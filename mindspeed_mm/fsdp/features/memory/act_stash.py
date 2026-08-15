"""ActStash: activation offload as a saved_tensors_hooks tenant of the shared
SwapCache (HBM<->DDR async swap).

pack  = SwapCache.put (the SwapHandle IS the unpack token)
pop   = SwapCache.pop (prefetch is self-driven by the cache's transition graph)

Physical resources (capacity, streams, CPU arena, eviction/prefetch policy)
are owned once by the SwapManager and shared with the other tenants (op
replay, ...); ActStash is a pure tenant with zero lifecycle code — the
iteration boundary is driven by the training loop via SwapManager.step_end().

Jurisdiction and contracts:
- A saved_tensors_hooks tenant covers exactly what autograd saves while the
  hook context is active: with the wrapper outside a non-reentrant checkpoint
  (the wiring order of FeaturesApplier), the checkpoint's initial forward runs
  under no_grad (nothing packed) and its internal saved tensors are taken over
  by the checkpoint frame hooks at recompute, so ActStash's coverage is
  "outside checkpoints + checkpoint boundary inputs" — complementary to op
  replay, which manages op outputs inside checkpoints.
- Tensors inside kwargs/containers never reach the pack hook (checkpoint
  saves only direct tensor arguments); this is a known blind spot shared by
  every hooks-based mechanism.
- A put tensor must not be mutated in place while its handle is alive (the
  saved-tensor autograd version counter already rejects such writes).
- Handle model: eviction only drops the cache's own reference. An externally
  retained tensor simply cannot be physically freed (benign degradation:
  correct gradients, no HBM saved).

Known interaction: the skip-recompute ops
(skip_recompute_flash_attn / GDN variants) schedule the freeing of their own
offloaded tensors via the legacy OffloadManager keyed by TrainingContext's
layer index/depth — a protocol only the legacy offload wrapper maintains.
ActStash deliberately does NOT maintain it (pure tenant, no global-state
side channels): with skip_*_recompute enabled, those ops keep their HBM
resident (~6.5GB in the qwen3_5 4B config). The intended pairing is
skip_*_recompute: false (+ op replay, which covers the same ops through
SwapCore).
"""
import functools
import logging

import torch
from torch.autograd.graph import saved_tensors_hooks

from mindspeed.fsdp.utils.log import print_rank
from mindspeed.fsdp.utils.str_match import module_name_match

from .swap_core import SwapCache


logger = logging.getLogger(__name__)


def _stashable(tensor: torch.Tensor) -> bool:
    """A tensor enters the swap cache iff it is a device tensor worth moving:
    skip non-device tensors, Parameters (and their views), non-strided layouts
    and empty-storage tensors; those pass through the hooks as their own
    token."""
    if tensor.device.type not in ('npu', 'cuda'):
        return False
    if isinstance(tensor, torch.nn.parameter.Parameter) \
            or isinstance(tensor._base, torch.nn.parameter.Parameter):
        return False
    if tensor.layout != torch.strided:
        # Non-strided layouts (e.g. sparse) have no flat storage to swap;
        # untyped_storage() below would raise, so pass them through instead.
        return False
    return tensor.untyped_storage().size() > 0


class ActStashHooks(saved_tensors_hooks):
    """saved_tensors_hooks tenant: pack = cache.put, unpack = cache.pop.

    Ineligible tensors (see _stashable) pass through as their own token, so
    unpack can tell the two token kinds apart without any bookkeeping. The
    autograd engine guarantees every packed tensor is eventually unpacked
    (retain_graph replays unpack), so a pop miss is always a bug, not a race.
    """

    def __init__(self, cache: SwapCache):
        self._cache = cache
        super().__init__(self._pack, self._unpack)

    def _pack(self, tensor: torch.Tensor):
        if not _stashable(tensor):
            return tensor
        return self._cache.put(tensor)

    def _unpack(self, token):
        if isinstance(token, torch.Tensor):
            return token
        tensor = self._cache.pop(token)
        if tensor is None:
            raise RuntimeError(
                "ActStash: saved-tensor token missing from the swap cache; "
                "a packed tensor must stay managed until its unpack")
        return tensor


def get_act_stash_modules(model, apply_modules):
    """Match apply_modules patterns against named modules (same semantics as
    the recompute wiring)."""
    matched_modules = []
    for plan_name in apply_modules:
        for name, module in model.named_modules():
            if module_name_match(plan_name, name):
                if (name, module) not in matched_modules:
                    matched_modules.append((name, module))
    if len(matched_modules) == 0:
        raise RuntimeError(f'[ActStash] No module named {apply_modules}.')
    return matched_modules


def _act_stash_wrapper(forward_func, cache):
    @functools.wraps(forward_func)
    def wrapper(*args, **kwargs):
        with ActStashHooks(cache):
            return forward_func(*args, **kwargs)
    return wrapper


def apply_act_stash_modules(model, apply_modules, cache: SwapCache):
    """Wrap each matched module's forward with ActStashHooks (in place).

    Wired after recompute in FeaturesApplier.pre_fully_shard_apply, so the
    hook context sits outside the checkpoint wrap and sees exactly the
    checkpoint boundary inputs (see module docstring)."""
    for name, module in get_act_stash_modules(model, apply_modules):
        print_rank(logger.info, f'Applying act stash to module: {name}')
        module.forward = _act_stash_wrapper(module.forward, cache)
    return model
