"""Op Replay (selective activation compression).

A policy of the recompute (checkpoint) mechanism, not a standalone feature:
inside a checkpoint boundary, replay zones (scopes) mark sub-regions:
- forward: OpReplayCachingMode (a TorchDispatchMode) intercepts whitelisted
  aten ops in the active scope and hands their outputs to the shared SwapCache
  (async HBM<->DDR swap, owned by the SwapManager), recording an op ->
  output-handle route in the checkpoint's session;
- recompute: OpReplayReplayingMode replays a hit op via the route (skipping
  computation); any miss falls back to recomputing that op.

It only activates via a non-reentrant checkpoint's context_fn, so the wiring
entry point is build_op_replay_context_fn, called by the recompute feature.
The feature is a pure tenant of the swap cache: it never constructs caches
and never delimits iterations (the step orchestrator drives the boundary).

Structure: each configured scope (region x whitelist x save_rng) is an
OpReplayScope patched onto its modules. The dispatch modes span the whole
checkpoint (context_fn granularity), so the active scope is signalled via a
scope stack pushed by the patcher (innermost zone wins). Each checkpoint gets
an OpReplaySession with an independent FIFO route table; the SwapCache is
shared globally. Nested checkpoints are unsupported: recompute_modules rejects
nesting recompute coverage with a ValueError when op replay is active.
"""
import contextlib
import functools
import inspect
import logging
import warnings
from collections import defaultdict
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten

from mindspeed.fsdp.utils.str_match import module_name_match
from ...utils.device import IS_NPU_AVAILABLE
from .swap_core import SwapCache, SwapHandle


logger = logging.getLogger(__name__)


def _get_rng_snapshot() -> dict:
    snap = {'cpu': torch.get_rng_state()}
    if IS_NPU_AVAILABLE:
        snap['npu'] = torch.npu.get_rng_state()
    return snap


def _set_rng_snapshot(snap: dict) -> None:
    torch.set_rng_state(snap['cpu'])
    if IS_NPU_AVAILABLE and 'npu' in snap:
        torch.npu.set_rng_state(snap['npu'])


# As in PyTorch SAC: ops whose call counts differ between forward and recompute
# must be excluded, otherwise cache misalignment corrupts gradients.
try:
    _OP_REPLAY_IGNORED_OPS = {
        torch.ops.aten.detach.default,
        torch.ops.prim.device.default,
    }
except Exception:
    _OP_REPLAY_IGNORED_OPS = set()

# Storage / in-place metadata ops that must never be cached or skipped:
# they mutate tensor storage and are not replay-safe across recompute.
_storage_op_names = [
    "aten.set_.source_Storage_storage_offset",
    "aten.set_.source_Tensor",
    "aten.resize_.default",
    "aten.resize_as_.default",
]
def _resolve_op(name: str):
    """Resolve a 'aten.op_name.overload' string via attribute lookup (no eval)."""
    return functools.reduce(getattr, name.split('.'), torch.ops)


def _try_resolve_op(name: str):
    """Like _resolve_op but returns None (with a warning) when the op is missing."""
    try:
        return _resolve_op(name)
    except Exception as err:
        logger.warning(f"Op '{name}' unavailable in this torch version, skipped: {err}")
        return None


for _op_name in _storage_op_names:
    _op = _try_resolve_op(_op_name)
    if _op is not None:
        _OP_REPLAY_IGNORED_OPS.add(_op)
# functional tensor metadata ops
try:
    _OP_REPLAY_IGNORED_OPS |= set(torch._subclasses.functional_tensor.FunctionalTensor.metadata_fns)
except Exception as err:
    # FunctionalTensor metadata ops may be unavailable in this torch version.
    logger.debug(f"FunctionalTensor metadata_fns unavailable, skipped: {err}")



def _maybe_detach(x, any_ret_has_alias_info):
    """Detach output tensors before caching (as PyTorch SAC does).

    Tensors created inside the reentrant dispatch scope carry autograd metadata
    and cannot be returned as-is during recompute (RuntimeError: differentiable
    view); detach produces a fresh alias without AutogradMeta. Excluding
    ADInplaceOrView keeps the version counter propagating correctly.
    """
    if isinstance(x, torch.Tensor) and (
            x.is_floating_point() or x.is_complex() or any_ret_has_alias_info):
        with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.ADInplaceOrView, False):
            x = x.detach()
    return x


DEFAULT_CACHE_OPS = frozenset({
    "aten.mm.default",
    "aten.addmm.default",
    "aten.bmm.default",
    "aten.linear.default",
    "aten._scaled_dot_product_flash_attention.default",
})


def _parse_op_list(op_names):
    """Parse a list of 'aten.op_name.overload' strings into torch op objects."""
    if op_names is None:
        return None
    ops = set()
    for name in op_names:
        try:
            ops.add(_resolve_op(name))
        except Exception as e:
            raise ValueError(f"Invalid op-replay op name '{name}': {e}")
    return ops


def _default_cache_ops():
    """Default whitelist, tolerant: an op missing in this torch version is
    skipped with a warning instead of crashing at startup. User-specified
    lists stay strict (typo in config must surface as an error)."""
    return {op for op in (_try_resolve_op(n) for n in DEFAULT_CACHE_OPS)
            if op is not None}


class _HandleInfo:
    """Pytree-leaf wrapper for SwapHandle so tree_flatten does not expand it."""
    __slots__ = ('handle',)

    def __init__(self, handle: SwapHandle):
        self.handle = handle


class OpReplayScope:
    """One replay scope: a region (its patched modules) with its own op
    whitelist and RNG policy. Immutable after construction.

    cache_ops: whitelist of ops to cache inside this scope. None falls back
               to the built-in default whitelist (matmul family + flash
               attention) at construction; an explicit empty set caches
               nothing. Caching is always whitelist-gated — there is no
               "cache everything" mode (it would swap out cheap-to-recompute
               and view/metadata ops whose payloads are pure waste).
    save_rng:  record/replay post-op RNG snapshots for this scope's cached ops
    """

    def __init__(self, cache_ops=None, save_rng=True, name=None):
        if cache_ops is None:
            cache_ops = _default_cache_ops()
        self.cache_ops = cache_ops
        self.save_rng = save_rng
        self.name = name

    @classmethod
    def from_config(cls, cfg) -> "OpReplayScope":
        """Build from a scope config (duck-typed on OpReplayScopeConfig).

        Whitelist fallback chain: scope.cache_ops -> built-in default whitelist
        (the fallback lives in the constructor, so both entry paths agree).
        """
        return cls(
            cache_ops=_parse_op_list(getattr(cfg, "cache_ops", None)),
            save_rng=getattr(cfg, "save_rng", True),
            name=getattr(cfg, "name", None),
        )

    def __repr__(self):
        return f"OpReplayScope(name={self.name!r}, save_rng={self.save_rng})"


class OpReplaySession:
    """Per-checkpoint session: one independent FIFO route table (correctness
    requires per-checkpoint routes), sharing the controller's scope stack and
    the global op_cache (capacity eviction and cross-checkpoint prefetch work
    on one physical store)."""

    def __init__(self, controller: "OpReplayController"):
        self._scope_stack = controller._scope_stack       # shared by reference
        self.op_cache = controller.op_cache               # shared
        # token_routes: func -> [(info_leaves, spec, post_rng_snap), ...], FIFO.
        # post_rng_snap is the RNG snapshot right after the op ran in forward;
        # replaying (skipping) the op restores it to keep the recompute RNG
        # stream aligned with forward op by op.
        self.token_routes: defaultdict = defaultdict(list)

    @property
    def zone_scope(self) -> Optional[OpReplayScope]:
        """The innermost active replay scope (None outside any zone)."""
        return self._scope_stack[-1] if self._scope_stack else None


class _OpReplayMode(TorchDispatchMode):
    """Base of the two replay modes: zone/whitelist gating.

    The modes span the whole checkpoint (context_fn granularity), so the zone
    signal comes from the scope stack. Nested checkpoints are rejected at
    wiring time (see recompute_modules), so at most one replay mode is active
    at a time and no dispatch-level nesting guard is needed.
    """

    def __init__(self, session: OpReplaySession):
        super().__init__()
        self.session = session

    def _zone_scope(self, func) -> Optional[OpReplayScope]:
        """The scope whose zone and whitelist cover func, or None (pass through)."""
        if func in _OP_REPLAY_IGNORED_OPS:
            return None
        scope = self.session.zone_scope
        if scope is None or func not in scope.cache_ops:
            return None
        return scope


class OpReplayCachingMode(_OpReplayMode):
    """Forward mode: run whitelisted ops and swap their outputs out, recording
    op -> route in this checkpoint's session."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        scope = self._zone_scope(func)
        if scope is None:
            return func(*args, **kwargs)

        out = func(*args, **kwargs)
        any_ret_has_alias_info = any(
            ret.alias_info is not None for ret in func._schema.returns)
        leaves, spec = tree_flatten(out)
        info_leaves = []
        has_tensor = False
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                handle = self.session.op_cache.put(
                    _maybe_detach(leaf, any_ret_has_alias_info))
                info_leaves.append(_HandleInfo(handle))
                has_tensor = True
            else:
                info_leaves.append(leaf)

        if has_tensor:
            # put consumes no RNG, so this is still the op's post-op state.
            post_rng_snap = _get_rng_snapshot() if scope.save_rng else None
            self.session.token_routes[func].append((info_leaves, spec, post_rng_snap))
        return out


class OpReplayReplayingMode(_OpReplayMode):
    """Recompute mode: replay cached op outputs (skipping computation), falling
    back to recompute on any miss; drains routes this checkpoint did not
    consume on exit (e.g. early stop). The iteration boundary itself is not
    delimited here: the step orchestrator drives it on the shared cache (see
    SwapManager)."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self._zone_scope(func) is None:
            return func(*args, **kwargs)

        routes = self.session.token_routes.get(func)
        if not routes:
            return func(*args, **kwargs)
        info_leaves, spec, post_rng_snap = routes.pop(0)

        # Any miss -> the whole op falls back to recompute. No RNG restore
        # is needed then: the stream is already at forward's pre-op position.
        handles = [info.handle for info in info_leaves if isinstance(info, _HandleInfo)]
        tensors: dict = {}
        for handle in handles:
            tensor = self.session.op_cache.pop(handle)
            if tensor is None:
                # Off the regular path: eviction never removes a handle from
                # the cache, so a miss means a mid-step clear() or a repeated
                # recompute. Falling back is correct (recompute is ground
                # truth); this route's unpopped handles stay orphaned until
                # the iteration boundary reclaims them (SwapCache.clear via
                # step_end).
                warnings.warn(
                    f"op replay cache miss on {func}: the recorded route's "
                    "tensor is absent from the swap cache (only a mid-step "
                    "clear() or a repeated recompute removes handles, "
                    "eviction never does); falling back to recompute.",
                    RuntimeWarning, stacklevel=2)
                return func(*args, **kwargs)
            tensors[handle] = tensor

        out_leaves = []
        for info in info_leaves:
            if isinstance(info, _HandleInfo):
                out_leaves.append(tensors[info.handle])
            else:
                out_leaves.append(info)
        out = tree_unflatten(out_leaves, spec)

        # The skipped op consumed no RNG; rewind the stream to its forward
        # post-op state so later ops draw from the same position as forward.
        if post_rng_snap is not None:
            _set_rng_snapshot(post_rng_snap)
        return out

    def __exit__(self, *args):
        ret = super().__exit__(*args)
        # Release routes this checkpoint did not consume (e.g. early stop).
        for func, routes in self.session.token_routes.items():
            for info_leaves, spec, post_rng_snap in routes:
                for info in info_leaves:
                    if isinstance(info, _HandleInfo):
                        self.session.op_cache.pop(info.handle)
        self.session.token_routes.clear()
        return ret


class OpReplayController:
    """Holds the replay scopes, the shared zone-scope stack and the shared
    op_cache; hands out per-checkpoint sessions via make_context_fn()."""

    def __init__(self, scopes: List[OpReplayScope], op_cache: SwapCache):
        self.scopes = list(scopes)
        self.op_cache = op_cache
        self._scope_stack: List[OpReplayScope] = []

    @contextlib.contextmanager
    def replay_zone(self, scope: OpReplayScope):
        """Mark a replay region for scope. The shared stack is the dispatch
        layer's in-zone signal (the modes span the whole checkpoint, wider
        than the zone); nested zones resolve to the innermost scope."""
        self._scope_stack.append(scope)
        try:
            yield
        finally:
            self._scope_stack.pop()

    def make_context_fn(self):
        """checkpoint context_fn returning (caching_mode, replaying_mode).

        Each checkpoint gets an OpReplaySession with independent token_routes
        (FIFO correctness), while the scope stack and op_cache are shared
        (capacity eviction and cross-checkpoint prefetch work on one store).
        """
        def _ctx():
            session = OpReplaySession(self)
            return OpReplayCachingMode(session), OpReplayReplayingMode(session)
        return _ctx


class OpReplayPatcher:
    def __init__(self, module: nn.Module, controller: OpReplayController, scope: OpReplayScope):
        self.module = module
        self.controller = controller
        self.scope = scope
        self._orig_forward = module.forward
        # Expose the wrapped forward's signature (PEP 362): __call__'s generic
        # (*args, **kwargs) would otherwise make signature consumers (e.g. the
        # recompute wrapper's past_key_values check) see VAR_POSITIONAL.
        self.__signature__ = inspect.signature(self._orig_forward)

    def __call__(self, *args, **kwargs):
        with self.controller.replay_zone(self.scope):
            return self._orig_forward(*args, **kwargs)


def apply_op_replay_inplace(model: nn.Module, fqns: List[str],
                            controller: OpReplayController, scope: OpReplayScope):
    """Patch target modules' forward in place without changing the module tree.
    A module already patched (by an earlier scope) keeps its first scope:
    list order is the priority order for overlapping scopes."""
    for fqn in fqns:
        module = model.get_submodule(fqn)
        if isinstance(module.forward, OpReplayPatcher):
            continue
        module.forward = OpReplayPatcher(module, controller, scope)
    return model


def build_op_replay_context_fn(model: nn.Module, scope_configs,
                               use_reentrant: bool,
                               op_cache: Optional[SwapCache] = None) -> Optional[Callable]:
    """Wire op replay into the recompute feature; returns the checkpoint context_fn.

    Called by the recompute feature before it wraps modules with checkpoint.
    scope_configs is the recompute_plan.op_replay_scopes list (empty disables
    the feature).
    Returns None (op replay inert) when:
    - the scope list is empty (feature disabled), or
    - the checkpoint is reentrant: context_fn would be silently dropped and
      replay would never activate, so skip with a warning.

    op_cache is the shared SwapCache from the SwapManager, required when the
    feature is enabled. Prerequisite (config-side, not enforced here): each
    scope's apply_modules must lie within the recompute_plan coverage; zones
    outside coverage stay inert.
    """
    if not scope_configs:
        return None
    # Op replay only activates via a non-reentrant checkpoint's context_fn;
    # with a reentrant checkpoint the context_fn is silently dropped and
    # replay would never activate, so skip with a warning.
    if use_reentrant:
        logger.warning(
            "recompute_plan.op_replay_scopes requires use_reentrant=false "
            f"(got use_reentrant={use_reentrant}); skipping op replay.")
        return None
    if op_cache is None:
        raise ValueError(
            "op replay requires the shared SwapCache from the SwapManager "
            "(swap_plan); none was provided")

    scopes = [OpReplayScope.from_config(cfg) for cfg in scope_configs]
    controller = OpReplayController(scopes, op_cache)

    for cfg, scope in zip(scope_configs, scopes):
        matched_fqns = []
        for plan_name in getattr(cfg, "apply_modules", None) or []:
            for name, _ in model.named_modules():
                if module_name_match(plan_name, name):
                    if name not in matched_fqns:
                        matched_fqns.append(name)
        label = f"scope {scope.name!r}" if scope.name else "scope"
        for fqn in matched_fqns:
            logger.info(f"Applying op replay {label} to module: {fqn}")
        if matched_fqns:
            apply_op_replay_inplace(model, matched_fqns, controller, scope)
    return controller.make_context_fn()
