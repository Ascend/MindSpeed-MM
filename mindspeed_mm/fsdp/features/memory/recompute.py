# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import inspect
import functools
from contextlib import nullcontext, ExitStack

from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint
from mindspeed.fsdp.utils.log import print_rank
from mindspeed.fsdp.utils.str_match import module_name_match

from .op_replay import build_op_replay_context_fn


logger = logging.getLogger(__name__)
_MXFP8_RECOMPUTE_CACHE = None


def _load_mxfp8_recompute_phase():
    global _MXFP8_RECOMPUTE_CACHE
    if _MXFP8_RECOMPUTE_CACHE is not None:
        return _MXFP8_RECOMPUTE_CACHE
    try:
        from fsdp_turbo.quantization.core.recompute_phase import (
            mark_module_for_recompute,
            mxfp8_recompute_context,
        )
        _MXFP8_RECOMPUTE_CACHE = (mark_module_for_recompute, mxfp8_recompute_context)
    except ImportError:
        _MXFP8_RECOMPUTE_CACHE = (None, None)
    return _MXFP8_RECOMPUTE_CACHE


def _compose_checkpoint_contexts(fns):
    def composed():
        contexts = [fn() for fn in fns]
        fwd_ctxs, bwd_ctxs = zip(*contexts)
        class _Composed:
            def __init__(self, managers):
                self._managers = managers
            def __enter__(self):
                self._stack = ExitStack()
                for m in self._managers:
                    self._stack.enter_context(m)
                return self
            def __exit__(self, *args):
                return self._stack.__exit__(*args)
        return _Composed(fwd_ctxs), _Composed(bwd_ctxs)
    return composed


def _build_checkpoint_context_fn(module, context_fn, mxfp8_ctx_fn):
    ctx_list = []
    if context_fn is not None:
        ctx_list.append(context_fn)
    if mxfp8_ctx_fn is not None:
        def _mxfp8_ctx(mod=module):
            return nullcontext(), mxfp8_ctx_fn(mod)
        ctx_list.append(_mxfp8_ctx)
    if not ctx_list:
        return None
    if len(ctx_list) == 1:
        return ctx_list[0]
    return _compose_checkpoint_contexts(ctx_list)


def recompute_modules(model, plan, op_cache=None):
    # Op replay (plan.op_replay_scopes, enabled when the list is non-empty) is a
    # policy of the checkpoint boundary: patch its replay zones first so the zones
    # fall inside the checkpoint wrap and the wrap picks up the context_fn.
    # op_cache is the shared SwapCache from the SwapManager, required when op
    # replay is enabled.
    context_fn = build_op_replay_context_fn(model, plan.op_replay_scopes, plan.use_reentrant, op_cache)
    modules = get_recompute_modules(model, plan.apply_modules)
    if context_fn is not None:
        _check_no_nested_checkpoints(modules)

    mxfp8_mark_fn, _ = _load_mxfp8_recompute_phase()
    for name, module in modules:
        print_rank(logger.info, f'Applying recompute to module: {name}')
        if mxfp8_mark_fn is not None:
            mxfp8_mark_fn(module)
        module.forward = recompute_wrapper(
            module.forward, plan.use_reentrant, context_fn,
            plan.flatten_inputs, module
        )
    return model


def _check_no_nested_checkpoints(modules):
    """Op replay does not support nested checkpoints.

    When the recompute coverage nests (one matched module is an ancestor of
    another), the two checkpoints' replay modes would double-handle the inner
    zone's ops (double put / replay misalignment). Reject the configuration at
    wiring time rather than degrading silently. Nested recompute without op
    replay is legal and not checked here.
    """
    names = []
    for name, _ in modules:
        if name not in names:
            names.append(name)
    for outer in names:
        prefix = outer + '.'
        for inner in names:
            if inner != outer and inner.startswith(prefix):
                raise ValueError(
                    "op replay does not support nested checkpoints: "
                    f"recompute_plan.apply_modules matches both '{outer}' and its "
                    f"descendant '{inner}'. Remove one of the overlapping patterns.")


def get_recompute_modules(modules, plan):
    matched_modules = []
    for plan_name in plan:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                matched_modules.append((name, module))
    if len(matched_modules) == 0:
        raise RuntimeError(f'[Recompute] No module named {plan}.')
    return matched_modules


def _flatten_call(function, args, kwargs):
    """Flatten the full (args, kwargs) input tree into unique leaves plus a
    rebuild recipe (call-shape normalization, semantics-preserving).

    Non-reentrant checkpoint only routes top-level positional tensors through
    save_for_backward (saved_tensors_hooks); kwargs are captured by reference in
    the checkpoint frame and held until backward. Flattening every leaf
    (keyword-only / **kwargs contents / tensors inside containers included) into
    positional arguments moves them all into the save_for_backward channel so
    tenants like ActStash can swap them out. The shim inside the checkpoint
    rebuilds the original structure before calling the wrapped function, so the
    function sees exactly the call it was invoked with.

    Leaves are deduplicated by identity: an object aliased across several slots
    occupies one leaf position, so recompute restores the alias (same object)
    instead of silently forking it into independent copies, and the saved
    channel packs it once instead of once per slot.
    """
    flat, spec = tree_flatten((args, kwargs))
    unique_leaves, index, slots = [], {}, []
    for leaf in flat:
        key = id(leaf)
        if key not in index:
            index[key] = len(unique_leaves)
            unique_leaves.append(leaf)
        slots.append(index[key])

    def flattened_function(*leaves):
        rebuild = [leaves[slot] for slot in slots]
        original_args, original_kwargs = tree_unflatten(rebuild, spec)
        return function(*original_args, **original_kwargs)

    return flattened_function, unique_leaves


def recompute_wrapper(function, use_reentrant, context_fn=None, flatten_inputs=False, module=None):
    # Only inject the transformers-style cache kwarg when the wrapped forward
    # actually accepts it. Native Wan blocks do not take this argument.
    sig = inspect.signature(function)
    has_past_key_values = 'past_key_values' in sig.parameters

    def wrapper(*args, **kwargs):
        if has_past_key_values:
            kwargs['past_key_values'] = None  # transformers kv cache must be set None, or model use_cache=False
        ckpt_kwargs = {}
        if not use_reentrant:
            _, mxfp8_ctx_fn = _load_mxfp8_recompute_phase()
            built_ctx_fn = _build_checkpoint_context_fn(module, context_fn, mxfp8_ctx_fn)
            if built_ctx_fn is not None:
                ckpt_kwargs['context_fn'] = built_ctx_fn

            if flatten_inputs:
                flattened_function, unique_leaves = _flatten_call(function, args, kwargs)
                return checkpoint(flattened_function, *unique_leaves,
                                  use_reentrant=use_reentrant, **ckpt_kwargs)
            return checkpoint(function, *args, use_reentrant=use_reentrant, **ckpt_kwargs, **kwargs)
        else:
            bound_function = functools.partial(function, **kwargs)
            return checkpoint(bound_function, *args, use_reentrant=use_reentrant, **ckpt_kwargs)
    return wrapper
