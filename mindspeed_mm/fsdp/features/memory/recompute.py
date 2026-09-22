# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import inspect
import functools
from contextlib import nullcontext, ExitStack

from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint
from mindspeed_mm.fsdp.log import print_rank
from mindspeed_mm.fsdp.utils.str_match import module_name_match

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
    # mxfp8 low-precision recompute only marks module .forward, so it must not
    # be attached when there is no owning module (method-level entries).
    if mxfp8_ctx_fn is not None and module is not None:
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
    specs = list(getattr(plan, "apply_modules", None) or [])

    # Dispatch each entry: matches a module in the model tree → wrap .forward;
    # otherwise treat as module-pattern.method_name (rpartition at last '.').
    module_entries = []
    method_entries = []
    for entry in specs:
        if any(module_name_match(entry, name) for name, _ in model.named_modules()):
            module_entries.append(entry)
        else:
            method_entries.append(entry)

    modules = get_recompute_modules(model, module_entries) if module_entries else []

    if context_fn is not None:
        _check_no_nested_checkpoints(modules)
        if method_entries:
            logger.warning(
                "op replay does not support nested checkpoints: make sure no "
                "method in apply_modules runs inside another apply_modules "
                "checkpoint coverage")

    mxfp8_mark_fn, _ = _load_mxfp8_recompute_phase()
    for name, module in modules:
        print_rank(logger.info, f'Applying recompute to module: {name}')
        if mxfp8_mark_fn is not None:
            mxfp8_mark_fn(module)
        module.forward = recompute_wrapper(
            module.forward, plan.use_reentrant, context_fn,
            plan.flatten_inputs, module
        )
    if method_entries:
        logger.warning(
            "low-precision recompute (mxfp8) does not support method-level "
            "apply_modules entries; only module .forward is covered. "
            f"Method entries {method_entries} will use standard recompute without "
            "low-precision marking.")
    for pattern in method_entries:
        _apply_recompute_method(model, pattern, plan, context_fn)

    return model


def _apply_recompute_method(model, pattern, plan, context_fn=None):
    """Wrap a method on every module matching the prefix of *pattern*.
    'model.layers.{*}._attention_forward' → module pattern 'model.layers.{*}',
    method '_attention_forward'; setattr on each matched instance.
    """
    module_pattern, _, method_name = pattern.rpartition('.')
    if not module_pattern or not method_name:
        raise ValueError(
            f'[Recompute] method entry must look like '
            f"'module.pattern.method_name', got {pattern!r}")
    matched = False
    for name, module in model.named_modules():
        if not module_name_match(module_pattern, name):
            continue
        method = getattr(module, method_name, None)
        if method is None or not callable(method):
            raise RuntimeError(
                f'[Recompute] No callable method {method_name!r} on module {name!r}.')
        print_rank(logger.info, f'Applying recompute to method: {name}.{method_name}')
        setattr(module, method_name,
                recompute_wrapper(method, plan.use_reentrant, context_fn,
                                  plan.flatten_inputs))
        matched = True
    if not matched:
        raise RuntimeError(f'[Recompute] No module named {module_pattern} (from {pattern!r}).')


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
            bound = sig.bind(*args, **kwargs)
            bound.arguments['past_key_values'] = None
            args, kwargs = bound.args, bound.kwargs
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
