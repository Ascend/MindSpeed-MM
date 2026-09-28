import math
import re
from typing import List
import logging

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.optimizer.grad_norm_overlap import manager as _grad_norm_overlap_mgr
from mindspeed_mm.fsdp.utils.device import get_device_type


logger = logging.getLogger(__name__)


def clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None,
    per_layer_stats: dict | None = None,
) -> torch.Tensor:
    """
    Clip gradients by their norm for distributed training with support for Expert Parallelism (EP).

    Args:
        model: The model containing parameters to clip.
        max_norm: Maximum norm of gradients. If 0, only compute norm without clipping.
        norm_type: Type of norm to compute (p-norm). Default is 2.0 (L2 norm).
        error_if_nonfinite: If True, raise error when gradients are non-finite.
        foreach: Whether to use foreach implementation for gradient clipping.
        per_layer_stats: Optional out-dict filled with per-layer gradient norms
            computed BEFORE clipping (raw, unclipped values). Works for EP models
            too. If None (default), no per-layer work is done.

    Returns:
        Total norm of gradients before clipping (or after clipping if max_norm > 0).
    """
    # Per-layer norms must be read before clipping; EP and non-EP share this hook.
    if per_layer_stats is not None:
        per_layer_stats.update(compute_per_layer_grad_norm(model))

    # EP-aware path (FSDP2 + EP)
    # _ep_param_groups is constructed in the optimizer
    # Check if model is configured for Expert Parallelism
    if hasattr(model, "_ep_param_groups"):
        return ep_fsdp2_clip_grad_norm(
            model,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    ps = get_parallel_state()
    fsdp_group = ps.get_fsdp_group()
    params: List[torch.nn.Parameter] = [p for p in model.parameters() if p.grad is not None]
    # Compute and reduce non-EP
    total_norm = _fsdp2_reduce_group(
        params=params,
        norm_type=norm_type,
        reduce_groups=[("fsdp", fsdp_group)],
    )
    if math.isinf(norm_type):
        total_norm = total_norm
    else:
        total_norm = total_norm ** (1.0 / float(norm_type))
    # Special case: max_norm = 0 means only compute gradient norm without clipping
    if max_norm > 0.:
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach=foreach)
    return total_norm


@torch.no_grad()
def ep_fsdp2_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    """
    EP-aware gradient clipping for composable FSDP2 with reductions mirroring FSDP1:

    - Compute local norms for non-EP and EP parameter groups separately.
    - For finite p: sum p-th powers across the appropriate groups, then take 1/p.
      • non-EP: all-reduce over FSDP group.
      • EP: all-reduce over EP-FSDP group, then over EP group.
    - For inf-norm: take elementwise MAX with the same reduction groups (MAX).
    - Use a single global clip coefficient for both groups.
    """

    ps = get_parallel_state()
    fsdp_group = ps.get_fsdp_group()
    ep_group = ps.get_ep_group() if ps.is_ep_enable() else None
    # For EP params sharded by FSDP2 along hidden dimension
    ep_fsdp_group = ps.get_efsdp_group()

    # Build param groups (filter out params without grads)
    ep_params: List[torch.nn.Parameter] = [p for p in model._ep_param_groups.get("ep", []) if p.grad is not None]
    non_ep_params: List[torch.nn.Parameter] = [p for p in model._ep_param_groups.get("non_ep", []) if p.grad is not None]
    non_ep_groups, ep_groups = _norm_reduce_groups(fsdp_group, ep_fsdp_group, ep_group)
    # Compute and reduce non-EP
    non_ep_total = _fsdp2_reduce_group(
        params=non_ep_params,
        norm_type=norm_type,
        reduce_groups=non_ep_groups,
    )
    # Compute and reduce EP: first across ep_fsdp, then across ep
    ep_total = _fsdp2_reduce_group(
        params=ep_params,
        norm_type=norm_type,
        reduce_groups=ep_groups,
    )

    if math.isinf(norm_type):
        total_norm = torch.maximum(non_ep_total, ep_total)
    else:
        total_norm = (non_ep_total + ep_total) ** (1.0 / float(norm_type))

    if max_norm == 0.:
        return total_norm
    # Apply the same clip coefficient to both groups
    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach=foreach)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach=foreach)

    return total_norm


# compute local sum of param gard norm
@torch.no_grad()
def _local_pth_sum(params: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    grads = [p.grad for p in params if p.grad is not None]
    # Keep prologue minimal: to_local() only (no_grad covers detach; foreach_norm(dtype=fp32)
    # matches materialize-then-norm bitwise, so no per-param .to(torch.float32)).
    grads_local = [g.to_local() if isinstance(g, DTensor) else g for g in grads]
    # Mixed dtypes: the pre-overlap baseline materialized every grad to fp32
    # first (single fp32 group per device); do the same here so mixed-dtype
    # models keep the baseline's summation order and stay bitwise identical.
    if len({g.dtype for g in grads_local}) > 1:
        grads_local = [g.to(torch.float32) for g in grads_local]
    default_device = grads_local[0].device if len(grads_local) > 0 else torch.device(get_device_type())
    res = torch.tensor(0.0, device=default_device, dtype=torch.float32)
    grouped_grads_local = _group_tensors_by_device_and_dtype([grads_local])
    for (device, dtype), ([device_grads_local], _) in grouped_grads_local.items():
        if _has_foreach_support(device_grads_local, device) or _device_has_foreach_support(device):
            # fp32 groups take the exact baseline kernel; only non-fp32 groups use the
            # probe-verified dtype=fp32 direct-compute path.
            if dtype == torch.float32:
                norms = torch._foreach_norm(device_grads_local, p)
            else:
                norms = torch._foreach_norm(device_grads_local, p, dtype=torch.float32)
            out = torch._foreach_pow_(norms, p)
            res += torch.sum(torch.stack(out)).to(default_device)
        else:
            for grad_local in device_grads_local:
                gn = torch.norm(grad_local.to(torch.float32), p=p)
                res = res + (gn**p).to(default_device)
    return res


def _local_max(params: List[torch.nn.Parameter]) -> torch.Tensor:
    dev = None
    mx = None
    for q in params:
        g = q.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            g_local = g.to_local()
        else:
            g_local = g
        if dev is None:
            dev = g_local.device
            mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
        gn = torch.max(torch.abs(g_local.detach().to(torch.float32)))
        mx = torch.maximum(mx, gn)
    if mx is None:
        dev = torch.device(get_device_type())
        mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
    return mx


def _norm_reduce_groups(fsdp_group, ep_fsdp_group, ep_group):
    """The reduce groups shared by the total grad norm and the per-layer view.

    Non-expert parameters sharded by FSDP2 are reduced over the FSDP group only;
    expert parameters are reduced over the expert FSDP group and then over the expert
    group, mirroring how they are sharded. Both callers build their reductions from
    this single definition so the total norm and the per-layer curves cannot drift
    apart when the group layout changes.
    """
    return [("fsdp", fsdp_group)], [("ep_fsdp", ep_fsdp_group), ("ep", ep_group)]


def _fsdp2_reduce_group(
    params: List[torch.nn.Parameter],
    norm_type: float,
    reduce_groups: List[tuple[str, dist.ProcessGroup | None]],
) -> torch.Tensor:
    """Compute local group statistic and reduce over provided groups.

    For finite p, returns the globally-reduced sum of p-th powers (not the final norm).
    For inf, returns the globally-reduced max.
    """
    if len(params) == 0:
        device = torch.device(get_device_type())
        val = torch.tensor(0.0, device=device, dtype=torch.float32)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.SUM, group=group)
        return val
    if math.isinf(norm_type):
        val = _local_max(params)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.MAX, group=group)
        return val
    else:
        p = float(norm_type)
        val = None
        if _grad_norm_overlap_mgr.enabled:
            val = _grad_norm_overlap_mgr.consume(params, p)
        if val is None:
            val = _local_pth_sum(params, p)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.SUM, group=group)
        return val


_BLOCK_INDEX_PATTERN = re.compile(r"\.(\d+)\.")


def _layer_group_name(param_name: str) -> str:
    """Map a parameter name to its layer group, e.g. ``model.layers.12``.

    The first ``.<index>.`` in the name identifies the repeating block and the
    container path is kept, so an encoder/decoder, an audio tower and a text tower
    or a vision tower never share a group. Parameters outside any indexed block
    (embedding, norm, head, ...) are grouped by their parent module path.
    """
    match = _BLOCK_INDEX_PATTERN.search(param_name)
    if match:
        return f"{param_name[:match.start()]}.{match.group(1)}"
    return param_name.rsplit(".", 1)[0] if "." in param_name else param_name


def _layer_group_sort_key(group: str):
    """Sort layer groups naturally: by container path, then by integer index."""
    prefix, _, index = group.rpartition(".")
    return (prefix, int(index)) if index.isdigit() else (group, -1)


def compute_per_layer_grad_norm(model) -> dict:
    """Compute the per-layer gradient L2 norm.

    Parameters are grouped by layer and split into an expert and a non-expert part
    (``model._ep_param_groups``, the same cache the EP clipping path reduces). Each
    part is stacked into one vector per reduction set and reduced with the groups
    the total norm uses: non-expert over the FSDP shard group, expert over the
    expert FSDP group and then the expert group. Summing the squared per-layer
    norms therefore reproduces the total grad norm squared, and the number of
    collectives stays independent of the layer count.

    No reduction across ``dp_replicate`` is needed: HSDP already all-reduces the
    gradients over the replica dimension during backward, so every replica holds the
    same value here -- which is what the total-norm path assumes as well.

    Must be called on every rank (collective) before gradients are clipped.
    Returns ``{group_name: {"ave": float}}``.
    """
    # Expert parameters come from the cache the optimizer built for the EP clipping
    # path (empty for a dense model), so both paths split the model identically.
    ep_param_groups = getattr(model, "_ep_param_groups", None) or {}
    ep_param_ids = {id(param) for param in ep_param_groups.get("ep", ())}

    # Group parameters that have gradients by layer, expert part kept separate.
    non_ep_by_layer, ep_by_layer = {}, {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        by_layer = ep_by_layer if id(param) in ep_param_ids else non_ep_by_layer
        by_layer.setdefault(_layer_group_name(name), []).append(param)

    # Deterministic, natural order on every rank so the vector layout matches.
    groups = sorted(set(non_ep_by_layer) | set(ep_by_layer), key=_layer_group_sort_key)
    if not groups:
        return {}

    fsdp_group = None
    ep_fsdp_group = None
    ep_group = None
    if dist.is_initialized():
        ps = get_parallel_state()
        fsdp_group = ps.get_fsdp_group()
        if ep_param_ids:
            ep_fsdp_group = ps.get_efsdp_group()
            ep_group = ps.get_ep_group()
    # The same group lists the total norm reduces over, so the two cannot drift apart.
    non_ep_groups, ep_groups = _norm_reduce_groups(fsdp_group, ep_fsdp_group, ep_group)

    # Non-expert part: all layers in ONE vector reduce; a layer without non-expert
    # parameters contributes 0.0, so the vector layout is the same on every rank.
    layer_sq = torch.stack([_local_pth_sum(non_ep_by_layer.get(g, ()), 2.0) for g in groups])
    for _, group in non_ep_groups:
        if group is not None:
            dist.all_reduce(layer_sq, op=dist.ReduceOp.SUM, group=group)

    # Expert part: reduced like ep_fsdp2_clip_grad_norm (same experts first, then
    # across experts). Sums of squares are additive, so the vectors simply add.
    if ep_param_ids:
        ep_sq = torch.stack([_local_pth_sum(ep_by_layer.get(g, ()), 2.0) for g in groups])
        for _, group in ep_groups:
            if group is not None:
                dist.all_reduce(ep_sq, op=dist.ReduceOp.SUM, group=group)
        layer_sq = layer_sq + ep_sq

    layer_norm = layer_sq.sqrt()

    # One host sync for the whole vector instead of one per layer group.
    norms = layer_norm.tolist()
    return {group: {"ave": norms[i]} for i, group in enumerate(groups)}
