# Copyright (c) 2025 Moonshot AI
import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate

from .dtensor_utils import (
    _chunk_range,
    _copy_replicated_to_dtensor_,
    _row_shards_preserve_heads,
    _single_shard_layout,
)


# Parameter-group entries that hold live objects (tensors, callbacks, process
# groups). They are left out of checkpoints and reattached after loading.

# Metrics consumed by the preflight but not yet applied to Q/K weights.
_QK_CLIP_PENDING_METRICS = "qk_clip_pending_max_logits"

_QK_CLIP_RUNTIME_KEYS = (
    "qk_clip_params",
    "qk_clip_max_logits",
    "qk_clip_metric_provider",
    "qk_clip_process_groups",
    _QK_CLIP_PENDING_METRICS,
)

# Per-group metric status, reduced with MAX across ranks, so a larger value
# wins when ranks disagree.
_METRICS_OK = 0
_METRICS_MISSING = 1
_METRICS_WRONG_SHAPE = 2


@torch.no_grad()
def qk_clip_(query, key, max_logits, threshold, alpha, metrics_validated=False):
    """Scale per-head query/key tensors in place by their QK-Clip factor.

    Heads above the threshold use ``gamma = threshold / max_logits``; others
    use 1. Query and key scales are ``gamma**alpha`` and ``gamma**(1-alpha)``.

    ``metrics_validated`` skips the host-side finite check.
    """
    if query.ndim == 0 or key.ndim == 0:
        raise ValueError("query and key must have a head dimension")
    if query.size(0) != key.size(0):
        raise ValueError(
            f"query and key head counts must match, got {query.size(0)} and {key.size(0)}"
        )
    _validate_qk_clip_controls(
        max_logits, threshold, alpha, query.size(0), check_finite=not metrics_validated
    )

    gamma = _qk_clip_gamma(max_logits, threshold, query.device)
    query_scale = gamma.pow(alpha).to(dtype=query.dtype)
    key_scale = gamma.pow(1.0 - alpha).to(dtype=key.dtype)
    query.mul_(query_scale.view(-1, *([1] * (query.ndim - 1))))
    key.mul_(key_scale.view(-1, *([1] * (key.ndim - 1))))


@torch.no_grad()
def qk_clip_mla_(
    query,
    key,
    max_logits,
    threshold,
    alpha,
    query_content_rows,
    query_rotary_rows,
    key_content_rows,
    metrics_validated=False,
):
    """Scale Kimi MLA query/key head blocks in place by their QK-Clip factor.

    Each query head is ``[q_content | q_rotary]`` and each fused KV head is
    ``[k_content | value]``. Content Q and K use ``gamma**alpha`` and
    ``gamma**(1-alpha)``. Rotary Q uses ``gamma``; V and shared rotary K are unscaled.
    """
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("MLA QK-Clip expects [heads, rows, columns] tensors")
    if query.size(0) != key.size(0):
        raise ValueError("MLA QK-Clip query and key head counts must match")
    if not all(
        isinstance(rows, int) and rows > 0
        for rows in (query_content_rows, query_rotary_rows, key_content_rows)
    ):
        raise ValueError("MLA QK-Clip row counts must be positive integers")
    if query_content_rows + query_rotary_rows != query.size(1):
        raise ValueError("MLA QK-Clip query row counts must cover each query head")
    if key_content_rows > key.size(1):
        raise ValueError("MLA QK-Clip key content rows exceed each fused KV head")
    _validate_qk_clip_controls(
        max_logits, threshold, alpha, query.size(0), check_finite=not metrics_validated
    )

    gamma = _qk_clip_gamma(max_logits, threshold, query.device)
    query_scale = gamma.pow(alpha).to(query.dtype)
    key_scale = gamma.pow(1.0 - alpha).to(key.dtype)
    shape = (-1, 1, 1)
    query[:, :query_content_rows].mul_(query_scale.view(shape))
    query[:, query_content_rows:].mul_(gamma.to(query.dtype).view(shape))
    key[:, :key_content_rows].mul_(key_scale.view(shape))


def _validate_qk_clip_controls(max_logits, threshold, alpha, num_heads, check_finite=True):
    """Reject QK-Clip inputs outside the supported domain.

    ``check_finite=False`` skips the host-side finite check to avoid a
    redundant device synchronization.
    """
    if not torch.is_tensor(max_logits) or max_logits.ndim != 1:
        raise ValueError("max_logits must be a 1-D tensor with one value per head")
    if max_logits.numel() != num_heads:
        raise ValueError(
            f"max_logits must have one value per head, got {max_logits.numel()} "
            f"for {num_heads} heads"
        )
    if not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError(f"threshold must be a finite non-negative number, got {threshold!r}")
    if not math.isfinite(float(alpha)) or not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be a finite number in [0, 1], got {alpha!r}")
    if check_finite and not torch.isfinite(max_logits).all():
        raise ValueError("max_logits must contain only finite values")


def _qk_clip_gamma(max_logits, threshold, device):
    """Return threshold/logit for heads above the bound, and 1 for other heads."""
    logits = max_logits.to(device=device, dtype=torch.float32)
    above = logits > threshold
    denominator = torch.where(above, logits, 1.0)
    return torch.where(above, threshold / denominator, 1.0)


def _prepare_qk_clip_tensor(tensor, num_heads, name):
    """View a complete 2-D Q/K projection as independent head blocks."""
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2-D projection tensor")
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("QK-Clip requires a positive integer muon_num_heads")
    if tensor.size(0) % num_heads != 0:
        raise ValueError(
            f"{name} first dimension {tensor.size(0)} is not divisible by "
            f"muon_num_heads={num_heads}"
        )
    return tensor.view(num_heads, tensor.size(0) // num_heads, tensor.size(1))


def _qk_clip_head_blocks(tensor, num_heads):
    """Return local complete head blocks and their first global head index."""
    layout = _single_shard_layout(tensor)
    if layout is None:
        return None
    _, shard_dim, process_group = layout
    if shard_dim != 0 or tensor.ndim != 2:
        return None
    rows, columns = (int(size) for size in tensor.shape)
    if not isinstance(num_heads, int) or num_heads <= 0 or rows % num_heads:
        return None
    world_size = dist.get_world_size(process_group)
    if not _row_shards_preserve_heads(rows, num_heads, world_size):
        return None
    head_rows = rows // num_heads
    start, end = _chunk_range(rows, world_size, dist.get_rank(process_group))
    local = tensor.to_local()
    if tuple(local.shape) != (end - start, columns):
        return None
    return start // head_rows, local.view((end - start) // head_rows, head_rows, columns)


def _clip_head_blocks(query_heads, key_heads, max_logits, group, metrics_validated=False):
    """Apply the group's QK-Clip factors to head-blocked query and key views."""
    key_rows = group.get("qk_clip_key_rows")
    query_content_rows = group.get("qk_clip_query_content_rows")
    query_rotary_rows = group.get("qk_clip_query_rotary_rows")
    alpha = group.get("qk_clip_alpha", 0.5)
    threshold = group["qk_clip_threshold"]
    if query_content_rows is not None or query_rotary_rows is not None:
        if query_content_rows is None or query_rotary_rows is None or key_rows is None:
            raise ValueError(
                "MLA QK-Clip requires content/rotary query rows and key content rows"
            )
        qk_clip_mla_(
            query_heads,
            key_heads,
            max_logits=max_logits,
            threshold=threshold,
            alpha=alpha,
            query_content_rows=query_content_rows,
            query_rotary_rows=query_rotary_rows,
            key_content_rows=key_rows,
            metrics_validated=metrics_validated,
        )
        return
    if key_rows is not None:
        if not isinstance(key_rows, int) or not 0 < key_rows <= key_heads.size(1):
            raise ValueError(
                "qk_clip_key_rows must select a non-empty prefix of each key head"
            )
        key_heads = key_heads[:, :key_rows]
    qk_clip_(
        query_heads,
        key_heads,
        max_logits=max_logits,
        threshold=threshold,
        alpha=alpha,
        metrics_validated=metrics_validated,
    )


def _has_zero_lr(group):
    """Whether the group has an explicit learning rate of exactly zero."""
    lr = group.get("lr")
    return lr is not None and float(lr) == 0.0


@torch.no_grad()
def _apply_qk_clip_groups(param_groups):
    """Reduce metrics and clip in one call, for callers outside ``Muon.step``."""
    _apply_qk_clip_updates(_prepare_qk_clip_groups(param_groups))


@torch.no_grad()
def _apply_qk_clip_updates(updates):
    """Apply already-validated clip factors and release pending metrics."""
    for group, max_logits in updates:
        _apply_qk_clip_group(group, max_logits)
        group.pop(_QK_CLIP_PENDING_METRICS, None)


@torch.no_grad()
def _prepare_qk_clip_groups(param_groups):
    """Consume, reduce and validate metrics; return the clip factors to apply.

    Raises on missing, wrongly shaped or non-finite metrics. Caches metrics
    only after all groups pass metric validation, so a later incomplete update can
    retry. Error flags are reduced before raising on any rank.
    """
    buckets = {}
    for group in param_groups:
        has_params = "qk_clip_params" in group
        provider = group.get("qk_clip_metric_provider")
        has_metrics = "qk_clip_max_logits" in group or provider is not None
        if not has_params and not has_metrics:
            continue
        if _has_zero_lr(group):
            # No weight update happened, for example in the zero-learning-rate
            # step that PyTorch distributed checkpointing runs to create the
            # optimizer state. Leave Q/K unchanged and keep the recorded
            # metrics for the next step. This also applies to a real step
            # with a learning rate of 0 (e.g. warmup starting at 0): the clip
            # is delayed to the next iteration. The learning rate is the same
            # on every rank, so all ranks skip the same groups.
            continue
        if group.get("qk_clip_threshold") is None:
            raise ValueError("QK-Clip metadata requires qk_clip_threshold")
        if not has_params or not has_metrics:
            raise ValueError("QK-Clip requires Q/K parameters and a metric provider or max_logits")
        params = group["qk_clip_params"]
        if not isinstance(params, (tuple, list)) or len(params) != 2:
            raise ValueError("qk_clip_params must be a (query, key) pair")
        query, key = params
        if isinstance(query, DTensor) != isinstance(key, DTensor):
            raise ValueError("QK-Clip query and key must use the same tensor layout")
        if isinstance(query, DTensor):
            if query.device_mesh != key.device_mesh:
                raise ValueError("QK-Clip query and key must use the same device mesh")
            process_groups = query.device_mesh.get_all_groups()
        else:
            process_groups = group.get("qk_clip_process_groups", ())
        process_groups = tuple(pg for pg in process_groups if dist.get_world_size(pg) > 1)
        num_heads = group.get("muon_num_heads")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("QK-Clip requires a positive integer muon_num_heads")
        # Reuse metrics cached by a previous incomplete step.
        max_logits = group.pop(_QK_CLIP_PENDING_METRICS, None)
        if max_logits is None:
            max_logits = provider() if provider is not None else group["qk_clip_max_logits"]
            if isinstance(max_logits, DTensor):
                max_logits = max_logits.full_tensor()
        status = _METRICS_OK
        if max_logits is None:
            status = _METRICS_MISSING
        elif not torch.is_tensor(max_logits) or max_logits.shape != (num_heads,):
            status = _METRICS_WRONG_SHAPE
        if status != _METRICS_OK:
            max_logits = torch.full((num_heads,), -torch.inf, device=query.device)
        metrics = max_logits.detach().to(device=query.device, dtype=torch.float32)
        buckets.setdefault((process_groups, query.device), []).append((group, metrics, status))

    reduced = []
    for (process_groups, device), entries in buckets.items():
        flags = torch.tensor([status for _, _, status in entries], device=device, dtype=torch.float32)
        metrics = torch.cat([entry[1] for entry in entries])
        # MAX does not reliably propagate NaNs on every backend.
        invalid = (torch.isnan(metrics) | torch.isposinf(metrics)).any().float().view(1)
        packet = torch.cat((invalid, flags, metrics))
        for process_group in process_groups:
            dist.all_reduce(packet, op=dist.ReduceOp.MAX, group=process_group)
        reduced.append((entries, packet))

    updates = []
    for entries, packet in reduced:
        metrics = packet[1 + len(entries):]
        # Single host transfer per bucket: read header flags and the
        # post-reduction finite check together. -Inf after MAX reduction
        # means no rank reported that head.
        header = torch.cat((
            packet[:1 + len(entries)],
            (~torch.isfinite(metrics)).any().to(packet.dtype).view(1),
        )).tolist()
        invalid, statuses, not_finite = header[0], header[1:-1], header[-1]
        if _METRICS_WRONG_SHAPE in statuses:
            raise ValueError("QK-Clip max_logits must have one value per head on every rank")
        if _METRICS_MISSING in statuses:
            raise RuntimeError(
                "QK-Clip did not receive fresh per-head logits since the previous "
                "optimizer step on every rank"
            )
        if invalid or not_finite:
            raise ValueError("max_logits must contain only finite values after reduction")
        head_counts = [group["muon_num_heads"] for group, _, _ in entries]
        for (group, _, _), max_logits in zip(entries, metrics.split(head_counts)):
            updates.append((group, max_logits))
    for group, max_logits in updates:
        group[_QK_CLIP_PENDING_METRICS] = max_logits
    # A corrected control can retry metrics already consumed from attention.
    for group, max_logits in updates:
        _validate_qk_clip_controls(
            max_logits, group["qk_clip_threshold"], group.get("qk_clip_alpha", .5),
            group["muon_num_heads"], check_finite=False,
        )
    return updates


def _apply_qk_clip_group(group, max_logits, metrics_validated=True):
    """Apply already reduced metrics to one Q/K parameter pair."""
    query, key = group["qk_clip_params"]
    num_heads = group.get("muon_num_heads")
    query_is_dtensor = isinstance(query, DTensor)
    if query_is_dtensor:
        local_query = _qk_clip_head_blocks(query, num_heads)
        local_key = _qk_clip_head_blocks(key, num_heads)
        if (
            local_query is not None
            and local_key is not None
            and local_query[0] == local_key[0]
            and local_query[1].size(0) == local_key[1].size(0)
        ):
            first_head, query_heads = local_query
            _, key_heads = local_key
            _clip_head_blocks(
                query_heads,
                key_heads,
                max_logits[first_head:first_head + query_heads.size(0)],
                group,
                metrics_validated=metrics_validated,
            )
            return
        replicated = [Replicate()] * query.device_mesh.ndim
        query_full, key_full = query.full_tensor(), key.full_tensor()
    else:
        query_full, key_full = query, key

    query_heads = _prepare_qk_clip_tensor(query_full, num_heads, "query")
    key_heads = _prepare_qk_clip_tensor(key_full, num_heads, "key")
    _clip_head_blocks(
        query_heads, key_heads, max_logits, group, metrics_validated=metrics_validated
    )
    if query_is_dtensor:
        _copy_replicated_to_dtensor_(query, query_full, replicated)
        _copy_replicated_to_dtensor_(key, key_full, replicated)
