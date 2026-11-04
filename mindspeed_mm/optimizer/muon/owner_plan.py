# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Check cross-rank agreement and batch the row-sharded weights for owner.py.

Before any rank updates momentum or exchanges row shards, one all-reduce per
process group checks that every rank has the same row-sharded 2-D weights,
shapes, dtypes, placements, gradient presence and Newton-Schulz settings. A
mismatch would otherwise hang the all-to-all, so it raises an error instead.
All ranks must build the same parameter groups and DTensor meshes, as FSDP
already requires. A missing gradient is allowed only if it is missing on
every rank.
"""

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from .coefficients import resolve_ns_coefficients
from .newton_schulz import uses_gram_newton_schulz
from .owner import _full_matrix_owner_layout, _is_expected_row_band


@dataclass
class _MeshWork:
    """The row-sharded weights of one process group, and whether all are row bands."""

    group: object
    descriptors: list = field(default_factory=list)
    entries: list = field(default_factory=list)
    layout_valid: bool = True


@lru_cache(maxsize=128)
def _manifest_digest(settings, descriptors):
    """Hash of the parameter manifest, cached because it rarely changes between steps."""
    manifest = repr((settings, descriptors))
    digest = hashlib.blake2b(manifest.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _validate_manifest(descriptors, group, device, layout_valid, settings):
    """Check with one all-reduce that every rank has the same manifest hash.

    Runs every step, even when the hash is cached or no gradient is present,
    so all ranks always issue the same collective. Returns whether every rank
    can use the row-band exchange.
    """
    digest = _manifest_digest(settings, tuple(descriptors))
    value = torch.tensor([digest, -digest, int(layout_valid)],
                         dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MIN, group=group)
    minimum, negative_maximum, valid = value.tolist()
    if minimum != -negative_maximum:
        raise RuntimeError("Muon owner matrix manifest differs across DTensor mesh ranks; "
                           "check parameter identities, gradients and scheduling settings")
    return bool(valid)


def collect_owner_batches(param_groups, *, wave_bytes, max_inflight):
    """Group row-sharded weights with gradients into batches for owner.py.

    Returns ``(batches, parameter_ids)``: batches keyed by process group,
    dtype, step count, coefficient schedule and Newton-Schulz variant, and the
    ``id()`` of every batched parameter (the others use the per-parameter path).
    The cross-rank check runs per process group *before* this split, because
    the split keys themselves could differ between ranks. Group and parameter
    positions (and names when given) tell apart matrices of the same shape.
    """
    meshes, schedules = {}, {}
    for group_index, group in enumerate(param_groups):
        if not group.get("use_muon", False):
            continue
        names = group.get("param_names")
        for parameter_index, parameter in enumerate(group["params"]):
            layout = _full_matrix_owner_layout(parameter, group)
            if layout is None:
                continue
            process_group, rows, columns = layout
            record = meshes.setdefault(id(process_group), _MeshWork(process_group))
            identity = (group_index, parameter_index,
                        names[parameter_index] if names is not None and len(names) == len(group["params"]) else None)
            if id(group) not in schedules:
                schedules[id(group)] = resolve_ns_coefficients(group.get("ns_coefficients"))
            coefficients = schedules[id(group)]
            record.descriptors.append((
                identity, (rows, columns), parameter.dtype, tuple(parameter.placements),
                group["ns_steps"], coefficients, group["momentum"], group["nesterov"],
                parameter.grad is not None,
            ))
            record.entries.append((parameter, group, rows, columns))
            if parameter.grad is not None:
                record.layout_valid &= isinstance(parameter.grad, DTensor) and _is_expected_row_band(
                    parameter.grad, parameter, (rows, columns),
                    dist.get_world_size(process_group), dist.get_rank(process_group))

    batches, deferred_ids = {}, set()
    for record in meshes.values():
        process_group, entries = record.group, record.entries
        valid = _validate_manifest(record.descriptors, process_group, entries[0][0].to_local().device,
                                   record.layout_valid, (wave_bytes, max_inflight))
        if wave_bytes <= 0 or max_inflight < 1:
            raise ValueError("Muon owner scheduling settings must be positive")
        # All ranks take the same fallback when any rank cannot use row bands.
        if not valid:
            continue
        for parameter, group, rows, columns in entries:
            if parameter.grad is None:
                continue
            key = (id(process_group), parameter.dtype, group["ns_steps"],
                   schedules[id(group)],
                   uses_gram_newton_schulz(rows, columns, group["ns_steps"]))
            batches.setdefault(key, (process_group, []))[1].append((parameter, group, rows, columns))
            deferred_ids.add(id(parameter))
    return batches, deferred_ids
