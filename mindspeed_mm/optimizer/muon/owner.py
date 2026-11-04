# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Orthogonalize row-sharded 2-D weights on owner ranks.

Each full matrix is assigned to one "owner" rank, balancing the estimated
Newton-Schulz FLOPs across ranks. The owner gathers the matrix's row bands
from the other ranks with a variable-size all-to-all, orthogonalizes it, and
sends each rank its rows of the result back. Matrices are exchanged in
"waves" that bound the bytes each owner gathers at once, and up to
``max_inflight`` waves overlap communication with compute. The per-head and
per-expert variants of this exchange live in ``distributed.py``.
"""

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from .coefficients import resolve_ns_coefficients
from .dtensor_utils import _chunk_range, _single_shard_layout, _flat_send, _peer_views
from .newton_schulz import (
    _newton_schulz,
    newton_schulz_dtype,
    newton_schulz_cost,
    uses_gram_newton_schulz,
    zeropower_via_newtonschulz5,
)


# An owner stacks same-shaped matrices into one batched Newton-Schulz call.
# Each stacked input is limited to 512 MiB (not configurable). The gathered
# matrices and the Newton-Schulz working memory are allocated separately.
_OWNER_BATCH_BYTES = 512 << 20


@lru_cache(maxsize=None)
def _mirror_group(group):
    """Build a second process group over every rank of ``group``."""
    return dist.new_group()


def _return_exchange_group(group):
    """Return the process group the owner results are sent back on.

    Each process group gets its own HCCL stream, so a wave's gather queues
    behind the previous wave's return exchange. Setting
    MINDSPEED_MM_MUON_OWNER_SCATTER_COMM=1 separates the two directions. Only
    a group spanning every rank is mirrored, since ``new_group`` is collective.
    """
    if os.getenv("MINDSPEED_MM_MUON_OWNER_SCATTER_COMM", "0") != "1":
        return group
    if dist.get_world_size(group) != dist.get_world_size():
        return group
    return _mirror_group(group)


def _full_matrix_owner_layout(parameter, group):
    """Return ``(group, rows, columns)`` for a simple 2-D row shard."""
    if group.get("muon_mode", "full_matrix") != "full_matrix":
        return None
    if not isinstance(parameter, DTensor) or parameter.ndim != 2:
        return None
    layout = _single_shard_layout(parameter)
    if layout is None:
        return None
    _, shard_dim, process_group = layout
    if shard_dim != 0 or dist.get_world_size(process_group) < 2:
        return None
    return process_group, int(parameter.shape[0]), int(parameter.shape[1])


def _is_expected_row_band(ns_input, parameter, shape, world_size, rank):
    """Check that a momentum shard matches its expected contiguous row band."""
    if ns_input.placements != parameter.placements:
        return False
    rows, columns = (int(size) for size in shape)
    start, end = _chunk_range(rows, world_size, rank)
    return tuple(ns_input.to_local().shape) == (end - start, columns)


@lru_cache(maxsize=64)
def _owner_schedule(shapes, world_size, steps, element_size, byte_budget, device_type):
    """Return ``(owner of each matrix, waves)``, cached because shapes rarely change."""
    owners = _owners(shapes, world_size, steps, device_type, element_size)
    waves = _waves(shapes, owners, world_size, element_size, byte_budget)
    return tuple(owners), tuple(tuple(wave) for wave in waves)


def _estimated_owner_batches(shapes, world_size, element_size):
    """Estimate the batch size each shape will be orthogonalized with.

    Assumes each shape spreads evenly over the ranks, capped by the same
    byte limit the batching uses. Rounds down to avoid over-crediting.
    """
    counts = Counter(shapes)
    batches = {}
    for shape, count in counts.items():
        rows, columns = shape
        matrix_bytes = int(rows) * int(columns) * element_size
        cap = max(1, _OWNER_BATCH_BYTES // matrix_bytes) if matrix_bytes else 1
        batches[shape] = max(1, min(count // world_size, cap))
    return batches


def _owners(shapes, world_size, steps, device_type, element_size):
    """Assign each matrix to an owner rank: largest cost first, to the least-loaded rank."""
    batches = _estimated_owner_batches(shapes, world_size, element_size)
    costs = [
        newton_schulz_cost(*shape, steps, device_type, batches[shape])
        for shape in shapes
    ]
    assignment = [0] * len(shapes)
    loads = [0] * world_size
    for index in sorted(
        range(len(shapes)), key=lambda item: (-costs[item], item)
    ):
        owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
        assignment[index] = owner
        loads[owner] += costs[index]
    return assignment


def _waves(shapes, owners, world_size, element_size, byte_budget):
    """Split matrix indices, in order, into waves.

    A new wave starts when the next matrix would make its owner gather more
    than ``byte_budget`` bytes of full matrices in the current wave. A single
    matrix larger than the budget gets a wave of its own.
    """
    result, wave = [], []
    held = [0] * world_size
    for index, (rows, columns) in enumerate(shapes):
        size = int(rows) * int(columns) * element_size
        owner = owners[index]
        if wave and held[owner] + size > byte_budget:
            result.append(wave)
            wave, held = [], [0] * world_size
        wave.append(index)
        held[owner] += size
    if wave:
        result.append(wave)
    return result


@dataclass
class _Gather:
    """State and buffers kept alive until an asynchronous gather completes."""

    locals_: list
    shapes: list
    owners: list
    bands: list
    owned: list
    input: torch.Tensor
    output: torch.Tensor
    output_splits: list
    work: object
    group: object
    steps: int
    return_dtype: torch.dtype
    coefficients: object


@dataclass
class _Scatter:
    """State and buffers kept alive until an asynchronous scatter completes."""

    updates: list
    owners: list
    bands: list
    input: torch.Tensor
    output: torch.Tensor
    output_splits: list
    work: object
    group: object


def _numel(shapes, bands, index, peer):
    start, end = bands[index][peer]
    return (end - start) * int(shapes[index][1])


def _launch_gather(locals_, shapes, owners, group, steps, return_dtype, coefficients):
    world_size, rank = dist.get_world_size(group), dist.get_rank(group)
    reference = locals_[0]
    bands = [
        [_chunk_range(int(rows), world_size, peer) for peer in range(world_size)]
        for rows, _ in shapes
    ]
    owned = [index for index, owner in enumerate(owners) if owner == rank]
    sends = {}
    for owner in range(world_size):
        if owner == rank:
            continue
        pieces = [
            locals_[index].reshape(-1)
            for index, target in enumerate(owners)
            if target == owner and _numel(shapes, bands, index, rank)
        ]
        if pieces:
            sends[owner] = pieces
    input_buffer, input_splits = _flat_send(sends, reference, world_size)
    output_splits = [
        sum(_numel(shapes, bands, index, source) for index in owned)
        if source != rank
        else 0
        for source in range(world_size)
    ]
    output = reference.new_empty(sum(output_splits))
    work = dist.all_to_all_single(
        output,
        input_buffer,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
        async_op=True,
    )
    return _Gather(
        locals_,
        shapes,
        owners,
        bands,
        owned,
        input_buffer,
        output,
        output_splits,
        work,
        group,
        steps,
        return_dtype,
        coefficients,
    )


def _compute_owned_updates(assembled, steps, coefficients, return_dtype):
    """Batch equally shaped matrices without combining their normalization."""
    grouped = defaultdict(list)
    for index, matrix in assembled.items():
        grouped[tuple(matrix.shape)].append(index)
    updates = {}
    for indices in grouped.values():
        reference = assembled[indices[0]]
        batch_size = max(
            1, _OWNER_BATCH_BYTES // (reference.numel() * reference.element_size())
        )
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            if len(batch) == 1:
                updates[batch[0]] = zeropower_via_newtonschulz5(
                    assembled[batch[0]], steps, coefficients, out_dtype=return_dtype
                )
            else:
                inputs = torch.stack([assembled[index] for index in batch])
                output = _newton_schulz(
                    inputs, steps, coefficients, out_dtype=return_dtype
                )
                updates.update(zip(batch, output.unbind()))
    return updates


def _finish_gather(gather):
    gather.work.wait()
    world_size, rank = dist.get_world_size(gather.group), dist.get_rank(gather.group)
    reference = gather.locals_[0]
    received = _peer_views(gather.output, gather.output_splits)
    assembled = {}
    for index in gather.owned:
        rows, columns = gather.shapes[index]
        assembled[index] = reference.new_empty((rows, columns))
        start, end = gather.bands[index][rank]
        if end > start:
            assembled[index][start:end].copy_(gather.locals_[index])
    for source, packed in received.items():
        offset = 0
        for index in gather.owned:
            start, end = gather.bands[index][source]
            count = _numel(gather.shapes, gather.bands, index, source)
            if count:
                assembled[index][start:end].copy_(
                    packed[offset : offset + count].view(
                        end - start, gather.shapes[index][1]
                    )
                )
                offset += count
    # Updates are returned, and sent back to the other ranks, in
    # gather.return_dtype, the Newton-Schulz compute dtype: FP16 for Gram
    # batches, BF16 for standard batches (see pipelined_owner_updates).
    owned_updates = _compute_owned_updates(
        assembled, gather.steps, gather.coefficients, gather.return_dtype
    )
    local_updates = [
        reference.new_empty(
            (
                gather.bands[index][rank][1] - gather.bands[index][rank][0],
                gather.shapes[index][1],
            ),
            dtype=gather.return_dtype,
        )
        for index in range(len(gather.shapes))
    ]
    for index in gather.owned:
        start, end = gather.bands[index][rank]
        if end > start:
            local_updates[index].copy_(owned_updates[index][start:end])

    sends = {}
    for peer in range(world_size):
        if peer == rank:
            continue
        pieces = []
        for index in gather.owned:
            start, end = gather.bands[index][peer]
            if end > start:
                pieces.append(owned_updates[index][start:end].reshape(-1))
        if pieces:
            sends[peer] = pieces
    input_buffer, input_splits = _flat_send(sends, local_updates[0], world_size)
    output_splits = [
        sum(
            _numel(gather.shapes, gather.bands, index, rank)
            for index, owner in enumerate(gather.owners)
            if owner == peer
        )
        if peer != rank
        else 0
        for peer in range(world_size)
    ]
    output = local_updates[0].new_empty(sum(output_splits))
    work = dist.all_to_all_single(
        output,
        input_buffer,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=_return_exchange_group(gather.group),
        async_op=True,
    )
    return _Scatter(
        local_updates,
        gather.owners,
        gather.bands,
        input_buffer,
        output,
        output_splits,
        work,
        _return_exchange_group(gather.group),
    )


def _finish_scatter(scatter):
    scatter.work.wait()
    for owner, packed in _peer_views(scatter.output, scatter.output_splits).items():
        offset = 0
        for index, target in enumerate(scatter.owners):
            if target != owner:
                continue
            count = scatter.updates[index].numel()
            if count:
                scatter.updates[index].copy_(
                    packed[offset : offset + count].view_as(scatter.updates[index])
                )
                offset += count
    return scatter.updates


def pipelined_owner_updates(
    local_shards,
    shapes,
    process_group,
    steps,
    *,
    wave_bytes,
    max_inflight=2,
    coefficients=None,
):
    """Orthogonalize row-sharded matrices; return this rank's rows of each result.

    ``local_shards[i]`` is this rank's row band of a matrix of ``shapes[i]``.
    Waves are processed in order: up to ``max_inflight`` gathers are started
    ahead, each finished gather is orthogonalized on its owners and its
    results are sent back asynchronously, and at most ``max_inflight`` of
    those return exchanges stay pending. Every rank issues the collectives in
    the same order.

    All matrices of one call must use the same Newton-Schulz variant (all Gram
    or all standard, see ``uses_gram_newton_schulz``); the results are
    returned in its compute dtype.
    """
    if not local_shards:
        return []
    if wave_bytes <= 0:
        raise ValueError("Muon owner wave_bytes must be positive")
    if max_inflight < 1:
        raise ValueError("Muon owner max_inflight must be positive")
    # shapes are full-matrix shapes and steps is a group setting, so every rank
    # makes the same decision here, before any collective.
    gram_flags = {uses_gram_newton_schulz(rows, columns, steps) for rows, columns in shapes}
    if len(gram_flags) != 1:
        raise ValueError(
            "pipelined_owner_updates requires all matrices in one call to use the "
            "same Newton-Schulz variant (all Gram or all standard); got a mix of "
            f"shapes {list(shapes)} for {steps} steps"
        )
    world_size = dist.get_world_size(process_group)
    owners, waves = _owner_schedule(
        tuple(tuple(shape) for shape in shapes),
        world_size,
        steps,
        local_shards[0].element_size(),
        wave_bytes,
        local_shards[0].device.type,
    )
    coefficients = resolve_ns_coefficients(coefficients)
    # Updates are exchanged in the Newton-Schulz compute dtype (FP16 for Gram,
    # BF16 for standard), because upcasting would only double the traffic.
    rows, columns = shapes[0]
    return_dtype = newton_schulz_dtype(rows, columns, steps)
    # _finish_scatter returns newly allocated tensors, so store them directly.
    results = [None] * len(local_shards)
    gathers, scatters, next_wave = {}, [], 0

    def launch(wave_index):
        indices = waves[wave_index]
        gathers[wave_index] = _launch_gather(
            [local_shards[index] for index in indices],
            [shapes[index] for index in indices],
            [owners[index] for index in indices],
            process_group,
            steps,
            return_dtype,
            coefficients,
        )

    while next_wave < min(max_inflight, len(waves)):
        launch(next_wave)
        next_wave += 1
    for wave_index in range(len(waves)):
        scatters.append((wave_index, _finish_gather(gathers.pop(wave_index))))
        if next_wave < len(waves):
            launch(next_wave)
            next_wave += 1
        if len(scatters) >= max_inflight:
            completed_index, completed = scatters.pop(0)
            for index, update in zip(waves[completed_index], _finish_scatter(completed)):
                results[index] = update
    for completed_index, completed in scatters:
        for index, update in zip(waves[completed_index], _finish_scatter(completed)):
            results[index] = update
    return results


__all__ = ["pipelined_owner_updates"]
