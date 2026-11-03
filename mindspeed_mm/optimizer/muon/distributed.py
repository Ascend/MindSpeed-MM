"""Sharded per-head and per-expert Muon updates.

Like ``owner.py`` for whole matrices, each attention head or expert is
assigned to one owner rank (head or expert ``i`` goes to rank
``i % world_size``). The owner assembles it from the pieces held by the other
ranks with an all-to-all, orthogonalizes it, and sends each rank its pieces
of the result back. The path taken for each parameter is chosen in
``_per_head_dtensor_update`` and ``_per_expert_dtensor_update``.
"""

from functools import lru_cache
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard

from .dtensor_utils import (
    _chunk_range,
    _from_local_like,
    _row_shards_preserve_heads,
    _run_all_to_all,
    _single_shard_layout,
)
from .newton_schulz import (
    _newton_schulz,
    newton_schulz_dtype,
    uses_column_sharded_gram_newton_schulz,
    zeropower_via_newtonschulz5_per_expert,
    zeropower_via_newtonschulz5_per_head,
)


def _muon_compute_placements(tensor, mode):
    """Return the placements to all-gather a tensor into for the fallback path.

    Every matrix shard becomes replicated, except that per-expert tensors keep
    their expert-parallel ``Shard(0)`` on the ``ep`` mesh dimension, because
    experts are orthogonalized independently.
    """
    mesh_dim_names = getattr(tensor.device_mesh, "mesh_dim_names", None)
    placements = []
    for mesh_dim, placement in enumerate(tensor.placements):
        mesh_name = mesh_dim_names[mesh_dim] if mesh_dim_names is not None else None
        keep_expert_shard = (
            mode == "per_expert"
            and mesh_name == "ep"
            and isinstance(placement, Shard)
            and placement.dim == 0
        )
        placements.append(
            placement
            if keep_expert_shard or not isinstance(placement, Shard)
            else Replicate()
        )
    return tuple(placements)


def _moving_shard_layout(tensor, mode):
    """Return ``(mesh_dim, shard_dim, group)`` of the one shard Muon must undo.

    Shards that ``_muon_compute_placements`` keeps (the expert-parallel shard)
    are ignored. Returns None unless exactly one plain ``Shard`` must move.
    """
    kept = _muon_compute_placements(tensor, mode)
    if any(
        not isinstance(placement, (Replicate, Shard))
        for placement in tensor.placements
    ):
        return None
    moving = [
        (mesh_dim, placement.dim)
        for mesh_dim, (placement, keep) in enumerate(zip(tensor.placements, kept))
        if isinstance(placement, Shard) and not isinstance(keep, Shard)
    ]
    if len(moving) != 1:
        return None
    mesh_dim, shard_dim = moving[0]
    if type(tensor.placements[mesh_dim]) is not Shard:
        # Owner exchange requires one contiguous chunk per rank.
        return None
    return mesh_dim, shard_dim, tensor.device_mesh.get_group(mesh_dim)


class _HeadFragment(NamedTuple):
    """A rectangular piece of one head that one rank sends to the head's owner.

    ``source_*`` is its position in the sending rank's local shard, and
    ``head_*`` its position inside the assembled ``[head_rows, columns]`` head
    on the owner.
    """

    head: int
    source_row0: int
    source_row1: int
    source_column0: int
    source_column1: int
    head_row0: int
    head_row1: int
    head_column0: int
    head_column1: int

    @property
    def numel(self):
        return (self.source_row1 - self.source_row0) * (
            self.source_column1 - self.source_column0
        )

    def source_view(self, local):
        return local[
            self.source_row0:self.source_row1,
            self.source_column0:self.source_column1,
        ]

    def head_view(self, owned, owner_rank, world_size):
        owned_head = (self.head - owner_rank) // world_size
        return owned[
            owned_head,
            self.head_row0:self.head_row1,
            self.head_column0:self.head_column1,
        ]


@lru_cache(maxsize=4096)
def _head_fragments(
    source_rank, owner_rank, world_size, rows, columns, num_heads, shard_dim
):
    """List the pieces of ``owner_rank``'s heads that ``source_rank`` holds.

    With row shards (``shard_dim == 0``) a piece is the overlap of a head's
    rows with the source rank's rows; with column shards it is the head's full
    rows over the source rank's columns. The result depends only on the
    shapes and ranks, so it is cached across steps.
    """
    head_rows = rows // num_heads
    specs = []
    if shard_dim == 0:
        shard_start, shard_end = _chunk_range(rows, world_size, source_rank)
        for head in range(owner_rank, num_heads, world_size):
            head_start = head * head_rows
            overlap_start = max(shard_start, head_start)
            overlap_end = min(shard_end, head_start + head_rows)
            if overlap_start < overlap_end:
                specs.append(
                    _HeadFragment(
                        head=head,
                        source_row0=overlap_start - shard_start,
                        source_row1=overlap_end - shard_start,
                        source_column0=0,
                        source_column1=columns,
                        head_row0=overlap_start - head_start,
                        head_row1=overlap_end - head_start,
                        head_column0=0,
                        head_column1=columns,
                    )
                )
    else:
        shard_start, shard_end = _chunk_range(columns, world_size, source_rank)
        if shard_start == shard_end:
            return ()
        for head in range(owner_rank, num_heads, world_size):
            specs.append(
                _HeadFragment(
                    head=head,
                    source_row0=head * head_rows,
                    source_row1=(head + 1) * head_rows,
                    source_column0=0,
                    source_column1=shard_end - shard_start,
                    head_row0=0,
                    head_row1=head_rows,
                    head_column0=shard_start,
                    head_column1=shard_end,
                )
            )
    return tuple(specs)


def _fragment_numel(specs):
    return sum(fragment.numel for fragment in specs)


def _pack_local_fragments(local, specs):
    pieces = [fragment.source_view(local).reshape(-1) for fragment in specs]
    if not pieces:
        return local.new_empty(0)
    return torch.cat(pieces).contiguous()


def _unpack_owner_fragments_(owned, packed, specs, owner_rank, world_size):
    offset = 0
    for fragment in specs:
        destination = fragment.head_view(owned, owner_rank, world_size)
        destination.copy_(
            packed[offset:offset + fragment.numel].view_as(destination)
        )
        offset += fragment.numel


def _pack_owner_fragments(owned, specs, owner_rank, world_size):
    pieces = [
        fragment.head_view(owned, owner_rank, world_size).reshape(-1)
        for fragment in specs
    ]
    if not pieces:
        return owned.new_empty(0)
    return torch.cat(pieces).contiguous()


def _unpack_local_fragments_(local, packed, specs):
    offset = 0
    for fragment in specs:
        destination = fragment.source_view(local)
        destination.copy_(
            packed[offset:offset + fragment.numel].view_as(destination)
        )
        offset += fragment.numel


def _validate_fragment_coverage(spec_groups, expected_numel, name):
    """Raise unless the fragments' total size equals the buffer's size.

    The receiving buffers are allocated uninitialized, so a missing fragment
    would silently leave garbage in the update.
    """
    covered = sum(_fragment_numel(specs) for specs in spec_groups)
    if covered != expected_numel:
        raise RuntimeError(
            f"Muon per-head fragments cover {covered} of {expected_numel} "
            f"{name} elements"
        )


def _per_head_owner_all_to_all(
    local, global_shape, num_heads, shard_dim, process_group, steps, coefficients=None
):
    """Owner exchange for per-head Muon; returns this rank's shard of the update.

    Each rank assembles only the heads it owns from the other ranks' fragments,
    orthogonalizes them, and sends every fragment of the result back to the
    rank that holds it.
    """
    rows, columns = map(int, global_shape)
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    head_rows = rows // num_heads
    owned_count = len(range(rank, num_heads, world_size))
    owned = local.new_empty((owned_count, head_rows, columns))

    specs_by_owner = {
        owner: _head_fragments(
            rank, owner, world_size, rows, columns, num_heads, shard_dim
        )
        for owner in range(world_size)
    }
    send_buffers = {
        owner: _pack_local_fragments(local, specs)
        for owner, specs in specs_by_owner.items()
        if owner != rank and specs
    }
    specs_by_source = {
        source: _head_fragments(
            source, rank, world_size, rows, columns, num_heads, shard_dim
        )
        for source in range(world_size)
    }
    receive_buffers = {
        source: local.new_empty(_fragment_numel(specs))
        for source, specs in specs_by_source.items()
        if source != rank and specs
    }
    _validate_fragment_coverage(specs_by_owner.values(), local.numel(), "local shard")
    _validate_fragment_coverage(specs_by_source.values(), owned.numel(), "owned heads")

    self_specs = specs_by_source[rank]
    if self_specs:
        _unpack_owner_fragments_(
            owned,
            _pack_local_fragments(local, self_specs),
            self_specs,
            rank,
            world_size,
        )
    _run_all_to_all(receive_buffers, send_buffers, local, process_group)
    for source, packed in receive_buffers.items():
        _unpack_owner_fragments_(
            owned, packed, specs_by_source[source], rank, world_size
        )

    # Updates are exchanged in the Newton-Schulz compute dtype: upcasting would
    # only double the traffic. Ranks without owned heads use the same dtype.
    update_dtype = newton_schulz_dtype(head_rows, columns, steps)
    if owned_count:
        owned_update = zeropower_via_newtonschulz5_per_head(
            owned, num_heads=owned_count, steps=steps, coefficients=coefficients
        )
    else:
        owned_update = owned.to(update_dtype)

    return_send_buffers = {
        source: _pack_owner_fragments(owned_update, specs, rank, world_size)
        for source, specs in specs_by_source.items()
        if source != rank and specs
    }
    local_update = torch.empty_like(local, dtype=update_dtype)
    return_receive_buffers = {
        owner: local_update.new_empty(_fragment_numel(specs))
        for owner, specs in specs_by_owner.items()
        if owner != rank and specs
    }
    if self_specs:
        _unpack_local_fragments_(
            local_update,
            _pack_owner_fragments(owned_update, self_specs, rank, world_size),
            self_specs,
        )
    _run_all_to_all(
        return_receive_buffers,
        return_send_buffers,
        local_update,
        process_group,
    )
    for owner, packed in return_receive_buffers.items():
        _unpack_local_fragments_(local_update, packed, specs_by_owner[owner])
    return local_update


def _per_head_dtensor_update(parameter, ns_input, group):
    """Return the per-head update of a sharded weight without gathering all heads.

    Cases, in order:
    - one rank: orthogonalize the local heads directly;
    - row shards whose boundaries fall between heads: each rank already holds
      complete heads, so no communication is needed;
    - column shards where Gram Newton-Schulz applies: orthogonalize in place,
      summing the partial Gram matrices across ranks;
    - otherwise: the owner exchange in ``_per_head_owner_all_to_all``.
    Returns None (use the all-gather fallback) for layouts it cannot handle.
    """
    layout = _single_shard_layout(ns_input)
    if layout is None:
        return None
    _, shard_dim, process_group = layout
    if shard_dim not in (0, 1):
        return None

    rows, columns = map(int, ns_input.shape)
    num_heads = group["muon_num_heads"]
    if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
        raise ValueError("per-head Muon requires a positive integer muon_num_heads")
    if rows <= 0 or rows % num_heads:
        raise ValueError("per-head Muon requires a positive row count divisible by muon_num_heads")
    head_rows = rows // num_heads
    local = ns_input.to_local()
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    shard_extent = rows if shard_dim == 0 else columns
    shard_start, shard_end = _chunk_range(shard_extent, world_size, rank)
    expected_shape = (
        (shard_end - shard_start, columns)
        if shard_dim == 0
        else (rows, shard_end - shard_start)
    )
    if tuple(local.shape) != expected_shape:
        return None

    if world_size == 1:
        local_heads = local.reshape(num_heads, head_rows, columns)
        local_update = zeropower_via_newtonschulz5_per_head(
            local_heads,
            num_heads=num_heads,
            steps=group["ns_steps"],
            coefficients=group.get("ns_coefficients"),
        ).reshape_as(local)
    elif shard_dim == 0 and _row_shards_preserve_heads(
        rows, num_heads, world_size
    ):
        local_heads = local.reshape(local.size(0) // head_rows, head_rows, columns)
        # Complete heads need no exchange; a rank past the final head has no work.
        local_update = local.to(
            newton_schulz_dtype(head_rows, columns, group["ns_steps"])
        ) if local_heads.size(0) == 0 else zeropower_via_newtonschulz5_per_head(
            local_heads,
            num_heads=local_heads.size(0),
            steps=group["ns_steps"],
            coefficients=group.get("ns_coefficients"),
        ).reshape_as(local)
    elif shard_dim == 1 and uses_column_sharded_gram_newton_schulz(
        head_rows, columns, group["ns_steps"]
    ):
        local_heads = local.reshape(num_heads, head_rows, local.size(1))
        local_update = _newton_schulz(
            local_heads,
            steps=group["ns_steps"],
            process_group=process_group,
            coefficients=group.get("ns_coefficients"),
        ).reshape_as(local)
    else:
        local_update = _per_head_owner_all_to_all(
            local,
            (rows, columns),
            num_heads,
            shard_dim,
            process_group,
            group["ns_steps"],
            group.get("ns_coefficients"),
        )

    return _from_local_like(local_update, parameter)


def _expert_owners(num_experts, world_size):
    """Assign expert ``e`` to group rank ``e % world_size``."""
    return [list(range(owner, num_experts, world_size)) for owner in range(world_size)]


def _per_expert_owner_all_to_all(local, rows, process_group, steps, coefficients=None):
    """Owner exchange for per-expert Muon; returns this rank's row shard of the update.

    ``local`` is ``[local_experts, local_rows, columns]``. Each rank assembles
    the full rows of the experts it owns, orthogonalizes them, and sends every
    rank its rows of the result back.
    """
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    num_experts, _, columns = (int(size) for size in local.shape)
    owners = _expert_owners(num_experts, world_size)
    owned = owners[rank]
    bands = [_chunk_range(rows, world_size, source) for source in range(world_size)]
    band_start, band_end = bands[rank]

    send_buffers = {
        owner: local[experts].reshape(-1).contiguous()
        for owner, experts in enumerate(owners)
        if owner != rank and experts and band_end > band_start
    }
    receive_buffers = {
        source: local.new_empty(len(owned) * (end - start) * columns)
        for source, (start, end) in enumerate(bands)
        if source != rank and owned and end > start
    }

    if sum(end - start for start, end in bands) != rows:
        raise RuntimeError("Muon per-expert row bands do not cover the matrix")
    if sum(len(experts) for experts in owners) != num_experts:
        raise RuntimeError("Muon per-expert owners do not cover every expert")
    assembled = local.new_empty((len(owned), rows, columns))
    if owned:
        assembled[:, band_start:band_end] = local[owned]
    _run_all_to_all(receive_buffers, send_buffers, local, process_group)
    for source, packed in receive_buffers.items():
        start, end = bands[source]
        assembled[:, start:end] = packed.view(len(owned), end - start, columns)

    # Updates are exchanged in the Newton-Schulz compute dtype: upcasting would
    # only double the traffic. Ranks without owned experts use the same dtype.
    update_dtype = newton_schulz_dtype(rows, columns, steps)
    if owned:
        owned_update = zeropower_via_newtonschulz5_per_expert(
            assembled, num_experts=len(owned), steps=steps, coefficients=coefficients
        )
    else:
        owned_update = assembled.to(update_dtype)

    local_update = torch.empty_like(local, dtype=update_dtype)
    if owned:
        local_update[owned] = owned_update[:, band_start:band_end]
    return_send_buffers = {
        peer: owned_update[:, start:end].reshape(-1).contiguous()
        for peer, (start, end) in enumerate(bands)
        if peer != rank and owned and end > start
    }
    return_receive_buffers = {
        owner: local_update.new_empty(
            len(experts) * (band_end - band_start) * columns
        )
        for owner, experts in enumerate(owners)
        if owner != rank and experts and band_end > band_start
    }
    _run_all_to_all(
        return_receive_buffers,
        return_send_buffers,
        local_update,
        process_group,
    )
    for owner, packed in return_receive_buffers.items():
        experts = owners[owner]
        local_update[experts] = packed.view(
            len(experts), band_end - band_start, columns
        )
    return local_update


def _per_expert_dtensor_update(parameter, ns_input, group):
    """Return the per-expert update of a row-sharded expert weight via the owner exchange.

    Returns None (use the all-gather fallback) unless exactly one mesh
    dimension shards the rows (dimension 1) across two or more ranks.
    """
    if ns_input.ndim != 3 or ns_input.placements != parameter.placements:
        return None
    layout = _moving_shard_layout(ns_input, "per_expert")
    if layout is None:
        return None
    _, shard_dim, process_group = layout
    if shard_dim != 1:
        return None
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return None

    rows, columns = (int(size) for size in ns_input.shape[1:])
    band_start, band_end = _chunk_range(
        rows, world_size, dist.get_rank(process_group)
    )
    local = ns_input.to_local()
    # The expert dimension is EP-local; validate each local matrix shape.
    if local.ndim != 3 or tuple(local.shape[1:]) != (band_end - band_start, columns):
        return None

    local_update = _per_expert_owner_all_to_all(
        local, rows, process_group, group["ns_steps"], group.get("ns_coefficients")
    )
    return _from_local_like(local_update, parameter)
