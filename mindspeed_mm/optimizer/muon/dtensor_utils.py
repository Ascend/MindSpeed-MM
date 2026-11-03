"""DTensor layout helpers and variable-size all-to-all packing."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard


def _chunk_range(size, chunks, index):
    """Return the contiguous range used by DTensor's chunk sharding."""
    chunk_size = (size + chunks - 1) // chunks
    start = min(index * chunk_size, size)
    return start, min(start + chunk_size, size)


def _single_shard_layout(tensor):
    """Return ``(mesh_dim, shard_dim, group)`` for a simple DTensor shard."""
    if any(
        isinstance(placement, Shard) and type(placement) is not Shard
        for placement in tensor.placements
    ):
        # Owner fragments require one contiguous chunk per rank.
        return None
    shards = [
        (mesh_dim, placement.dim)
        for mesh_dim, placement in enumerate(tensor.placements)
        if isinstance(placement, Shard)
    ]
    if len(shards) != 1 or any(
        not isinstance(placement, (Replicate, Shard))
        for placement in tensor.placements
    ):
        return None
    mesh_dim, shard_dim = shards[0]
    return mesh_dim, shard_dim, tensor.device_mesh.get_group(mesh_dim)


def _row_shards_preserve_heads(rows, num_heads, world_size):
    """Whether every row-shard boundary falls between complete heads."""
    if (not isinstance(num_heads, int) or isinstance(num_heads, bool)
            or num_heads <= 0 or rows <= 0 or rows % num_heads):
        return False
    head_rows = rows // num_heads
    return all(
        start % head_rows == 0 and end % head_rows == 0
        for rank in range(world_size)
        for start, end in [_chunk_range(rows, world_size, rank)]
    )


def _from_local_like(local, reference, placements=None):
    """Wrap a local shard as a DTensor carrying a reference parameter's layout."""
    return DTensor.from_local(
        local,
        reference.device_mesh,
        reference.placements if placements is None else placements,
        run_check=False,
        shape=reference.shape,
        stride=reference.stride(),
    )


def _copy_replicated_to_dtensor_(parameter, full_tensor, placements):
    """Reshard a modified replicated tensor and copy it into a DTensor parameter."""
    replicated = DTensor.from_local(
        full_tensor.reshape(parameter.shape),
        parameter.device_mesh,
        placements,
        run_check=False,
    )
    parameter.copy_(replicated.redistribute(parameter.device_mesh, parameter.placements))


def _flat_send(peer_pieces, reference, world_size):
    """Concatenate each peer's tensors into one send buffer, in group-rank order.

    Returns the buffer and the per-peer split sizes for ``all_to_all_single``.
    Used by both owner.py and distributed.py.
    """
    splits = [sum(int(piece.numel()) for piece in peer_pieces.get(peer, ()))
              for peer in range(world_size)]
    pieces = [piece for peer in range(world_size) for piece in peer_pieces.get(peer, ())]
    return (torch.cat(pieces).contiguous() if pieces else reference.new_empty(0), splits)


def _peer_views(flat, splits):
    """Return nonempty peer views into a packed receive buffer."""
    result, offset = {}, 0
    for peer, count in enumerate(splits):
        if count:
            result[peer] = flat[offset:offset + count]
            offset += count
    return result


def _run_all_to_all(receive_buffers, send_buffers, reference, process_group):
    """Exchange flattened peer buffers with variable split sizes.

    Missing peers use zero-length splits, but every rank enters the collective.
    """
    world_size = dist.get_world_size(process_group)
    input_buffer, input_split_sizes = _flat_send(
        {peer: [buffer.reshape(-1)] for peer, buffer in send_buffers.items()}, reference, world_size)
    output_split_sizes = [
        int(receive_buffers[peer].numel()) if peer in receive_buffers else 0
        for peer in range(world_size)
    ]
    output_buffer = reference.new_empty(sum(output_split_sizes))
    work = dist.all_to_all_single(
        output_buffer,
        input_buffer,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=process_group,
        async_op=True,
    )
    work.wait()
    for peer, view in _peer_views(output_buffer, output_split_sizes).items():
        receive_buffers[peer].copy_(view)
