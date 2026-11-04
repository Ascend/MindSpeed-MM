"""CPU/Gloo checks for uneven exchanges and owner-computed matrix updates."""
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mindspeed_mm.optimizer.muon.dtensor_utils import _chunk_range, _run_all_to_all
from mindspeed_mm.optimizer.muon.newton_schulz import zeropower_via_newtonschulz5
from mindspeed_mm.optimizer.muon.owner import pipelined_owner_updates


def _exchange_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3,
                            timeout=timedelta(seconds=30))
    try:
        # Rank 2 participates without sending or receiving any elements.
        if rank == 0:
            send = {1: torch.tensor([10., 11.])}
            receive = {1: torch.empty(3)}
            expected = torch.tensor([20., 21., 22.])
        elif rank == 1:
            send = {0: torch.tensor([20., 21., 22.])}
            receive = {0: torch.empty(2)}
            expected = torch.tensor([10., 11.])
        else:
            send, receive = {}, {}
        _run_all_to_all(receive, send, torch.empty(0), dist.group.WORLD)
        if rank != 2:
            torch.testing.assert_close(receive[1 - rank], expected)
    finally:
        dist.destroy_process_group()


def test_all_to_all_exchanges_uneven_and_empty_splits(tmp_path):
    mp.spawn(_exchange_worker, args=((tmp_path / "exchange").as_uri(),), nprocs=3)


# Shapes, NS steps, wave limit, and number of exchanges allowed in flight.
_OWNER_CASES = [
    ([(1, 7), (2, 5), (5, 7)], 2, 1 << 20, 2),
    ([(5, 7), (8, 4), (3, 9)], 2, 64, 1),
    ([(5, 7)] * 10, 2, 300, 2),
    # Standard Newton-Schulz at 5 steps; the Gram cases follow.
    ([(16, 16), (12, 16)], 5, 1 << 20, 2),
    ([(4, 32)] * 6 + [(32, 4)] * 4, 5, 1 << 20, 2),
    ([(4, 32)] * 6 + [(32, 4)] * 4, 10, 1 << 20, 2),
]


def _owner_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3,
                            timeout=timedelta(seconds=60))
    try:
        for shapes, steps, wave_bytes, inflight in _OWNER_CASES:
            generator = torch.Generator().manual_seed(13)
            matrices = [torch.randn(shape, generator=generator) for shape in shapes]
            # Batched normalization must preserve zero, tiny, and ordinary inputs independently.
            matrices[0].zero_()
            matrices[1].mul_(1e-8)
            bands = [_chunk_range(rows, 3, rank) for rows, _ in shapes]
            shards = [matrix[start:end].clone() for matrix, (start, end) in zip(matrices, bands)]
            actual = pipelined_owner_updates(shards, shapes, dist.group.WORLD, steps,
                                             wave_bytes=wave_bytes, max_inflight=inflight)
            assert len(actual) == len(matrices)
            for update, shard, matrix, (start, end) in zip(actual, shards, matrices, bands):
                expected = zeropower_via_newtonschulz5(matrix, steps)[start:end]
                assert update.dtype == expected.dtype  # the Newton-Schulz compute dtype
                torch.testing.assert_close(update.float(), expected.float(), rtol=.02, atol=.02)
                torch.testing.assert_close(shard, matrix[start:end], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_owner_updates_match_complete_matrices(tmp_path):
    mp.spawn(_owner_worker, args=((tmp_path / "owners").as_uri(),), nprocs=3)


def test_owner_updates_reject_mixed_gram_and_standard_shapes():
    # Checked before any collective, so no process group is needed.
    with pytest.raises(ValueError, match="same Newton-Schulz variant"):
        pipelined_owner_updates([torch.zeros(4, 32), torch.zeros(16, 16)], [(4, 32), (16, 16)],
                                None, 5, wave_bytes=1 << 20)


@pytest.mark.parametrize("shard_dim", [0, 1])
@pytest.mark.parametrize("num_heads", [1, 4])
def test_head_fragments_cover_shards_and_heads_once(shard_dim, num_heads):
    from mindspeed_mm.optimizer.muon.distributed import _head_fragments

    world_size, head_rows, columns = 3, 4, 6
    rows = num_heads * head_rows
    for source in range(world_size):
        start, end = _chunk_range(rows if shard_dim == 0 else columns, world_size, source)
        local = torch.zeros(end - start, columns) if shard_dim == 0 else torch.zeros(rows, end - start)
        for owner in range(world_size):
            for fragment in _head_fragments(source, owner, world_size, rows, columns, num_heads, shard_dim):
                fragment.source_view(local).add_(1)
        torch.testing.assert_close(local, torch.ones_like(local))

    for owner in range(world_size):
        owned = torch.zeros(len(range(owner, num_heads, world_size)), head_rows, columns)
        for source in range(world_size):
            for fragment in _head_fragments(source, owner, world_size, rows, columns, num_heads, shard_dim):
                fragment.head_view(owned, owner, world_size).add_(1)
        torch.testing.assert_close(owned, torch.ones_like(owned))
