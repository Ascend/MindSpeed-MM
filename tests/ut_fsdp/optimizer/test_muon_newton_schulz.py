"""Muon iterations agree across local batches and column shards."""
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mindspeed_mm.optimizer.muon import kernels, newton_schulz as ns
from mindspeed_mm.optimizer.muon.dtensor_utils import _chunk_range, _row_shards_preserve_heads


@pytest.mark.parametrize("steps", [0, 2, 4, 5, 6, 8, 10, 11])
def test_gram_newton_schulz_supports_only_five_and_ten_steps(steps):
    supported = steps in (5, 10)
    assert ns.uses_gram_newton_schulz(4, 32, steps) == supported
    assert ns.uses_gram_newton_schulz(32, 4, steps) == supported
    assert ns.uses_column_sharded_gram_newton_schulz(4, 32, steps) == supported
    assert not ns.uses_column_sharded_gram_newton_schulz(32, 4, steps)
    assert not ns.uses_gram_newton_schulz(16, 16, steps)
    if not supported:
        with pytest.raises(ValueError, match="only 5 or 10"):
            ns._gram_iterations(torch.ones(2, 4, 8), steps, ns.resolve_ns_coefficients(None),
                                *kernels.symmetric_kernels("cpu"))


@pytest.mark.parametrize("steps", [1, 5, 10])
def test_iterations_match_direct_polynomial(steps):
    x = torch.randn(2, 4, 19, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    x /= torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True)
    schedule = ns.resolve_ns_coefficients(None)
    expected = x.clone()
    identity = torch.eye(4, dtype=x.dtype)
    # Apply each polynomial directly to the matrix as an independent reference.
    for i in range(steps):
        a, b, c = schedule[min(i, 4)]
        gram = expected @ expected.transpose(-2, -1)
        expected = (a * identity + b * gram + c * (gram @ gram)) @ expected
    if steps == 1:
        actual = ns._newton_schulz(x.float(), steps)
        torch.testing.assert_close(actual.double(), expected, rtol=.02, atol=.01)
    else:
        actual = ns._gram_iterations(x, steps, schedule, *kernels.symmetric_kernels("cpu"))
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("steps", [5, 10])
@pytest.mark.parametrize("shape", [(4, 19), (19, 4)])
def test_single_head_expert_and_owner_batches_share_the_iteration(steps, shape):
    generator = torch.Generator().manual_seed(4)
    inputs = torch.randn(3, *shape, generator=generator)
    expected = torch.stack([ns.zeropower_via_newtonschulz5(x, steps) for x in inputs])
    for actual in (ns._newton_schulz(inputs, steps),
                   ns.zeropower_via_newtonschulz5_per_head(inputs, steps=steps),
                   ns.zeropower_via_newtonschulz5_per_expert(inputs, steps=steps)):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _column_shard_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=3, timeout=timedelta(seconds=90))
    try:
        for steps in (5, 10):
            for shape in ((2, 4, 19), (2, 1, 2)):
                full = torch.randn(shape, generator=torch.Generator().manual_seed(71))
                a, b = _chunk_range(shape[-1], 3, rank)
                actual = ns._newton_schulz(full[..., a:b], steps, process_group=dist.group.WORLD)
                expected = ns._newton_schulz(full, steps)[..., a:b]
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, rtol=.04, atol=.02,
                                           msg=f"steps={steps}, shape={shape}, rank={rank}")
    finally:
        dist.destroy_process_group()


def test_column_shards_match_complete_gns_for_five_and_ten_steps(tmp_path):
    mp.spawn(_column_shard_worker, args=((tmp_path / "columns").as_uri(),), nprocs=3)


def test_owner_cost_charges_per_call_overhead():
    overhead = ns.newton_schulz_call_overhead("cpu")
    flops = ns.newton_schulz_flops(1024, 4096, 10, "cpu", 1)
    assert ns.newton_schulz_cost(1024, 4096, 10, "cpu", 1) == pytest.approx(flops + overhead)
    batched = ns.newton_schulz_cost(1024, 4096, 10, "cpu", 8)
    assert batched == pytest.approx(
        ns.newton_schulz_flops(1024, 4096, 10, "cpu", 8) + overhead / 8
    )


@pytest.mark.parametrize("rows,heads", [(4, 0), (4, 8), (6, 4), (0, 2)])
def test_invalid_head_layout_is_rejected_before_shard_access(rows, heads):
    assert not _row_shards_preserve_heads(rows, heads, 3)
