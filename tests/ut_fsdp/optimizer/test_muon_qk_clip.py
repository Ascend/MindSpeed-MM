"""QK-Clip scales per-head weights and agrees across CPU/Gloo ranks."""
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from mindspeed_mm.optimizer.muon.qk_clip import qk_clip_, qk_clip_mla_, _apply_qk_clip_groups, _prepare_qk_clip_groups


def _group(query, key, metrics, num_heads, process_groups=()):
    return dict(
        params=[query], use_muon=True, muon_mode="per_head", muon_num_heads=num_heads,
        qk_clip_params=(query, key), qk_clip_max_logits=metrics,
        qk_clip_process_groups=process_groups, qk_clip_threshold=5., qk_clip_alpha=.5,
    )


@pytest.mark.parametrize("mla", [False, True])
@pytest.mark.parametrize("alpha", [0., .25, .5, 1.])
@pytest.mark.parametrize("threshold,factors", [(0., [1., 1., 0., 0., 0.]),
                                                (2., [1., 1., 1., 1., .5])])
def test_clips_only_heads_above_threshold(mla, alpha, threshold, factors):
    query = torch.arange(1., 61.).reshape(5, 4, 3)
    key = torch.arange(1., 51.).reshape(5, 5, 2)
    original_query, original_key = query.clone(), key.clone()
    metrics = torch.tensor([-3., 0., 1., 2., 4.])
    gamma = torch.tensor(factors).view(-1, 1, 1)
    expected_query, expected_key = query.clone(), key.clone()
    if mla:
        qk_clip_mla_(query, key, metrics, threshold, alpha, 2, 2, 2)
        expected_query[:, :2] *= gamma**alpha
        expected_query[:, 2:] *= gamma
        expected_key[:, :2] *= gamma**(1 - alpha)
        torch.testing.assert_close(key[:, 2:], original_key[:, 2:], rtol=0, atol=0)
    else:
        qk_clip_(query, key, metrics, threshold, alpha)
        expected_query *= gamma**alpha
        expected_key *= gamma**(1 - alpha)
    torch.testing.assert_close(query, expected_query)
    torch.testing.assert_close(key, expected_key)
    within_bound = metrics <= threshold
    torch.testing.assert_close(query[within_bound], original_query[within_bound], rtol=0, atol=0)
    torch.testing.assert_close(key[within_bound], original_key[within_bound], rtol=0, atol=0)


def _distributed_checks(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=4, timeout=timedelta(seconds=30))
    try:
        mesh = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "cp"))
        process_groups = tuple(mesh.get_all_groups())
        for layout in ("plain", "sharded", "mixed"):
            groups, expected = [], []
            for layer, heads in enumerate((4, 6, 8)):
                query = torch.ones(heads * 2, 3)
                key = query.clone()
                if layout == "sharded" or (layout == "mixed" and layer != 0):
                    query = distribute_tensor(query, mesh, [Shard(0), Replicate()])
                    key = distribute_tensor(key, mesh, [Shard(0), Replicate()])
                query, key = torch.nn.Parameter(query), torch.nn.Parameter(key)
                first = (rank % 2) * (heads // 2)
                metrics = torch.full((heads,), -torch.inf)
                metrics[first:first + heads // 2] = torch.arange(first, first + heads // 2) + 10. + 10 * (rank // 2) + layer
                groups.append(_group(query, key, metrics, heads, process_groups))
                expected.append((5. / (torch.arange(heads) + 20. + layer)).sqrt())
            _apply_qk_clip_groups(groups)
            for group, scales in zip(groups, expected):
                for parameter in group["qk_clip_params"]:
                    actual = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
                    torch.testing.assert_close(actual.view(len(scales), 2, 3), scales[:, None, None].expand(-1, 2, 3))

        # Every rank finishes the reductions and raises before any Q/K scaling.
        for failure in ("missing", "all_missing", "shape", "type", "nan"):
            query = torch.nn.Parameter(torch.ones(8, 3))
            key = torch.nn.Parameter(torch.ones(8, 3))
            metrics = torch.full((4,), 20.)
            if failure == "all_missing" or (rank == 0 and failure == "missing"):
                metrics = None
            elif rank == 0 and failure == "shape":
                metrics = torch.ones(3)
            elif rank == 0 and failure == "type":
                metrics = [20.] * 4
            elif rank == 0 and failure == "nan":
                metrics[0] = torch.nan
            group = _group(query, key, metrics, 4, process_groups)
            group["qk_clip_metric_provider"] = lambda: metrics
            error = RuntimeError if "missing" in failure else ValueError
            with pytest.raises(error):
                _apply_qk_clip_groups([group])
            torch.testing.assert_close(query, torch.ones_like(query))
            torch.testing.assert_close(key, torch.ones_like(key))
            metrics = torch.full((4,), 20.)
            _apply_qk_clip_groups([group])
            torch.testing.assert_close(query, torch.full_like(query, .5))
            torch.testing.assert_close(key, torch.full_like(key, .5))
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_distributed_qk_clip_agreement_and_failures(tmp_path):
    mp.spawn(_distributed_checks, args=((tmp_path / "qk").as_uri(),), nprocs=4)



@pytest.mark.parametrize("name,value", [("qk_clip_alpha", 2.), ("qk_clip_threshold", -1.)])
def test_invalid_controls_keep_consumed_metrics_for_retry(name, value):
    query = torch.nn.Parameter(torch.ones(4, 3))
    key = torch.nn.Parameter(torch.ones_like(query))
    metrics = [torch.full((2,), 20.)]
    calls = []

    def provider():
        calls.append(1)
        return metrics.pop() if metrics else None

    group = _group(query, key, None, 2)
    group["qk_clip_metric_provider"] = provider
    original = group[name]
    group[name] = value
    with pytest.raises(ValueError):
        _prepare_qk_clip_groups([group])
    torch.testing.assert_close(query, torch.ones_like(query))
    torch.testing.assert_close(key, torch.ones_like(key))
    group[name] = original
    _apply_qk_clip_groups([group])
    assert len(calls) == 1
    torch.testing.assert_close(query, torch.full_like(query, .5))
    torch.testing.assert_close(key, torch.full_like(key, .5))
