"""QK-Clip scales only out-of-bound heads and agrees across CPU/Gloo ranks."""
import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from mindspeed_mm.optimizer.muon import Muon, qk_clip_, qk_clip_mla_
from mindspeed_mm.optimizer.muon.qk_clip import _apply_qk_clip_groups, _QK_CLIP_RUNTIME_KEYS


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


def test_runtime_groups_are_not_checkpointed():
    query = torch.nn.Parameter(torch.ones(4, 3))
    key = torch.nn.Parameter(torch.ones(4, 3))
    group = _group(query, key, torch.ones(2), 2)
    group["qk_clip_metric_provider"] = lambda: torch.ones(2)
    optimizer = Muon([group])
    saved = copy.deepcopy(optimizer.state_dict())
    assert not set(_QK_CLIP_RUNTIME_KEYS).intersection(saved["param_groups"][0])
    optimizer.load_state_dict(saved)
    assert optimizer.param_groups[0]["qk_clip_metric_provider"] is group["qk_clip_metric_provider"]
    assert optimizer.param_groups[0]["qk_clip_params"][0] is query


class _TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 4, bias=False)
        self.k_proj = torch.nn.Linear(3, 4, bias=False)
        self.norm = torch.nn.LayerNorm(3)


def _qk_clip_muon(model, provider):
    """Per-head Muon for Q and K with QK-Clip on Q, and AdamW for the norm."""
    query, key = model.q_proj.weight, model.k_proj.weight
    per_head = dict(use_muon=True, muon_mode="per_head", muon_num_heads=2)
    return Muon([
        dict(params=[query], **per_head, qk_clip_params=(query, key),
             qk_clip_metric_provider=provider, qk_clip_threshold=1e-3, qk_clip_alpha=.5),
        dict(params=[key], **per_head),
        dict(params=list(model.norm.parameters()), use_muon=False),
    ], lr=.01)


def test_distributed_checkpoint_round_trip_into_fresh_optimizer():
    """DCP creates state with a zero-lr step(closure=None) before save and load."""
    from torch.distributed.checkpoint.state_dict import (
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    torch.manual_seed(0)
    model = _TinyAttention()
    fresh_metrics = {"value": None}

    def provider():
        value, fresh_metrics["value"] = fresh_metrics["value"], None
        return value

    optimizer = _qk_clip_muon(model, provider)
    before = copy.deepcopy(model.state_dict())
    # The initialization step sees no fresh QK-Clip metrics and must not change weights.
    get_optimizer_state_dict(model, optimizer)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    fresh_metrics["value"] = torch.tensor([10., 0.])
    optimizer.step()
    saved = copy.deepcopy(get_optimizer_state_dict(model, optimizer))

    restored_model = copy.deepcopy(model)
    restored = _qk_clip_muon(restored_model, provider)
    set_optimizer_state_dict(restored_model, restored, saved)
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    for source, target in zip(model.parameters(), restored_model.parameters()):
        assert optimizer.state[source].keys() == restored.state[target].keys()
        for key, value in optimizer.state[source].items():
            torch.testing.assert_close(restored.state[target][key], value)
    assert restored.param_groups[0]["qk_clip_metric_provider"] is provider


def test_adamw_initializes_moments_with_unrelated_state():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = Muon([dict(params=[parameter], use_muon=False)], lr=.01, weight_decay=.1, adamw_eps=1e-3)
    adamw = torch.optim.AdamW([reference], lr=.01, weight_decay=.1, betas=(.95, .95), eps=1e-3)
    optimizer.state[parameter]["momentum_buffer"] = torch.full_like(parameter, 7.)
    for _ in range(2):
        # Small gradients make the epsilon placement affect the update.
        parameter.grad = torch.full_like(parameter, 2e-6)
        reference.grad = parameter.grad.clone()
        optimizer.step()
        adamw.step()
        torch.testing.assert_close(parameter, reference)


@pytest.mark.parametrize("name,value", [("qk_clip_alpha", 2.), ("qk_clip_threshold", -1.)])
def test_invalid_controls_fail_before_any_optimizer_update(name, value):
    model = _TinyAttention()
    metrics = [torch.tensor([10., 12.])]
    calls = []

    def provider():
        calls.append(1)
        return metrics.pop() if metrics else None

    optimizer = _qk_clip_muon(model, provider)
    original = optimizer.param_groups[0][name]
    optimizer.param_groups[0][name] = value
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        optimizer.step()
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert not optimizer.state
    optimizer.param_groups[0][name] = original
    optimizer.step()
    assert len(calls) == 1
    assert optimizer.state
