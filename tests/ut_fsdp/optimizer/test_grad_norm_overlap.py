"""Unit tests for mindspeed_mm.fsdp.optimizer.grad_norm_overlap.

Most tests run single-process on CPU tensors, feeding the manager plain
(param, grad, event) jobs as produced by fully_shard._collect_grad_norm_jobs;
the device side-stream path has NPU-gated tests (skipped where NPU is
unavailable). The real DTensor/offload wiring is covered by e2e off/on pairs.
"""
from types import SimpleNamespace

import pytest
import torch

from mindspeed_mm.fsdp.optimizer import grad_norm_overlap as gno
from mindspeed_mm.fsdp.optimizer.clip_grad_norm import _local_pth_sum

try:
    import torch_npu  # noqa: F401
    _HAS_NPU = torch.npu.is_available()
except ImportError:
    _HAS_NPU = False

npu_only = pytest.mark.skipif(not _HAS_NPU, reason="NPU unavailable")


@pytest.fixture()
def fresh_manager():
    mgr = gno._GradNormOverlapManager()
    mgr.enabled = True
    yield mgr
    if mgr._queue is not None:
        mgr._queue.join()


def _cpu_job(param, event=None):
    return (param, param.grad, event)


def _dev_job(param):
    return (param, param.grad)


def _params_with_grads(values_per_param):
    params = []
    for values in values_per_param:
        p = torch.nn.Parameter(torch.zeros(len(values)))
        p.grad = torch.tensor(values, dtype=torch.float32)
        params.append(p)
    return params


def test_disabled_record_and_consume_are_noop():
    mgr = gno._GradNormOverlapManager()
    (p,) = _params_with_grads([[3.0, 4.0]])
    mgr.record([_cpu_job(p)], [])
    assert mgr._worker is None
    assert mgr.consume([p], 2.0) is None


def test_consume_matches_serial_path_bitwise(fresh_manager):
    params = _params_with_grads([[3.0, 4.0], [1.0, 2.0, 2.0], [0.5, -0.5]])
    # record in shuffled group order; consume must follow `params` order
    fresh_manager.record([_cpu_job(params[2])], [])
    fresh_manager.record([_cpu_job(params[0])], [])
    fresh_manager.record([_cpu_job(params[1])], [])
    val = fresh_manager.consume(params, 2.0)
    ref = _local_pth_sum(params, 2.0)
    assert val is not None
    assert torch.equal(val, ref), f"{val.item()} != {ref.item()}"


def test_last_invocation_overwrites(fresh_manager):
    (p,) = _params_with_grads([[3.0, 4.0]])
    fresh_manager.record([_cpu_job(p)], [])
    p.grad.mul_(2.0)  # second accumulation lands in the same buffer
    fresh_manager.record([_cpu_job(p)], [])
    val = fresh_manager.consume([p], 2.0)
    assert val is not None
    assert val.item() == pytest.approx(100.0)  # (6^2 + 8^2)


def test_d2h_event_is_synchronized_before_read(fresh_manager):
    (p,) = _params_with_grads([[1.0, 1.0]])
    calls = []
    ev = SimpleNamespace(synchronize=lambda: calls.append(1))
    fresh_manager.record([_cpu_job(p, event=ev)], [])
    val = fresh_manager.consume([p], 2.0)
    assert val is not None and calls, "event.synchronize() must be called"
    assert val.item() == pytest.approx(2.0)


def test_uncovered_param_falls_back_with_warning(fresh_manager, caplog):
    params = _params_with_grads([[3.0, 4.0], [1.0]])
    fresh_manager.record([_cpu_job(params[0])], [])
    with caplog.at_level("WARNING"):
        assert fresh_manager.consume(params, 2.0) is None
    assert "not covered" in caplog.text


def test_norm_type_mismatch_disables(fresh_manager):
    (p,) = _params_with_grads([[3.0, 4.0]])
    fresh_manager.record([_cpu_job(p)], [])
    assert fresh_manager.consume([p], 3.0) is None
    assert fresh_manager.enabled is False  # hard-disabled, not per-step fallback
    # and stays off: subsequent records are no-ops
    fresh_manager.record([_cpu_job(p)], [])
    assert fresh_manager._worker is not None  # worker exists from the first record
    assert p.grad is not None


def test_worker_exception_surfaces_at_consume(fresh_manager, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("foreach_norm exploded")

    monkeypatch.setattr(torch, "_foreach_norm", boom)
    (p,) = _params_with_grads([[3.0, 4.0]])
    fresh_manager.record([_cpu_job(p)], [])
    with pytest.raises(RuntimeError, match="worker error"):
        fresh_manager.consume([p], 2.0)


def test_bf16_grads_normed_in_fp32(fresh_manager):
    p = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    p.grad = torch.tensor([3.0, 4.0, 1.0, 2.0], dtype=torch.bfloat16)
    fresh_manager.record([_cpu_job(p)], [])
    val = fresh_manager.consume([p], 2.0)
    assert val is not None and val.dtype == torch.float32
    # same call sequence as the serial path (foreach_norm(dtype=fp32) -> pow -> stack -> sum)
    assert torch.equal(val, _local_pth_sum([p], 2.0))


def test_configure_requires_planted_hook():
    mgr = gno._GradNormOverlapManager()
    mgr.configure(True)
    assert mgr.enabled is False  # no patched post_backward -> stays off
    mgr.mark_hook_planted()
    mgr.configure(True)
    assert mgr.enabled is True


def test_mixed_dtype_single_record_falls_back(fresh_manager, caplog):
    """One record carrying fp32+bf16 grads: consume must fall back loudly (never
    silently diverge or crash on a dtype-mismatched torch.stack)."""
    p_fp32 = torch.nn.Parameter(torch.zeros(3))
    p_fp32.grad = torch.tensor([1.5, -2.5, 3.5])
    p_bf16 = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    p_bf16.grad = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    fresh_manager.record([_cpu_job(p_fp32), _cpu_job(p_bf16)], [])
    with caplog.at_level("WARNING"):
        assert fresh_manager.consume([p_fp32, p_bf16], 2.0) is None
    assert "mixed grad dtypes" in caplog.text


def test_mixed_dtype_falls_back_and_serial_matches_baseline(fresh_manager, caplog):
    """Mixed fp32/bf16 grads across records: overlap falls back; the serial path
    must reproduce the pre-overlap baseline bitwise (materialize every grad to
    fp32 first, single fp32 group per device)."""
    p_fp32 = torch.nn.Parameter(torch.zeros(3))
    p_fp32.grad = torch.tensor([1.5, -2.5, 3.5])
    p_bf16 = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    p_bf16.grad = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    p_fp32_b = torch.nn.Parameter(torch.zeros(2))
    p_fp32_b.grad = torch.tensor([0.25, -0.75])
    params = [p_fp32, p_bf16, p_fp32_b]
    for p in params:
        fresh_manager.record([_cpu_job(p)], [])
    with caplog.at_level("WARNING"):
        assert fresh_manager.consume(params, 2.0) is None
    assert "mixed grad dtypes" in caplog.text
    ref_grads = [p.grad.to(torch.float32) for p in params]
    ref = torch.sum(torch.stack(torch._foreach_pow_(torch._foreach_norm(ref_grads, 2.0), 2.0)))
    assert torch.equal(_local_pth_sum(params, 2.0), ref)


def test_clear_drops_leftovers(fresh_manager):
    (p,) = _params_with_grads([[3.0, 4.0]])
    fresh_manager.record([_cpu_job(p)], [])
    fresh_manager._queue.join()
    assert fresh_manager._norms
    fresh_manager.clear()
    assert not fresh_manager._norms
    # registry emptied -> param no longer covered -> serial fallback
    assert fresh_manager.consume([p], 2.0) is None


def test_clear_noop_when_disabled():
    mgr = gno._GradNormOverlapManager()
    mgr._norms[1] = ((torch.device("cpu"), torch.float32), torch.tensor(1.0))
    mgr.clear()
    assert mgr._norms  # untouched while disabled


def _collect_jobs(fsdp_params, device):
    from mindspeed_mm.fsdp.ops.fully_shard.fully_shard import _collect_grad_norm_jobs
    return _collect_grad_norm_jobs(fsdp_params, device)


def _fake_fp(param, offload=False, with_event=False):
    ns = SimpleNamespace(sharded_param=param, offload_to_cpu=offload)
    if with_event:
        ns.grad_offload_event = object()
    return ns


def test_collect_jobs_splits_by_placement():
    """Collector owns all FSDPParam attribute reads; placements split into
    cpu/dev jobs, grads=None are skipped, and a missing grad_offload_event
    attribute degrades to None instead of raising."""
    (p_off,) = _params_with_grads([[3.0, 4.0]])
    (p_dev,) = _params_with_grads([[1.0, 2.0]])
    p_nograd = torch.nn.Parameter(torch.zeros(2))
    cpu_jobs, dev_jobs = _collect_jobs(
        [_fake_fp(p_off, offload=True), _fake_fp(p_dev), _fake_fp(p_nograd)],
        torch.device("cpu"))
    assert len(cpu_jobs) == 1 and len(dev_jobs) == 1
    assert cpu_jobs[0][0] is p_off and cpu_jobs[0][1] is p_off.grad
    assert cpu_jobs[0][2] is None  # no grad_offload_event attr -> None, no raise
    assert dev_jobs[0][0] is p_dev and dev_jobs[0][1] is p_dev.grad


def test_collect_jobs_skips_unexpected_placement():
    """A grad living somewhere other than the group's device (and not
    offloaded) is left out, so consume falls back to the serial path."""
    (p,) = _params_with_grads([[3.0, 4.0]])
    cpu_jobs, dev_jobs = _collect_jobs([_fake_fp(p)], torch.device("meta"))
    assert cpu_jobs == [] and dev_jobs == []


def test_foreach_norm_dtype_fp32_matches_materialize_first_cpu():
    """Pin the bitwise identity the serial-path rewrite relies on:
    foreach_norm(bf16, dtype=fp32) == foreach_norm(fp32-materialized bf16)."""
    torch.manual_seed(0)
    grads = [torch.randn(33, 65, dtype=torch.bfloat16),
             torch.randn(5, 7, 129, dtype=torch.bfloat16),
             torch.randn(1024, dtype=torch.bfloat16)]
    direct = torch._foreach_norm(grads, 2.0, dtype=torch.float32)
    materialized = torch._foreach_norm([g.to(torch.float32) for g in grads], 2.0)
    assert len(direct) == len(materialized)
    for a, b in zip(direct, materialized):
        assert torch.equal(a, b)


def _npu_params_with_grads(values_per_param, dtype=torch.bfloat16):
    params = []
    for values in values_per_param:
        p = torch.nn.Parameter(torch.zeros(len(values), device="npu:0", dtype=dtype))
        p.grad = torch.tensor(values, dtype=dtype, device="npu:0")
        params.append(p)
    return params


def _recorded_npu_event():
    ev = torch.npu.Event()
    ev.record()  # grads were produced on the current stream
    return ev


@npu_only
def test_foreach_norm_dtype_fp32_matches_materialize_first_npu():
    torch.manual_seed(0)
    grads = [torch.randn(33, 65, dtype=torch.bfloat16, device="npu:0"),
             torch.randn(5, 7, 129, dtype=torch.bfloat16, device="npu:0"),
             torch.randn(1024, dtype=torch.bfloat16, device="npu:0")]
    direct = torch._foreach_norm(grads, 2.0, dtype=torch.float32)
    materialized = torch._foreach_norm([g.to(torch.float32) for g in grads], 2.0)
    assert len(direct) == len(materialized)
    for a, b in zip(direct, materialized):
        assert torch.equal(a, b)


@npu_only
def test_device_record_consume_bitwise(fresh_manager):
    torch.manual_seed(0)
    params = _npu_params_with_grads([[3.0, 4.0], [1.0, 2.0, 2.0], [0.5, -0.5]])
    fresh_manager.record([], [_dev_job(params[2])], _recorded_npu_event())
    fresh_manager.record([], [_dev_job(params[0]), _dev_job(params[1])],
                         _recorded_npu_event())
    assert fresh_manager._worker is None  # device path starts no CPU worker
    val = fresh_manager.consume(params, 2.0)
    assert val is not None and val.device.type == "npu"
    assert torch.equal(val.cpu(), _local_pth_sum(params, 2.0).cpu())


@npu_only
def test_device_and_cpu_mixed_bitwise(fresh_manager):
    """One offloaded (CPU) group + one NPU-resident group consumed together."""
    (p_cpu,) = _params_with_grads([[3.0, 4.0]])
    (p_npu,) = _npu_params_with_grads([[1.0, 2.0, 2.0]])
    params = [p_npu, p_cpu]
    fresh_manager.record([_cpu_job(p_cpu)], [])
    fresh_manager.record([], [_dev_job(p_npu)], _recorded_npu_event())
    val = fresh_manager.consume(params, 2.0)
    assert val is not None
    assert torch.equal(val.cpu(), _local_pth_sum(params, 2.0).cpu())


@npu_only
def test_device_record_without_event_falls_back(fresh_manager, caplog):
    (p,) = _npu_params_with_grads([[3.0, 4.0]])
    fresh_manager.record([], [_dev_job(p)], None)
    with caplog.at_level("WARNING"):
        assert fresh_manager.consume([p], 2.0) is None
    assert "post_reduce_event" in caplog.text


@npu_only
def test_device_mixed_dtype_single_record_falls_back(fresh_manager, caplog):
    """Device path, one record with fp32+bf16 grads: norms themselves must not
    crash (always computed in fp32) and consume must fall back loudly."""
    p_fp32 = torch.nn.Parameter(torch.zeros(3, device="npu:0"))
    p_fp32.grad = torch.tensor([1.5, -2.5, 3.5], device="npu:0")
    p_bf16 = torch.nn.Parameter(torch.zeros(2, device="npu:0", dtype=torch.bfloat16))
    p_bf16.grad = torch.tensor([3.0, 4.0], dtype=torch.bfloat16, device="npu:0")
    fresh_manager.record([], [_dev_job(p_fp32), _dev_job(p_bf16)],
                         _recorded_npu_event())
    with caplog.at_level("WARNING"):
        assert fresh_manager.consume([p_fp32, p_bf16], 2.0) is None
    assert "mixed grad dtypes" in caplog.text
