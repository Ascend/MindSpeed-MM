"""Preloader stream handoff tests; the NPU regression is explicitly opt-in."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mindspeed_mm.fsdp.data.dataloader import dataloader as loader


def test_next_protects_nested_device_storage_before_prefetch(monkeypatch):
    events = []
    consumer = Mock()
    consumer.wait_stream.side_effect = lambda stream: events.append("wait")

    def tensor(device):
        value = Mock(spec=torch.Tensor)
        value.device = SimpleNamespace(type=device)
        value.record_stream.side_effect = lambda stream: events.append("record")
        return value

    pixels, ids, video, cpu = [tensor(d) for d in ("npu", "npu", "cuda", "cpu")]
    batch = {
        "pixel_values": pixels,
        "nested": {"ids": [ids], "videos": (video, cpu)},
        "metadata": [None, 3, "sample"],
    }
    # Construct only the handoff state: this test must not initialize a device.
    preloader = loader.Preloader.__new__(loader.Preloader)
    preloader._h2d_thread = None
    preloader._fetch_error = preloader._h2d_error = None
    preloader.h2dstream = Mock()
    preloader.next_batch = batch
    preloader.state_dict_fn = lambda: events.append("state") or {"consumed": 1}
    preloader._start_cpu_prefetch = lambda: events.append("prefetch")
    monkeypatch.setattr(loader, "get_current_stream", lambda: consumer)

    assert preloader.next() is batch
    assert events == ["wait", "record", "record", "record", "state", "prefetch"]
    for value in (pixels, ids, video):
        value.record_stream.assert_called_once_with(consumer)
    cpu.record_stream.assert_not_called()
    consumer.wait_stream.assert_called_once_with(preloader.h2dstream)
    assert preloader.next_batch is None
    assert preloader.last_consumed_state == {"consumed": 1}


def test_record_batch_stream_leaves_cpu_batch_unchanged():
    batch = {"labels": torch.tensor([1, 2]), "metadata": None}
    # Tensor.record_stream would reject this object if called for CPU tensors.
    loader._record_batch_stream(batch, object())
    assert batch["labels"].tolist() == [1, 2]


@pytest.mark.skipif(
    os.environ.get("MM_RUN_PRELOADER_NPU_TEST") != "1",
    reason="set MM_RUN_PRELOADER_NPU_TEST=1 to run the single-NPU lifetime regression",
)
def test_preloaded_input_survives_pending_backward(monkeypatch):
    """A second H2D must not overwrite input still read by queued backward work."""
    npu = torch.npu
    npu.set_device(0)
    # Training installs this accelerator compatibility alias during startup.
    monkeypatch.setattr(torch.accelerator, "current_device", npu.current_device, raising=False)
    shape = (1024, 1024)
    batches = iter([
        {"pixel_values": torch.ones(shape).pin_memory()},
        {"pixel_values": torch.full(shape, 1e20).pin_memory()},
    ])
    preloader = loader.Preloader(batches, param_dtype=torch.float32)
    batch = preloader.next()
    weight = torch.ones(shape, device="npu", requires_grad=True)
    loss = (weight * batch["pixel_values"]).sum()
    npu.synchronize()
    # Widen the race window without a full model or distributed training.
    npu._sleep(100000000)
    batch.clear()
    loss.backward()
    preloader.trigger_h2d()
    # Join submission only; do not synchronize compute before the next allocation.
    preloader._h2d_thread.join()
    npu.synchronize()
    try:
        torch.testing.assert_close(weight.grad.cpu(), torch.ones(shape), rtol=0, atol=0)
        assert weight.grad.norm().item() == 1024.0
        next_batch = preloader.next()
        assert next_batch["pixel_values"][0, 0].item() == torch.tensor(1e20).item()
    finally:
        if preloader._cpu_thread is not None:
            preloader._cpu_thread.join()
        npu.synchronize()
