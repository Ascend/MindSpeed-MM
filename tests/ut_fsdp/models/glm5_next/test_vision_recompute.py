from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

pytest.importorskip("transformers", minversion="5.16.1", reason="GLM-5.3-Flash example dependency")
from mindspeed_mm.fsdp.models.glm5_next import modeling_glm5_next as glm
from mindspeed_mm.fsdp.ops.swiglu import eager_swiglu


def tiny_vision_config():
    return glm.Glm5NextVisionConfig(
        depth=2, hidden_size=16, num_heads=2, intermediate_size=16, out_hidden_size=16,
        projection_intermediate_size=16, patch_size=1, temporal_patch_size=1, attn_implementation="eager",
    )


@pytest.mark.parametrize("source_device", ["cpu", "npu"])
def test_blocks_share_host_sequence_lengths(monkeypatch, source_device):
    monkeypatch.setattr(glm, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(glm, "swiglu", eager_swiglu)
    vision = glm.Glm5NextVisionModel(tiny_vision_config())
    lengths = torch.tensor([0, 4, 12], dtype=torch.int32)
    source = lengths
    if source_device == "npu":
        source = Mock(device=SimpleNamespace(type="npu"))
        source.cpu.return_value = lengths
    monkeypatch.setattr(glm, "get_vision_attention_seqlens", lambda *a, **kw: (source, 8))
    seen = []

    def observe(block, args, kwargs):
        seen.append(kwargs["cu_seqlens"])
        assert kwargs["cu_seqlens"] is lengths
        assert kwargs["max_seqlen"] == 8

    for block in vision.blocks:
        block.register_forward_pre_hook(observe, with_kwargs=True)
    vision(torch.ones(12, 3), grid_thw=torch.tensor([[1, 2, 2], [1, 2, 4]]))
    assert len(seen) == 2
    if source_device == "npu":
        source.cpu.assert_called_once_with()


@pytest.mark.parametrize("npu,flash", [(True, True), (False, True), (True, False)])
def test_sync_order_during_checkpoint_replay(monkeypatch, npu, flash):
    monkeypatch.setattr(glm, "IS_NPU_AVAILABLE", False)
    attn = glm.Glm5NextVisionAttention(tiny_vision_config())
    # Keep tensor arithmetic on CPU while observing the NPU synchronization boundary.
    attn.q_norm = nn.Identity()
    attn.k_norm = nn.Identity()
    events = []
    stream = Mock()
    stream.synchronize.side_effect = lambda: events.append("sync")
    current_stream = Mock(return_value=stream)
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "current_stream", current_stream)
    else:
        monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=current_stream), raising=False)
    monkeypatch.setattr(glm, "IS_NPU_AVAILABLE", npu)
    monkeypatch.setattr(glm, "is_flash_attention_requested", lambda config: flash)
    lengths = torch.tensor([0, 2, 5], dtype=torch.int32)
    positions = (torch.ones(5, attn.head_dim), torch.zeros(5, attn.head_dim))

    def max_length(cu, config, kwargs):
        assert cu is lengths
        events.append("lengths")
        return 3

    def attention(module, q, k, v, **kwargs):
        events.append("attention")
        if flash:
            assert kwargs["cu_seq_lens_q"] is kwargs["cu_seq_lens_k"] is lengths
        return (q * k + v).transpose(1, 2), None

    monkeypatch.setattr(glm, "get_max_seqlen", max_length)
    monkeypatch.setattr(glm.ALL_ATTENTION_FUNCTIONS, "get_interface", lambda *args: attention)
    context = glm.TrainingContext()
    stage = glm.TrainingStage.FORWARD
    monkeypatch.setattr(context, "get_training_stage", lambda: stage)
    x = torch.ones(5, attn.dim, requires_grad=True)
    y = checkpoint(attn, x, lengths, positions, use_reentrant=False)
    forward_events = ["lengths", "attention"] if flash else ["attention", "attention"]
    assert events == forward_events
    events.clear()
    stage = glm.TrainingStage.BACKWARD
    y.sum().backward()
    if npu and flash:
        assert events == ["sync", *forward_events]
        current_stream.assert_called_once_with(x.device)
    else:
        assert events == forward_events
        current_stream.assert_not_called()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not glm.IS_NPU_AVAILABLE, reason="requires Ascend NPU Flash Attention")
def test_npu_flash_replay_matches_regular_backward(monkeypatch):
    from mindspeed_mm.fsdp.ops.flash_attn import flash_attn

    # This test runs the packed FA kernel on one device, without a distributed job.
    single_rank = SimpleNamespace(is_ring_enable=lambda: False, is_ulysses_enable=lambda: False,
                                  is_kvallgather_enable=lambda: False)
    monkeypatch.setattr(flash_attn, "get_parallel_state", lambda: single_rank)

    # Select the same attention backend as the trainer without modifying its global registry.
    monkeypatch.setattr(glm.ALL_ATTENTION_FUNCTIONS, "get_interface", lambda *args: flash_attn.flash_attention_forward)
    cfg = glm.Glm5NextVisionConfig(hidden_size=128, num_heads=2, attention_dropout=0.0)
    attn = glm.Glm5NextVisionAttention(cfg).to(device="npu", dtype=torch.bfloat16)
    values = torch.randn(128, 128, device="npu", dtype=torch.bfloat16)
    lengths = torch.tensor([0, 48, 128], dtype=torch.int32)
    positions = (values.new_ones(128, 64), values.new_zeros(128, 64))
    stage = glm.TrainingStage.FORWARD
    monkeypatch.setattr(glm.TrainingContext(), "get_training_stage", lambda: stage)

    def run(replay):
        nonlocal stage
        stage = glm.TrainingStage.FORWARD
        x = values.detach().requires_grad_()
        if replay:
            output = checkpoint(attn, x, lengths, positions, max_seqlen=80, use_reentrant=False)
        else:
            output = attn(x, lengths, positions, max_seqlen=80)
        stage = glm.TrainingStage.BACKWARD
        grads = torch.autograd.grad(output.float().square().mean(), (x, *attn.parameters()))
        return (output.detach(), *grads)

    baseline = run(replay=False)
    for actual, expected in zip(run(replay=True), baseline):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=1e-5)
