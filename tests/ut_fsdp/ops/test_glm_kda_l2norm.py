# Copyright © 2026 Huawei Technologies Co., Ltd.
import importlib.util
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts


def load_wrapper(filename):
    path = Path(__file__).resolve().parents[3] / "mindspeed_mm/fsdp/ops/kda/ascendc" / filename
    spec = importlib.util.spec_from_file_location("glm_kda", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def glm_kda_triton(monkeypatch):
    def identity(fn):
        return fn

    triton = SimpleNamespace(
        input_guard=identity, autocast_custom_fwd=identity, autocast_custom_bwd=identity,
        chunk_kda_fwd=None, chunk_kda_bwd=Mock(), prepare_chunk_indices=None,
        l2norm_fwd=Mock(side_effect=lambda x: (x.clone(), x.float().sum(-1))),
        l2norm_bwd=Mock(side_effect=lambda x, rstd, grad: grad),
    )
    monkeypatch.setitem(sys.modules, "triton_ascend_kernels.attention.fla.kda.chunk", triton)
    return triton


@pytest.mark.parametrize(
    "optimized,atlas_a3,tokens,safe_gate",
    [
        (False, True, 64, True),
        (False, True, 65, True),
        (False, False, 64, True),
        (True, False, 64, True),
        pytest.param(False, True, 64, False, id="reject-unsafe-gate"),
    ],
)
def test_qk_norm_forward_backward_pair(monkeypatch, glm_kda_triton, optimized, atlas_a3, tokens, safe_gate):
    fused = optimized or atlas_a3

    saved = []

    def forward(q, k, v, g, *args, layout, **options):
        assert layout == "BNSD"
        length = q.shape[2]
        assert length == ((tokens + 63) // 64) * 64
        assert options.get("use_qk_l2norm_in_kernel", False) == fused
        if fused:
            for index, name in enumerate(("q", "k"), start=1):
                hat, rstd = options[f"{name}_hat_out"], options[f"{name}_rstd_out"]
                assert (hat.shape, hat.dtype) == (q.shape, torch.bfloat16)
                assert (rstd.shape, rstd.dtype) == (q.shape[:-1], torch.float32)
                hat.fill_(index)
                rstd.fill_(index + 2)
                saved.extend((hat, rstd))
        aqk, akk = (q.new_empty((1, 2, length, 64)) for _ in range(2))
        state = q.new_empty((1, length // 64, 2, 128, 128))
        return (v.transpose(1, 2).clone(), None, g.float(), aqk, akk,
                q.clone(), None, q.clone(), k.clone(), v.clone(), state, None)

    def backward(q, k, v, beta, *args, layout, q_rstd, k_rstd, implementation, raw_g, **kwargs):
        assert layout == "BNSD"
        assert implementation == ("optimized" if optimized else "auto")
        assert beta.dtype == (torch.float32 if optimized else torch.bfloat16)
        if fused:
            torch.testing.assert_close((q, q_rstd, k, k_rstd), tuple(saved), rtol=0, atol=0)
        else:
            assert q_rstd is None and k_rstd is None
        return q, k, v, beta, raw_g, None, None, None

    ops = SimpleNamespace(chunk_kda_fwd=Mock(side_effect=forward), chunk_kda_bwd=backward)
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", ops)
    module = load_wrapper("chunk_kda_ascendc_glm.py")
    monkeypatch.setattr(module, "_is_atlas_a3_device", lambda device: atlas_a3)

    q, k, v = [torch.ones(1, 2, tokens, 128, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    g = torch.zeros_like(q.transpose(1, 2)).contiguous()
    beta = g[..., 0].contiguous()
    kda = partial(
        module.chunk_kda_ascendc, qkv_layout="BNSD", disable_recompute=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=safe_gate,
        lower_bound=-5.0, A_log=torch.ones(2), use_optimized_bwd=optimized,
    )
    if not safe_gate:
        with pytest.raises(ValueError, match="BNSD KDA requires safe_gate=True"):
            kda(q, k, v, g, beta)
        ops.chunk_kda_fwd.assert_not_called()
        return

    context = partial(create_selective_checkpoint_contexts, [torch.ops.mindspeed_mm.chunk_kda_fwd_bnsd.default])
    out, _ = checkpoint(kda, q, k, v, g, beta, use_reentrant=False, context_fn=context)
    assert out.shape == (1, tokens, 2, 128)
    out.sum().backward()
    ops.chunk_kda_fwd.assert_called_once()
    assert q.grad.shape == k.grad.shape == q.shape
    assert glm_kda_triton.l2norm_fwd.call_count == (0 if fused else 4)
    assert glm_kda_triton.l2norm_bwd.call_count == (0 if fused else 2)


def test_glm_bsnd_with_recompute_uses_triton_backward(monkeypatch, glm_kda_triton):
    q, k, v, g = [torch.ones(1, 64, 2, 128, dtype=torch.bfloat16, requires_grad=True) for _ in range(4)]
    beta = torch.ones(1, 64, 2, dtype=torch.bfloat16, requires_grad=True)
    head_major = q.detach().transpose(1, 2).contiguous()
    matrix = q.new_empty((1, 2, 64, 64))
    state = q.new_empty((1, 1, 2, 128, 128))
    forward = Mock(return_value=(v, None, head_major.float(), matrix, matrix, head_major, None,
                                 head_major, head_major, head_major, state, None))
    ascendc_backward = Mock()
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", SimpleNamespace(
        chunk_kda_fwd=forward, chunk_kda_bwd=ascendc_backward,
    ))
    glm_kda_triton.chunk_kda_bwd.return_value = (q, k, v, beta, g, None, None, None)
    module = load_wrapper("chunk_kda_ascendc_glm.py")

    out, _ = module.chunk_kda_ascendc(q, k, v, g, beta, use_gate_in_kernel=True, A_log=torch.ones(2))
    out.sum().backward()
    assert forward.call_args.kwargs["disable_recompute"] is False
    ascendc_backward.assert_not_called()
    glm_kda_triton.chunk_kda_bwd.assert_called_once()
    assert glm_kda_triton.chunk_kda_bwd.call_args.kwargs["disable_recompute"] is False


@pytest.mark.parametrize("option", ["state_v_first", "transpose_state_layout"])
def test_glm_rejects_v_first_state_layout(monkeypatch, glm_kda_triton, option):
    ops = SimpleNamespace(chunk_kda_fwd=Mock(), chunk_kda_bwd=Mock())
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", ops)
    module = load_wrapper("chunk_kda_ascendc_glm.py")

    with pytest.raises(NotImplementedError, match="does not support state_v_first=True"):
        if option == "transpose_state_layout":
            with pytest.warns(DeprecationWarning, match="transpose_state_layout"):
                module.chunk_kda_ascendc(None, None, None, None, None, **{option: True})
        else:
            module.chunk_kda_ascendc(None, None, None, None, None, **{option: True})
    ops.chunk_kda_fwd.assert_not_called()


def test_bsnd_replay_passes_token_major_backward_inputs(monkeypatch):
    def identity(fn):
        return fn

    monkeypatch.setitem(sys.modules, "triton_ascend_kernels.attention.fla.kda.fla_utils", SimpleNamespace(
        input_guard=identity, autocast_custom_fwd=identity, autocast_custom_bwd=identity,
    ))
    monkeypatch.setitem(sys.modules, "triton_ascend_kernels.attention.fla.kda.l2norm_kda", SimpleNamespace(
        l2norm_fwd=None, l2norm_bwd=None,
    ))
    custom_op = Mock(return_value=identity)
    monkeypatch.setattr(torch.library, "custom_op", custom_op)

    def forward(q, k, v, g, beta, *args, layout, **kwargs):
        assert layout == "BSND"
        head_major = q.transpose(1, 2).contiguous()
        matrix = q.new_empty((1, 2, 64, 64))
        state = q.new_empty((1, 1, 2, 128, 128))
        return v, None, head_major.float(), matrix, matrix, head_major, None, head_major, head_major, head_major, state, None

    def backward(q, k, v, beta, *args, layout, raw_g, **kwargs):
        assert layout == "BSND"
        assert q.shape == k.shape == v.shape == raw_g.shape == args[-2].shape == (1, 64, 2, 128)
        assert beta.shape == (1, 64, 2)
        return q, k, v, beta, raw_g, None, None, None

    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", SimpleNamespace(
        chunk_kda_fwd=forward, chunk_kda_bwd=backward,
    ))
    module = load_wrapper("chunk_kda_wrapper.py")
    custom_op.assert_called_once_with("mindspeed_mm::chunk_kda_fwd", mutates_args=())
    q, k, v, g = [torch.ones(1, 64, 2, 128, dtype=torch.bfloat16, requires_grad=True) for _ in range(4)]
    beta = torch.ones(1, 64, 2, dtype=torch.bfloat16, requires_grad=True)
    out, _ = module.chunk_kda_ascendc(q, k, v, g, beta, use_gate_in_kernel=True, A_log=torch.ones(2))
    out.sum().backward()
    assert all(t.grad.shape == t.shape for t in (q, k, v, g, beta))
