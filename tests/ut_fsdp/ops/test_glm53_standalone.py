# Copyright © 2026 Huawei Technologies Co., Ltd.
"""CPU contracts for GLM expert activation, QKV convolution and routing."""

import ast
from functools import partial
import contextlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional, Unpack
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, namespace, class_name=None):
    """Load a production function without requiring the model's NPU imports."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(node.body for node in nodes
                     if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in nodes
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def load_class(path, name, namespace):
    """Load a class definition without importing its full module."""
    tree = ast.parse((ROOT / path).read_text())
    cls_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[cls_node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("limit", [0.0, -1.0, 10.0])
def test_expert_activation_matches_split_clamp(dtype, limit):
    def swiglu(value, dim, fused):
        gate, up = value.chunk(2, dim=dim)
        return torch.nn.functional.silu(gate) * up

    activation = load_function(
        "mindspeed_mm/fsdp/models/glm5_next/modeling_glm5_next.py", "_ep_activation",
        dict(torch=torch, swiglu=swiglu, clamp_with_fused_backward=torch.clamp),
        "Glm5NextTextExperts",
    )
    module = SimpleNamespace(swiglu_limit=limit, intermediate_dim=8,
                             _min_bound=None, _max_bound=None)
    value = torch.linspace(-12, 12, 64, dtype=dtype).reshape(4, 16).requires_grad_()
    reference_input = value.detach().clone().requires_grad_()
    gate, up = reference_input.chunk(2, dim=-1)
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    reference = torch.nn.functional.silu(gate) * up
    result = activation(module, value, fused=False)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    gradient = torch.randn_like(result)
    result.backward(gradient)
    reference.backward(gradient)
    torch.testing.assert_close(value.grad, reference_input.grad, rtol=0, atol=0)


def test_qkv_convolution_concatenation_and_head_order():
    calls = []

    def convolution(x, weight, heads, bias, activation):
        calls.append(heads)
        # A deterministic per-channel operation preserves the depthwise
        # convolution's channel independence while checking the wrapper layout.
        result = x * weight[:, 0] + bias
        return result.reshape(2, 5, heads, -1).transpose(1, 2).contiguous(), None

    function = load_function(
        "mindspeed_mm/fsdp/ops/gdn/causal_conv1d_ascendc.py", "causal_conv1d_qkv_ascendc",
        dict(torch=torch, Optional=Optional,
             causal_conv1d_ascendc=convolution),
    )
    value = torch.arange(2 * 5 * 24, dtype=torch.float32).reshape(2, 5, 24)
    weight = torch.randn(24, 4)
    bias = torch.randn(24)
    query, key, output = function(value, weight, 2, bias=bias, activation="silu")
    assert calls == [6]
    expected = (value * weight[:, 0] + bias).reshape(2, 5, 6, 4).transpose(1, 2)
    for index, tensor in enumerate((query, key, output)):
        assert tensor.shape == (2, 2, 5, 4)
        assert tensor.is_contiguous()
        torch.testing.assert_close(tensor, expected[:, index * 2:(index + 1) * 2])


@pytest.mark.parametrize("conv_impl,device,optimized", [
    ("ascendc", "Ascend910_9391", False), ("ascendc", "Ascend950", True), ("triton", "Ascend910_9391", False),
])
def test_glm_routes_short_conv_to_matching_kda_layout(conv_impl, device, optimized):
    hidden_states = torch.ones(1, 5, 256, dtype=torch.bfloat16)
    qkv = tuple(hidden_states.reshape(1, 5, 2, 128).transpose(1, 2).contiguous().clone() for _ in range(3))
    output = hidden_states.reshape(1, 5, 2, 128)
    triton_short_conv = Mock(side_effect=lambda **kwargs: (kwargs["x"], None))
    fused_short_conv = Mock(return_value=qkv)
    bsnd_kda = Mock(return_value=(output, None))
    bnsd_kda = Mock(return_value=(output, None))
    forget_gate = Mock(return_value=output)
    forget_gate.raw.return_value = hidden_states
    forget_gate.safe_gate_lower_bound = -5.0
    forget_gate.A_log = torch.ones(2)
    forget_gate.dt_bias = torch.zeros(256)

    forward = load_function(
        "mindspeed_mm/fsdp/models/glm5_next/modeling_glm5_next.py", "forward",
        dict(torch=torch, Cache=object, Unpack=Unpack, TransformersKwargs=dict,
             force_accelerate_hooks=lambda _: lambda fn: fn,
             apply_mask_to_padding_states=lambda x, _mask: x, IMPL_ASCENDC="ascendc", IMPL_TRITON="triton",
             causal_conv1d=triton_short_conv, causal_conv1d_qkv_ascendc=fused_short_conv,
             chunk_kda_ascendc=bsnd_kda, chunk_kda_bnsd=bnsd_kda, _require_kernel=Mock(),
             torch_npu=SimpleNamespace(npu=SimpleNamespace(get_device_name=lambda _: device))),
        "Glm5NextTextLinearAttention",
    )
    model = SimpleNamespace(
        qkv_proj=lambda x: x.repeat(1, 1, 3), conv1d=SimpleNamespace(weight=torch.ones(768, 1, 4), bias=None),
        forget_gate=forget_gate, b_proj=lambda x: x[..., :2], num_heads=2, head_dim=128, qkv_dim=256,
        activation="silu", causal_conv1d_implementation=conv_impl, kda_implementation="ascendc",
        g_a_proj=lambda x: x, g_b_proj=lambda x: x, o_norm=lambda out, gate: out, o_proj=lambda x: x,
    )
    torch.testing.assert_close(forward(model, hidden_states), hidden_states)
    if conv_impl == "ascendc":
        fused_short_conv.assert_called_once()
        triton_short_conv.assert_not_called()
        bsnd_kda.assert_not_called()
        args = bnsd_kda.call_args.kwargs
        assert all(args[name] is tensor for name, tensor in zip("qkv", qkv, strict=True))
        assert args["use_optimized_bwd"] is optimized
    else:
        fused_short_conv.assert_not_called()
        bnsd_kda.assert_not_called()
        triton_short_conv.assert_called_once()
        assert bsnd_kda.call_args.kwargs["q"].shape == output.shape


def test_clamp_fallback_preserves_boundary_and_special_value_gradients():
    clamp = load_function(
        "mindspeed_mm/fsdp/ops/swiglu.py", "clamp_with_fused_backward",
        dict(torch=torch, IS_NPU_AVAILABLE=False),
    )
    value = torch.tensor([[float("-inf"), -10, 0, 10, float("inf"), float("nan")]],
                         requires_grad=True)
    reference_input = value.detach().clone().requires_grad_()
    lower = torch.full((6,), -10.0)
    upper = torch.full((6,), 10.0)
    result = clamp(value, lower, upper)
    expected = torch.clamp(reference_input, min=lower, max=upper)
    torch.testing.assert_close(result, expected, equal_nan=True)
    gradient = torch.tensor([[float("nan"), 2, -3, 4, float("inf"), 6]])
    result.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(value.grad, reference_input.grad, equal_nan=True)


def test_clamp_fallback_preserves_3d_shape_and_gradients():
    clamp = load_function(
        "mindspeed_mm/fsdp/ops/swiglu.py", "clamp_with_fused_backward",
        dict(torch=torch, IS_NPU_AVAILABLE=False),
    )
    value = torch.linspace(-12, 12, 24).reshape(1, 4, 6).requires_grad_()
    ref_input = value.detach().clone().requires_grad_()
    lower = torch.tensor([-float("inf")] * 3 + [-10.0] * 3)
    upper = torch.full((6,), 10.0)
    result = clamp(value, lower, upper)
    expected = torch.clamp(ref_input, min=lower, max=upper)
    assert result.shape == (1, 4, 6)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    grad = torch.randn_like(result)
    result.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(value.grad, ref_input.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_bias", [True, False])
def test_vision_downsample_linear_equivalence(dtype, use_bias):
    Downsample = load_class(
        "mindspeed_mm/fsdp/models/glm5_next/modeling_glm5_next.py",
        "_Glm5NextVisionDownsample",
        dict(torch=torch, nn=torch.nn, F=torch.nn.functional, DTensor=tuple()),
    )
    torch.manual_seed(42)
    merge, in_c, out_c, n_merged = 2, 16, 24, 5
    conv = torch.nn.Conv2d(in_c, out_c, kernel_size=merge, stride=merge, bias=use_bias).to(dtype)
    fast = Downsample(in_c, out_c, kernel_size=merge, stride=merge, bias=use_bias).to(dtype)
    with torch.no_grad():
        fast.weight.copy_(conv.weight)
        if use_bias:
            fast.bias.copy_(conv.bias)

    x_ref = torch.randn((n_merged * merge * merge, in_c), dtype=dtype, requires_grad=True)
    x_fast = x_ref.detach().clone().requires_grad_()
    y_ref = conv(x_ref.view(-1, merge, merge, in_c).permute(0, 3, 1, 2)).view(-1, out_c)
    y_fast = fast(x_fast.view(-1, merge, merge, in_c).permute(0, 3, 1, 2)).view(-1, out_c)

    # BF16 conv and linear kernels can accumulate weight gradients in different orders.
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    tol = dict(rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(y_fast, y_ref, **tol)
    g = torch.randn_like(y_ref)
    y_ref.backward(g)
    y_fast.backward(g)
    torch.testing.assert_close(x_fast.grad, x_ref.grad, **tol)
    torch.testing.assert_close(fast.weight.grad, conv.weight.grad, **tol)
    if use_bias:
        torch.testing.assert_close(fast.bias.grad, conv.bias.grad, **tol)

    # Module invocation must preserve forward hooks and fallback cleanly when dilation != (1, 1).
    hook_calls = []
    handle = fast.register_forward_hook(lambda _mod, _inp, out: (hook_calls.append(out.shape), out + 1.0)[1])
    try:
        y_hooked = fast(x_fast.detach().view(-1, merge, merge, in_c).permute(0, 3, 1, 2))
    finally:
        handle.remove()
    assert hook_calls == [(n_merged, out_c, 1, 1)]
    torch.testing.assert_close(y_hooked.view(-1, out_c), y_fast.detach() + 1.0, rtol=0, atol=0)

    dilated = Downsample(in_c, out_c, kernel_size=merge, stride=merge, dilation=2, bias=use_bias).to(dtype)
    dilated_ref = torch.nn.Conv2d(in_c, out_c, kernel_size=merge, stride=merge, dilation=2, bias=use_bias).to(dtype)
    with torch.no_grad():
        dilated.weight.copy_(dilated_ref.weight)
        if use_bias:
            dilated.bias.copy_(dilated_ref.bias)
    x_dil = torch.randn((2, in_c, 4, 4), dtype=dtype)
    torch.testing.assert_close(dilated(x_dil), dilated_ref(x_dil), rtol=0, atol=0)


@pytest.mark.parametrize(
    "dispatcher,is_npu,fused,in_dtype,expected_w_dtype",
    [
        ("chunkmoe", True, True, torch.bfloat16, torch.bfloat16),
        ("alltoall", True, True, torch.bfloat16, torch.bfloat16),
        ("mc2", True, True, torch.bfloat16, torch.bfloat16),
        ("mc2_udma", True, True, torch.bfloat16, torch.bfloat16),
        ("allgather", True, True, torch.bfloat16, torch.float32),
        ("chunkmoe", True, False, torch.bfloat16, torch.float32),
        ("chunkmoe", False, True, torch.bfloat16, torch.float32),
        ("chunkmoe", True, True, torch.float16, torch.float32),
    ],
)
def test_ep_forward_casts_top_k_weights_to_hidden_dtype(
    monkeypatch, dispatcher, is_npu, fused, in_dtype, expected_w_dtype
):
    seen = {}

    def fake_dispatcher(num_experts, routing_weights, selected_experts, hidden_states, **kwargs):
        seen["routing_weights_dtype"] = routing_weights.dtype
        assert kwargs["fc1_weight"].shape == (2, 8, 16)
        assert kwargs["fc2_weight"].shape == (2, 8, 8)
        return hidden_states

    disp_mod = ModuleType("mindspeed_mm.fsdp.distributed.expert_parallel.ep_dispatcher")
    for fn_name in ("ep_forward", "ep_mc2_forward", "ep_allgather_forward", "ep_mc2_udma_forward"):
        setattr(disp_mod, fn_name, fake_dispatcher)
    chunk_mod = ModuleType("mindspeed_mm.fsdp.distributed.expert_parallel.ep_chunkmoe")
    chunk_mod.ep_chunkmoe_forward = fake_dispatcher
    monkeypatch.setitem(sys.modules, disp_mod.__name__, disp_mod)
    monkeypatch.setitem(sys.modules, chunk_mod.__name__, chunk_mod)

    ep_forward_fn = load_function(
        "mindspeed_mm/fsdp/models/glm5_next/modeling_glm5_next.py", "ep_forward",
        dict(torch=torch, IS_NPU_AVAILABLE=is_npu, DTensor=tuple(), ProcessGroup=object, EPPlanConfig=object, partial=partial),
        "Glm5NextTextExperts",
    )
    expert_mod = SimpleNamespace(
        num_experts=16,
        hidden_dim=8,
        swiglu_limit=7.0,
        gate_up_proj=torch.zeros(2, 16, 8, dtype=in_dtype),
        down_proj=torch.zeros(2, 8, 8, dtype=in_dtype),
        _ep_activation=lambda x, fused=True: x,
    )
    hs = torch.randn(4, 8, dtype=in_dtype)
    w = torch.rand(4, 2, dtype=torch.float32)
    idx = torch.zeros(4, 2, dtype=torch.int64)
    plan = SimpleNamespace(dispatcher=dispatcher, use_npu_fused_ops=fused)
    out = ep_forward_fn(expert_mod, hs, idx, w, ep_group=None, ep_plan=plan)
    torch.testing.assert_close(out, hs)
    assert seen["routing_weights_dtype"] == expected_w_dtype


def test_rms_norm_gated_npu_bf16_matches_fp32_reference(monkeypatch):
    triton_calls = []

    def fake_rmsnorm_gated(x, gate, weight, eps, activation):
        triton_calls.append(x.dtype)
        xf = x.float()
        norm = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype).float() * weight.float()
        if torch.promote_types(x.dtype, weight.dtype) != torch.float32:
            norm = norm.to(x.dtype).float()
        return (norm * torch.sigmoid(gate.float())).to(x.dtype)

    triton_mod = ModuleType("mindspeed_mm.fsdp.ops.rmsnorm_gated.rmsnorm_gated_triton")
    triton_mod._supported_inputs = lambda x, g, w: x.dtype == torch.bfloat16 and x.shape == g.shape
    triton_mod.rmsnorm_gated = fake_rmsnorm_gated
    monkeypatch.setitem(sys.modules, triton_mod.__name__, triton_mod)

    def fake_npu_rms_norm(x, weight, eps):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype), None

    fake_torch_npu = SimpleNamespace(npu_rms_norm=fake_npu_rms_norm)
    act = {"sigmoid": torch.sigmoid}
    fn_npu = load_function(
        "mindspeed_mm/fsdp/models/glm5_next/modeling_glm5_next.py", "forward",
        dict(torch=torch, IS_NPU_AVAILABLE=True, DTensor=tuple(), torch_npu=fake_torch_npu, ACT2FN=act),
        "Glm5NextTextRMSNormGated",
    )
    fn_ref = load_function(
        "mindspeed_mm/fsdp/models/glm5_next/modeling_glm5_next.py", "forward",
        dict(torch=torch, IS_NPU_AVAILABLE=False, DTensor=tuple(), ACT2FN=act),
        "Glm5NextTextRMSNormGated",
    )

    torch.manual_seed(11)
    x0_data = torch.randn(1, 16, 4, 32, dtype=torch.bfloat16)
    x0_data[0, 0, 0, :] = 1.0
    x0 = x0_data.clone().requires_grad_()
    g0_data = torch.randn(1, 16, 4, 32, dtype=torch.bfloat16)
    g0_data[0, 0, 0, 0] = 7.0  # BF16 sigmoid rounds to 1.0, so naive BF16 backward drops grad to 0.
    g0 = g0_data.clone().requires_grad_()
    w0_data = torch.randn(32, dtype=torch.bfloat16)
    w0_data[0] = 1.0
    w0 = w0_data.clone().requires_grad_()
    x1, g1, w1 = [t.detach().clone().requires_grad_() for t in (x0, g0, w0)]

    y_npu = fn_npu(SimpleNamespace(weight=w0, variance_epsilon=1e-6, activation="sigmoid"), x0, g0)
    y_ref = fn_ref(SimpleNamespace(weight=w1, variance_epsilon=1e-6, activation="sigmoid"), x1, g1)
    assert triton_calls == [torch.bfloat16]
    torch.testing.assert_close(y_npu, y_ref, rtol=1e-2, atol=1e-2)

    grad_out = torch.ones_like(y_npu)
    y_npu.backward(grad_out)
    y_ref.backward(grad_out)
    assert abs(g0.grad[0, 0, 0, 0].item()) > 5e-4
    for gn, gr in ((x0.grad, x1.grad), (g0.grad, g1.grad), (w0.grad, w1.grad)):
        torch.testing.assert_close(gn, gr, rtol=2e-2, atol=2e-2)

    # Non-BF16 and gate=None stay off the Triton path and retain strict FP32 math.
    triton_calls.clear()
    _ = fn_npu(SimpleNamespace(weight=w0.detach().half(), variance_epsilon=1e-6, activation="sigmoid"), x0.detach().half(), g0.detach().half())
    y_nogate = fn_npu(SimpleNamespace(weight=w0.detach(), variance_epsilon=1e-6, activation="sigmoid"), x0.detach(), None)
    y_nogate_ref = fn_ref(SimpleNamespace(weight=w1.detach(), variance_epsilon=1e-6, activation="sigmoid"), x1.detach(), None)
    assert triton_calls == []
    torch.testing.assert_close(y_nogate, y_nogate_ref, rtol=1e-2, atol=1e-2)


def test_clamp_fused_backward_supports_create_graph(monkeypatch):
    def fake_kernel(x, dy, lower, upper, dx, *_args):
        dx.copy_(torch.where((x >= lower) & (x <= upper), dy, 0))

    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=lambda _d: contextlib.nullcontext()), raising=False)

    clamp_cls = load_class(
        "mindspeed_mm/fsdp/ops/glu/clamp_triton.py",
        "ClampWithFusedBackward",
        dict(
            torch=torch,
            torch_npu=SimpleNamespace(get_npu_format=lambda _t: 2),
            triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
            _vector_cores=lambda _idx: 1,
            _clamp_dx={(1,): fake_kernel},
        ),
    )
    lo = torch.full((4,), -10.0)
    hi = torch.full((4,), 10.0)
    x = torch.tensor([[-12.0, -3.0, 4.0, 12.0]], requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    y = clamp_cls.apply(x, lo, hi)
    y_ref = torch.clamp(x_ref, min=lo, max=hi)
    (dx,) = torch.autograd.grad(0.5 * (y ** 2).sum(), x, create_graph=True)
    (dx_ref,) = torch.autograd.grad(0.5 * (y_ref ** 2).sum(), x_ref, create_graph=True)
    (d2x,) = torch.autograd.grad(dx.sum(), x)
    (d2x_ref,) = torch.autograd.grad(dx_ref.sum(), x_ref)
    torch.testing.assert_close(d2x, d2x_ref, rtol=0, atol=0)
