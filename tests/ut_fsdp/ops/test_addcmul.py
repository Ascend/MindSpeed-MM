"""Forward, first-order gradients, and dispatch regressions for addcmul."""
import importlib

import pytest
import torch

from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE


@pytest.fixture
def addcmul():
    pytest.importorskip('triton')
    return importlib.import_module('mindspeed_mm.fsdp.ops.addcmul.addcmul_triton')


@pytest.fixture
def npu_device():
    if not IS_NPU_AVAILABLE:
        pytest.skip('An Ascend NPU is required for the Triton fast path')
    return torch.device('npu', torch.npu.current_device())


def reference(x, w, r, rounding):
    if rounding:
        return r + (x.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)
    return torch.addcmul(r.unflatten(-1, (w.shape[-1], x.shape[-1])),
                        x.unsqueeze(-2), w.unsqueeze(-1)).flatten(-2)


def check_forward_backward(addcmul, x, w, r, g, rounding):
    actual = addcmul.triton_addcmul(x, w, r, round_product=rounding)
    expected = reference(x, w, r, rounding)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradients = torch.autograd.grad(actual, (x, w, r), g)
    expected_gradients = torch.autograd.grad(expected, (x, w, r), g)
    atol, rtol = (1e-5, .01) if x.dtype == torch.bfloat16 else (1e-3, 1e-3)
    for actual_grad, expected_grad in zip(gradients[:2], expected_gradients[:2]):
        torch.testing.assert_close(actual_grad, expected_grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(gradients[2], g, atol=0, rtol=0)


# Keep the model's H=2560/C=4 and [B, S, H] layout, with smaller B/S for UT.
# Reuse these specializations below; exercise generic boundaries only in BF16
# instead of compiling every shape/dtype/rounding combination in CI.
@pytest.mark.parametrize('leading,h,c,dtype,rounding', [
    pytest.param((2, 17), 2560, 4, torch.bfloat16, True, id='model-bf16-tail'),
    pytest.param((1, 32), 2560, 4, torch.bfloat16, True, id='model-bf16-aligned'),
    pytest.param((2, 17), 2560, 4, torch.float16, True, id='model-fp16'),
    pytest.param((2, 17), 2560, 4, torch.bfloat16, False, id='addcmul-bf16'),
    pytest.param((2, 17), 2560, 4, torch.float16, False, id='addcmul-fp16'),
    pytest.param((7,), 17, 1, torch.bfloat16, True, id='single-channel-tail'),
    pytest.param((17,), 4096, 16, torch.bfloat16, True, id='max-channels-reduce'),
    pytest.param((19,), 4097, 5, torch.bfloat16, True, id='unaligned-hidden-reduce'),
])
def test_fast_forward_and_gradients(addcmul, npu_device, monkeypatch,
                                    leading, h, c, dtype, rounding):
    monkeypatch.setenv('ADDCMUL_TRITON', '1')
    torch.manual_seed(20261120)
    x, w, r = [torch.randn((*leading, d), device=npu_device, dtype=dtype, requires_grad=True)
               for d in (h, c, h*c)]
    check_forward_backward(addcmul, x, w, r, torch.randn_like(r), rounding)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('layout', ['strided', 'expanded'])
def test_noncontiguous_gradient(addcmul, npu_device, monkeypatch, dtype, layout):
    monkeypatch.setenv('ADDCMUL_TRITON', '1')
    x, w, r = [torch.randn((2, 17, d), device=npu_device, dtype=dtype, requires_grad=True)
               for d in (2560, 4, 10240)]
    if layout == 'strided':
        g = torch.randn((2, 17, 20480), device=npu_device, dtype=dtype)[..., ::2]
    else:
        g = torch.randn((2, 17, 1), device=npu_device, dtype=dtype).expand_as(r)
    assert not g.is_contiguous()
    check_forward_backward(addcmul, x, w, r, g, True)


@pytest.mark.parametrize('kind', ['disabled', 'small_auto', 'strided', 'broadcast',
                                  'float32', 'channels17', 'empty'])
def test_native_fallback(addcmul, npu_device, monkeypatch, kind):
    monkeypatch.setenv('ADDCMUL_TRITON', {'disabled': '0', 'small_auto': 'auto'}.get(kind, '1'))
    dtype = torch.float32 if kind == 'float32' else torch.bfloat16
    h, c = 17, 17 if kind == 'channels17' else 3
    leading = (0, 7) if kind == 'empty' else (2, 7)
    shapes = [(*leading, d) for d in (h, c, h*c)]
    if kind == 'broadcast':
        shapes[:2] = [(2, 1, h), (1, 7, c)]
    tensors = []
    for shape in shapes:
        z = torch.randn((*shape[:-1], shape[-1]*2) if kind == 'strided' else shape,
                        device=npu_device, dtype=dtype)
        tensors.append((z[..., ::2] if kind == 'strided' else z).detach().requires_grad_())
    def unexpected_fast_path(*args, **kwargs):
        raise AssertionError('Unsupported inputs must use the native path')
    monkeypatch.setattr(addcmul._Addcmul, 'apply', unexpected_fast_path)
    x, w, r = tensors
    check_forward_backward(addcmul, x, w, r, torch.randn_like(r), True)


@pytest.mark.parametrize('fused,npu_available,expected', [(True, True, 'fused'),
                                                       (False, True, 'eager'),
                                                       (True, False, 'eager')])
def test_gated_dispatch(monkeypatch, fused, npu_available, expected):
    module = importlib.import_module('mindspeed_mm.fsdp.ops.addcmul.gated_residual')
    monkeypatch.setattr(module, 'IS_NPU_AVAILABLE', npu_available)
    monkeypatch.setattr(module, 'fused_residual_injection', lambda *args: 'fused')
    monkeypatch.setattr(module, 'eager_residual_injection', lambda *args: 'eager')
    assert module.gated_residual_injection(None, None, None, fused=fused) == expected


@pytest.mark.parametrize('fused', [True, False])
def test_gated_numerics(npu_device, monkeypatch, fused):
    module = importlib.import_module('mindspeed_mm.fsdp.ops.addcmul.gated_residual')
    monkeypatch.setenv('ADDCMUL_TRITON', '1')
    x, w, r = [torch.randn((2, 17, d), device=npu_device, dtype=torch.bfloat16, requires_grad=True)
               for d in (2560, 4, 10240)]
    actual = module.gated_residual_injection(x, w, r, fused=fused)
    expected = reference(x, w, r, True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    g = torch.randn_like(r)
    for a, z in zip(torch.autograd.grad(actual, (x, w, r), g),
                    torch.autograd.grad(expected, (x, w, r), g)):
        torch.testing.assert_close(a, z, atol=1e-5, rtol=.01)


@pytest.mark.parametrize('bad', ['scalar', 'residual_size'])
def test_invalid_shape(addcmul, bad):
    x, w, r = torch.randn(2, 17), torch.randn(2, 3), torch.randn(2, 51)
    if bad == 'scalar':
        x = torch.tensor(1.)
    else:
        r = r[..., :-1]
    with pytest.raises(ValueError):
        addcmul.triton_addcmul(x, w, r)
