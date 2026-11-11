"""Fused HC forward with explicit small-operator backward. First-order only.

Requires cann_ops_transformer and its matching installed ACLNN package.
Returns mixed and optional inject; forward cache stays private to autograd.
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

def backward_from_cache(x, w_down, w_up, w_gate, gamma, channels,
                        grad_mixed, grad_inject, cache, eps=1e-6):
    use_combine = w_gate is not None
    """Reuse fused caches with the native-dtype small-op backward path."""
    *lead, d = x.shape
    h = d // channels
    xg32 = x.reshape(*lead, channels, h).float()
    inv = cache["inv_rms"].unsqueeze(-1).float()
    normalized = xg32 * inv
    if gamma is not None:
        normalized = normalized * (1.0 + gamma.reshape(channels, h).float())
    xn = normalized.reshape(*lead, d).to(x.dtype)
    down = cache["down_pre_act"]
    silu = F.silu(down)
    # New kernels return up_pre_act with the same dtype as x. Keep the native
    # dtype before sigmoid to preserve the BF16 small-op backward path.
    up = cache.get("up_pre_act")
    if up is None:
        up = torch.matmul(silu, w_up.transpose(-1, -2))
    gate = torch.sigmoid(up.to(x.dtype))
    gate_g = gate.reshape(*lead, channels, h)
    xn_g = xn.reshape(*lead, channels, h)
    dg = grad_mixed.to(x.dtype).unsqueeze(-2).expand_as(xn_g) / channels
    dup = torch.ops.aten.sigmoid_backward(
        (dg * xn_g).reshape(*lead, d), gate)
    dw_up = torch.matmul(dup.reshape(-1, d).transpose(0, 1),
                         silu.reshape(-1, silu.shape[-1]))
    dsilu = torch.matmul(dup, w_up)
    # Divide before the GEMMs, matching backward through down/C.
    ddown = torch.ops.aten.silu_backward(dsilu, down) / channels
    dw_down = torch.matmul(ddown.reshape(-1, down.shape[-1]).transpose(0, 1),
                           xn.reshape(-1, d))
    dxn = torch.matmul(ddown, w_down)
    dw_gate = None
    if use_combine and grad_inject is not None:
        inject_gate = cache["inject_optional"] / 2
        dlogits = torch.ops.aten.sigmoid_backward(
            grad_inject.to(x.dtype) * 2, inject_gate) / channels
        dw_gate = torch.matmul(dlogits.reshape(-1, channels).transpose(0, 1),
                               xn.reshape(-1, d))
        dxn_right = torch.matmul(dlogits, w_gate)
    direct = (dg * gate_g).reshape(*lead, d)
    dxn_total = ((dxn_right + direct) + dxn) if use_combine and grad_inject is not None else (direct + dxn)
    q = dxn_total.reshape(
        *lead, channels, h).float()
    if gamma is not None:
        q_gamma = q * (1.0 + gamma.reshape(channels, h).float())
        dgamma = (q * xg32 * inv).sum(dim=tuple(range(len(lead)))).reshape_as(gamma)
    else:
        q_gamma = q
        dgamma = None
    dxg = inv * (q_gamma - xg32 * (q_gamma * xg32).mean(
        dim=-1, keepdim=True) * inv.square())
    return dxg.reshape_as(x).to(x.dtype), dw_down, dw_up, dw_gate, dgamma


class TextGatedResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_down, w_up, w_gate, gamma, channels, eps):
        use_combine = w_gate is not None
        from cann_ops_transformer.ops.attention.text_gated_residual import (
            text_gated_residual as fused_forward,
        )
        outputs = fused_forward(
            x, w_down, w_up, w_gate=w_gate, gamma=gamma,
            channels=channels, eps=eps,
            need_backward=True,
        )
        if use_combine:
            mixed, inject, inv_rms, down_pre_act = outputs[:4]
            up_pre_act = outputs[4] if len(outputs) == 5 else None
        else:
            mixed, inv_rms, down_pre_act = outputs[:3]
            up_pre_act = outputs[3] if len(outputs) == 4 else None
            inject = None
        ctx.channels, ctx.eps, ctx.use_combine = channels, eps, use_combine
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, w_down, w_up, w_gate, gamma,
                              inv_rms, down_pre_act, inject, up_pre_act)
        return mixed, inject

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_mixed, grad_inject):
        x, wd, wu, wg, gamma, inv, down, inject, up = ctx.saved_tensors
        if grad_mixed is None:
            grad_mixed = x.new_zeros((*x.shape[:-1], x.shape[-1] // ctx.channels))
        cache = {"inv_rms": inv, "down_pre_act": down, "up_pre_act": up}
        if ctx.use_combine:
            cache["inject_optional"] = inject
        grads = backward_from_cache(
            x, wd, wu, wg, gamma, ctx.channels, grad_mixed,
            grad_inject,
            cache, ctx.eps,
        )
        inputs = (x, wd, wu, wg, gamma)
        grads = tuple(g.to(t.dtype) if g is not None and t is not None and needed else None
                      for g, t, needed in zip(grads, inputs, ctx.needs_input_grad[:5]))
        return (*grads, None, None)


def qhc_ascendc(
    x, w_down, w_up, w_gate=None, gamma=None, channels=4, eps=1e-6,
):
    """Fused forward + small-op backward for x/weights/optional gamma.

    Passing w_gate returns (mixed, inject); omitting it returns mixed.
    Supports first-order gradients, including losses using only one output.
    """
    if gamma is not None and (gamma.ndim != 1 or gamma.numel() != x.shape[-1]):
        raise ValueError('gamma must have shape [C*H], matching x.shape[-1]')
    mixed, inject = TextGatedResidualFunction.apply(
        x, w_down, w_up, w_gate, gamma, channels, eps,
    )
    return (mixed, inject) if w_gate is not None else mixed
