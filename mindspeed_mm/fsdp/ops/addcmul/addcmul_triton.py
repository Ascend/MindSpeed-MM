
"""Broadcast multiply/add fusion for Ascend, with a fused first-order backward.

Model API: triton_addcmul(X[..., H], W[..., C], R[..., C*H]).
The leading dimensions (batch/sequence etc.) are arbitrary. H and C are
compile-time parameters derived from the inputs, not fixed model shapes.

round_product=True preserves R + flatten(X[...,None,:] * W[...,:,None]).
round_product=False follows torch.addcmul's single output rounding instead.
Only first-order AD is supported by the fast path. Set ADDCMUL_TRITON=0
for native operations, including higher-order derivatives.
"""
import os

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from mindspeed_mm.fsdp.utils.device import get_vector_num, is_arch35


@triton.jit(do_not_specialize=["T"])
def _addcmul_fwd_kernel(X, W, R, Y, T, H: tl.constexpr,
                       C: tl.constexpr, BR: tl.constexpr, BH: tl.constexpr,
                       BC: tl.constexpr, ROUND_PRODUCT: tl.constexpr):
    rr = tl.arange(0, BR)
    hh = tl.arange(0, BH)
    cc = tl.arange(0, BC)
    NT: tl.constexpr = triton.cdiv(H, BH)
    for task in range(tl.program_id(0), tl.cdiv(T, BR)*NT, tl.num_programs(0)):
        row = ((task // NT)*BR + rr).to(tl.int64)
        col = (task % NT)*BH + hh
        x = tl.load(X + row[:, None]*H + col[None, :],
                    (row[:, None] < T) & (col[None, :] < H), other=0).to(tl.float32)
        w = tl.load(W + row[:, None]*C + cc[None, :],
                    (row[:, None] < T) & (cc[None, :] < C), other=0).to(tl.float32)
        off = row[:, None, None]*(C*H) + cc[None, :, None]*H + col[None, None, :]
        mask = ((row[:, None, None] < T) & (cc[None, :, None] < C)
                & (col[None, None, :] < H))
        r = tl.load(R + off, mask, other=0).to(tl.float32)
        p = x[:, None, :]*w[:, :, None]
        if ROUND_PRODUCT:
            p = p.to(X.dtype.element_ty).to(tl.float32)
        tl.store(Y + off, r + p, mask)


@triton.jit(do_not_specialize=["T"])
def _addcmul_bwd_kernel(X, W, G, DX, P, T, H: tl.constexpr,
                        C: tl.constexpr, BR: tl.constexpr, BH: tl.constexpr,
                        IS_ARCH35: tl.constexpr):
    rr = tl.arange(0, BR)
    hh = tl.arange(0, BH)
    BC: tl.constexpr = triton.next_power_of_2(C)
    cc = tl.arange(0, BC)
    NT: tl.constexpr = triton.cdiv(H, BH)
    for task in range(tl.program_id(0), tl.cdiv(T, BR)*NT, tl.num_programs(0)):
        row = ((task // NT)*BR + rr).to(tl.int64)
        tile = task % NT
        col = tile*BH + hh
        mask = (row[:, None] < T) & (col[None, :] < H)
        if IS_ARCH35:
            # A5 benefits from one contiguous W tile followed by gathers.
            weights = tl.load(W + row[:, None]*C + cc[None, :],
                              (row[:, None] < T) & (cc[None, :] < C), other=0).to(tl.float32)
        x = tl.load(X + row[:, None]*H + col[None, :], mask, other=0).to(tl.float32)
        dx = tl.full((BR, BH), 0, tl.float32)
        for c in tl.static_range(C):
            if IS_ARCH35:
                w = tl.gather(weights, tl.full((BR, 1), c, tl.int32), axis=1).reshape((BR,))
            else:
                # Avoid gather overhead on A2/A3; the branch is compile-time.
                w = tl.load(W + row*C + c, row < T, other=0).to(tl.float32)
            g = tl.load(G + row[:, None]*C*H + c*H + col[None, :],
                        mask, other=0).to(tl.float32)
            # Native autograd materializes each product in the input dtype.
            dx += (g*w[:, None]).to(X.dtype.element_ty).to(tl.float32)
            pw = (g*x).to(X.dtype.element_ty).to(tl.float32)
            tl.store(P + (row*C + c)*NT + tile, tl.sum(pw, axis=1), row < T)
        tl.store(DX + row[:, None]*H + col[None, :], dx, mask)


@triton.jit(do_not_specialize=["N"])
def _addcmul_dw_reduce_kernel(P, DW, N, NT: tl.constexpr,
                             B: tl.constexpr, BT: tl.constexpr):
    rr = tl.arange(0, B)
    tt = tl.arange(0, BT)
    for start in range(tl.program_id(0)*B, N, tl.num_programs(0)*B):
        row = (start + rr).to(tl.int64)
        accum = tl.full((B, BT), 0, tl.float32)
        for tile in range(triton.cdiv(NT, BT)):
            part = tile*BT + tt
            p = tl.load(P + row[:, None]*NT + part[None, :],
                        (row[:, None] < N) & (part[None, :] < NT), other=0)
            accum += p
        tl.store(DW + row, tl.sum(accum, axis=1), row < N)


def _opts(arch35, backward=False):
    opts = dict(num_stages=1 if backward and not arch35 else 2,
                multibuffer=not backward or arch35, enable_fp_fusion=False)
    if arch35:
        opts['compile_mode'] = 'simd'
    return opts


def _config(x, w, arch35, backward=False):
    h, c = x.shape[-1], w.shape[-1]
    t, cores = x.numel()//h, get_vector_num(x.device)
    bc = triton.next_power_of_2(c)
    if backward:
        if arch35:
            br, bh = 16, min(256, triton.next_power_of_2(h))
        else:
            # Bound the backward tile for the 192 KiB UB on A2/A3.
            bh = min(4096 if c <= 4 else 1024, triton.next_power_of_2(h))
            budget = 8192 if c <= 4 else 4096
            br = min(8, budget//bh)
    else:
        bh = min(256 if not arch35 and bc == 16 else 512,
                 triton.next_power_of_2(h))
        # Preserve each architecture's measured schedule, including A5.
        budget = (8192 if bc == 16 else 32768) if arch35 else (
            4096 if bc == 16 else 8192)
        br = max(1, min(16, budget//(bh*bc)))
        br = 2**(br.bit_length()-1)
    return t, h, c, cores, br, bh, bc


def forward(x, w, r, round_product=True):
    """Internal fast path: use triton_addcmul() to validate the inputs."""
    arch35 = is_arch35()
    t, h, c, cores, br, bh, bc = _config(x, w, arch35)
    y = torch.empty_like(r)
    grid = min(cores, triton.cdiv(t, br)*triton.cdiv(h, bh))
    _addcmul_fwd_kernel[(grid,)](x, w, r, y, t, h, c, br, bh, bc,
                               round_product, **_opts(arch35))
    return y


def backward(x, w, g):
    """Internal contiguous fast path; R's gradient aliases G."""
    arch35 = is_arch35()
    t, h, c, cores, br, bh, _ = _config(x, w, arch35, backward=True)
    nt = triton.cdiv(h, bh)
    dx, dw = torch.empty_like(x), torch.empty_like(w)
    # A2/A3's common model H fits one tile, allowing a direct dW store.
    # Retain A5's FP32 partial sums and separate reduction for every H.
    needs_reduce = arch35 or nt > 1
    partial = (torch.empty((w.numel(), nt), device=x.device, dtype=torch.float32)
               if needs_reduce else dw)
    grid = min((2 if arch35 else 1)*cores, triton.cdiv(t, br)*nt)
    opts = _opts(arch35, backward=True)
    _addcmul_bwd_kernel[(grid,)](x, w, g, dx, partial, t, h, c, br, bh,
                               arch35, **opts)
    if needs_reduce:
        block, max_tiles = (256, 64) if arch35 else (128, 32)
        grid_reduce = min(cores, triton.cdiv(w.numel(), block))
        _addcmul_dw_reduce_kernel[(grid_reduce,)](
            partial, dw, w.numel(), nt, block,
            min(max_tiles, triton.next_power_of_2(nt)), **opts)
    return dx, dw, g


def _reference(x, w, r, round_product):
    if round_product:
        return r + (x.unsqueeze(-2)*w.unsqueeze(-1)).flatten(-2)
    c, h = w.shape[-1], x.shape[-1]
    return torch.addcmul(r.unflatten(-1, (c, h)), x.unsqueeze(-2),
                        w.unsqueeze(-1)).flatten(-2)


class _Addcmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, r, round_product):
        ctx.save_for_backward(x, w)
        return forward(x, w, r, round_product)

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        if g.is_contiguous():
            dx, dw, dr = backward(x, w, g)
        else:
            gg = g.unflatten(-1, (w.shape[-1], x.shape[-1]))
            dx = (gg*w.unsqueeze(-1)).sum(-2)
            dw = (gg*x.unsqueeze(-2)).sum(-1)
            dr = g
        return dx, dw, dr, None


def triton_addcmul(hidden_states, injection_weights, hyper_input, *, round_product=True):
    """Return R + flatten(X[...,None,:] * W[...,:,None]).

    Shapes: X=[...,H], W=[...,C], R=[...,C*H]. Fast path supports arbitrary
    identical leading dimensions, positive H, 1<=C<=16, contiguous NPU
    BF16/FP16, and first-order gradients. Other inputs use native operations.
    Non-contiguous dY uses a native backward without an extra contiguous copy.
    Default ADDCMUL_TRITON=auto uses native operations below 32M output elements
    to avoid launch overhead. 1 forces supported fast paths; 0 disables them.
    Default rounding matches the original Mul+Add and the supplied reference.
    Set round_product=False to select torch.addcmul-style forward rounding.
    """
    x, w, r = hidden_states, injection_weights, hyper_input
    if any(z.ndim < 1 for z in (x, w, r)):
        raise ValueError('X, W and R must have a last dimension H, C and C*H')
    if r.shape[-1] != x.shape[-1]*w.shape[-1]:
        raise ValueError('R last dimension must equal hidden size * group size')
    supported = (x.device.type == 'npu' and x.device == w.device == r.device
                 and x.dtype == w.dtype == r.dtype
                 and x.dtype in (torch.bfloat16, torch.float16)
                 and x.shape[:-1] == w.shape[:-1] == r.shape[:-1]
                 and x.numel() > 0 and 1 <= w.shape[-1] <= 16
                 and x.is_contiguous() and w.is_contiguous() and r.is_contiguous())
    mode = os.getenv('ADDCMUL_TRITON', 'auto')
    small = mode == 'auto' and r.numel() < 32*1024*1024
    if mode == '0' or not supported or small:
        return _reference(x, w, r, round_product)
    return _Addcmul.apply(x, w, r, bool(round_product))
