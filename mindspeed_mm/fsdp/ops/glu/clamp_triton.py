"""Fused input gradient for BF16 clamp with per-column bounds on Ascend."""

from functools import lru_cache

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _clamp_dx(X, DY, LO, HI, DX, ROWS: tl.constexpr, WIDTH: tl.constexpr,
              BLOCK: tl.constexpr, FLAT: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    if FLAT:
        col = lane % WIDTH
        lower = tl.load(LO + col).to(tl.float32)
        upper = tl.load(HI + col).to(tl.float32)
        count = ROWS * WIDTH // BLOCK
    else:
        lower = tl.load(LO + lane, lane < WIDTH, other=0).to(tl.float32)
        upper = tl.load(HI + lane, lane < WIDTH, other=0).to(tl.float32)
        count = ROWS
    per_core = tl.cdiv(count, tl.num_programs(0))
    start = tl.program_id(0) * per_core
    for tile in range(start, tl.minimum(start + per_core, count)):
        if FLAT:
            offset = tile * BLOCK + lane
            valid = lane < BLOCK
        else:
            offset = tile * WIDTH + lane
            valid = lane < WIDTH
        x = tl.load(X + offset, valid, other=0).to(tl.float32)
        dy = tl.load(DY + offset, valid, other=0)
        dx = tl.where((x >= lower) & (x <= upper), dy, 0)
        tl.store(DX + offset, dx, valid)


@lru_cache(maxsize=None)
def _vector_cores(device_index):
    return triton.runtime.driver.active.utils.get_device_properties(device_index)["num_vectorcore"]


class ClampWithFusedBackward(torch.autograd.Function):
    """First-order clamp for contiguous ND tensors, with constant column bounds."""

    @staticmethod
    def forward(ctx, x, lower, upper):
        ctx.save_for_backward(x, lower, upper)
        return torch.clamp(x, min=lower, max=upper)

    @staticmethod
    def backward(ctx, dy):
        x, lower, upper = ctx.saved_tensors
        if torch.is_grad_enabled():
            return torch.where((x >= lower) & (x <= upper), dy, 0), None, None
        rows, width = x.shape
        with torch.npu.device(x.device):
            if not dy.is_contiguous() or torch_npu.get_npu_format(dy) != 2:
                return torch.where((x >= lower) & (x <= upper), dy, 0), None, None
            flat = width == 4096 and rows % 2 == 0
            block = 8192 if flat else triton.next_power_of_2(width)
            count = rows // 2 if flat else rows
            dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            _clamp_dx[(min(_vector_cores(x.device.index), count),)](
                x, dy, lower, upper, dx, rows, width, block, flat
            )
        return dx, None, None
