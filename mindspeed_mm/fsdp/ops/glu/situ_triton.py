"""SituAndMul 融合 Triton 算子（前向 + 反向合一接口）。

来源：
- 前向 kernel：op_22_SituAndMul/situ_and_mul_generated.py（geomean 4.21x）
- 反向 kernel：op_23_SituAndMulBackward/situ_and_mul_backward_generated.py（geomean 4.26x）

对外接口：
- SituAndMulTriton(beta=1.0, linear_beta=None)：nn.Module，可直接替换
  MindSpeed-MM modeling_kimi_linear.py 中的 SituAndMul（ACT2FN["situ"]）。
- situ_and_mul(x, beta=1.0, linear_beta=None)：函数式接口。

训练路径通过 torch.autograd.Function 接入：前向只保存 x（t/s 在反向 kernel
内单 vexp 重算，无需保存任何中间张量）。推理路径（no_grad/inference_mode）
自动跳过反向图构建。
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


_NUM_CORES = None


def _get_num_cores():
    global _NUM_CORES
    if _NUM_CORES is None:
        try:
            device = torch.npu.current_device()
            props = triton.runtime.driver.active.utils.get_device_properties(device)
            num_cores = props.get("num_vectorcore", -1)
            if num_cores <= 0:
                num_cores = props.get("num_aicore", 48)
            _NUM_CORES = max(int(num_cores), 1)
        except Exception:
            _NUM_CORES = 48
    return _NUM_CORES


def _select_block(d, cap=4096):
    if d <= 64:
        return 64
    if d <= 128:
        return 128
    if d <= 256:
        return 256
    if d <= 512:
        return 512
    if d <= 1024:
        return 1024
    if d <= 2048:
        return 2048
    return cap


@triton.jit
def _situ_and_mul_fwd_kernel(
    x_ptr, y_ptr,
    total_rows,
    d: tl.constexpr, beta: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr, linear_beta: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_programs = tl.num_programs(0).to(tl.int32)

    rows_per_prog = (total_rows + num_programs - 1) // num_programs
    row_start = pid * rows_per_prog
    row_end = row_start + rows_per_prog
    if row_end > total_rows:
        row_end = total_rows

    for row in range(row_start, row_end):
        x_row = x_ptr + row * 2 * d
        y_row = y_ptr + row * d
        for col in range(0, d, BLOCK):
            cols = (col + tl.arange(0, BLOCK)).to(tl.int32)
            mask = cols < d

            gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(x_row + d + cols, mask=mask, other=0.0).to(tl.float32)

            if beta == 1.0:
                e = tl.exp(-gate)
                s = 1.0 / (1.0 + e)
                e2 = e * e
                t = (1.0 - e2) / (1.0 + e2)
            else:
                s = tl.sigmoid(gate)
                t = 2.0 * tl.sigmoid(2.0 * gate / beta) - 1.0
            situ_a = beta * t * s
            if HAS_LINEAR_BETA:
                up = linear_beta * (2.0 * tl.sigmoid(2.0 * up / linear_beta) - 1.0)
            out = situ_a * up

            tl.store(y_row + cols, out.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _situ_and_mul_bwd_kernel(
    x_ptr, dout_ptr, dx_ptr,
    total_rows,
    d: tl.constexpr, beta: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr, linear_beta,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_programs = tl.num_programs(0).to(tl.int32)

    rows_per_prog = (total_rows + num_programs - 1) // num_programs
    row_start = pid * rows_per_prog
    row_end = row_start + rows_per_prog
    if row_end > total_rows:
        row_end = total_rows

    for row in range(row_start, row_end):
        x_row = x_ptr + row * 2 * d
        dout_row = dout_ptr + row * d
        dx_row = dx_ptr + row * 2 * d
        for col in range(0, d, BLOCK):
            cols = (col + tl.arange(0, BLOCK)).to(tl.int32)
            mask = cols < d

            gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(x_row + d + cols, mask=mask, other=0.0).to(tl.float32)
            gout = tl.load(dout_row + cols, mask=mask, other=0.0).to(tl.float32)

            if beta == 1.0:
                e = tl.exp(-gate)
                s = 1.0 / (1.0 + e)
                e2 = e * e
                t = (1.0 - e2) / (1.0 + e2)
                ts = t * s
                a = ts
                da = s + ts * (1.0 - s - t)
            else:
                t = 2.0 * tl.sigmoid(2.0 * gate / beta) - 1.0
                s = tl.sigmoid(gate)
                a = beta * t * s
                da = (1.0 - t * t) * s + beta * t * s * (1.0 - s)

            v = up
            dv = 1.0
            if HAS_LINEAR_BETA:
                tu = 2.0 * tl.sigmoid(2.0 * up / linear_beta) - 1.0
                v = linear_beta * tu
                dv = 1.0 - tu * tu

            dx_gate = gout * v * da
            dx_up = gout * a * dv

            tl.store(dx_row + cols, dx_gate.to(dx_ptr.dtype.element_ty), mask=mask)
            tl.store(dx_row + d + cols, dx_up.to(dx_ptr.dtype.element_ty), mask=mask)


def _launch_fwd(x, beta, linear_beta):
    d = x.shape[-1] // 2
    x = x.contiguous()
    out_shape = x.shape[:-1] + (d,)
    x2 = x.view(-1, 2 * d)
    total_rows = x2.shape[0]
    y2 = torch.empty((total_rows, d), dtype=x.dtype, device=x.device)

    num_cores = _get_num_cores()
    num_programs = total_rows if total_rows < num_cores else num_cores
    _situ_and_mul_fwd_kernel[(num_programs,)](
        x2, y2,
        total_rows,
        d, float(beta),
        linear_beta is not None,
        float(linear_beta) if linear_beta is not None else 1.0,
        BLOCK=_select_block(d),
        multibuffer=True,
    )
    return y2.view(out_shape)


def _launch_bwd(x, dout, beta, linear_beta):
    d = x.shape[-1] // 2
    x = x.contiguous()
    dout = dout.contiguous()
    out_shape = x.shape
    x2 = x.view(-1, 2 * d)
    dout2 = dout.reshape(-1, d)
    total_rows = x2.shape[0]
    dx2 = torch.empty((total_rows, 2 * d), dtype=x.dtype, device=x.device)

    num_cores = _get_num_cores()
    num_programs = total_rows if total_rows < num_cores else num_cores
    _situ_and_mul_bwd_kernel[(num_programs,)](
        x2, dout2, dx2,
        total_rows,
        d, float(beta),
        linear_beta is not None,
        float(linear_beta) if linear_beta is not None else 1.0,
        BLOCK=_select_block(d),
        multibuffer=True,
    )
    return dx2.view(out_shape)


class _SituAndMulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, linear_beta):
        y = _launch_fwd(x, beta, linear_beta)
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.beta = beta
            ctx.linear_beta = linear_beta
        return y

    @staticmethod
    def backward(ctx, dout):
        (x,) = ctx.saved_tensors
        dx = _launch_bwd(x, dout, ctx.beta, ctx.linear_beta)
        return dx, None, None


def situ_and_mul(x: torch.Tensor, beta: float = 1.0, linear_beta=None) -> torch.Tensor:
    """函数式接口：y = beta * tanh(gate/beta) * sigmoid(gate) * v(up)。"""
    return _SituAndMulFunc.apply(x, beta, linear_beta)


class SituAndMulTriton(nn.Module):
    """即插即用替换 MindSpeed-MM modeling_kimi_linear.py 中的 SituAndMul。"""

    def __init__(self, beta: float = 1.0, linear_beta=None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _SituAndMulFunc.apply(x, self.beta, self.linear_beta)
