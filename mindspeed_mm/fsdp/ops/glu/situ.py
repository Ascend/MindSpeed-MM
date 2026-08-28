"""
situ_glu.py
SituGLU 完整实现：autograd Function + eager 小算子 + 对外接口，单文件版本。
"""

import torch
import torch.nn as nn

from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    import torch_npu

def situ_glu_eager(
    x: torch.Tensor,
    *,
    dim: int = -1,
    beta: float = 1.0,
    linear_beta: float = 0.0,
    activate_left: bool = True,
) -> torch.Tensor:
    d = x.shape[dim] // 2
    if dim == -1:
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
    else:
        # 通用 dim 切片
        slices_gate = [slice(None)] * x.ndim
        slices_gate[dim] = slice(d)
        slices_up = [slice(None)] * x.ndim
        slices_up[dim] = slice(d, 2 * d)
        gate = x[tuple(slices_gate)].to(torch.float32)
        up = x[tuple(slices_up)].to(torch.float32)

    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)

    if linear_beta not in (0.0, None):
        up = linear_beta * torch.tanh(up / linear_beta)

    result = situ_a * up

    return result.to(x.dtype)


class SituGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        dim: int = -1,
        beta: float = 1.0,
        linear_beta: float = 0.0,
        activate_left: bool = True,
    ) -> torch.Tensor:
        # 非 Tensor 参数存 ctx（不能进 saved_tensors）
        ctx.dim = dim
        ctx.beta = beta
        ctx.linear_beta = linear_beta
        ctx.activate_left = activate_left
        ctx.save_for_backward(x)

        out = torch.ops.cann_ops_nn.situ_glu(
            x,
            dim=dim,
            beta=beta,
            linear_beta=linear_beta,
            activate_left=activate_left,
        )
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple:
        x, = ctx.saved_tensors

        grad_x = torch.ops.cann_ops_nn.situ_glu_grad(
            grad_output,
            x,
            dim=ctx.dim,
            beta=ctx.beta,
            linear_beta=ctx.linear_beta,
            activate_left=ctx.activate_left,
        )

        # 对应 forward 入参: (x, dim, beta, linear_beta, activate_left)
        return grad_x, None, None, None, None


def fused_situ_glu(
    x: torch.Tensor,
    *,
    dim: int = -1,
    beta: float = 1.0,
    linear_beta: float = 0.0,
    activate_left: bool = True,
    situ_glu_implementation: str = "eager"
) -> torch.Tensor:
    if not IS_NPU_AVAILABLE or situ_glu_implementation == "eager":
        return situ_glu_eager(x, dim=dim, beta=beta, linear_beta=linear_beta, activate_left=activate_left)
    elif situ_glu_implementation == "ascendc":
        import cann_ops_nn.ops
        return SituGLUFunction.apply(x, dim, beta, linear_beta, activate_left)
    elif situ_glu_implementation == "triton":
        # triton kernel 只支持最后一维切分且 gate 在左的布局
        if dim != -1 or not activate_left:
            raise ValueError(
                "Triton situ_glu only supports dim=-1 and activate_left=True."
            )
        from mindspeed_mm.fsdp.ops.glu.situ_triton import situ_and_mul
        # eager/ascendc 语义中 linear_beta 为 0.0 表示不启用，triton kernel 以 None 表示，
        # 直接传 0.0 会在 kernel 内产生除零
        linear_beta = linear_beta if linear_beta not in (0.0, None) else None
        return situ_and_mul(x, beta=beta, linear_beta=linear_beta)
    else:
        raise ValueError(
            f"Unsupported situ_glu_implementation: {situ_glu_implementation}. "
            "Expected 'eager', 'triton' or 'ascendc'."
        )
