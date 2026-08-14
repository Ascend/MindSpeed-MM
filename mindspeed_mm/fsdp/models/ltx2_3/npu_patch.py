from __future__ import annotations

import types

import torch
import torch_npu
from einops import rearrange

from mindspeed_mm.fsdp.ops.flash_attn.flash_attn import flash_attention_forward
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE


class NpuFlashAttention:
    """NPU Flash Attention，委托给框架统一的 flash_attention_forward。

    与其他后端一致：q/k/v 输入 (B, T, H*D)，输出 (B, T, H*D)，内部 reshape 到 BNSD。
    支持 CP（上下文并行）/ Ulysses / skip-recompute。
    """

    label = "NpuFlashAttention"

    def __init__(self) -> None:
        self.config = types.SimpleNamespace(_attn_implementation="flash_attention_2")
        self.is_causal = False

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads

        # (B, T, H*D) -> (B, H, T, D)  [BNSD layout]
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
        q, k = q.to(v.dtype), k.to(v.dtype)

        attn_output, _ = flash_attention_forward(
            module=self,
            query=q,
            key=k,
            value=v,
            attention_mask=mask,
            scaling=dim_head ** -0.5,
            input_layout="BNSD",
        )

        return attn_output.reshape(b, -1, heads * dim_head)


class NpuRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def _npu_apply_split_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    if sin_freqs.shape != cos_freqs.shape:
        raise ValueError(
            f"apply_split_rotary_emb: sin_freqs.shape {tuple(sin_freqs.shape)} must equal "
            f"cos_freqs.shape {tuple(cos_freqs.shape)}."
        )
    needs_reshape = input_tensor.ndim != 4 and cos_freqs.ndim == 4
    if needs_reshape:
        b_freq = cos_freqs.shape[0]
        h = cos_freqs.shape[1]
        b_in = input_tensor.shape[0]
        if b_freq not in (1, b_in):
            raise ValueError(
                f"apply_split_rotary_emb: cos_freqs batch ({b_freq}) must be 1 "
                f"(broadcast) or equal input_tensor batch ({b_in})."
            )
        input_tensor = input_tensor.unflatten(-1, (h, -1)).transpose(1, 2)

    if IS_NPU_AVAILABLE:
        # NPU fused rotary mul: npu_rotary_mul(x, cos, sin) = x*cos + rotate_half(x)*sin
        # cos/sin are half-dim here, duplicate to full-dim for the kernel
        cos_full = torch.cat([cos_freqs, cos_freqs], dim=-1)
        sin_full = torch.cat([sin_freqs, sin_freqs], dim=-1)
        output = torch_npu.npu_rotary_mul(input_tensor, cos_full, sin_full)
    else:
        split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
        first_half_input = split_input[..., :1, :]
        second_half_input = split_input[..., 1:, :]

        output = split_input * cos_freqs.unsqueeze(-2)
        first_half_output = output[..., :1, :]
        second_half_output = output[..., 1:, :]

        first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)
        second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)

        output = rearrange(output, "... d r -> ... (d r)")

    if needs_reshape:
        output = output.transpose(1, 2).flatten(-2)

    return output


def _npu_rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """NPU 版 rms_norm，替换 ltx_core.utils.rms_norm。

    torch_npu.npu_rms_norm 要求 weight 必须显式传入，weight 为 None 时构造默认全 1。
    """
    if weight is None:
        weight = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
    return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]


def apply_ltx2_3_npu_patch() -> None:
    if not IS_NPU_AVAILABLE:
        return

    from ltx_core.model.transformer import attention as ltx_attention
    from ltx_core.model.transformer import rope as ltx_rope
    from ltx_core.model.transformer import transformer as ltx_transformer
    from ltx_core import utils as ltx_utils

    ltx_attention._select_primary_attention = lambda: NpuFlashAttention()
    ltx_attention._select_masked_attention = lambda: NpuFlashAttention()
    ltx_attention.automatic_attention.cache_clear()
    ltx_attention.automatic_masked_attention.cache_clear()
    npu_attention_ops = ltx_attention.AttentionOps(
        attention_function=NpuFlashAttention(),
        masked_attention_function=NpuFlashAttention(),
    )
    npu_transformer_ops = ltx_transformer.TransformerOpsConfig(attention_ops=npu_attention_ops)
    ltx_transformer.DEFAULT_TRANSFORMER_OPS = npu_transformer_ops

    # model.py / model_configurator.py import DEFAULT_TRANSFORMER_OPS by value,
    # so update their module globals as well.
    from ltx_core.model.transformer import model as ltx_model
    from ltx_core.model.transformer import model_configurator as ltx_model_configurator

    ltx_model.DEFAULT_TRANSFORMER_OPS = npu_transformer_ops
    ltx_model_configurator.DEFAULT_TRANSFORMER_OPS = npu_transformer_ops
    ltx_model.LTXModel.__init__.__kwdefaults__["ops"] = npu_transformer_ops
    ltx_model_configurator.LTXModelConfigurator.from_config.__func__.__defaults__ = (npu_transformer_ops,)
    ltx_model_configurator.LTXVideoOnlyModelConfigurator.from_config.__func__.__defaults__ = (npu_transformer_ops,)

    # q/k_norm：patch Attention.__init__，把 q_norm/k_norm 替换为 NpuRMSNorm
    _orig_init = ltx_attention.Attention.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        inner_dim = self.heads * self.dim_head
        norm_eps = kwargs.get("norm_eps", 1e-6)
        self.q_norm = NpuRMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = NpuRMSNorm(inner_dim, eps=norm_eps)

    ltx_attention.Attention.__init__ = _patched_init

    # RoPE：patch apply_split_rotary_emb
    ltx_rope.apply_split_rotary_emb = _npu_apply_split_rotary_emb

    # utils.rms_norm
    ltx_utils.rms_norm = _npu_rms_norm
