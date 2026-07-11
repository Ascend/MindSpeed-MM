import copy
import math
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio
from diffusers.models.activations import get_activation
from diffusers.models.attention import (
    GEGLU,
    GELU,
    AdaLayerNorm,
    AdaLayerNormZero,
    ApproximateGELU,
)
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import pack, rearrange, repeat
from flash_attn import flash_attn_varlen_func
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from .cosy24k_vocoder import Cosy24kVocoder

logger = logging.get_logger(__name__)


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    if not channels % 2 == 0:
        raise AssertionError("channels % 2 must be 0")
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def get_sequence_mask(inputs, inputs_length):
    if inputs.dim() == 3:
        bsz, tgt_len, _ = inputs.size()
    else:
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)
    sequence_mask = torch.arange(0, tgt_len).to(inputs.device)
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(bsz, tgt_len, 1)
    unpacking_index = torch.cumsum(sequence_mask.to(torch.int64).view(-1), dim=0) - 1  # 转成下标
    return sequence_mask, unpacking_index


def unpack_hidden_states(hidden_states, lengths):
    bsz = lengths.shape[0]
    sequence_mask, unpacking_index = get_sequence_mask(hidden_states, lengths)
    hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
        bsz, torch.max(lengths), hidden_states.shape[-1]
    )
    hidden_states = torch.where(
        sequence_mask, hidden_states, 0
    )  # 3d (bsz, max_input_len, d)
    return hidden_states


def uniform_init(*shape):
    t = torch.zeros(shape)
    nn.init.kaiming_uniform_(t)
    return t


def cdist(x, y):
    x2 = torch.sum(x ** 2, dim=-1, keepdims=True)  # (b, 1)
    y2 = torch.sum(y ** 2, dim=-1).reshape(1, -1)  # (1, c)
    xy = torch.einsum('bd,cd->bc', x, y) * -2
    return (x2 + y2 + xy).clamp(min=0).sqrt()  # (b, c)


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise AssertionError("mask.dtype need equals torch.bool")
    if dtype not in [torch.float32, torch.bfloat16, torch.float16]:
        raise AssertionError("dtype should in torch.float32, torch.bfloat16, torch.float16")
    mask = mask.to(dtype)
    mask = (1.0 - mask) * torch.finfo(dtype).min
    return mask


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    pos_idx = torch.arange(size, device=device)
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode='trunc') + 1) * chunk_size
    ret = pos_idx.unsqueeze(0) < block_value.unsqueeze(1)
    return ret


def add_optional_chunk_mask(xs: torch.Tensor,
                            masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True):
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            chunk_size = torch.randint(1, max_len, (1,)).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks,
                                                    (1,)).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


class EuclideanCodebook(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            init_std=0.02,
    ):
        super().__init__()
        self.init_std = init_std
        self.dim = dim
        self.codebook_size = codebook_size

        embed = uniform_init(codebook_size, dim).to(torch.float32)
        self.cluster_size = nn.Parameter(torch.ones(codebook_size))
        self.embed_avg = nn.Parameter(embed.clone())
        self.embed = nn.Parameter(embed)
        del embed

    @autocast(enabled=True, dtype=torch.float32)
    @torch.no_grad()
    def forward(self, x):
        if not (len(x.shape) == 2):
            raise AssertionError("len(x.shape) needs equal to 2")
        if not (x.dtype == torch.float32):
            raise AssertionError("x.dtype needs to be equal with torch.float32")
        embed = self.embed.detach().to(x.device)
        dist = -cdist(x, embed)  # dist((bs*sl, d), (c, d)) --> (bs*sl, c)
        embed_ind = dist.argmax(dim=-1)
        quantize = embed[embed_ind]  # (bs*sl, d)
        return quantize, embed_ind, dist


class VectorQuantize(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.codebook = EuclideanCodebook(dim=config.dim, codebook_size=config.codebook_size)

    def forward(self, x, input_length):
        batch_size, seq_len, _ = x.shape
        mask, unpacking_index = get_sequence_mask(x, input_length)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.masked_select(x, mask).reshape(-1, self.config.dim)  # (bs*sl?, d)
        quantize, embed_ind, _ = self.codebook(x)
        quantize = torch.index_select(quantize, 0, unpacking_index).view(batch_size, seq_len, self.config.dim)
        quantize = torch.where(mask, quantize, 0)
        embed_ind = torch.index_select(embed_ind.reshape(-1, 1), 0, unpacking_index).view(batch_size, seq_len, 1)
        embed_ind = torch.where(mask, embed_ind, -1).squeeze()
        return quantize, embed_ind

    def get_output_from_indices(self, indices):
        indices = indices.to(self.codebook.embed.device)
        return self.codebook.embed[indices]


class SnakeBeta(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            alpha=1.0,
            alpha_trainable=True,
            alpha_logscale=True,
    ):
        super().__init__()
        self.in_features = (
            out_features if isinstance(out_features, list) else [out_features]
        )
        self.proj = LoRACompatibleLinear(in_features, out_features)

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta

        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(x * alpha), 2
        )

        return x


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            dropout: float = 0.0,
            activation_fn: str = "geglu",
            final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)
        elif activation_fn == "snakebeta":
            act_fn = SnakeBeta(dim, inner_dim)
        else:
            raise ValueError(f"activation_fn not in [gelu,gelu-approximate,geglu,geglu-approximate,snakebeta]")

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(LoRACompatibleLinear(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
            use_omni_attn: bool = False,
    ):
        super().__init__()

        self.use_omni_attn = use_omni_attn
        self.dim = dim

        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (
                                               num_embeds_ada_norm is not None
                                       ) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (
                                          num_embeds_ada_norm is not None
                                  ) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        if self.use_omni_attn:
            if only_cross_attention:
                raise NotImplementedError
            print(
                "Use OmniWhisperAttention with flash attention. Dropout is ignored."
            )
            self.attn1 = OmniWhisperAttention(
                embed_dim=dim, num_heads=num_attention_heads, causal=False
            )
        else:
            self.attn1 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=(
                    cross_attention_dim if only_cross_attention else None
                ),
                upcast_attention=upcast_attention,
            )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=(
                    cross_attention_dim if not double_self_attention else None
                ),
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
    ):

        bsz, tgt_len, d_model = hidden_states.shape

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )

        if self.use_omni_attn:
            seq_len = attention_mask[:, 0, :].float().long().sum(dim=1)
            var_len_attention_mask, unpacking_index = get_sequence_mask(
                norm_hidden_states, seq_len
            )
            norm_hidden_states = torch.masked_select(
                norm_hidden_states, var_len_attention_mask
            )
            norm_hidden_states = norm_hidden_states.view(torch.sum(seq_len), self.dim)
            attn_output = self.attn1(norm_hidden_states, seq_len)
            # unpacking
            attn_output = torch.index_select(attn_output, 0, unpacking_index).view(
                bsz, tgt_len, d_model
            )
            attn_output = torch.where(var_len_attention_mask, attn_output, 0)
        else:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=(
                    encoder_hidden_states if self.only_cross_attention else None
                ),
                attention_mask=(
                    encoder_attention_mask
                    if self.only_cross_attention
                    else attention_mask
                ),
                **cross_attention_kwargs,
            )

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(hid_slice)
                    for hid_slice in norm_hidden_states.chunk(
                    num_chunks, dim=self._chunk_dim
                )
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class CausalBlock1D(Block1D):
    def __init__(self, dim: int, dim_out: int):
        super(CausalBlock1D, self).__init__(dim, dim_out)
        self.block = torch.nn.Sequential(
            CausalConv1d(dim, dim_out, 3),
            Transpose(1, 2),
            nn.LayerNorm(dim_out),
            Transpose(1, 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        output = self.block(x * mask)
        return output * mask


class CausalResnetBlock1D(ResnetBlock1D):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super(CausalResnetBlock1D, self).__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding=0, dilation=dilation,
                                           groups=groups, bias=bias,
                                           padding_mode=padding_mode,
                                           device=device, dtype=dtype)
        if not stride == 1:
            raise AssertionError("stride needs to be 1")
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.causal_padding)
        x = super(CausalConv1d, self).forward(x)
        return x


class BASECFM(torch.nn.Module, ABC):
    def __init__(
            self,
            n_feats,
            cfm_params,
            n_spks=1,
            spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction="sum") / (
                torch.sum(mask) * u.shape[1]
        )
        return loss, y


class ConditionalDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            causal=False,
            channels=(256, 256),
            dropout=0.05,
            attention_head_dim=64,
            n_blocks=1,
            num_mid_blocks=2,
            num_heads=4,
            act_fn="snake",
            gradient_checkpointing=False,
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.static_chunk_size = 2 * 25 * 2  # 2*input_frame_rate*token_mel_ratio
        self.gradient_checkpointing = gradient_checkpointing

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel,
                                         time_emb_dim=time_embed_dim) if self.causal else \
                ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else
                CausalConv1d(output_channel, output_channel, 3) if self.causal else nn.Conv1d(output_channel,
                                                                                              output_channel, 3,
                                                                                              padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for _ in range(num_mid_blocks):
            input_channel = channels[-1]
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel,
                                         time_emb_dim=time_embed_dim) if self.causal else \
                ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = CausalResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            ) if self.causal else ResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else CausalConv1d(output_channel, output_channel, 3) if self.causal else nn.Conv1d(output_channel,
                                                                                                   output_channel, 3,
                                                                                                   padding=1)
            )
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
        self.final_block = CausalBlock1D(channels[-1], channels[-1]) if self.causal else Block1D(channels[-1],
                                                                                                 channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        t = self.time_embeddings(t)
        t = t.to(x.dtype)
        t = self.time_mlp(t)
        x = pack([x, mu], "b * t")[0]
        mask = mask.to(x.dtype)
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(transformer_block),
                        x,
                        attn_mask,
                        t,
                    )
                else:
                    x = transformer_block(
                        hidden_states=x,
                        attention_mask=attn_mask,
                        timestep=t,
                    )
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            x_in, attn_mask_in, t_in = inputs
                            return module(x=x_in, attn_mask=attn_mask_in, t=t_in)

                        return custom_forward

                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(transformer_block),
                        x=x,
                        attn_mask=attn_mask,
                        t=t,
                        use_reentrant=False
                    )
                else:
                    x = transformer_block(
                        hidden_states=x,
                        attention_mask=attn_mask,
                        timestep=t,
                    )
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(transformer_block),
                        x,
                        attn_mask,
                        t,
                    )
                else:
                    x = transformer_block(
                        hidden_states=x,
                        attention_mask=attn_mask,
                        timestep=t,
                    )
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate

    @torch.inference_mode()
    def forward(self, estimator, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(estimator, z, t_span=t_span.to(mu.dtype), mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, estimator, x, t_span, mu, mask, spks, cond):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = estimator(x, mask, mu, t, spks, cond)
            # Classifier-Free Guidance inference introduced in VoiceBox
            if self.inference_cfg_rate > 0:
                cfg_dphi_dt = estimator(
                    x, mask,
                    torch.zeros_like(mu), t,
                    torch.zeros_like(spks) if spks is not None else None,
                    cond=cond
                )
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt -
                           self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, estimator, x1, mask, mu, spks=None, cond=None):
        org_dtype = x1.dtype

        b, _, t = mu.shape
        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            if spks is not None:
                spks = spks * cfg_mask.view(-1, 1)
            if cond is not None:
                cond = cond * cfg_mask.view(-1, 1, 1)

        pred = estimator(y, mask, mu, t.squeeze(), spks, cond)
        pred = pred.float()
        u = u.float()
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        loss = loss.to(org_dtype)
        return loss, y


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        if not self.dim % 2 == 0:
            raise AssertionError("SinusoidalPosEmb requires dim to be even")

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class TimestepEmbedding(nn.Module):
    def __init__(
            self,
            in_channels: int,
            time_embed_dim: int,
            act_fn: str = "silu",
            out_dim: int = None,
            post_act_fn: Optional[str] = None,
            cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Upsample1D(nn.Module):
    def __init__(
            self,
            channels,
            use_conv=False,
            use_conv_transpose=True,
            out_channels=None,
            name="conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        if not inputs.shape[1] == self.channels:
            raise AssertionError("inputs.shape[1] needs to the same with self.channels")
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class OmniWhisperAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.causal = causal

    def forward(self, hidden_states: torch.Tensor, seq_len: torch.Tensor):
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)

        cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(torch.int32)
        max_seqlen = torch.max(seq_len).to(torch.int32).detach()
        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_len, cu_len, max_seqlen,
                                             max_seqlen, causal=self.causal)  # (bsz * qlen, nheads, headdim)
        attn_output = attn_output.reshape(bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class OmniWhisperTransformerLayer(nn.Module):
    def __init__(
            self,
            act,
            d_model,
            encoder_attention_heads,
            encoder_ffn_dim,
            causal,
            ln_type="LayerNorm",
    ):
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = OmniWhisperAttention(
            self.embed_dim, encoder_attention_heads, causal
        )

        if ln_type == "LayerNorm":
            self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        elif ln_type == "RMSNorm":
            self.self_attn_layer_norm = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unknown ln_type: {ln_type}")

        self.activation_fn = act
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)

        if ln_type == "LayerNorm":
            self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        elif ln_type == "RMSNorm":
            self.final_layer_norm = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unknown ln_type: {ln_type}")

    def forward(
            self, hidden_states: torch.Tensor, seq_len: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, seq_len)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if (
                hidden_states.dtype == torch.float16
                or hidden_states.dtype == torch.bfloat16
        ) and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        return hidden_states


class LongcatNextAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_source_positions = (
                                            config.max_audio_seconds * config.sampling_rate // config.hop_length) // config.stride_size
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, kernel_size=config.kernel_size, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=config.kernel_size,
                               stride=config.stride_size, padding=1)
        self.register_buffer("positional_embedding", sinusoids(self.max_source_positions, config.d_model))  # 1500 * d

        self.layers = nn.ModuleList([OmniWhisperTransformerLayer(
            ACT2FN[config.activation_function],
            config.d_model,
            config.encoder_attention_heads,
            config.encoder_ffn_dim,
            False) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
            self,
            input_features,
            output_length,
    ):
        input_features = input_features.to(self.conv1.weight.dtype)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))  # (bs, channels, frames)
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))  # (bs, channels, frames // 2)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # (bs, frams, channels)
        bsz, tgt_len, _ = inputs_embeds.size()
        if tgt_len < self.positional_embedding.shape[0]:
            current_positional_embedding = self.positional_embedding[:tgt_len]
        else:
            current_positional_embedding = self.positional_embedding
        hidden_states = (inputs_embeds.to(torch.float32) + current_positional_embedding).to(inputs_embeds.dtype)

        # packing hidden states
        attention_mask, unpacking_index = get_sequence_mask(hidden_states, output_length)
        hidden_states = torch.masked_select(hidden_states, attention_mask).view(torch.sum(output_length),
                                                                                self.config.d_model)

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states, output_length)
        hidden_states = self.layer_norm(hidden_states)
        # unpacking
        hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(bsz, tgt_len, self.config.d_model)
        hidden_states = torch.where(attention_mask, hidden_states, 0)
        return hidden_states


class CasualConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.GroupNorm(1, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, hidden_states, input_length, output_dim=None):
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        bsz = input_length.shape[0]

        if output_dim is None:
            output_dim = hidden_states.dim()
        if hidden_states.dim() <= 2:  # unpack sequence to 3d
            sequence_mask, unpacking_index = get_sequence_mask(hidden_states, input_length)
            hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(bsz, torch.max(input_length),
                                                                                       self.in_channels)
            hidden_states = torch.where(sequence_mask, hidden_states, 0)  # 3d (bsz, max_input_len, d)

        hidden_states = hidden_states.transpose(2, 1)  # (N, L, C) -> (N, C, L)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(2, 1)  # (N, C, L) -> (N, L, C)

        casual_padding_right = max(0, kernel_size - stride)
        hidden_states = hidden_states[:, :hidden_states.shape[1] - casual_padding_right,
                        :]
        output_length = (input_length - 1) * stride + kernel_size - casual_padding_right
        sequence_mask, _ = get_sequence_mask(hidden_states, output_length)
        if output_dim <= 2:
            hidden_states = torch.masked_select(hidden_states, sequence_mask).view(-1, self.out_channels)
        else:
            hidden_states = torch.where(sequence_mask, hidden_states, 0)
            hidden_states = hidden_states[:, :torch.max(output_length), :]
        return hidden_states, output_length


class MelSpecRefineNet(nn.Module):
    def __init__(self, encoder_config, vocoder_config):
        super().__init__()
        self.encoder_config = encoder_config
        self.vocoder_config = vocoder_config

        layers = nn.ModuleList([])
        in_channels = self.vocoder_config.num_mel_bins
        for i, out_channels in enumerate(self.vocoder_config.channels[:-1]):
            module = nn.Conv1d(in_channels, out_channels, 5, 1, 2)  # cosyvoice kernel=3, stride=1, pad=1
            in_channels = out_channels
            norm = nn.GroupNorm(1, out_channels)
            act = nn.Mish()
            layers.extend([module, norm, act])
        layers.append(nn.Conv1d(in_channels, self.vocoder_config.num_mel_bins, 1, 1))  # projector
        self.layers = nn.Sequential(*layers)

    def compute_output_length(self, input_length):
        output_length = input_length.to(
            torch.float32) * self.encoder_config.hop_length / self.encoder_config.sampling_rate
        output_length = output_length * self.vocoder_config.sampling_rate / self.vocoder_config.hop_length
        return output_length.to(torch.int64)

    def forward(self, coarse_mel, input_length, output_length=None):
        bsz, _, d = coarse_mel.shape
        if not (d == self.vocoder_config.num_mel_bins):
            raise AssertionError("d must be the same with self.vocoder_config.num_mel_bins")
        if output_length is None or not self.training:
            output_length = self.compute_output_length(input_length)
        coarse_mel, default_dtype = coarse_mel[:, :torch.max(input_length), :], coarse_mel.dtype
        coarse_mel = F.interpolate(coarse_mel.to(torch.float32).transpose(1, 2).contiguous(), size=output_length.max(),
                                   mode='nearest').to(default_dtype)
        refined_mel = self.layers(coarse_mel).transpose(1, 2).contiguous()  # (bs, t, d)
        coarse_mel = coarse_mel.transpose(1, 2)  # (bs, max(output_length), d)
        refined_mel += coarse_mel  # residual conntection
        sequence_mask, _ = get_sequence_mask(refined_mel, output_length)
        coarse_mel = torch.where(sequence_mask, coarse_mel, 0)
        refined_mel = torch.where(sequence_mask, refined_mel, 0)
        return refined_mel, coarse_mel, output_length


@dataclass
class OmniAudioDecoderOutput(ModelOutput):
    refined_mel: Optional[torch.FloatTensor] = None
    coarse_mel: Optional[torch.FloatTensor] = None
    mel_length: Optional[torch.Tensor] = None
    hidden_states_before_dconv2: Optional[torch.FloatTensor] = None
    output_length_before_dconv2: Optional[torch.Tensor] = None


class LongcatNextAudioDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocoder_config = config.vocoder_config
        self.max_source_positions = self.config.max_audio_seconds * self.config.sampling_rate // self.config.hop_length

        self.dconv1 = CasualConvTranspose1d(
            self.config.d_model,
            self.config.d_model,
            self.config.decoder_kernel_size,
            self.config.avg_pooler,
        )
        self.register_buffer("positional_embedding", sinusoids(self.max_source_positions, self.config.d_model))
        # causal transformer layers
        self.layers = nn.ModuleList(
            [OmniWhisperTransformerLayer(
                ACT2FN[self.config.activation_function],
                self.config.d_model,
                self.config.decoder_attention_heads,
                self.config.decoder_ffn_dim,
                True  # causal
            ) for _ in range(self.config.decoder_layers)
            ])
        self.layer_norm = nn.LayerNorm(self.config.d_model)
        self.dconv2 = CasualConvTranspose1d(
            self.config.d_model,
            self.vocoder_config.num_mel_bins,
            self.config.decoder_kernel_size,
            self.config.decoder_stride_size
        )
        self.post_net = MelSpecRefineNet(self.config, self.vocoder_config)
        self.gradient_checkpointing = False

    def forward(self,
                audio_embed,
                input_length,
                mel_labels=None,
                mel_labels_length=None,
                ):
        if not (audio_embed.shape[-1] == self.config.d_model):
            raise AssertionError("audio_embed.shape[-1] must be the same with self.config.d_model")
        audio_embed = audio_embed.to(self.layer_norm.weight)  # device and type
        audio_embed, output_length = self.dconv1(audio_embed, input_length, output_dim=3)  # (b, l*2, d_model)
        _, tgt_len, _ = audio_embed.size()
        if tgt_len < self.positional_embedding.shape[0]:
            current_positional_embedding = self.positional_embedding[:tgt_len]
        else:
            current_positional_embedding = self.positional_embedding
        hidden_states = (audio_embed.to(torch.float32) + current_positional_embedding).to(audio_embed.dtype)

        # packing hidden states
        attention_mask, _ = get_sequence_mask(hidden_states, output_length)
        hidden_states = torch.masked_select(hidden_states, attention_mask).view(torch.sum(output_length),
                                                                                self.config.d_model)

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states, output_length)

        hidden_states = self.layer_norm(hidden_states)
        hidden_states_before_dconv2 = hidden_states
        output_length_before_dconv2 = output_length

        coarse_mel, output_length = self.dconv2(hidden_states, output_length, output_dim=3)
        refined_mel, coarse_mel, mel_labels_length = self.post_net(coarse_mel, output_length, mel_labels_length)

        return OmniAudioDecoderOutput(
            refined_mel=refined_mel,
            coarse_mel=coarse_mel,
            mel_length=mel_labels_length,
            hidden_states_before_dconv2=hidden_states_before_dconv2,
            output_length_before_dconv2=output_length_before_dconv2,
        )


class LongcatNextAudioVQBridger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.intermediate_dim = self.config.d_model * self.config.avg_pooler
        self.gate_proj = nn.Conv1d(self.config.d_model, self.intermediate_dim, self.config.avg_pooler,
                                   self.config.avg_pooler, bias=False)
        self.up_proj = nn.Conv1d(self.config.d_model, self.intermediate_dim, self.config.avg_pooler,
                                 self.config.avg_pooler, bias=False)

        self.down_proj = nn.Linear(self.intermediate_dim, self.intermediate_dim, bias=False)
        self.act_fn = ACT2FN['silu']
        self.layer_norm = nn.LayerNorm(self.intermediate_dim)
        self.proj_decoder = nn.Linear(self.intermediate_dim, self.config.d_model)

        self.vq_list = nn.ModuleList([])
        for idx, codebook_size in enumerate(self.config.vq_config.codebook_sizes):
            vq_config = copy.deepcopy(self.config.vq_config)
            vq_config.dim = self.intermediate_dim
            vq_config.codebook_size = codebook_size
            self.vq_list.append(VectorQuantize(vq_config))

    def rvq_op(self, inputs, output_length):
        def rvq_layer_op(vq_layer, residual_encoding, output_length):
            q_v_i, code_ids_i = vq_layer(residual_encoding, output_length)
            residual_encoding = residual_encoding.float() - q_v_i.float()
            residual_encoding = residual_encoding.to(inputs.dtype)
            return residual_encoding, code_ids_i

        cmt_loss, residual_encoding = 0, inputs
        code_ids_list = []
        for i, vq_layer in enumerate(self.vq_list):
            residual_encoding, code_ids_i = rvq_layer_op(vq_layer, residual_encoding, output_length)
            code_ids_list.append(code_ids_i)
        return torch.stack(code_ids_list, -1)

    def forward(self, x, output_length):
        batch_size, _, _ = x.shape
        output_length = output_length.to(x.device)

        if x.shape[1] % self.config.avg_pooler != 0:
            x = F.pad(x, (0, 0, 0, self.config.avg_pooler - x.shape[1] % self.config.avg_pooler), "constant", 0)
        xt = x.permute(0, 2, 1)
        g = self.gate_proj(xt).permute(0, 2, 1)  # (bs, sl//poolersizre+1, d*2)
        u = self.up_proj(xt).permute(0, 2, 1)
        x = x.reshape(batch_size, -1, self.intermediate_dim)  # (bs, sl//poolersizre+1, d*2)

        c = self.down_proj(self.act_fn(g) * u)
        res = self.layer_norm(c + x)
        valid_mask, _ = get_sequence_mask(res, output_length)
        code_ids = self.rvq_op(res, output_length)
        code_ids = torch.masked_select(code_ids, valid_mask).reshape(-1,
                                                                     len(self.vq_list))  # (sum(valid_sequence_length), vq_num)
        return code_ids

    @torch.no_grad()
    def decode(self, code_ids):
        vq_num = code_ids.shape[-1]
        res = sum(
            self.vq_list[i].get_output_from_indices(code_ids[:, i]).float() for i in range(vq_num - 1, -1, -1)).to(
            self.proj_decoder.weight)
        decoder_emb = self.proj_decoder(res.to(self.proj_decoder.weight))
        return decoder_emb

    @torch.no_grad()
    def recover(self, code_ids):
        vq_num = code_ids.shape[-1]
        res = sum(
            self.vq_list[i].get_output_from_indices(code_ids[:, i]).float() for i in range(vq_num - 1, -1, -1)).to(
            self.proj_decoder.weight)
        return res


class FlowmatchingPrenet(nn.Module):
    def __init__(
            self,
            input_feat_dim,
            out_feat_dim,
            d_model,
            attention_heads,
            ffn_dim,
            nlayers,
            activation_function,
            max_source_positions,
            target_mel_length_scale_ratio,
    ):
        super().__init__()

        self.d_model = d_model
        self.target_mel_length_scale_ratio = target_mel_length_scale_ratio
        self.gradient_checkpointing = False

        self.register_buffer(
            "positional_embedding", sinusoids(max_source_positions, d_model)
        )

        self.in_mlp = nn.Sequential(
            nn.Linear(input_feat_dim, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.transformer_layers = nn.ModuleList(
            [
                OmniWhisperTransformerLayer(
                    act=ACT2FN[activation_function],
                    d_model=d_model,
                    encoder_attention_heads=attention_heads,
                    encoder_ffn_dim=ffn_dim,
                    causal=True,  # causal
                    ln_type="RMSNorm",
                )
                for _ in range(nlayers)
            ]
        )

        self.final_norm = RMSNorm(self.d_model)
        self.out_proj = nn.Linear(d_model, out_feat_dim, bias=False)

    def compute_output_length(self, input_length):
        output_length = input_length.float() * self.target_mel_length_scale_ratio
        return output_length.to(torch.int64)

    def forward(self, input_feat, input_length, output_length=None):
        """
        Args:
            input_feat: [B, T, input_feat_dim]
            input_length: [B]
            output_length: [B]
        """
        if output_length is None or not self.training:
            output_length = self.compute_output_length(input_length)

        input_feat = input_feat[:, : input_length.max(), :]  # [B, T, D]
        orig_dtype = input_feat.dtype

        input_feat = F.interpolate(
            input=input_feat.to(torch.float32).transpose(1, 2).contiguous(),
            size=output_length.max(),
            mode="nearest",
        ).to(orig_dtype)
        input_feat = input_feat.transpose(1, 2).contiguous()  # [B, T, D]
        hidden_states = self.in_mlp(input_feat)

        # packing hidden states
        bsz, tgt_len, d_model = hidden_states.shape
        attention_mask, unpacking_index = get_sequence_mask(
            hidden_states, output_length
        )
        hidden_states = torch.masked_select(hidden_states, attention_mask).view(
            torch.sum(output_length), self.d_model
        )

        for idx, encoder_layer in enumerate(self.transformer_layers):
            hidden_states = encoder_layer(hidden_states, output_length)

        # unpacking
        hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
            bsz, tgt_len, d_model
        )
        hidden_states = torch.where(attention_mask, hidden_states, 0)

        hidden_states = self.final_norm(hidden_states)
        output = self.out_proj(hidden_states)
        return output, output_length


@dataclass
class OmniAudioFlowMatchingDecoderOutput(ModelOutput):
    flow_matching_mel: Optional[torch.FloatTensor] = None
    flow_matching_mel_lengths: Optional[torch.FloatTensor] = None


class LongcatNextAudioFlowMatchingDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.flow_matching_config
        self.in_channels = self.config.in_channels
        self.spk_emb_dim = self.config.spk_emb_dim
        self.diffusion_steps = self.config.diffusion_steps
        self.cal_mel_mae = self.config.cal_mel_mae
        self.forward_step = -1

        self.prenet = FlowmatchingPrenet(
            input_feat_dim=self.config.prenet_in_dim,
            out_feat_dim=self.config.prenet_out_dim,
            d_model=self.config.prenet_d_model,
            attention_heads=self.config.prenet_attention_heads,
            ffn_dim=self.config.prenet_ffn_dim,
            nlayers=self.config.prenet_nlayers,
            activation_function=self.config.prenet_activation_function,
            max_source_positions=self.config.prenet_max_source_positions,
            target_mel_length_scale_ratio=self.config.prenet_target_mel_length_scale_ratio,
        )

        self.conditional_decoder = ConditionalDecoder(
            in_channels=self.in_channels * 2 + self.spk_emb_dim,
            out_channels=self.in_channels,
            causal=True,
            channels=self.config.channels,
            dropout=self.config.dropout,
            attention_head_dim=self.config.attention_head_dim,
            n_blocks=self.config.n_blocks,
            num_mid_blocks=self.config.num_mid_blocks,
            num_heads=self.config.num_heads,
            act_fn=self.config.act_fn,
        )

        self.cfm = ConditionalCFM(
            in_channels=self.in_channels,
            cfm_params=self.config.cfm_params,
            n_spks=0,
            spk_emb_dim=self.spk_emb_dim,
        )

    def unpack_hidden_states(self, hidden_states, output_length):
        unpacked = unpack_hidden_states(hidden_states, output_length)
        return unpacked, output_length

    def forward(
            self, refined_mel, input_length, mel_labels=None, mel_labels_length=None
    ):
        self.forward_step += 1
        if self.prenet is not None:
            refined_mel = refined_mel[:, : torch.max(input_length), :]
            if mel_labels_length is None:
                mel_labels_length = self.prenet.compute_output_length(input_length)
            refined_mel, input_length = self.prenet(
                refined_mel, input_length, mel_labels_length
            )

        float_dtype = refined_mel.dtype
        refined_mel = refined_mel.float()
        input_length = input_length.long()

        refined_mel = refined_mel[:, : torch.max(input_length), :]
        sequence_mask, unpacking_index = get_sequence_mask(refined_mel, input_length)
        refined_mel = refined_mel.transpose(1, 2)  # (bs, mel_bin, max_input_len)
        sequence_mask = sequence_mask.transpose(2, 1)  # (bs, 1, sl)

        fm_mel = self.cfm.forward(
            estimator=self.conditional_decoder,
            mu=refined_mel.to(float_dtype),
            mask=sequence_mask.float(),
            n_timesteps=self.diffusion_steps,
        )
        return OmniAudioFlowMatchingDecoderOutput(
            flow_matching_mel=fm_mel.transpose(1, 2),
            flow_matching_mel_lengths=mel_labels_length,
        )


@torch.no_grad()
def decode_wave_vocoder2(response, vocoder, audio_tokenizer):
    response_len = (response[:, :, 0] == audio_tokenizer.config.audio_config.vq_config.codebook_sizes[0]).long().argmax(
        dim=1)
    valid_response_list = [response[i, :response_len[i], :] for i in range(response.shape[0]) if
                           int(response_len[i]) > 0]

    if len(valid_response_list) == 0:
        return []
    flatten_response = torch.cat(valid_response_list, dim=0) if len(valid_response_list) > 1 else valid_response_list[0]
    valid_response_len = response_len[response_len > 0]
    ret = audio_tokenizer.decode(flatten_response.view(-1, response.shape[-1]),
                                 bridge_length=valid_response_len)
    batch_size = response.shape[0]
    valid_start = 0
    r = []
    for i in range(batch_size):
        if response_len[i] == 0:
            r.append(None)
            continue
        if isinstance(ret, torch.Tensor):
            r.append(ret[valid_start:valid_start + 1])
            valid_start += 1
            continue
        decode_wave = vocoder.decode(
            ret.flow_matching_mel[valid_start][:ret.flow_matching_mel_lengths[valid_start], :].transpose(0, 1).to(
                torch.float32).unsqueeze(0))
        r.append(decode_wave.cpu())
        valid_start += 1
    return r


@torch.no_grad()
def decode_save_concat2(response_list, vocoder, model, path, sampling_rate=16000, wave_concat_overlap=800):
    wave_list = []
    for response in response_list:
        wave_list.extend([wave_i for wave_i in decode_wave_vocoder2(response, vocoder, model) if wave_i is not None])
    new_wave_list = [wave_list[0]]
    if not wave_list:
        raise ValueError(f"wave_list is empty")
    for w in wave_list[1:]:
        if new_wave_list[-1].shape[1] > wave_concat_overlap and w.shape[1] > wave_concat_overlap:
            new_wave_list.append((new_wave_list[-1][:, -wave_concat_overlap:] * torch.linspace(1.0, 0.0,
                                                                                               wave_concat_overlap,
                                                                                               device=new_wave_list[
                                                                                                   -1].device)[None, :]
                                  + w[:, :wave_concat_overlap] * torch.linspace(0.0, 1.0, wave_concat_overlap,
                                                                                device=new_wave_list[-1].device)[None,
                                                                 :]))
        new_wave_list.append(w)
    full_wave = torch.cat(new_wave_list, dim=1) if len(new_wave_list) > 1 else new_wave_list[0]
    torchaudio.save(path, full_wave, sampling_rate)


class LongcatNextAudioTokenizer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_model = LongcatNextAudioEncoder(config.audio_config)
        self.audio_bridge_model = LongcatNextAudioVQBridger(config.audio_config)
        self.audio_decoder = LongcatNextAudioDecoder(config.audio_config)
        self.audio_flow_matching_decoder = LongcatNextAudioFlowMatchingDecoder(config.audio_config)
        self.cosy24kvocoder = None

    @torch.no_grad()
    def encode(self, x, encoder_length: Optional[torch.Tensor] = None, bridge_length: Optional[torch.Tensor] = None):
        audio_emb = self.audio_model(x, encoder_length)
        audio_tokens = self.audio_bridge_model(audio_emb, bridge_length)
        return audio_tokens

    @torch.no_grad()
    def decode(self, audio_ids, bridge_length: Optional[torch.Tensor] = None):
        audio_emb = self.audio_bridge_model.decode(audio_ids)
        audio_dec = self.audio_decoder(
            audio_emb.to(next(self.audio_decoder.parameters())), bridge_length
        )
        if self.config.audio_config.flow_matching_config.use_hidden_states_before_dconv2:
            hidden_states, hidden_states_length = (
                self.audio_flow_matching_decoder.unpack_hidden_states(
                    audio_dec.hidden_states_before_dconv2,
                    audio_dec.output_length_before_dconv2,
                )
            )
            audio_flow_matching_decoder_ret = self.audio_flow_matching_decoder(
                hidden_states, hidden_states_length
            )
        else:
            audio_flow_matching_decoder_ret = self.audio_flow_matching_decoder(
                audio_dec.refined_mel, audio_dec.mel_length
            )
        return audio_flow_matching_decoder_ret

    @torch.no_grad()
    def lazy_decode_and_save(self, audio_ids, sampling_rate, wave_concat_overlap, save_path):
        if self.cosy24kvocoder is None:
            print("lazy load cosy24kvocoder ...")
            device = next(self.parameters()).device
            self.cosy24kvocoder = Cosy24kVocoder.from_pretrained(
                self.config.audio_config.cosy24kvocoder_config.weight_path).to(device)

        if audio_ids[-1, 0] != self.config.audio_config.vq_config.codebook_sizes[0]:  # exceed max_new_tokens
            audio_ids = F.pad(audio_ids, (0, 0, 0, 1), value=self.config.audio_config.vq_config.codebook_sizes[0])

        audio_end_pos = [-1] + (audio_ids[:, 0] == self.config.audio_config.vq_config.codebook_sizes[0]).nonzero().view(
            -1).tolist()

        audio_ids_chunk = []
        for i in range(len(audio_end_pos) - 1):
            start = audio_end_pos[i] + 1
            end = audio_end_pos[i + 1] + 1
            audio_ids_chunk.append(audio_ids[start:end].unsqueeze(0))

        audio_ids = audio_ids_chunk

        decode_save_concat2(
            response_list=audio_ids,
            vocoder=self.cosy24kvocoder,
            model=self,
            path=save_path,
            sampling_rate=sampling_rate,
            wave_concat_overlap=wave_concat_overlap,
        )
