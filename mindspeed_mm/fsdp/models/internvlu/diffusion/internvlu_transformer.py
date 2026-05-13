# --------------------------------------------------------
# InternVL-U
# Modifications Copyright (c) 2026 OpenGVLab
# This file includes code from Qwen-Image and HuggingFace,
# licensed under the Apache License, Version 2.0.
# --------------------------------------------------------
# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.transformers.transformer_qwenimage import (
    QwenTimestepProjEmbeddings,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm, FP32LayerNorm
from diffusers.models.modeling_outputs import dataclass, BaseOutput

from .attention_processor import AttentionVE

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or
        `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: "torch.Tensor"  # noqa: F821


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _basic_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        w = module.weight
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    elif isinstance(module, RMSNorm):
        if module.weight is not None:
            nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def apply_rotary_emb_ms(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(-2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class UnifiedMSRoPE(nn.Module):
    """
    Unified multi-scale RoPE module for 3D position encodings.

    Args:
        theta (`int`): Base frequency for rotary embeddings.
        axes_dim (`List[int]`): Per-axis embedding dimensions (frame, height, width).
        scale_rope (`bool`, *optional*, defaults to `False`): Whether to apply scaling to cosine/sine outputs.
    """

    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        inv_freqs_list = []
        self.axis_offsets = [0]

        for dim in self.axes_dim:
            assert dim % 2 == 0, f"Dimension {dim} must be even"
            inv_freq = 1.0 / torch.pow(
                theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)
            )
            inv_freqs_list.append(inv_freq)
            self.axis_offsets.append(self.axis_offsets[-1] + len(inv_freq))

        self.all_inv_freqs = torch.cat(inv_freqs_list, dim=0).float()

    def get_inv_freq(self, axis_idx, device):
        start_idx = self.axis_offsets[axis_idx]
        end_idx = self.axis_offsets[axis_idx + 1]
        return self.all_inv_freqs[start_idx:end_idx].to(device)

    def rope_params(self, positions, axis_idx):
        """
        Args:
            positions (`torch.Tensor`):
                Position tensor of shape `[N]` (supports float or negative values).
            axis_idx (`int`):
                Axis index (0: frame, 1: height, 2: width).

        Returns:
            `torch.Tensor`: Complex frequencies of shape `[N, dim // 2]`.
        """
        positions = positions.float()
        inv_freq = self.get_inv_freq(axis_idx, positions.device).float()
        freqs = torch.outer(positions, inv_freq)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex

    def forward(self, position_ids_3d, device=None):
        if device is None:
            device = position_ids_3d.device

        position_ids_3d = position_ids_3d.to(device)
        L = position_ids_3d.shape[0]

        total_dim = sum(self.axes_dim) // 2
        freqs_result = torch.zeros(L, total_dim, dtype=torch.complex64, device=device)

        dim_offset = 0
        for axis_idx in range(3):
            axis_dim = self.axes_dim[axis_idx] // 2
            axis_positions = position_ids_3d[:, axis_idx]

            axis_freqs = self.rope_params(axis_positions, axis_idx)
            freqs_result[:, dim_offset : dim_offset + axis_dim] = axis_freqs

            dim_offset += axis_dim

        return freqs_result

    def get_cos_sin(self, position_ids_3d, device=None):
        freqs_complex = self.forward(position_ids_3d, device)

        cos_freqs = freqs_complex.real
        sin_freqs = freqs_complex.imag

        cos = torch.cat([cos_freqs, cos_freqs], dim=-1)
        sin = torch.cat([sin_freqs, sin_freqs], dim=-1)

        if self.scale_rope:
            attention_scaling = 1.0
            cos = cos * attention_scaling
            sin = sin * attention_scaling

        return cos, sin


def get_video_scale_factors(video_scale_factor, batch_size, video_fhw):
    """Convert video_scale_factor to per-video scale factors"""
    if video_scale_factor is None:
        return [
            [1.0] * len(sample_videos) if sample_videos else [1.0]
            for sample_videos in (video_fhw or [[]] * batch_size)
        ]

    if isinstance(video_scale_factor, (int, float)):
        return [
            (
                [video_scale_factor] * len(sample_videos)
                if sample_videos
                else [video_scale_factor]
            )
            for sample_videos in (video_fhw or [[]] * batch_size)
        ]

    if isinstance(video_scale_factor, list):
        if len(video_scale_factor) == 0:
            return [
                [1.0] * len(sample_videos) if sample_videos else [1.0]
                for sample_videos in (video_fhw or [[]] * batch_size)
            ]

        if isinstance(video_scale_factor[0], list):
            result = []
            for batch_idx in range(batch_size):
                if batch_idx < len(video_scale_factor):
                    result.append(video_scale_factor[batch_idx])
                else:
                    sample_videos = (
                        video_fhw[batch_idx]
                        if video_fhw and batch_idx < len(video_fhw)
                        else []
                    )
                    if not isinstance(sample_videos, list) or (
                        len(sample_videos) > 0
                        and not isinstance(sample_videos[0], (list, tuple))
                    ):
                        sample_videos = [sample_videos] if sample_videos else []
                    result.append([1.0] * len(sample_videos))
            return result
        else:
            result = []
            for batch_idx in range(batch_size):
                batch_scale = (
                    video_scale_factor[batch_idx]
                    if batch_idx < len(video_scale_factor)
                    else 1.0
                )
                sample_videos = (
                    video_fhw[batch_idx]
                    if video_fhw and batch_idx < len(video_fhw)
                    else []
                )
                if not isinstance(sample_videos, list) or (
                    len(sample_videos) > 0
                    and not isinstance(sample_videos[0], (list, tuple))
                ):
                    sample_videos = [sample_videos] if sample_videos else []
                result.append([batch_scale] * len(sample_videos))
            return result

    return [
        [1.0] * len(sample_videos) if sample_videos else [1.0]
        for sample_videos in (video_fhw or [[]] * batch_size)
    ]


def create_position_ids_3d_v2(
    video_fhw: Optional[Union[List, List[List]]] = None,
    input_token_mask: Optional[torch.Tensor] = None,
    scale_rope: bool = True,
    video_scale_factor: Optional[Union[float, List[float], List[List[float]]]] = None,
    device: Optional[torch.device] = None,
):
    """
    Optimized helper method to create 3D position IDs for interleaved text-video sequences
    Uses tensor operations instead of loops for better performance

    Args:
        video_fhw: batch of video specs, where each element is a list of [frame, height, width] for each video
                  Shape: [batch_size, num_videos_per_sample, 3] or [batch_size, 3] for single video per sample
        input_token_mask: batch of token type masks, where True indicates video token, False indicates text token
                         Shape: [batch_size, sequence_length]
                         The number of True values should match the total f*h*w from video_fhw
        scale_rope: bool, whether to use centered indexing for position IDs
        video_scale_factor: float, list of floats, or nested list of floats for scaling video position IDs
                           - float: same scale for all videos
                           - List[float]: scale per batch (all videos in a batch use same scale)
                           - List[List[float]]: scale per video (each video can have different scale)
        device: torch device for the output tensor

    Returns:
        position_ids_3d: Tensor of shape [total_tokens, 3] where each row is [frame_idx, height_idx, width_idx]
    """
    if video_fhw is None and input_token_mask is None:
        return torch.empty(0, 3, dtype=torch.float32, device=device)

    batch_size = (
        len(input_token_mask) if input_token_mask is not None else len(video_fhw)
    )

    video_scale_factors_per_video = get_video_scale_factors(
        video_scale_factor, batch_size, video_fhw
    )

    batch_position_ids = []

    for batch_idx in range(batch_size):
        if input_token_mask is not None and batch_idx < len(input_token_mask):
            token_mask = input_token_mask[batch_idx]
        else:
            total_video_tokens = sum(
                f * h * w for f, h, w in (video_fhw[batch_idx] if video_fhw else [])
            )
            token_mask = torch.ones(total_video_tokens, dtype=torch.bool, device=device)

        seq_len = len(token_mask)

        sample_video_fhw = []
        if video_fhw is not None and batch_idx < len(video_fhw):
            sample_video_fhw = video_fhw[batch_idx]

            if not isinstance(sample_video_fhw, list):
                sample_video_fhw = [sample_video_fhw]

            if len(sample_video_fhw) > 0 and not isinstance(
                sample_video_fhw[0], (list, tuple)
            ):
                sample_video_fhw = [sample_video_fhw]

        current_batch_scales = (
            video_scale_factors_per_video[batch_idx]
            if batch_idx < len(video_scale_factors_per_video)
            else [1.0]
        )

        video_position_blocks = []
        video_token_counts = []

        if sample_video_fhw:
            for video_idx, fhw in enumerate(sample_video_fhw):
                frame, height, width = fhw
                video_token_counts.append(frame * height * width)

                current_video_scale = (
                    current_batch_scales[video_idx]
                    if video_idx < len(current_batch_scales)
                    else 1.0
                )

                video_pos = _create_single_video_positions(
                    frame,
                    height,
                    width,
                    cum_frame=0,
                    scale_rope=scale_rope,
                    scale_factor=current_video_scale,
                    device=device,
                )
                video_position_blocks.append(video_pos)

        video_mask = token_mask

        position_ids = torch.zeros(seq_len, 3, dtype=torch.float32, device=device)

        if len(video_position_blocks) > 0:
            mask_diff = torch.diff(
                video_mask.float(), prepend=torch.tensor([0.0], device=device)
            )
            video_starts = torch.where(mask_diff == 1)[0]
            video_ends = torch.where(mask_diff == -1)[0]

            if len(video_ends) < len(video_starts):
                video_ends = torch.cat(
                    [video_ends, torch.tensor([seq_len], device=device)]
                )

            current_text_pos = 0
            video_block_idx = 0

            if len(video_starts) == 0 or video_starts[0] > 0:
                first_video_start = (
                    video_starts[0] if len(video_starts) > 0 else seq_len
                )
                text_positions = torch.arange(
                    current_text_pos,
                    current_text_pos + first_video_start,
                    dtype=torch.float32,
                    device=device,
                )
                position_ids[:first_video_start] = text_positions.unsqueeze(1).expand(
                    -1, 3
                )
                current_text_pos += first_video_start

            for seg_idx in range(len(video_starts)):
                video_start = video_starts[seg_idx]
                video_end = video_ends[seg_idx]
                video_length = video_end - video_start

                if video_block_idx < len(video_position_blocks):
                    video_pos = video_position_blocks[video_block_idx]
                    if len(video_pos) == video_length:
                        adjusted_video_pos = video_pos.clone()
                        adjusted_video_pos[:, 0] += current_text_pos
                        position_ids[video_start:video_end] = adjusted_video_pos

                        max_pos = torch.max(adjusted_video_pos).item()
                        current_text_pos = int(max_pos) + 1
                    else:
                        fallback_pos = torch.arange(
                            current_text_pos,
                            current_text_pos + video_length,
                            dtype=torch.float32,
                            device=device,
                        )
                        position_ids[video_start:video_end] = fallback_pos.unsqueeze(
                            1
                        ).expand(-1, 3)
                        current_text_pos += video_length

                    video_block_idx += 1

                next_video_start = (
                    video_starts[seg_idx + 1]
                    if seg_idx + 1 < len(video_starts)
                    else seq_len
                )
                text_segment_length = next_video_start - video_end

                if text_segment_length > 0:
                    text_positions = torch.arange(
                        current_text_pos,
                        current_text_pos + text_segment_length,
                        dtype=torch.float32,
                        device=device,
                    )
                    position_ids[video_end:next_video_start] = text_positions.unsqueeze(
                        1
                    ).expand(-1, 3)
                    current_text_pos += text_segment_length

        else:
            text_positions = torch.arange(seq_len, dtype=torch.float32, device=device)
            position_ids = text_positions.unsqueeze(1).expand(-1, 3)

        batch_position_ids.append(position_ids)

    if batch_position_ids:
        result = torch.cat(batch_position_ids, dim=0)
    else:
        result = torch.empty(0, 3, dtype=torch.float32, device=device)

    if device is not None:
        result = result.to(device)

    return result


def create_position_ids_3d_v3(
    video_fhw: Optional[Union[List, List[List], List[List[List]]]] = None,
    input_token_mask: Optional[torch.Tensor] = None,
    scale_rope: bool = True,
    video_scale_factor: Optional[
        Union[float, List[float], List[List[float]], List[List[List[float]]]]
    ] = None,
    device: Optional[torch.device] = None,
):
    """
    Optimized helper method to create 3D position IDs for interleaved text-video sequences
    Uses tensor operations instead of loops for better performance

    Args:
        video_fhw: batch of video specs, supports multiple formats:
                  - [batch_size, 3]: single video per sample
                  - [batch_size, num_videos_per_sample, 3]: multiple videos per sample
                  - [batch_size, num_videos_per_sample, num_flips_per_video, 3]: multiple videos with flips per sample
        input_token_mask: batch of token type masks, where True indicates video token, False indicates text token
                         Shape: [batch_size, sequence_length]
                         The number of True values should match the total f*h*w from video_fhw
        scale_rope: bool, whether to use centered indexing for position IDs
        video_scale_factor: scaling factors, supports multiple formats:
                           - float: same scale for all videos
                           - List[float]: scale per batch
                           - List[List[float]]: scale per video
                           - List[List[List[float]]]: scale per flip
        device: torch device for the output tensor

    Returns:
        position_ids_3d: Tensor of shape [total_tokens, 3] where each row is [frame_idx, height_idx, width_idx]
    """
    if video_fhw is None and input_token_mask is None:
        return torch.empty(0, 3, dtype=torch.float32, device=device)

    batch_size = (
        len(input_token_mask) if input_token_mask is not None else len(video_fhw)
    )

    video_scale_factors_per_video = get_video_scale_factors_with_flips(
        video_scale_factor, batch_size, video_fhw
    )

    batch_position_ids = []

    for batch_idx in range(batch_size):
        if input_token_mask is not None and batch_idx < len(input_token_mask):
            token_mask = input_token_mask[batch_idx]
        else:
            total_video_tokens = calculate_total_video_tokens(
                video_fhw[batch_idx] if video_fhw else []
            )
            token_mask = torch.ones(total_video_tokens, dtype=torch.bool, device=device)

        seq_len = len(token_mask)

        sample_video_fhw = []
        if video_fhw is not None and batch_idx < len(video_fhw):
            sample_video_fhw = video_fhw[batch_idx]

            if not isinstance(sample_video_fhw, list):
                sample_video_fhw = [sample_video_fhw]

            if len(sample_video_fhw) > 0 and not isinstance(
                sample_video_fhw[0], (list, tuple)
            ):
                sample_video_fhw = [sample_video_fhw]

        current_batch_scales = (
            video_scale_factors_per_video[batch_idx]
            if batch_idx < len(video_scale_factors_per_video)
            else [[[1.0]]]
        )

        video_position_blocks = []

        if sample_video_fhw:
            for video_idx, video_data in enumerate(sample_video_fhw):
                video_scales = (
                    current_batch_scales[video_idx]
                    if video_idx < len(current_batch_scales)
                    else [[1.0]]
                )

                if (
                    isinstance(video_data, list)
                    and len(video_data) > 0
                    and isinstance(video_data[0], list)
                    and len(video_data[0]) == 3
                ):

                    video_flip_positions = []

                    cum_frame = 0
                    for flip_idx, fhw in enumerate(video_data):
                        frame, height, width = fhw

                        flip_scale = (
                            video_scales[flip_idx]
                            if flip_idx < len(video_scales)
                            else 1.0
                        )
                        if isinstance(flip_scale, list):
                            flip_scale = flip_scale[0] if len(flip_scale) > 0 else 1.0

                        flip_pos = _create_single_video_positions(
                            frame,
                            height,
                            width,
                            cum_frame=cum_frame,
                            scale_rope=scale_rope,
                            scale_factor=flip_scale,
                            device=device,
                        )
                        video_flip_positions.append(flip_pos)
                        cum_frame += frame

                    if video_flip_positions:
                        video_pos = torch.cat(video_flip_positions, dim=0)
                        video_position_blocks.append(video_pos)

                else:
                    frame, height, width = video_data

                    video_scale = video_scales[0] if len(video_scales) > 0 else 1.0
                    if isinstance(video_scale, list):
                        video_scale = video_scale[0] if len(video_scale) > 0 else 1.0
                        if isinstance(video_scale, list):
                            video_scale = (
                                video_scale[0] if len(video_scale) > 0 else 1.0
                            )

                    video_pos = _create_single_video_positions(
                        frame,
                        height,
                        width,
                        cum_frame=0,
                        scale_rope=scale_rope,
                        scale_factor=video_scale,
                        device=device,
                    )
                    video_position_blocks.append(video_pos)

        video_mask = token_mask

        position_ids = torch.zeros(seq_len, 3, dtype=torch.float32, device=device)

        if len(video_position_blocks) > 0:
            mask_diff = torch.diff(
                video_mask.float(), prepend=torch.tensor([0.0], device=device)
            )
            video_starts = torch.where(mask_diff == 1)[0]
            video_ends = torch.where(mask_diff == -1)[0]

            if len(video_ends) < len(video_starts):
                video_ends = torch.cat(
                    [video_ends, torch.tensor([seq_len], device=device)]
                )

            current_text_pos = 0
            video_block_idx = 0

            if len(video_starts) == 0 or video_starts[0] > 0:
                first_video_start = (
                    video_starts[0] if len(video_starts) > 0 else seq_len
                )
                text_positions = torch.arange(
                    current_text_pos,
                    current_text_pos + first_video_start,
                    dtype=torch.float32,
                    device=device,
                )
                position_ids[:first_video_start] = text_positions.unsqueeze(1).expand(
                    -1, 3
                )
                current_text_pos += first_video_start

            for seg_idx in range(len(video_starts)):
                video_start = video_starts[seg_idx]
                video_end = video_ends[seg_idx]
                video_length = video_end - video_start

                if video_block_idx < len(video_position_blocks):
                    video_pos = video_position_blocks[video_block_idx]
                    if len(video_pos) == video_length:
                        adjusted_video_pos = video_pos.clone()
                        adjusted_video_pos[:, 0] += current_text_pos
                        position_ids[video_start:video_end] = adjusted_video_pos

                        max_pos = torch.max(adjusted_video_pos).item()
                        current_text_pos = int(max_pos) + 1
                    else:
                        fallback_pos = torch.arange(
                            current_text_pos,
                            current_text_pos + video_length,
                            dtype=torch.float32,
                            device=device,
                        )
                        position_ids[video_start:video_end] = fallback_pos.unsqueeze(
                            1
                        ).expand(-1, 3)
                        current_text_pos += video_length

                    video_block_idx += 1

                next_video_start = (
                    video_starts[seg_idx + 1]
                    if seg_idx + 1 < len(video_starts)
                    else seq_len
                )
                text_segment_length = next_video_start - video_end

                if text_segment_length > 0:
                    text_positions = torch.arange(
                        current_text_pos,
                        current_text_pos + text_segment_length,
                        dtype=torch.float32,
                        device=device,
                    )
                    position_ids[video_end:next_video_start] = text_positions.unsqueeze(
                        1
                    ).expand(-1, 3)
                    current_text_pos += text_segment_length

        else:
            text_positions = torch.arange(seq_len, dtype=torch.float32, device=device)
            position_ids = text_positions.unsqueeze(1).expand(-1, 3)

        batch_position_ids.append(position_ids)

    if batch_position_ids:
        result = torch.cat(batch_position_ids, dim=0)
    else:
        result = torch.empty(0, 3, dtype=torch.float32, device=device)

    if device is not None:
        result = result.to(device)

    return result


def get_video_scale_factors_with_flips(video_scale_factor, batch_size, video_fhw):
    """
    Helper function to handle video scale factors with flip support
    """
    if video_scale_factor is None:
        return [[[1.0]] for _ in range(batch_size)]

    if isinstance(video_scale_factor, (int, float)):
        return [[[video_scale_factor]] for _ in range(batch_size)]

    if isinstance(video_scale_factor, list):
        if len(video_scale_factor) == 0:
            return [[[1.0]] for _ in range(batch_size)]

        if isinstance(video_scale_factor[0], (int, float)):
            result = []
            for batch_idx in range(batch_size):
                scale = (
                    video_scale_factor[batch_idx]
                    if batch_idx < len(video_scale_factor)
                    else 1.0
                )
                result.append([[scale]])
            return result

        elif isinstance(video_scale_factor[0], list):
            if len(video_scale_factor[0]) > 0 and isinstance(
                video_scale_factor[0][0], (int, float)
            ):
                result = []
                for batch_idx in range(batch_size):
                    batch_scales = (
                        video_scale_factor[batch_idx]
                        if batch_idx < len(video_scale_factor)
                        else [1.0]
                    )
                    video_scales = []
                    for video_scale in batch_scales:
                        video_scales.append([video_scale])
                    result.append(video_scales)
                return result

            elif len(video_scale_factor[0]) > 0 and isinstance(
                video_scale_factor[0][0], list
            ):
                return video_scale_factor

    return [[[1.0]] for _ in range(batch_size)]


def calculate_total_video_tokens(sample_video_fhw):
    """
    Calculate total number of video tokens for a sample
    """
    if not sample_video_fhw:
        return 0

    total = 0

    if not isinstance(sample_video_fhw, list):
        return 0

    for video_data in sample_video_fhw:
        if (
            isinstance(video_data, list)
            and len(video_data) > 0
            and isinstance(video_data[0], list)
            and len(video_data[0]) == 3
        ):
            for fhw in video_data:
                f, h, w = fhw
                total += f * h * w
        elif isinstance(video_data, list) and len(video_data) == 3:
            f, h, w = video_data
            total += f * h * w

    return total


def _create_single_video_positions(
    frame,
    height,
    width,
    cum_frame=0,
    scale_rope=True,
    scale_factor=1.0,
    device=None,
):
    """Helper function to create position IDs for a single video"""

    if scale_rope:
        h_neg_count = height - height // 2
        w_neg_count = width - width // 2

        h_indices = torch.cat(
            [
                torch.arange(-h_neg_count, 0, dtype=torch.float32, device=device),
                torch.arange(0, height // 2, dtype=torch.float32, device=device),
            ]
        )
        w_indices = torch.cat(
            [
                torch.arange(-w_neg_count, 0, dtype=torch.float32, device=device),
                torch.arange(0, width // 2, dtype=torch.float32, device=device),
            ]
        )

        if scale_factor != 1.0:
            target_h_neg_count = int(h_neg_count / scale_factor)
            target_h_pos_count = int((height // 2) / scale_factor)
            target_w_neg_count = int(w_neg_count / scale_factor)
            target_w_pos_count = int((width // 2) / scale_factor)

            h_min_orig, h_max_orig = -h_neg_count, height // 2 - 1
            h_min_target, h_max_target = -target_h_neg_count, target_h_pos_count - 1
            if h_max_orig != h_min_orig:
                h_indices = h_min_target + (h_indices - h_min_orig) * (
                    h_max_target - h_min_target
                ) / (h_max_orig - h_min_orig)

            w_min_orig, w_max_orig = -w_neg_count, width // 2 - 1
            w_min_target, w_max_target = -target_w_neg_count, target_w_pos_count - 1
            if w_max_orig != w_min_orig:
                w_indices = w_min_target + (w_indices - w_min_orig) * (
                    w_max_target - w_min_target
                ) / (w_max_orig - w_min_orig)

    else:
        h_indices = torch.arange(height, dtype=torch.float32, device=device)
        w_indices = torch.arange(width, dtype=torch.float32, device=device)

        if scale_factor != 1.0:
            target_height = int(height / scale_factor)
            target_width = int(width / scale_factor)

            h_min_orig, h_max_orig = 0, height - 1
            h_min_target, h_max_target = 0, target_height - 1
            if h_max_orig != h_min_orig:
                h_indices = h_min_target + (h_indices - h_min_orig) * (
                    h_max_target - h_min_target
                ) / (h_max_orig - h_min_orig)

            w_min_orig, w_max_orig = 0, width - 1
            w_min_target, w_max_target = 0, target_width - 1
            if w_max_orig != w_min_orig:
                w_indices = w_min_target + (w_indices - w_min_orig) * (
                    w_max_target - w_min_target
                ) / (w_max_orig - w_min_orig)

    f_indices = torch.arange(
        cum_frame, cum_frame + frame, dtype=torch.float32, device=device
    )

    f_grid, h_grid, w_grid = torch.meshgrid(
        f_indices, h_indices, w_indices, indexing="ij"
    )

    video_positions = torch.stack(
        [f_grid.flatten(), h_grid.flatten(), w_grid.flatten()], dim=1
    )

    return video_positions
