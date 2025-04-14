from typing import Optional, Dict, Tuple
from contextlib import nullcontext

from einops import rearrange, repeat
import torch
import torch.nn as nn
from einops import rearrange, repeat
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps
)
from megatron.legacy.model.enums import AttnType
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args

from mindspeed_mm.models.common import MultiModalModule
from mindspeed_mm.models.common.embeddings.pos_embeddings import RoPE3DStepVideo
from mindspeed_mm.models.common.attention import ParallelAttention
from mindspeed_mm.models.common.ffn import FeedForward
from mindspeed_mm.models.common.communications import split_forward_gather_backward, gather_forward_split_backward


class StepVideoDiT(MultiModalModule):
    def __init__(
        self,
        num_attention_heads: int = 48,
        attention_head_dim: int = 128,
        channel_split: list = None,
        in_channels: int = 64,
        out_channels: Optional[int] = 64,
        num_layers: int = 48,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_norm_type: str = "rmsnorm",
        attention_norm_elementwise_affine: bool = False,
        attention_norm_eps: float = 1e-6,
        fa_layout: str = "bsnd",
        use_additional_conditions: Optional[bool] = False,
        caption_channels: Optional[list] = None,
        **kwargs
    ):
        super().__init__(config=None)

        # Set some common variables used across the board.
        args = get_args()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.caption_channels = caption_channels
        self.use_additional_conditions = use_additional_conditions
        self.sequence_parallel = args.sequence_parallel
        self.recompute_granularity = args.recompute_granularity
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        self.distribute_saved_activations = args.distribute_saved_activations
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in VideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")

        self.pos_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=self.in_channels,
            embed_dim=self.inner_dim
        )

        # Rotary positional embeddings
        self.rope = RoPE3DStepVideo(
            ch_split=channel_split
        )

        self.global_layer_idx = tuple(range(num_layers))
        self.transformer_blocks = nn.ModuleList(
            [
                StepVideoTransformerBlock(
                    dim=self.inner_dim,
                    attention_head_dim=self.attention_head_dim,
                    attention_norm_type=attention_norm_type,
                    attention_norm_elementwise_affine=attention_norm_elementwise_affine,
                    attention_norm_eps=attention_norm_eps,
                    fa_layout=fa_layout,
                    rope=self.rope,
                )
                for _ in range(self.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels)
        self.patch_size = patch_size

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=self.use_additional_conditions
        )

        if isinstance(self.caption_channels, int):
            caption_channel = self.caption_channels
        else:
            caption_channel, clip_channel = self.caption_channels
            self.clip_projection = nn.Linear(clip_channel, self.inner_dim) 

        self.caption_norm = nn.LayerNorm(caption_channel, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channel, hidden_size=self.inner_dim
        )

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    def patchfy(self, hidden_states):
        hidden_states = rearrange(hidden_states, 'b f c h w -> (b f) c h w')
        hidden_states = self.pos_embed(hidden_states)
        return hidden_states

    def prepare_attn_mask(self, encoder_attention_mask, encoder_hidden_states, q_seqlen):
        kv_seqlens = encoder_attention_mask.sum(dim=1).int()
        mask = torch.ones([len(kv_seqlens), q_seqlen, max(kv_seqlens)], dtype=torch.bool, device=encoder_attention_mask.device)
        encoder_hidden_states = encoder_hidden_states[:, : max(kv_seqlens)]
        for i, kv_len in enumerate(kv_seqlens):
            mask[i, :, :kv_len] = 0

        return encoder_hidden_states, mask

    def forward(
        self,
        hidden_states: torch.Tensor, 
        timestep: Optional[torch.LongTensor] = None,
        prompt: Optional[list] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        fps: torch.Tensor = None,
        **kwargs
    ):
        if hidden_states.ndim != 5:
            raise ValueError("hidden_states's shape should be (bsz, f, ch, h ,w)")
        
        # RNG context.
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        encoder_hidden_states = prompt[0]
        encoder_hidden_states_2 = prompt[1]

        # Only retain stepllm's mask
        if isinstance(prompt_mask, list):
            encoder_attention_mask = prompt_mask[0]
        # Padding 1 on the mask of the stepllm
        len_clip = encoder_hidden_states_2.shape[1]
        encoder_attention_mask = encoder_attention_mask.squeeze(1).to(hidden_states.device) # stepchat_tokenizer_mask: b 1 s => b s
        encoder_attention_mask = torch.nn.functional.pad(encoder_attention_mask, (len_clip, 0), value=1)   # pad attention_mask with clip's length 

        bsz, frame, _, height, width = hidden_states.shape
        if mpu.get_context_parallel_world_size() > 1:
            frame //= mpu.get_context_parallel_world_size()
            hidden_states = split_forward_gather_backward(hidden_states, mpu.get_context_parallel_group(), dim=1,
                                                        grad_scale='down')
        
        height, width = height // self.patch_size, width // self.patch_size
        hidden_states = self.patchfy(hidden_states) 
        len_frame = hidden_states.shape[1]

        if self.use_additional_conditions:
            added_cond_kwargs = {
                "resolution": torch.tensor([(height, width)] * bsz, device=hidden_states.device, dtype=hidden_states.dtype),
                "nframe": torch.tensor([frame] * bsz, device=hidden_states.device, dtype=hidden_states.dtype),
                "fps": fps
            }    
        else:
            added_cond_kwargs = {}
        
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs=added_cond_kwargs
        )

        encoder_hidden_states = self.caption_projection(self.caption_norm(encoder_hidden_states))
        if encoder_hidden_states_2 is not None and hasattr(self, 'clip_projection'):
            clip_embedding = self.clip_projection(encoder_hidden_states_2)
            encoder_hidden_states = torch.cat([clip_embedding, encoder_hidden_states], dim=1)

        hidden_states = rearrange(hidden_states, '(b f) l d->  b (f l) d', b=bsz, f=frame, l=len_frame).contiguous()

        encoder_hidden_states, attn_mask = self.prepare_attn_mask(encoder_attention_mask, encoder_hidden_states, q_seqlen=frame * len_frame)
        
        # Rotary positional embeddings
        rotary_pos_emb = self.rope(bsz, frame * mpu.get_context_parallel_world_size(), height, width, hidden_states.device)# s b 1 d
        if mpu.get_context_parallel_world_size() > 1:
            rotary_pos_emb = rotary_pos_emb.chunk(mpu.get_context_parallel_world_size(), dim=0)[mpu.get_context_parallel_rank()]

        with rng_context:
            if self.recompute_granularity == "full":
                hidden_states = self._checkpointed_forward(
                    hidden_states,
                    encoder_hidden_states,
                    timestep,
                    attn_mask,
                    rotary_pos_emb
                )
            else:
                for _, block in zip(self.global_layer_idx, self.transformer_blocks):
                    hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        timestep,
                        attn_mask,
                        rotary_pos_emb
                    )
        
        hidden_states = rearrange(hidden_states, 'b (f l) d -> (b f) l d', b=bsz, f=frame, l=len_frame)
        embedded_timestep = repeat(embedded_timestep, 'b d -> (b f) d', f=frame).contiguous()
        
        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        
        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        
        hidden_states = rearrange(hidden_states, 'n h w p q c -> n c h p w q')
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        output = rearrange(output, '(b f) c h w -> b f c h w', f=frame)

        if mpu.get_context_parallel_world_size() > 1:
            output = gather_forward_split_backward(output, mpu.get_context_parallel_group(), dim=1,
                                                        grad_scale='up')

        return output

    def _get_block(self, layer_number):
        return self.transformer_blocks[layer_number]

    def _checkpointed_forward(
            self,
            latents,
            prompt,
            timestep,
            prompt_mask,
            rotary_pos_emb):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_block(index)
                    x_ = layer(x_, *args, **kwargs)
                return x_
            return custom_forward

        if self.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_num = 0
            while layer_num < self.num_layers:
                latents = tensor_parallel.checkpoint(
                    custom(layer_num, layer_num + self.recompute_num_layers),
                    self.distribute_saved_activations,
                    latents,
                    prompt,
                    timestep,
                    prompt_mask,
                    rotary_pos_emb
                )
                layer_num += self.recompute_num_layers
        elif self.recompute_method == "block":
            for layer_num in range(self.num_layers):
                if layer_num < self.recompute_num_layers:
                    latents = tensor_parallel.checkpoint(
                        custom(layer_num, layer_num + 1),
                        self.distribute_saved_activations,
                        latents,
                        prompt,
                        timestep,
                        prompt_mask,
                        rotary_pos_emb
                    )
                else:
                    block = self._get_block(layer_num)
                    latents = block(
                        latents,
                        prompt,
                        timestep,
                        prompt_mask,
                        rotary_pos_emb
                    )
        else:
            raise ValueError("Invalid activation recompute method.")
        return latents


class StepVideoTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            attention_head_dim: int,
            norm_eps: float = 1e-5,
            attention_norm_type: str = "rmsnorm",
            attention_norm_elementwise_affine: bool = False,
            attention_norm_eps: float = 1e-6,
            fa_layout: str = "bsnd",
            rope=None
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn1 = ParallelAttention(
            query_dim=dim,
            key_dim=dim,
            num_attention_heads=dim // attention_head_dim,
            hidden_size=dim,
            use_qk_norm=True,
            norm_type=attention_norm_type,
            norm_elementwise_affine=attention_norm_elementwise_affine,
            norm_eps=attention_norm_eps,
            is_qkv_concat=True,
            attention_type=AttnType.self_attn,
            rope=rope,
            fa_layout=fa_layout
        )

        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn2 = ParallelAttention(
            query_dim=dim,
            key_dim=dim,
            num_attention_heads=dim // attention_head_dim,
            hidden_size=dim,
            use_qk_norm=True,
            norm_type=attention_norm_type,
            norm_elementwise_affine=attention_norm_elementwise_affine,
            norm_eps=attention_norm_eps,
            is_kv_concat=True,
            attention_type=AttnType.cross_attn,
            fa_layout=fa_layout,
            split_kv_in_forward=False
        )

        self.ff = FeedForward(
            dim=dim,
            activation_fn="gelu-approximate",
            bias=False
        )

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)

    def forward(
            self,
            q: torch.Tensor,
            kv: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            attn_mask=None,
            rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale_shift_table_expanded = self.scale_shift_table[None] + timestep.reshape(-1, 6, self.dim)
        chunks = scale_shift_table_expanded.chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (torch.clone(chunk) for chunk in chunks)

        scale_shift_q = self.norm1(q) * (1 + scale_msa) + shift_msa
        
        # self attention
        attn_q = self.attn1(
            scale_shift_q,
            input_layout="bsh",
            rotary_pos_emb=rotary_pos_emb
        )
        q = attn_q * gate_msa + q

        # cross attention
        attn_q = self.attn2(
            q,
            kv,
            attn_mask,
            input_layout="bsh"
        )
        q = attn_q + q
        scale_shift_q = self.norm2(q) * (1 + scale_mlp) + shift_mlp

        # feed forward
        ff_output = self.ff(scale_shift_q)
        q = ff_output * gate_mlp + q

        return q


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
            self,
            patch_size=64,
            in_channels=3,
            embed_dim=768,
            layer_norm=False,
            flatten=True,
            bias=True,
    ):
        super().__init__()

        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )

    def forward(self, latent):
        latent = self.proj(latent).to(latent.dtype)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)
        if self.layer_norm:
            latent = self.norm(latent)

        return latent


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if self.use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.nframe_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
            self.fps_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, resolution=None, nframe=None, fps=None):
        hidden_dtype = next(self.timestep_embedder.parameters()).dtype

        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            batch_size = timestep.shape[0]
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            nframe_emb = self.additional_condition_proj(nframe.flatten()).to(hidden_dtype)
            nframe_emb = self.nframe_embedder(nframe_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + resolution_emb + nframe_emb

            if fps is not None:
                fps_emb = self.additional_condition_proj(fps.flatten()).to(hidden_dtype)
                fps_emb = self.fps_embedder(fps_emb).reshape(batch_size, -1)
                conditioning = conditioning + fps_emb
        else:
            conditioning = timesteps_emb

        return conditioning


class AdaLayerNormSingle(nn.Module):
    r"""
        Norm layer adaptive layer norm single (adaLN-single).

        As proposed in PixArt-Alpha.

        Parameters:
            embedding_dim (`int`): The size of each embedding vector.
            use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False, time_step_rescale=1000):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 2, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

        self.time_step_rescale = time_step_rescale  # timestep usually in [0, 1], we rescale it to [0,1000] for stability

    def forward(
            self,
            timestep: torch.Tensor,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep * self.time_step_rescale, **added_cond_kwargs)

        out = self.linear(self.silu(embedded_timestep))

        return out, embedded_timestep