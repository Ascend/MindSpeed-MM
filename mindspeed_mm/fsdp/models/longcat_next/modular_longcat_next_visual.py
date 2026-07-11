from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
from einops import rearrange
from flash_attn import flash_attn_varlen_func
from safetensors.torch import load_file
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2RMSNorm,
    Qwen2_5_VisionTransformerPretrainedModel,
)
from transformers.utils import logging

from .image_refiner import (
    ImageRefinerContainer,
    RefinerImageProcessor,
    RefinerPipeline,
    de_transform,
    tensor2pil,
)
from .refiner_modules import FlowMatchEulerDiscreteScheduler

logger = logging.get_logger(__name__)


def uniform_init(*shape):
    t = torch.zeros(shape)
    nn.init.kaiming_uniform_(t)
    return t


class VQEmbedding(nn.Module):
    """VQ embedding module with ema update."""

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5, init_std=0.02):
        super().__init__()

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed
        self.init_std = init_std

        if not self.ema:
            raise AssertionError("self.ema should be True or not-none")
        embed = uniform_init(n_embed + 1, embed_dim).to(torch.float32)
        self.embed = nn.Parameter(embed)
        self.embed_ema = nn.Parameter(embed[:-1, :].clone())
        self.cluster_size_ema = nn.Parameter(torch.ones(n_embed))
        del embed
        _ = [p.requires_grad_(False) for p in self.parameters()]

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.embed[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        if not inputs_shape[-1] == embed_dim:
            raise AssertionError(f"error: inputs_shape[-1] != embed_dim")

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, h, w, n_embed or n_embed+1]
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)  # [B, h, w, n_embed or n_embed+1]
        embed_idxs = distances.argmin(dim=-1)  # use padding index or not

        return embed_idxs

    @autocast('cuda', enabled=True, dtype=torch.float32)
    @torch.no_grad()
    def forward(self, inputs):
        if inputs.dtype != torch.float32:
            inputs = inputs.to(torch.float32)
        embed_idxs = self.find_nearest_embedding(inputs)
        embeds = self.embed[embed_idxs]
        return embeds, embed_idxs


class RQBottleneck(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.
    Arguments:
        latent_shape (Tuple[int, int, int]): the shape of latents, denoted (H, W, D)
        code_shape (Tuple[int, int, int]): the shape of codes, denoted (h, w, d)
        n_embed (int, List, or Tuple): the number of embeddings (i.e., the size of codebook)
            If isinstance(n_embed, int), the sizes of all codebooks are same.
        shared_codebook (bool): If True, codebooks are shared in all location. If False,
            uses separate codebooks along the ``depth'' dimension. (default: False)
        restart_unused_codes (bool): If True, it randomly assigns a feature vector in the curruent batch
            as the new embedding of unused codes in training. (default: True)
    """

    def __init__(self,
                 latent_shape,
                 code_shape,
                 n_embed,
                 decay=0.99,
                 shared_codebook=False,
                 restart_unused_codes=True,
                 commitment_loss='cumsum'
                 ):
        super().__init__()

        if not len(code_shape) == len(latent_shape) == 3:
            raise ValueError("incompatible code shape or latent shape")
        if any([y % x != 0 for x, y in zip(code_shape[:2], latent_shape[:2])]):
            raise ValueError("incompatible code shape or latent shape")

        # residual quantization does not divide feature dims for quantization.
        embed_dim = np.prod(latent_shape[:2]) // np.prod(code_shape[:2]) * latent_shape[2]

        self.latent_shape = torch.Size(latent_shape)
        self.code_shape = torch.Size(code_shape)
        self.shape_divisor = torch.Size([latent_shape[i] // code_shape[i] for i in range(len(latent_shape))])

        self.shared_codebook = shared_codebook
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError("Shared codebooks are incompatible \
                                    with list types of momentums or sizes: Change it into int")

        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed if isinstance(n_embed, Iterable) else [n_embed for _ in range(self.code_shape[-1])]
        self.decay = decay if isinstance(decay, Iterable) else [decay for _ in range(self.code_shape[-1])]
        if not len(self.n_embed) == self.code_shape[-1]:
            raise AssertionError("error: len(self.n_embed) != self.code_shape[-1]")
        if not len(self.decay) == self.code_shape[-1]:
            raise AssertionError("error: len(self.decay) != self.code_shape[-1]")

        if self.shared_codebook:
            codebook0 = VQEmbedding(self.n_embed[0],
                                    embed_dim,
                                    decay=self.decay[0],
                                    restart_unused_codes=restart_unused_codes,
                                    ).to(torch.float32)
            self.codebooks = nn.ModuleList([codebook0 for _ in range(self.code_shape[-1])])
        else:
            codebooks = [VQEmbedding(self.n_embed[idx],
                                     embed_dim,
                                     decay=self.decay[idx],
                                     restart_unused_codes=restart_unused_codes,
                                     ).to(torch.float32) for idx in range(self.code_shape[-1])]
            self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss

    def to_code_shape(self, x):
        (B, H, W, D) = x.shape
        (rH, rW, _) = self.shape_divisor

        x = x.reshape(B, H // rH, rH, W // rW, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H // rH, W // rW, -1)

        return x

    def to_latent_shape(self, x):
        (B, h, w, _) = x.shape
        (_, _, D) = self.latent_shape
        (rH, rW, _) = self.shape_divisor

        x = x.reshape(B, h, w, rH, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, h * rH, w * rW, D)

        return x

    def quantize(self, x):
        r"""
        Return list of quantized features and the selected codewords by the residual quantization.
        The code is selected by the residuals between x and quantized features by the previous codebooks.
        Arguments:
            x (Tensor): bottleneck feature maps to quantize.
        Returns:
            quant_list (list): list of sequentially aggregated and quantized feature maps by codebooks.
            codes (LongTensor): codewords index, corresponding to quants.
        Shape:
            - x: (B, h, w, embed_dim)
            - quant_list[i]: (B, h, w, embed_dim)
            - codes: (B, h, w, d)
        """
        B, h, w, embed_dim = x.shape
        ori_dtype = x.dtype
        x = x.to(torch.float32)
        self.codebooks = self.codebooks.to(torch.float32)

        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.code_shape[-1]):
            quant, code = self.codebooks[i](residual_feature)
            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)
            quant_list.append(aggregated_quants.clone().to(dtype=ori_dtype))
            code_list.append(code.unsqueeze(-1))

        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def forward(self, x):
        x_reshaped = self.to_code_shape(x)
        # 强制使用float32精度来执行
        quant_list, codes = self.quantize(x_reshaped)

        commitment_loss = self.compute_commitment_loss(x_reshaped, quant_list)
        quants_trunc = self.to_latent_shape(quant_list[-1])
        quants_trunc = x + (quants_trunc - x).detach()
        codebook_usage = 0
        # (vq_loss, commit_loss, entropy_loss, codebook_usage) # 格式对齐
        codebook_loss = [0, commitment_loss, 0, codebook_usage]

        return quants_trunc, codebook_loss, codes

    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []

        for idx, quant in enumerate(quant_list):
            partial_loss = (x - quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)

        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss


class Qwen2_5_VisionRotaryEmbedding_Modified(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))

    def forward(self, seqlen: int, device: torch.device) -> torch.Tensor:
        self.inv_freq = self.inv_freq.to(device)
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class VisualEncoder(Qwen2_5_VisionTransformerPretrainedModel):

    def __init__(self, config):
        config._attn_implementation = 'flash_attention_2'
        super().__init__(config)
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding_Modified(config.hidden_size // config.num_heads // 2)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        self.merge_size = config.merge_size if hasattr(config, 'merge_size') else 2
        del self.merger  # register visual.merger in visual_bridge_model

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.down_proj.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.down_proj.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size, device=grid_thw.device)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
            self,
            pixel_values: torch.Tensor,
            grid_thw: torch.Tensor,
            require_window_index: bool = False,
    ):
        '''
        pixel_values.shape=[NumOfPatches, 1176]
        grid_thw.shape=[NumOfSamples, 3]. [grid_t,grid_h,grid_w]
        '''
        hidden_states = pixel_values.to(torch.bfloat16)
        grid_thw = grid_thw.to(pixel_values.device)

        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(blk.__call__, hidden_states, cu_seqlens_now, None,
                                                                  position_embeddings)
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    position_embeddings=position_embeddings,
                )

        if require_window_index:
            return hidden_states, window_index
        return hidden_states


class OmniVisualBridge(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.merge_size = self.config.merge_size if hasattr(self.config, 'merge_size') else 2
        self.hidden_size = self.config.hidden_size * (self.merge_size ** 2)
        self.window_index = self.config.window_size
        self.ln_q = Qwen2RMSNorm(self.config.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.config.out_hidden_size),
        )

    def forward(self, x: torch.Tensor, window_index) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices, :]

        return x


class VisualQuantizer(nn.Module):
    def __init__(self, quantizer_config):
        super().__init__()

        self.config = quantizer_config
        self.depth = self.config.depth
        self.decay = self.config.decay
        self.codebook_size = self.config.codebook_size
        self.codebook_dim = self.config.codebook_dim
        self.shared_codebook = self.config.shared_codebook
        self.restart_unused_codes = self.config.restart_unused_codes
        self.in_channels = self.config.in_channels

        self.vq_loss_ratio = self.config.vq_loss_ratio
        self.entropy_loss_ratio = self.config.entropy_loss_ratio
        self.commit_loss_ratio = self.config.commit_loss_ratio

        code_h_w = int(448 / 14)
        latent_shape = [code_h_w, code_h_w, self.codebook_dim]
        code_shape = [code_h_w, code_h_w, self.depth]

        self.quantize = RQBottleneck(
            latent_shape=latent_shape,
            code_shape=code_shape,
            n_embed=self.codebook_size,
            decay=self.decay,
            shared_codebook=self.shared_codebook,
            restart_unused_codes=self.restart_unused_codes,
        )

        if self.config.quant_conv:
            self.quant_conv = nn.Sequential(
                nn.LayerNorm(self.in_channels),
                nn.Linear(self.in_channels, self.in_channels),
                nn.GELU(),
                nn.Linear(self.in_channels, self.codebook_dim)
            )
        else:
            self.quant_conv = None

    def encode(self, x):
        L, D = x.shape
        to_qnt_feat = x.clone()
        to_qnt_feat = to_qnt_feat.unsqueeze(0)  # [L, D] -> [1, L, D]
        N = 1

        if self.quant_conv is not None:
            to_qnt_feat = self.quant_conv(to_qnt_feat)

        # quantizer needs nchw format. N,L,d -> N,1,L,d -> N,d,1,L
        to_qnt_feat = to_qnt_feat.reshape(N, 1, L, self.codebook_dim).permute(0, 3, 1, 2)
        if self.config.quantizer_type == "rq":
            to_qnt_feat = to_qnt_feat.permute(0, 2, 3, 1).contiguous()  # N,d,1,L -> N,1,L,d
            quant, emb_loss, info = self.quantize(to_qnt_feat)
            info = info.reshape(-1, info.shape[-1])  # n,h,w,lv -> n*h*w,lv
            info = [None, None, info]
            quant = quant.permute(0, 3, 1, 2).contiguous()  # N,1,L,d -> N,d,1,L
        else:
            quant, emb_loss, info = self.quantize(to_qnt_feat)
        return quant, emb_loss, info, x.detach()

    def forward(self, x):
        quant, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings,
                                                                      min_encoding_indices), align_feature = \
            self.encode(x)
        return min_encoding_indices


class MLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.visual_embedding_layer_intermediate_size,
            hidden_act=config.visual_embedding_layer_hidden_act,
        )
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.pre_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VisualEmbeddingBridge(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_buffer = DecoderLayer(config)

    def forward(self, embeding):
        return self.pre_buffer(embeding)


class VisualVQBridge(nn.Module):
    def __init__(self, visual_config):
        super().__init__()
        self.bridge = OmniVisualBridge(visual_config)
        self.quantizer = VisualQuantizer(visual_config.vq_config)

    def forward(
            self,
            visual_embed: torch.Tensor,
            window_index: torch.Tensor,
    ):
        visual_embed = self.bridge(visual_embed, window_index)
        indices = self.quantizer(visual_embed)
        return indices


class LongcatNextVisualTokenizer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual_model = VisualEncoder(config.visual_config)
        self.visual_bridge_model = VisualVQBridge(config.visual_config)
        self.visual_embedding_layer = VisualEmbeddingBridge(config)
        self.image_decoder = None
        self._refiner_pipeline = None

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor, visual_grid_thw: torch.Tensor):
        visual_embed, window_index = self.visual_model(pixel_values, grid_thw=visual_grid_thw,
                                                       require_window_index=True)
        indices = self.visual_bridge_model(visual_embed, window_index)
        return indices

    @torch.no_grad()
    def lazy_decode_and_save(self, visual_ids, tokens_h, tokens_w, save_path):
        device = next(self.parameters()).device
        if self.image_decoder is None:
            print("lazy load image_decoder / image_refiner / _refiner_pipeline ...")
            vdc = self.config.visual_config.visual_decoder_config
            self.image_decoder = VisionTransformerDecoder.from_pretrained(
                vdc.image_decoder_config,
                vdc.weight_path,
            ).to(device=device, dtype=torch.bfloat16)
            image_refiner = ImageRefinerContainer.from_pretrained(vdc, vdc.weight_path).to(device=device,
                                                                                           dtype=torch.bfloat16)

            sc = vdc.scheduler_config
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=sc.num_train_timesteps,
                dynamic_time_shift=sc.dynamic_time_shift)
            self._refiner_pipeline = RefinerPipeline(
                vae=image_refiner.vae,
                transformer=image_refiner.base_transformer,
                scheduler=scheduler,
                cond_proj=image_refiner.cond_proj,
            )
            self._refiner_pipeline.set_progress_bar_config(disable=False)

        data = torch.as_tensor(visual_ids, dtype=torch.long)
        if data.ndim == 1:
            data = data.view(-1, len(self.config.visual_config.vq_config.codebook_sizes))
        if data.ndim == 2:
            data = data.unsqueeze(0)
        batch_size = data.shape[0]

        quant_features = None
        for idx in range(len(self.config.visual_config.vq_config.codebook_sizes)):
            embed = self.visual_bridge_model.quantizer.quantize.codebooks[idx].embed
            feat = embed[data[..., idx].to(embed.device)]
            quant_features = feat if quant_features is None else quant_features + feat
        quant_features = quant_features.to(device)

        # tokens_h/tokens_w are the merged grid; expand to the full (unmerged) grid
        s = self.image_decoder.spatial_merge_size
        grid_thw_list = [(1, tokens_h * s, tokens_w * s)]
        grid_thw_batch = list(grid_thw_list) * batch_size

        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]

        emb_2d = quant_features.reshape(-1, quant_features.shape[-1]).contiguous()
        device_type = "cuda" if str(device).startswith("cuda") else str(device)
        with torch.amp.autocast(device_type=device_type, enabled=True, dtype=torch.float32):
            decoder_out = self.image_decoder(emb_2d, grid_thw_batch, return_pixel_features=False)

        decoded_tensors = decoder_out.get("images") or []
        decoded_images = [tensor2pil(t, image_mean, image_std) for t in decoded_tensors]
        decoded_path = save_path.replace(".png", "_decoded.png")

        ref_input = []
        for t in decoded_tensors:
            img_01 = de_transform(t, mean=image_mean, std=image_std, rescale_factor=1 / 255)
            img_norm = RefinerImageProcessor.normalize(img_01)
            ref_input.append(img_norm.squeeze(0).to(device))

        generators = [torch.Generator(device=device).manual_seed(42 + b) for b in range(batch_size)]
        out = self._refiner_pipeline(
            encoder_hidden_states=quant_features,
            grid_thw_list=grid_thw_list,
            image=ref_input,
            generator=generators[0] if batch_size == 1 else generators,
            output_type="pil",
            return_dict=True,
        )
        refined_images = out.images
        refined_path = save_path.replace(".png", "_refined.png")
        refined_images[0].save(refined_path)

        return [refined_path]


# ---------------------------------------------------------------------------
# Vision Transformer Decoder
# ---------------------------------------------------------------------------

def _rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRoPE2D(nn.Module):
    """2D Rotary Position Embedding for Q/K in vision decoder attention."""

    def __init__(self, theta: float = 10000.0):
        super().__init__()
        self.theta = theta

    def _rope_half(self, x_half, pos_1d, theta):
        BH, T, d_half = x_half.shape
        idx = torch.arange(0, d_half, 2, device=x_half.device, dtype=torch.float32)
        inv_freq = (1.0 / (theta ** (idx / d_half))).to(x_half.dtype)
        angles = pos_1d.to(x_half.dtype)[:, None] * inv_freq[None, :]
        cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1).unsqueeze(0)
        sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1).unsqueeze(0)
        return x_half * cos + _rotate_half(x_half) * sin

    def forward(self, x, positions_2d):
        d_half = x.shape[-1] // 2
        x_y = self._rope_half(x[:, :, :d_half], positions_2d[:, 0], self.theta)
        x_x = self._rope_half(x[:, :, d_half:], positions_2d[:, 1], self.theta)
        return torch.cat([x_y, x_x], dim=-1)


class VisionAttention(nn.Module):
    """Multi-headed attention with 2D RoPE + FlashAttention varlen."""

    def __init__(self, config, rope=None, rope_shift=0):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, num_heads={self.num_heads})"
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.subln = config.subln
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=getattr(config, "k_bias", True))
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=getattr(config, "v_bias", True))
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=getattr(config, "q_bias", True))
        self.inner_attn_ln = Qwen2RMSNorm(self.embed_dim, eps=config.layer_norm_eps) if config.subln else nn.Identity()
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.rope = rope
        self.rope_shift = int(rope_shift)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _maybe_flash_attention(self, query_states, key_states, value_states, seq_lens, training):
        if not (query_states.is_cuda and (query_states.dtype in (torch.float16, torch.bfloat16, torch.float32))):
            return None
        if seq_lens is None:
            return None
        try:
            BxH, T, hd = query_states.shape
            H = self.num_heads
            if not BxH % H == 0:
                raise AssertionError("error: BxH % H != 0")
            B = BxH // H
            if int(seq_lens.sum().item()) != T:
                return None
            q = query_states.view(B, H, T, hd).transpose(1, 2).reshape(-1, H, hd).contiguous()
            k = key_states.view(B, H, T, hd).transpose(1, 2).reshape(-1, H, hd).contiguous()
            v = value_states.view(B, H, T, hd).transpose(1, 2).reshape(-1, H, hd).contiguous()
            cu_q = torch.zeros(seq_lens.numel() + 1, dtype=torch.int32, device=seq_lens.device)
            cu_q[1:] = torch.cumsum(seq_lens.to(torch.int32), dim=0)
            cu_k = cu_q
            max_seqlen = int(seq_lens.max().item())
            orig_dtype = q.dtype
            use_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
            if q.dtype != use_dtype:
                q = q.to(use_dtype)
                k = k.to(use_dtype)
                v = v.to(use_dtype)
            out = flash_attn_varlen_func(
                q, k, v, cu_q, cu_k, max_seqlen, max_seqlen,
                dropout_p=self.dropout if training else 0.0,
                softmax_scale=None, causal=False, return_attn_probs=False
            )
            if out.dtype != orig_dtype:
                out = out.to(orig_dtype)
            return out.view(B, -1, H, hd).transpose(1, 2).contiguous().view(B * H, T, hd)
        except Exception:
            return None

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
            positions_2d: Optional[torch.Tensor] = None,
            seq_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = self._shape(query_states, tgt_len, bsz).view(bsz * self.num_heads, tgt_len, self.head_dim)
        key_states = self._shape(key_states, tgt_len, bsz).view(bsz * self.num_heads, tgt_len, self.head_dim)
        value_states = self._shape(value_states, tgt_len, bsz).view(bsz * self.num_heads, tgt_len, self.head_dim)
        if self.rope is not None and positions_2d is not None:
            if self.rope_shift > 0:
                q_pref = query_states[:, :self.rope_shift, :]
                k_pref = key_states[:, :self.rope_shift, :]
                q_rot = self.rope(query_states[:, self.rope_shift:, :], positions_2d[self.rope_shift:])
                k_rot = self.rope(key_states[:, self.rope_shift:, :], positions_2d[self.rope_shift:])
                query_states = torch.cat([q_pref, q_rot], dim=1).type_as(value_states)
                key_states = torch.cat([k_pref, k_rot], dim=1).type_as(value_states)
            else:
                query_states = self.rope(query_states, positions_2d).type_as(value_states)
                key_states = self.rope(key_states, positions_2d).type_as(value_states)
        attn_output = self._maybe_flash_attention(
            query_states, key_states, value_states, seq_lens=seq_lens, training=self.training
        )
        if attn_output is not None:
            attn_weights_reshaped = None
        else:
            src_len = key_states.size(1)
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
            if causal_attention_mask is not None:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            if attention_mask is not None:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            if output_attentions:
                attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            else:
                attn_weights_reshaped = None
            attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.inner_attn_ln(attn_output)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped


class VisionSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.w2 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.w3 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.SiLU()
        self.ffn_ln = Qwen2RMSNorm(self.intermediate_size, eps=config.layer_norm_eps) if config.subln else nn.Identity()

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act_fn(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        return x


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_ln = Qwen2RMSNorm(config.intermediate_size,
                                   eps=config.layer_norm_eps) if config.subln else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class VisionEncoderLayer(nn.Module):
    def __init__(self, config, rope=None, rope_shift=0):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = VisionAttention(config, rope=rope, rope_shift=rope_shift)
        self.layer_norm1 = Qwen2RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = VisionSwiGLU(config) if config.swiglu else VisionMLP(config)
        self.layer_norm2 = Qwen2RMSNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            causal_attention_mask: Optional[torch.Tensor],
            output_attentions: Optional[bool] = False,
            positions_2d: Optional[torch.Tensor] = None,
            seq_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            positions_2d=positions_2d,
            seq_lens=seq_lens,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class VisionEncoder(nn.Module):
    def __init__(self, config, rope=None, rope_shift=0):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [VisionEncoderLayer(config, rope=rope, rope_shift=rope_shift) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    def forward(
            self,
            inputs_embeds: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            positions_2d: Optional[torch.Tensor] = None,
            seq_lens: Optional[torch.Tensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = True if return_dict is None else return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds

        for layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                def custom_forward(hs, attn, causal, pos2d, seqlens):
                    return layer(
                        hs,
                        attention_mask=attn,
                        causal_attention_mask=causal,
                        output_attentions=False,
                        positions_2d=pos2d,
                        seq_lens=seqlens,
                    )[0]

                hidden_states = self._gradient_checkpointing_func(
                    custom_forward,
                    hidden_states,
                    attention_mask if attention_mask is not None else torch.tensor(0., device=hidden_states.device),
                    causal_attention_mask if causal_attention_mask is not None else torch.tensor(0.,
                                                                                                 device=hidden_states.device),
                    positions_2d,
                    seq_lens if seq_lens is not None else torch.tensor([], device=hidden_states.device),
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                    positions_2d=positions_2d,
                    seq_lens=seq_lens,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class PatchUnMerger(nn.Module):
    """Learnable inverse of Qwen2_5_VLPatchMerger."""

    def __init__(self, dim, context_dim, spatial_merge_size=2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.context_dim = context_dim
        hidden = context_dim * (spatial_merge_size ** 2)
        self.ln_q = Qwen2RMSNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, x):
        x = self.mlp(self.ln_q(x))
        return x.view(x.shape[0] * (self.spatial_merge_size ** 2), self.context_dim)


def restore_spatial_structure_and_convert_to_images(patches, grid_thw_list, patch_size,
                                                    channel_dim=3, temporal_patch_size=2, merge_size=2):
    """Convert decoder pixel features back to image tensors [3, H, W]."""
    if isinstance(patches, tuple):
        patches = patches[0]
    image_tensors = []
    ptr = 0
    for grid in grid_thw_list:
        gt, gh, gw = (int(x) for x in (grid if not isinstance(grid, torch.Tensor) else grid.tolist()))
        n = gt * gh * gw
        chunk = patches[ptr:ptr + n]
        ptr += n
        r = chunk.reshape(gt, gh // merge_size, gw // merge_size, merge_size, merge_size,
                          channel_dim, temporal_patch_size, patch_size, patch_size)
        r = r.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
        image_tensors.append(r.reshape(gt * temporal_patch_size, channel_dim, gh * patch_size, gw * patch_size)[0])
    return image_tensors


class VisionTransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.codebook_dim = config.codebook_dim
        self.temporal_patch_size = config.temporal_patch_size

        self.rope2d = VisionRoPE2D(theta=10000.0)
        self.post_quant_conv = nn.Linear(self.codebook_dim, self.embed_dim)
        self.post_quant_norm = Qwen2RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.patch_unmerger = PatchUnMerger(self.embed_dim, self.embed_dim, self.spatial_merge_size)
        self.norm_in = Qwen2RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.encoder = VisionEncoder(config, rope=self.rope2d, rope_shift=0)
        self.norm_out = Qwen2RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.decoder_head = nn.Sequential(
            nn.Linear(self.embed_dim, config.intermediate_size), nn.GELU(),
            nn.Linear(config.intermediate_size, 3 * self.patch_size * self.patch_size * self.temporal_patch_size),
        )

    @classmethod
    def from_pretrained(cls, config, model_path: str):
        """Load a pretrained model from a checkpoint."""
        model = cls(config)
        weight_dict = load_file(model_path, device="cpu")
        model.load_state_dict(
            {k.removeprefix("image_decoder."): v for k, v in weight_dict.items() if k.startswith("image_decoder.")},
            strict=True)
        model.eval()
        return model

    def _build_2d_positions(self, grid_thw_list):
        pos_list = []
        for (t, gh, gw) in grid_thw_list:
            for _ in range(int(t)):
                for y in range(int(gh)):
                    for x in range(int(gw)):
                        pos_list.append([y, x])
        return torch.tensor(pos_list, dtype=torch.long)

    def _build_attention_mask(self, grid_thw_list, device, dtype, B, num_heads):
        counts = [int(t) * int(h) * int(w) for (t, h, w) in grid_thw_list]
        L = sum(counts)
        mask = torch.zeros((B, num_heads, L, L), device=device, dtype=dtype)
        s = 0
        for c in counts:
            e = s + c
            if s > 0:
                mask[:, :, s:e, :s] = float("-inf")
            if e < L:
                mask[:, :, s:e, e:] = float("-inf")
            s = e
        return mask

    def forward(self, embeddings, grid_thw, return_pixel_features=False, return_last_latent=False):
        device = embeddings.device
        grid_thw_list = ([(int(t), int(h), int(w)) for t, h, w in grid_thw.detach().cpu().numpy()]
                         if isinstance(grid_thw, torch.Tensor) else list(grid_thw))

        if embeddings.shape[-1] == self.codebook_dim:
            embeddings = self.post_quant_conv(embeddings)
            embeddings = self.post_quant_norm(embeddings)

        unmerged = self.patch_unmerger(embeddings)
        if unmerged.dim() == 2:
            unmerged = unmerged.unsqueeze(0)
        B, L, D = unmerged.shape
        hidden_states = self.norm_in(unmerged)

        positions_2d = self._build_2d_positions(grid_thw_list).to(device)
        seq_lens = torch.tensor([int(t) * int(h) * int(w) for (t, h, w) in grid_thw_list],
                                device=device, dtype=torch.int32)
        if not positions_2d.shape[0] == L:
            raise AssertionError(f"error: positions_2d {positions_2d.shape[0]} != L {L}")

        last_latent = hidden_states.detach().squeeze(0) if return_last_latent else None
        enc_out = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            positions_2d=positions_2d,
            seq_lens=seq_lens,
        )
        hidden_states = enc_out.last_hidden_state

        hidden_states = self.norm_out(hidden_states)
        pixel_features = self.decoder_head(hidden_states).squeeze(0)

        out_imgs = (None if return_pixel_features else
                    restore_spatial_structure_and_convert_to_images(
                        pixel_features, grid_thw_list, self.patch_size,
                        temporal_patch_size=self.temporal_patch_size, merge_size=self.spatial_merge_size))
        ret = {"images": out_imgs, "pixel_features": pixel_features}
        if last_latent is not None:
            ret["last_latent"] = last_latent
        return ret
