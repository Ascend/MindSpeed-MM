# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import check_model_inputs
from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4vMoeTextConfig, Glm4vMoeVisionConfig
from transformers.models.glm4v_moe.modeling_glm4v_moe import (
    Glm4vMoeForConditionalGeneration,
    Glm4vMoeCausalLMOutputWithPast,
    Glm4vMoeTextDecoderLayer,
    Glm4vMoeTextMLP,
    Glm4vMoeVisionModel,
    Glm4vMoeRMSNorm,
    Glm4vMoeTextTopkRouter,
    Glm4vMoeVisionBlock,
    Glm4vMoeTextModel,
    Glm4vMoeVisionEmbeddings,
)

from mindspeed_mm.fsdp.utils.register import model_register
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.distributed.context_parallel.communication import (
    gather_forward_split_backward,
    packed_data_gather_forward_split_backward_with_cp,
    packed_data_split_forward_gather_backward_with_cp,
    split_forward_gather_backward_with_cp,
)
from mindspeed_mm.fsdp.distributed.context_parallel.utils import cal_split_sizes_multi

if IS_NPU_AVAILABLE:
    import torch_npu
    from mindspeed_mm.fsdp.ops.moe_ops.gemm import fused_group_gemm


class Glm4vMoeTextExperts(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.num_experts = config.n_routed_experts
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.intermediate_size))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.intermediate_size, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]
        self.use_grouped_expert_matmul = getattr(config, "use_grouped_expert_matmul", True)

        ps = get_parallel_state()
        self.enable_ep_balance = getattr(config, "enable_ep_balance", False) and ps.is_ep_enable()
        if self.enable_ep_balance:
            from mindspeed_mm.fsdp.distributed.expert_parallel.ep_balance.ep_balance_strategy import EPBalanceStrategy

            self.ep_balance_strategy = EPBalanceStrategy(
                ep_group=ps.get_ep_group(),
                num_experts=self.num_experts,
                max_dup_experts_num=config.max_dup_experts_num,
            )

            def clear_ep_balance_plan(*args, **kwargs):
                self.ep_balance_strategy.planner.clear_record_planner_result()

            self.register_full_backward_hook(clear_ep_balance_plan)

    def _local_experts_weight(self):
        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
        return gate_up_proj, down_proj

    def forward(self, hidden_states, topk_weights, topk_indices):
        gate_up_proj, down_proj = self._local_experts_weight()
        if self.use_grouped_expert_matmul and IS_NPU_AVAILABLE and hidden_states.device.type == "npu":
            batch_size = hidden_states.shape[0]
            orig_dtype = hidden_states.dtype
            hidden_states = hidden_states.reshape(-1, self.hidden_size)
            permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
                hidden_states, topk_indices.to(torch.int32)
            )
            tokens_per_expert = torch.histc(
                topk_indices, bins=self.num_experts, min=0, max=self.num_experts
            )
            intermediate_hidden_states = fused_group_gemm(
                permuted_hidden_states, gate_up_proj, tokens_per_expert
            )
            intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
            output = fused_group_gemm(intermediate_activations, down_proj, tokens_per_expert)
            final_hidden_states = torch_npu.npu_moe_token_unpermute(
                output.to(topk_weights.dtype), row_ids_map, probs=topk_weights
            )
            return final_hidden_states.view(batch_size, -1, self.hidden_size).type(orig_dtype)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=self.num_experts).permute(2, 0, 1)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            with torch.no_grad():
                token_idx, weight_idx = torch.where(expert_mask[expert_idx])
            gate_up = hidden_states[token_idx] @ gate_up_proj[expert_idx]
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = (self.act_fn(gate) * up) @ down_proj[expert_idx]
            weighted_output = expert_output * topk_weights[token_idx, weight_idx, None]
            final_hidden_states.index_add_(0, token_idx, weighted_output)
        return final_hidden_states.type(hidden_states.dtype)

    def ep_forward(self, hidden_states, topk_weights, topk_indices, ep_group, ep_plan):
        orig_shape = hidden_states.shape
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        gate_up_proj, down_proj = self._local_experts_weight()

        from mindspeed_mm.fsdp.distributed.expert_parallel.ep_dispatcher import (
            ep_allgather_forward,
            ep_forward,
            ep_mc2_forward,
        )

        if self.enable_ep_balance:
            self.ep_balance_strategy.executor.register_backward_dup_experts_grad_acc_hook(
                gate_up_proj, name="fc1"
            )
            self.ep_balance_strategy.executor.register_backward_dup_experts_grad_acc_hook(
                down_proj, name="fc2"
            )
            if ep_plan.dispatcher != "alltoall":
                raise NotImplementedError("EP load balancing strategy currently only supports alltoall dispatch.")

        dispatchers = {
            "alltoall": ep_forward,
            "mc2": ep_mc2_forward,
            "allgather": ep_allgather_forward,
        }
        if ep_plan.dispatcher not in dispatchers:
            raise NotImplementedError(f"EP dispatcher {ep_plan.dispatcher} is not implemented for GLM-4.5V.")

        hidden_states = dispatchers[ep_plan.dispatcher](
            self.num_experts,
            topk_weights,
            topk_indices,
            hidden_states,
            fc1_weight=gate_up_proj,
            fc2_weight=down_proj,
            ep_group=ep_group,
            fused=ep_plan.use_npu_fused_ops,
            ep_balance_strategy=self.ep_balance_strategy if self.enable_ep_balance else None,
        )
        return hidden_states.view(*orig_shape).to(orig_dtype)


class Glm4vFusedMoeTextMoE(nn.Module):
    """MoE block: stacked 3D experts (+ EP via framework) + HF router + shared experts."""

    def __init__(self, config: Glm4vMoeTextConfig):
        super().__init__()
        self.config = config
        self.experts = Glm4vMoeTextExperts(config, intermediate_size=config.moe_intermediate_size)
        self.gate = Glm4vMoeTextTopkRouter(config)
        self.shared_experts = Glm4vMoeTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, topk_weights, topk_indices).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class Glm4vFusedMoeRMSNorm(Glm4vMoeRMSNorm):
    def forward(self, hidden_states):
        if IS_NPU_AVAILABLE:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
        return super().forward(hidden_states)


class Glm4vFusedMoeVisionEmbeddings(Glm4vMoeVisionEmbeddings):
    """NPU has no bicubic grid_sample yet; use bilinear as a temporary stand-in."""

    def forward(self, embeddings, lengths, image_shapes, h_coords, w_coords) -> torch.Tensor:
        if not IS_NPU_AVAILABLE:
            return super().forward(embeddings, lengths, image_shapes, h_coords, w_coords)

        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device
        h_coords, w_coords = h_coords.to(device), w_coords.to(device)
        if total_seq == 0:
            adapted_pos_embed = torch.empty(0, hidden_size, device=device, dtype=pos_embed_weight.dtype)
        else:
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(image_shapes, device=device, dtype=torch.long)
            orig_size = int(pos_embed_weight.shape[0] ** 0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            )
            target_h = torch.cat([image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]).to(device=device, dtype=torch.float32)
            target_w = torch.cat([image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]).to(device=device, dtype=torch.float32)
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)
            interpolated = F.grid_sample(pos_embed_2d, grid, mode="bilinear", align_corners=False, padding_mode="border")
            adapted_pos_embed = interpolated.squeeze(0).squeeze(-1).permute(1, 0).to(pos_embed_weight.dtype).to(embeddings.device)
        return embeddings + adapted_pos_embed


class Glm4vFusedMoeVisionBlock(Glm4vMoeVisionBlock):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.norm1 = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Glm4vFusedMoeVisionModel(Glm4vMoeVisionModel):
    config: Glm4vMoeVisionConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        self.embeddings = Glm4vFusedMoeVisionEmbeddings(config)
        self.blocks = nn.ModuleList([Glm4vFusedMoeVisionBlock(config) for _ in range(config.depth)])
        self.post_conv_layernorm = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_layernorm = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)

        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        sequence_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
        if IS_NPU_AVAILABLE:
            cu_seqlens = cu_seqlens.cpu()
        hidden_states = self.embeddings(
            hidden_states,
            sequence_lengths.tolist(),
            grid_thw,
            image_type_ids[:, 0],
            image_type_ids[:, 1],
        )

        ps = get_parallel_state()
        attention_kwargs = {}
        if ps.is_cp_enable():
            rotary_pos_emb = packed_data_split_forward_gather_backward_with_cp(
                rotary_pos_emb, dim=0, seq_lens=sequence_lengths
            )
            hidden_states = packed_data_split_forward_gather_backward_with_cp(
                hidden_states, dim=0, seq_lens=sequence_lengths
            )
            if ps.is_ring_enable():
                ring_split_sizes = cal_split_sizes_multi(sequence_lengths, ps.get_ring_group_size())
                cu_seqlens = F.pad(ring_split_sizes.cumsum(dim=1)[ps.get_ring_rank()], (1, 0), value=0)
                attention_kwargs["seq_split_lens"] = ring_split_sizes

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **attention_kwargs,
            )

        if ps.is_cp_enable():
            hidden_states = packed_data_gather_forward_split_backward_with_cp(
                hidden_states, dim=0, seq_lens=sequence_lengths
            )

        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)
        if ps.is_cp_enable():
            merged_sequence_lengths = sequence_lengths // self.spatial_merge_size**2
            hidden_states = packed_data_split_forward_gather_backward_with_cp(
                hidden_states, dim=0, seq_lens=merged_sequence_lengths
            )

        hidden_states = self.merger(hidden_states)
        if ps.is_cp_enable():
            hidden_states = packed_data_gather_forward_split_backward_with_cp(
                hidden_states, dim=0, seq_lens=merged_sequence_lengths
            )
        return hidden_states


class Glm4vFusedMoeTextDecoderLayer(Glm4vMoeTextDecoderLayer):
    def __init__(self, config: Glm4vMoeTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = Glm4vFusedMoeTextMoE(config)
        else:
            self.mlp = Glm4vMoeTextMLP(config)
        self.input_layernorm = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Glm4vFusedMoeTextModel(Glm4vMoeTextModel):
    config: Glm4vMoeTextConfig

    def __init__(self, config: Glm4vMoeTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Glm4vFusedMoeTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        text_position_ids = position_ids[0]

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        ps = get_parallel_state()
        if self.training and ps.is_cp_enable():
            position_ids = split_forward_gather_backward_with_cp(position_ids, dim=2)
            text_position_ids = split_forward_gather_backward_with_cp(text_position_ids, dim=1)
            inputs_embeds = split_forward_gather_backward_with_cp(inputs_embeds, dim=1)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@model_register.register("glm4v_moe")
class Glm4vMoeFSDPForConditionalGeneration(Glm4vMoeForConditionalGeneration):
    """Pure-FSDP2 GLM-4.5V: native HF model with NPU-fused vision/text towers."""

    def __init__(self, config):
        super().__init__(config)
        self.model.visual = Glm4vFusedMoeVisionModel._from_config(config.vision_config)
        self.model.language_model = Glm4vFusedMoeTextModel._from_config(config.text_config)
        self.enable_chunk_loss = getattr(config.text_config, "enable_chunk_loss", False)
        self.enable_dynamic_chunk_loss = getattr(config.text_config, "enable_dynamic_chunk_loss", False)
        self.post_init()

    @staticmethod
    def overwrite_transformer_config(transformer_config, model_args, feature_args=None):
        transformer_config.text_config.use_grouped_expert_matmul = getattr(
            model_args, "use_grouped_expert_matmul", True
        )
        loss_cfg = getattr(feature_args, "loss_cfg", None)
        transformer_config.text_config.router_aux_loss_coef = (
            getattr(loss_cfg, "router_aux_loss_coef", 0.0) if loss_cfg else 0.0
        )
        transformer_config.text_config.enable_chunk_loss = getattr(feature_args, "enable_chunk_loss", False)
        transformer_config.text_config.enable_dynamic_chunk_loss = getattr(
            feature_args, "enable_dynamic_chunk_loss", False
        )
        transformer_config.text_config.enable_ep_balance = getattr(feature_args, "enable_ep_balance", False)
        ep_balance_plan = getattr(feature_args, "ep_balance_plan", None)
        transformer_config.text_config.max_dup_experts_num = getattr(
            ep_balance_plan, "max_dup_experts_num", 2
        )
        return transformer_config

    @property
    def visual(self):
        return self.model.visual

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Glm4vMoeCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        if getattr(self, "enable_chunk_loss", False) or getattr(self, "enable_dynamic_chunk_loss", False):
            logits = None
            loss = self.lm_head(hidden_states[:, slice_indices, :], self.loss_function)
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            loss = None
            if labels is not None:
                loss = self.loss_function(
                    logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
                )

        ps = get_parallel_state()
        if loss is not None and ps.is_cp_enable():
            loss = gather_forward_split_backward(loss.unsqueeze(0), ps.get_cp_group(), dim=0)
            loss = loss.sum()

        return Glm4vMoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(outputs, "rope_deltas", None),
        )


if IS_NPU_AVAILABLE:
    from .npu_patch import apply_glm4v_npu_patch

    apply_glm4v_npu_patch()
