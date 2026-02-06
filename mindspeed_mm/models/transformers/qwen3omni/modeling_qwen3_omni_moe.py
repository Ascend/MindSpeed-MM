# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
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

from typing import Optional, Union

import transformers
# support newer transformers for qwen3omni
try:
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerConfig,
        Qwen3OmniMoeTextConfig,
        Qwen3OmniMoeConfig
    )
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        load_balancing_loss_func,
        Qwen3OmniMoeThinkerCausalLMOutputWithPast,
        Qwen3OmniMoeThinkerTextDecoderLayer,
        Qwen3OmniMoeThinkerTextAttention,
        Qwen3OmniMoeThinkerTextSparseMoeBlock,
        Qwen3OmniMoeThinkerTextRMSNorm,
        Qwen3OmniMoeThinkerTextRotaryEmbedding
    )
    has_qwen3omni_support = True
except ImportError:
    Qwen3OmniMoeThinkerConfig = None
    Qwen3OmniMoeTextConfig = None
    Qwen3OmniMoeConfig = None
    load_balancing_loss_func = None
    Qwen3OmniMoeThinkerCausalLMOutputWithPast = None
    Qwen3OmniMoeThinkerTextDecoderLayer = None
    Qwen3OmniMoeThinkerTextAttention = None
    Qwen3OmniMoeThinkerTextSparseMoeBlock = None
    Qwen3OmniMoeThinkerTextRMSNorm = None
    Qwen3OmniMoeThinkerTextRotaryEmbedding = None
    has_qwen3omni_support = False
from transformers.generation import GenerationMixin
from transformers.activations import ACT2FN
from transformers.utils import auto_docstring, can_return_tuple
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils.generic import OutputRecorder, check_model_inputs
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache


import torch
import torch_npu
from torch import nn
import torch.nn.functional as F

from mindspeed_mm.models.common.gmm import npu_group_gemm
from mindspeed_mm.utils.async_offload import async_save_on_cpu
from .modules import Qwen3OmniLMHead


@auto_docstring
class Qwen3OmniMoePreTrainedModel(PreTrainedModel):
    config: Qwen3OmniMoeConfig
    base_model_prefix = "model"
    input_modalities = ["image", "video", "audio", "text"]
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3OmniMoeDecoderLayer", "Qwen3OmniMoeVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


class Qwen3OmniMoeThinkerTextExperts(nn.ModuleList):
    """
    ModuleList of experts.
    """

    def __init__(self, config: Qwen3OmniMoeThinkerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size \
            if config.moe_intermediate_size is None else config.moe_intermediate_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))
        self.act_fn = ACT2FN[config.hidden_act]


    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size * sequence_length, hidden_dim)
            selected_experts: (batch_size * sequence_length, top_k)
            routing_weights: (batch_size * sequence_length, top_k)
        Returns:
            (batch_size * sequence_length, hidden_dim)
        """
        # Fixes the memory reorganization problem triggered when fast host dispatch
        # and tensor multi-stream reuse occur simultaneously.
        torch.npu.synchronize()

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, top_k_index.to(torch.int32))
        tokens_per_expert = torch.histc(top_k_index, bins=self.num_experts, min=0, max=self.num_experts)
        intermediate_hidden_states = npu_group_gemm(permuted_hidden_states, self.gate_up_proj, tokens_per_expert)
        intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
        output = npu_group_gemm(intermediate_activations, self.down_proj, tokens_per_expert)
        next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=top_k_weights)
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states
    

@auto_docstring(
    custom_intro=(
        "Text part of Qwen3OmniMoeThinker, "
        "not a pure text-only model, as DeepStack integrates visual features into the early hidden states."
    )
)
class Qwen3OmniMoeThinkerTextModel(Qwen3OmniMoePreTrainedModel):
    config: Qwen3OmniMoeTextConfig
    _no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]
    config_class = Qwen3OmniMoeTextConfig
    _can_record_outputs = {
        "hidden_states": Qwen3OmniMoeThinkerTextDecoderLayer,
        "attentions": Qwen3OmniMoeThinkerTextAttention,
        "router_logits": OutputRecorder(Qwen3OmniMoeThinkerTextSparseMoeBlock, index=1),
    }

    def __init__(self, config: Qwen3OmniMoeTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3OmniMoeThinkerTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3OmniMoeThinkerTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(config)
        self.gradient_checkpointing = False

        if config.activation_offload:
            self.swap_stream = torch.npu.Stream()

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            if self.config.activation_offload:
                with async_save_on_cpu(
                    h2d_stream=self.swap_stream,
                    d2h_stream=self.swap_stream,
                    block_idx=layer_idx,
                    depth=len(self.layers),
                    custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr(),
                    prefetch=True,
                ):
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this

        return hidden_states
    

class Qwen3OmniMoeThinkerForConditionalGeneration(transformers.Qwen3OmniMoeThinkerForConditionalGeneration, GenerationMixin):
    def __init__(self, config):
        self._apply_moe_block_patch()
        super().__init__(config)
        self.model = Qwen3OmniMoeThinkerTextModel._from_config(config.text_config)
        self.lm_head = Qwen3OmniLMHead(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
    
    def _apply_moe_block_patch(self):
        if has_qwen3omni_support:
            modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextExperts = Qwen3OmniMoeThinkerTextExperts

    @classmethod
    def from_pretrained(cls, hf_path, **kwargs):
        load_kwargs = {
            "_from_auto": True,
            "device_map": None,
            "dtype": None,
            "low_cpu_mem_usage": False
        }

        # get thinker_config
        config = kwargs.get("config")
        if config is not None:
            load_kwargs["config"] = config.thinker_config

        kwargs.update(load_kwargs)
        return super().from_pretrained(hf_path, **kwargs)
    
    @classmethod
    def _from_config(cls, config, **kwargs):
        # thinker_config
        if config.thinker_config is not None:
            config = config.thinker_config
        
        return super()._from_config(config, **kwargs)
    
    @check_model_inputs
    @can_return_tuple
    @auto_docstring  
    def forward(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        output_router_logits: Optional[bool] = None,
        use_audio_in_video=None,
        cache_position=None,
        video_second_per_grid=None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        loss_ctx=None,
        **kwargs,
    ) -> Union[tuple, Qwen3OmniMoeThinkerCausalLMOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        use_audio_in_video (`bool`, *optional*):
            Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
        video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            Number of seconds per grid for each video, used for temporal feature mapping.
        """
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        visual_embeds_multiscale = None
        visual_pos_masks = None
        image_mask, video_mask = None, None
        # 2. Merge text , audios , image and video
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_embeds, image_embeds_multiscale = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, video_embeds_multiscale = self.get_video_features(pixel_values_videos, video_grid_thw)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask | image_mask
            visual_embeds_multiscale_joint = ()
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(image_embeds_multiscale, video_embeds_multiscale):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1])
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                visual_embeds_multiscale_joint = visual_embeds_multiscale_joint + (embed_joint,)
            visual_embeds_multiscale = visual_embeds_multiscale_joint
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_embeds_multiscale = image_embeds_multiscale
            visual_pos_masks = image_mask
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_embeds_multiscale = video_embeds_multiscale
            visual_pos_masks = video_mask

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        position_ids = position_ids.to(torch.bfloat16)
        self.rope_deltas = None

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            deepstack_visual_embeds=visual_embeds_multiscale,
            visual_pos_masks=visual_pos_masks,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        if loss_ctx:
            logits, loss = self.lm_head(hidden_states[:, slice_indices, :], loss_ctx=loss_ctx)
        else:
            logits, loss = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return Qwen3OmniMoeThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )