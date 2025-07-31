#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

from typing import List, Optional, Tuple, Union
import torch

from modeling_bailingmm import BailingMMNativeForConditionalGeneration, BailingMMCausalLMOutputWithPast


class AscendBailingMMNativeForConditionalGeneration(BailingMMNativeForConditionalGeneration):

    _supports_flash_attn_2 = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        audio_feats: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_feats_lengths: Optional[torch.LongTensor] = None,
        audio_placeholder_loc_lens: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_whisper_encoder: bool = False,
    ) -> Union[Tuple, BailingMMCausalLMOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if (
            pixel_values is not None or pixel_values_videos is not None or audio_feats is not None
        ) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos/pixel_values_audios and inputs_embeds at the same time, and must specify either one"
            )

        image_embeds, video_embeds, audio_embeds, audio_embeds_lengths = None, None, None, None
        if pixel_values is not None:
            image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)
        if pixel_values_videos is not None:
            video_embeds = self.extract_image_feature(pixel_values_videos, grid_thw=video_grid_thw)
        if audio_feats is not None:
            audio_embeds, audio_embeds_lengths = self.extract_audio_feature(
                audio_feats, audio_feats_lengths, use_whisper_encoder=use_whisper_encoder
            )

        if (
            image_embeds is None and video_embeds is None and audio_embeds is None
        ) or input_ids.size(1) == 1:
            words_embeddings = self.model.get_input_embeddings()(
                input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1)
            )
            image_mask = None
            audio_mask = None

        else:
            words_embeddings, image_mask, audio_mask = self.prompt_wrap_navit(
                input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1),
                image_embeds,
                video_embeds,
                audio_embeds,
                audio_embeds_lengths,
                audio_placeholder_loc_lens,
                None,  # noqa
            )

        if (
            self.config.llm_config.rope_scaling is not None
            and self.config.llm_config.rope_scaling["type"] == "3D"
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_token_id=self.config.llm_config.image_patch_token,
                video_token_id=self.config.llm_config.image_patch_token,
                image_start_token_id=self.config.llm_config.image_start_token,
                video_start_token_id=self.config.llm_config.video_start_token,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            rope_deltas = None

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=words_embeddings,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_mask=image_mask,
            audio_mask=audio_mask,
        )

        return BailingMMCausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
