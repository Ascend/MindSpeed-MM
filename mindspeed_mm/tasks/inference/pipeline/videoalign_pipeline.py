# coding=utf-8
# Copyright 2025 The KwaiVGI team and the HuggingFace Inc. team. All rights reserved.


import copy
from copy import deepcopy
from collections.abc import Mapping

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

import mindspeed.megatron_adaptor
from megatron.training import get_args
from megatron.training.checkpointing import load_checkpoint
from mindspeed_mm.utils.transformer_model_config import get_model_config
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.generation_mixin import GenerationMixin
from mindspeed_mm.models.text_encoder import Tokenizer
from mindspeed_mm.models.reward_model import Qwen2VLRewardModelBT
from mindspeed_mm.data.data_utils.reward_preprocess import fill_data_template, clean_examples
from mindspeed_mm.data.data_utils.func_utils.convert import reward_setting_processor


class VideoAlignPipeline(GenerationMixin):
    def __init__(self, model_config, preprocess_params, norm_param=None):
        self.model_config = model_config
        self.preprocess_params = preprocess_params
        self.norm_param = norm_param

        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16 if getattr(model_config.text_decoder, 'bf16', False) else torch.float32
        model_preprocess_params = copy.deepcopy(preprocess_params)

        self.video_reader, self.video_processor, self.tokenizer, self.processor, self.model_args = reward_setting_processor(
            model_preprocess_params)
        self.model = self.model_provider()
        load_checkpoint([self.model], None, None, 'load')

        self.model = self.model.to(self.device, self.dtype)

    def model_provider(self):
        """Builds the model."""
        vlm_config = deepcopy(self.model_config)

        vlm_config.pre_process = True
        vlm_config.post_process = True
        vlm_config.reward_process = True

        if vlm_config.image_encoder and vlm_config.text_decoder:
            vlm_config.image_encoder.vision_encoder = get_model_config(vlm_config.image_encoder.vision_encoder)
            vlm_config.image_encoder.vision_projector = get_model_config(vlm_config.image_encoder.vision_projector)
            vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)

        model = Qwen2VLRewardModelBT(config=vlm_config, extra_config=self.model_args)
        model.freeze(freeze_image_encoder=True, freeze_image_projection=True, freeze_text_decoder=True)
        return model

    def _norm(self, reward):
        if self.norm_param is None:
            return reward
        else:
            reward['VQ'] = (reward['VQ'] - self.norm_param['VQ_mean']) / self.norm_param['VQ_std']
            reward['MQ'] = (reward['MQ'] - self.norm_param['MQ_mean']) / self.norm_param['MQ_std']
            reward['TA'] = (reward['TA'] - self.norm_param['TA_mean']) / self.norm_param['TA_std']
            return reward

    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            def mapping_key(k):
                if k == 'video_grid_thw':
                    return 'image_grid_thw'
                elif k == 'pixel_values_videos':
                    return 'pixel_values'
                else:
                    return k

            return type(data)({mapping_key(k): self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before HH them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs

    def prepare_batch(self, data_folder, batch_data):
        setting_fps = self.preprocess_params.get('fps', None)
        video_max_pixels = self.preprocess_params.get('video_max_pixels', None)
        eval_dim = self.preprocess_params.get('eval_dim', None)
        sample_nframe = self.preprocess_params.get('sample_nframe', None)
        sample_type = self.preprocess_params.get('sample_type', None)
        prompt_template_type = self.preprocess_params.get('prompt_template_type', None)

        video_inputs = []
        chat_datas = [
            fill_data_template(data_folder=data_folder,
                               relative_path=video_path,
                               prompt=prompt,
                               fps=setting_fps if setting_fps else fps,
                               max_pixels=video_max_pixels,
                               num_frames=num_frames,
                               eval_dim=eval_dim,
                               sample_nframe=sample_nframe,
                               sample_type=sample_type,
                               prompt_template_type=prompt_template_type)
            for (video_path, prompt, fps, num_frames) in batch_data
        ]

        clean_chat_datas = []
        for chat_data in chat_datas:
            clean_chat_data = clean_examples(chat_data)
            clean_chat_datas.append(clean_chat_data)
            video_input = self.video_processor(self.video_reader(clean_chat_data[0]['content'][0]["video"]))
            video_inputs.append(video_input)

        batch = self.processor(
            text=self.processor.apply_chat_template(clean_chat_datas, tokenize=False, add_generation_prompt=True),
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )

        batch = self._prepare_inputs(batch)
        batch['pixel_values'] = batch['pixel_values'].to(self.dtype)

        return batch

    def __call__(self, data_folder, batch_data, use_norm=False, model_reward_return=False):
        batch = self.prepare_batch(data_folder=data_folder, batch_data=batch_data)

        rewards = self.model(
            return_dict=True,
            **batch
        )
        if model_reward_return:
            return rewards

        rewards = [{'VQ': reward[0].item(), 'MQ': reward[1].item(), 'TA': reward[2].item()} for reward in rewards]

        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]['Overall'] = rewards[i]['VQ'] + rewards[i]['MQ'] + rewards[i]['TA']

        return rewards