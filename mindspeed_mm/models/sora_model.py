# coding=utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

from logging import getLogger
from typing import Any, Mapping

import torch
import torch_npu
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn

from mindspeed_mm.data.data_utils.constants import (
    LATENTS,
    PROMPT,
    PROMPT_MASK,
    VIDEO_MASK,
)
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.common.communications import collect_tensors_across_ranks
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.utils.utils import unwrap_single

logger = getLogger(__name__)


class SoRAModel(nn.Module):
    """
    Instantiate a video generation model from config.
    SoRAModel is an assembled model, which may include text_encoder, video_encoder, predictor, and diffusion model

    Args:
        config (dict): the general config for Multi-Modal Model
        {
            "ae": {...},
            "text_encoder": {...},
            "predictor": {...},
            "diffusion": {...},
            "load_video_features":False,
            ...
        }
    """

    def __init__(self, config):
        super().__init__()
        args = get_args()
        self.config = core_transformer_config_from_args(args)

        # training task
        self.task = getattr(config, "task", "t2v")
        self.is_enable_lora = bool(getattr(args, "lora_target_modules", None))

        # encoder inference interleaves with DIT training, encoder_offload_interval = 1 means don't offload
        self.interleaved_steps = getattr(config, "encoder_offload_interval", 1)
        self.enable_encoder_dp = getattr(config, "enable_encoder_dp", False)
        self.cache = {}
        self.index = 0
        self.i2v_keys = None
        if self.enable_encoder_dp and mpu.get_pipeline_model_parallel_world_size() > 1:
            raise AssertionError("Encoder DP cannot be used with PP")

        # build inner moudule
        if args.dist_train:
            from mindspeed.multi_modal.dist_train.parallel_state import is_in_subworld
        self.pre_process = mpu.is_pipeline_first_stage() if not args.dist_train else is_in_subworld(
            'vae')  # vae subworld
        self.post_process = mpu.is_pipeline_last_stage()
        self.input_tensor = None

        if mpu.get_pipeline_model_parallel_rank() == 0:
            self.load_video_features = config.load_video_features
            self.load_text_features = config.load_text_features
            if args.dist_train and not is_in_subworld('vae'):
                self.load_video_features = self.load_text_features = False
            if not self.load_video_features:
                print_rank_0(f"init AEModel....")
                self.ae = AEModel(config.ae).eval()
                self.ae.requires_grad_(False)
            if not self.load_text_features:
                print_rank_0(f"init TextEncoder....")
                self.text_encoder = TextEncoder(config.text_encoder).eval()
                self.text_encoder.requires_grad_(False)
                self.text_encoder_num = len(self.text_encoder.text_encoders) \
                    if isinstance(self.text_encoder.text_encoders, nn.ModuleList) else 1
                self.offload_cpu = self.interleaved_steps > 1

        self.diffusion = DiffusionModel(config.diffusion).get_model()
        if args.dist_train:
            from mindspeed.multi_modal.dist_train.parallel_state import is_in_subworld
            if is_in_subworld('dit'):
                self.predictor = PredictModel(config.predictor).get_model()
        else:
            self.predictor = PredictModel(config.predictor).get_model()

        # to avoid grad all-reduce and reduce-scatter in megatron, since SoRAModel has no embedding layer.
        self.share_embeddings_and_output_weights = False

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor
        if get_args().dist_train and mpu.is_pipeline_first_stage(is_global=False):  # enable dist_train will apply Patch
            self.input_tensor = input_tensor
        else:
            self.predictor.set_input_tensor(input_tensor)

    def forward(self, video, prompt_ids, video_mask=None, prompt_mask=None, skip_encode=False, **kwargs):
        """
        video: raw video tensors, or ae encoded latent
        prompt_ids: tokenized input_ids, or encoded hidden states
        video_mask: mask for video/image
        prompt_mask: mask for prompt(text)
        skip_encode: get feature from the cache in some steps
        """
        args = get_args()
        if self.pre_process:
            with torch.no_grad():
                if not skip_encode:
                    self.index = 0
                    i2v_results = None

                    # Visual Encode
                    if self.load_video_features:
                        latents = video
                    else:
                        if self.task == "t2v":
                            latents, _ = self.ae.encode(video)
                        elif self.task == "i2v":
                            latents, i2v_results = self.ae.encode(video, **kwargs)
                        else:
                            raise NotImplementedError(f"Task {self.task} is not Implemented!")
                    
                    # Text Encode
                    if self.load_text_features:
                        prompt = prompt_ids
                        if isinstance(prompt_ids, list) or isinstance(prompt_ids, tuple):
                            prompt = [p.npu() for p in prompt]
                    else:
                        prompt, prompt_mask = self.text_encoder.encode(prompt_ids, prompt_mask,
                                                                       offload_cpu=self.offload_cpu, **kwargs)

            # Gather the results after encoding of ae and text_encoder
            if self.enable_encoder_dp or self.interleaved_steps > 1:
                if self.index == 0:
                    self.init_cache(latents, prompt, video_mask, prompt_mask, i2v_results)
                latents, prompt, video_mask, prompt_mask, i2v_results = self.get_feature_from_cache()
            else:
                kwargs.update({"shape": latents.shape})

            if self.task == "i2v" and i2v_results is not None:
                kwargs.update(i2v_results)
            noised_latents, noise, timesteps = self.diffusion.q_sample(latents, model_kwargs=kwargs, mask=video_mask)
            predictor_input_latent, predictor_timesteps, predictor_prompt = noised_latents, timesteps, prompt
            predictor_video_mask, predictor_prompt_mask = video_mask, prompt_mask

            if args.dist_train:
                return [predictor_input_latent, predictor_timesteps, predictor_prompt, predictor_video_mask,
                        predictor_prompt_mask,
                        latents, noised_latents, timesteps, noise, video_mask]
        else:
            if args.dist_train and mpu.is_pipeline_first_stage(is_global=False):
                [predictor_input_latent, predictor_timesteps, predictor_prompt, predictor_video_mask,
                 predictor_prompt_mask,
                 latents, noised_latents, timesteps, noise, video_mask] = self.input_tensor
            else:
                if not hasattr(self.predictor, "pipeline_set_prev_stage_tensor"):
                    raise ValueError(f"PP has not been implemented for {self.predictor_cls} yet. ")
                predictor_input_list, training_loss_input_list = self.predictor.pipeline_set_prev_stage_tensor(
                    self.input_tensor, extra_kwargs=kwargs)
                predictor_input_latent, predictor_timesteps, predictor_prompt, predictor_video_mask, predictor_prompt_mask \
                    = predictor_input_list
                latents, noised_latents, timesteps, noise, video_mask = training_loss_input_list

        output = self.predictor(
            predictor_input_latent,
            timestep=predictor_timesteps,
            prompt=predictor_prompt,
            video_mask=predictor_video_mask,
            prompt_mask=predictor_prompt_mask,
            **kwargs,
        )

        if self.post_process:
            loss = self.compute_loss(
                output if isinstance(output, torch.Tensor) else output[0],
                latents,
                noised_latents,
                timesteps,
                noise,
                video_mask,
                **kwargs
            )
            return [loss]

        return self.predictor.pipeline_set_next_stage_tensor(
            input_list=[latents, noised_latents, timesteps, noise, video_mask],
            output_list=output,
            extra_kwargs=kwargs)

    def compute_loss(
            self, model_output, latents, noised_latents, timesteps, noise, video_mask, **kwargs
    ):
        """compute diffusion loss"""
        loss_dict = self.diffusion.training_losses(
            model_output=model_output,
            x_start=latents,
            x_t=noised_latents,
            noise=noise,
            t=timesteps,
            mask=video_mask,
            **kwargs
        )
        return loss_dict

    def train(self, mode=True):
        if hasattr(self, "predictor"):
            self.predictor.train()

    def state_dict_for_save_lora_checkpoint(self, state_dict):
        state_dict_ = dict()
        for key in state_dict:
            if 'lora' in key:
                state_dict_[key] = state_dict[key]
        return state_dict_

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        if not get_args().dist_train:
            state_dict = self.predictor.state_dict(prefix=prefix, keep_vars=keep_vars)
            if self.is_enable_lora:
                state_dict = self.state_dict_for_save_lora_checkpoint(state_dict)
            return state_dict
        from mindspeed.multi_modal.dist_train.parallel_state import is_in_subworld
        if is_in_subworld('dit'):
            return self.predictor.state_dict(prefix=prefix, keep_vars=keep_vars)
        return None

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Customized load."""
        if get_args().dist_train:
            from mindspeed.multi_modal.dist_train.parallel_state import is_in_subworld
            if is_in_subworld('vae'):
                return None
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

        missing_keys, unexpected_keys = self.predictor.load_state_dict(state_dict, False)

        if missing_keys is not None:
            logger.info(f"Missing keys in state_dict: {missing_keys}.")
        if unexpected_keys is not None:
            logger.info(f"Unexpected key(s) in state_dict: {unexpected_keys}.")
        return None

    def init_cache(self, latents, prompt, video_mask, prompt_mask, i2v_results=None):
        """
        Initialize feature cache in the first step of one round when encoder dp or encoder interleave offload is enabled.
        example with latents and prompt:
        input:
            latents [step_1, step_2..., step_n]
            prompt  [step_n][encoder_1, encoder_2, ...encoder_n]
        cache:
            latents [step_n][rank1_data, rank2_data, ...]"
            prompt  [step_n][encoder_n][rank1_data, rank2_data, ...]
        """
        # empty cache
        self.cache = {}
        group = mpu.get_tensor_and_context_parallel_group()
        for key, value in [(LATENTS, latents), (VIDEO_MASK, video_mask)]:
            if value is None or len(value) < 0:
                continue
            self.cache[key] = [[item] for item in value] if not self.enable_encoder_dp \
                else collect_tensors_across_ranks(value, group=group, dynamic_shape=False)

        for key, value in [(PROMPT, prompt), (PROMPT_MASK, prompt_mask)]:
            if value is None or len(value) < 0:
                continue
            if not self.enable_encoder_dp:
                self.cache[key] = [[item] for item in value]
                continue
            self.cache[key] = [[[] for _ in range(self.text_encoder_num)] for _ in range(self.interleaved_steps)]
            for encoder_idx in range(self.text_encoder_num):
                # Features from the same text encoder have identical shapes, concat to reduce communication overhead.
                encoder_step_tensors = torch.stack([value[step][encoder_idx] for step in range(self.interleaved_steps)])
                collected_tensors = collect_tensors_across_ranks(encoder_step_tensors, group=group, dynamic_shape=False)

                for step_idx in range(self.interleaved_steps):
                    for collected_tensor in collected_tensors:
                        self.cache[key][step_idx][encoder_idx].append(
                            collected_tensor[step_idx:step_idx + 1].squeeze(0).contiguous())

        # handle extra i2v cache, source i2v_results：{"key_n":[step_1_data, step_2_date ... step_n_data]}
        if self.task != "i2v" or not i2v_results:
            return

        self.i2v_keys = i2v_results.keys() if self.i2v_keys is None else self.i2v_keys
        for i2v_key, value in i2v_results.items():
            if not self.enable_encoder_dp:
                self.cache[i2v_key] = [[item] for item in value]
                continue
            self.cache[i2v_key] = collect_tensors_across_ranks(value, group=group, dynamic_shape=False)

    def get_feature_from_cache(self):
        """Get from the cache while several features have been already encoded and cached."""
        divisor = mpu.get_tensor_and_context_parallel_world_size() if self.enable_encoder_dp else 1
        step_idx = self.index // divisor
        rank_idx = self.index % divisor

        latents = unwrap_single(self.cache[LATENTS][step_idx][rank_idx] if LATENTS in self.cache else None)
        video_mask = unwrap_single(self.cache[VIDEO_MASK][step_idx][rank_idx] if VIDEO_MASK in self.cache else None)
        prompt = unwrap_single([self.cache[PROMPT][step_idx][encoder_idx][rank_idx] \
                                for encoder_idx in range(self.text_encoder_num)] if PROMPT in self.cache else None)
        prompt_mask = unwrap_single([self.cache[PROMPT_MASK][step_idx][encoder_idx][rank_idx] \
                                     for encoder_idx in range(self.text_encoder_num)] if PROMPT_MASK in self.cache else None)

        i2v_results = {}
        if self.task == "i2v":
            for key in self.i2v_keys:
                i2v_results[key] = unwrap_single(self.cache[key][step_idx][rank_idx] if key in self.cache else None)

        self.index += 1
        return latents, prompt, video_mask, prompt_mask, i2v_results
