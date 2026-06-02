# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
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

from typing import Optional, List, Union
import inspect
import PIL

import torch
from diffusers.video_processor import VideoProcessor

from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin


class CogVideoXPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model, config=None):
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
            predict_model=predict_model, scheduler=scheduler
        )

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.predict_model = predict_model

        config = config.to_dict()
        self.num_frames, self.height, self.width = config.get("input_size", [49, 480, 720])
        self.generator = torch.Generator().manual_seed(config.get("seed", 1234))
        self.num_videos_per_prompt = 1
        self.guidance_scale = config.get("guidance_scale", 6.0)

        self.scheduler.use_dynamic_cfg = config.get("use_dynamic_cfg", True)

        self.vae_scale_factor_temporal = self.vae.vae_scale_factor[0]
        self.vae_scale_factor_spatial = self.vae.vae_scale_factor[1]
        self.vae_scaling_factor = self.vae.vae_scale_factor[2]
        self.vae_invert_scale_latents = config.get("vae_invert_scale_latents", False)

        self.use_tiling = config.get("use_tiling", True)
        self.cogvideo_version = config.get("version", 1.0)
        self.additional_frames = 0

        if self.use_tiling:
            self.vae.enable_tiling()
        else:
            self.vae.disable_tiling()

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)


    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def prepare_image_latents(self, image, height, width, device, dtype):
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=dtype)
        image = image.unsqueeze(2).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]

        image_latents = [self.vae.encode(img.unsqueeze(0), invert_scale_latents=self.vae_invert_scale_latents,
                                         generator=self.generator) for img in image]
        image_latents = torch.cat(image_latents, dim=0)  # [B, C, T, H, W]

        if self.vae_invert_scale_latents:
            image_latents = 1 / self.vae_scaling_factor * image_latents

        padding_shape = (
            image_latents.shape[0],
            self.predict_model.in_channels // 2,
            (self.num_frames - 1) // self.vae_scale_factor_temporal,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial
        )

        latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
        image_latents = torch.cat([image_latents, latent_padding], dim=2)

        if self.predict_model.patch_size[0] is not None:
            first_frame = image_latents[:, :, : image_latents.size(2) % self.predict_model.patch_size[0], ...]
            image_latents = torch.cat([first_frame, image_latents], dim=2)
        return image_latents

    @torch.no_grad()
    def __call__(self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        height = self.height or self.predict_model.config.sample_size * self.vae_scale_factor_spatial
        width = self.width or self.predict_model.config.sample_size * self.vae_scale_factor_spatial

        # 1. Check inputs.
        self.text_prompt_checks(
            prompt,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if image is not None:
            self.image_prompt_checks(image)

        self.generate_params_checks(height, width)
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.text_encoder.device or self._execution_device

        # 3. Encode input prompt
        prompt_embeds, prompt_embeds_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = self.encode_texts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=device,
            do_classifier_free_guidance=True,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=False,
            prompt_to_lower=False
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 5. Prepare latents
        latent_frames = (self.num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.predict_model.patch_size[0]
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            self.additional_frames = patch_size_t - latent_frames % patch_size_t
            self.num_frames += self.additional_frames * self.vae_scale_factor_temporal

        # prepare image_latents for i2v task
        if image is not None:
            image_latents = self.prepare_image_latents(
                image=image,
                height=height,
                width=width,
                device=device,
                dtype=prompt_embeds.dtype
            )

            # do_classifier_free_guidence
            image_latents = torch.cat([image_latents] * 2)
        else:
            image_latents = None

        # prepare latents for all task
        latent_channels = self.predict_model.in_channels if image is None else self.predict_model.in_channels // 2
        batch_size = batch_size * self.num_videos_per_prompt
        shape = (
            batch_size,
            (self.num_frames - 1) // self.vae_scale_factor_temporal + 1,
            latent_channels,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial
        )
        latents = self.prepare_latents(shape, generator=self.generator, device=device, dtype=prompt_embeds.dtype,
                                       latents=latents)

        # 6 prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.generator, eta)

        model_kwargs = {"prompt": prompt_embeds.unsqueeze(1),
                        "prompt_mask": prompt_embeds_attention_mask,
                        "masked_video": image_latents}

        self.scheduler.guidance_scale = self.guidance_scale
        latents = self.scheduler.sample(model=self.predict_model, shape=shape, latents=latents,
                                        model_kwargs=model_kwargs,
                                        extra_step_kwargs=extra_step_kwargs)

        latents = latents[:, self.additional_frames:]
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor * latents
        video = self.decode_latents(latents, cogvideo_version=self.cogvideo_version)
        return video

    def callback_on_step_end_tensor_inputs_checks(self, callback_on_step_end_tensor_inputs):
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
