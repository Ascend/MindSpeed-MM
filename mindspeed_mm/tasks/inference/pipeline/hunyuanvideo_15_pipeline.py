# This file was adapted from Tencent's HunyuanVideo 1.5 pipeline (Tencent Hunyuan Community License).
# It is now distributed under the AGPL-3.0-or-later for SimpleTuner contributors.

import inspect
import os
from typing import Optional, Union, List

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch_npu
import torchvision.transforms as transforms
import transformers
from PIL.Image import Image
from accelerate import cpu_offload_with_hook
from diffusers.video_processor import VideoProcessor, VaeImageProcessor
from megatron.core import mpu
from transformers import CLIPImageProcessor

from mindspeed_mm.models.predictor.dits.hunyuanvideo15.utils import get_parallel_state, generate_crop_size_list, \
    get_closest_ratio, resize_and_center_crop, auto_offload_model
from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin
from mindspeed_mm.utils.extra_processor.i2v_processors import I2VProcessor

NEGATIVE_PROMPT = ""


class HunyuanVideo15Pipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds"
    ]

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @num_timesteps.setter
    def num_timesteps(self, value):
        if value <= 0:
            raise ValueError("num_timesteps must be positive")
        self._num_timesteps = value

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def vae_spatial_compression_ratio(self):
        if hasattr(self.vae, "ffactor_spatial"):
            return self.vae.ffactor_spatial
        else:
            return 16

    @property
    def vae_temporal_compression_ratio(self):
        if hasattr(self.vae, "ffactor_temporal"):
            return self.vae.ffactor_temporal
        else:
            return 4

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def ideal_resolution(self):
        return self._ideal_resolution

    @property
    def ideal_task(self):
        return self.predict_model.task_type

    @property
    def use_meanflow(self):
        return self._use_meanflow

    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model, config=None, progress_bar_config=None):

        super().__init__()

        self.register_modules(
            vae=vae,
            scheduler=scheduler,
            predict_model=predict_model
        )
        self.text_encoders = text_encoder
        self.tokenizers = tokenizer

        self.glyph_byT5_v2 = config.get("glyph_byT5_v2", False)
        self.byt5_max_length = tokenizer[1].byt5_max_length
        self.vision_num_semantic_tokens = config.get("vision_num_semantic_tokens", 729)
        self.vision_states_dim = predict_model.vision_states_dim
        self.flow_shift = config.get("flow_shift", 5.0)
        self.num_inference_steps = scheduler.num_inference_timesteps
        self._ideal_resolution = config.get("ideal_resolution", "720p")
        self._clip_skip = config.get("clip_skip", None)
        self._use_meanflow = config.get("use_meanflow", False)
        self.use_attention_mask = config.get("use_attention_mask", False)
        self.aspect_ratio = config.get("aspect_ratio", "16:9")

        self.frames, self.height, self.width = config.get("input_size", [121, 256, 256])
        self.video_length = self.frames
        self.generate_params_checks(self.height, self.width)
        self._guidance_scale = config.get("guidance_scale", 6.0)
        self._guidance_rescale = config.get("guidance_rescale", 0.0)
        self.embedded_guidance_scale = config.get("embedded_guidance_scale", None)

        self.eta = config.get("eta", 0.0)
        self.cpu_offload = config.get("cpu_offload", False)
        if self.cpu_offload:
            local_rank = int(os.getenv("LOCAL_RANK"))
            self.enable_model_cpu_offload(local_rank)

        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.vae_scale_factor = 2 ** (len(self.vae.encoder.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.text_len = config.get("text_max_length", 1000)
        self.target_dtype = torch.bfloat16
        self.vae_dtype = torch.float16
        self.autocast_enabled = True
        self.vae_autocast_enabled = True
        self.enable_offloading = config.get("enable_offloading", False)
        self.execution_device = torch.device("npu")
        self._ideal_task = config.get("task", "t2v")

        if self.ideal_task == "i2v":
            self.i2v_processor_config = config.get("i2v_processor", None).to_dict()
            self.vision_encoder = I2VProcessor(self.i2v_processor_config).get_processor().vision_encoder
        else:
            self.vision_encoder = None

        # Default i2v target size configurations
        self.target_size_config = {
            "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
            "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
            "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
            "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
        }

        self.noise_init_device = torch.device('cpu')
        self.seed = config.get("seed", 42)
        self.generator = torch.Generator(device=self.noise_init_device).manual_seed(self.seed)

    @torch.no_grad()
    def __call__(
            self,
            prompt: Optional[Union[str, List[str]]] = None,
            image: Optional[Union[Image, List[Image]]] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            device: torch.device = "npu",
            data_type: str = "video",
            attention_mask: Optional[torch.Tensor] = None,
            negative_attention_mask: Optional[torch.Tensor] = None,
            clip_skip: Optional[int] = None,
            use_prompt_preprocess: Optional[bool] = False,
            return_dict: bool = True,
            return_pre_sr_video: bool = False,
            enable_sr: bool = False,
            **kwargs
    ):

        # 1. Check inputs
        self.text_prompt_checks(
            prompt,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds
        )

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = NEGATIVE_PROMPT
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")
        negative_prompt = [negative_prompt.strip()]

        num_videos_per_prompt = 1
        target_resolution = self.ideal_resolution

        guidance_scale = self.guidance_scale
        embedded_guidance_scale = self.embedded_guidance_scale
        flow_shift = self.flow_shift
        num_inference_steps = self.num_inference_steps

        if embedded_guidance_scale is not None:
            if self.do_classifier_free_guidance:
                raise AssertionError(
                    "embedded_guidance_scale and do_classifier_free_guidance can not use in the same time. "
                )
            if not self.predict_model.guidance_embed:
                raise ValueError(
                    "when use embedded_guidance_scale，self.predict_model.guidance_embed must be True."
                )
        else:
            if self.predict_model.guidance_embed:
                raise ValueError(
                    "when no embedded_guidance_scale，self.predict_model.guidance_embed must be False."
                )
        reference_image = image[0] if image is not None else None
        user_reference_image = reference_image
        user_prompt = prompt

        if reference_image is not None:
            task_type = "i2v"
            if isinstance(reference_image, str):
                reference_image = Image.open(reference_image).convert('RGB')
            elif not isinstance(reference_image, Image):
                raise ValueError("reference_image must be a PIL Image or path to image file")
            semantic_images_np = np.array(reference_image)
        else:
            task_type = "t2v"
            semantic_images_np = None

        if self.ideal_task is not None and self.ideal_task != task_type:
            raise ValueError(
                f"The loaded pipeline is trained for '{self.ideal_task}' task, but received input for '{task_type}' task. "
                "Please load a pipeline trained for the correct task, or check and update your arguments accordingly."
            )

        seed = self.seed
        if get_parallel_state().sp_enabled:
            if dist.is_initialized():
                obj_list = [seed]
                group_src_rank = dist.get_global_rank(get_parallel_state().sp_group, 0)
                dist.broadcast_object_list(obj_list, src=group_src_rank, group=get_parallel_state().sp_group)
                seed = obj_list[0]

        generator = self.generator

        if reference_image is not None:
            if self.ideal_resolution is not None and target_resolution != self.ideal_resolution:
                raise ValueError(
                    f'The loaded pipeline is trained for {self.ideal_resolution} resolution, but received input for {target_resolution} resolution. '
                )
            height, width = self.get_closest_resolution_given_reference_image(reference_image, target_resolution)
        else:
            if self.ideal_resolution is not None:
                if ":" not in self.aspect_ratio:
                    raise ValueError("aspect_ratio must be separated by a colon")
                width, height = self.aspect_ratio.split(":")
                # check if width and height are integers
                size_cond = not width.isdigit() or not height.isdigit() or int(width) <= 0 or int(height) <= 0
                if size_cond:
                    raise ValueError(
                        "width and height must be positive integers and separated by a colon in aspect_ratio")
                width = int(width)
                height = int(height)
                height, width = self.get_closest_resolution_given_original_size((width, height), self.ideal_resolution)
            else:
                raise ValueError("ideal_resolution is not set")

        latent_target_length, latent_height, latent_width = self.get_latent_size(self.video_length, height, width)
        n_tokens = latent_target_length * latent_height * latent_width
        multitask_mask = self.get_task_mask(task_type, latent_target_length)

        device = self.execution_device

        if int(os.environ.get('RANK', '0')) == 0:
            print(
                '\n'
                f"{'=' * 60}\n"
                f"🎬  HunyuanVideo Generation Task\n"
                f"{'-' * 60}\n"
                f"User Prompt:               {user_prompt}\n"
                f"Aspect Ratio:              {self.aspect_ratio if task_type == 't2v' else f'{width}:{height}'}\n"
                f"Video Length:              {self.video_length}\n"
                f"Reference Image:           {user_reference_image} {reference_image.size if reference_image is not None else ''}\n"
                f"Guidance Scale:            {guidance_scale}\n"
                f"Guidance Embedded Scale:   {embedded_guidance_scale}\n"
                f"Shift:                     {flow_shift}\n"
                f"Seed:                      {seed}\n"
                f"Video Resolution:          {width} x {height}\n"
                f'Attn mode:                 {self.predict_model.attn_mode}\n'
                f"Transformer dtype:         {self.predict_model.dtype}\n"
                f"Sampling Steps:            {num_inference_steps}\n"
                f"Use Meanflow:              {self.use_meanflow}\n"
                f"{'=' * 60}"
                '\n'
            )

        with auto_offload_model(self.text_encoders[0], self.execution_device, enabled=self.enable_offloading):
            (
                prompt_embeds,
                negative_prompt_embeds,
                prompt_mask,
                negative_prompt_mask,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                clip_skip=self.clip_skip,
                data_type="video",
                text_tokenizer=self.tokenizers[0],
                text_encoder=self.text_encoders[0]
            )

        prompt_embeds_2 = None
        negative_prompt_embeds_2 = None
        prompt_mask_2 = None
        negative_prompt_mask_2 = None

        extra_kwargs = {}
        if self.glyph_byT5_v2:
            with auto_offload_model(self.text_encoders[1], self.execution_device, enabled=self.enable_offloading):
                extra_kwargs = self._prepare_byt5_embeddings(prompt, device)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )

        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            **extra_set_timesteps_kwargs,
        )

        num_channels_latents = self.predict_model.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            latent_height,
            latent_width,
            latent_target_length,
            self.target_dtype,
            device,
            generator,
        )

        with auto_offload_model(self.vae, self.execution_device, enabled=self.enable_offloading):
            image_cond = self.get_image_condition_latents(task_type, reference_image, height, width)

        cond_latents = self._prepare_cond_latents(
            task_type, image_cond, latents, multitask_mask
        )
        with auto_offload_model(self.vision_encoder, self.execution_device, enabled=self.enable_offloading):
            vision_states = self._prepare_vision_states(
                semantic_images_np, target_resolution, latents, device
            )

        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": kwargs.get("eta", 0.0)},
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        cache_helper = getattr(self, 'cache_helper', None)
        if cache_helper is not None:
            cache_helper.clear_states()

        with self.progress_bar(total=num_inference_steps) as progress_bar, auto_offload_model(self.predict_model,
                                                                                              self.execution_device,
                                                                                              enabled=self.enable_offloading):
            for i, t in enumerate(timesteps):
                if cache_helper is not None:
                    cache_helper.cur_timestep = i
                latents_concat = torch.concat([latents, cond_latents], dim=1)
                latent_model_input = torch.cat(
                    [latents_concat] * 2) if self.do_classifier_free_guidance else latents_concat

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])
                if self.use_meanflow:
                    if i == len(timesteps) - 1:
                        timesteps_r = torch.tensor([0.0], device=self.execution_device)
                    else:
                        timesteps_r = timesteps[i + 1]
                    timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
                else:
                    timesteps_r = None

                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(self.target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                byt5_text_states = extra_kwargs.get("byt5_text_states", None)
                byt5_text_mask = extra_kwargs.get("byt5_text_mask", None)
                output = self.predict_model(
                    latent_model_input,
                    t_expand,
                    prompt=[prompt_embeds.unsqueeze(0), byt5_text_states.unsqueeze(0)],
                    prompt_mask=[prompt_mask.unsqueeze(0), byt5_text_mask.unsqueeze(0)],
                    timestep_r=timesteps_r,
                    guidance=guidance_expand,
                    kwargs=kwargs,
                    vision_states=vision_states,
                    mode="infer",
                )
                noise_pred = output[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = self.rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()

        if len(latents.shape) == 4:
            latents = latents.unsqueeze(2)
        elif len(latents.shape) != 5:
            raise ValueError(
                f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
            )

        if hasattr(self.vae, "shift_factor") and self.vae.shift_factor:
            latents = latents / self.vae.scaling_factor + self.vae.shift_factor
        else:
            latents = latents / self.vae.scaling_factor

        if hasattr(self.vae, 'enable_tile_parallelism'):
            self.vae.enable_tile_parallelism()

        if return_pre_sr_video or not enable_sr:
            with torch.autocast(device_type="npu", dtype=self.vae_dtype,
                                enabled=self.vae_autocast_enabled), auto_offload_model(self.vae, self.execution_device,
                                                                                       enabled=self.enable_offloading), self.vae.memory_efficient_context():
                video_frames = self.vae.decode(latents, return_dict=False, generator=generator)[0]
        else:
            video_frames = None  # reserve for sr
        if video_frames.ndim == 5:
            if video_frames.shape[0] != 1:
                raise AssertionError("video.shape[0] needs to be 1. ")
            video_frames = video_frames[0]
        if mpu.get_context_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
            video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()
            video_frames = (video_frames * 255).clamp(0, 255).to(torch.uint8)
            video_frames = einops.rearrange(video_frames, 'c f h w -> f h w c').unsqueeze(0)

        return video_frames

    def get_task_mask(self, task_type, latent_target_length):
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported !")
        return mask

    def get_closest_resolution_given_reference_image(self, reference_image, target_resolution):
        if reference_image is None:
            raise AssertionError("reference image must be not None")
        if isinstance(reference_image, Image):
            origin_size = reference_image.size
        elif isinstance(reference_image, np.ndarray):
            H, W, C = reference_image.shape
            origin_size = (W, H)
        else:
            raise ValueError(
                f"Unsupported reference_image type: {type(reference_image)}. Must be PIL Image or numpy array")

        return self.get_closest_resolution_given_original_size(origin_size, target_resolution)

    def get_closest_resolution_given_original_size(self, origin_size, target_size):
        bucket_hw_base_size = self.target_size_config[target_size]["bucket_hw_base_size"]
        bucket_hw_bucket_stride = self.target_size_config[target_size]["bucket_hw_bucket_stride"]

        bucket_cond = bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960, 1440]
        if not bucket_cond:
            raise AssertionError(
                f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960, 1440], but got {bucket_hw_base_size}")

        crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
        aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
        closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

        height = closest_size[0]
        width = closest_size[1]

        return height, width

    def encode_prompt(
            self,
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=None,
            attention_mask=None,
            negative_prompt_embeds=None,
            negative_attention_mask=None,
            clip_skip=None,
            text_tokenizer=None,
            text_encoder=None,
            data_type="image",
    ):

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:

            text_inputs = text_tokenizer(text=prompt, data_type=data_type, max_length=self.text_len)
            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, device=device, use_attention_mask=self.use_attention_mask
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    device=device,
                    use_attention_mask=self.use_attention_mask
                )
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.model.dtype
        elif self.predict_model is not None:
            prompt_embeds_dtype = self.predict_model.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = text_tokenizer(uncond_tokens, data_type=data_type, max_length=self.text_len)

            negative_prompt_outputs = text_encoder.encode(uncond_input, use_attention_mask=self.use_attention_mask)
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(1, num_videos_per_prompt)
                negative_attention_mask = negative_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, -1)
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

    def _prepare_byt5_embeddings(self, prompts, device):
        if not self.glyph_byT5_v2:
            return {}

        if isinstance(prompts, str):
            prompt_list = [prompts]
        elif isinstance(prompts, list):
            prompt_list = prompts
        else:
            raise ValueError("prompts must be str or list of str")

        positive_embeddings = []
        positive_masks = []
        negative_embeddings = []
        negative_masks = []

        for prompt in prompt_list:
            pos_emb, pos_mask = self._process_single_byt5_prompt(prompt, device)
            positive_embeddings.append(pos_emb)
            positive_masks.append(pos_mask)

            if self.do_classifier_free_guidance:
                neg_emb, neg_mask = self._process_single_byt5_prompt("", device)
                negative_embeddings.append(neg_emb)
                negative_masks.append(neg_mask)

        byt5_positive = torch.cat(positive_embeddings, dim=0)
        byt5_positive_mask = torch.cat(positive_masks, dim=0)

        if self.do_classifier_free_guidance:
            byt5_negative = torch.cat(negative_embeddings, dim=0)
            byt5_negative_mask = torch.cat(negative_masks, dim=0)

            byt5_embeddings = torch.cat([byt5_negative, byt5_positive], dim=0)
            byt5_masks = torch.cat([byt5_negative_mask, byt5_positive_mask], dim=0)
        else:
            byt5_embeddings = byt5_positive
            byt5_masks = byt5_positive_mask

        return {
            "byt5_text_states": byt5_embeddings,
            "byt5_text_mask": byt5_masks
        }

    def _process_single_byt5_prompt(self, prompt_text, device):
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=device)
        byt5_mask = torch.zeros((1, self.byt5_max_length), device=device, dtype=torch.int64)

        glyph_texts = self._extract_glyph_texts(prompt_text)

        if len(glyph_texts) > 0:
            text_styles = [{'color': None, 'font-family': None} for _ in range(len(glyph_texts))]
            formatted_text = self.tokenizers[1].format_prompt(glyph_texts, text_styles)

            text_ids, text_mask = self.get_byt5_text_tokens(
                self.tokenizers[1], self.byt5_max_length, formatted_text
            )
            text_ids = text_ids.to(device=device)
            text_mask = text_mask.to(device=device)

            byt5_outputs = self.text_encoders[1](text_ids, attention_mask=text_mask.float())
            byt5_embeddings = byt5_outputs[0]
            byt5_mask = text_mask

        return byt5_embeddings, byt5_mask

    @staticmethod
    def get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return byt5_text_inputs.input_ids, byt5_text_inputs.attention_mask

    def _extract_glyph_texts(self, prompt):
        en_results = []
        start = 0
        while True:
            open_idx = prompt.find('"', start)
            if open_idx == -1:
                break
            close_idx = prompt.find('"', open_idx + 1)
            if close_idx == -1:
                break
            en_results.append(prompt[open_idx + 1: close_idx])
            start = close_idx + 1

        zh_results = []
        start = 0
        while True:
            open_idx = prompt.find('“', start)
            if open_idx == -1:
                break
            close_idx = prompt.find('”', open_idx + 1)
            if close_idx == -1:
                break
            zh_results.append(prompt[open_idx + 1: close_idx])
            start = close_idx + 1

        seen = set()
        final = []
        for t in en_results + zh_results:
            if t not in seen:
                seen.add(t)
                final.append(t)
        return final

    def retrieve_timesteps(
            self,
            scheduler,
            num_inference_steps: Optional[int] = None,
            device: Optional[Union[str, torch.device]] = None,
            timesteps: Optional[List[int]] = None,
            sigmas: Optional[List[float]] = None,
            **kwargs,
    ):
        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
            )
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        std_text = noise_pred_text.std(
            dim=list(range(1, noise_pred_text.ndim)), keepdim=True
        )
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = (
                guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        )
        return noise_cfg

    def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            latent_height,
            latent_width,
            video_length,
            dtype,
            device,
            generator,
            latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            latent_height,
            latent_width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = torch.randn(shape, generator=generator, device=self.noise_init_device, dtype=dtype).to(device)
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_image_condition_latents(self, task_type, reference_image, height, width):

        if task_type == "t2v":
            cond_latents = None

        elif task_type == "i2v":
            origin_size = reference_image.size

            target_height, target_width = height, width
            original_width, original_height = origin_size

            scale_factor = max(target_width / original_width, target_height / original_height)
            resize_width = int(round(original_width * scale_factor))
            resize_height = int(round(original_height * scale_factor))

            ref_image_transform = transforms.Compose([
                transforms.Resize((resize_height, resize_width),
                                  interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop((target_height, target_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            ref_images_pixel_values = ref_image_transform(reference_image).unsqueeze(0).unsqueeze(2).to(
                self.execution_device)
            cond_latents = self.vae.encode(ref_images_pixel_values)

        else:
            raise ValueError(f"Unsupported task_type: {task_type}. Must be 't2v' or 'i2v'")

        return cond_latents

    def _prepare_cond_latents(self, task_type, cond_latents, latents, multitask_mask):
        latents_concat = None
        mask_concat = None

        if cond_latents is not None and task_type == 'i2v':
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3],
                                         latents.shape[4]).to(latents.device)

        mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_concat = self.merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(
            device=latents.device)

        cond_latents = torch.concat([latents_concat, mask_concat], dim=1)

        return cond_latents

    def merge_tensor_by_mask(self, tensor_1, tensor_2, mask, dim):
        if tensor_1.shape != tensor_2.shape:
            raise AssertionError("tensor_1.shape need to be the same with tensor_2.shape")
        # Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1
        masked_indices = torch.nonzero(mask).squeeze(1)
        tmp = tensor_1.clone()
        if dim == 0:
            tmp[masked_indices] = tensor_2[masked_indices]
        elif dim == 1:
            tmp[:, masked_indices] = tensor_2[:, masked_indices]
        elif dim == 2:
            tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
        return tmp

    def _prepare_vision_states(self, reference_image, target_resolution, latents, device):
        if reference_image is None:
            vision_states = torch.zeros(latents.shape[0], self.vision_num_semantic_tokens, self.vision_states_dim).to(
                latents.device)
        else:
            reference_image = np.array(reference_image) if isinstance(reference_image, Image) else reference_image
            if len(reference_image.shape) == 4:
                reference_image = reference_image[0]

            height, width = self.get_closest_resolution_given_reference_image(reference_image, target_resolution)

            # Encode reference image to vision states
            if self.vision_encoder is not None:
                input_image_np = resize_and_center_crop(reference_image, target_width=width, target_height=height)
                vision_states = self.vision_encoder.encode_images(input_image_np)
                vision_states = vision_states.last_hidden_state.to(device=device, dtype=self.target_dtype)
            else:
                vision_states = None

        # Repeat image features for batch size if needed (for classifier-free guidance)
        if self.do_classifier_free_guidance and vision_states is not None:
            vision_states = vision_states.repeat(2, 1, 1)

        return vision_states

    def get_latent_size(self, video_length, height, width):
        spatial_compression_ratio = self.vae_spatial_compression_ratio
        temporal_compression_ratio = self.vae_temporal_compression_ratio
        video_length = (video_length - 1) // temporal_compression_ratio + 1
        height, width = height // spatial_compression_ratio, width // spatial_compression_ratio

        size_cond = height > 0 and width > 0 and video_length > 0
        if not size_cond:
            raise AssertionError(f"height: {height}, width: {width}, video_length: {video_length}")

        return video_length, height, width

    def prepare_extra_func_kwargs(self, func, kwargs):
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs
