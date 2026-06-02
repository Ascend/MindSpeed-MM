import inspect
from typing import Optional, Union, List
import os
from PIL.Image import Image
import torch
from accelerate import cpu_offload_with_hook
from diffusers.video_processor import VideoProcessor
import transformers
from transformers import CLIPImageProcessor

from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin


NEGATIVE_PROMPT = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"


class HunyuanVideoPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds"
    ]

    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model, config=None):
        self.predict_model = predict_model
        self.vae = vae
        self.text_encoders = text_encoder
        self.tokenizers = tokenizer
        self.scheduler = scheduler

        config = config.to_dict()
        self.generator = torch.Generator().manual_seed(config.get("seed", 1234))
        self.frames, self.height, self.width = config.get("input_size", [65, 256, 256])
        self.generate_params_checks(self.frames, self.height, self.width)
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.guidance_rescale = config.get("guidance_rescale", 0.0)
        self.embedded_guidance_scale = config.get("embedded_guidance_scale", None)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.tokenizers[0].init_kwargs["name_or_path"])

        self.eta = config.get("eta", 0.0)
        self.cpu_offload = config.get("cpu_offload", False)
        if self.cpu_offload:
            local_rank = int(os.getenv("LOCAL_RANK"))
            self.enable_model_cpu_offload(local_rank)

    @staticmethod
    def generate_params_checks(frames, height, width):
        if height % 16 != 0 or width % 16 != 0 or (frames - 1) % 4 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

    def enable_model_cpu_offload(self, npu_id: Optional[int] = 0, device: Union[torch.device, str] = "npu"):
        torch_device = torch.device(device)

        device = torch.device(f"{torch_device.type}:{npu_id or torch_device.index or 0}")

        model_sequence = [
            self.text_encoders[0],
            self.text_encoders[1],
            self.predict_model,
            self.vae
        ]
        hook = None

        for cpu_offload_model in model_sequence:
            cpu_offload_model.cpu()
            _, hook = cpu_offload_with_hook(cpu_offload_model, device, prev_module_hook=hook)

    def prepare_extra_func_kwargs(self, func, kwargs):
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

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
        **kwargs
    ):
        """
        The call function to the pipeline for generation

        Inputs:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide video/image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in video/image generation.
                Ignored when not using guidance (`guidance_scale < 1`)
        Returns:
            video (`torch.Tensor` or `List[torch.Tensor]`)
        """

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

        do_classifier_free_guidance = self.guidance_scale > 1

        # 3. Encode input prompt
        if image is not None:
            image_tensor = self.video_processor.preprocess(image, self.height, self.width).to(device, self.vae.dtype)
            img_latents = self.vae.encode(image_tensor.unsqueeze(2))
        else:
            img_latents = None

        i2v_kwargs = {
            "i2v_mode": image is not None,
            "i2v_condition_type": "token_replace",
            "img_latents": img_latents
        }

        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self.mllm_encode(
            prompt=prompt,
            device=device,
            tokenizer=self.tokenizers[0],
            text_encoder=self.text_encoders[0],
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            image=image,
        )

        prompt_embeds_2, _, negative_prompt_embeds_2, _ = self.encode_texts(
            prompt,
            device,
            tokenizer=self.tokenizers[1],
            text_encoder=self.text_encoders[1],
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            clip_skip=clip_skip,
            max_length=self.tokenizers[1].model_max_length,
            use_prompt_preprocess=use_prompt_preprocess
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])

        # 4. Prepare latents
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.predict_model.in_channels,
            height=self.height // self.vae.spatial_compression_ratio,
            width=self.width // self.vae.spatial_compression_ratio,
            video_length=(self.frames - 1) // self.vae.time_compression_ratio + 1,
            generator=self.generator,
            device=device,
            dtype=prompt_embeds.dtype,
            latents=None,
            img_latents=img_latents,
            i2v_mode=image is not None,
            i2v_condition_type="token_replace"
        )

        # 5. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": self.generator, "eta": self.eta}
        )

        # 6. denoise
        latents = self.scheduler.sample(
            model=self.predict_model,
            latents=latents.to(self.predict_model.dtype),
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=self.guidance_scale,
            guidance_rescale=self.guidance_rescale,
            embedded_guidance_scale=self.embedded_guidance_scale,
            model_kwargs={"prompt": [prompt_embeds, prompt_embeds_2], "prompt_mask": prompt_mask},
            extra_step_kwargs=extra_step_kwargs,
            **i2v_kwargs
        )

        if hasattr(self.vae, "shift_factor"):
            latents = latents / self.vae.scaling_factor + self.vae.shift_factor
        else:
            latents = latents / self.vae.scaling_factor

        self.vae.enable_tiling()
        video = self.decode_latents(latents.to(self.vae.dtype))

        return video

    def mllm_encode(
            self,
            prompt,
            device,
            tokenizer,
            text_encoder,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            image=None
    ):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        if image is not None:
            image_embeds = self.image_processor(image, return_tensors="pt").pixel_values.to(device)
            image_input_kwargs = {"pixel_values": image_embeds}
        else:
            image_input_kwargs = {}

        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = text_encoder(
            input_ids,
            attention_mask=attention_mask,
            **image_input_kwargs
        )

        if hasattr(text_encoder, "output_key"):
            prompt_embeds = prompt_embeds[text_encoder.output_key]
        elif isinstance(prompt_embeds, transformers.utils.ModelOutput):
            prompt_embeds = prompt_embeds[0]
        if hasattr(text_encoder, "hidden_state_skip_layer") and text_encoder.hidden_state_skip_layer is not None:
            prompt_embeds = prompt_embeds[-(text_encoder.hidden_state_skip_layer + 1)]

        # negative
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [""] * len(prompt)

            uncond_inputs = tokenizer(
                negative_prompt,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )

            if image is not None:
                # i2v task, black image
                uncond_images = [Image.new("RGB", (image.size[0], image.size[1]), (0, 0, 0)) for img in image]
                uncond_image_embeds = self.image_processor(uncond_images, return_tensors="pt").pixel_values.to(device)
                image_input_kwargs = {"pixel_values": uncond_image_embeds}
            else:
                image_input_kwargs = {}

            uncond_input_ids = uncond_inputs.input_ids.to(device)
            uncond_attention_mask = uncond_inputs.attention_mask.to(device)
            uncond_prompt_embeds = text_encoder(
                input_ids=uncond_input_ids,
                attention_mask=uncond_attention_mask,
                **image_input_kwargs
            )

        else:
            uncond_prompt_embeds = None
            uncond_attention_mask = None

        return prompt_embeds, attention_mask, uncond_prompt_embeds, uncond_attention_mask

    def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            video_length,
            generator,
            device,
            dtype,
            latents=None,
            img_latents=None,
            i2v_mode=False,
            i2v_condition_type=None,
            i2v_stability=True
    ):
        if i2v_mode and i2v_condition_type == "latent_concat":
            num_channels_latents = (num_channels_latents - 1) // 2

        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            int(height),
            int(width),
        )

        if i2v_mode and i2v_stability:
            if img_latents.shape[2] == 1:
                img_latents = img_latents.repeat(1, 1, video_length, 1, 1)
            x0 = super().prepare_latents(shape, generator=generator, device=device, dtype=dtype)
            x1 = img_latents

            t = torch.tensor([0.999]).to(device=device)
            latents = x0 * t + x1 * (1 - t)
            latents = latents.to(dtype=dtype)

        if latents is None:
            latents = super().prepare_latents(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents.to(device=device, dtype=dtype)

        return latents
