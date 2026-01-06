from dataclasses import dataclass
from typing import Optional, Union, List

import PIL
import numpy as np
import torch
from torchvision import transforms

from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin

POSITIVE_MAGIC_T2V = "超高清、HDR 视频、环境光、杜比全景声、画面稳定、流畅动作、逼真的细节、专业级构图、超现实主义、自然、生动、超细节、清晰。"
NEGATIVE_MAGIC_T2V = "画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。"

POSITIVE_MAGIC_I2V = "画面中的主体动作表现生动自然、画面流畅、生动细节、光线统一柔和、超真实动态捕捉、大师级运镜、整体不变形、超高清、画面稳定、逼真的细节、专业级构图、超细节、清晰。"
NEGATIVE_MAGIC_I2V = "动画、模糊、变形、毁容、低质量、拼贴、粒状、标志、抽象、插图、计算机生成、扭曲、动作不流畅、面部有褶皱、表情僵硬、畸形手指"


@dataclass
class ImageLatentsConfig:
    image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]]
    batch_size: int
    num_channels_latents: int
    height: int
    width: int
    num_frames: int
    device: torch.device
    dtype: torch.dtype


class StepVideoPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):
    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model, config=None):
        self.vae = vae
        self.text_encoders = text_encoder
        self.tokenizers = tokenizer
        self.scheduler = scheduler
        self.predict_model = predict_model

        config = config.to_dict()
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.num_frames, self.height, self.width = config.get("input_size", [204, 768, 768])
        self.generator = torch.Generator(device="npu")
        self.motion_score = config.get("motion_score", 1.0)
        self.model_type = config.get("model_type", "t2v")

    @staticmethod
    def apply_template(text, template):
        if isinstance(text, str):
            return [template.format(text)]
        elif isinstance(text, list) or isinstance(text, tuple):
            return [template.format(one_text) for one_text in text]
        else:
            raise NotImplementedError(f"Not Support text type {type(text)}") 

    def check_inputs(self, num_frames, width, height):
        num_frames = max(num_frames // 17 * 17, 1)
        width = max(width // 16 * 16, 16)
        height = max(height // 16 * 16, 16)
        return num_frames, width, height

    def resize_to_desired_aspect_ratio(self, video, aspect_size):
        # video is in shape [f, c, h, w]
        height, width = video.shape[-2:]

        aspect_ratio = [w / h for h, w in aspect_size]
        # resize
        aspect_ratio_fact = width / height
        bucket_idx = np.argmin(np.abs(aspect_ratio_fact - np.array(aspect_ratio)))
        aspect_ratio = aspect_ratio[bucket_idx]
        target_size_height, target_size_width = aspect_size[bucket_idx]

        if aspect_ratio_fact < aspect_ratio:
            scale = target_size_width / width
        else:
            scale = target_size_height / height

        width_scale = int(round(width * scale))
        height_scale = int(round(height * scale))

        # crop
        delta_h = height_scale - target_size_height
        delta_w = width_scale - target_size_width
        if delta_w < 0 or delta_h < 0:
            raise ValueError("the delta_w and delta_h must be greater than or equal to 0.")

        top = delta_h // 2
        left = delta_w // 2

        # resize image and crop
        resize_crop_transform = transforms.Compose([
            transforms.Resize((height_scale, width_scale)),
            lambda x: transforms.functional.crop(x, top, left, target_size_height, target_size_width),
        ])

        video = torch.stack([resize_crop_transform(frame.contiguous()) for frame in video], dim=0)
        return video

    def prepare_image_latents(self, params: ImageLatentsConfig):
        num_frames, width, height = self.check_inputs(params.num_frames, params.width, params.height)
        img_tensor = transforms.ToTensor()(params.image[0].convert('RGB')) * 2 - 1
        img_tensor = self.resize_to_desired_aspect_ratio(img_tensor[None], aspect_size=[(height, width)])[None]
        img_tensor = img_tensor.to(params.dtype).to(params.device)
        img_emb = self.vae.encode(img_tensor).repeat(params.batch_size, 1, 1, 1, 1).to(params.device)

        padding_tensor = torch.zeros((params.batch_size, max(num_frames // 17 * 3, 1) - 1, params.num_channels_latents,
                                      int(height) // 16,
                                      int(width) // 16,), device=params.device)
        condition_hidden_states = torch.cat([img_emb, padding_tensor], dim=1)

        condition_hidden_states = condition_hidden_states.repeat(2, 1, 1, 1, 1)  # for CFG
        return condition_hidden_states.to(params.dtype)

    def get_positive_magic(self):
        if self.model_type == "t2v":
            return POSITIVE_MAGIC_T2V
        else:
            return POSITIVE_MAGIC_I2V

    def get_negative_magic(self):
        if self.model_type == "t2v":
            return NEGATIVE_MAGIC_T2V
        else:
            return NEGATIVE_MAGIC_I2V

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        device: torch.device = "npu",
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
        height = self.height
        width = self.width

        # 1. Check inputs
        self.text_prompt_checks(
            prompt,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds
        )

        if image is not None:
            self.image_prompt_checks(image)

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        do_classifier_free_guidance = self.guidance_scale > 1
        
        # 3. Encode input prompt
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.get_negative_magic()
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")
        
        if isinstance(prompt, str):
            prompt = [prompt + self.get_positive_magic()]
        elif isinstance(prompt, list) or isinstance(prompt, tuple):
            prompt = [one_text + self.get_positive_magic() for one_text in prompt]
        else:
            raise NotImplementedError(f"Not Support text type {type(prompt)}")

        # Text Encoder load to device
        self.text_encoders.to(device)

        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self.encode_texts(
            prompt,
            device,
            tokenizer=self.tokenizers[0],
            text_encoder=self.text_encoders[0],
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
            max_length=self.tokenizers[0].model_max_length,
            use_prompt_preprocess=use_prompt_preprocess
        )
        clip_embedding, clip_mask, negative_clip_embedding, negative_clip_mask = self.encode_texts(
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
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])# 2, s1, d1
            clip_embedding = torch.cat([negative_clip_embedding, clip_embedding, ]).to(prompt_embeds.dtype)# 2, s2, d2 
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
                clip_mask = torch.cat([negative_clip_mask, clip_mask])

        # Text Encoder offload to `cpu`
        self.text_encoders.to("cpu")
        torch.cuda.empty_cache()

        # prepare image_latents for i2v task
        if image is not None:
            params = ImageLatentsConfig(
                image=image,
                batch_size=batch_size,
                num_channels_latents=self.predict_model.in_channels,
                height=height,
                width=width,
                num_frames=self.num_frames,
                device=device,
                dtype=prompt_embeds.dtype
            )
            image_latents = self.prepare_image_latents(params=params)
        else:
            image_latents = None

        # 4. Prepare latents
        latent_channels = self.predict_model.in_channels
        shape = (
            batch_size,
            max(1, self.num_frames // self.vae.frame_len * self.vae.latent_len),
            latent_channels,
            int(self.height) // 16,
            int(self.width) // 16,
        )
        latents = self.prepare_latents(
            shape,
            generator=self.generator,
            device=device,
            dtype=prompt_embeds.dtype
        )

        # 5. Denoising
        prompt_embeds = prompt_embeds.unsqueeze(1)  # b s d -> b 1 s d
        clip_embedding = clip_embedding.unsqueeze(1)  # b s d -> b 1 s d
        self.predict_model.to(device)
        latents = self.scheduler.sample(
            model=self.predict_model,
            latents=latents.to(self.predict_model.dtype),
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=self.guidance_scale,
            model_kwargs={"prompt": [prompt_embeds, clip_embedding], "prompt_mask": [prompt_mask, clip_mask],
                          "motion_score": self.motion_score, "image_latents": image_latents}
        )
        # predict model offload to 'cpu'
        self.predict_model.to("cpu")
        torch.cuda.empty_cache()

        # 6. Decode
        video = self.decode_latents(latents.to(self.vae.dtype))  # b t c h w
        
        return video