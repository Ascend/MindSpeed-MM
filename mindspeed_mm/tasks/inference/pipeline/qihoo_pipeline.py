from typing import Optional, Union, List, Callable

import math
import torch
from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin


class QihooPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):

    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model, config=None):
        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler,
                              predict_model=predict_model)

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.predict_model = predict_model
        text_encoder.use_attention_mask = config.pipeline_config.use_attention_mask
        self.num_frames, self.height, self.width = config.pipeline_config.input_size
        self.guidance_scale = config.diffusion.cfg_scale
        self.max_sequence_length = config.model_max_length
        self.vae_type = config.ae.model_id

    @torch.no_grad()
    def __call__(self,
                 prompt,
                 prompt_embeds: Optional[torch.Tensor] = None,
                 negative_prompt: Optional[str] = None,
                 negative_prompt_embeds: Optional[torch.Tensor] = None,
                 eta: float = 0.0,
                 num_images_per_prompt: Optional[int] = 1,
                 guidance_scale: float = 4.5,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 latents: Optional[torch.FloatTensor] = None,
                 clean_caption: bool = True,
                 fps: int = None,
                 enable_temporal_attentions: bool = True,
                 added_cond_kwargs: dict = None,
                 model_args: Optional[dict] = None,
                 device: torch.device = "npu",
                 dtype: torch.dtype = None,
                 **kwargs,
                 ):

        # 1 check prompts
        self.text_prompt_checks(prompt, negative_prompt, prompt_embeds, negative_prompt_embeds)

        prompt = self.preprocess_text(prompt, clean=clean_caption)

        # 2
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = self.guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, prompt_embeds_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = self.encode_texts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_length=self.max_sequence_length,
            clean_caption=False,
            prompt_to_lower=False
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
            prompt_embeds_attention_mask = torch.cat([prompt_embeds_attention_mask, negative_prompt_attention_mask], dim=0)

        if model_args:
            model_args.update(dict(encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_embeds_attention_mask))
        else:
            model_args = dict(encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_embeds_attention_mask)

        model_args["fps"] = torch.tensor([fps], device=device, dtype=dtype).repeat(batch_size)
        model_args["height"] = torch.tensor([self.height], device=device, dtype=dtype).repeat(batch_size)
        model_args["width"] = torch.tensor([self.width], device=device, dtype=dtype).repeat(batch_size)
        model_args["num_frames"] = torch.tensor([self.num_frames], device=device, dtype=dtype).repeat(batch_size)

        # 5. Prepare latents
        image_size = (self.height, self.width)
        batch_size = batch_size * num_images_per_prompt
        input_size = (self.num_frames, *image_size)
        try:
            latent_size = self.vae.get_latent_size(input_size)
            shape = (batch_size, self.vae.out_channels, *latent_size)
        except Exception as e:
            latent_size = (
                (math.ceil((int(self.num_frames) - 1) / self.vae.vae_scale_factor[0]) + 1) if int(
                self.num_frames) % 2 == 1 else math.ceil(int(self.num_frames) / self.vae.vae_scale_factor[0]),
                math.ceil(int(self.height) / self.vae.vae_scale_factor[1]),
                math.ceil(int(self.width) / self.vae.vae_scale_factor[2]),
            )
            shape = (batch_size, self.predict_model.in_channels, *latent_size)

        z = torch.randn(shape, device=device, dtype=dtype)
        latents = self.scheduler.sample(model=self.predict_model, shape=shape, clip_denoised=False, latents=z,
                                        mask=None,
                                        model_kwargs=model_args, progress=True)  # b,c,t,h,w

        if self.vae_type in ["casualvae", "wfvae"]:
            video = self.decode_latents(latents.to(self.vae.dtype))
            video = video.permute(0, 2, 1, 3, 4)  # [b,t,c,h,w -> [b,c,t,h,w]
        return video
