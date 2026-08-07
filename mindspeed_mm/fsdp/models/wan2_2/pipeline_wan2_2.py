
"""Wan2.2 flow-matching T2V training pipeline (generic ``FlowMatchT2VPipeline`` + Wan2.2 specialization).
Data flow: condition encoding (grad-enabled for trainable sub-models) -> ``scheduler.q_sample`` noising -> predictor forward (grad) -> ``scheduler.training_losses``."""

import torch

from mindspeed_mm.fsdp.train.generative_pipeline import (
    GenerativePipelineBase,
    GenerativePipelineOutput,
    grad_context_for,
)
from mindspeed_mm.fsdp.models.wan2_2.modeling_wan2_2 import (
    Wan2_2ModelOutput,
    get_wan_flow_match_scheduler,
)


class FlowMatchT2VPipeline(GenerativePipelineBase):
    """Flow-matching T2V training pipeline shared by video DiT models."""

    # Sub-model role names on the container (default convention).
    ae_name = "ae"
    text_encoder_name = "text_encoder"
    predictor_name = "predictor"

    # Output type wrapping the loss; overridden by model-specific pipelines.
    output_type = GenerativePipelineOutput

    def __init__(self, scheduler):
        # Weight-free component only; the pipeline itself is not an nn.Module.
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # Stage 1: condition encoding (grad-enabled for trainable sub-models)
    # ------------------------------------------------------------------

    def prepare_conditions(
        self,
        container,
        video: torch.Tensor = None,
        prompt_ids: torch.Tensor = None,
        prompt_mask: torch.Tensor = None,
        video_mask: torch.Tensor = None,
        i2v_clip_feature: torch.Tensor = None,
        i2v_vae_feature: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        ae = getattr(container, self.ae_name, None)
        text_encoder = getattr(container, self.text_encoder_name, None)
        with grad_context_for(ae):
            latents = self.encode_latents(container, video, kwargs)
            i2v_vae_feature = self.prepare_image_conditions(
                container, video, latents, i2v_vae_feature
            )
        with grad_context_for(text_encoder):
            prompt, prompt_mask = self.encode_prompt(
                container, video, latents, prompt_ids, prompt_mask, kwargs
            )
        return {
            "latents": latents,
            "prompt": prompt,
            "prompt_mask": prompt_mask,
            "i2v_clip_feature": i2v_clip_feature,
            "i2v_vae_feature": i2v_vae_feature,
            # Extra batch kwargs threaded through q_sample / predictor /
            # training_losses; hooks may update this dict in place.
            "extra_kwargs": kwargs,
        }

    def encode_latents(self, container, video, extra_kwargs):
        """VAE-encode the video into latents (identity when no ae is attached)."""
        ae = getattr(container, self.ae_name, None)
        if ae is None:
            return video
        ae_dtype = next(ae.parameters()).dtype
        video_ae = video.to(ae_dtype) if video is not None else None
        encode_out = ae(video_ae)
        return encode_out[0] if isinstance(encode_out, tuple) else encode_out

    def prepare_image_conditions(self, container, video, latents, i2v_vae_feature):
        """Hook for image-conditioning models; default is a pass-through."""
        return i2v_vae_feature

    def encode_prompt(self, container, video, latents, prompt_ids, prompt_mask, extra_kwargs):
        """Text-encode the prompt (pass-through when no text encoder is attached)."""
        text_encoder = getattr(container, self.text_encoder_name, None)
        if text_encoder is None:
            return prompt_ids, prompt_mask
        if prompt_ids is None:
            return self.default_prompt(container, video, latents, prompt_mask)
        te_dtype = next(text_encoder.parameters()).dtype
        prompt_mask_te = (
            prompt_mask.to(te_dtype)
            if prompt_mask is not None and prompt_mask.is_floating_point()
            else prompt_mask
        )
        # Call through forward() so that FSDP2 pre/post hooks
        # (unshard/reshard) are triggered when the text encoder
        # is wrapped by FSDP2.
        prompt, prompt_mask = text_encoder(prompt_ids, prompt_mask_te, **extra_kwargs)
        return prompt, prompt_mask

    def default_prompt(self, container, video, latents, prompt_mask):
        """Fallback prompt when a text encoder is attached but prompt_ids is None."""
        return None, prompt_mask

    # ------------------------------------------------------------------
    # Stage 2: forward-diffusion noising
    # ------------------------------------------------------------------

    def noising(self, container, conditions, **batch) -> dict:
        kwargs = conditions["extra_kwargs"]
        noised_latents, noise, timesteps = self.scheduler.q_sample(
            conditions["latents"],
            model_kwargs=kwargs,
            mask=batch.get("video_mask"),
        )
        inputs = dict(conditions)
        inputs.update(
            noised_latents=noised_latents,
            noise=noise,
            timesteps=timesteps,
        )
        return inputs

    # ------------------------------------------------------------------
    # Stage 3: predictor forward (with gradients)
    # ------------------------------------------------------------------

    def predict(self, container, **inputs):
        predictor = getattr(container, self.predictor_name)
        return predictor(
            inputs["noised_latents"],
            timestep=inputs["timesteps"],
            prompt=inputs["prompt"],
            prompt_mask=inputs["prompt_mask"],
            i2v_clip_feature=inputs["i2v_clip_feature"],
            i2v_vae_feature=inputs["i2v_vae_feature"],
            **inputs["extra_kwargs"],
        )

    # ------------------------------------------------------------------
    # Stage 4: loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, container, conditions, inputs, model_output, **batch):
        loss = self.scheduler.training_losses(
            model_output=model_output,
            x_start=conditions["latents"],
            x_t=inputs["noised_latents"],
            noise=inputs["noise"],
            t=inputs["timesteps"],
            mask=batch.get("video_mask"),
            **inputs["extra_kwargs"],
        )
        return self.output_type(loss=loss)


class WanT2VPipeline(FlowMatchT2VPipeline):
    """Wan2.2 T2V/I2V training pipeline."""

    output_type = Wan2_2ModelOutput

    def __init__(self, task: str = "t2v"):
        super().__init__(scheduler=get_wan_flow_match_scheduler()())
        self.task = task

    def encode_latents(self, container, video, extra_kwargs):
        ae = getattr(container, self.ae_name, None)
        if ae is None:
            return video
        ae_dtype = next(ae.parameters()).dtype
        video_ae = video.to(ae_dtype) if video is not None else None
        if self.task == "t2v":
            encode_out = ae(video_ae)
            latents = encode_out[0] if isinstance(encode_out, tuple) else encode_out
        elif self.task == "i2v":
            ae_kwargs = {
                k: v for k, v in extra_kwargs.items()
                if k not in ("first_frame", "use_cache")
            }
            encode_out = ae(video_ae, **ae_kwargs)
            latents = encode_out[0] if isinstance(encode_out, tuple) else encode_out
            i2v_results = (
                encode_out[1]
                if isinstance(encode_out, tuple) and len(encode_out) > 1
                else None
            )
            if i2v_results is not None:
                extra_kwargs.update(i2v_results)
        else:
            raise NotImplementedError(f"Task {self.task} is not implemented!")
        return latents

    def prepare_image_conditions(self, container, video, latents, i2v_vae_feature):
        if (
            self.task == "i2v"
            and i2v_vae_feature is None
            and video is not None
        ):
            i2v_vae_feature = self._prepare_i2v_condition(container, video, latents)
        return i2v_vae_feature

    def default_prompt(self, container, video, latents, prompt_mask):
        B = (
            video.shape[0]
            if video is not None
            else latents.shape[0]
        )
        device = (
            video.device
            if video is not None
            else latents.device
        )
        text_dim = getattr(container, self.predictor_name).text_dim
        prompt = torch.zeros(
            (B, 1, text_dim), device=device, dtype=torch.float32
        )
        return prompt, prompt_mask

    def _prepare_i2v_condition(
        self, container, video: torch.Tensor, latents: torch.Tensor
    ) -> torch.Tensor:
        B, C, T, H, W = video.shape
        _, _, T_latent, H_latent, W_latent = latents.shape

        first_frame = video[:, :, 0:1, :, :]
        zeros = torch.zeros(
            B, C, T - 1, H, W, device=video.device, dtype=video.dtype
        )
        vae_input = torch.cat([first_frame, zeros], dim=2)

        ae = getattr(container, self.ae_name, None)
        if ae is None:
            raise RuntimeError("AE model is required for the I2V task but is not attached to the container.")
        with grad_context_for(ae):
            latent_condition = ae(vae_input)
            if isinstance(latent_condition, tuple):
                latent_condition = latent_condition[0]

        msk = torch.ones(
            B, T, H_latent, W_latent, device=latents.device, dtype=latents.dtype
        )
        msk[:, 1:] = 0
        msk = torch.cat(
            [msk[:, 0:1].repeat(1, 4, 1, 1), msk[:, 1:]], dim=1
        )
        msk = msk.view(B, (T + 3) // 4, 4, H_latent, W_latent).transpose(1, 2)

        i2v_vae_feature = torch.cat([msk, latent_condition], dim=1)
        return i2v_vae_feature
