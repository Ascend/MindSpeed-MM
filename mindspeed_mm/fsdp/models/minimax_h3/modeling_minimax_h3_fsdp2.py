"""MiniMax-H3 DiT adapter for MindSpeed-MM FSDP2 training."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mindspeed_mm.fsdp.checkpoint.convert import (
    WEIGHT_TRANSFORM_PIPELINES,
    WeightTransformPipeline,
)
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.models.base_model import BaseModel, WeightInitMixin
from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.utils.register import model_register

from .vendor.diffsynth.diffusion.flow_match import FlowMatchScheduler
from .vendor.diffsynth.models.minimax_constant import (
    AUDIO_COND_NOISE_AUG,
    IMGVID_COND_NOISE_AUG,
)
from .vendor.diffsynth.models.minimax_h3_dit import (
    MiniMaxH3DiT,
    pack_audio,
    patchify_video,
    unpack_audio,
    unpatchify_video,
)


@dataclass
class MiniMaxH3TrainingOutput:
    loss: torch.Tensor


class _MiniMaxH3WeightTransform(WeightTransformPipeline):
    def __init__(self, hf_dir=None, mtp_num_layers=None):
        # MiniMax-H3 uses identity key mapping. Accept the common loader
        # constructor contract even though neither argument is needed here.
        del hf_dir, mtp_num_layers
        super().__init__()

    def hf_to_dcp(self, key, tensor):
        if tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16).to(torch.float32)
        return key, tensor

    def dcp_to_hf(self, key, tensor):
        return {key: tensor}


WEIGHT_TRANSFORM_PIPELINES["minimax_h3"] = _MiniMaxH3WeightTransform


def _load_model_config(config: ModelArguments):
    config_path = Path(config.model_name_or_path) / "config.json"
    with config_path.open(encoding="utf-8") as file:
        model_config = json.load(file)
    if getattr(config, "dit_num_layers", None) is not None:
        model_config["num_layers"] = int(config.dit_num_layers)
    return model_config


@model_register.register("minimax_h3")
class MiniMaxH3ForTraining(WeightInitMixin, MiniMaxH3DiT, BaseModel):
    """Preserve the DiffSynth DiT module tree and add the SFT loss."""

    def __init__(
        self,
        *args,
        loss_replay_seed=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_replay_seed = loss_replay_seed
        self.loss_replay_step = 0
        self.scheduler = FlowMatchScheduler("MiniMax-H3")
        self.scheduler_audio = FlowMatchScheduler("MiniMax-H3")
        self.scheduler.set_timesteps(1000, training=True, shift=12.0)
        self.scheduler_audio.set_timesteps(1000, training=True, shift=3.0)

    @classmethod
    def _from_config(cls, config: ModelArguments):
        return cls(
            **_load_model_config(config),
            loss_replay_seed=getattr(config, "loss_replay_seed", None),
            attn_implementation=config.attn_implementation,
        )

    @classmethod
    def from_pretrained(cls, config: ModelArguments):
        # Formal configs build on meta and let MindSpeed-MM's HF checkpointer load weights.
        return cls._from_config(config)

    @staticmethod
    def _restore_position_ids(position_ids_bits):
        return position_ids_bits.view(torch.float64)

    def _model_fn(
        self,
        video_latents,
        audio_latents,
        packed,
        prompt_embeds,
        t_video,
        t_audio,
        keyframe_cond_anchor,
        ref_visual_anchor,
        ref_audio_anchor,
    ):
        device = video_latents.device
        dtype = video_latents.dtype
        video_rows = patchify_video(video_latents)
        audio_rows = pack_audio(audio_latents)

        seq_len = packed["seq_len"]
        img_pos = packed["img_pos"]
        audio_pos = packed["audio_pos"]
        text_pos = packed["text_pos"]
        text_len = packed["text_len"]
        cond_rows = packed["cond_rows"]
        ref_audio_rows = packed["ref_audio_rows"]

        x = torch.zeros(1, seq_len, 96, dtype=dtype, device=device)
        audio_x = torch.zeros(1, seq_len, 32, dtype=dtype, device=device)
        if ref_visual_anchor is not None:
            # ref2va
            x[0].index_copy_(0, img_pos[:cond_rows], ref_visual_anchor)
            x[0].index_copy_(0, img_pos[cond_rows:], video_rows)
        elif keyframe_cond_anchor is not None:
            # fl2va
            x[0].index_copy_(0, img_pos[:cond_rows], keyframe_cond_anchor)
            x[0].index_copy_(0, img_pos[cond_rows:], video_rows)
        else:
            # t2va
            x[0].index_copy_(0, img_pos, video_rows)

        if ref_audio_anchor is not None:
            audio_x[0].index_copy_(0, audio_pos[:ref_audio_rows], ref_audio_anchor)
            audio_x[0].index_copy_(0, audio_pos[ref_audio_rows:], audio_rows)
        else:
            audio_x[0].index_copy_(0, audio_pos, audio_rows)

        timesteps = torch.full((seq_len,), float(t_video), dtype=torch.float32, device=device)
        timesteps[audio_pos] = float(t_audio)
        if cond_rows:
            timesteps[img_pos[:cond_rows]] = max(float(t_video), IMGVID_COND_NOISE_AUG)
        if ref_audio_rows:
            timesteps[audio_pos[:ref_audio_rows]] = max(
                float(t_audio), AUDIO_COND_NOISE_AUG
            )
        unique_timesteps, inverse_indices = torch.unique(
            timesteps, sorted=True, return_inverse=True
        )
        refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)

        video_rows_pred, audio_rows_pred = super().forward(
            x=x,
            audio_x=audio_x,
            img_position_ids=packed["img_position_ids"],
            unique_timesteps=unique_timesteps,
            inverse_indices=inverse_indices,
            update_mask=packed["update_mask"],
            token_tags=packed["token_tags"],
            prompt_embeds=prompt_embeds,
            img_pos_info={"position_ids": img_pos},
            audio_pos_info={"position_ids": audio_pos},
            text_pos_info={"position_ids": text_pos},
            img_pos_for_infer_output_info={"position_ids": img_pos},
            packed_seq_params={
                "cu_seqlens_q": packed["cu_seqlens"],
                "max_seqlen_q": int(packed["cu_seqlens"][1]),
            },
            refiner_packed_seq_params={
                "cu_seqlens_q": refiner_cu,
                "max_seqlen_q": text_len,
            },
            skip_mask_out_condition=False,
        )

        if cond_rows:
            video_rows_pred = video_rows_pred[cond_rows:]
        if ref_audio_rows:
            audio_rows_pred = audio_rows_pred[ref_audio_rows:]
        video_pred = unpatchify_video(
            video_rows_pred,
            packed["latent_t"],
            packed["latent_h_patched"],
            packed["latent_w_patched"],
        )
        audio_pred = unpack_audio(
            audio_rows_pred,
            packed["audio_channel"],
            packed["audio_t"],
        )
        return -video_pred, -audio_pred

    def forward(
        self,
        input_latents,
        audio_input_latents,
        prompt_embeds,
        packed_seq_len,
        packed_img_pos,
        packed_audio_pos,
        packed_text_pos,
        packed_update_mask,
        packed_img_position_ids_bits,
        packed_token_tags,
        packed_cu_seqlens,
        packed_text_len,
        packed_audio_channel,
        packed_audio_t,
        packed_latent_t,
        packed_latent_h_patched,
        packed_latent_w_patched,
        packed_cond_rows=0,
        packed_ref_audio_rows=0,
        keyframe_cond_anchor=None,
        ref_visual_anchor=None,
        ref_audio_anchor=None,
        use_cache=False,
        **_,
    ):
        del use_cache
        packed = {
            "seq_len": packed_seq_len,
            "img_pos": packed_img_pos,
            "audio_pos": packed_audio_pos,
            "text_pos": packed_text_pos,
            "update_mask": packed_update_mask,
            "img_position_ids": self._restore_position_ids(
                packed_img_position_ids_bits
            ),
            "token_tags": packed_token_tags,
            "cu_seqlens": packed_cu_seqlens,
            "text_len": packed_text_len,
            "audio_channel": packed_audio_channel,
            "audio_t": packed_audio_t,
            "latent_t": packed_latent_t,
            "latent_h_patched": packed_latent_h_patched,
            "latent_w_patched": packed_latent_w_patched,
            "cond_rows": packed_cond_rows,
            "ref_audio_rows": packed_ref_audio_rows,
        }
        generator = None
        if self.loss_replay_seed is not None:
            if torch.distributed.is_initialized():
                parallel_state = get_parallel_state()
                data_parallel_rank = parallel_state.get_dp_rank()
                data_parallel_size = parallel_state.get_dp_group_size()
            else:
                data_parallel_rank = 0
                data_parallel_size = 1
            seed = (
                self.loss_replay_seed
                + self.loss_replay_step * data_parallel_size
                + data_parallel_rank
            )
            self.loss_replay_step += 1
            generator = torch.Generator("cpu").manual_seed(seed)
        timestep_id = torch.randint(0, len(self.scheduler.timesteps), (1,), generator=generator)
        # Scheduler lookup and scalar conversion are CPU operations. Keeping the
        # timestep on CPU avoids an NPU -> CPU synchronization in add_noise and
        # training_weight while preserving the existing dtype rounding.
        timestep_video = self.scheduler.timesteps[timestep_id].to(
            dtype=input_latents.dtype
        )
        timestep_audio = self.scheduler_audio.timesteps[timestep_id].to(
            dtype=input_latents.dtype
        )

        if generator is None:
            video_noise = torch.randn_like(input_latents)
            audio_noise = torch.randn_like(audio_input_latents)
        else:
            video_noise = torch.randn(
                input_latents.shape, generator=generator, dtype=input_latents.dtype
            ).to(input_latents.device)
            audio_noise = torch.randn(
                audio_input_latents.shape,
                generator=generator,
                dtype=audio_input_latents.dtype,
            ).to(audio_input_latents.device)
        video_latents = self.scheduler.add_noise(
            input_latents, video_noise, timestep_video
        )
        video_target = self.scheduler.training_target(
            input_latents, video_noise, timestep_video
        )
        audio_latents = self.scheduler_audio.add_noise(
            audio_input_latents, audio_noise, timestep_audio
        )
        audio_target = self.scheduler_audio.training_target(
            audio_input_latents, audio_noise, timestep_audio
        )

        video_pred, audio_pred = self._model_fn(
            video_latents,
            audio_latents,
            packed,
            prompt_embeds,
            1.0 - float(timestep_video) / self.scheduler.num_train_timesteps,
            1.0 - float(timestep_audio) / self.scheduler_audio.num_train_timesteps,
            keyframe_cond_anchor,
            ref_visual_anchor,
            ref_audio_anchor,
        )
        video_loss = F.mse_loss(video_pred.float(), video_target.float())
        video_loss *= self.scheduler.training_weight(timestep_video)
        audio_loss = F.mse_loss(audio_pred.float(), audio_target.float())
        audio_loss *= self.scheduler_audio.training_weight(timestep_audio)
        return MiniMaxH3TrainingOutput(loss=video_loss + audio_loss)
