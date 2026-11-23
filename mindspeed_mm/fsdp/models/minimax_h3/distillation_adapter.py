# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""H3 noise-prediction interface, separate from the existing SFT loss path."""

from dataclasses import dataclass
import math

import torch

from mindspeed_mm.fsdp.checkpoint.convert import WEIGHT_TRANSFORM_PIPELINES
from mindspeed_mm.fsdp.utils.register import model_register
from mindspeed_mm.fsdp.models.model_container import ModelContainer
from .modeling_minimax_h3_fsdp2 import MiniMaxH3ForTraining

WEIGHT_TRANSFORM_PIPELINES["minimax_h3_distillation"] = WEIGHT_TRANSFORM_PIPELINES["minimax_h3"]


class H3DistillationRoles(ModelContainer):
    """Independent sharding roots; the frozen teacher never enters optimizer state."""

    checkpoint_submodels = ("student", "critic")

    def __init__(self, student, teacher, critic):
        teacher.requires_grad_(False)
        teacher.eval()
        super().__init__({"student": student, "teacher": teacher, "critic": critic})


@dataclass
class H3DistillationPrediction:
    """Flow velocities in scheduler convention: noise minus clean sample."""

    video_velocity: torch.Tensor
    audio_velocity: torch.Tensor


def _flow_time(sigma, name):
    # Schedules are prepared on the host. Reject a device tensor rather than
    # silently introducing device-to-host synchronization on every prediction.
    if not isinstance(sigma, (float, int)) or isinstance(sigma, bool):
        raise TypeError(f"{name} must be a host scalar in [0, 1]")
    if not math.isfinite(sigma) or not 0 <= sigma <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return 1.0 - float(sigma)


def conditioning_from_cache(batch):
    """Restore the existing H3 cache fields without recasting position IDs."""
    names = (
        "seq_len",
        "img_pos",
        "audio_pos",
        "text_pos",
        "update_mask",
        "token_tags",
        "cu_seqlens",
        "text_len",
        "audio_channel",
        "audio_t",
        "latent_t",
        "latent_h_patched",
        "latent_w_patched",
    )
    packed = {name: batch[f"packed_{name}"] for name in names}
    bits = batch["packed_img_position_ids_bits"]
    if bits.dtype != torch.int64:
        raise ValueError("Position IDs must retain their original FP64 bit pattern as int64")
    packed["img_position_ids"] = bits.view(torch.float64)
    packed["cond_rows"] = batch.get("packed_cond_rows", 0)
    packed["ref_audio_rows"] = batch.get("packed_ref_audio_rows", 0)
    return packed


@model_register.register("minimax_h3_distillation")
class MiniMaxH3ForDistillation(MiniMaxH3ForTraining):
    """Call through the wrapped module, never a private method of an FSDP peer.

    This module only predicts velocities. Rollout, DMD2 losses and role
    updates belong to the distillation task engine, not this adapter.
    """

    def forward(self, noisy_video, noisy_audio, *, batch, sigma_video, sigma_audio):
        t_video = _flow_time(sigma_video, "sigma_video")
        t_audio = _flow_time(sigma_audio, "sigma_audio")
        for name, noisy, clean in (
            ("video", noisy_video, batch["input_latents"]),
            ("audio", noisy_audio, batch["audio_input_latents"]),
        ):
            if noisy.shape != clean.shape:
                raise ValueError(f"Noisy {name} shape {tuple(noisy.shape)} != reference {tuple(clean.shape)}")
        if noisy_video.device != noisy_audio.device or noisy_video.dtype != noisy_audio.dtype:
            raise ValueError("Video and audio inputs must share device and dtype")
        video, audio = self._model_fn(
            noisy_video,
            noisy_audio,
            conditioning_from_cache(batch),
            batch["prompt_embeds"],
            t_video,
            t_audio,
            batch.get("keyframe_cond_anchor"),
            batch.get("ref_visual_anchor"),
            batch.get("ref_audio_anchor"),
        )
        # _model_fn already negates native H3 predictions into scheduler
        # velocity convention. Do not negate again or treat them as epsilon.
        return H3DistillationPrediction(video, audio)
