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

"""Validated reference samples on top of the existing H3 cache reader."""

import torch

from mindspeed_mm.fsdp.utils.register import data_register
from .minimax_h3_precomputed_dataset import MiniMaxH3PrecomputedDataset, _param, flatten_h3_cache


def validate_h3_reference(batch, *, expected_video_shape=None, expected_audio_shape=None, check_finite=False):
    """Validate on the host, before distributing reference samples to devices.

    Shape expectations are optional user settings. Cache metadata alone cannot
    establish which VAE/version produced it; callers must also match provenance.
    """
    video, audio, prompt = (batch[name] for name in ("input_latents", "audio_input_latents", "prompt_embeds"))
    if video.ndim != 5 or video.shape[:2] != (1, 24) or min(video.shape) < 1:
        raise ValueError("H3 reference video must be [1, 24, T, H, W]")
    if video.shape[-2] % 2 or video.shape[-1] % 2:
        raise ValueError("H3 latent H/W must be divisible by spatial patch size 2")
    if audio.ndim != 3 or audio.shape[1] != 32 or min(audio.shape) < 1:
        raise ValueError("H3 reference audio must be [channels, 32, T]")
    if prompt.ndim != 2 or min(prompt.shape) < 1:
        raise ValueError("H3 prompt embeddings must be nonempty [tokens, features]")
    for name, tensor, expected in (
        ("video", video, expected_video_shape),
        ("audio", audio, expected_audio_shape),
    ):
        if not tensor.is_floating_point():
            raise ValueError(f"Reference {name} must be floating point")
        if expected is not None and tuple(tensor.shape) != tuple(expected):
            raise ValueError(f"Reference {name} shape {tuple(tensor.shape)} != configured {tuple(expected)}")
    if not prompt.is_floating_point():
        raise ValueError("Prompt embeddings must be floating point")
    if check_finite and not all(torch.isfinite(t).all().item() for t in (video, audio, prompt)):
        raise ValueError("Reference sample contains non-finite values")
    integers = {
        "latent_t": video.shape[2],
        "latent_h_patched": video.shape[3] // 2,
        "latent_w_patched": video.shape[4] // 2,
        "audio_channel": audio.shape[0],
        "audio_t": audio.shape[2],
        "text_len": prompt.shape[0],
    }
    for name, actual in integers.items():
        if batch[f"packed_{name}"] != actual:
            raise ValueError(f"packed_{name} disagrees with reference tensor: expected {actual}")
    seq = batch["packed_seq_len"]
    bounds = batch["packed_cu_seqlens"]
    if (
        bounds.dtype not in (torch.int32, torch.int64)
        or bounds.ndim != 1
        or bounds.numel() < 2
        or bounds[0] != 0
        or bounds[-1] != seq
    ):
        raise ValueError("Packed sequence boundaries must start at zero and cover seq_len")
    if (bounds[1:] < bounds[:-1]).any() or not 0 < int(bounds[1]) <= seq:
        raise ValueError("Invalid packed sequence boundaries")
    # One real sample, followed only by optional isolated alignment padding.
    valid = int(bounds[1])
    if (bounds[1:-1] != valid).any():
        raise ValueError("Only one real sample and optional isolated alignment padding are supported")
    if batch["packed_token_tags"].shape != (seq,):
        raise ValueError("Packed token_tags must have seq_len rows")
    bits = batch["packed_img_position_ids_bits"]
    if bits.dtype != torch.int64 or bits.shape != (1, seq, 3):
        raise ValueError("Position IDs must be [1, seq_len, 3] int64 FP64 bit patterns")
    positions = []
    for name in ("text", "img", "audio"):
        pos = batch[f"packed_{name}_pos"]
        if pos.ndim != 1 or pos.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"packed_{name}_pos must be a one-dimensional integer tensor")
        if pos.numel() and ((pos < 0).any() or (pos >= valid).any()):
            raise ValueError(f"packed_{name}_pos is outside the real sample")
        positions.append(pos.to(torch.int64))
    all_positions = torch.cat(positions)
    if all_positions.numel() != valid or torch.unique(all_positions).numel() != valid:
        raise ValueError("Modality positions must cover the real sample exactly once")
    cond = batch.get("packed_cond_rows", 0)
    ref_audio = batch.get("packed_ref_audio_rows", 0)
    counts = (
        prompt.shape[0],
        video.shape[2] * (video.shape[3] // 2) * (video.shape[4] // 2) + cond,
        audio.shape[0] * audio.shape[2] + ref_audio,
    )
    if tuple(pos.numel() for pos in positions) != counts:
        raise ValueError("Packed modality lengths disagree with reference and condition rows")
    if batch["packed_update_mask"].numel() != counts[1]:
        raise ValueError("Video update_mask must include all video and condition rows")
    if cond:
        # MiniMaxH3ForTraining._build_inputs uses the reference anchor when it is
        # present (ref2va), otherwise the keyframe anchor (fl2va). Only that
        # effective anchor feeds the condition rows, so only it is validated.
        anchor = batch.get("ref_visual_anchor")
        if anchor is None:
            anchor = batch.get("keyframe_cond_anchor")
        if anchor is None:
            raise ValueError("Visual condition rows require a matching [rows, 96] keyframe or reference anchor")
        shape = tuple(anchor.shape) if isinstance(anchor, torch.Tensor) else None
        if shape != (cond, 96):
            raise ValueError(f"Visual condition anchor shape {shape} != required {(cond, 96)}")
    elif any(key in batch for key in ("keyframe_cond_anchor", "ref_visual_anchor")):
        raise ValueError("Visual anchor is present but packed_cond_rows is zero")
    if ref_audio and ("ref_audio_anchor" not in batch or batch["ref_audio_anchor"].shape != (ref_audio, 32)):
        raise ValueError("Audio reference rows require a matching [rows, 32] anchor")
    if not ref_audio and "ref_audio_anchor" in batch:
        raise ValueError("Audio anchor is present but packed_ref_audio_rows is zero")
    tags = batch["packed_token_tags"]
    if tags.dtype not in (torch.int32, torch.int64):
        raise ValueError("Packed token_tags must be integer")
    for pos, tag in zip(positions, (1, 0, 2)):
        if (tags[pos] != tag).any():
            raise ValueError("Packed modality tags disagree with position indices")


@data_register.register("minimax_h3_distillation_precomputed")
class MiniMaxH3DistillationDataset(MiniMaxH3PrecomputedDataset):
    def __init__(self, basic_param, **kwargs):
        super().__init__(basic_param, **kwargs)
        if self.dataset_repeat < 1:
            raise ValueError("dataset_repeat must be positive")
        self.expected_video_shape = _param(basic_param, "expected_video_shape")
        self.expected_audio_shape = _param(basic_param, "expected_audio_shape")
        self.check_finite = _param(basic_param, "check_finite", False)
        if not isinstance(self.check_finite, bool):
            raise ValueError("check_finite must be a YAML boolean")
        self.load_negative = _param(basic_param, "load_negative", False)
        if not isinstance(self.load_negative, bool):
            raise ValueError("load_negative must be a YAML boolean")
        self.cache_root = str(_param(basic_param, "dataset_dir"))

    def __getitem__(self, index):
        source = self.cache_files[index % len(self.cache_files)]
        shared, positive, negative = torch.load(source, map_location="cpu", weights_only=False)
        batch = flatten_h3_cache(shared, positive, complete_layout=True)
        try:
            validate_h3_reference(
                batch,
                expected_video_shape=self.expected_video_shape,
                expected_audio_shape=self.expected_audio_shape,
                check_finite=self.check_finite,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid H3 reference cache {source}: {error}") from error
        batch["sample_id"] = str(source)
        if self.load_negative:
            if not isinstance(negative, dict) or "packed" not in negative or "prompt_embeds" not in negative:
                raise ValueError(f"Teacher CFG needs precomputed negative embeddings and packed layout: {source}")
            negative_batch = flatten_h3_cache(shared, negative, complete_layout=True)
            validate_h3_reference(negative_batch, check_finite=self.check_finite)
            batch["negative_batch"] = negative_batch
        return batch
