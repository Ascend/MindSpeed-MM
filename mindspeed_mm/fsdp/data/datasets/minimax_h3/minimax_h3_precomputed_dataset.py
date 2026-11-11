"""DiffSynth MiniMax-H3 precomputed-cache dataset."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from mindspeed_mm.fsdp.utils.register import data_register


_PACKED_TENSOR_KEYS = (
    "img_pos",
    "audio_pos",
    "text_pos",
    "token_tags",
    "cu_seqlens",
)
_PACKED_INT_KEYS = (
    "seq_len",
    "text_len",
    "audio_channel",
    "audio_t",
    "latent_t",
    "latent_h_patched",
    "latent_w_patched",
)


def _param(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


@data_register.register("minimax_h3_precomputed")
class MiniMaxH3PrecomputedDataset(Dataset):
    """Load DiffSynth sft:data_process outputs and flatten packed."""

    def __init__(self, basic_param, **_):
        root = Path(_param(basic_param, "dataset_dir")).expanduser()
        self.cache_files = sorted(root.rglob("*.pth"))
        if not self.cache_files:
            raise ValueError(f"No .pth cache files found under {root}.")
        self.dataset_repeat = int(_param(basic_param, "dataset_repeat", 1))

    def __len__(self):
        return len(self.cache_files) * self.dataset_repeat

    def __getitem__(self, index):
        shared, positive, _ = torch.load(
            self.cache_files[index % len(self.cache_files)],
            map_location="cpu",
            # DiffSynth Ref2VA caches include PIL reference objects alongside
            # the encoded tensors consumed below.
            weights_only=False,
        )
        packed = positive["packed"]
        output = {
            "input_latents": shared["input_latents"],
            "audio_input_latents": shared["audio_input_latents"],
            "prompt_embeds": positive["prompt_embeds"],
            # The common device mover casts floating tensors to BF16. Keep the
            # original FP64 position IDs as bits and restore them in the model.
            "packed_img_position_ids_bits": packed["img_position_ids"]
            .contiguous()
            .view(torch.int64),
        }
        if "keyframe_cond_anchor" in shared:
            output["keyframe_cond_anchor"] = shared["keyframe_cond_anchor"]
        if "ref_blocks" in shared:
            visual_parts = [
                block["visual_rows"]
                for block in shared["ref_blocks"]
                if "visual_rows" in block
            ]
            audio_parts = [
                block["audio_rows"]
                for block in shared["ref_blocks"]
                if "audio_rows" in block
            ]
            if visual_parts:
                output["ref_visual_anchor"] = torch.cat(visual_parts, dim=0)
            if audio_parts:
                output["ref_audio_anchor"] = torch.cat(audio_parts, dim=0)
        for key in _PACKED_TENSOR_KEYS:
            output[f"packed_{key}"] = packed[key]
        for key in _PACKED_INT_KEYS:
            if key in packed:
                output[f"packed_{key}"] = int(packed[key])
        self._fill_packed_fields(output, shared, packed)
        return output

    @staticmethod
    def _fill_packed_fields(output, shared, packed):
        """Fill packed fields, deriving ones missing from newer DiffSynth caches.

        DiffSynth-Studio v2.1.0 caches store ``update_mask``, ``cond_rows`` and
        the packed metadata ints directly. Since the pipeline refactor
        (v2.1.5+), caches only carry ``img_pos``/``audio_pos``/``text_pos``/
        ``cu_seqlens``/``seq_len``/``token_tags``/``img_position_ids`` and the
        fields below are re-derived at train time exactly like DiffSynth's
        ``model_fn`` does. Both cache formats therefore end up with the same
        batch contract.

        Raises:
            ValueError: if the cached/derived values violate the packed
                geometry that ``model_fn`` relies on (latent shapes or
                ``update_mask`` length), so layout mismatches fail at load
                time instead of as reshape errors at the first training step.
        """
        video_latents = shared["input_latents"]
        audio_latents = shared["audio_input_latents"]
        # patchify_video consumes [B, C, T, H, W]; pack_audio consumes
        # [audio_channel, latent_dim, T] (no batch dim).
        if video_latents.dim() != 5:
            raise ValueError(
                f"input_latents must be [B, C, T, H, W], got "
                f"{tuple(video_latents.shape)}."
            )
        if audio_latents.dim() != 3:
            raise ValueError(
                f"audio_input_latents must be [audio_channel, latent_dim, T], "
                f"got {tuple(audio_latents.shape)}."
            )

        cond_rows = packed.get("cond_rows")
        if cond_rows is None:
            # ref2va anchors take precedence over fl2va keyframes, same as the
            # DiffSynth model_fn cond anchor selection.
            cond_anchor = output.get("ref_visual_anchor")
            if cond_anchor is None:
                cond_anchor = output.get("keyframe_cond_anchor")
            cond_rows = 0 if cond_anchor is None else cond_anchor.shape[0]

        ref_audio_rows = packed.get("ref_audio_rows")
        if ref_audio_rows is None:
            ref_audio_anchor = output.get("ref_audio_anchor")
            ref_audio_rows = 0 if ref_audio_anchor is None else ref_audio_anchor.shape[0]

        img_pos = packed["img_pos"]
        update_mask = packed.get("update_mask")
        if update_mask is None:
            # Every video row updates except the condition (reference /
            # keyframe) rows at the front of img_pos, which model_fn masks out
            # and excludes from the loss. Audio condition rows are handled by
            # model_fn slicing only; audio logits are gated by
            # update_audio_mask (None during training), not by update_mask.
            update_mask = torch.ones(img_pos.shape[0], dtype=torch.bool)
            if cond_rows:
                update_mask[:cond_rows] = False
        elif update_mask.shape[0] != img_pos.shape[0]:
            raise ValueError(
                f"update_mask length {update_mask.shape[0]} does not match "
                f"the number of img rows ({img_pos.shape[0]})."
            )

        latent_t = int(packed.get("latent_t", video_latents.shape[2]))
        latent_h_patched = int(
            packed.get("latent_h_patched", video_latents.shape[3] // 2)
        )
        latent_w_patched = int(
            packed.get("latent_w_patched", video_latents.shape[4] // 2)
        )
        audio_channel = int(packed.get("audio_channel", audio_latents.shape[0]))
        audio_t = int(packed.get("audio_t", audio_latents.shape[2]))

        # These row counts are what unpatchify_video / unpack_audio reshape
        # against; validate them here so mismatched layouts fail with a clear
        # message instead of a confusing reshape error at the first step.
        video_rows = latent_t * latent_h_patched * latent_w_patched
        if video_rows != img_pos.shape[0] - cond_rows:
            raise ValueError(
                f"latent_t*latent_h_patched*latent_w_patched ({video_rows}) "
                f"does not match the number of video rows "
                f"({img_pos.shape[0]} - cond_rows {cond_rows})."
            )
        audio_rows = audio_channel * audio_t
        if audio_rows != packed["audio_pos"].shape[0] - ref_audio_rows:
            raise ValueError(
                f"audio_channel*audio_t ({audio_rows}) does not match the "
                f"number of audio rows ({packed['audio_pos'].shape[0]} - "
                f"ref_audio_rows {ref_audio_rows})."
            )

        fields = {
            "packed_update_mask": update_mask,
            "packed_cond_rows": int(cond_rows),
            "packed_ref_audio_rows": int(ref_audio_rows),
            # DiffSynth builders store cu_seqlens as [0, used, seq_len], so the
            # last entry is the padded seq_len; only fills when the cache
            # omits seq_len (setdefault keeps an explicit value authoritative).
            "packed_seq_len": int(packed.get("seq_len", packed["cu_seqlens"][-1])),
            "packed_text_len": int(
                packed.get("text_len", packed["text_pos"].shape[0])
            ),
            "packed_audio_channel": audio_channel,
            "packed_audio_t": audio_t,
            "packed_latent_t": latent_t,
            "packed_latent_h_patched": latent_h_patched,
            "packed_latent_w_patched": latent_w_patched,
        }
        for key, value in fields.items():
            output.setdefault(key, value)

    @staticmethod
    def collate_fn(features):
        if len(features) != 1:
            raise ValueError(
                "MiniMax-H3 currently requires micro_batch_size=1; "
                f"received {len(features)} samples. Multi-sample packing is not implemented."
            )
        return features[0]
