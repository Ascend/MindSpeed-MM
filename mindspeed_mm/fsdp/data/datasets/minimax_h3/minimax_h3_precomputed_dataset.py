"""DiffSynth MiniMax-H3 precomputed-cache dataset."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from mindspeed_mm.fsdp.utils.register import data_register


_PACKED_TENSOR_KEYS = (
    "img_pos",
    "audio_pos",
    "text_pos",
    "update_mask",
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
            "packed_cond_rows": int(packed.get("cond_rows", 0)),
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
            output[f"packed_{key}"] = int(packed[key])
        if "ref_audio_rows" in packed:
            output["packed_ref_audio_rows"] = int(packed["ref_audio_rows"])
        return output

    @staticmethod
    def collate_fn(features):
        if len(features) != 1:
            raise ValueError(
                "MiniMax-H3 currently requires micro_batch_size=1; "
                f"received {len(features)} samples. Multi-sample packing is not implemented."
            )
        return features[0]
