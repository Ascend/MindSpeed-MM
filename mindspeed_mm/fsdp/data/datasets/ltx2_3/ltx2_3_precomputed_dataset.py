from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from mindspeed_mm.fsdp.models.ltx2_3.bootstrap import install_ltx2_3_aliases
from mindspeed_mm.fsdp.models.ltx2_3.npu_patch import apply_ltx2_3_npu_patch
from mindspeed_mm.fsdp.utils.register import data_register

install_ltx2_3_aliases()
apply_ltx2_3_npu_patch()

from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy


@dataclass
class LTX23DatasetArgs:
    dataset_dir: str = "/data/ltx2_3"
    max_samples: int | None = None
    first_frame_conditioning_p: float = 0.0
    with_audio: bool = True
    audio_latents_dir: str = "audio_latents"
    timestep_sampling_mode: str = "shifted_logit_normal"
    timestep_sampling_params: dict[str, Any] = field(default_factory=dict)


@data_register.register("ltx2_3_precomputed")
class LTX23PrecomputedDataset(torch.utils.data.Dataset):
    """MindSpeed-MM dataset adapter for the vendored LTX-2.3 precomputed dataset."""

    def __init__(self, basic_param, preprocess_param=None, dataset_param=None, **kwargs):
        _ = (preprocess_param, kwargs)
        args = self._resolve_args(basic_param, dataset_param)
        self.args = LTX23DatasetArgs(**args)

        strategy_config = TextToVideoConfig(
            first_frame_conditioning_p=self.args.first_frame_conditioning_p,
            with_audio=self.args.with_audio,
            audio_latents_dir=self.args.audio_latents_dir,
        )
        self.strategy = TextToVideoStrategy(strategy_config)
        self.dataset = PrecomputedDataset(
            data_root=self.args.dataset_dir,
            data_sources=self.strategy.get_data_sources(),
        )

        if self.args.max_samples is not None:
            self._length = min(int(self.args.max_samples), len(self.dataset))
        else:
            self._length = len(self.dataset)

    @staticmethod
    def _to_dict(obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return {}

    @classmethod
    def _resolve_args(cls, basic_param: Any, dataset_param: Any) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        dataset_param_dict = cls._to_dict(dataset_param)
        custom = dataset_param_dict.get("ltx2_3_dataset_custom", {})
        if isinstance(custom, dict):
            merged.update(custom)
        if isinstance(basic_param, dict):
            merged.update(basic_param)

        valid_keys = set(LTX23DatasetArgs.__dataclass_fields__)
        return {key: value for key, value in merged.items() if key in valid_keys}

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]

    def collate_fn(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        raw_batch = self._collate_values(features)
        batch: dict[str, Any] = {
            "latents": raw_batch["latents"],
            "conditions": raw_batch["conditions"],
            "first_frame_conditioning_p": self.args.first_frame_conditioning_p,
            "with_audio": self.args.with_audio,
            "timestep_sampling_mode": self.args.timestep_sampling_mode,
        }

        timestep_params = self.args.timestep_sampling_params or {}
        for key in ("std", "eps", "uniform_prob"):
            if key in timestep_params:
                batch[f"timestep_{key}"] = float(timestep_params[key])

        if raw_batch.get("audio_latents") is not None:
            batch["audio_latents"] = raw_batch["audio_latents"]

        if raw_batch.get("idx") is not None:
            batch["sample_idx"] = raw_batch["idx"]

        return batch

    def _collate_values(self, values: list[Any]) -> Any:
        first = values[0]
        if isinstance(first, torch.Tensor):
            if any((not isinstance(v, torch.Tensor)) or v.shape != first.shape for v in values):
                raise ValueError(f"Shape mismatch while collating LTX2.3 batch: {tuple(first.shape)}")
            return torch.stack(values, dim=0)
        if isinstance(first, dict):
            return {key: self._collate_values([value[key] for value in values]) for key in first.keys()}
        if isinstance(first, bool):
            return torch.tensor(values, dtype=torch.bool)
        if isinstance(first, int):
            return torch.tensor(values, dtype=torch.int64)
        if isinstance(first, float):
            return torch.tensor(values, dtype=torch.float32)
        return values
