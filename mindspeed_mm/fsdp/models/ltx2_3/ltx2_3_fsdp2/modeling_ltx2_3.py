from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import types
from typing import Any

import torch

from mindspeed_mm.fsdp.models.base_model import BaseModel
from mindspeed_mm.fsdp.models.ltx2_3.bootstrap import install_ltx2_3_aliases
from mindspeed_mm.fsdp.models.ltx2_3.npu_patch import apply_ltx2_3_npu_patch
from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.utils.register import model_register

install_ltx2_3_aliases()
apply_ltx2_3_npu_patch()

from mindspeed_mm.fsdp.models.ltx2_3.ltx2_3_fsdp2.modified import (
    _process_transformer_blocks,
    block_forward,
    model_forward,
)
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.loader.helpers import read_model_config
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.model_configurator import (
    LTXModelConfigurator,
    LTXV_MODEL_COMFY_RENAMING_MAP,
)
from ltx_core.text_encoders.gemma.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    EmbeddingsProcessor,
    convert_to_additive_mask,
)
from ltx_trainer.timestep_samplers import SAMPLERS
from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy


@dataclass
class LTX23ModelOutput:
    loss: torch.Tensor
    video_pred: torch.Tensor | None = None
    audio_pred: torch.Tensor | None = None


class _EmbeddingsConnectorOnlyConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> EmbeddingsProcessor:
        return EmbeddingsProcessor(
            video_connector=Embeddings1DConnectorConfigurator.from_config(config),
            audio_connector=AudioEmbeddings1DConnectorConfigurator.from_config(config),
            feature_extractor=None,
        )



@model_register.register("ltx2_3")
class LTX23ForTraining(torch.nn.Module, BaseModel):
    """Thin MindSpeed-MM FSDP2 adapter around vendored LTX-2.3 modules."""

    _checkpoint_conversion_mapping = {
        r"^text_embedding_projection\.aggregate_embed\.": "embeddings_processor.feature_extractor.aggregate_embed.",
        r"^text_embedding_projection\.video_aggregate_embed\.": "embeddings_processor.feature_extractor.video_aggregate_embed.",
        r"^text_embedding_projection\.audio_aggregate_embed\.": "embeddings_processor.feature_extractor.audio_aggregate_embed.",
        r"^model\.diffusion_model\.video_embeddings_connector\.": "embeddings_processor.video_connector.",
        r"^model\.diffusion_model\.audio_embeddings_connector\.": "embeddings_processor.audio_connector.",
        r"^model\.diffusion_model\.": "transformer.",
    }

    def __init__(
        self,
        transformer: LTXModel,
        embeddings_processor: EmbeddingsProcessor | None = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.embeddings_processor = embeddings_processor

        if self.embeddings_processor is not None:
            self.embeddings_processor.requires_grad_(False)
            self.embeddings_processor.eval()

        self.transformer._process_transformer_blocks = types.MethodType(
            _process_transformer_blocks, self.transformer
        )
        for block in self.transformer.transformer_blocks:
            block.forward = types.MethodType(block_forward, block)
        self.transformer.forward = types.MethodType(model_forward, self.transformer)

    @staticmethod
    def _to_mapping(obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return {key: value for key, value in vars(obj).items() if not key.startswith("_")}
        return {}

    @classmethod
    def _normalized_transformer_config(cls, model_args: ModelArguments) -> dict[str, Any]:
        transformer_cfg = {
            **cls._checkpoint_transformer_config(model_args),
            **cls._to_mapping(getattr(model_args, "transformer", {})),
        }
        if "audio_num_attention_heads" not in transformer_cfg:
            transformer_cfg["audio_num_attention_heads"] = transformer_cfg.get("num_attention_heads", 32)

        transformer_cfg.setdefault("connector_num_attention_heads", transformer_cfg.get("num_attention_heads", 32))
        transformer_cfg.setdefault("connector_attention_head_dim", transformer_cfg.get("attention_head_dim", 128))
        transformer_cfg.setdefault("audio_connector_num_attention_heads", transformer_cfg["audio_num_attention_heads"])
        transformer_cfg.setdefault(
            "audio_connector_attention_head_dim",
            transformer_cfg.get("audio_attention_head_dim", transformer_cfg.get("attention_head_dim", 128)),
        )
        return transformer_cfg

    @classmethod
    def _checkpoint_transformer_config(cls, model_args: ModelArguments) -> dict[str, Any]:
        ckpt_path = getattr(model_args, "checkpoint_path", None) or getattr(model_args, "model_name_or_path", None)
        if not ckpt_path:
            return {}

        ckpt_path = Path(str(ckpt_path)).expanduser()
        if not ckpt_path.is_file():
            return {}

        config = read_model_config(str(ckpt_path), SafetensorsModelStateDictLoader())
        return dict(cls._to_mapping(config.get("transformer", {})))

    @classmethod
    def _build_transformer_from_config(cls, model_args: ModelArguments) -> LTXModel:
        transformer_cfg = cls._normalized_transformer_config(model_args)
        transformer = LTXModelConfigurator.from_config({"transformer": transformer_cfg})
        return cls._limit_transformer_layers(transformer, model_args)

    @staticmethod
    def _limit_transformer_layers(
        transformer: LTXModel,
        model_args: ModelArguments,
    ) -> LTXModel:
        num_layers = getattr(model_args, "transformer_num_layers", None)
        if num_layers is None:
            return transformer
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(
                f"`transformer_num_layers` must be a positive integer, got {num_layers!r}."
            )

        total_layers = len(transformer.transformer_blocks)
        if num_layers > total_layers:
            raise ValueError(
                f"transformer_num_layers={num_layers} exceeds checkpoint model depth "
                f"({total_layers})."
            )
        transformer.transformer_blocks = torch.nn.ModuleList(
            list(transformer.transformer_blocks[:num_layers])
        )
        return transformer

    @classmethod
    def _from_config(cls, config: ModelArguments) -> "LTX23ForTraining":
        transformer = cls._build_transformer_from_config(config)
        embeddings_processor = cls._build_embeddings_processor_from_config(config)
        return cls(transformer=transformer, embeddings_processor=embeddings_processor)

    @classmethod
    def from_pretrained(cls, config: ModelArguments) -> "LTX23ForTraining":
        ckpt_path = getattr(config, "checkpoint_path", None) or config.model_name_or_path
        if ckpt_path is None:
            raise ValueError("`model_name_or_path` or `checkpoint_path` must be provided for LTX2.3.")

        ckpt_path = str(Path(ckpt_path).expanduser())
        builder = SingleGPUModelBuilder(
            model_path=ckpt_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        )
        if getattr(config, "transformer_num_layers", None) is not None:
            builder = builder.with_module_ops(
                (
                    ModuleOps(
                        name="limit_transformer_layers",
                        matcher=lambda model: hasattr(model, "transformer_blocks"),
                        mutator=lambda model: cls._limit_transformer_layers(model, config),
                    ),
                )
            )
        transformer = builder.build(device=torch.device("cpu"), dtype=torch.float32)

        embeddings_processor = cls._build_embeddings_processor(config, ckpt_path)
        return cls(transformer=transformer, embeddings_processor=embeddings_processor)

    @staticmethod
    def _build_embeddings_processor(config: ModelArguments, checkpoint_path: str) -> EmbeddingsProcessor | None:
        if not bool(getattr(config, "load_embeddings_processor", True)):
            return None

        embeddings_processor = SingleGPUModelBuilder(
            model_path=checkpoint_path,
            model_class_configurator=_EmbeddingsConnectorOnlyConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
        ).build(device=torch.device("cpu"), dtype=torch.bfloat16)
        return embeddings_processor

    @staticmethod
    def _build_embeddings_processor_from_config(config: ModelArguments) -> EmbeddingsProcessor | None:
        if not bool(getattr(config, "load_embeddings_processor", True)):
            return None
        transformer_cfg = LTX23ForTraining._normalized_transformer_config(config)
        return _EmbeddingsConnectorOnlyConfigurator.from_config({"transformer": transformer_cfg})

    def _ensure_embeddings_processor_device(self, device: torch.device) -> None:
        if self.embeddings_processor is None:
            return
        first_param = next(self.embeddings_processor.parameters(), None)
        if first_param is None:
            return
        if first_param.device != device or first_param.dtype != torch.bfloat16:
            self.embeddings_processor.to(device=device, dtype=torch.bfloat16)

    def _apply_embeddings_processor(self, conditions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.embeddings_processor is None:
            if "video_prompt_embeds" not in conditions:
                raise KeyError("`video_prompt_embeds` is required when embeddings processor is disabled.")
            return conditions

        if "video_prompt_embeds" in conditions:
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            video_features = conditions["prompt_embeds"]
            audio_features = conditions["prompt_embeds"]

        mask = conditions["prompt_attention_mask"]
        self._ensure_embeddings_processor_device(video_features.device)
        additive_mask = convert_to_additive_mask(mask, video_features.dtype)
        video_embeds, audio_embeds, attention_mask = self.embeddings_processor.create_embeddings(
            video_features,
            audio_features,
            additive_mask,
        )

        out = dict(conditions)
        out["video_prompt_embeds"] = video_embeds
        out["audio_prompt_embeds"] = audio_embeds
        out["prompt_attention_mask"] = attention_mask
        return out

    @staticmethod
    def _build_timestep_sampler(
        mode: str,
        std: float | None = None,
        eps: float | None = None,
        uniform_prob: float | None = None,
    ):
        sampler_cls = SAMPLERS.get(mode)
        if sampler_cls is None:
            raise ValueError(f"Unsupported timestep sampler: {mode}")

        params: dict[str, float] = {}
        if std is not None:
            params["std"] = float(std)
        if eps is not None:
            params["eps"] = float(eps)
        if uniform_prob is not None:
            params["uniform_prob"] = float(uniform_prob)
        return sampler_cls(**params)

    def forward(
        self,
        latents: dict[str, torch.Tensor],
        conditions: dict[str, torch.Tensor],
        first_frame_conditioning_p: float = 0.0,
        with_audio: bool = True,
        timestep_sampling_mode: str = "shifted_logit_normal",
        timestep_std: float | None = None,
        timestep_eps: float | None = None,
        timestep_uniform_prob: float | None = None,
        audio_latents: dict[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> LTX23ModelOutput:
        for key in ("latents", "num_frames", "height", "width"):
            if latents.get(key) is None:
                raise ValueError(
                    "LTX2.3 batches must include latents, num_frames, height, width, "
                    "and prompt_attention_mask."
                )
        if conditions.get("prompt_attention_mask") is None:
            raise ValueError(
                "LTX2.3 batches must include prompt_attention_mask."
            )

        conditions = self._apply_embeddings_processor(conditions)

        raw_batch: dict[str, Any] = {"latents": latents, "conditions": conditions}
        if audio_latents is not None:
            raw_batch["audio_latents"] = audio_latents

        strategy = TextToVideoStrategy(
            TextToVideoConfig(
                first_frame_conditioning_p=float(first_frame_conditioning_p),
                with_audio=bool(with_audio),
            )
        )
        timestep_sampler = self._build_timestep_sampler(
            timestep_sampling_mode,
            std=timestep_std,
            eps=timestep_eps,
            uniform_prob=timestep_uniform_prob,
        )
        model_inputs = strategy.prepare_training_inputs(raw_batch, timestep_sampler)

        video_pred, audio_pred = self.transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )
        per_sample_loss = strategy.compute_loss(video_pred, audio_pred, model_inputs)
        loss = per_sample_loss.mean()
        return LTX23ModelOutput(loss=loss, video_pred=video_pred, audio_pred=audio_pred)
