from typing import Dict

import torch
import torch.nn as nn

from mindspeed_mm.fsdp.models.base_model import GenerativeBaseModel
from mindspeed_mm.fsdp.models.model_container import ModelContainer
from mindspeed_mm.fsdp.models.wan2_2.modeling_wan2_2 import (
    WanDiTFSDP2,
    get_wan_video_vae,
)
from mindspeed_mm.fsdp.models.wan2_2.text_encoder import build_text_encoder
from mindspeed_mm.fsdp.models.wan2_2.pipeline_wan2_2 import WanT2VPipeline
from mindspeed_mm.fsdp.utils.register import model_register
from mindspeed_mm.fsdp.utils.dtype import get_dtype


def _config_to_dict(config):
    return config.to_dict() if hasattr(config, "to_dict") else dict(config)


def _resolve_torch_dtype(value):
    if value in (None, ""):
        return None
    if isinstance(value, torch.dtype):
        return value
    return get_dtype(str(value).strip().lower())


# ---------------------------------------------------------------------------
# Bridge classes for sub-models that are NOT registered in model_register
# by the original codebase. These bridges allow ModelHub._build_custom_model
# to construct ae / text_encoder / predictor independently in the new
# model-list mode while keeping the legacy single-model path untouched.
# ---------------------------------------------------------------------------

@model_register.register("wan_video_vae")
class WanVideoVAEBridge(GenerativeBaseModel):
    """Bridge to build WanVideoVAE (ae) in model-list mode."""

    @classmethod
    def _from_config(cls, config):
        cfg = _config_to_dict(config)
        cfg.pop("freeze", None)
        cfg.pop("meta_init", None)
        target_dtype = _resolve_torch_dtype(cfg.get("dtype", None))
        if target_dtype is not None:
            cfg["dtype"] = target_dtype
        return get_wan_video_vae()(**cfg)

    @classmethod
    def from_pretrained(cls, config):
        return cls._from_config(config)


@model_register.register("UMT5")
class UMT5Bridge(GenerativeBaseModel):
    """Bridge to build UMT5 text encoder in model-list mode."""

    @classmethod
    def _from_config(cls, config):
        cfg = _config_to_dict(config)
        cfg.pop("freeze", None)
        cfg.pop("meta_init", None)
        return build_text_encoder(cfg)

    @classmethod
    def from_pretrained(cls, config):
        return cls._from_config(config)


class WanDiTFSDP2WithSetup(WanDiTFSDP2, GenerativeBaseModel):
    """WanDiTFSDP2 subclass keeping only the Wan-specific dtype re-cast and init-skipping guard on top of the ``GenerativeBaseModel`` default ``setup_weights``."""

    weight_transform_model_type = "wan2_2"

    @classmethod
    def from_pretrained(cls, config):
        raise NotImplementedError(
            "WanDiTFSDP2WithSetup is constructed via WanDiTFSDP2Bridge."
        )

    @classmethod
    def _from_config(cls, config):
        raise NotImplementedError(
            "WanDiTFSDP2WithSetup is constructed via WanDiTFSDP2Bridge."
        )

    def post_load(self, loaded: bool):
        target_dtype = getattr(self, "_wan_target_dtype", None)
        if target_dtype is not None and target_dtype != torch.float32:
            self.to(dtype=target_dtype)

    def init_weights(self):
        if getattr(self, "_weights_loaded", False):
            return
        super().init_weights()


@model_register.register("wan2_2_predictor")
class WanDiTFSDP2Bridge(GenerativeBaseModel):
    """Bridge to build the native WanDiTFSDP2 predictor in model-list mode."""

    @classmethod
    def _from_config(cls, config):
        cfg = _config_to_dict(config)
        cfg.pop("freeze", None)
        cfg.pop("meta_init", None)
        target_dtype = _resolve_torch_dtype(cfg.pop("dtype", None))
        model_backend = cfg.pop("model_backend", None)
        if model_backend not in (None, "", "native"):
            raise ValueError(
                "Wan2.2 FSDP2 ModelContainer only supports the native predictor. "
                "Remove 'model_backend' or set it to 'native'."
            )
        model = WanDiTFSDP2WithSetup(**cfg)
        if target_dtype is not None:
            model._wan_target_dtype = target_dtype
            model.to(dtype=target_dtype)
        return model

    @classmethod
    def from_pretrained(cls, config):
        return cls._from_config(config)


@model_register.register("wan2_2_container")
class Wan2_2ModelContainer(ModelContainer):
    """Wan2.2 multi-model container managing ae / text_encoder / predictor lifecycle; data flow is delegated to :class:`WanT2VPipeline`."""

    # Checkpoint policy is dynamic (trainable sub-models only); flat
    # (unprefixed) keys are treated as predictor weights for backward
    # compatibility with older checkpoints.
    legacy_flat_key_target = "predictor"

    def __init__(self, models: Dict[str, nn.Module], task: str = "t2v"):
        super().__init__(models)
        self.task = task
        # Inter-model data-flow orchestration (condition encoding, noising,
        # predictor forward, loss) lives in the pipeline; the container only
        # owns component management.  The pipeline is a plain object (not an
        # nn.Module), so named_modules() paths and FSDP patterns are unaffected.
        self.pipeline = WanT2VPipeline(task=task)
