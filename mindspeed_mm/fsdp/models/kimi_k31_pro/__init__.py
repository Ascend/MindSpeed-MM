"""MindSpeed-MM FSDP2 adapter for Kimi-K3.1-Pro."""

from transformers import AutoConfig

from mindspeed_mm.fsdp.models.kimi_k3 import (
    KimiK3ForConditionalGeneration as _KimiK3ForConditionalGeneration,
)
from mindspeed_mm.fsdp.utils.register import model_register

from .configuration_kimi_k3 import KimiK3Config
from .modeling_kimi_linear import apply_kimi_k31_dsa_patch


class KimiK31ForConditionalGeneration(_KimiK3ForConditionalGeneration):
    @staticmethod
    def overwrite_transformer_config(transformer_config, model_args, feature_args):
        transformer_config = _KimiK3ForConditionalGeneration.overwrite_transformer_config(
            transformer_config, model_args, feature_args
        )
        dsa_implementation = getattr(model_args, "dsa_implementation", "eager")
        if dsa_implementation not in ("eager", "ascendc"):
            raise ValueError(
                f"Unsupported dsa_implementation: {dsa_implementation}. "
                "Expected 'eager' or 'ascendc'."
            )
        transformer_config.text_config.dsa_implementation = dsa_implementation
        return transformer_config


try:
    AutoConfig.register("kimi_k31", KimiK3Config)
except ValueError:
    pass

apply_kimi_k31_dsa_patch()
KimiK31ForConditionalGeneration = model_register.register("kimi_k31_pro")(
    KimiK31ForConditionalGeneration
)

__all__ = ["KimiK31ForConditionalGeneration"]
