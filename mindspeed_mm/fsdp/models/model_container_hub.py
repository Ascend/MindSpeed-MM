import os
import logging

import torch

from mindspeed.fsdp.utils.str_match import module_name_match
from mindspeed.fsdp.utils.log import print_rank

from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.params.feature_args import FeatureArguments
from mindspeed_mm.fsdp.params.training_args import TrainingArguments
from mindspeed_mm.fsdp.utils.register import model_register
from mindspeed_mm.fsdp.models.modelhub import ModelHub
from mindspeed_mm.fsdp.utils.device import get_device_type


logger = logging.getLogger(__name__)


def _parse_bool_field(value, default: bool = False) -> bool:
    """Parse a boolean-ish config value robustly (accepts booleans and strings)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return bool(value) if value is not None else default


class ModelContainerHub:
    """Dedicated hub for building ModelContainer-based (multi-sub-model) models; does not affect the legacy single-model build flow."""

    @staticmethod
    def _apply_freeze(model: torch.nn.Module, freeze_cfg):
        """Freeze modules according to the per-sub-model freeze configuration."""
        if isinstance(freeze_cfg, bool) and freeze_cfg:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            print_rank(logger.info, "freezing sub-model (global)...")
        elif isinstance(freeze_cfg, list) and len(freeze_cfg) > 0:
            freezed_named_modules = []
            for name, module in model.named_modules():
                for pattern in freeze_cfg:
                    if module_name_match(pattern, name):
                        freezed_named_modules.append((name, module))
            for name, module in freezed_named_modules:
                print_rank(logger.info, f"freezing module {name}...")
                for param in module.parameters():
                    param.requires_grad_(False)

    @staticmethod
    def build(
        model_args: ModelArguments,
        feature_args: FeatureArguments,
        training_args: TrainingArguments,
    ):
        """Build a ModelContainer by constructing each sub-model independently."""
        models_config = getattr(model_args, "models", None)
        if not models_config:
            raise ValueError(
                "ModelContainerHub.build() requires model_args.models to be set. "
                "For single-model builds, use ModelHub.build() instead."
            )

        container_type = getattr(model_args, "model_id", None)
        if container_type is None:
            raise ValueError(
                "model_id must be specified when using models list "
                "(e.g., model_id: wan2_2_container)."
            )

        container_cls = model_register.get(container_type)
        if container_cls is None:
            raise ValueError(
                f"Container type '{container_type}' is not registered in MODEL_MAPPINGS."
            )

        built_models = {}
        orig_meta_init = training_args.init_model_with_meta_device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        target_device = torch.device(f"{get_device_type()}:{local_rank}")

        for name, sub_cfg in models_config.items():
            if not isinstance(sub_cfg, dict):
                raise ValueError(
                    f"Each entry under 'models' must be a dict, got {type(sub_cfg)} for '{name}'."
                )

            sub_cfg = dict(sub_cfg)  # shallow copy
            freeze_cfg = sub_cfg.pop("freeze", False)
            meta_init_cfg = sub_cfg.pop("meta_init", orig_meta_init)

            # Normalise meta_init so that string "false" (from YAML) is handled
            # correctly in addition to the native boolean false.
            use_meta_init = _parse_bool_field(meta_init_cfg, default=orig_meta_init)

            # Build ModelArguments for the sub-model (exclude per-model control keys)
            sub_model_args = ModelArguments(**sub_cfg)

            # Per-sub-model meta-init control.
            #
            # Each sub-model declares its own ``meta_init`` field in YAML
            # (e.g. ``meta_init: false`` for ae).  It is passed explicitly to
            # ModelHub.build(), so frozen sub-models are built with real
            # weights instead of meta tensors.
            sub_model = ModelHub.build(sub_model_args, feature_args, training_args, meta_init=use_meta_init)

            # Call setup_weights() to load pretrained weights (or random-init
            # fallback) through the unified GenerativeBaseModel flow. Skipped
            # when a DCP checkpoint is being loaded, because DCP load will
            # overwrite the weights anyway; the hasattr guard keeps
            # non-generative custom models (no setup_weights) working.
            if (
                hasattr(sub_model, "setup_weights")
                and callable(getattr(sub_model, "setup_weights"))
                and training_args.load is None
            ):
                sub_model.setup_weights(sub_model_args)

            # Per-model freeze (bool or list of patterns)
            ModelContainerHub._apply_freeze(sub_model, freeze_cfg)

            # Move non-meta frozen models to the accelerator immediately so FSDP2 does not
            # manage them. Sub-models may opt out via ``_ms_mm_skip_to_device``
            # (e.g. UMT5 text encoder CPU offload). Opted-out models remain on CPU and are handled by
            # FSDP2/CPUOffloadPolicy instead.
            # Record whether this sub-model was built with meta-init so the
            # trainer knows whether it needs explicit weight initialization.
            sub_model._ms_mm_meta_init = use_meta_init

            # CPU-offloaded (skip_to_device) sub-models only run forward on the
            # fly; a trainable sub-model must stay on device for backward.
            is_globally_frozen = isinstance(freeze_cfg, bool) and freeze_cfg
            if not is_globally_frozen and getattr(sub_model, "_ms_mm_skip_to_device", False):
                raise ValueError(
                    f"Sub-model '{name}' is trainable (freeze is not true) but requests "
                    "skip_to_device/CPU offload, which is incompatible with backward. "
                    "Disable skip_to_device or freeze the sub-model."
                )

            if (
                not use_meta_init
                and (isinstance(freeze_cfg, bool) and freeze_cfg)
                and not getattr(sub_model, "_ms_mm_skip_to_device", False)
            ):
                print_rank(logger.info, f"moving sub-model <{name}> to {target_device}...")
                sub_model = sub_model.to(target_device)

            built_models[name] = sub_model

        # Collect extra kwargs for container construction.  ``task`` selects
        # the container's data-flow variant.
        container_kwargs = {}
        for key in ("task",):
            value = getattr(model_args, key, None)
            if value is None:
                value = model_args.to_dict().get(key, None)
            if value is not None:
                container_kwargs[key] = value

        container = container_cls(built_models, **container_kwargs)
        return container
