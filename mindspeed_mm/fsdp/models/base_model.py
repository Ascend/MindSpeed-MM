import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from mindspeed_mm.fsdp.params.model_args import ModelArguments


logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Base Model Abstract Base Class
    All custom models should inherit from this class and implement the required methods.
    """

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        config: ModelArguments,
    ):
        """
        Load model from pretrained weights.

        Args:
            config: ModelArguments

        Returns:
            Loaded model instance
        """
        pass

    @classmethod
    @abstractmethod
    def _from_config(cls, config: ModelArguments) -> "BaseModel":
        """Create a model instance from configuration without loading pretrained weights (meta-device or from-scratch init)."""
        pass


class GenerativeBaseModel(BaseModel):
    """Base class for generative-model (diffusion family) components.

    Adds the unified weight-setup flow on top of :class:`BaseModel`:
    ``setup_weights`` loads pretrained weights (HF/Diffusers/DCP) or falls
    back to random init, so each sub-model (ae / text_encoder / predictor)
    is built and loaded through the same interface.
    """

    # Registry key of ``WEIGHT_TRANSFORM_PIPELINES`` selecting the
    # key-conversion pipeline for diffusers checkpoint sources (e.g.
    # ``"wan2_2"``).  ``None`` loads weights without key conversion.
    # Can be overridden per sub-model via the ``weight_transform_model_type``
    # yaml config key, so selecting a pipeline needs no code change.
    weight_transform_model_type: Optional[str] = None

    def get_weight_load_module(self):
        """Module that receives checkpoint weights; defaults to ``self``, wrappers override to point at the inner module."""
        return self

    def resolve_weight_source(self, config):
        """Resolve ``(ckpt_path, declared_format)`` from the sub-model config.

        ``checkpoint_path`` wins over ``from_pretrained`` when set;
        ``declared_format`` is the optional ``load_format`` config value
        (``"auto"``/``"dcp"``/``"hf"``/``"diffusers"``); overridable for
        construction-time fallbacks.
        """
        cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        ckpt_path = cfg.get("checkpoint_path") or cfg.get("from_pretrained")
        declared_format = cfg.get("load_format") or "auto"
        return ckpt_path, declared_format

    def setup_weights(self, config):
        """Default generative-model weight setup: pretrained load or random init; a failed load raises.

        Loading reuses the mainline checkpoint interfaces directly: DCP
        directories go through ``load_dcp_weights``; everything else (HF or
        diffusers safetensors, distinguished by filename inside
        ``locate_hf_weight_files``) goes through ``load_hf_weights`` with the
        key-conversion pipeline selected by ``weight_transform_model_type``.
        """
        from mindspeed_mm.fsdp.checkpoint.convert import build_weight_transform
        from mindspeed_mm.fsdp.checkpoint.dcp_utils import (
            load_dcp_weights,
            looks_like_dcp_checkpoint_dir,
        )
        from mindspeed_mm.fsdp.checkpoint.hf_utils import load_hf_weights
        from mindspeed_mm.fsdp.utils.utils import setup_module_weights

        module = self.get_weight_load_module()
        ckpt_path, declared_format = self.resolve_weight_source(config)
        cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        # The yaml key takes priority over the class-attribute default.
        model_type = cfg.get("weight_transform_model_type") or self.weight_transform_model_type

        def load_fn():
            is_dcp = declared_format == "dcp" or (
                declared_format == "auto" and looks_like_dcp_checkpoint_dir(ckpt_path)
            )
            logger.info(
                "Loading %s weights: path=%s format=%s",
                type(self).__name__,
                ckpt_path,
                "dcp" if is_dcp else "safetensors",
            )
            if is_dcp:
                load_dcp_weights(module, ckpt_path)
            else:
                weight_transform = None
                if model_type is not None:
                    weight_transform = build_weight_transform(
                        model_type, hf_dir=ckpt_path
                    )
                # Raises on failure: silently falling back to random
                # initialization would train from garbage weights.
                load_hf_weights(module, ckpt_path, weight_transform=weight_transform)
            return True

        def init_fn():
            if hasattr(self, "init_weights"):
                self.init_weights()

        loaded = setup_module_weights(
            module=module,
            ckpt_path=ckpt_path,
            load_fn=load_fn,
            init_fn=init_fn,
        )
        self._weights_loaded = loaded
        self.post_load(loaded)

    def post_load(self, loaded: bool):
        """Hook invoked after :meth:`setup_weights`; default no-op (``loaded`` marks pretrained vs random-init fallback)."""
        pass


class WeightInitMixin:
    """
    Weight Initialization Mixin Class

    Provides general model weight initialization functionality, supporting multiple layer types
    and composite model structures. Can be used as a mixin class with other torch.nn.Module subclasses.
    """

    def _init_weights(self, module, std=0.02):
        """
        Initialize the weights. This is quite general on purpose, in the spirit of what we usually do. For more complex
        initialization scheme, it should be overridden by the derived `PreTrainedModel` class. In case a model adds an explicit
        `nn.Parameter`, this method should also be overridden in order to initialize it correctly.
        """
        if getattr(module, "_is_initialized", False):
            return

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding) and module.padding_idx is None:
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.MultiheadAttention):
            # This uses torch's original init
            module._reset_parameters()
        # We cannot use `isinstance` on the RMSNorms or LayerNorms, as they usually are custom modules which change names
        # between modelings (because they are prefixed with the model name)
        elif (
                isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                or "norm" in module.__class__.__name__.lower()
        ):
            # Norms can exist without weights (in which case they are None from torch primitives)
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        # 3. Added: Generic parameter scanning and initialization for unhandled module types
        else:
            # Scan all Parameter attributes of the module
            for name, param in module.named_parameters(recurse=False):
                # Only process parameters that directly belong to this module (not recursive to submodules)
                if "weight" in name.lower():
                    param.data.normal_(mean=0.0, std=std)
                elif "bias" in name.lower():
                    param.data.zero_()
                else:
                    # Use default initialization for unknown parameter types
                    param.data.normal_(mean=0.0, std=std)

        module._is_initialized = True

    @torch.no_grad()
    def init_weights(self):
        """
        This is equivalent to calling `self.apply(self._initialize_weights)`, but correctly handles composite models.
        This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
        module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
        model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
        is extremely error prone and inefficient.

        Note that the `torch.no_grad()` decorator is very important as well, as most of our `_init_weights` do not use
        `torch.nn.init` functions (which are all no_grad by default), but simply do in-place ops such as
        `module.weight.data.zero_()`.
        """

        # This function is equivalent to `torch.nn.Module.apply`, except that it dynamically adjust the function
        # to apply as we go down the graph
        def smart_apply(self, fn):
            for module in self.children():
                module.smart_apply(fn)
            fn(self)
            return self

        torch.nn.Module.smart_apply = smart_apply

        # Let the magic happen with this simple call
        self.smart_apply(self._init_weights)
