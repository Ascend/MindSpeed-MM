import logging
from typing import Dict, Optional, Tuple

import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys


logger = logging.getLogger(__name__)


class ModelContainer(nn.Module):
    """Lightweight container for top-level sub-models (own meta-init/parallel/freeze policy), registered as direct
    attributes for flat FSDP pattern paths; ``forward`` delegates to an attached ``pipeline`` unless overridden.
    ``checkpoint_submodels`` selects sub-models entering checkpoints (default: trainable ones); ``legacy_flat_key_target`` receives prefix-less keys."""

    checkpoint_submodels: Optional[Tuple[str, ...]] = None
    legacy_flat_key_target: Optional[str] = None
    pipeline = None

    def __init__(self, models: Dict[str, nn.Module]):
        super().__init__()
        for name, model in models.items():
            setattr(self, name, model)
        self._model_names = list(models.keys())

    def forward(self, *args, **kwargs):
        if self.pipeline is not None:
            return self.pipeline.forward(self, *args, **kwargs)
        raise NotImplementedError(
            "ModelContainer.forward() must be overridden by a subclass."
        )

    def train(self, mode: bool = True):
        """Sub-models follow ``mode``; fully frozen sub-models are forced back to eval (e.g. to keep frozen dropout off)."""
        super().train(mode)
        for name in self._model_names:
            sub_model = getattr(self, name, None)
            if sub_model is not None and self._is_fully_frozen(sub_model):
                sub_model.eval()
        return self

    @staticmethod
    def _is_fully_frozen(module: nn.Module) -> bool:
        params = list(module.parameters())
        return len(params) > 0 and all(not p.requires_grad for p in params)

    def get_sub_models(self) -> Dict[str, nn.Module]:
        """Return a shallow copy of the sub-model mapping."""
        return {name: getattr(self, name) for name in self._model_names}

    def get_checkpoint_submodel_names(self):
        """Names of sub-models whose weights go into training checkpoints."""
        if self.checkpoint_submodels is not None:
            return [name for name in self.checkpoint_submodels if name in self._model_names]
        return [
            name
            for name in self._model_names
            if not self._is_fully_frozen(getattr(self, name))
        ]

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save the checkpointed sub-models' weights with the container prefix."""
        if destination is None:
            destination = {}
        for name in self.get_checkpoint_submodel_names():
            sub_model = getattr(self, name, None)
            if sub_model is None:
                continue
            sub_prefix = f"{prefix}{name}."
            for key, value in sub_model.state_dict(prefix="", keep_vars=keep_vars).items():
                destination[f"{sub_prefix}{key}"] = value
        return destination

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Dispatch keys to sub-models by ``<name>.`` prefix (flat keys go to ``legacy_flat_key_target``); with ``strict=True`` missing/unexpected keys raise."""
        sub_sds = {name: {} for name in self._model_names}
        missing_keys = []
        unexpected_keys = []

        for key, value in state_dict.items():
            dispatched = False
            for name in self._model_names:
                sub_prefix = f"{name}."
                if key.startswith(sub_prefix):
                    sub_sds[name][key[len(sub_prefix):]] = value
                    dispatched = True
                    break
            if not dispatched:
                if self.legacy_flat_key_target is not None and self.legacy_flat_key_target in sub_sds:
                    sub_sds[self.legacy_flat_key_target][key] = value
                else:
                    unexpected_keys.append(key)

        for name, sub_sd in sub_sds.items():
            sub_model = getattr(self, name, None)
            if sub_model is None or not sub_sd:
                continue
            ret = sub_model.load_state_dict(sub_sd, strict=False, **kwargs)
            missing_keys.extend(f"{name}.{key}" for key in ret.missing_keys)
            unexpected_keys.extend(f"{name}.{key}" for key in ret.unexpected_keys)

        error_msgs = []
        if missing_keys:
            error_msgs.append(f"Missing key(s) in state_dict: {missing_keys}")
        if unexpected_keys:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected_keys}")
        if strict and error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    type(self).__name__, "\n\t".join(error_msgs)
                )
            )
        for msg in error_msgs:
            logger.info(msg)
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    # ------------------------------------------------------------------
    # FSDP2 integration helpers
    # ------------------------------------------------------------------

    def fully_shard(self, fsdp_plan):
        """Opt out of root-level FSDP wrapping (container holds no trainable parameters; sub-models are wrapped beforehand)."""
        return True

    def set_is_last_backward(self, is_last_step: bool):
        """Delegate to any wrapped sub-models."""
        for _name, module in self.named_modules():
            if hasattr(module, "set_is_last_backward") and module is not self:
                module.set_is_last_backward(is_last_step)

    def set_requires_all_reduce(self, is_last_step: bool):
        """Delegate to any wrapped sub-models."""
        for _name, module in self.named_modules():
            if hasattr(module, "set_requires_all_reduce") and module is not self:
                module.set_requires_all_reduce(is_last_step)
