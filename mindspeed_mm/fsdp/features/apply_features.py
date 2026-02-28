import torch

from mindspeed.fsdp.utils.str_match import module_name_match
from ..params.model_args import ModelArguments
from ..features.memory.async_offload import async_offload_modules, get_offload_modules
from ..features.memory.chunkloss_lm_head import apply_chunkloss_module, get_chunkloss_module


class FeaturesApplier:
    def __init__(self, model_config: ModelArguments):
        self.config = model_config

    def get_needed_modules(self, modules, plan):
        matched_submodules = []
        for plan_name in plan:
            for name, module in modules.named_modules():
                if module_name_match(plan_name, name):
                    if (name, module) not in matched_submodules:
                        matched_submodules.append((name, module))
        return matched_submodules

    def apply_activation_offload_modules(self, model):
        if getattr(self.config, "activation_offload_plan", None) is None or getattr(self.config.activation_offload_plan, "apply_modules", None) is None:
            return

        activation_offload_modules = get_offload_modules(model, getattr(self.config.activation_offload_plan, "apply_modules"))
        async_offload_modules(activation_offload_modules)

    def apply_chunkloss(self, model):
        if not self.config.enable_chunk_loss:
            return

        setattr(model, "enable_chunk_loss", True)
        chunkloss_module = get_chunkloss_module(model, self.config.chunkloss_plan)
        apply_chunkloss_module(chunkloss_module)

    def __call__(self, model):
        self.apply_activation_offload_modules(model=model)
        self.apply_chunkloss(model=model)