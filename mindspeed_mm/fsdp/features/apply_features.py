import torch
from packaging import version
import transformers 
if version.parse(transformers.__version__) >= version.parse("5.2.0"):
    from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY
    
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
        if (
            getattr(self.config, "activation_offload_plan", None) is None
            or not getattr(self.config, "enable_activation_offload", False)
            or getattr(self.config.activation_offload_plan, "apply_modules", None) is None
        ):
            return

        activation_offload_modules = get_offload_modules(model, getattr(self.config.activation_offload_plan, "apply_modules"))
        async_offload_modules(activation_offload_modules)

    def apply_chunkloss(self, model):
        if self.config.enable_chunk_loss:
            setattr(model, "enable_chunk_loss", True)
            setattr(model, "chunk_size", self.config.chunkloss_plan.chunk_size)
        elif self.config.enable_dynamic_chunk_loss:
            setattr(model, "enable_dynamic_chunk_loss", True)
        else:
            return
        chunkloss_module = get_chunkloss_module(model, self.config.chunkloss_plan)
        apply_chunkloss_module(chunkloss_module)

    def apply_aux_loss_capture(self, model):
        # This function is designed to automatically capture router logits from each MoE layer
        # when 'loss_cfg.router_aux_loss_coef' is configured and greater than 0.
        # These captured logits are essential for calculating the auxiliary loss.
        if (
            getattr(self.config, "loss_cfg", None) is None
            or getattr(self.config.loss_cfg, "router_aux_loss_coef", 0.0) <= 0.0
        ):
            return 
        
        # This logic applies to transformers version 5.2.0 and later.
        # Please use with caution for earlier versions.
        if version.parse(transformers.__version__) >= version.parse("5.2.0"):
            for sub_module in model.modules():
                if hasattr(sub_module, "_can_record_outputs") and len(sub_module._can_record_outputs) > 0:
                    # After applying FSDP sharding via fully_shard, the module paths change
                    # (e.g., 'model.layers.0' becomes 'model.layers.fsdp.0'), causing a mismatch with the
                    # registry keys which are based on the original model structure from
                    # from_pretrained. We need to update the _CAN_RECORD_REGISTRY with the
                    # new class keys from the sharded sub-modules.
                    _CAN_RECORD_REGISTRY[str(sub_module.__class__)] = sub_module._can_record_outputs

    def __call__(self, model):
        self.apply_activation_offload_modules(model=model)
        self.apply_chunkloss(model=model)
        self.apply_aux_loss_capture(model=model)
