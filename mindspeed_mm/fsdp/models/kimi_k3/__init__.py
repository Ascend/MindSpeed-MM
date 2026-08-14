"""MindSpeed-MM FSDP adapter for the Kimi-K3 model.

This ``__init__.py`` makes the ``Kimi-k3`` HuggingFace model directory directly
importable as a MindSpeed-MM plugin. It registers the custom config/model
classes and adds the small amount of training glue needed by the FSDP2 trainer.
"""

from transformers import AutoConfig

import torch
import torch.nn as nn

from mindspeed_mm.fsdp.distributed.context_parallel.communication import (
    gather_forward_split_backward,
)
from mindspeed_mm.fsdp.loss.loss_func import build_loss_func
from mindspeed_mm.fsdp.utils.register import model_register
from mindspeed_mm.fsdp.models.base_model import WeightInitMixin
from .configuration_kimi_k3 import KimiK3Config, KimiLinearConfig
from .modeling_kimi_k3 import KimiK3ForConditionalGeneration as _KimiK3ForConditionalGeneration

for _model_type, _config_cls in (
    ("kimi_linear", KimiLinearConfig),
    ("kimi_k3", KimiK3Config),
):
    try:
        AutoConfig.register(_model_type, _config_cls)
    except ValueError:
        # Already registered (e.g. by a previous plugin import or remote-code load)
        pass


class KimiK3ForConditionalGeneration(WeightInitMixin, _KimiK3ForConditionalGeneration):
    """MindSpeed-MM FSDP adapter for the Kimi-K3 model.

    This is a thin subclass of the upstream ``KimiK3ForConditionalGeneration``.
    It keeps the original model weights and forward logic intact while adding
    the hooks required by the MindSpeed-MM FSDP2 trainer.
    """

    @staticmethod
    def overwrite_transformer_config(transformer_config, model_args, feature_args):
        """Adjust the loaded config for training.

        Disables KV cache and propagates the selected attention implementation
        and MoE execution switches to ``text_config``.
        """
        transformer_config.use_cache = False
        if getattr(transformer_config, "text_config", None) is not None:
            transformer_config.text_config.use_cache = False

        attn_implementation = getattr(model_args, "attn_implementation", None)
        if attn_implementation is not None:
            transformer_config._attn_implementation = attn_implementation
            if getattr(transformer_config, "vision_config", None) is not None:
                transformer_config.vision_config._attn_implementation = attn_implementation

        # Propagate MoE execution switches so the patched sparse MoE block can read them.
        use_grouped_expert_matmul = getattr(model_args, "use_grouped_expert_matmul", False)
        transformer_config.text_config.use_grouped_expert_matmul = use_grouped_expert_matmul

        # ep balance
        transformer_config.text_config.enable_ep_balance = getattr(feature_args, "enable_ep_balance", False)
        transformer_config.text_config.max_dup_experts_num = getattr(feature_args.ep_balance_plan, "max_dup_experts_num", 2)

        # Selective-recompute switches: keep the expensive attention kernels (FA / KDA)
        # out of gradient-checkpoint replay and offload their small intermediates to host.
        # Requires features.recompute and features.enable_activation_offload to be on,
        # with activation_offload_plan covering the same modules as recompute_plan.
        skip_flash_attn_recompute = getattr(model_args, "skip_flash_attn_recompute", False)
        skip_kda_recompute = getattr(model_args, "skip_kda_recompute", False)
        transformer_config.text_config.skip_flash_attn_recompute = skip_flash_attn_recompute
        transformer_config.text_config.skip_kda_recompute = skip_kda_recompute
        if getattr(transformer_config, "vision_config", None) is not None:
            transformer_config.vision_config.skip_flash_attn_recompute = skip_flash_attn_recompute

        # KDA kernel selection: 'fused' (chunk_kda from triton_ascend_kernels) or
        # 'naive' (in-repo small-op implementation chunk_kda_naive).
        kda_implementation = getattr(model_args, "kda_implementation", "fused")
        if kda_implementation not in ("fused", "naive"):
            raise ValueError(
                f"Unsupported kda_implementation: {kda_implementation}. "
                "Expected 'fused' or 'naive'."
            )
        transformer_config.text_config.kda_implementation = kda_implementation

        # Causal conv1d kernel selection for KDA short convolutions: 'triton' (default)
        # or 'ascendc' (AscendC fused op from fla_npu, same as qwen3_5, NPU only).
        causal_conv1d_implementation = getattr(model_args, "causal_conv1d_implementation", "triton")
        if causal_conv1d_implementation not in ("triton", "ascendc"):
            raise ValueError(
                f"Unsupported causal_conv1d_implementation: {causal_conv1d_implementation}. "
                "Expected 'triton' or 'ascendc'."
            )
        transformer_config.text_config.causal_conv1d_implementation = causal_conv1d_implementation

        return transformer_config

    def tie_weights(self, *args, **kwargs):
        """Forward tie_weights to the language model, accepting extra kwargs.

        Transformers' ``init_weights`` may call ``tie_weights(recompute_mapping=False)``,
        but the upstream Kimi-K3 wrapper does not accept that argument. Accept
        ``*args, **kwargs`` here so the call signature stays compatible.
        """
        return self.language_model.tie_weights(*args, **kwargs)

    def set_modules_to_prefetch(self, fsdp_plan, ep_plan):
        """Configure FSDP forward/backward prefetch chains along execution order.

        Mirrors the kimi_k2.5 prefetch hook, with routed experts living under
        ``block_sparse_moe.experts`` instead of ``mlp.experts``. Dense layers
        (no ``block_sparse_moe``) only prefetch the next decoder layer.
        """
        layers = self.language_model.model.layers
        vt_blocks = self.vision_tower.encoder.blocks

        def _experts_of(layer):
            moe = getattr(layer, "block_sparse_moe", None)
            return [moe.experts] if moe is not None else []

        if fsdp_plan.num_to_forward_prefetch > 0:
            self.language_model.model.embed_tokens.set_modules_to_forward_prefetch([self.vision_tower])
            self.vision_tower.set_modules_to_forward_prefetch([vt_blocks[0]])
            for idx in range(len(vt_blocks) - 1):
                vt_blocks[idx].set_modules_to_forward_prefetch([vt_blocks[idx + 1]])
            vt_blocks[-1].set_modules_to_forward_prefetch([self.mm_projector])
            self.mm_projector.set_modules_to_forward_prefetch([layers[0]])
            for idx in range(len(layers) - 1):
                layers[idx].set_modules_to_forward_prefetch(_experts_of(layers[idx + 1]) + [layers[idx + 1]])
            layers[-1].set_modules_to_forward_prefetch([self.language_model.model, self.language_model.lm_head])

        if fsdp_plan.num_to_backward_prefetch > 0:
            self.language_model.lm_head.set_modules_to_backward_prefetch(
                [self.language_model.model] + _experts_of(layers[-1]) + [layers[-1]])
            for idx in range(len(layers) - 1, 0, -1):
                layers[idx].set_modules_to_backward_prefetch(_experts_of(layers[idx - 1]) + [layers[idx - 1]])
            layers[0].set_modules_to_backward_prefetch([self.mm_projector])
            self.mm_projector.set_modules_to_backward_prefetch([vt_blocks[-1]])
            for idx in range(len(vt_blocks) - 1, 0, -1):
                vt_blocks[idx].set_modules_to_backward_prefetch([vt_blocks[idx - 1]])
            vt_blocks[0].set_modules_to_backward_prefetch([self.vision_tower])
            self.vision_tower.set_modules_to_backward_prefetch([self.language_model.model.embed_tokens])
        return True


try:
    KimiK3ForConditionalGeneration = model_register.register("kimi_k3")(
        KimiK3ForConditionalGeneration
    )
except KeyError:
    # Already registered by another import path (e.g. deprecated adapter).
    raise ValueError("error register!")


# Apply the optional MoE patch that replaces KimiSparseMoeBlock with a 3-D tensor
# expert implementation supporting EP and NPU fused GMM/permute/unpermute.
from .kimi_moe_patch import apply_kimi_k3_moe_patch

apply_kimi_k3_moe_patch()


__all__ = ["KimiK3ForConditionalGeneration"]
