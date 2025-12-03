import transformers
# support newer transformers for qwen3omni
try:
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe
    has_qwen3omni_support = True
except ImportError:
    Qwen3OmniMoeThinkerConfig = None
    has_qwen3omni_support = False
from transformers.activations import ACT2FN
import torch
from torch import nn
import torch_npu
from torch.distributed.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl
)

from megatron.training import print_rank_0
from mindspeed_mm.models.transformers.base_model import FSDP2Mixin
from mindspeed_mm.models.common.gmm import npu_group_gemm


class Qwen3OmniFSDP2Mixin(FSDP2Mixin):
    """
    Mixin class for FSDP2 of the Qwen3Omni-series
    """

    def _fully_shard(self, fsdp2_kwargs, fsdp2_config):
        # recompute
        for i, layer in enumerate(self.audio_tower.layers):
            self.audio_tower.layers[i] = checkpoint_wrapper(layer, CheckpointImpl.REENTRANT)

        for i, layer in enumerate(self.model.layers):
            self.model.layers[i] = checkpoint_wrapper(layer, CheckpointImpl.REENTRANT)

        # fully_shard
        fully_shard(self.audio_tower.positional_embedding, **fsdp2_kwargs)
        for layer in self.audio_tower.layers:
            fully_shard(layer, **fsdp2_kwargs)

        fully_shard(self.visual.merger, **fsdp2_kwargs)
        for merger in self.visual.merger_list:
            fully_shard(merger, **fsdp2_kwargs)
        fully_shard(self.visual, **fsdp2_kwargs)

        fully_shard(self.model.embed_tokens, **fsdp2_kwargs)
        for layer in self.model.layers:
            fully_shard(layer, **fsdp2_kwargs)
        fully_shard(self.lm_head, **fsdp2_kwargs)
        fully_shard(self, **fsdp2_kwargs)

    def freeze(self, config):
        forbidden_modules = set()
        if config.image_encoder.vision_encoder.freeze:
            vision_model_keys = ['visual.patch_embed', 'visual.blocks']
            print_rank_0(f"Set vision model not trainable: {vision_model_keys}")
            forbidden_modules.update(vision_model_keys)

        if config.image_encoder.vision_projector.freeze:
            projector_keys = ["visual.merger"]
            print_rank_0(f"Set multi model projector not trainable: {projector_keys}")
            forbidden_modules.update(projector_keys)

        if config.audio_encoder.audio_encoder.freeze:
            projector_keys = ["audio_tower"]
            print_rank_0(f"Set audio model not trainable: {projector_keys}")
            forbidden_modules.update(projector_keys)

        if config.text_decoder.freeze:
            language_model_keys = ["model", "lm_head"]
            print_rank_0(f"Set language model not trainable: {language_model_keys}")
            forbidden_modules.update(language_model_keys)

        for name, param in self.named_parameters():
            if any(forbidden_module in name for forbidden_module in forbidden_modules):
                param.requires_grad_(False)


class Qwen3OmniMoeThinkerTextExperts(nn.ModuleList):
    """
    ModuleList of experts.
    """

    def __init__(self, config: Qwen3OmniMoeThinkerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size \
            if config.moe_intermediate_size is None else config.moe_intermediate_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))
        self.act_fn = ACT2FN[config.hidden_act]


    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size * sequence_length, hidden_dim)
            selected_experts: (batch_size * sequence_length, top_k)
            routing_weights: (batch_size * sequence_length, top_k)
        Returns:
            (batch_size * sequence_length, hidden_dim)
        """
        # Fixes the memory reorganization problem triggered when fast host dispatch
        # and tensor multi-stream reuse occur simultaneously.
        torch.npu.synchronize()

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, top_k_index.to(torch.int32))
        tokens_per_expert = torch.histc(top_k_index, bins=self.num_experts, min=0, max=self.num_experts)
        intermediate_hidden_states = npu_group_gemm(permuted_hidden_states, self.gate_up_proj, tokens_per_expert)
        intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
        output = npu_group_gemm(intermediate_activations, self.down_proj, tokens_per_expert)
        next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=top_k_weights)
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states


class Qwen3OmniMoeThinkerForConditionalGeneration(transformers.Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniFSDP2Mixin):
    def __init__(self, config):
        self._apply_moe_block_patch()
        super().__init__(config)
    
    def _apply_moe_block_patch(self):
        if has_qwen3omni_support:
            modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextExperts = Qwen3OmniMoeThinkerTextExperts
    
    @classmethod
    def from_pretrained(cls, hf_path, **kwargs):
        load_kwargs = {
            "_from_auto": True,
            "device_map": None,
            "dtype": None,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": False
        }

        # get thinker_config
        config = kwargs.get("config")
        if config is not None:
            load_kwargs["config"] = config.thinker_config

        kwargs.update(load_kwargs)
        return super().from_pretrained(hf_path, **kwargs)
    
    @classmethod
    def _from_config(cls, config, **kwargs):
        # thinker_config
        if config.thinker_config is not None:
            config = config.thinker_config
        
        load_kwargs = {
            "attn_implementation": "eager",
        }
        kwargs.update(load_kwargs)
        return super()._from_config(config, **kwargs)