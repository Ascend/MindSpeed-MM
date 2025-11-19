# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.

import transformers
from torch.distributed.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl
)
from megatron.training import print_rank_0, get_args
from mindspeed_mm.models.hf_src.base_model import FSDP2Mixin

from mindspeed_mm.models.hf_src.qwen3vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration
from mindspeed_mm.models.hf_src.qwen3vl.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration as HFQwen3VLMoeForConditionalGeneration


class Qwen3VLFSDP2Mixin(FSDP2Mixin):
    """
    Mixin class for FSDP2 of the Qwen3VL-series
    """
    def _fully_shard(self, fsdp2_kwargs, fsdp2_config):
        # recompute
        for i, block in enumerate(self.model.visual.blocks):
            self.model.visual.blocks[i] = checkpoint_wrapper(block, CheckpointImpl.REENTRANT)

        for i, layer in enumerate(self.model.language_model.layers):
            self.model.language_model.layers[i] = checkpoint_wrapper(layer, CheckpointImpl.REENTRANT)
            
        last_module_kwargs = fsdp2_kwargs.copy()
        last_module_kwargs["reshard_after_forward"] = False

        # fully_shard
        for block in self.model.visual.blocks:
            fully_shard(block, **fsdp2_kwargs)
        fully_shard(self.model.visual.merger, **fsdp2_kwargs)
        for merger in self.model.visual.deepstack_merger_list:
            fully_shard(merger, **fsdp2_kwargs)
        fully_shard(self.model.visual, **fsdp2_kwargs)            

        llm_num_layers = len(self.model.language_model.layers)
        fully_shard(self.model.language_model.embed_tokens, **fsdp2_kwargs)
        for idx, layer in enumerate(self.model.language_model.layers):
            if idx == (llm_num_layers - 1) and fsdp2_config.num_to_forward_prefetch > 0:
                # Skip resharding after forward for the last layer if prefetching is enabled
                fully_shard(layer, **last_module_kwargs)
            else:
                fully_shard(layer, **fsdp2_kwargs)
        fully_shard(self.lm_head, **last_module_kwargs)
        fully_shard(self, **fsdp2_kwargs)
        
        # prefetch
        if fsdp2_config.num_to_forward_prefetch > 0:
            for i, (curr_block, next_block) in enumerate(zip(self.model.visual.blocks[:-1], self.model.visual.blocks[1:])):
                prefetch_modules = []
                if i in self.model.visual.deepstack_visual_indexes:
                    prefetch_modules.append(self.model.visual.deepstack_merger_list[self.model.visual.deepstack_visual_indexes.index(i)])
                prefetch_modules.append(next_block)
                curr_block.set_modules_to_forward_prefetch(prefetch_modules)
                
            self.model.visual.blocks[-1].set_modules_to_forward_prefetch([self.model.visual.merger])
            self.model.visual.merger.set_modules_to_forward_prefetch([self.model.language_model.embed_tokens])
            self.model.language_model.embed_tokens.set_modules_to_forward_prefetch([self.model.language_model.layers[0]])
            
            for curr_layer, next_layer in zip(self.model.language_model.layers[:-1], self.model.language_model.layers[1:]):
                curr_layer.set_modules_to_forward_prefetch([next_layer])
            self.model.language_model.layers[-1].set_modules_to_forward_prefetch([self.lm_head])

    def freeze(self, config):
        forbidden_modules = set()
        if config.image_encoder.vision_encoder.freeze:
            vision_model_keys = ["visual.patch_embed", "visual.blocks"]
            print_rank_0(f"Set vision model not trainable: {vision_model_keys}")
            forbidden_modules.update(vision_model_keys)

        if config.image_encoder.vision_projector.freeze:
            projector_keys = ["visual.merger"]
            print_rank_0(f"Set vision model not trainable: {projector_keys}")
            forbidden_modules.update(projector_keys)

        if config.text_decoder.freeze:
            language_model_keys = ["language_model", "lm_head"]
            print_rank_0(f"Set vision model not trainable: {language_model_keys}")
            forbidden_modules.update(language_model_keys)

        for name, param in self.model.named_parameters():
            if any(forbidden_module in name for forbidden_module in forbidden_modules):
                param.requires_grad_(False)

    @staticmethod
    def overwrite_transformer_config(transformer_config):
        args = get_args()
        model_cfg = args.mm.model

        # attn_implementation support eager, sdpa(layout BNSD), flash_attention_2(layout BNSD), var_len_fa(layout TND), default flash_attention_2
        vit_attn_implementation = getattr(model_cfg.image_encoder, "attn_implementation", "flash_attention_2")
        llm_attn_implementation = getattr(model_cfg.text_decoder, "attn_implementation", "flash_attention_2")
        # set attn type configuration
        transformer_config.vision_config._attn_implementation = vit_attn_implementation
        transformer_config.text_config._attn_implementation = llm_attn_implementation

        return transformer_config


class Qwen3VLForConditionalGeneration(HFQwen3VLForConditionalGeneration, Qwen3VLFSDP2Mixin):
    pass


class Qwen3VLMoeForConditionalGeneration(HFQwen3VLMoeForConditionalGeneration, Qwen3VLFSDP2Mixin):
    pass