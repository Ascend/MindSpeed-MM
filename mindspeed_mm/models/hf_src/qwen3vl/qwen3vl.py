# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.

import transformers
from torch.distributed.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl
)
from megatron.training import print_rank_0
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

        # fully_shard
        for block in self.model.visual.blocks:
            fully_shard(block, **fsdp2_kwargs)
        fully_shard(self.model.visual.merger, **fsdp2_kwargs)
        for merger in self.model.visual.deepstack_merger_list:
            fully_shard(merger, **fsdp2_kwargs)
        fully_shard(self.model.visual, **fsdp2_kwargs)

        fully_shard(self.model.language_model.embed_tokens, **fsdp2_kwargs)
        for layer in self.model.language_model.layers:
            fully_shard(layer, **fsdp2_kwargs)
        fully_shard(self.lm_head, **fsdp2_kwargs)
        fully_shard(self, **fsdp2_kwargs)

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


class Qwen3VLForConditionalGeneration(HFQwen3VLForConditionalGeneration, Qwen3VLFSDP2Mixin):
    pass


class Qwen3VLMoeForConditionalGeneration(HFQwen3VLMoeForConditionalGeneration, Qwen3VLFSDP2Mixin):
    pass