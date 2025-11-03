import transformers
from torch.distributed.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl
)
from megatron.training import print_rank_0
from mindspeed_mm.models.hf_src.base_model import FSDP2Mixin


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

    def post_meta_init(self):
        """
        Moves the model from meta device to NPU and reinitializes buffers
        (e.g., rotary position embeddings) after FSDP sharding.
        """
        # reset buffer
        self.visual.rotary_pos_emb.__init__(dim=self.config.vision_config.hidden_size // self.config.vision_config.num_heads // 2)
        self.language_model.rotary_emb.__init__(self.config.text_config)

        self.visual.rotary_pos_emb.to(device="cuda")
        self.language_model.rotary_emb.to(device="cuda")

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


class Qwen3VLForConditionalGeneration(transformers.Qwen3VLForConditionalGeneration, Qwen3VLFSDP2Mixin):
    pass


class Qwen3VLMoeForConditionalGeneration(transformers.Qwen3VLMoeForConditionalGeneration, Qwen3VLFSDP2Mixin):
    pass