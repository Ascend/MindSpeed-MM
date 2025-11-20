import transformers
from torch.distributed.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl
)

from megatron.training import print_rank_0
from mindspeed_mm.models.transformers.base_model import FSDP2Mixin


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


class Qwen3OmniMoeThinkerForConditionalGeneration(transformers.Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniFSDP2Mixin):
    @classmethod
    def from_pretrained(cls, hf_path, **kwargs):
        # get thinker_config
        config = kwargs.get("config")
        if config is not None:
            kwargs["config"] = config.thinker_config

        load_kwargs = {
            "trust_remote_code": False,
            "_from_auto": True,
            "device_map": None,
            "dtype": None,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": False
        }
        load_kwargs.update(kwargs)

        return super().from_pretrained(hf_path, **load_kwargs)