
# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.

from megatron.training import print_rank_0
from transformers import Mistral3ForConditionalGeneration

from mindspeed_mm.models.transformers.base_model import FSDP2Mixin, WeightInitMixin


class MultiModelMistral3ForConditionalGeneration(Mistral3ForConditionalGeneration, FSDP2Mixin, WeightInitMixin):

    def freeze(self, config):
        forbidden_modules = set()
        if config.vision_encoder.freeze:
            vision_model_keys = ["vision_tower"]
            print_rank_0(f"Set vision model not trainable: {vision_model_keys}")
            forbidden_modules.update(vision_model_keys)

        if config.vision_projector.freeze:
            projector_keys = ["multi_modal_projector"]
            print_rank_0(f"Set vision model not trainable: {projector_keys}")
            forbidden_modules.update(projector_keys)

        if config.text_decoder.freeze:
            language_model_keys = ["language_model", "lm_head"]
            print_rank_0(f"Set vision model not trainable: {language_model_keys}")
            forbidden_modules.update(language_model_keys)

        for name, param in self.model.named_parameters():
            if any(forbidden_module in name for forbidden_module in forbidden_modules):
                param.requires_grad_(False)
