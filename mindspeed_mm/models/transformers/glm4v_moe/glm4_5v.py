# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl
)
from megatron.training import print_rank_0, get_args
from mindspeed_mm.models.transformers.glm4v_moe.modeling_glm4v_moe import Glm4vFusedMoeForConditionalGeneration
from mindspeed_mm.models.transformers.base_model import FSDP2Mixin, WeightInitMixin
from mindspeed_mm.models.transformers.custom_model_registry import register_model


class Glm4VFSDP2Minxin(FSDP2Mixin):
    """
    Mixin class for FSDP2 of the  GLM4.5V
    """
    def _fully_shard(self, fsdp2_kwargs, fsdp2_config):
        # recompute
        for i, block in enumerate(self.model.visual.blocks):
            self.model.visual.blocks[i] = checkpoint_wrapper(block, CheckpointImpl.REENTRANT)

        for i, layer in enumerate(self.model.language_model.layers):
            self.model.language_model.layers[i] = checkpoint_wrapper(layer, CheckpointImpl.REENTRANT)

        args = get_args()
        if args.init_model_with_meta_device:
            for module in self.model.modules():
                if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    module.weight.data[module.padding_idx].zero_()

        # fully_shard
        for block in self.model.visual.blocks:
            fully_shard(block, **fsdp2_kwargs)
        fully_shard(self.model.visual.merger, **fsdp2_kwargs)
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
            forbidden_modules.update(vision_model_keys)

        if config.image_encoder.vision_projector.freeze:
            projector_keys = ["visual.merger"]
            forbidden_modules.update(projector_keys)

        if config.text_decoder.freeze:
            language_model_keys = ["language_model", "lm_head"]
            forbidden_modules.update(language_model_keys)

        # modules that are finally frozen
        final_forbidden_modules = set()
        for name, param in self.named_parameters():
            for forbidden_module in forbidden_modules:
                if forbidden_module in name:
                    param.requires_grad_(False)
                    final_forbidden_modules.add(forbidden_module)
        print_rank_0(f"Set modules not trainable: {final_forbidden_modules}")


@register_model("glm4v_moe")
class Glm4vMoeForConditionalGeneration(WeightInitMixin, Glm4vFusedMoeForConditionalGeneration, Glm4VFSDP2Minxin):
    pass
