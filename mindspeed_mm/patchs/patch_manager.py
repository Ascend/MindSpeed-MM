# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

from megatron.training import get_args
from mindspeed.patch_utils import MindSpeedPatchesManager as pm

from mindspeed_mm.patchs import (
    adaptive_clip_grad_patch,
    infer_fa_patch,
    models_patches,
    fsdp1_patches,
    hetero_patches,
    training_patches,
    fsdp2_patches
)


class PatchesManager:
    configs = {
        "ae_float32": [
            ("megatron.core.transformer.module.Float16Module.__init__", models_patches.float16Module_init),
            ("megatron.core.transformer.module.Float16Module.forward", models_patches.float16Module_forward)
        ],
        "adaptive_clip_grad_norm": [
            ("megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__", adaptive_clip_grad_patch.adaptive_clip_grad_norm_optimizer_init_wrapper),
            ("megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.clip_grad_norm", adaptive_clip_grad_patch.adaptive_clip_grad_norm_wrapper)
        ],
        "infer_fa": [("megatron.core.transformer.dot_product_attention.DotProductAttention.forward", infer_fa_patch.dot_product_attention_forward_infer_wrapper)],
        "use_fsdp1": [
            ("megatron.training.training.get_model", fsdp1_patches.fsdp1_get_model)
        ],
        "clip_grad_async": [
            ("megatron.core.optimizer.clip_grads.get_grad_norm_fp32", adaptive_clip_grad_patch.get_grad_norm_fp32_async),
            ("megatron.core.optimizer.clip_grads.clip_grad_by_total_norm_fp32", adaptive_clip_grad_patch.clip_grad_by_total_norm_fp32_async)
        ],
        
        # Enable this patch when loading model weights from a .pt checkpoint file in distributed training.
        # This will override the default model loading behavior to handle distributed checkpoint format.
        "get_dist_model_load_from_pt": [
            ("megatron.training.training.get_model", training_patches.get_dist_model_load_from_pt)
        ],
        "hetero_parallel_cp": [
            ("megatron.core.transformer.attention.Attention.__init__", \
                hetero_patches.hetero_attention_init_wrapper),
            ("mindspeed_mm.models.vlm_model.get_vit_layer_spec", \
                hetero_patches.hetero_spec_wrapper),
            ("mindspeed_mm.models.vlm_model.get_llm_layer_spec", \
                hetero_patches.hetero_spec_wrapper),
            ("mindspeed_mm.models.vlm_model.get_audio_layer_spec", \
                hetero_patches.hetero_spec_wrapper),
        ],
        "scale_grad": [
            ("megatron.core.distributed.TorchFullyShardedDataParallel.scale_gradients", fsdp2_patches.scale_gradients)
        ]
    }

    @staticmethod
    def register_patch(orig_func_name, new_func=None):
        pm.register_patch(orig_func_name, new_func, force_patch=True)

    @staticmethod
    def apply_patches():
        pm.apply_patches()

    @staticmethod
    def apply_patches_from_config():
        cfg = get_args().mm.model
        if hasattr(cfg, "patch"):
            cfg = cfg.patch.to_dict()
            for key in cfg.keys() & PatchesManager.configs.keys():
                if not cfg.get(key):
                    continue
                for orig_func_name, new_func in PatchesManager.configs[key]:
                    PatchesManager.register_patch(orig_func_name, new_func)
        PatchesManager.apply_patches()

    
