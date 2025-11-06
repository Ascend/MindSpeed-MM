from packaging import version

import mindspore
import transformers
from mindspeed.patch_utils import MindSpeedPatchesManager as aspm
from mindspeed.mindspore.ops.npu_rotary_position_embedding import npu_rotary_position_embedding

from mindspeed_mm.mindspore.data.datasets.utils import process_in_cpu_wrapper
from mindspeed_mm.mindspore.data.data_utils.func_utils.convert import preprocess_dataset
from mindspeed_mm.mindspore.models.vision.vision_encoders.qwen2vl_vit_model import get_window_index, qwen2vlvit_selfattention_forward
from mindspeed_mm.mindspore.utils.transformer_model_config import get_model_config
from mindspeed_mm.mindspore.models.predictor.dits.sparseu_mmdit import block_forward, sparsemmditblock_forward


def masked_scatter_(self, mask, updates):
    origin_dtype = None
    if self.dtype in (mindspore.float16, mindspore.bfloat16):
        origin_dtype = self.dtype
        self = self.to(mindspore.float32)
    if updates.dtype in (mindspore.float16, mindspore.bfloat16):
        updates = updates.to(mindspore.float32)
    self = mindspore.ops.MaskedScatter()(self, mask, updates)
    if origin_dtype is not None:
        self = self.to(origin_dtype)
    return self


def apply_mindspore_patch():
    aspm.register_patch('mindspeed_mm.data.datasets.qwen2vl_dataset.get_qwen2vl_dataset', process_in_cpu_wrapper) # process dataset on cpu
    aspm.register_patch('torch.Tensor.masked_scatter', masked_scatter_)
    aspm.register_patch('mindspeed_mm.data.data_utils.func_utils.convert.SupervisedDatasetProcessor.preprocess_dataset', preprocess_dataset)
    aspm.register_patch('mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model.Qwen2VLViT.get_window_index', get_window_index)
    aspm.register_patch('mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model.Qwen2vlVitSelfAttention.forward', qwen2vlvit_selfattention_forward)
    aspm.register_patch('mindspeed_mm.utils.transformer_model_config.get_model_config', get_model_config)

    # patch llava
    aspm.register_patch(
        'mindspeed.ops.npu_rotary_position_embedding.npu_rotary_position_embedding',
        npu_rotary_position_embedding
    )

    # patch glm
    if version.parse(transformers.__version__) >= version.parse('4.54.0.dev0'):
        from mindspeed_mm.mindspore.third_party.transformers.masking_utils import sdpa_mask_older_torch
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa'] = sdpa_mask_older_torch

    aspm.apply_patches()
    #patch opensoraplan1.5t2v
    aspm.register_patch('mindspeed_mm.models.predictor.dits.sparseu_mmdit.SparseUMMDiT.block_forward', block_forward)
    aspm.register_patch('mindspeed_mm.models.predictor.dits.sparseu_mmdit.SparseMMDiTBlock.forward',
                        sparsemmditblock_forward)

apply_mindspore_patch()
