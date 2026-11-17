# Copyright 2026 Huawei Technologies Co., Ltd.
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

from transformers.models.qwen2 import modeling_qwen2


def apply_cosyvoice_npu_patch():
    from mindspeed_mm.fsdp.ops.npu_patch import npu_fused_operator
    # Patches for CosyVoice3 Model
    modeling_qwen2.Qwen2RMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
    modeling_qwen2.apply_rotary_pos_emb = npu_fused_operator.apply_transformers_rope_half_npu
