# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
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

import torch

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE


def build_mla_qk_clip_metadata(module):
    """Build runtime MLA QK-Clip bindings."""
    attn_implementation = getattr(module.config, "_attn_implementation", "eager")
    if attn_implementation not in {"eager", "flash_attention_2", "flash_attention_3"}:
        raise ValueError(
            "Kimi MLA QK-Clip requires eager attention or the MindSpeed NPU "
            "flash-attention path (Ulysses and ring context parallel included)"
        )
    if attn_implementation != "eager" and not IS_NPU_AVAILABLE:
        raise ValueError("Kimi MLA QK-Clip with flash attention requires the NPU fused kernel")
    query_name = "q_b_proj" if module.q_lora_rank is not None else "q_proj"
    query = getattr(getattr(module, query_name, None), "weight", None)
    key = getattr(getattr(module, "kv_b_proj", None), "weight", None)
    provider = getattr(module, "consume_qk_clip_max_logits", None)
    key_rows = getattr(module, "qk_nope_head_dim", None)
    query_rotary_rows = getattr(module, "qk_rope_head_dim", None)
    if not isinstance(query, torch.nn.Parameter) or not isinstance(key, torch.nn.Parameter):
        raise ValueError("Kimi MLA requires Q and KV projection weights for QK-Clip")
    if not callable(provider):
        raise ValueError("Kimi MLA does not publish QK-Clip metrics")
    if type(key_rows) is not int or key_rows <= 0:
        raise ValueError("Kimi MLA requires a positive qk_nope_head_dim for QK-Clip")
    if type(query_rotary_rows) is not int or query_rotary_rows <= 0:
        raise ValueError("Kimi MLA requires a positive qk_rope_head_dim for QK-Clip")
    module._qk_clip_enabled = True
    process_groups = ()
    if torch.distributed.is_initialized():
        process_groups = tuple(get_parallel_state().get_hsdp_device_mesh().get_all_groups())
    return {
        "qk_clip_params": (query, key),
        "qk_clip_metric_provider": provider,
        "qk_clip_process_groups": process_groups,
        "qk_clip_key_rows": key_rows,
        "qk_clip_query_content_rows": key_rows,
        "qk_clip_query_rotary_rows": query_rotary_rows,
    }
