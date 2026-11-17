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

import logging

import torch

from mindspeed_mm.fsdp.log import print_rank
from .flash_attn.flash_attn import apply_transformers_attention_patch
from .fully_shard.fully_shard import apply_fully_shard_patch


logger = logging.getLogger(__name__)


def apply_ops_patch():
    apply_transformers_attention_patch()
    print_rank(logger.info, "✅ MindSpeed-MM ops patch applied.")

    # apply modify fully_shard patch
    apply_fully_shard_patch()
    print_rank(logger.info, "✅ MindSpeed-MM fully_shard patch applied.")
