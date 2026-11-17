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

from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_qwen3_asr import Qwen3ASRConfig
from .modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from .processing_qwen3_asr import Qwen3ASRProcessor


logger = logging.getLogger(__name__)


try:
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
except ValueError:
    logger.debug("Qwen3-ASR AutoConfig is already registered.")

try:
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
except ValueError:
    logger.debug("Qwen3-ASR AutoModel is already registered.")

try:
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
except ValueError:
    logger.debug("Qwen3-ASR AutoProcessor is already registered.")


__all__ = [
    "Qwen3ASRConfig",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRProcessor",
]
