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

from .build_prompt_internvl import InternvlPromptTemplate
from .build_prompt_qwen2vl import Qwen2vlPromptTemplate

eval_model_prompt_dict = {
    "internvl2_8b": InternvlPromptTemplate,
    "qwen2_vl_7b": Qwen2vlPromptTemplate
    }
