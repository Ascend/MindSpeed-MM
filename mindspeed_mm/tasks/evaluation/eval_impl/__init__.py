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

from mindspeed_mm.tasks.evaluation.eval_impl.impl_mmmu import MMMUEvalImpl
from mindspeed_mm.tasks.evaluation.eval_impl.impl_vqa import VQAEvalImpl
from mindspeed_mm.tasks.evaluation.eval_impl.impl_ai2d import AI2DEvalImpl

from mindspeed_mm.tasks.inference.pipeline import vlm_pipeline_dict, sora_pipeline_dict

eval_impl_dict = {"mmmu_dev_val": MMMUEvalImpl, "ai2d_test": AI2DEvalImpl, "chartqa_test": VQAEvalImpl,
                  "docvqa_val": VQAEvalImpl}

eval_pipeline_dict = {"internvl2_8b": vlm_pipeline_dict["InternVLPipeline"],
                      "qwen2_vl_7b": vlm_pipeline_dict["Qwen2VlPipeline"]
                      }
