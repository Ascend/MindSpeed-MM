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

__all__ = ["sora_pipeline_dict", "vlm_pipeline_dict"]

from mindspeed_mm.tasks.inference.pipeline.qwen2vl_pipeline import Qwen2VlPipeline
from mindspeed_mm.tasks.inference.pipeline.stepvideo_pipeline import StepVideoPipeline
from mindspeed_mm.tasks.inference.pipeline.wan_pipeline import WanPipeline
from mindspeed_mm.tasks.inference.pipeline.hunyuanvideo_15_pipeline import HunyuanVideo15Pipeline

sora_pipeline_dict = {
                      "WanPipeline": WanPipeline,
                      "StepVideoPipeline": StepVideoPipeline,
                      "HunyuanVideo15Pipeline": HunyuanVideo15Pipeline,
                      }

vlm_pipeline_dict = {
    "Qwen2VlPipeline": Qwen2VlPipeline,
}
