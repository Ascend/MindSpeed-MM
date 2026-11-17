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

from dataclasses import field
from typing import Optional

from mindspeed_mm.fsdp.config.arguments.base_args import BaseArguments
from mindspeed_mm.fsdp.params.inference_args import InferenceArguments


class EvaluationRuntimeArguments(BaseArguments):
    dataset_type: str = field(
        default="vqa2_val",
        metadata={"help": "Dataset type used for evaluation."},
    )
    dataset_path: str = field(
        metadata={"help": "Root path of the evaluation dataset."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of evaluation samples. None uses the full dataset."},
    )
    result_output_path: str = field(
        default="./evaluation_outputs",
        metadata={"help": "Directory used to save evaluation predictions and metrics."},
    )


class EvaluationArguments(InferenceArguments):
    evaluation: EvaluationRuntimeArguments = field(default_factory=EvaluationRuntimeArguments)
