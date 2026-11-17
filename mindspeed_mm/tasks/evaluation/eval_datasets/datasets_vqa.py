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

from mindspeed_mm.tasks.evaluation.eval_datasets.datasets_base import BaseEvalDataset


class VQAEvalDataset(BaseEvalDataset):

    def __init__(self, text_path, image_path):
        super().__init__(text_path, image_path)

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        if msgs[-1]['type'] != 'text':
            raise ValueError(
                "The last message in the list is expected to be of type 'text', but got '{}'.".format(msgs[-1]['type']))
        msgs[-1]['value'] += '\nAnswer the question using a single word or phrase.'
        return msgs
