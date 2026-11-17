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

from typing import Callable

from mindspeed_mm.tasks.evaluation.eval_prompt.build_prompt_base import BasePromptTemplate
from mindspeed_mm.tasks.evaluation.eval_datasets.datasets_base import datasets_type


class InternvlPromptTemplate(BasePromptTemplate):

    def __init__(self, use_custom_prompt):
        super().__init__(use_custom_prompt)

    def build_prompt(self, line, dump_image: Callable, dataset_name=None):
        tgt_path = dump_image(line)

        if datasets_type[dataset_name] == "MCQ":
            prompt = self.build_multi_choice_prompt(line)
        elif datasets_type[dataset_name] == "VQA":
            prompt = line['question'] + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def build_multi_choice_prompt(self, line):
        return super()._build_prompt(line)

    @staticmethod
    def is_use_custom_prompt(dataset_name):
        if dataset_name in ['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'MMBench-Video', 'Video-MME', 'MVBench',
                            'Video']:
            return False
        else:
            return True
