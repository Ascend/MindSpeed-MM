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

import string
from typing import Callable

import pandas as pd

from mindspeed_mm.tasks.evaluation.utils.string_utils import is_cn_string
from mindspeed_mm.tasks.evaluation.eval_prompt.build_prompt_base import BasePromptTemplate
from mindspeed_mm.tasks.evaluation.eval_datasets.datasets_base import datasets_type


class Qwen2vlPromptTemplate(BasePromptTemplate):

    def __init__(self, use_custom_prompt):
        super().__init__(use_custom_prompt)
        self.tgt_path = None

    def build_prompt(self, line, dump_image: Callable, dataset_name=None):

        self.tgt_path = dump_image(line)

        if dataset_name in {'mmmu_dev_val', 'mmmu_test'}:
            return self._build_mmmu_prompt(line)
        if datasets_type[dataset_name] == 'MCQ':
            return self._build_mcq_prompt(line)
        elif datasets_type[dataset_name] == 'VQA':
            return self._build_vqa_prompt(line)
        else:
            raise ValueError(f'Unsupported dataset: {dataset_name}')

    def is_use_custom_prompt(self, dataset_name):
        dataset_type = datasets_type[dataset_name]
        if not self._use_custom_prompt:
            return False
        if dataset_name in {'mmmu_dev_val', 'mmmu_test'}:
            return True
        if dataset_type == 'MCQ':
            return True
        if dataset_type == 'VQA' and dataset_name not in {'MMVet'}:  # MMVet VQA has it's own prompt
            return True
        return False

    def _build_mmmu_prompt(self, line):
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()
        msgs = self._build_message_template(prompt)
        return msgs

    def _build_mcq_prompt(self, line) -> list[dict[str, str]]:
        """change the prompt for MCQ dataset: use chinese prompt if the question contains chinese characters."""
        MCQ_CN_PROMPT = '请直接回答选项字母。'
        MCQ_EN_PROMPT = 'Please select the correct answer from the options above.'

        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if is_cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()

        msgs = self._build_message_template(prompt)

        return msgs

    def _build_vqa_prompt(self, line) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'

        question = line['question']
        msgs = self._build_message_template(question)

        if not msgs[-1]['type'] == 'text':
            raise ValueError("The type of  message must be 'text'")
        msgs[-1]['value'] += VQA_PROMPT
        return msgs

    def _build_message_template(self, question) -> list[dict[str, str]]:

        messages = []
        if isinstance(self.tgt_path, list):
            messages.extend([dict(type='image', value=p) for p in self.tgt_path])
        else:
            messages = [dict(type='image', value=self.tgt_path)]
        messages.append(dict(type='text', value=question))
        return messages
