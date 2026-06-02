from typing import Callable
import string

import torch
import pandas as pd

from mindspeed_mm.tasks.evaluation.utils.file_utils import parse_file
from mindspeed_mm.tasks.evaluation.utils.string_utils import is_cn_string


class BasePromptTemplate:

    def __init__(self, use_custom_prompt):
        self._use_custom_prompt = use_custom_prompt
        self.device = torch.cuda.current_device()

    def build_prompt(self, line, dump_image: Callable, dataset_name=None):
        raise NotImplementedError('you must implement build_prompt')

    @staticmethod
    def check_content_type(message):
        """Check the content type of the input. Four types are allowed: str, dict, ListOfString, ListOfDict.
        """
        if isinstance(message, str):
            return 'String'
        elif isinstance(message, dict):
            return 'Dict'
        elif isinstance(message, list):
            types = [BasePromptTemplate.check_content_type(m) for m in message]
            if all(t == 'String' for t in types):
                return 'ListOfString'
            elif all(t == 'Dict' for t in types):
                return 'ListOfDict'
            else:
                raise ValueError(f'Unknown prompt type')
        else:
            raise ValueError(f'Unknown prompt type')

    @staticmethod
    def _build_prompt(line):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += '\n请直接回答选项字母。' if is_cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if is_cn_string(prompt) else '\nAnswer the question directly.'
        return prompt

    @staticmethod
    def preprocess_content(inputs):

        """Convert the raw input messages to a list of dicts.

        Args:
            inputs: raw input messages.

        Returns:
            list(dict): The preprocessed input messages. Will return None if failed to preprocess the input.
        """
        if BasePromptTemplate.check_content_type(inputs) == 'String':
            return [dict(type='text', value=inputs)]
        elif BasePromptTemplate.check_content_type(inputs) == 'Dict':
            if not ('type' in inputs and 'value' in inputs):
                raise ValueError("inputs must contain both 'type' and 'value' keys")
            return [inputs]
        elif BasePromptTemplate.check_content_type(inputs) == 'ListOfString':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime is None or mime == 'Unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif BasePromptTemplate.check_content_type(inputs) == 'ListOfDict':
            for item in inputs:
                if not ('type' in item and 'value' in item):
                    raise ValueError("item must contain both 'type' and 'value' keys")
                mime, s = parse_file(item['value'])
                if mime is None:
                    if item['type'] != 'text':
                        raise ValueError("item type must be 'text'")
                else:
                    if not mime.split('/')[0] == item['type']:
                        raise ValueError(f"The MIME type '{mime}' does not match the expected type '{item['type']}'")
                    item['value'] = s
            return inputs
        else:
            return None
