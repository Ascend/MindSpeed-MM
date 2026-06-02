import string

import pandas as pd

from mindspeed_mm.tasks.evaluation.eval_datasets.datasets_base import BaseEvalDataset
from mindspeed_mm.tasks.evaluation.utils.string_utils import string_to_list, is_expected_type


class MMMUEvalDataset(BaseEvalDataset):

    def __init__(self, dataset_path, dataset_name):
        super().__init__(dataset_path, dataset_name)

    @staticmethod
    def split_MMMU(msgs):
        text, images = None, []
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                if text is not None:
                    raise ValueError("text messages found")
                text = s['value']
        text_segs = text.split('<image ')
        if len(text_segs) == 1:
            return msgs

        segs = [dict(type='text', value=text_segs[0])]
        for i, seg in enumerate(text_segs):
            if i == 0:
                continue
            if not (is_expected_type(seg[0], int) and seg[1] == '>'):
                raise ValueError("Invalid image tag format")
            image_idx = int(seg[0]) - 1
            segs.append(dict(type='image', value=images[image_idx]))
            segs.append(dict(type='text', value=seg[2:]))
        return segs

    def _build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = string_to_list(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
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

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def build_prompt(self, line):
        msgs = self._build_prompt(line)
        msgs = self.split_MMMU(msgs)
        return msgs
