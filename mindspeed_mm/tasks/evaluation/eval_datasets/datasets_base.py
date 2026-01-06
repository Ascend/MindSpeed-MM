import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from mindspeed_mm.tasks.evaluation.utils.string_utils import string_to_list, is_expected_type
from mindspeed_mm.tasks.evaluation.utils.file_utils import is_valid_image, decode_base64_to_image_file

datasets_type = {"ai2d_test": "MCQ", "docvqa_val": "VQA", "chartqa_test": "VQA", "mmmu_dev_val": "MCQ"}


class BaseEvalDataset(Dataset):

    def __init__(self, dataset_path, dataset_name):
        self.meta_only = True
        self.dataset_name = dataset_name
        self.dataset_type = None
        self.image_path = os.path.join(os.path.split(dataset_path)[0], dataset_name, 'images')
        data = self.prepare_tsv(dataset_path)
        self.data = self.prepare_image(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    @staticmethod
    def prepare_tsv(data_path):
        data = pd.read_csv(data_path, sep='\t')
        data = data[~pd.isna(data['image'])]
        data['index'] = [str(x) for x in data['index']]
        return data

    def prepare_image(self, data):

        if 'image' in data:
            data['image'] = [str(x) for x in data['image']]
            image_map = {x: y for x, y in zip(data['index'], data['image'])}
            for k in image_map:
                if len(image_map[k]) <= 64:  # Swap k & v when image < 64
                    idx = image_map[k]
                    if idx not in image_map or len(image_map.get(idx, "")) <= 64:
                        raise ValueError(
                            f"Key {k} maps to a value of length {len(image_map[k])}, but the target key {idx} "
                            f"is either not found or has a value of length {len(image_map.get(idx, ''))}.")
                    image_map[k] = image_map[idx]

            images = [string_to_list(image_map[k]) for k in data['index']]
            data['image'] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if 'image_path' in data:
            paths = [string_to_list(x) for x in data['image_path']]
            data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([is_expected_type(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        return data

    def dump_image(self, line):
        os.makedirs(self.image_path, exist_ok=True)
        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                if 'image_path' not in line:
                    raise ValueError("The required key 'image_path' is missing from the provided data.")
                for img, im_name in zip(line['image'], line['image_path']):
                    path = os.path.join(self.image_path, im_name)
                    if not is_valid_image(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = os.path.join(self.image_path, f"{line['index']}.jpg")
                if not is_valid_image(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            if 'image_path' not in line:
                raise AssertionError("Either image or image_path must be non-empty.")
            tgt_path = string_to_list(line['image_path'])

        return tgt_path

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = string_to_list(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs
