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

import os.path

from mindspeed_mm.data.datasets.mm_base_dataset import MMBaseDataset


class BaseGenEvalDataset(MMBaseDataset):
    def __init__(self,
            basic_param: dict,
            **kwargs
    ):
        super().__init__(**basic_param, **kwargs)

    def __getitem__(self, index):
        return self.data_samples[index]


class DimCounter:
    def __init__(self):
        self.cnt = {}

    def count(self, dims):
        for dim in set(dims):
            self.cnt[dim] = self.cnt.get(dim, 0) + 1

    def min_count(self, dims):
        return min(self.cnt.get(dim, 0) for dim in set(dims))


vbench_dim_proj = {
    "subject_consistency": "subject_consistency",
    "background_consistency": "scene",
    "aesthetic_quality": "overall_consistency",
    "imaging_quality": "overall_consistency",
    "object_class": "object_class",
    "multiple_objects": "multiple_objects",
    "color": "color",
    "spatial_relationship": "spatial_relationship",
    "scene": "scene",
    "temporal_style": "temporal_style",
    "overall_consistency": "overall_consistency",
    "human_action": "human_action",
    "temporal_flickering": "temporal_flickering",
    "motion_smoothness": "subject_consistency",
    "dynamic_degree": "subject_consistency",
    "appearance_style": "appearance_style"
}


def prepare_dims(dimensions):
    if dimensions is None or len(dimensions) == 0:
        return []
    dims = set(dimensions)
    for dim in set(dimensions):
        if dim in vbench_dim_proj:
            dims.add(vbench_dim_proj[dim])
    return list(dims)


class VbenchGenEvalDataset(BaseGenEvalDataset):
    def __init__(self,
            basic_param: dict,
            extra_param: dict,
            dimensions: list = None,
    ):
        super().__init__(basic_param)
        self.dimensions = prepare_dims(dimensions)
        self.augment = extra_param.get("augment", False)
        self.prompts_per_dim = extra_param.get("prompts_per_dim", 0)
        self.samples_per_prompt = extra_param.get("samples_per_prompt", 5)
        self.prompt_file = extra_param.get("prompt_file", "all_dimension.txt")
        self.augmented_prompt_file = extra_param.get("augmented_prompt_file",
                                                     "augmented_prompts/gpt_enhanced_prompts/all_dimension_longer.txt")
        self.captions = []
        self.augmented_captions = []
        if self.dimensions is not None and len(self.dimensions) > 0:
            prompt_files = [os.path.join("prompts_per_dimension", f"{dim}.txt") for dim in self.dimensions]
            for prompt_file in prompt_files:
                if not os.path.exists(os.path.join(self.data_folder, prompt_file)):
                    continue
                if self.prompts_per_dim > 0:
                    self.captions += self.get_data(os.path.join(self.data_folder, prompt_file))[:self.prompts_per_dim]
                else:
                    self.captions += self.get_data(os.path.join(self.data_folder, prompt_file))
            if self.augment:
                augmented_prompt_files = [
                    os.path.join("augmented_prompts/gpt_enhanced_prompts/prompts_per_dimension_longer",
                                 f"{dim}_longer.txt") for dim in self.dimensions
                ]
                for prompt_file in augmented_prompt_files:
                    if not os.path.exists(os.path.join(self.data_folder, prompt_file)):
                        continue
                    if self.prompts_per_dim > 0:
                        self.augmented_captions += self.get_data(os.path.join(self.data_folder, prompt_file))[
                                                   :self.prompts_per_dim]
                    else:
                        self.augmented_captions += self.get_data(os.path.join(self.data_folder, prompt_file))
        else:
            self.captions += self.get_data(os.path.join(self.data_folder, self.prompt_file))
            if self.augment:
                self.augmented_captions += self.get_data(os.path.join(self.data_folder, self.augmented_prompt_file))

    def prepare_item(self):
        return {
            "caption": "",
            "prefix": "",
        }

    def __getitem__(self, index):
        caption_index = index // self.samples_per_prompt
        sample_index = index % self.samples_per_prompt
        item = self.prepare_item()
        item["prefix"] = self.captions[caption_index] + f"-{sample_index}"
        item["caption"] = self.augmented_captions[caption_index] if self.augment else self.captions[caption_index]
        return item

    def __len__(self):
        return len(self.captions) * self.samples_per_prompt


class VbenchI2VGenEvalDataset(BaseGenEvalDataset):
    def __init__(self,
            basic_param: dict,
            extra_param: dict,
            dimensions: list,
    ):
        super().__init__(basic_param)
        self.ratio = extra_param.get("ratio", "16-9")
        self.prompts_per_dim = extra_param.get("prompts_per_dim", 0)
        self.samples_per_prompt = extra_param.get("samples_per_prompt", 5)
        self.filter_by_dimension(dimensions)

    def prepare_item(self):
        return {
            "caption": "",
            "prefix": "",
            "image": "",
        }

    def filter_by_dimension(self, dimensions):
        if dimensions is None or len(dimensions) == 0:
            return
        dims = set(dimensions)
        new_data_samples = []
        cnt = DimCounter()
        for sample in self.data_samples:
            if any(dim in dims for dim in sample.get("dimension")):
                if 0 < self.prompts_per_dim <= cnt.min_count(sample.get("dimension")):
                    continue
                new_data_samples += [sample]
                cnt.count(sample.get("dimension"))
        self.data_samples = new_data_samples

    def __getitem__(self, index):
        caption_index = index // self.samples_per_prompt
        sample_index = index % self.samples_per_prompt
        item = self.prepare_item()
        data = self.data_samples[caption_index]
        item["caption"] = data.get("prompt_en")
        item["prefix"] = data.get("prompt_en") + f"-{sample_index}"
        item["image"] = os.path.join(self.data_folder, "crop", self.ratio, data.get("image_name"))
        return item

    def __len__(self):
        return len(self.data_samples) * self.samples_per_prompt
