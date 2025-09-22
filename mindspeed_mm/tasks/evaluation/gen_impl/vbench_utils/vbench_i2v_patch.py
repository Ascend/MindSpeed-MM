# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

import importlib
import inspect
import json
import os
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from megatron.training.utils import print_rank_0
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.utils import CONFIG_NAME
from transformers.utils import PushToHubMixin


def evaluate_i2v(self, videos_path, name, dimension_list=None, custom_image_folder=None, mode='vbench_standard',
                 local=False, read_frame=False, resolution="1-1"):
    from vbench2_beta_i2v.utils import init_submodules, save_json
    from vbench.distributed import get_rank

    results_dict = {}
    if dimension_list is None:
        dimension_list = self.build_full_dimension_list()
    submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame, resolution=resolution)
    cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list,
                                                   custom_image_folder=custom_image_folder, mode=mode)
    for dimension in dimension_list:
        try:
            if dimension in self.i2v_dims:
                dimension_module = importlib.import_module('mindspeed_mm.tasks.evaluation.gen_impl.vbench_utils.vbench_i2v_patch')
            else:
                dimension_module = importlib.import_module(f'vbench.{dimension}')
            evaluate_func = getattr(dimension_module, f'compute_{dimension}')
        except Exception as e:
            raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}') from e
        submodules_list = submodules_dict[dimension]
        print(f'cur_full_info_path: {cur_full_info_path}')
        results = evaluate_func(cur_full_info_path, self.device, submodules_list)
        results_dict[dimension] = results
    output_name = os.path.join(self.output_path, name + '_eval_results.json')
    if get_rank() == 0:
        save_json(results_dict, output_name)
        print(f'Evaluation results saved to {output_name}')


def compute_i2v_subject(json_dir, device, submodules_list, **kwargs):
    from vbench2_beta_i2v.utils import load_i2v_dimension_info
    from vbench2_beta_i2v.i2v_subject import i2v_subject
    from vbench.distributed import distribute_list_to_rank, get_world_size, gather_list_of_dict

    dino_model = torch.hub.load(**submodules_list).to(device)
    resolution = submodules_list['resolution']
    print("Initialize DINO success")
    video_pair_list, _ = load_i2v_dimension_info(json_dir, dimension='i2v_subject', lang='en', resolution=resolution)
    video_pair_list = distribute_list_to_rank(video_pair_list)

    all_results, video_results = i2v_subject(dino_model, video_pair_list, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results


def cuda_available_fun():
    return False


def compute_i2v_background(json_dir, device, submodules_list, **kwargs):
    from vbench2_beta_i2v.utils import load_i2v_dimension_info
    from vbench2_beta_i2v.i2v_background import i2v_background
    ori_cuda_available_fun = torch.cuda.is_available
    torch.cuda.is_available = cuda_available_fun
    from dreamsim import dreamsim
    from vbench.distributed import distribute_list_to_rank, get_world_size, gather_list_of_dict

    dream_model, preprocess = dreamsim(pretrained=True)
    torch.cuda.is_available = ori_cuda_available_fun
    resolution = submodules_list['resolution']
    print("Initialize DreamSim success")

    video_pair_list, _ = load_i2v_dimension_info(json_dir, dimension='i2v_background', lang='en', resolution=resolution)
    video_pair_list = distribute_list_to_rank(video_pair_list)

    all_results, video_results = i2v_background(dream_model, video_pair_list, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results) if len(
            video_results) != 0 else None
    return all_results, video_results


def compute_camera_motion(json_dir, device, submodules_list, **kwargs):
    from vbench.utils import load_dimension_info
    from vbench2_beta_i2v.camera_motion import camera_motion, CameraPredict
    from vbench.distributed import distribute_list_to_rank, get_world_size, gather_list_of_dict

    camera = CameraPredict(device, submodules_list)
    video_list, _ = load_dimension_info(json_dir, dimension='camera_motion', lang='en')
    video_list = distribute_list_to_rank(video_list)

    all_results, diff_type_results, video_results = camera_motion(camera, video_list)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results) if len(
            video_results) != 0 else None
    return all_results, diff_type_results, video_results


class PatchPeftConfigMixin(PushToHubMixin):
    @classmethod
    def _split_kwargs(cls, kwargs):
        hf_hub_download_kwargs = {}
        class_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            elif key in list(cls.__annotations__):
                class_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, class_kwargs, other_kwargs

    @classmethod
    def from_json_file(cls, path_json_file: str):
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = None, **kwargs):
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                print_rank_0(f"Warning: download {CONFIG_NAME}")
                config_file = hf_hub_download(
                    pretrained_model_name_or_path, CONFIG_NAME, subfolder=subfolder, local_files_only=True,
                    **hf_hub_download_kwargs
                )
            except Exception as e:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'") from e

        loaded_attributes = cls.from_json_file(config_file)

        if "peft_type" in loaded_attributes:
            peft_type = loaded_attributes["peft_type"]
            config_cls = PEFT_TYPE_TO_CONFIG_MAPPING.get(peft_type, cls)
        else:
            config_cls = cls

        kwargs = {**class_kwargs, **loaded_attributes}

        kwargs.pop('layer_replication', None)
        kwargs.pop('use_dora', None)
        kwargs.pop('use_rslora', None)
        config = config_cls(**kwargs)
        return config