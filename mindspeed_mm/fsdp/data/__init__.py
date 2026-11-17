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

# pylint: skip-file
from mindspeed_mm.fsdp.data.dataloader.dataloader import (
    prepare_base_dataloader,
    prepare_sampler_dataloader,
)
from mindspeed_mm.fsdp.utils.register import data_register


def build_mm_dataset(dataset_param):
    """
    Build a multimodal dataset based on different tasks

    Args:
        dataset_param
    Return:
        dataset
    """
    if not isinstance(dataset_param, dict):
        dataset_param = dataset_param.to_dict()
    for check_key in ["dataset_type", "basic_parameters", "preprocess_parameters"]:
        if check_key not in dataset_param:
            raise AssertionError(f"Key parameter missing: {check_key}")
    dataset_type = dataset_param["dataset_type"]
    basic_param = dataset_param["basic_parameters"]
    preprocess_param = dataset_param["preprocess_parameters"]
    dataset_cls_or_func = data_register.get(dataset_type)
    return dataset_cls_or_func(
        basic_param=basic_param,
        preprocess_param=preprocess_param,
        dataset_param=dataset_param
    )


def build_mm_dataloader(
        dataset,
        dataloader_param,
        process_group=None,
        dataset_param=None,
        model=None,
    ):
    """
    Build a multimodal dataloader based on different tasks

    dataloader_type interpretation:
    base: raw dataloader based on torch.utils.data.DataLoader
    sampler: prepare a dataloader for distributed training by building a specific sampler
    variable: used for variable dataset

    Args:
        dataloader_param_dict
    Return:
        dataloader
    """
    if not isinstance(dataloader_param, dict):
        dataloader_param = dataloader_param.to_dict()
    if "dataloader_mode" not in dataloader_param:
        raise AssertionError("Key parameter missing: dataloader_mode")
    dataloader_mode = dataloader_param.pop("dataloader_mode")

    if dataloader_mode == "base":
        data_loader = prepare_base_dataloader(
            dataset, **dataloader_param,
            dataset_param=dataset_param,
            model=model
        )
        return data_loader
    elif dataloader_mode == "sampler":
        data_loader = prepare_sampler_dataloader(
            dataset, **dataloader_param,
            process_group=process_group,
            dataset_param=dataset_param,
            model=model,
        )
        return data_loader
    else:
        # keyerror, pop before
        raise NotImplementedError(f"Unsupported dataloader_mode: {dataloader_mode}")
