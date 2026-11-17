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

import os

import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS

from mindspeed_mm.data.data_utils.utils import VID_EXTENSIONS, DataFileReader


class MMBaseDataset(Dataset):
    """
    A base mutilmodal dataset,  it's to provide basic parameters and method

    Args: some basic parameters from dataset_param_dict in config.
        data_path(str):  csv/json/parquat file path
        data_folder(str): the root path of multimodal data
    """

    def __init__(
        self,
        data_path: str = "",
        data_folder: str = "",
        return_type: str = "list",
        data_storage_mode: str = "standard",
        **kwargs,
    ):
        self.data_path = data_path
        self.data_folder = data_folder
        self.data_storage_mode = data_storage_mode
        self.get_data = DataFileReader(data_storage_mode=data_storage_mode, **kwargs)
        self.data_samples = self.get_data(self.data_path, return_type=return_type)

    def __len__(self):
        return len(self.data_samples)

    # must be reimplemented in the subclass
    def __getitem__(self, index):
        raise AssertionError("__getitem__() in dataset is required.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        elif ext.lower() in IMG_EXTENSIONS:
            return "image"
        else:
            raise NotImplementedError(f"Unsupported file format: {ext}")
