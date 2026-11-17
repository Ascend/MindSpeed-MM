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
import random

import numpy as np
import torch


def set_seed(seed: int, set_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.accelerator.manual_seed(seed)
    torch.accelerator.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if set_deterministic:
        set_deterministic_algorithms()


def set_deterministic_algorithms():
    '''
    HCCL_DETERMINISTIC: is a deterministic switch in ops level, set it to 'True' to enable ops level deterministic,
        set it to 'False' to disable ops level deterministic.
    CLOSE_MATMUL_K_SHIFT: is a switch of matmul K-axis shift, set it to '1' to close matmul K-axis shift,
        set it to '0' to enable matmul K-axis shift.
    '''
    os.environ['HCCL_DETERMINISTIC'] = 'True'
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '1'
    torch.use_deterministic_algorithms(True)
