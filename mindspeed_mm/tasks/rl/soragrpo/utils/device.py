# Copyright 2024 Huawei Technologies Co., Ltd.
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
from enum import IntEnum, unique

from functools import lru_cache
import torch_npu


@unique
class NPUVersion(IntEnum):
    NONE = 0
    A2 = 2
    A3 = 3
    A5 = 5
    MAX_VERSION = 999


def check_npu_version(min_version=None, max_version=None):
    version = get_npu_version()
    if version == NPUVersion.NONE:
        return False
    if min_version is not None:
        if version < min_version:
            return False
    if max_version is not None:
        if version > max_version:
            return False
    return True


@lru_cache(maxsize=None)
def get_npu_version():
    try:
        device_name = torch_npu.npu.get_device_name()
    except Exception:
        return NPUVersion.NONE
    if "Ascend910_95" in device_name or "Ascend950" in device_name:
        return NPUVersion.A5
    elif "Ascend910_93" in device_name:
        return NPUVersion.A3
    elif "Ascend910B" in device_name or "A2G" in device_name:
        return NPUVersion.A2
    else:
        return NPUVersion.MAX_VERSION
