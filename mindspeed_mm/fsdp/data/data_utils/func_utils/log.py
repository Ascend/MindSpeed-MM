# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 Optuna, HuggingFace Inc. and the LlamaFactory team
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

import logging
from functools import lru_cache

from mindspeed_mm.fsdp import envs


def get_logger(name):
    _logger = logging.getLogger(name)
    # 设置日志level
    _logger.setLevel("INFO")
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(module)s - %(lineno)d - %(levelname)s - %(message)s')
    # 控制台打印
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)
    return _logger


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if envs.LOCAL_RANK == 0:
        self.info(*args, **kwargs)


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if envs.LOCAL_RANK == 0:
        self.warning(*args, **kwargs)


@lru_cache(None)
def warning_rank0_once(self: "logging.Logger", *args, **kwargs) -> None:
    if envs.LOCAL_RANK == 0:
        self.warning(*args, **kwargs)


logging.Logger.info_rank0 = info_rank0
logging.Logger.warning_rank0 = warning_rank0
logging.Logger.warning_rank0_once = warning_rank0_once
