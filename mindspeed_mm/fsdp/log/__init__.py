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

"""
Unified logging utilities for the MindSpeed-MM FSDP backend.

This package hosts parallel, independent modules:
- ``logging`` (planned by RFC #664): unified text logging.
- ``tensorboard``: TensorBoard scalar recording backend.
- ``metrics``: unified metrics entry point with pluggable handler backends.

The modules are independent of each other and can be enabled separately.
"""

from mindspeed_mm.fsdp.log.tensorboard import tb_writer
from mindspeed_mm.fsdp.log.metrics import metrics, TensorBoardHandler
from mindspeed_mm.fsdp.log.logging import init_logger, get_logger
from mindspeed_mm.fsdp.log.utils import print_rank


__all__ = [
    "tb_writer",
    "metrics",
    "TensorBoardHandler",
    "init_logger",
    "get_logger",
    "print_rank",
]
