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

import logging

import torch
from torch.distributed._composable import replicate

from mindspeed_mm.fsdp.log import print_rank
from mindspeed_mm.fsdp.params.training_args import TrainingArguments
from mindspeed_mm.fsdp.utils.device import get_device_type, get_torch_device


logger = logging.getLogger(__name__)


def distributed_data_parallel_modules(
    model: torch.nn.Module,
    process_group,
    training_config: TrainingArguments,
) -> torch.nn.Module:
    """Replicate without changing parameter dtype or replacing forward.

    The reducer collects parameters on the first forward, after the existing
    meta materialization, weight loading and embedding re-tying steps. Nested
    fully_shard modules remain managed by FSDP2.
    """
    device_type = get_device_type()
    if not training_config.init_model_with_meta_device:
        model = model.to(device_type)

    if torch.distributed.get_world_size(process_group) == 1:
        print_rank(logger.info, "Single-rank DDP group: no reducer is needed.")
        return model

    model = replicate(
        model,
        process_group=process_group,
        find_unused_parameters=True,
        device_id=torch.device("cpu") if device_type == "cpu"
        else torch.device(device_type, get_torch_device().current_device()),
    )
    print_rank(
        logger.info,
        "DDP enabled (fully_shard_parallel_size=1); parameter dtype is unchanged "
        "and nested FSDP2-managed modules are excluded from the DDP reducer.",
    )
    return model


def configure_ddp_gradient_sync(model: torch.nn.Module, is_last_step: bool) -> None:
    """Defer DDP reduction while preserving nested expert FSDP synchronization."""
    # Single-rank models have no DDP reducer.
    if hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(is_last_step)
    for module in model.modules():
        if hasattr(module, "set_is_last_backward") and hasattr(module, "set_requires_all_reduce"):
            module.set_is_last_backward(is_last_step)
            module.set_requires_all_reduce(is_last_step, recurse=False)
