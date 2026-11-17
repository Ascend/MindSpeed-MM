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

import torch

from megatron.core import mpu


def scale_gradients(self, scaling_factor: float):
    """
    Scale all gradients inside the buffers by `scaling_factor`.

    The following code will be used for token-level loss calculation when calculate_per_token_loss=True.
    Token-level loss calculation does not require averaging over gradient accumulation steps,
    nor averaging within the DP (Data Parallelism) domain.
    FSDP2 automatically performs averaging within the DP domain, which is not what we desire.
    Therefore, in the following code, the gradient (grad) needs to be additionally multiplied by the DP size.
    """
    dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    for param in self.parameters():
        if param.grad is not None:
            param.grad.mul_(scaling_factor * dp_size)
