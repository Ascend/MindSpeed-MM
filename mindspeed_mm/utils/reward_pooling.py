# Copyright (c) Huawei Technologies Co., Ltd.
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


def mean_pool_logits(logits, sequence_lengths):
    """Mean-pool logits through each sequence's last valid token.

    ``sequence_lengths`` contains last-token indices. An integer sentinel is
    used when no padding token is configured, in which case the full sequence
    is valid.
    """
    if isinstance(sequence_lengths, int):
        valid_lengths = torch.full(
            (logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device
        )
    else:
        valid_lengths = (sequence_lengths + 1).clamp(min=1, max=logits.size(1))

    return torch.stack([
        logits[i, :valid_length].mean(dim=0) for i, valid_length in enumerate(valid_lengths)
    ])
