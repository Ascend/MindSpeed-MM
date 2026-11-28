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

from mindspeed_mm.utils.reward_pooling import mean_pool_logits


def test_mean_pool_includes_last_valid_token_for_padded_batch():
    logits = torch.tensor([
        [[1.0], [3.0], [5.0], [100.0]],
        [[10.0], [30.0], [100.0], [100.0]],
    ])
    last_token_indices = torch.tensor([2, 1])

    pooled = mean_pool_logits(logits, last_token_indices)

    torch.testing.assert_close(pooled, torch.tensor([[3.0], [20.0]]))


def test_mean_pool_handles_single_valid_token_without_nan():
    logits = torch.tensor([[[7.0], [100.0], [100.0]]])
    last_token_indices = torch.tensor([0])

    pooled = mean_pool_logits(logits, last_token_indices)

    torch.testing.assert_close(pooled, torch.tensor([[7.0]]))
    assert torch.isfinite(pooled).all()


def test_mean_pool_uses_full_sequence_when_padding_token_is_unavailable():
    logits = torch.tensor([[[1.0], [3.0], [5.0]]])

    pooled = mean_pool_logits(logits, -1)

    torch.testing.assert_close(pooled, torch.tensor([[3.0]]))
