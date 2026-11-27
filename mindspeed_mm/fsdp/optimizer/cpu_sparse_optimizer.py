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

"""SparseAdam for host-resident embeddings whose gradient arrives as a row ledger.

The parameters are DTensors: each rank owns one row window on a size-one ``vfsdp`` shard, so FSDP treats
them as ordinary parameters and the moments are sharded DTensors too. The checkpoint therefore needs no
window-specific keys; the ledger stays on the host and each step applies the standard sparse Adam update
to a parameter's local shard.
"""
import logging

import torch
from torch.optim import _functional as _sparse_functional


# Name of the sparse member inside MultiOptimizer: its moments are stored under this entry next to the
# dense ones (see optimizer.py).
CPU_SPARSE_KEY = "cpu_sparse"

logger = logging.getLogger(__name__)


class CpuSparseAdam(torch.optim.SparseAdam):
    """SparseAdam over host-resident embeddings, driven by the row ledger of each parameter."""

    def __init__(self, params, *, lr, betas, eps):
        self._params = params  # [(parameter, row accumulator)]
        super().__init__([param for param, _ in self._params], lr=lr, betas=betas, eps=eps)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param, accumulator in self._params:
                grad = accumulator.pending_sparse_grad()
                if grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
                # The moments are DTensors, the update runs on their local shards: same arithmetic as
                # SparseAdam on the host window, and the result is already checkpointed as a shard.
                _sparse_functional.sparse_adam(
                    [param.to_local()],
                    [grad],
                    [state["exp_avg"].to_local()],
                    [state["exp_avg_sq"].to_local()],
                    [state["step"]],
                    lr=group["lr"],
                    beta1=beta1,
                    beta2=beta2,
                    eps=group["eps"],
                    maximize=False,
                )

    def zero_grad(self, set_to_none: bool = True):
        for _, accumulator in self._params:
            accumulator.clear()

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        # The DCP optimizer load rebuilds the groups from the checkpoint and drops every hyperparameter
        # except ``lr``, so a resumed member fails on its first step without them.
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)


def build_cpu_sparse_member(params, *, lr, betas, eps):
    """The sparse member for ``params`` (``[(parameter, accumulator)]``), or None without them."""
    if not params:
        return None
    elements = sum(param.to_local().numel() for param, _ in params)
    logger.info(
        "cpu sparse optimizer: SparseAdam over %d embedding(s), fp32 moments up to %.1f GiB "
        "(created on the first step)", len(params), elements * 8 / 1024 ** 3,
    )
    return CpuSparseAdam(params, lr=lr, betas=betas, eps=eps)
