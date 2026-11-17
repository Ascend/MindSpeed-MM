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

from .greedy_dup_experts_planner import GreedyDupExpertsPlanner
from .greedy_dup_experts_executor import DupExpertExecutor


class EPBalanceStrategy:
    def __init__(self, ep_group, num_experts, max_dup_experts_num):
        self.planner = GreedyDupExpertsPlanner(
            ep_group=ep_group,
            num_experts=num_experts,
            max_dup_experts_num=max_dup_experts_num,
        )

        self.executor = DupExpertExecutor(
            ep_group=ep_group,
            num_experts=num_experts,
            max_dup_experts_num=max_dup_experts_num,
        )
