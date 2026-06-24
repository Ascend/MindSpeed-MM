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
