import pytest
import torch
import numpy as np
from unittest.mock import patch
from numba.typed import List
from numba import types as nb_types

from mindspeed_mm.fsdp.distributed.expert_parallel.ep_balance.greedy_dup_experts_planner import (
    update_experts,
    select_min_workload_rank_numba,
    rearrange_rank_select_expert_ids_np,
    GreedyDupExpertsPlanner,
)


# =============================================================================
# 1. PyTorch 工具函数测试
# =============================================================================
class TestUpdateExperts:
    def test_no_matches(self):
        """当 tensor 中没有 old_id 时，应原样返回"""
        tensor = torch.tensor([0, 1, 2, 3])
        result = update_experts(tensor, old_dup_experts_id=99, dup_experts_id=100, rearrange_workload=2)
        assert torch.equal(result, tensor)

    def test_full_replace(self):
        """当 rearrange_workload >= 匹配数时，全部替换"""
        tensor = torch.tensor([5, 5, 5, 1])
        result = update_experts(tensor, old_dup_experts_id=5, dup_experts_id=8, rearrange_workload=10)
        expected = torch.tensor([8, 8, 8, 1])
        assert torch.equal(result, expected)

    def test_partial_replace(self):
        """只替换前 rearrange_workload 个匹配项"""
        tensor = torch.tensor([5, 1, 5, 2, 5])
        result = update_experts(tensor, old_dup_experts_id=5, dup_experts_id=8, rearrange_workload=2)
        # 前两个5被替换，第三个5保留
        expected = torch.tensor([8, 1, 8, 2, 5])
        assert torch.equal(result, expected)

    def test_preserves_shape(self):
        """确保多维 tensor 处理后形状不变"""
        tensor = torch.tensor([[5, 1], [5, 5]])
        result = update_experts(tensor, old_dup_experts_id=5, dup_experts_id=8, rearrange_workload=2)
        assert result.shape == tensor.shape


# =============================================================================
# 2. Numba 核心算法测试
# =============================================================================
class TestSelectMinWorkloadRankNumba:
    def test_basic_selection(self):
        """选择负载最小且有槽位的 rank"""
        ep_size = 4
        max_dup_num = 2
        workload = np.array([100, 50, 80, 60], dtype=np.int64)
        # rank1 和 rank3 有空槽位 (-1)，rank1 负载更低
        dup_map = np.array([
            [0, 1],
            [-1, -1],  # rank1 有空位
            [2, 3],
            [-1, 4],   # rank3 有空位
        ], dtype=np.int64)

        chosen, success = select_min_workload_rank_numba(ep_size, max_dup_num, workload, dup_map, source_rank_id=0)
        assert success is True
        assert chosen == 1  # rank1 workload=50 最小

    def test_exclude_source_rank(self):
        """不应选择 source_rank_id 自身"""
        ep_size = 3
        max_dup_num = 1
        workload = np.array([10, 100, 50], dtype=np.int64)
        dup_map = np.array([[-1], [-1], [-1]], dtype=np.int64)

        # source=0, 虽然rank0负载最低，但应选rank2(50)而非rank1(100)
        chosen, success = select_min_workload_rank_numba(ep_size, max_dup_num, workload, dup_map, source_rank_id=0)
        assert success is True
        assert chosen == 2

    def test_no_available_slots(self):
        """所有非源 rank 都没有空槽位时返回失败"""
        ep_size = 3
        max_dup_num = 1
        workload = np.array([10, 20, 30], dtype=np.int64)
        dup_map = np.array([[0], [1], [2]], dtype=np.int64)  # 全部占满

        chosen, success = select_min_workload_rank_numba(ep_size, max_dup_num, workload, dup_map, source_rank_id=0)
        assert success is False
        assert chosen == -1


class TestRearrangeRankSelectExpertIdsNp:
    def _make_empty_update_list(self):
        return List.empty_list(nb_types.UniTuple(nb_types.int64, 4))

    def test_zero_rearrange_workload(self):
        """dup_workload <= 0 时不做任何修改"""
        num_local_experts = 4
        input_splits = np.array([10, 10], dtype=np.int64)
        group_workload = np.zeros((2, 8), dtype=np.int64)
        num_global = np.zeros((2, 12), dtype=np.int64)
        ul = self._make_empty_update_list()

        ng, isp, dw, gw, ul_out = rearrange_rank_select_expert_ids_np(
            num_local_experts, input_splits, ep_rank=0, max_dup_experts_num=2,
            num_global_tokens_per_local_expert_np=num_global, dup_workload=0,
            group_workload_np=group_workload, rearrange_rank=1, min_workload_ep_rank=1,
            dup_experts_source_id=0, dup_experts_id=5, update_experts_lists=ul
        )
        assert dw == 0
        assert len(ul_out) == 0

    def test_rearrange_on_current_ep_rank(self):
        """当 rearrange_rank == ep_rank 时，应记录到 update_experts_lists 并更新 input_splits"""
        num_local_experts = 4
        max_dup = 2
        input_splits = np.array([100, 50], dtype=np.int64)
        group_workload = np.zeros((2, 8), dtype=np.int64)
        group_workload[0, 0] = 30  # rearrange_rank=0, source_expert=0 有30的负载
        num_global = np.zeros((2, 12), dtype=np.int64)
        num_global[0, 0] = 30
        ul = self._make_empty_update_list()

        ng, isp, dw, gw, ul_out = rearrange_rank_select_expert_ids_np(
            num_local_experts, input_splits, ep_rank=0, max_dup_experts_num=max_dup,
            num_global_tokens_per_local_expert_np=num_global, dup_workload=20,
            group_workload_np=group_workload, rearrange_rank=0, min_workload_ep_rank=0,
            dup_experts_source_id=0, dup_experts_id=5, update_experts_lists=ul
        )

        # dup_workload 应减少 20
        assert dw == 0
        # group_workload 应减少
        assert gw[0, 0] == 10
        # 应有一条更新记录
        assert len(ul_out) == 1
        _, _, rw, _ = ul_out[0]
        assert rw == 20


# =============================================================================
# 3. GreedyDupExpertsPlanner 集成测试 (Mock Distributed)
# =============================================================================
class TestGreedyDupExpertsPlanner:
    @pytest.fixture
    def mock_dist_env(self):
        """Mock torch.distributed 以模拟单机单卡环境"""
        with patch("mindspeed_mm.fsdp.distributed.expert_parallel.ep_balance.greedy_dup_experts_planner.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 2
            mock_dist.get_rank.return_value = 0
            yield mock_dist

    def test_init(self, mock_dist_env):
        planner = GreedyDupExpertsPlanner(ep_group=None, num_experts=8, max_dup_experts_num=2)
        assert planner.ep_size == 2
        assert planner.num_local_experts == 4
        assert planner.ep_rank == 0

    def test_select_dup_experts_basic(self, mock_dist_env):
        """基本功能：运行后应生成有效的规划结果"""
        ep_size = 2
        num_experts = 8
        num_local_experts = 4
        max_dup = 2

        planner = GreedyDupExpertsPlanner(ep_group=None, num_experts=num_experts, max_dup_experts_num=max_dup)

        # 构造不均衡的 group_workload: rank0 负载远高于 rank1
        group_workload = np.zeros((ep_size, num_experts), dtype=np.int64)
        group_workload[0, :4] = 100  # rank0 每个expert 100
        group_workload[1, :4] = 10   # rank1 每个expert 10

        # 模拟 selected_experts: 20个token全部指向 rank0 的 experts
        selected_experts = torch.randint(0, num_local_experts, (20,), dtype=torch.long)

        planner.select_dup_experts_and_rearrange_experts(group_workload, selected_experts)

        # 验证结果已生成
        assert planner.dup_experts_map is not None
        assert planner.selected_experts_with_dup is not None
        assert planner.input_splits is not None
        assert planner.output_splits is not None
        assert planner.num_global_tokens_per_local_expert is not None

        # 验证 output_splits 总和等于总 token 数
        assert planner.output_splits.sum().item() == 220

    def test_idempotent_guard(self, mock_dist_env):
        """第二次调用应直接返回（幂等保护）"""
        planner = GreedyDupExpertsPlanner(ep_group=None, num_experts=8, max_dup_experts_num=2)
        gw = np.ones((2, 8), dtype=np.int64)
        se = torch.zeros(5, dtype=torch.long)

        planner.select_dup_experts_and_rearrange_experts(gw, se)
        first_result = planner.selected_experts_with_dup.clone()

        # 第二次调用不应改变结果
        planner.select_dup_experts_and_rearrange_experts(gw * 10, se)
        assert torch.equal(planner.selected_experts_with_dup, first_result)

    def test_clear_record(self, mock_dist_env):
        """clear 后应重置所有状态"""
        planner = GreedyDupExpertsPlanner(ep_group=None, num_experts=8, max_dup_experts_num=2)
        gw = np.ones((2, 8), dtype=np.int64)
        se = torch.zeros(5, dtype=torch.long)

        planner.select_dup_experts_and_rearrange_experts(gw, se)
        assert planner.dup_experts_map is not None

        planner.clear_record_planner_result()
        assert planner.dup_experts_map is None
        assert planner.selected_experts_with_dup is None
        assert planner.input_splits is None

    def test_balanced_workload_no_rearrange(self, mock_dist_env):
        """完全均衡的负载不应产生重排"""
        planner = GreedyDupExpertsPlanner(ep_group=None, num_experts=8, max_dup_experts_num=2)
        # 完全均匀
        gw = np.full((2, 8), 50, dtype=np.int64)
        se = torch.randint(0, 4, (20,), dtype=torch.long)

        planner.select_dup_experts_and_rearrange_experts(gw, se)

        # 均衡情况下 dup_experts_map 应全为 -1（无重复专家分配）
        assert np.all(planner.dup_experts_map == -1)
