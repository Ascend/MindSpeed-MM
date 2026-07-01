import os
import tempfile
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mindspeed_mm.fsdp.distributed.expert_parallel.ep_balance.greedy_dup_experts_executor import DupExpertExecutor


# =============================================================================
# 分布式环境初始化 / 销毁
# =============================================================================

def _init_pg(rank: int, world_size: int, init_file: str):
    """初始化进程组，优先使用 NPU，回退到 GPU/CPU"""
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(rank)
        backend = "hccl"
    elif torch.cuda.is_available():
        torch.cuda.set_device(rank)
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"npu:{rank}")
    )


def _destroy_pg():
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_test_wrapper(rank, world_size, init_file, test_func, test_kwargs=None):
    """统一的 spawn 入口：初始化 → 执行测试 → 销毁"""
    _init_pg(rank, world_size, init_file)
    try:
        test_func(**(test_kwargs or {}))
    finally:
        _destroy_pg()


def local_device_for_rank(rank: int):
    """获取当前 rank 对应的设备"""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device(f"npu:{rank}")
    elif torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


# =============================================================================
# 真实 P2P 通信测试逻辑 (在 spawn 子进程中执行)
# =============================================================================

def _test_param_p2p_comm():
    """
    验证 async_experts_param_comm 的真实 P2P 数据传输。
    2 ranks, 4 experts (each rank has 2 local experts), max_dup=1
    Rank0 需要 Rank1 的 expert3; Rank1 需要 Rank0 的 expert0
    """
    ep_size = dist.get_world_size()
    assert ep_size == 2, "This test requires exactly 2 ranks"

    num_experts = 4
    max_dup = 1
    input_dim, output_dim = 64, 128

    executor = DupExpertExecutor(
        ep_group=None,
        num_experts=num_experts,
        max_dup_experts_num=max_dup,
    )

    device = local_device_for_rank(dist.get_rank())
    # 构造确定性参数：expert_i 的值全部为 float(i+1)
    local_expert_start = executor.ep_rank * executor.num_local_experts
    local_params = torch.stack([
        torch.full((input_dim, output_dim), float(local_expert_start + i + 1),
                   device=device, dtype=torch.float32)
        for i in range(executor.num_local_experts)
    ])

    # Rank0 wants expert3 (from rank1), Rank1 wants expert0 (from rank0)
    dup_map = [
        [3],   # rank0: dup0 ← expert3
        [0],   # rank1: dup0 ← expert0
    ]

    dup_experts = executor.async_experts_param_comm(dup_map, local_params, "param_test")
    executor.wait_async_works_finished("param_test")
    dist.barrier()

    # 验证接收到的参数值
    rank = dist.get_rank()
    expected_val = 4.0 if rank == 0 else 1.0
    received = dup_experts[0]  # dup_id=0
    assert torch.allclose(received, torch.full_like(received, expected_val)), \
        f"Rank{rank}: expected {expected_val}, got {received.mean().item()}"


def _test_grad_p2p_comm_and_accumulate():
    """
    验证梯度异步通信 + 累加的完整链路。
    dup_map=[[3],[0]] 的参数通信方向:
      Plan A: src_rank=1 → dst_rank=0 (Rank1的expert3发给Rank0)
      Plan B: src_rank=0 → dst_rank=1 (Rank0的expert0发给Rank1)
    梯度反向:
      Rank0 持有 expert3 的 dup grad → 发回 Rank1 (Plan A 反向)
      Rank1 持有 expert0 的 dup grad → 发回 Rank0 (Plan B 反向)
    """
    ep_size = dist.get_world_size()
    assert ep_size == 2

    num_experts = 4
    max_dup = 1
    input_dim, output_dim = 64, 128

    executor = DupExpertExecutor(
        ep_group=None,
        num_experts=num_experts,
        max_dup_experts_num=max_dup,
    )

    device = local_device_for_rank(dist.get_rank())
    rank = dist.get_rank()

    dup_map = [
        [3],   # rank0: dup0 ← global expert3 (owned by rank1)
        [0],   # rank1: dup0 ← global expert0 (owned by rank0)
    ]

    # 每个 rank 上 dup expert 的梯度
    # Rank0 持有 expert3 的副本 → grad=2.0 → 应发回给 Rank1
    # Rank1 持有 expert0 的副本 → grad=3.0 → 应发回给 Rank0
    dup_grad_val = 2.0 if rank == 0 else 3.0
    dup_experts_grad = torch.full(
        (max_dup, input_dim, output_dim), dup_grad_val,
        device=device, dtype=torch.bfloat16
    )

    # 发起梯度异步通信
    executor.async_experts_grad_comm(dup_map, dup_experts_grad, "grad_test")

    # 先 barrier 确保两端都发起了 P2P ops，再 wait
    dist.barrier()
    executor.wait_async_works_finished("grad_test")
    dist.barrier()

    # 检查是否真的收到了远程梯度
    received_grad_keys = list(executor.dup_experts_grad.get("grad_test", {}).keys())

    # 本地梯度初始化为 1.0
    local_grad = torch.ones(
        executor.num_local_experts, input_dim, output_dim,
        device=device, dtype=torch.bfloat16
    )

    # 累加远程梯度
    result_grad = executor.add_dup_experts_grad(local_grad, "grad_test")
    dist.barrier()

    # 验证累加结果
    result_view = result_grad.view(executor.num_local_experts, input_dim, output_dim)
    if rank == 0:
        actual_expert1 = result_view[1].mean().item()
        actual_expert0 = result_view[0].mean().item()
        assert torch.allclose(result_view[1], torch.full_like(result_view[1], 1.0)), \
            f"Rank0: expert1 grad expected 1.0, got {actual_expert1}. " \
            f"Received keys: {received_grad_keys}"
        assert torch.allclose(result_view[0], torch.full_like(result_view[0], 4.0)), \
            f"Rank0: expert0 grad expected 4.0, got {actual_expert0}"
    else:
        actual_expert0 = result_view[0].mean().item()
        actual_expert1 = result_view[1].mean().item()
        assert torch.allclose(result_view[0], torch.full_like(result_view[0], 1.0)), \
            f"Rank1: expert0 grad expected 1.0, got {actual_expert0}. " \
            f"Received keys: {received_grad_keys}"
        assert torch.allclose(result_view[1], torch.full_like(result_view[1], 3.0)), \
            f"Rank1: expert1 grad expected 3.0, got {actual_expert1}"


def _test_no_dup_no_hang():
    """全 -1 map 时不应 hang，且返回空结果"""
    executor = DupExpertExecutor(
        ep_group=None, num_experts=4, max_dup_experts_num=2,
    )
    device = local_device_for_rank(dist.get_rank())
    local_params = torch.randn(2, 64, 128, device=device, dtype=torch.float32)
    dup_map = [[-1, -1], [-1, -1]]

    dup_experts = executor.async_experts_param_comm(dup_map, local_params, "empty")
    executor.wait_async_works_finished("empty")
    dist.barrier()

    assert "empty" in executor.wait_works
    assert len(executor.wait_works["empty"]) == 0


# =============================================================================
# Pytest 测试类 (mp.spawn 调度器)
# =============================================================================

class TestDupExpertExecutorP2P:
    """真实分布式 P2P 通信测试，通过 mp.spawn 启动多进程验证"""

    def _spawn_test(self, world_size, test_func, test_kwargs=None):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as f:
            init_file = f.name
        try:
            mp.spawn(
                _run_test_wrapper,
                args=(world_size, init_file, test_func, test_kwargs),
                nprocs=world_size,
                join=True,
            )
        finally:
            if os.path.exists(init_file):
                os.remove(init_file)

    @pytest.mark.skipif(
        torch.npu.device_count() < 2,
        reason="Requires at least 2 NPUs for P2P communication test"
    )
    def test_param_p2p_comm(self):
        """验证参数异步 P2P 通信的数据正确性"""
        self._spawn_test(world_size=2, test_func=_test_param_p2p_comm)

    @pytest.mark.skipif(
        torch.npu.device_count() < 2,
        reason="Requires at least 2 NPUs for gradient P2P test"
    )
    def test_grad_p2p_comm_and_accumulate(self):
        """验证梯度异步 P2P 通信 + 累加的数值正确性"""
        self._spawn_test(world_size=2, test_func=_test_grad_p2p_comm_and_accumulate)

    @pytest.mark.skipif(
        torch.npu.device_count() < 2,
        reason="Requires at least 2 NPUs"
    )
    def test_no_dup_no_hang(self):
        """验证空 dup map 不会导致死锁"""
        self._spawn_test(world_size=2, test_func=_test_no_dup_no_hang)
