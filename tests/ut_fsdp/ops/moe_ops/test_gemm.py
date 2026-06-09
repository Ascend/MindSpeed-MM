import torch
from mindspeed_mm.fsdp.ops.moe_ops.gemm import grouped_matmul
from mindspeed_mm.fsdp.utils.device import get_device_type
from tests.ut_fsdp.utils.utils import judge_expression


def test_fused_vs_eager_consistency_for_grouped_matmul():
    device = torch.device(get_device_type())
    test_cases = [
        # (batch_size, input_dim, output_dim, num_experts, group_list)
        (8, 16, 32, 2, [4, 4]),  # Baseline: evenly split across 2 experts
        (10, 8, 16, 3, [5, 3, 2]),  # Uneven splits: different group sizes
        (6, 4, 8, 3, [2, 2, 2]),  # Small scale: input_dim < output_dim (upscale)
        (12, 16, 8, 4, [3, 3, 3, 3]),  # More experts: 4 experts with equal splits
    ]

    for batch_size, input_dim, output_dim, num_experts, group_sizes in test_cases:
        group_list = torch.tensor(group_sizes, device=device)

        x = torch.randn(batch_size, input_dim, dtype=torch.bfloat16, device=device)
        weight = torch.randn(num_experts, input_dim, output_dim, dtype=torch.bfloat16, device=device)

        fused_output = grouped_matmul(x, weight, group_list, fused=True)
        eager_output = grouped_matmul(x, weight, group_list, fused=False)

        judge_expression(fused_output.shape == eager_output.shape)
        judge_expression(fused_output.shape == (batch_size, output_dim))
        torch.testing.assert_close(fused_output, eager_output, rtol=1e-2, atol=1e-2)
