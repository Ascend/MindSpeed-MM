import torch
from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.ops.moe_ops.unpermute import unpermute
from tests.ut_fsdp.utils.utils import judge_expression


def test_fused_vs_eager_consistency_for_unpermute_when_probs_is_none():
    device = torch.device(get_device_type())
    num_tokens = 10
    hidden_size = 32
    restore_shape = [8, hidden_size]

    permuted_tokens = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    indices = torch.randperm(num_tokens, dtype=torch.int32, device=device)
    eager_result = unpermute(permuted_tokens, indices, fused=False)
    fused_result = unpermute(permuted_tokens, indices, fused=True)
    torch.testing.assert_close(eager_result, fused_result)

    indices = torch.randperm(restore_shape[0], dtype=torch.int32, device=device)
    eager_result = unpermute(permuted_tokens, indices, restore_shape=restore_shape, fused=False)
    fused_result = unpermute(permuted_tokens, indices, restore_shape=restore_shape, fused=True)
    torch.testing.assert_close(eager_result, fused_result)



def test_fused_vs_eager_consistency_for_unpermute_when_probs_is_notnone():
    device = torch.device(get_device_type())
    num_tokens = 10
    hidden_size = 32
    topk_list = [1, 2]

    permuted_tokens = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    indices = torch.randperm(num_tokens, dtype=torch.int32, device=device)
    for topk in topk_list:
        probs = torch.rand(num_tokens // topk, topk, dtype=torch.bfloat16, device=device)
        eager_result = unpermute(permuted_tokens, indices, probs, fused=False)
        fused_result = unpermute(permuted_tokens, indices, probs, fused=True)

        torch.testing.assert_close(eager_result, fused_result, rtol=5e-2, atol=1e-2)

    restore_shape = [8, hidden_size]
    permuted_tokens = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    for topk in topk_list:
        indices = torch.randperm(restore_shape[0], dtype=torch.int32, device=device)
        probs = torch.rand(restore_shape[0] // topk, topk, dtype=torch.bfloat16, device=device)
        eager_result = unpermute(permuted_tokens, indices, probs, restore_shape=restore_shape, fused=False)
        fused_result = unpermute(permuted_tokens, indices, probs, restore_shape=restore_shape, fused=True)
