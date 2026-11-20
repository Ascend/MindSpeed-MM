"""Check autograd and host-count reads at the boundary of each fused MC2 kernel."""
from unittest.mock import Mock

import pytest
import torch

from mindspeed_mm.fsdp.ops.moe_ops import gemm_mc2


@pytest.mark.parametrize("op", [gemm_mc2.AllToAllGroupedMatmul, gemm_mc2.GroupedMatmulAllToAll])
@pytest.mark.parametrize("counts_mode", ["legacy", "fallback", "cached"])
def test_mc2_autograd_host_counts(monkeypatch, op, counts_mode):
    # A single rank with one expert reduces both fused operations to the same matmul.
    backend = Mock(wraps=torch)
    backend.distributed.get_world_size.return_value = 1
    backend.distributed.get_rank.return_value = 0
    backend.distributed.get_global_rank.return_value = 0
    backend.device.return_value = torch.device("cpu")
    monkeypatch.setattr(gemm_mc2, "torch", backend)

    def gmm(x, weight, comm, size, send, recv, trans_gmm_weight=False, **kwargs):
        assert send == recv == [3]
        weight = weight[0].T if trans_gmm_weight else weight[0]
        return x @ weight, None

    def dispatch_gmm(x, *args, **kwargs):
        return (*gmm(x, *args, **kwargs), x)

    kernels = Mock()
    kernels.npu_gmm_alltoallv.side_effect = gmm
    kernels.npu_alltoallv_gmm.side_effect = dispatch_gmm
    kernels.npu_grouped_matmul.side_effect = lambda a, b, **kw: [(a[0] @ b[0]).unsqueeze(0)]
    monkeypatch.setattr(gemm_mc2, "torch_npu", kernels, raising=False)

    x = torch.arange(6, dtype=torch.float32).reshape(3, 2).requires_grad_()
    w = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4).requires_grad_()
    send, recv = torch.tensor([3]), torch.tensor([3])
    optional_args = {"legacy": (), "fallback": (None,), "cached": (([3], [3]),)}
    reads = []
    to_list = torch.Tensor.tolist

    def record_read(tensor):
        reads.append(id(tensor))
        return to_list(tensor)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", record_read)
        result, _ = op.apply(x, w, Mock(), send, recv, None, None, *optional_args[counts_mode])
        grads = torch.autograd.grad(result.sum(), (x, w))
    assert reads == ([] if counts_mode == "cached" else [id(send), id(recv)])
    expected = x @ w[0]
    torch.testing.assert_close(result, expected)
    for got, ref in zip(grads, torch.autograd.grad(expected.sum(), (x, w))):
        torch.testing.assert_close(got, ref)
