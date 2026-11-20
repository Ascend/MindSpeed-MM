"""Exercise the MC2 dispatcher with two asymmetric routes and mocked collectives."""
from unittest.mock import Mock

import pytest
import torch

from mindspeed_mm.fsdp.distributed.expert_parallel import ep_dispatcher as ep


@pytest.mark.parametrize("reuse", [True, False], ids=["default", "opt_out"])
def test_mc2_host_counts_follow_each_route(monkeypatch, reuse):
    if reuse:
        monkeypatch.delenv("MM_MC2_REUSE_HOST_COUNTS", raising=False)
    else:
        monkeypatch.setenv("MM_MC2_REUSE_HOST_COUNTS", "0")
    monkeypatch.setattr(ep, "FORCE_EP_BALANCE", False)
    monkeypatch.setattr(ep, "EP_BINCOUNT_DISPATCH", True)
    monkeypatch.setattr(ep.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(ep.dist, "get_rank", lambda group: 1)

    def gather(output, local, group):
        output.copy_(torch.stack([local.new_tensor([1, 0, 0, 2]), local]))

    monkeypatch.setattr(ep.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(ep, "permute", lambda x, indices, fused: (x, None))
    monkeypatch.setattr(ep, "unpermute", lambda x, indices, probs, fused: x)
    fc1 = Mock(side_effect=lambda **kw: kw["inputs"])
    fc2 = Mock(side_effect=lambda **kw: kw["inputs"])
    monkeypatch.setattr(ep, "all2all_grouped_matmul", fc1)
    monkeypatch.setattr(ep, "grouped_matmul_all2all", fc2)

    for experts, send, recv in (
        ([0, 2, 2], [1, 0, 2, 0], [0, 2, 2, 0]),
        ([1, 3, 3], [0, 1, 0, 2], [0, 2, 0, 2]),
    ):
        indices = torch.tensor(experts).view(-1, 1)
        ep.ep_mc2_forward(4, torch.ones(3, 1), indices, torch.ones(3, 4),
                          None, None, activation=torch.nn.Identity())
        first, second = fc1.call_args.kwargs, fc2.call_args.kwargs
        assert first["send_counts"].tolist() == second["recv_counts"].tolist() == send
        assert first["recv_counts"].tolist() == second["send_counts"].tolist() == recv
        if reuse:
            assert first["host_counts"] == (send, recv)
            assert second["host_counts"][0] is first["host_counts"][1]
            assert second["host_counts"][1] is first["host_counts"][0]
        else:
            assert first["host_counts"] is None and second["host_counts"] is None
