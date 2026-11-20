from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("transformers", minversion="5.16.1", reason="GLM-5.3-Flash example dependency")
from mindspeed_mm.fsdp.distributed.expert_parallel import ep_chunkmoe, ep_dispatcher
from mindspeed_mm.fsdp.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
from mindspeed_mm.fsdp.params.parallel_args import EPPlanConfig


@pytest.mark.parametrize("dispatcher", ["alltoall", "chunkmoe"])
def test_expert_dispatch_passes_plan_only_to_chunkmoe(monkeypatch, dispatcher):
    experts = Glm5NextTextExperts(SimpleNamespace(
        num_local_experts=2, hidden_size=4, moe_intermediate_size=2, swiglu_limit=0.0,
    ))
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    plan = EPPlanConfig(dispatcher=dispatcher)
    group = object()
    impl = Mock(return_value=x.reshape(-1, 4))
    if dispatcher == "chunkmoe":
        monkeypatch.setattr(ep_chunkmoe, "ep_chunkmoe_forward", impl)
    else:
        monkeypatch.setattr(ep_dispatcher, "ep_forward", impl)
    indices = torch.zeros(6, 1, dtype=torch.long)
    result = experts.ep_forward(x, indices, torch.ones(6, 1), group, plan)
    impl.assert_called_once()
    assert impl.call_args.kwargs["ep_group"] is group
    if dispatcher == "chunkmoe":
        assert impl.call_args.kwargs["ep_plan"] is plan
    else:
        assert "ep_plan" not in impl.call_args.kwargs
    torch.testing.assert_close(result, x)
