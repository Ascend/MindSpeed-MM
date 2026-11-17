from types import SimpleNamespace
import pytest
import torch

from mindspeed_mm.fsdp.optimizer.muon_routing import _mark_muon_param_groups
from mindspeed_mm.fsdp.optimizer import optimizer as builders
from mindspeed_mm.fsdp.optimizer.optimizer import build_ep_fsdp2_optimizer
from mindspeed_mm.fsdp.params.training_args import TrainingArguments
from mindspeed_mm.optimizer.muon import (
    Muon,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_per_expert,
)
from tests.ut_fsdp.utils.utils import judge_expression


class TinyMuonModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.embedding = torch.nn.Embedding(8, 4)
        self.image_index_embedding = torch.nn.Parameter(torch.randn(4, 4))
        self.output_layer = torch.nn.Linear(4, 8, bias=False)


class TestZeroPowerViaNewtonSchulz5:
    def test_preserves_2d_matrix_shape(self):
        for shape in [(2, 4), (4, 2), (3, 3)]:
            matrix = torch.randn(*shape, dtype=torch.float32)

            result = zeropower_via_newtonschulz5(matrix, steps=2)

            judge_expression(result.shape == matrix.shape)
            # Standard Newton-Schulz returns its BF16 compute dtype.
            judge_expression(result.dtype == torch.bfloat16)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="expects a 2-D tensor"):
            zeropower_via_newtonschulz5(torch.randn(2, 3, 4), steps=2)

    @pytest.mark.parametrize("scale", [0.0, 1e-8])
    @pytest.mark.parametrize("shape", [(2, 4), (4, 2), (4, 5)])
    @pytest.mark.parametrize("steps", [5, 10])
    def test_zero_and_tiny_gradients_remain_finite(self, scale, shape, steps):
        matrix = torch.full(shape, scale, dtype=torch.float32)

        result = zeropower_via_newtonschulz5(matrix, steps=steps)

        assert torch.isfinite(result).all()
        if scale == 0.0:
            assert (result == 0).all()
        else:
            assert (result != 0).any()

    @pytest.mark.parametrize("scale", [0.0, 1e-8])
    def test_per_expert_zero_and_tiny_gradients_preserve_signal(self, scale):
        matrix = torch.full((2, 4, 5), scale, dtype=torch.float32)

        result = zeropower_via_newtonschulz5_per_expert(matrix, steps=5)

        assert torch.isfinite(result).all()
        if scale == 0.0:
            assert (result == 0).all()
        else:
            assert (result != 0).any()


class TestMarkMuonParamGroups:
    @pytest.mark.parametrize("weight_decay", [0., .1])
    def test_splits_matrix_weights_from_adamw_fallback_params(self, weight_decay):
        model = TinyMuonModel()
        named_params = dict(model.named_parameters())
        param_groups = [
            {
                "params": list(named_params.values()),
                "weight_decay": weight_decay,
                "lr": 3.0e-5,
            }
        ]

        marked_groups = _mark_muon_param_groups(model, param_groups)

        muon_param_ids = {
            id(param)
            for group in marked_groups
            if group["use_muon"]
            for param in group["params"]
        }
        fallback_param_ids = {
            id(param)
            for group in marked_groups
            if not group["use_muon"]
            for param in group["params"]
        }

        muon_names = {name for name, param in named_params.items() if id(param) in muon_param_ids}
        fallback_names = {name for name, param in named_params.items() if id(param) in fallback_param_ids}
        judge_expression(muon_names == {"linear.weight"})
        judge_expression(fallback_names == {
            "linear.bias",
            "embedding.weight",
            "image_index_embedding",
            "output_layer.weight",
        })
        judge_expression(all(group["weight_decay"] == weight_decay for group in marked_groups))
        judge_expression(all(group["lr"] == 3.0e-5 for group in marked_groups))

        # Explicit group decay wins over exclusions in the EP builder.
        optimizer = build_ep_fsdp2_optimizer(
            model, optimizer_type="muon", param_groups=param_groups, weight_decay=.2,
            no_decay_params=["bias"],
        )
        assert all(group["weight_decay"] == weight_decay for group in optimizer.param_groups)

    @pytest.mark.parametrize("keywords", [None, [], ["CUSTOM_HEAD"], "CUSTOM_HEAD"])
    def test_preserves_fallback_keywords(self, keywords):
        model = torch.nn.Module()
        model.Embed_Tokens = torch.nn.Parameter(torch.randn(4, 4))
        model.custom_head = torch.nn.Linear(4, 4, bias=False)
        model.projection = torch.nn.Linear(4, 4, bias=False)

        groups = _mark_muon_param_groups(
            model, [{"params": list(model.parameters())}], muon_fallback_param_keywords=keywords
        )
        routes = {
            parameter: group["use_muon"]
            for group in groups
            for parameter in group["params"]
        }
        assert not routes[model.Embed_Tokens]
        assert routes[model.projection.weight]
        assert routes[model.custom_head.weight] == (not bool(keywords))


class TestMuonMomentum:
    def test_uses_native_ema_momentum_buffer(self):
        parameter = torch.nn.Parameter(torch.zeros(2, 2))
        optimizer = Muon(
            [{"params": [parameter], "use_muon": True}],
            momentum=0.75,
            nesterov=False,
        )
        parameter.grad = torch.full_like(parameter, 4.0)

        ns_input = optimizer._muon_ns_input(parameter, optimizer.param_groups[0])

        torch.testing.assert_close(ns_input, torch.ones_like(parameter))
        assert "momentum_buffer" in optimizer.state[parameter]



@pytest.mark.parametrize("config,decayed_names", [
    ({}, {"0.weight", "0.bias", "1.weight", "1.bias"}),
    ({"no_decay_modules": [], "no_decay_params": []}, {"0.weight", "0.bias", "1.weight", "1.bias"}),
    ({"no_decay_params": ["bias"]}, {"0.weight", "1.weight"}),
    ({"no_decay_modules": ["LayerNorm"], "no_decay_params": []}, {"0.weight", "0.bias"}),
])
def test_no_decay_configuration_reaches_groups_and_updates(config, decayed_names, monkeypatch):
    for name in ("LOCAL_RANK", "RANK"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(builders, "get_parallel_state", lambda: SimpleNamespace(get_ep_group_size=lambda: 1))
    args = TrainingArguments(lr=.01, weight_decay=.1, optimizer="muon", **config)
    for builder in (builders.build_optimizer, build_ep_fsdp2_optimizer):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        optimizer = builder(
            model, optimizer_type=args.optimizer, lr=args.lr, weight_decay=args.weight_decay,
            no_decay_modules=args.no_decay_modules, no_decay_params=args.no_decay_params,
        )
        decay = {parameter: group["weight_decay"] for group in optimizer.param_groups for parameter in group["params"]}
        for name, parameter in model.named_parameters():
            assert decay[parameter] == (.1 if name in decayed_names else 0.)
            parameter.data.fill_(1.)
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, .999 if name in decayed_names else 1.))
