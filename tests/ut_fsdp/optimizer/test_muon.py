import pytest
import torch

from mindspeed_mm.fsdp.optimizer.optimizer import _mark_muon_param_groups
from mindspeed_mm.optimizer.muon import zeropower_via_newtonschulz5
from tests.ut_fsdp.utils.utils import judge_expression


class TinyMuonModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.embedding = torch.nn.Embedding(8, 4)
        self.output_layer = torch.nn.Linear(4, 8, bias=False)


class TestZeroPowerViaNewtonSchulz5:
    def test_preserves_2d_matrix_shape(self):
        for shape in [(2, 4), (4, 2), (3, 3)]:
            matrix = torch.randn(*shape, dtype=torch.float32)

            result = zeropower_via_newtonschulz5(matrix, steps=2)

            judge_expression(result.shape == matrix.shape)
            judge_expression(result.dtype == matrix.dtype)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="expects a 2-D tensor"):
            zeropower_via_newtonschulz5(torch.randn(2, 3, 4), steps=2)


class TestMarkMuonParamGroups:
    def test_splits_matrix_weights_from_adamw_fallback_params(self):
        model = TinyMuonModel()
        named_params = dict(model.named_parameters())
        param_groups = [
            {
                "params": list(named_params.values()),
                "weight_decay": 0.1,
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
            "output_layer.weight",
        })
        judge_expression(all(group["weight_decay"] == 0.1 for group in marked_groups))
