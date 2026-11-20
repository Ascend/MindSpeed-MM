from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import torch
import yaml
from torch import nn

from mindspeed_mm.fsdp.features.memory.recompute import recompute_modules
from mindspeed_mm.fsdp.params.feature_args import RecomputePlanConfig
from mindspeed_mm.fsdp.utils.str_match import module_name_match


def a5_features():
    root = Path(__file__).resolve().parents[4]
    return yaml.safe_load((root / "examples/glm5.3_flash/glm5.3_next_config_A5.yaml").read_text())["features"]


def test_vit_checkpoint_inputs_stay_on_device():
    features = a5_features()
    vision = "model.visual.blocks.0"
    language = "model.language_model.layers.0"
    checkpoint = features["recompute_plan"]["apply_modules"]
    offload = features["activation_offload_plan"]["apply_modules"]
    assert features["recompute"] and features["enable_activation_offload"]
    assert all(any(module_name_match(p, name) for p in checkpoint) for name in (vision, language))
    assert not any(module_name_match(p, vision) for p in offload)
    assert any(module_name_match(p, language) for p in offload)


def test_o_proj_replay_hits_cache_and_preserves_gradients():
    # Match the module names and checkpoint boundary in the shipped A5 config.
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.o_proj = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.self_attn.o_proj(x).sin()

    root = nn.Module()
    root.model = nn.Module()
    root.model.language_model = nn.Module()
    layers = root.model.language_model.layers = nn.ModuleList([Layer()])
    baseline = deepcopy(layers[0])
    config = a5_features()["recompute_plan"]
    # Other scopes select NPU-only operators outside this test's LM layer.
    scope, = [s for s in config["op_replay_scopes"] if s["name"] == "save_attention_o_proj"]
    plan = RecomputePlanConfig(**{**config, "op_replay_scopes": [scope],
                                   "apply_modules": [p for p in config["apply_modules"] if "language_model" in p]})
    cache = Mock()
    handle = object()
    cache.put.return_value = handle
    recompute_modules(root, plan, op_cache=cache)

    x = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    expected = baseline(x_ref)
    observed = layers[0](x)
    torch.testing.assert_close(observed, expected)
    cache.put.assert_called_once()
    cache.pop.return_value = cache.put.call_args.args[0]
    expected.sum().backward()
    observed.sum().backward()
    cache.pop.assert_called_once_with(handle)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(layers[0].self_attn.o_proj.weight.grad, baseline.self_attn.o_proj.weight.grad)
