# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import pytest
import torch
from torch import nn

from mindspeed_mm.fsdp.params.training_args import QuantizeConfig
from mindspeed_mm.fsdp.distributed.torch_parallelize import ParallelApplier


class _TinyModel(nn.Module):
    """Minimal model with a couple of Linear layers inside ``layers.N`` so
    that the W8A16 module filter (which matches ``layers.\\d+``) picks them up.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.embed = nn.Linear(hidden, hidden)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.ModuleDict({
                    "q_proj": nn.Linear(hidden, hidden),
                    "k_proj": nn.Linear(hidden, hidden),
                }),
                "mlp": nn.ModuleDict({
                    "gate_proj": nn.Linear(hidden, hidden * 2),
                    "down_proj": nn.Linear(hidden * 2, hidden),
                }),
            })
            for _ in range(2)
        ])
        self.lm_head = nn.Linear(hidden, hidden)


def _make_parallel_applier(quantization_plan):
    """Build a minimal ParallelApplier without initialising the real
    distributed process group.

    We bypass ``__init__`` and set only the attributes consumed by
    ``apply_quantization_modules``.
    """

    class _FakeFeatureConfig:
        recompute = False

    class _FakeParallelConfig:
        fully_shard_parallel_size = 1

    applier = ParallelApplier.__new__(ParallelApplier)
    applier.config = _FakeParallelConfig()
    applier.training_config = type("TC", (), {"quantization_plan": quantization_plan})()
    applier.feature_config = _FakeFeatureConfig()
    applier.parallel_state = None
    return applier


# --------------------------------------------------------------------------- #
# 1. QuantizeConfig field defaults & W8A16 configuration
# --------------------------------------------------------------------------- #

def test_quantize_config_defaults():
    cfg = QuantizeConfig()
    assert cfg.quant_recipe is None
    assert cfg.quant_format is None
    assert cfg.block_size == 32
    assert cfg.quant_apply_modules is None
    assert cfg.quant_ignored_modules is None
    assert cfg.converters is None
    assert cfg.enable_fsdp_low_precision_all_gather is True
    assert cfg.fsdp_low_precision_all_gather_mode == "on-demand"
    assert cfg.fsdp_world_size == 1


def test_quantize_config_w8a16():
    cfg = QuantizeConfig(
        quant_recipe="w8a16",
        quant_apply_modules=["model.language_model.layers.{*}"],
        quant_ignored_modules=[],
        converters=["quantize.linear.w8a16"],
    )
    assert cfg.quant_recipe == "w8a16"
    assert cfg.converters == ["quantize.linear.w8a16"]
    assert cfg.quant_apply_modules == ["model.language_model.layers.{*}"]


# --------------------------------------------------------------------------- #
# 2. apply_quantization_modules trigger logic
# --------------------------------------------------------------------------- #

def test_apply_quantization_skips_when_recipe_is_none():
    """When quant_recipe is None the method should return immediately
    without touching the model (no fsdp_turbo import needed)."""
    plan = QuantizeConfig(quant_recipe=None)
    applier = _make_parallel_applier(plan)
    model = _TinyModel()
    linear_types_before = {type(m) for m in model.modules()}
    applier.apply_quantization_modules(model)
    assert {type(m) for m in model.modules()} == linear_types_before


def test_apply_quantization_skips_when_recipe_is_empty():
    plan = QuantizeConfig(quant_recipe="")
    applier = _make_parallel_applier(plan)
    model = _TinyModel()
    applier.apply_quantization_modules(model)
    assert isinstance(model.layers[0]["self_attn"]["q_proj"], nn.Linear)


# --------------------------------------------------------------------------- #
# 3. W8A16 converter registration & model conversion (requires fsdp_turbo)
# --------------------------------------------------------------------------- #

def _import_fsdp_turbo():
    """Import fsdp_turbo.quantization which triggers W8A16 converter
    registration via the package __init__ side-effect."""
    pytest.importorskip("fsdp_turbo")
    import fsdp_turbo.quantization  # noqa: F401 – side-effect import
    from fsdp_turbo.quantization.converter.model_converter import (
        _registry_model_converter_cls,
    )
    return _registry_model_converter_cls


def test_w8a16_converter_registered():
    registry = _import_fsdp_turbo()
    assert "quantize.linear.w8a16" in registry
    assert "quantize.moe.w8a16" in registry


def test_w8a16_config_mapping():
    """Verify QuantizeConfig fields are correctly mapped to W8A16Config."""
    pytest.importorskip("fsdp_turbo")
    from fsdp_turbo.quantization.qat.w8a16.w8a16_config import (
        get_w8a16_config,
        W8A16Config,
    )

    plan = QuantizeConfig(
        quant_recipe="w8a16",
        quant_apply_modules=["model.layers.{*}"],
        quant_ignored_modules=["*lm_head"],
    )
    w8cfg = get_w8a16_config(plan)
    assert isinstance(w8cfg, W8A16Config)
    assert w8cfg.w8a16_apply_modules == ["model.layers.{*}"]
    assert w8cfg.w8a16_ignored_modules == ["*lm_head"]


def test_apply_quantization_converts_linears_to_w8a16():
    """End-to-end: ParallelApplier.apply_quantization_modules replaces
    nn.Linear inside ``layers.N`` with W8A16Linear, while embed / lm_head
    are left untouched."""
    pytest.importorskip("fsdp_turbo")
    plan = QuantizeConfig(
        quant_recipe="w8a16",
        quant_apply_modules=["model.layers.{*}"],
        quant_ignored_modules=[],
        converters=["quantize.linear.w8a16"],
    )
    applier = _make_parallel_applier(plan)
    model = _TinyModel()
    applier.apply_quantization_modules(model)

    from fsdp_turbo.quantization.qat.w8a16.w8a16_linear import W8A16Linear

    # Linears inside layers.* should be converted
    assert isinstance(model.layers[0]["self_attn"]["q_proj"], W8A16Linear)
    assert isinstance(model.layers[0]["mlp"]["gate_proj"], W8A16Linear)
    assert isinstance(model.layers[1]["self_attn"]["k_proj"], W8A16Linear)
    assert isinstance(model.layers[1]["mlp"]["down_proj"], W8A16Linear)

    # Linears outside layers.* should remain nn.Linear
    assert type(model.embed) is nn.Linear
    assert type(model.lm_head) is nn.Linear


def test_apply_quantization_respects_ignored_modules():
    """Modules matched by ignored patterns are skipped even if they fall
    inside a layers.N prefix."""
    pytest.importorskip("fsdp_turbo")
    plan = QuantizeConfig(
        quant_recipe="w8a16",
        quant_apply_modules=["model.layers.{*}"],
        quant_ignored_modules=["*down_proj"],
        converters=["quantize.linear.w8a16"],
    )
    applier = _make_parallel_applier(plan)
    model = _TinyModel()
    applier.apply_quantization_modules(model)

    from fsdp_turbo.quantization.qat.w8a16.w8a16_linear import W8A16Linear

    # down_proj is ignored → remains nn.Linear
    assert type(model.layers[0]["mlp"]["down_proj"]) is nn.Linear
    # gate_proj is not ignored → converted
    assert isinstance(model.layers[0]["mlp"]["gate_proj"], W8A16Linear)


# --------------------------------------------------------------------------- #
# 4. W8A16Linear forward / backward semantics (requires fsdp_turbo)
# --------------------------------------------------------------------------- #

def test_w8a16_linear_forward_shape_and_dtype():
    pytest.importorskip("fsdp_turbo")
    from fsdp_turbo.quantization.qat.w8a16.w8a16_linear import W8A16Linear

    linear = nn.Linear(64, 32)
    W8A16Linear.from_float(linear)
    x = torch.randn(4, 8, 64, dtype=torch.float32)
    y = linear(x)
    assert y.shape == (4, 8, 32)
    assert y.dtype == torch.float32
    assert not torch.isnan(y).any()


def test_w8a16_linear_backward_uses_ste():
    """STE: gradient should flow through unchanged (identity)."""
    pytest.importorskip("fsdp_turbo")
    from fsdp_turbo.quantization.qat.w8a16.w8a16_linear import (
        W8A16Linear,
        W8A16FakeQuantization,
    )

    w = torch.randn(32, 64, requires_grad=True)
    w_q = W8A16FakeQuantization.apply(w)
    w_q.sum().backward()
    assert w.grad is not None
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_w8a16_linear_forward_quantizes_weight():
    """The fake-quantized weight should differ from the original weight
    (quantization introduces non-trivial rounding)."""
    pytest.importorskip("fsdp_turbo")
    from fsdp_turbo.quantization.qat.w8a16.w8a16_linear import (
        W8A16Linear,
        W8A16FakeQuantization,
    )

    linear = nn.Linear(64, 32)
    W8A16Linear.from_float(linear)
    w_orig = linear.weight.detach().clone()
    w_q = W8A16FakeQuantization.apply(linear.weight)
    assert not torch.allclose(w_q, w_orig, atol=1e-6)


# --------------------------------------------------------------------------- #
# 5. W8A16 converter build via ModelConvertersContainer
# --------------------------------------------------------------------------- #

def test_build_model_converter_with_w8a16():
    """build_model_converter should build a container that converts the
    model in-place."""
    pytest.importorskip("fsdp_turbo")
    from fsdp_turbo.quantization.converter.model_converter import (
        build_model_converter,
    )

    plan = QuantizeConfig(
        quant_recipe="w8a16",
        quant_apply_modules=["model.layers.{*}"],
        quant_ignored_modules=[],
        converters=["quantize.linear.w8a16"],
    )
    container = build_model_converter(plan)
    model = _TinyModel()
    container.convert(model)

    from fsdp_turbo.quantization.qat.w8a16.w8a16_linear import W8A16Linear
    assert isinstance(model.layers[0]["self_attn"]["q_proj"], W8A16Linear)
