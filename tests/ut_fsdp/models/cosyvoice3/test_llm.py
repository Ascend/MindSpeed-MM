"""Unit tests for CosyVoice3 sampling configuration."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn

from mindspeed_mm.fsdp.config.arguments.base_args import BaseArguments
from mindspeed_mm.fsdp.utils import register


class SamplingConfig(BaseArguments):
    pass


@pytest.fixture
def cosyvoice3_lm(monkeypatch):
    monkeypatch.setattr(register, "model_register", register.Register())
    path = Path(__file__).resolve().parents[4] / "mindspeed_mm/fsdp/models/cosyvoice3/llm.py"
    spec = importlib.util.spec_from_file_location("_cosyvoice3_llm_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    # Do not load pretrained weights; sampling setup uses real CPU layers.
    monkeypatch.setattr(module, "Qwen2Encoder", lambda _: nn.Identity())
    return module.CosyVoice3LM


def model_config(sampling):
    return SimpleNamespace(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=8,
        llm_encoder="unused",
        lsm_weight=0.1,
        length_normalized_loss=True,
        mix_ratio=[5, 15],
        sampling=sampling,
    )


class TestCosyVoice3LM:
    @pytest.mark.parametrize("values", [
        {"top_p": 0.9, "top_k": 12, "win_size": 6, "tau_r": 0.2},
        {"top_k": 12},
    ], ids=["configured", "partial_defaults"])
    def test_sampling_config(self, cosyvoice3_lm, values):
        model = cosyvoice3_lm(model_config(SamplingConfig(**values)))
        expected = {"top_p": 0.8, "top_k": 25, "win_size": 10, "tau_r": 0.1}
        expected.update(values)
        assert model.sampling.keywords == expected
