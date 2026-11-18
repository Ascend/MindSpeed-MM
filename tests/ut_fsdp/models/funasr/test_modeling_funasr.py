"""Unit tests for FunASR model construction and module freezing."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.nn as nn

from mindspeed_mm.fsdp.config.arguments.base_args import BaseArguments
from mindspeed_mm.fsdp.utils import register


class FreezeConfig(BaseArguments):
    freeze: bool


class ModelConfig(BaseArguments):
    model_name_or_path: str
    trust_remote_code: bool = True
    audio_encoder_conf: FreezeConfig
    llm_conf: FreezeConfig
    audio_adaptor_conf: FreezeConfig
    ctc_decoder_conf: FreezeConfig


@pytest.fixture
def funasr_model(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text("model: fake\n")
    model = nn.ModuleDict({
        name: nn.Linear(2, 2)
        for name in ("audio_encoder", "llm", "audio_adaptor", "ctc_decoder")
    })
    tokenizer, frontend = object(), object()
    funasr = ModuleType("funasr")
    funasr.AutoModel = Mock(return_value=SimpleNamespace(
        model=model, kwargs={"tokenizer": tokenizer, "frontend": frontend}
    ))
    monkeypatch.setitem(sys.modules, "funasr", funasr)
    # Keep the test's model registration out of the shared registry.
    monkeypatch.setattr(register, "model_register", register.Register())
    path = Path(__file__).resolve().parents[4] / "mindspeed_mm/fsdp/models/funasr/modeling_funasr.py"
    spec = importlib.util.spec_from_file_location("_modeling_funasr_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(module, "get_device_type", lambda: "cpu")
    values = {
        "model_name_or_path": str(tmp_path),
        "trust_remote_code": True,
        "audio_encoder_conf": {"freeze": True},
        "llm_conf": {"freeze": False},
        "audio_adaptor_conf": {"freeze": True},
        "ctc_decoder_conf": {"freeze": False},
    }
    return module, model, values, tokenizer, frontend


class TestGetFunasrModel:
    def test_freeze_modules(self, funasr_model):
        module, model, values, tokenizer, frontend = funasr_model
        result, result_tokenizer, result_frontend = module.get_funasr_model(
            ModelConfig(**values), lambda model: model
        )
        assert result is model
        assert result_tokenizer is tokenizer
        assert result_frontend is frontend
        for name, component in model.items():
            frozen = values[f"{name}_conf"]["freeze"]
            assert all(param.requires_grad is (not frozen) for param in component.parameters())
            assert component.training is (not frozen)
