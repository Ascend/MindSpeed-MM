"""Unit tests for multi-level configuration extensions with dot notation."""

# 在仓库根目录执行以下命令，可单独运行 config 测试用例：
# NON_MEGATRON=true python -m pytest tests/ut_fsdp/config/test_config_arguments.py -v

import sys
from dataclasses import field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
import yaml
from pydantic import ValidationError

from mindspeed_mm.fsdp.config.arguments.base_args import BaseArguments
from mindspeed_mm.fsdp.config.config_manager import ConfigManager
from mindspeed_mm.fsdp.params.argument import Arguments
from mindspeed_mm.fsdp.params.feature_args import FeatureArguments
from mindspeed_mm.fsdp.params.parallel_args import ParallelArguments
from mindspeed_mm.fsdp.params.tools_args import ToolsArguments
from mindspeed_mm.fsdp.params.training_args import TrainingArguments


@pytest.fixture(autouse=True)
def isolate_config_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(sys, "argv", ["test_config_arguments.py"])
    # YAML registration changes class metadata. Restore it after every test
    # so these tests do not affect other configuration consumers in the suite.
    pending = [BaseArguments]
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if "_dynamic_fields_info" in cls.__dict__:
            monkeypatch.setattr(cls, "_dynamic_fields_info", cls._dynamic_fields_info.copy())


def test_multi_level_registration_initializes_defaults_and_input_values():
    class Config(BaseArguments):
        pass

    Config.register_field("model.encoder.hidden_size", int, 128)
    Config.register_field("model.encoder.dropout", float, 0.1)
    Config.register_field("model.decoder.num_layers", int, 2)

    default = Config()
    supplied = Config(model={"encoder": {"hidden_size": 256}})

    assert default.model.encoder.hidden_size == 128
    assert supplied.model.encoder.hidden_size == 256
    assert supplied.model.encoder.dropout == 0.1
    assert supplied.model.decoder.num_layers == 2


def test_registration_extends_existing_annotated_config():
    class Encoder(BaseArguments):
        hidden_size: int = 128

    class Config(BaseArguments):
        encoder: Encoder = field(default_factory=Encoder)

    Config.register_field("encoder.attention.num_heads", int, 8)
    config = Config(encoder={"hidden_size": 256, "attention": {"num_heads": 16}})

    assert config.encoder.hidden_size == 256
    assert config.encoder.attention.num_heads == 16


def test_nested_config_accepts_an_instance_of_registered_type():
    class Config(BaseArguments):
        pass

    Config.register_field("encoder.hidden_size", int, 128)
    encoder = Config(encoder={"hidden_size": 256}).encoder
    config = Config(encoder=encoder)

    assert config.encoder is encoder
    assert config.encoder.hidden_size == 256


@pytest.mark.parametrize("missing", [{}, {"encoder": {}}])
def test_required_nested_field_cannot_be_omitted(missing):
    class Config(BaseArguments):
        pass

    Config.register_field("encoder.model_path", str, required=True)
    with pytest.raises(ValueError, match="model_path"):
        Config(**missing)

    assert Config(encoder={"model_path": "/models/encoder"}).encoder.model_path == "/models/encoder"


def test_nested_default_instances_are_independent():
    class Config(BaseArguments):
        pass

    Config.register_field("encoder.hidden_size", int, 128)
    first, second = Config(), Config()
    first.encoder.hidden_size = 256

    assert second.encoder.hidden_size == 128


@pytest.mark.parametrize("value_type", [int, Dict[str, int]])
def test_nested_registration_rejects_non_config_prefix(value_type):
    class Config(BaseArguments):
        pass

    Config.register_field("encoder", value_type)
    with pytest.raises(TypeError, match="not a BaseArguments subclass"):
        Config.register_field("encoder.hidden_size", int, 128)


@pytest.mark.parametrize(("value_type", "valid", "invalid"), [
    (int, 3, "invalid"),
    (float, 0.1, "invalid"),
    (bool, True, "invalid"),
    (List[int], [1, 2], [1, "invalid"]),
    (Dict[str, int], {"layers": 2}, {"layers": "invalid"}),
])
def test_dynamic_field_checks_initialization_and_assignment(value_type, valid, invalid):
    class Config(BaseArguments):
        pass

    Config.register_field("value", value_type)
    with pytest.raises(TypeError, match="value"):
        Config(value=invalid)

    config = Config(value=valid)
    with pytest.raises(TypeError, match="value"):
        config.value = invalid
    assert config.value == valid


def test_invalid_dynamic_default_is_rejected_at_registration():
    class Config(BaseArguments):
        pass

    with pytest.raises(TypeError, match="Default value type error"):
        Config.register_field("num_layers", int, "invalid")


def test_optional_dynamic_field_accepts_none():
    class Config(BaseArguments):
        pass

    Config.register_field("model_path", Optional[str])
    config = Config(model_path=None)
    config.model_path = "/models/encoder"
    config.model_path = None
    assert config.model_path is None


def test_subclass_registration_does_not_add_fields_to_parent_or_sibling():
    class Parent(BaseArguments):
        pass

    Parent.register_field("num_layers", int, 2)

    class Child(Parent):
        pass

    class Sibling(Parent):
        pass

    Child.register_field("num_heads", int, 8)
    assert Child().num_layers == 2
    assert Child().num_heads == 8
    assert not hasattr(Parent(), "num_heads")
    assert not hasattr(Sibling(), "num_heads")


def test_flat_dynamic_fields_and_static_nested_fields_export_to_dict():
    class Encoder(BaseArguments):
        hidden_size: int = 128

    class Config(BaseArguments):
        encoder: Encoder = field(default_factory=Encoder)

    Config.register_field("num_layers", int, 2)
    assert Config(num_layers=4).to_dict() == {"encoder": {"hidden_size": 128}, "num_layers": 4}


@pytest.mark.parametrize(("config_class", "data"), [
    (ParallelArguments, {"tensor_parallel_szie": 1}),
    (ParallelArguments, {"fsdp_plan": {"param_dtyep": "bf16"}}),
    (ParallelArguments, {"tp_plan": {"colwise_paralell": []}}),
    (ParallelArguments, {"ep_plan": {"dispatcherr": "alltoall"}}),
    (ToolsArguments, {"profil": {}}),
    (ToolsArguments, {"tensorboard": {"enabel": True}}),
    (ToolsArguments, {"profile": {"profile_typ": "static"}}),
    (ToolsArguments, {"memory_profile": {"start_stepp": 1}}),
    (FeatureArguments, {"enable_chunk_los": True}),
    (FeatureArguments, {"recompute_plan": {"use_reentrent": True}}),
    (FeatureArguments, {"chunkloss_plan": {"chunk_szie": 512}}),
    (FeatureArguments, {"loss_cfg": {"loss_typ": "raw"}}),
    (FeatureArguments, {"activation_offload_plan": {"apply_moduels": []}}),
    (FeatureArguments, {"chunkmbs_plan": {"batch_dimm": 0}}),
    (FeatureArguments, {"ep_balance_plan": {"max_dup_experts_numm": 2}}),
])
def test_fsdp_sections_allow_extra_fields(config_class, data):
    # Existing model-specific options remain accepted in every section.
    config = config_class(**data)
    dumped = config.to_dict()
    for name, value in data.items():
        if isinstance(value, dict):
            assert all(dumped[name][key] == item for key, item in value.items())
        else:
            assert dumped[name] == value


def test_parallel_arguments_read_launcher_context_and_compute_parallel_sizes(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    config = ParallelArguments(expert_parallel_size=2)

    assert (config.local_rank, config.global_rank, config.world_size) == (1, 3, 8)
    assert config.fully_shard_parallel_size == 8
    assert config.data_parallel_size == 8
    assert config.expert_fully_shard_parallel_size == 4


def test_parallel_arguments_reject_inconsistent_data_parallel_size():
    with pytest.raises(ValidationError, match="data_parallel_size should match"):
        ParallelArguments(data_parallel_size=2)


def test_tools_arguments_accept_declared_nested_options():
    config = ToolsArguments(
        profile={"enable": True, "ranks": [0], "static_param": {"start_step": 2}},
        tensorboard={"enable": True, "dir": "/tmp/tensorboard"},
    )
    assert config.profile.enable is True
    assert config.profile.static_param.start_step == 2
    assert config.tensorboard.dir == "/tmp/tensorboard"


def test_feature_arguments_keep_enabled_chunk_loss_options():
    config = FeatureArguments(
        enable_chunk_loss=True,
        chunkloss_plan={"impl_type": "legacy", "chunk_size": 512},
    )
    assert config.chunkloss_plan.chunk_size == 512
    assert config.chunkloss_plan.vocab_tile_size is None


@pytest.mark.parametrize(("accumulation", "expected_batch_size"), [(None, 8), (3, 24)])
def test_training_arguments_compute_global_batch_size(accumulation, expected_batch_size):
    config = TrainingArguments(micro_batch_size=2, gradient_accumulation_steps=accumulation)
    config.compute_distributed_training(SimpleNamespace(data_parallel_size=4))

    assert config.global_batch_size == expected_batch_size
    assert config.gradient_accumulation_steps == (accumulation or 1)


def test_config_manager_registers_multi_level_fields():
    class Config(BaseArguments):
        pass

    manager = ConfigManager(config_class=Config, allow_cli_override=False)
    manager.register_dynamic_field("encoder.attention.num_heads", int, 8)
    config = manager.load_and_parse()

    assert config.encoder.attention.num_heads == 8
    assert "encoder.attention.num_heads" in manager.get_defined_fields()
    assert manager.get_config() is config


def test_config_manager_loads_nested_yaml_extensions(tmp_path):
    class Config(BaseArguments):
        pass

    path = tmp_path / "config.yaml"
    path.write_text("encoder:\n  attention:\n    num_heads: 8\n  hidden_size: 128\n")
    manager = ConfigManager(config_class=Config, config_file_path=str(path), allow_cli_override=False)
    config = manager.load_and_parse()

    assert config.encoder.attention.num_heads == 8
    assert config.encoder.hidden_size == 128


def test_yaml_collate_extensions_support_attribute_access(tmp_path):
    class Data(BaseArguments):
        pass

    class Config(BaseArguments):
        data: Data = field(default_factory=Data)

    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n"
        "  dataloader_param:\n"
        "    collate_param:\n"
        "      model_name: qwen3vl\n"
        "      ignore_pad_token_for_loss: true\n"
    )
    # YAML fields are registered automatically; no manual registration is needed.
    config = ConfigManager(
        config_class=Config, config_file_path=str(path), allow_cli_override=False,
    ).load_and_parse()

    assert config.data.dataloader_param.collate_param.model_name == "qwen3vl"
    assert config.data.dataloader_param.collate_param.ignore_pad_token_for_loss is True
    assert config.to_dict() == yaml.safe_load(path.read_text())


def test_yaml_extends_existing_fsdp_collate_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "16")
    example = Path(__file__).resolve().parents[3] / "examples/qwen3_5/qwen3_5_35B_config.yaml"
    payload = yaml.safe_load(example.read_text())
    payload["data"]["dataloader_param"]["collate_param"].update({
        "model_name": "qwen3vl",
        "ignore_pad_token_for_loss": True,
        "processor_options": {"image": {"max_pixels": 1024}},
    })
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    config = ConfigManager(
        config_class=Arguments, config_file_path=str(path), allow_cli_override=False,
    ).load_and_parse()

    assert config.data.dataloader_param.collate_param.model_name == "qwen3vl"
    assert config.data.dataloader_param.collate_param.ignore_pad_token_for_loss is True
    assert config.data.dataloader_param.collate_param.processor_options.image.max_pixels == 1024


def test_config_manager_prioritizes_cli_then_code_then_yaml(tmp_path, monkeypatch):
    class Config(BaseArguments):
        learning_rate: float = 0.1
        batch_size: int = 1
        seed: int = 0

    path = tmp_path / "config.yaml"
    path.write_text("learning_rate: 0.01\nbatch_size: 2\nseed: 42\n")
    monkeypatch.setattr(sys, "argv", ["train.py", str(path), "learning_rate=0.001"])
    manager = ConfigManager(
        config_class=Config, config_file_path=str(path),
        additional_args={"learning_rate": 0.005, "batch_size": 4},
    )
    config = manager.load_and_parse()

    assert config.learning_rate == 0.001
    assert config.batch_size == 4
    assert config.seed == 42


def test_nested_dynamic_configuration_exports_plain_dictionaries():
    class Config(BaseArguments):
        pass

    Config.register_field("encoder.attention.num_heads", int, 8)
    config = Config(encoder={"attention": {"num_heads": 16}})
    expected = {"encoder": {"attention": {"num_heads": 16}}}
    assert config.to_dict() == expected
    assert Config(**config.to_dict()).to_dict() == expected


@pytest.mark.parametrize("invalid", [123, "invalid", []])
def test_nested_dynamic_configuration_rejects_non_config_values(invalid):
    class Config(BaseArguments):
        pass

    Config.register_field("encoder.hidden_size", int, 128)
    with pytest.raises(TypeError, match="encoder"):
        Config(encoder=invalid)
    config = Config()
    with pytest.raises(TypeError, match="encoder"):
        config.encoder = invalid
    assert config.encoder.hidden_size == 128


def test_nested_dictionary_assignment_keeps_attribute_access():
    class Config(BaseArguments):
        pass

    Config.register_field("encoder.hidden_size", int, 128)
    config = Config()
    config.encoder = {"hidden_size": 256}
    assert config.encoder.hidden_size == 256


def test_root_registration_extends_existing_and_new_sections():
    class Model(BaseArguments):
        pass

    class Config(BaseArguments):
        model: Model = field(default_factory=Model)

    manager = ConfigManager(config_class=Config, allow_cli_override=False)
    manager.register_dynamic_field("model.encoder.hidden_size", int, 128)
    manager.register_dynamic_field("custom.encoder.hidden_size", int, 256)
    config = manager.load_and_parse()
    assert config.model.encoder.hidden_size == 128
    assert config.custom.encoder.hidden_size == 256


@pytest.mark.parametrize("filename", [
    "examples/qwen3_5/qwen3_5_35B_config.yaml",
    "examples/ltx2/ltx2_config_t2v.yaml",
    "examples/ltx2/ltx2_config_t2av.yaml",
    "examples/magihuman/magihuman_config_t2av.yaml",
])
def test_examples_load_with_nested_extensions(monkeypatch, filename):
    monkeypatch.setenv("WORLD_SIZE", "16")
    path = Path(__file__).resolve().parents[3] / filename
    config = ConfigManager(
        config_class=Arguments, config_file_path=str(path), allow_cli_override=False,
    ).load_and_parse()
    assert all(getattr(config, section) is not None for section in Arguments.model_fields)
    if "qwen3_5" in filename:
        assert config.features.recompute is True
    else:
        custom_key = "magihuman_dataset_custom" if "magihuman" in filename else "ltx2_dataset_custom"
        custom = config.data.dataset_param.to_dict()[custom_key]
        assert isinstance(custom, dict)
        assert isinstance(custom["timestep_sampling_params"], dict)
        assert custom["with_audio"] is filename.endswith("t2av.yaml")


def test_yaml_accepts_extra_root_and_nested_sections(tmp_path):
    class Config(BaseArguments):
        features: FeatureArguments = field(default_factory=FeatureArguments)

    path = tmp_path / "config.yaml"
    path.write_text(
        "custom:\n  encoder:\n    hidden_size: 128\n"
        "features:\n  model_specific_option:\n    value: 1\n"
    )
    config = ConfigManager(
        config_class=Config, config_file_path=str(path), allow_cli_override=False,
    ).load_and_parse()
    assert config.custom.encoder.hidden_size == 128
    assert config.features.model_specific_option.value == 1
