import re
import pytest
from checkpoint.vlm_model.converters.qwen3_5 import Qwen35Converter


class TestQwen35MtpExpertKeyDetection:
    @pytest.mark.parametrize(
        "key",
        [
            "mtp.layers.0.mlp.experts.0.gate_proj.weight",
            "mtp.layers.0.mlp.experts.0.up_proj.weight",
            "mtp.layers.0.mlp.experts.1.down_proj.weight",
        ],
    )
    def test_mtp_expert_gate_proj_is_collected(self, key):
        assert Qwen35Converter.is_mtp_expert_key(key) is True

    @pytest.mark.parametrize(
        "key",
        [
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        ],
    )
    def test_mtp_shared_expert_is_not_collected(self, key):
        assert not Qwen35Converter.is_mtp_expert_key(key)

    @pytest.mark.parametrize(
        "key",
        [
            "model.language_model.layers.4.mlp.experts.gate_up_proj",
            "model.language_model.layers.4.mlp.experts.down_proj",
            "model.language_model.layers.42.mlp.gate.weight",
        ],
    )
    def test_non_mtp_key_is_not_collected(self, key):
        assert not Qwen35Converter.is_mtp_expert_key(key)


class TestQwen35ExpertWeightNamePatterns:
    _patterns = Qwen35Converter.expert_weight_name_patterns

    @pytest.mark.parametrize(
        "key",
        [
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "model.language_model.layers.5.mlp.experts.down_proj",
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.1.mlp.experts.down_proj",
        ],
    )
    def test_pattern_match(self, key):
        matched = any(re.fullmatch(pat, key) for pat in self._patterns)
        assert matched is True

    @pytest.mark.parametrize(
        "key",
        [
            "mtp.layers.0.mlp.experts.0.gate_proj.weight",
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        ],
    )
    def test_pattern_does_not_match(self, key):
        matched = any(re.fullmatch(pat, key) for pat in self._patterns)
        assert matched is False
