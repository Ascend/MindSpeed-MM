from unittest.mock import MagicMock, patch, PropertyMock
from mindspeed_mm.fsdp.tools.flops_tool.flops_qwen3_5 import Qwen35FlopsCounter


class TestQwen35FlopsCounter:

    @patch("transformers.AutoConfig")
    def setup_method(self, method, mock_autoconfig):
        self.mock_text_config = MagicMock()
        self.mock_text_config.hidden_size = 1024
        self.mock_text_config.vocab_size = 10000
        self.mock_text_config.num_hidden_layers = 2
        self.mock_text_config.num_attention_heads = 16
        self.mock_text_config.num_key_value_heads = 16
        self.mock_text_config.full_attention_interval = 2
        self.mock_text_config.intermediate_size = 4096

        self.mock_text_config.linear_num_key_heads = 8
        self.mock_text_config.linear_key_head_dim = 64
        self.mock_text_config.linear_num_value_heads = 8
        self.mock_text_config.linear_value_head_dim = 64
        self.mock_text_config.linear_conv_kernel_dim = 16

        self.mock_vision_config = MagicMock()
        self.mock_vision_config.num_heads = 8
        self.mock_vision_config.depth = 2
        self.mock_vision_config.hidden_size = 512
        self.mock_vision_config.intermediate_size = 2048
        self.mock_vision_config.out_hidden_size = 512
        self.mock_vision_config.spatial_merge_size = 2
        self.mock_vision_config.in_channels = 3
        self.mock_vision_config.temporal_patch_size = 2
        self.mock_vision_config.patch_size = 14

        mock_config = MagicMock()
        mock_config.text_config = self.mock_text_config
        mock_config.vision_config = self.mock_vision_config

        mock_autoconfig.from_pretrained.return_value = mock_config

        self.counter = Qwen35FlopsCounter(config=mock_config)

    def test_estimate_flops_text_only_dense(self, mocker):
        """
        Test Scenario: Text-only input, Dense architecture.
        Objective: Verify core FLOPs calculation flow.
        """
        mock_estimate_family = mocker.patch.object(
            self.counter,
            "_estimate_qwen3_5_family_flops",
        )

        batch_seqlens = [128, 128]
        step_time = 6.9

        result = self.counter.estimate_flops(batch_seqlens=batch_seqlens, step_time=step_time)

        mock_estimate_family.assert_called_once()

    def test_estimate_flops_text_only_moe(self, mocker):
        """
        Test Scenario: Text-only input, MoE architecture.
        Objective: Verify MoE branch logic is triggered.
        """
        type(self.mock_text_config).num_experts = PropertyMock(return_value=8)
        self.mock_text_config.num_experts_per_tok = 2
        self.mock_text_config.moe_intermediate_size = 1024
        self.mock_text_config.shared_expert_intermediate_size = 512

        mock_estimate_family = mocker.patch.object(
            self.counter,
            "_estimate_qwen3_5_family_flops",
        )

        batch_seqlens = [64]
        step_time = 6.9

        result = self.counter.estimate_flops(batch_seqlens=batch_seqlens, step_time=step_time)

        mock_estimate_family.assert_called_once()

    def test_estimate_flops_multimodal_with_vit(self, mocker):
        """
        Test Scenario: Multimodal input (containing images).
        Objective: Verify ViT calculation logic is invoked when images_seqlens is passed.
        """
        mock_estimate_family = mocker.patch.object(
            self.counter,
            "_estimate_qwen3_5_family_flops",
        )

        mock_estimate_vit = mocker.patch.object(
            self.counter,
            "_estimate_qwen3_vit_flop",
        )

        batch_seqlens = [256]
        images_seqlens = [100]
        step_time = 6.9

        result = self.counter.estimate_flops(
            batch_seqlens=batch_seqlens, images_seqlens=images_seqlens, step_time=step_time
        )

        mock_estimate_family.assert_called_once()

        _, kwargs = mock_estimate_family.call_args
        assert "images_seqlens" in kwargs
        assert kwargs["images_seqlens"] == images_seqlens
