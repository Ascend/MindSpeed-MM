import os
import sys
import unittest
import torch


# Use the native aux_loss calculation from transformers as a reference.
# The calculation formula in the qwen3vl model is the same as the aux_loss implementation in the qwen3.5 model.
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import load_balancing_loss_func
from mindspeed_mm.utils.aux_loss import load_balancing_loss_func_optimized


class TestLoadBalancingLossConsistency(unittest.TestCase):
    """Test the numerical consistency between load_balancing_loss_func and its optimized version."""

    def test_none_input(self):
        """Test with None input."""
        original_result = load_balancing_loss_func(None, num_experts=8, top_k=2)
        optimized_result = load_balancing_loss_func_optimized(None, num_experts=8, top_k=2)
        self.assertEqual(original_result, 0)
        self.assertEqual(optimized_result, 0)

    def test_single_layer_no_mask_bf16(self):
        """Test single layer without attention_mask (bf16)."""
        batch_size, seq_len, num_experts = 2, 4, 8
        top_k = 2

        gate_logits = torch.randn(batch_size * seq_len, num_experts, dtype=torch.bfloat16)
        gate_logits_tuple = (gate_logits.float(),)

        original_loss = load_balancing_loss_func(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=None
        )

        optimized_loss = load_balancing_loss_func_optimized(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=None
        )

        self.assertTrue(torch.allclose(original_loss, optimized_loss, rtol=1e-3, atol=1e-4))

    def test_multi_layer_no_mask_bf16(self):
        """Test multiple layers without attention_mask (bf16)."""
        batch_size, seq_len, num_experts = 2, 4, 8
        num_layers = 3
        top_k = 2

        gate_logits_list = []
        for _ in range(num_layers):
            gate_logits_layer = torch.randn(batch_size * seq_len, num_experts, dtype=torch.bfloat16)
            gate_logits_list.append(gate_logits_layer.float())

        gate_logits_tuple = tuple(gate_logits_list)

        original_loss = load_balancing_loss_func(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=None
        )

        optimized_loss = load_balancing_loss_func_optimized(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=None
        )

        self.assertTrue(torch.allclose(original_loss, optimized_loss, rtol=1e-3, atol=1e-4))

    def test_single_layer_with_mask_bf16(self):
        """Test single layer with attention_mask (bf16)."""
        batch_size, seq_len, num_experts = 3, 5, 6
        top_k = 2

        gate_logits = torch.randn(batch_size * seq_len, num_experts, dtype=torch.bfloat16)
        gate_logits_tuple = (gate_logits.float(),)

        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
        attention_mask[0, 2:] = 0
        attention_mask[1, 3:] = 0

        original_loss = load_balancing_loss_func(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attention_mask
        )

        optimized_loss = load_balancing_loss_func_optimized(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attention_mask
        )

        self.assertTrue(torch.allclose(original_loss, optimized_loss, rtol=1e-3, atol=1e-4))

    def test_multi_layer_with_mask_bf16(self):
        """Test multiple layers with attention_mask (bf16)."""
        batch_size, seq_len, num_experts = 1, 5, 16
        num_layers = 4
        top_k = 2
        gate_logits_list = []
        for _ in range(num_layers):
            gate_logits_layer = torch.randn(batch_size * seq_len, num_experts, dtype=torch.bfloat16)
            gate_logits_list.append(gate_logits_layer.float())

        gate_logits_tuple = tuple(gate_logits_list)

        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
        attention_mask[0, 2:] = 0

        original_loss = load_balancing_loss_func(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attention_mask
        )

        optimized_loss = load_balancing_loss_func_optimized(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attention_mask
        )

        self.assertTrue(torch.allclose(original_loss, optimized_loss, rtol=1e-3, atol=1e-4))

    def test_multi_layer_with_mask_batchsize_more_than_1_bf16(self):
        """Test multiple layers with attention_mask (bf16) with batch size > 1."""
        batch_size, seq_len, num_experts = 4, 8, 16
        num_layers = 4
        top_k = 2
        gate_logits_list = []
        for _ in range(num_layers):
            gate_logits_layer = torch.randn(batch_size * seq_len, num_experts, dtype=torch.bfloat16)
            gate_logits_list.append(gate_logits_layer.float())

        gate_logits_tuple = tuple(gate_logits_list)

        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
        attention_mask[0, 2:] = 0
        attention_mask[1, 3:] = 0
        attention_mask[2, 1:] = 0
        attention_mask[3, 4:] = 0

        original_loss = load_balancing_loss_func(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attention_mask
        )

        optimized_loss = load_balancing_loss_func_optimized(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attention_mask
        )

        self.assertTrue(torch.allclose(original_loss, optimized_loss, rtol=1e-3, atol=1e-4))

    def test_multi_layer_with_mask_fp32(self):
        """Test multiple layers with attention_mask (fp32)."""
        batch_size, seq_len, num_experts = 2, 4, 16
        num_layers = 4
        top_k = 2
        gate_logits_list = []
        for _ in range(num_layers):
            gate_logits_layer = torch.randn(batch_size * seq_len, num_experts, dtype=torch.bfloat16)
            gate_logits_list.append(gate_logits_layer.float())

        gate_logits_tuple = tuple(gate_logits_list)

        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
        attention_mask[0, 2:] = 0
        attention_mask[1, 3:] = 0

        original_loss = load_balancing_loss_func(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attention_mask
        )

        optimized_loss = load_balancing_loss_func_optimized(
            gate_logits_tuple,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attention_mask
        )

        self.assertTrue(torch.allclose(original_loss, optimized_loss, rtol=1e-3, atol=1e-4))


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    unittest.main(testRunner=runner)
