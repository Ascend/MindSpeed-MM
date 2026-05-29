# coding=utf-8
import os
import torch
import torch.nn as nn
import pytest


from mindspeed_mm.fsdp.models.mimov2_5.modeling_mimo_v2 import PatchMiMoV2TopkRouter, PatchMiMoV2NaiveMoe, PatchMiMoV2MoE
from mindspeed_mm.fsdp.models.mimov2_5.configuration_mimo_v2 import MiMoV2Config
from tests.ut_fsdp.utils.utils import judge_expression


class TestPatchMiMoV2TopkRouter:

    def test_init(self):
        config = MiMoV2Config(
            hidden_size=4096,
            n_routed_experts=8
        )
        router = PatchMiMoV2TopkRouter(config)
        judge_expression(router.n_routed_experts == 8)
        judge_expression(router.weight.shape == (8, 4096))
        judge_expression(router.e_score_correction_bias.shape == (8,))

    def test_forward(self):
        config = MiMoV2Config(
            hidden_size=4096,
            n_routed_experts=4
        )
        router = PatchMiMoV2TopkRouter(config)
        hidden_states = torch.randn(2, 10, 4096)
        router_logits = router(hidden_states)
        judge_expression(router_logits.shape == (20, 4))
        judge_expression(router_logits.dtype == torch.float32)

    def test_forward_with_different_batch_size(self):
        config = MiMoV2Config(
            hidden_size=2048,
            n_routed_experts=6
        )
        router = PatchMiMoV2TopkRouter(config)
        hidden_states = torch.randn(5, 15, 2048)
        router_logits = router(hidden_states)
        judge_expression(router_logits.shape == (75, 6))


class TestPatchMiMoV2NaiveMoe:

    def test_init(self):
        config = MiMoV2Config(
            hidden_size=4096,
            n_routed_experts=4,
            moe_intermediate_size=1024,
            hidden_act="silu"
        )
        moe = PatchMiMoV2NaiveMoe(config)
        judge_expression(moe.num_experts == 4)
        judge_expression(moe.hidden_size == 4096)
        judge_expression(moe.intermediate_dim == 1024)
        judge_expression(moe.gate_up_proj.shape == (4, 4096, 2048))
        judge_expression(moe.down_proj.shape == (4, 1024, 4096))

    def test_forward(self):
        config = MiMoV2Config(
            hidden_size=512,
            n_routed_experts=4,
            moe_intermediate_size=256,
            hidden_act="silu"
        )
        moe = PatchMiMoV2NaiveMoe(config)
        hidden_states = torch.randn(10, 512)
        top_k_weights = torch.rand(10, 2)
        top_k_index = torch.randint(0, 4, (10, 2))
        output = moe(hidden_states, top_k_weights, top_k_index)
        judge_expression(output.shape == (10, 512))
        judge_expression(output.dtype == hidden_states.dtype)

    def test_forward_with_single_expert(self):
        config = MiMoV2Config(
            hidden_size=256,
            n_routed_experts=2,
            moe_intermediate_size=128,
            hidden_act="silu"
        )
        moe = PatchMiMoV2NaiveMoe(config)
        hidden_states = torch.randn(5, 256)
        top_k_weights = torch.rand(5, 1)
        top_k_index = torch.zeros(5, 1, dtype=torch.long)
        output = moe(hidden_states, top_k_weights, top_k_index)
        judge_expression(output.shape == (5, 256))

    def test_forward_with_all_experts(self):
        config = MiMoV2Config(
            hidden_size=128,
            n_routed_experts=3,
            moe_intermediate_size=64,
            hidden_act="silu"
        )
        moe = PatchMiMoV2NaiveMoe(config)
        hidden_states = torch.randn(6, 128)
        top_k_weights = torch.ones(6, 3) / 3
        top_k_index = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 0, 1]])
        output = moe(hidden_states, top_k_weights, top_k_index)
        judge_expression(output.shape == (6, 128))


class TestPatchMiMoV2MoE:

    def test_init(self):
        config = MiMoV2Config(
            hidden_size=4096,
            n_routed_experts=8,
            moe_intermediate_size=1024,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            hidden_act="silu"
        )
        moe = PatchMiMoV2MoE(config)
        judge_expression(moe.n_routed_experts == 8)
        judge_expression(moe.n_group == 2)
        judge_expression(moe.topk_group == 1)
        judge_expression(moe.norm_topk_prob == True)
        judge_expression(moe.top_k == 2)

    def test_route_tokens_to_experts(self):
        config = MiMoV2Config(
            hidden_size=512,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0
        )
        moe = PatchMiMoV2MoE(config)
        router_logits = torch.randn(10, 4)
        topk_indices, topk_weights = moe.route_tokens_to_experts(router_logits)
        judge_expression(topk_indices.shape == (10, 2))
        judge_expression(topk_weights.shape == (10, 2))
        judge_expression((topk_weights >= 0).all())

    def test_route_tokens_to_experts_with_norm(self):
        config = MiMoV2Config(
            hidden_size=256,
            n_routed_experts=6,
            num_experts_per_tok=3,
            n_group=3,
            topk_group=2,
            norm_topk_prob=True,
            routed_scaling_factor=2.0
        )
        moe = PatchMiMoV2MoE(config)
        router_logits = torch.randn(5, 6)
        topk_indices, topk_weights = moe.route_tokens_to_experts(router_logits)
        judge_expression(topk_indices.shape == (5, 3))
        judge_expression(topk_weights.shape == (5, 3))


    def test_route_tokens_to_experts_without_norm(self):
        config = MiMoV2Config(
            hidden_size=256,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            norm_topk_prob=False,
            routed_scaling_factor=1.0
        )
        moe = PatchMiMoV2MoE(config)
        router_logits = torch.randn(8, 4)
        topk_indices, topk_weights = moe.route_tokens_to_experts(router_logits)
        judge_expression(topk_indices.shape == (8, 2))
        judge_expression(topk_weights.shape == (8, 2))
        judge_expression((topk_weights >= 0).all())

    def test_forward(self):
        config = MiMoV2Config(
            hidden_size=512,
            n_routed_experts=4,
            moe_intermediate_size=256,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            hidden_act="silu"
        )
        moe = PatchMiMoV2MoE(config)
        hidden_states = torch.randn(2, 10, 512)
        output = moe(hidden_states)
        judge_expression(output.shape == (2, 10, 512))

    def test_forward_with_different_config(self):
        config = MiMoV2Config(
            hidden_size=1024,
            n_routed_experts=8,
            moe_intermediate_size=512,
            num_experts_per_tok=3,
            n_group=4,
            topk_group=2,
            norm_topk_prob=True,
            routed_scaling_factor=0.5,
            hidden_act="silu"
        )
        moe = PatchMiMoV2MoE(config)
        hidden_states = torch.randn(3, 15, 1024)
        output = moe(hidden_states)
        judge_expression(output.shape == (3, 15, 1024))

    def test_forward_with_single_batch(self):
        config = MiMoV2Config(
            hidden_size=256,
            n_routed_experts=4,
            moe_intermediate_size=128,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            hidden_act="silu"
        )
        moe = PatchMiMoV2MoE(config)
        hidden_states = torch.randn(1, 5, 256)
        output = moe(hidden_states)
        judge_expression(output.shape == (1, 5, 256))

    def test_forward_with_scaling_factor(self):
        config = MiMoV2Config(
            hidden_size=512,
            n_routed_experts=4,
            moe_intermediate_size=256,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=2.5,
            hidden_act="silu"
        )
        moe = PatchMiMoV2MoE(config)
        hidden_states = torch.randn(2, 8, 512)
        output = moe(hidden_states)
        judge_expression(output.shape == (2, 8, 512))
