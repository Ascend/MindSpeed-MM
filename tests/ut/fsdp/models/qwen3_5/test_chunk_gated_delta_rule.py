# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
import torch.nn.functional as F
import pytest

from mindspeed_mm.fsdp.models.qwen3_5.chunk_gated_delta_rule import chunk_gated_delta_rule as triton_chunk_gated_delta_rule
from tests.ut.utils import judge_expression


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def ref_torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


TEST_PARAM_NAMES = ('B', 'T', 'H', 'K', 'V', 'chunk_size', 'use_qk_l2norm_in_kernel')

TEST_CASES = [
    (1, 128, 4, 64, 64, 64, False),
    (2, 256, 4, 64, 64, 64, False),
    (1, 128, 8, 128, 128, 64, False),
    (1, 128, 8, 128, 128, 64, True),
    (21, 195, 32, 128, 128, 64, True),
    (1, 360, 32, 128, 128, 64, True),
]


def _make_test_params():
    params = []
    for test in TEST_CASES:
        test_id = '-'.join(f'{name}{value}' for name, value in zip(TEST_PARAM_NAMES, test))
        params.append(pytest.param(*test, id=test_id))
    return params


class TestChunkGatedDeltaRule:
    """Test chunk_gated_delta_rule API forward and backward precision."""

    @pytest.mark.parametrize(
        TEST_PARAM_NAMES,
        _make_test_params()
    )
    def test_chunk_gated_delta_rule_forward_backward(self, B, T, H, K, V, chunk_size, use_qk_l2norm_in_kernel):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.npu.is_available():
            device = torch.device("npu")
        else:
            pytest.skip("CUDA or NPU is not available")
        dtype = torch.bfloat16

        q = torch.randn(B, T, H, K, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(B, T, H, K, dtype=dtype, device=device, requires_grad=True)
        if not use_qk_l2norm_in_kernel:
            q = l2norm(q, dim=-1, eps=1e-6).detach().clone().requires_grad_(True)
            k = l2norm(k, dim=-1, eps=1e-6).detach().clone().requires_grad_(True)
        v = torch.randn(B, T, H, V, dtype=dtype, device=device, requires_grad=True)
        beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
        g = F.logsigmoid(torch.rand(B, T, H, dtype=dtype, device=device)).requires_grad_(True)

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        beta_ref = beta.detach().clone().requires_grad_(True)
        g_ref = g.detach().clone().requires_grad_(True)

        o_triton, _ = triton_chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
        )
        o_ref, _ = ref_torch_chunk_gated_delta_rule(
            query=q_ref, key=k_ref, value=v_ref, g=g_ref, beta=beta_ref, chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
        )

        forward_close = torch.allclose(o_triton.float(), o_ref.float(), rtol=1e-2, atol=1e-2)
        judge_expression(forward_close)

        do = torch.randn_like(o_triton)
        o_triton.backward(do)
        o_ref.backward(do)

        backward_close = torch.allclose(q.grad.float(), q_ref.grad.float(), rtol=1e-2, atol=1e-2)
        judge_expression(backward_close)
        backward_close = torch.allclose(k.grad.float(), k_ref.grad.float(), rtol=1e-2, atol=1e-2)
        judge_expression(backward_close)
        backward_close = torch.allclose(v.grad.float(), v_ref.grad.float(), rtol=1e-2, atol=1e-2)
        judge_expression(backward_close)
        backward_close = torch.allclose(beta.grad.float(), beta_ref.grad.float(), rtol=1e-2, atol=1e-2)
        judge_expression(backward_close)
        backward_close = torch.allclose(g.grad.float(), g_ref.grad.float(), rtol=1e-2, atol=1e-2)
        judge_expression(backward_close)

