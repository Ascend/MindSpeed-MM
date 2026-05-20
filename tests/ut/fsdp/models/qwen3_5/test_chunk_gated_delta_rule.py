# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
import torch.nn.functional as F
import pytest

from mindspeed_mm.fsdp.ops.gdn.chunk_gated_delta_rule import chunk_gated_delta_rule as triton_chunk_gated_delta_rule
from tests.ut.utils import judge_expression


def varlen_to_nonvarlen(cu_seqlens, *inputs):
    B = len(cu_seqlens) - 1  # batch size
    max_len = max(cu_seqlens[i + 1] - cu_seqlens[i] for i in range(B))  # 最长序列长度
    
    nonvarlens = [torch.zeros(B, max_len, *v.shape[2:], device=v.device, dtype=v.dtype) for v in inputs]
    
    for i in range(B):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        seq_len = end - start
        if seq_len > 0:
            for nonvarlen, var in zip(nonvarlens, inputs):
                nonvarlen[i, :seq_len] = var[0, start:end]
    
    return nonvarlens


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


TEST_PARAM_NAMES = ('B', 'T', 'H', 'K', 'V', 'chunk_size', 'use_qk_l2norm_in_kernel', 'cu_seqlens')

TEST_CASES = [
    (1, 128, 4, 64, 64, 64, False, None),
    (2, 256, 4, 64, 64, 64, False, None),
    (1, 128, 8, 128, 128, 64, False, None),
    (1, 128, 8, 128, 128, 64, True, None),
    (1, 360, 32, 128, 128, 64, True, None),
    (1, 1024, 64, 128, 128, 64, True, None),
    (1, 1024, 32, 128, 128, 64, True, [0, 112, 209, 240, 281, 489, 523, 566, 689, 721, 785, 837, 985] + [i for i in range(986, 1025)]),
    (1, 1121, 2, 128, 128, 64, True, [0, 112, 209, 240, 281, 489, 523, 566, 689, 721, 785, 837, 985, 1071, 1121]),
    (1, 1121, 4, 128, 128, 64, True, [0, 112, 209, 240, 281, 489, 523, 566, 689, 721, 785, 837, 985, 1071, 1121]),
    (1, 1121, 8, 128, 128, 64, True, [0, 112, 209, 240, 281, 489, 523, 566, 689, 721, 785, 837, 985, 1071, 1121]),
    (1, 1121, 16, 128, 128, 64, True, [0, 112, 209, 240, 281, 489, 523, 566, 689, 721, 785, 837, 985, 1071, 1121]),
    (1, 1121, 32, 128, 128, 64, True, [0, 112, 209, 240, 281, 489, 523, 566, 689, 721, 785, 837, 985, 1071, 1121]),
    (1, 1121, 64, 128, 128, 64, True, [0, 112, 209, 240, 281, 489, 523, 566, 689, 721, 785, 837, 985, 1071, 1121]),
    (21, 195, 32, 128, 128, 64, True, None),
    (1, 160, 32, 128, 128, 64, True, None),
    (1, 4608, 2, 128, 128, 64, True, None),
    (1, 4608, 4, 128, 128, 64, True, None),
    (1, 4608, 8, 128, 128, 64, True, None),
    (1, 4608, 16, 128, 128, 64, True, None),
    (1, 4608, 32, 128, 128, 64, True, None),
    (1, 4608, 64, 128, 128, 64, True, None),
    (1, 808, 32, 128, 128, 64, True, [0] + [i for i in range(804, 809)]),
    (1, 32832, 32, 128, 128, 64, True, [0, 14012, 26600, 32766, 32832]),
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
    def test_chunk_gated_delta_rule_forward_backward(self, B, T, H, K, V, chunk_size, use_qk_l2norm_in_kernel, cu_seqlens):
        torch.manual_seed(42)
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

        if cu_seqlens is not None:
            cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
        o_triton, _ = triton_chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens
        )
        do = torch.randn_like(o_triton)
        o_triton.backward(do)

        # ==== torch ====
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        beta_ref = beta.detach().clone().requires_grad_(True)
        g_ref = g.detach().clone().requires_grad_(True)
        if cu_seqlens is not None:
            q_ref, k_ref, v_ref, beta_ref, g_ref = varlen_to_nonvarlen(cu_seqlens, q_ref, k_ref, v_ref, beta_ref, g_ref)
        q_ref.retain_grad()
        k_ref.retain_grad()
        v_ref.retain_grad()
        beta_ref.retain_grad()
        g_ref.retain_grad()
        o_ref, _ = ref_torch_chunk_gated_delta_rule(
            query=q_ref, key=k_ref, value=v_ref, g=g_ref, beta=beta_ref, chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
        )
        if cu_seqlens is not None:
            do_ref = varlen_to_nonvarlen(cu_seqlens, do)
        else:
            do_ref = do
        o_ref.backward(do_ref)

        # ==== bsnd triton ====
        if cu_seqlens is not None:
            q_tt_sbh = q_ref.detach().clone().requires_grad_(True)
            k_tt_sbh = k_ref.detach().clone().requires_grad_(True)
            v_tt_sbh = v_ref.detach().clone().requires_grad_(True)
            beta_tt_sbh = beta_ref.detach().clone().requires_grad_(True)
            g_tt_sbh = g_ref.detach().clone().requires_grad_(True)

            o_tt_sbh, _ = triton_chunk_gated_delta_rule(
                q=q_tt_sbh, k=k_tt_sbh, v=v_tt_sbh, g=g_tt_sbh, beta=beta_tt_sbh, chunk_size=chunk_size,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
            )
            o_tt_sbh.backward(do_ref)

        if cu_seqlens is not None:
            o_tt, q_g, k_g, v_g, beta_g, g_g = varlen_to_nonvarlen(cu_seqlens, o_triton, q.grad, k.grad, v.grad, beta.grad, g.grad)
        else:
            o_tt, q_g, k_g, v_g, beta_g, g_g = o_triton, q.grad, k.grad, v.grad, beta.grad, g.grad

        # ==== accuracy ====
        if cu_seqlens is not None:
            torch.testing.assert_close(o_tt, o_tt_sbh, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(q_g, q_tt_sbh.grad, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(k_g, k_tt_sbh.grad, rtol=0.01, atol=0.01)
            torch.testing.assert_close(v_g, v_tt_sbh.grad, rtol=0.01, atol=0.01)
            torch.testing.assert_close(beta_g, beta_tt_sbh.grad, rtol=0.01, atol=0.01)
            torch.testing.assert_close(g_g, g_tt_sbh.grad, rtol=0.01, atol=0.01)
        forward_close = torch.allclose(o_tt.float(), o_ref.float(), rtol=1e-2, atol=1e-2)
        judge_expression(forward_close)
        backward_close = torch.allclose(q_g.float(), q_ref.grad.float(), rtol=1e-2, atol=1e-2)
        judge_expression(backward_close)
        backward_close = torch.allclose(k_g.float(), k_ref.grad.float(), rtol=1e-2, atol=1e-2)
        judge_expression(backward_close)
        backward_close = torch.allclose(v_g.float(), v_ref.grad.float(), rtol=1e-2, atol=1e-2)
        judge_expression(backward_close)
        backward_close = torch.allclose(beta_g.float(), beta_ref.grad.float(), rtol=1e-2, atol=1e-2)
        judge_expression(backward_close)
        backward_close = torch.allclose(g_g.float(), g_ref.grad.float(), rtol=1e-2, atol=1e-2)
        judge_expression(backward_close)

