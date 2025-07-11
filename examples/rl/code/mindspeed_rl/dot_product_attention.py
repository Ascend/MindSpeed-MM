# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math
from functools import wraps

import torch
import torch.nn.functional as F
import torch_npu
from torch import Tensor
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from megatron.training import get_args
from megatron.core import mpu, parallel_state
from mindspeed.core.models.common.embeddings.rotary_pos_embedding import yarn_get_mscale
from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.model.transformer import get_attention_mask
from mindspeed.utils import get_actual_seq_len
from mindspeed.core.context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
from mindspeed.core.context_parallel.utils import get_scheduling_info

try:
    from einops import rearrange, repeat
except ImportError:
    rearrange = None


def dot_product_attention_init(
    self,
    config: TransformerConfig,
    layer_number: int,
    attn_mask_type: AttnMaskType,
    attention_type: str,
    attention_dropout: float = None,
):
    cp_size = config.context_parallel_size
    config.context_parallel_size = 1

    super(DotProductAttention, self).__init__(config=config)
    if self.config.context_parallel_size != 1:
        raise ValueError("Context parallelism is only supported by TEDotProductAttention!")

    if self.config.window_size is not None:
        raise ValueError("Sliding Window Attention is only supported by TEDotProductAttention!")

    self.layer_number = max(1, layer_number)
    self.attn_mask_type = attn_mask_type
    self.attention_type = attention_type  # unused for now

    projection_size = self.config.kv_channels * self.config.num_attention_heads
    args = get_args()
    # Per attention head and per partition values.
    world_size = args.tp_x if args.tp_2d else parallel_state.get_tensor_model_parallel_world_size()
    self.hidden_size_per_partition = divide(projection_size, world_size)
    self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
    self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
    self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

    coeff = None
    self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
    if self.config.apply_query_key_layer_scaling:
        coeff = self.layer_number
        self.norm_factor *= coeff

    self.scale_mask_softmax = FusedScaleMaskSoftmax(
        input_in_fp16=self.config.fp16,
        input_in_bf16=self.config.bf16,
        attn_mask_type=self.attn_mask_type,
        scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
        mask_func=attention_mask_func,
        softmax_in_fp32=self.config.attention_softmax_in_fp32,
        scale=coeff,
    )

    # Dropout. Note that for a single iteration, this layer will generate
    # different outputs on different number of parallel partitions but
    # on average it should not be partition dependent.
    self.attention_dropout = torch.nn.Dropout(
        self.config.attention_dropout if attention_dropout is None else attention_dropout
    )

    config.context_parallel_size = cp_size

    # add pse
    self.pse = None
    self.pse_type = args.alibi_fusion_attn_type

    if args.multi_head_latent_attention:
        self.scale_mask_softmax.scale = True
        self.hidden_size_per_partition = config.num_attention_heads * args.v_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)

        if args.rope_scaling_type is not None:
            mscale_all_dim = args.rope_scaling_mscale_all_dim if args.rope_scaling_mscale_all_dim else 0
            scaling_factor = args.rope_scaling_factor

            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.norm_factor = 1.0 / self.softmax_scale

    if self.pse_type is None:
        self.pse_type = 1 # not use pse
    elif self.pse_type == 0:
        alibi = AlibiForFusionAttnSingleton.get_alibi_tensor_for_fusion_attn(args.seq_length,
                                                config.num_attention_heads,
                                                config.params_dtype,
                                                args.alibi_diagonal_opposite,
                                                1024)
        self.pse = alibi
    elif self.pse_type == 2 or self.pse_type == 3:
        self.pse = AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(config.num_attention_heads)


def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params):
        if attention_mask is None and self.attn_mask_type == AttnMaskType.causal:
            attention_mask = get_attention_mask()
        if get_args().use_flash_attn:
            return dot_product_attention_forward(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params)
        return fn(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params)

    return wrapper


def dot_product_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask,
        attn_mask_type,
        packed_seq_params,
):
    is_vit = (getattr(self.config, 'model_id', None) in ['qwen2vit', 'qwen_omni'])
    use_remove_padding = getattr(self.config, 'use_remove_padding', False)
    seq_length, bsz, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
    if not is_vit:
        if attention_mask is not None and 0 not in attention_mask:
            attention_mask = None
    is_inference = hasattr(get_args().mm.model, 'generation_config')
    if is_inference:
        generation_config = get_args().mm.model.generation_config
    if is_inference and getattr(generation_config, 'kv_cache', None) and not is_vit:
        # only for inference!
        query = query.transpose(0, 1).contiguous()  # [b s h d]
        key = key.transpose(0, 1).contiguous()
        value = value.transpose(0, 1).contiguous()
        if query.shape[1] == 1:
            attention_mask_npu = None
        else:
            attention_mask_npu = torch.triu(
                torch.ones([query.shape[1], key.shape[1]], dtype=torch.bool, device=query.device), diagonal=1)

        attn_output = torch_npu.npu_fused_infer_attention_score(query, key, value,
                                                                pse_shift=None,
                                                                atten_mask=attention_mask_npu,
                                                                actual_seq_lengths=[query.shape[1]],
                                                                actual_seq_lengths_kv=[key.shape[1]],
                                                                num_heads=query.shape[2],
                                                                num_key_value_heads=key.shape[2],
                                                                scale=1.0 / math.sqrt(query.shape[-1]),
                                                                input_layout="BSND",
                                                                )[0]
        attn_output = rearrange(attn_output, 'b s h d -> s b (h d)', s=query.shape[1], b=bsz)
        return attn_output
    else:
        if is_vit:
            actual_seq_len = get_actual_seq_len()
            query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]
            attn_output = torch_npu.npu_fusion_attention(
                query, key, value, n_head,
                pse=None,
                padding_mask=None,
                atten_mask=None,
                scale=1.0 / math.sqrt(query.shape[-1]),
                keep_prob=1,
                input_layout='TND',
                actual_seq_qlen=actual_seq_len,
                actual_seq_kvlen=actual_seq_len,
                pre_tockens=2147483647,
                next_tockens=2147483647,
                sparse_mode=0)[0].reshape(seq_length, bsz, -1)
        elif use_remove_padding:
            actual_seq_len = get_actual_seq_len()
            query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]
            attention_mask_npu = torch.triu(
                torch.ones([2048, 2048], dtype=torch.bool, device=query.device), diagonal=1)
            attn_output = torch_npu.npu_fusion_attention(
                query, key, value, n_head,
                pse=None,
                padding_mask=None,
                atten_mask=attention_mask_npu,
                scale=1.0 / math.sqrt(query.shape[-1]),
                keep_prob=1,
                input_layout="TND",
                actual_seq_qlen=actual_seq_len,
                actual_seq_kvlen=actual_seq_len,
                sparse_mode=3)[0].reshape(seq_length, bsz, -1)
        elif attention_mask is not None:
            query = query.transpose(0, 1).contiguous()
            key = key.transpose(0, 1).contiguous()
            value = value.transpose(0, 1).contiguous()
            query, key, value, indices_q, cu_seq_lens, max_seq_lens = _unpad_input(
                query, key, value, attention_mask, seq_length
            )
            attention_mask_npu = torch.triu(
                torch.ones([max_seq_lens, max_seq_lens], dtype=torch.bool, device=query.device), diagonal=1)
            attn_output_unpad = torch_npu.npu_fusion_attention(
                query, key, value, n_head,
                pse=None,
                padding_mask=None,
                atten_mask=attention_mask_npu,
                scale=1.0 / math.sqrt(query.shape[-1]),
                keep_prob=1,
                input_layout='TND',
                actual_seq_qlen=tuple(cu_seq_lens[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seq_lens[1:].cpu().numpy().tolist()),
                pre_tockens=2147483647,
                next_tockens=0)[0]
            attn_output = _pad_input(attn_output_unpad, indices_q, bsz, seq_length)
            attn_output = rearrange(attn_output, 'b s h d -> s b (h d)', s=seq_length, b=bsz)
        else:
            query = query.transpose(0, 1).contiguous()
            key = key.transpose(0, 1).contiguous()
            value = value.transpose(0, 1).contiguous()
            attention_mask_npu = torch.triu(
                torch.ones([query.shape[1], key.shape[1]], dtype=torch.bool, device=query.device), diagonal=1)
            attn_output = torch_npu.npu_fusion_attention(
                query, key, value, n_head, 'BSND',
                keep_prob=1.0,
                scale=1.0 / math.sqrt(query.shape[-1]),
                atten_mask=attention_mask_npu)[0]
            attn_output = rearrange(attn_output, 'b s h d -> s b (h d)', s=seq_length, b=bsz)
        return attn_output


def _unpad_input(
        query_state: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_state.shape
    key_state = index_first_axis(key_state.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices)
    value_state = index_first_axis(value_state.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices)
    if query_length != kv_seq_len:
        raise AssertionError("Only support query_length that equals to kv_seq_len!")
    query_state = index_first_axis(query_state.reshape(batch_size * kv_seq_len, -1, head_dim), indices)

    return(
        query_state,
        key_state,
        value_state,
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _pad_input(hidden_states, indices, batch, seqlen):
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, indices):
        ctx.save_for_backward(indices)
        ctx.first_axis_dim, other_shape = input_tensor.shape[0], input_tensor.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input_tensor, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply