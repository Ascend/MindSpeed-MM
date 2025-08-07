# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
import torch
import torch_npu

from megatron.core.transformer.utils import attention_mask_func
from mindspeed.core.context_parallel import AttnMaskType
from mindspeed.core.context_parallel import FusedScaleMaskSoftmax
from mindspeed.core.context_parallel import divide
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.core.transformer.flash_attention.alibi.alibi import AlibiForFusionAttnSingleton
from mindspeed.core.context_parallel import mpu as parallel_state
from mindspeed.model.transformer import get_attention_mask
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.core.context_parallel.ring_context_parallel.context_parallel_kv_cache import get_cache_policy
from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import ulyssesattn_context_parallel
from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.context_parallel.utils import get_scheduling_info
from mindspeed.core.context_parallel.adaptive_context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
from mindspeed.core.context_parallel.model_parallel_utils import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)


try:
    from einops import rearrange
except ImportError:
    rearrange = None


class CPDotProductAttentionImpl:
    """
    Implementation of dot product attention with cp support.
    """

    def __init__(self,
                 config,
                 layer_number,
                 attn_mask_type,
                 attention_type,
                 attention_dropout: float = None,
                 softmax_scale: float = None,
                 cp_comm_type: str = None):
        cp_size = config.context_parallel_size
        config.context_parallel_size = 1
        self.config = config
        super().__init__(config, layer_number, attn_mask_type, attention_type, attention_dropout, softmax_scale, cp_comm_type)

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = self.config.kv_channels * self.config.num_attention_heads
        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

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
        self.pse_type = self.config.alibi_fusion_attn_type

        if self.pse_type is None:
            self.pse_type = 1 # not use pse
        elif self.pse_type == 0:
            alibi = (
                AlibiForFusionAttnSingleton.get_alibi_tensor_for_fusion_attn(
                    self.config.seq_length,
                    self.config.num_attention_heads,
                    self.config.params_dtype,
                    self.config.alibi_diagonal_opposite,
                    1024
                )
            )
            self.pse = alibi
        elif self.pse_type == 2 or self.pse_type == 3:
            self.pse = (
                AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(
                    self.config.num_attention_heads
                )
            )

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        if attention_mask is None and self.attn_mask_type == AttnMaskType.causal:
            if not getattr(self.config, 'is_llava', False):
                attention_mask = get_attention_mask()
                if self.config.attention_mask_type == 'causal':
                    self.config.sparse_mode = 2
                if self.config.reset_attention_mask:
                    if self.config.attention_mask_type == 'general':
                        self.config.sparse_mode = 2
                        if not (self.config.context_parallel_size == 1 or self.config.context_parallel_algo == 'ulysses_cp_algo'):
                            self.config.sparse_mode = 1

        sparse_mode = self.config.sparse_mode

        seq_length, bsz, n_head = query.shape[0], query.shape[1], query.shape[2]

        if attn_mask_type == AttnMaskType.no_mask:
            sparse_mode = 0  # default mask

        scale = 1.0 / math.sqrt(
            self.hidden_size_per_attention_head) if self.scale_mask_softmax.scale is None else self.softmax_scale

        cp_expanded_by_2d_tp = getattr(self.config, 'tp_2d', False) and getattr(self.config, 'tp_y', 1) > 1
        if cp_expanded_by_2d_tp:
            tp_y_cp_sz = TensorParallelYUnionCP().get_parallel_group_world_size()
        else:
            tp_y_cp_sz = self.config.context_parallel_size

        if (self.config.context_parallel_size > 1 and self.config.context_parallel_algo == "ulysses_cp_algo"
                and self.config.context_parallel_kv_cache_policy):
            self.ulysses_comm_para['cache_policy'] = get_cache_policy(
                self.layer_number, self.config.context_parallel_kv_cache_policy, self.config.context_parallel_cache_interval
            )
            self.ulysses_comm_para['use_ulysses_allgather_kv'] = self.config.use_ulysses_allgather_kv

            attn_para = dict()
            attn_para['packed_seq_params'] = packed_seq_params
            attn_para['attention_mask'] = attention_mask
            attn_para['scale'] = scale
            attn_para['pre_tokens'] = self.config.pre_tockens
            attn_para['next_tokens'] = self.config.next_tockens
            attn_para['keep_prob'] = 1 - self.attention_dropout.p
            attn_para['sparse_mode'] = sparse_mode
            output = ulyssesattn_context_parallel(query, key, value, attn_para, self.ulysses_comm_para)

            return output
        if tp_y_cp_sz > 1 and self.config.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo',
                                                             'adaptive_cp_algo', 'hybrid_adaptive_cp_algo']:
            in_hybrid_mode = False
            if get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None:
                in_hybrid_mode = True

            if not in_hybrid_mode:
                if cp_expanded_by_2d_tp:
                    tp_y_cp = TensorParallelYUnionCP()
                    cp_group = tp_y_cp.group
                    cp_size = tp_y_cp.get_parallel_group_world_size()
                    rank = tp_y_cp.get_parallel_rank()
                    cp_global_ranks = tp_y_cp.global_ranks
                else:
                    cp_group = parallel_state.get_context_parallel_group()
                    cp_size = parallel_state.get_context_parallel_world_size()
                    rank = parallel_state.get_context_parallel_rank()
                    cp_global_ranks = parallel_state.get_context_parallel_global_ranks()
            else:
                cp_group = get_context_parallel_group_for_hybrid_ring()
                cp_size = get_context_parallel_for_hybrid_ring_world_size()
                rank = get_context_parallel_for_hybrid_ring_rank()
                cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()

            cp_para = dict()
            cp_para['megatron_cp_in_bnsd'] = self.config.megatron_cp_in_bnsd
            cp_para['causal'] = self.config.attention_mask_type == 'causal'
            cp_para['cp_group'] = cp_group
            cp_para['cp_size'] = cp_size
            cp_para['rank'] = rank

            query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
            if self.config.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
                cp_para['cp_global_ranks'] = cp_global_ranks
                if self.config.use_cp_send_recv_overlap:
                    if cp_expanded_by_2d_tp:
                        cp_para['cp_group_for_send_recv_overlap'] = tp_y_cp.overlap_group
                    else:
                        cp_para[
                            'cp_group_for_send_recv_overlap'] = parallel_state.get_context_parallel_group_for_send_recv_overlap()
                else:
                    cp_para['cp_group_for_send_recv_overlap'] = None
                cp_para['pse'] = self.pse
                cp_para['pse_type'] = self.pse_type

                if self.config.context_parallel_size > 1 and not getattr(self.config, 'tp_2d', False):
                    cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
                    cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
                    cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
                    cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
                    cp_para[
                        'cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()
                    cp_para['cache_policy'] = get_cache_policy(
                        self.layer_number, self.config.context_parallel_kv_cache_policy, self.config.context_parallel_cache_interval
                    )
                output = ringattn_context_parallel(query, key, value, n_head, cp_para, scale, attention_mask,
                                                   self.attention_dropout.p,
                                                   packed_seq_params)

            else:
                cp_para['scheduling_info'] = get_scheduling_info()
                output = adaptive_attn_context_parallel(query, key, value, n_head, cp_para, scale, attention_mask,
                                                        self.attention_dropout.p)

        else:
            if not getattr(self.config, 'use_remove_padding', False):
                if packed_seq_params is not None:  # TND
                    cp_size = parallel_state.get_context_parallel_world_size()
                    actual_seq_qlen = packed_seq_params.cu_seqlens_q.tolist()
                    actual_seq_kvlen = packed_seq_params.cu_seqlens_kv.tolist()
                    query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]
                    shape_order = 'TND'
                else:  # SBH
                    actual_seq_qlen = None
                    actual_seq_kvlen = None
                    query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
                    shape_order = 'SBH'
            if self.config.use_fusion_attn_v2:
                output = npu_fusion_attention(
                    query, key, value, n_head, shape_order,
                    pse=self.pse,
                    padding_mask=None,
                    atten_mask=attention_mask,
                    scale=scale,
                    pse_type=self.pse_type,
                    pre_tokens=self.config.pre_tockens,
                    next_tokens=self.config.next_tockens,
                    keep_prob=1 - self.attention_dropout.p,
                    inner_precise=0,
                    sparse_mode=sparse_mode,
                    actual_seq_qlen=actual_seq_qlen,
                    actual_seq_kvlen=actual_seq_kvlen
                )[0]
            else:
                use_remove_padding = getattr(self.config, 'use_remove_padding', False)
                if use_remove_padding:
                    attention_mask_npu = torch.triu(
                        torch.ones([2048, 2048], dtype=torch.bool, device=query.device), diagonal=1
                    )
                    query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]
                    from mindspeed.utils import get_actual_seq_len
                    actual_seq_len = get_actual_seq_len()
                    output = torch_npu.npu_fusion_attention(
                        query, key, value, n_head,
                        pse=None,
                        padding_mask=None,
                        atten_mask=attention_mask_npu,
                        scale=1.0 / math.sqrt(query.shape[-1]),
                        keep_prob=1,
                        input_layout="TND",
                        sparse_mode=3,
                        actual_seq_qlen=actual_seq_len,
                        actual_seq_kvlen=actual_seq_len
                    )[0].reshape(seq_length, bsz, -1)
                else:
                    output = torch_npu.npu_fusion_attention(
                        query, key, value, n_head, shape_order,
                        pse=None,
                        padding_mask=None,
                        atten_mask=attention_mask,
                        scale=scale,
                        pre_tockens=self.config.pre_tockens,
                        next_tockens=self.config.next_tockens,
                        keep_prob=1 - self.attention_dropout.p,
                        inner_precise=0,
                        sparse_mode=sparse_mode,
                        actual_seq_qlen=actual_seq_qlen,
                        actual_seq_kvlen=actual_seq_kvlen
                    )[0]
            if packed_seq_params is not None:
                output = rearrange(output, '(b s) h d -> s b (h d)', s=seq_length, b=bsz)
        return output
