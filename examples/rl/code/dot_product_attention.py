# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import math
from typing import Optional

import torch
from torch import Tensor
import torch_npu

from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide

try:
    from einops import rearrange
except ImportError:
    rearrange = None


class DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

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

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        use_remove_padding = getattr(self.config, 'use_remove_padding', False)
        if use_remove_padding:
            from mindspeed.utils import get_actual_seq_len
            seq_length, bsz, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
            actual_seq_len = get_actual_seq_len()
            query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]
            attention_mask_npu = torch.triu(
                torch.ones([2048, 2048], dtype=torch.bool, device=query.device), diagonal=1)
            output = torch_npu.npu_fusion_attention(
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

        else:
            seq_length, bsz, n_head, head_dim = (
                query.shape[0], query.shape[1], query.shape[2], query.shape[3]
            )

            sparse_mode = self.config.sparse_mode
            if attn_mask_type == AttnMaskType.no_mask:
                sparse_mode = 0  # default mask

            scale = (
                1.0 / math.sqrt(self.hidden_size_per_attention_head) 
                if self.scale_mask_softmax.scale is None 
                else self.softmax_scale
            )

            if packed_seq_params is not None: # TND
                actual_seq_qlen = packed_seq_params.cu_seqlens_q.tolist()
                actual_seq_kvlen = packed_seq_params.cu_seqlens_kv.tolist()
                query, key, value = (
                    [
                        rearrange(x, 's b h d -> (b s) h d')
                        for x in [query, key, value]
                    ]
                )
                shape_order = 'TND'
            else: # SBH
                actual_seq_qlen = None
                actual_seq_kvlen = None
                query, key, value = (
                    [
                        rearrange(x, 's b h d -> s b (h d)')
                        for x in [query, key, value]
                    ]
                )
                shape_order = 'SBH'

            output = torch_npu.npu_fusion_attention(
                query, key, value,
                n_head,
                shape_order,
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
                output = (
                    rearrange(
                        output,
                        '(b s) h d -> s b (h d)',
                        s=seq_length, b=bsz
                    )
                )
                
        return output
