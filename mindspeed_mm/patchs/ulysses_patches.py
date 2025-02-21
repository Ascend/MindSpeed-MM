# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Tuple, List

import torch 
from torch import Tensor 
from torch.nn import Module 
from mindspeed.megatron_adaptor import get_mindspeed_args
from mindspeed.patch_utils import MindSpeedPatchesManager as pm

from mindspeed_mm.models.common.communications import cal_split_sizes
from mindspeed_mm.models.common.communications import all_to_all




class UlyssesContextAttention(torch.nn.Module):
    """Initialization.
    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """
    def __init__(
            self,
            local_attention: Module,
            sequence_process_group: torch.distributed.ProcessGroup,
            scatter_idx: int = 2,
            gather_idx: int = 0,
    ) -> None:
        super(UlyssesContextAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        attention_mask = args[0]
        act_seq_len = attention_mask.shape[-1]
        scatter_sizes_query = cal_split_sizes(query.shape[self.scatter_idx], dist.get_world_size(self.spg))
        scatter_sizes_key = cal_split_sizes(key.shape[self.scatter_idx], dist.get_world_size(self.spg))
        scatter_sizes_value = cal_split_sizes(value.shape[self.scatter_idx], dist.get_world_size(self.spg))

        gather_sizes = cal_split_sizes(act_seq_len, dist.get_world_size(self.spg))

        query_layer = all_to_all(query, self.spg, self.scatter_idx, self.gather_idx, scatter_sizes_query, gather_sizes)
        key_layer = all_to_all(key, self.spg, self.scatter_idx, self.gather_idx, scatter_sizes_key, gather_sizes)
        value_layer = all_to_all(value, self.spg, self.scatter_idx, self.gather_idx, scatter_sizes_value, gather_sizes)

        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)
        context_shape = context_layer.shape
        context_layer = context_layer.reshape(context_shape[0], context_shape[1],
                                            scatter_sizes_query[dist.get_rank(self.spg)], -1)
        
        output = all_to_all(context_layer, self.spg, self.gather_idx, self.scatter_idx, gather_sizes, scatter_sizes_query)
        output = output.reshape(output.shape[0], output.shape[1], -1)

        # out e.g., [s/p::h]
        return output

mindspeed_args = get_mindspeed_args()
if hasattr(mindspeed_args, 'context_parallel_algo') and hasattr(mindspeed_args, 'context_parallel_size'):
    if mindspeed_args.context_parallel_algo == "ulysses_cp_algo" and int(mindspeed_args.context_parallel_size) > 1:
        pm.register_patch('mindspeed.core.context_parallel.ulysses_context_parallel.UlyssesContextAttention', 
                        UlyssesContextAttention, force_patch=True)
        pm.apply_patches()