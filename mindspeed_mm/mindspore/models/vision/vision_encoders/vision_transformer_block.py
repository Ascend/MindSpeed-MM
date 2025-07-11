# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from contextlib import nullcontext

import torch
from torch import Tensor
from megatron.core import InferenceParams, parallel_state, tensor_parallel, mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import make_viewless_tensor
from mindspeed.utils import set_actual_seq_len


def qwen2vl_block_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        window_mask=None,
        cu_seqlens=None,
        cu_window_seqlens=None,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
):
    # hidden_states (float): [s, b, h]
    # attention_mask (bool): [1, 1, s, s]

    if not self.pre_process:
        # See set_input_tensor()
        hidden_states = self.input_tensor

    # Viewless tensor.
    # - We only need to create a viewless tensor in the case of micro batch
    #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
    #   above creates a view tensor, and '.contiguous()' is a pass-through.
    #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
    #   the need to make it viewless.
    #
    #   However, we don't explicitly check mbs == 1 here because
    #   make_viewless_tensor() has negligible overhead when its input
    #   is already viewless.
    #
    # - For the 'else' case above, calling make_viewless_tensor() here is
    #   likely redundant, since p2p_communication.py (likely originator)
    #   already creates viewless tensors. That said, make_viewless_tensor()
    #   is called here to be future-proof and corner-case-proof.
    hidden_states = make_viewless_tensor(
        inp=hidden_states, requires_grad=True, keep_graph=True,
    )

    if self.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    if self.config.fp8:
        import transformer_engine  # To keep out TE dependency when not training in fp8

        if self.config.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif self.config.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
            margin=self.config.fp8_margin,
            interval=self.config.fp8_interval,
            fp8_format=fp8_format,
            amax_compute_algo=self.config.fp8_amax_compute_algo,
            amax_history_len=self.config.fp8_amax_history_len,
            override_linear_precision=(False, False, not self.config.fp8_wgrad),
        )
        fp8_group = None
        if parallel_state.model_parallel_is_initialized():
            fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True)
        fp8_context = transformer_engine.pytorch.fp8_autocast(
            enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
        )
    else:
        fp8_context = nullcontext()
    if getattr(self.config, "window_attn_size", None) is not None:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        fullatt_block_indexes_now = []
        for x in self.config.fullatt_block_indexes:
            fullatt_block_indexes_now.append(x - sum(self.config.pipeline_num_layers[:pp_rank]))
    with rng_context and fp8_context:
        # Forward pass.
        if self.config.recompute_granularity == 'full' and self.training:
            hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            )
        else:
            cu_seqlens_np = tuple(cu_seqlens.numpy()[1:].tolist())
            cu_window_seqlens_np = tuple(cu_window_seqlens.numpy()[1:].tolist())
            for layer_num, layer in enumerate(self.layers):
                with self.offload_context:
                    if getattr(self.config, "window_attn_size", None) is not None:
                        if layer_num in fullatt_block_indexes_now:
                            attention_mask_now = attention_mask
                            set_actual_seq_len(cu_seqlens_np)
                        else:
                            attention_mask_now = window_mask
                            set_actual_seq_len(cu_window_seqlens_np)
                    else:
                        attention_mask_now = attention_mask
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask_now,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                    )

                if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(hidden_states)
    # Final layer norm.
    if self.post_process and self.post_layer_norm:
        hidden_states = self.final_layernorm(hidden_states)

    return hidden_states
