# Copyright 2025 Huawei Technologies Co., Ltd.
# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.masking_utils import create_causal_mask

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.distributed.context_parallel.communication import split_forward_gather_backward_with_cp, gather_forward_split_backward
from mindspeed_mm.fsdp.distributed.context_parallel.utils import cal_split_sizes, generate_ulysses_cu_seqlen_params


_DEFAULT_MASK_BUILDER = object()


def _mask_embedding_keyword(mask_builder):
    """Select the embedding keyword used by this mask builder's API."""
    if mask_builder is not None:
        parameters = inspect.signature(mask_builder).parameters
        if "input_embeds" in parameters and "inputs_embeds" not in parameters:
            return "input_embeds"
    return "inputs_embeds"


def shift_tensor(tensor, shifts=-1, dims=-1, fill_value=0, cu_seqlens=None):
    """Shift a tensor and clear positions that would cross packed boundaries."""
    if cu_seqlens is not None:
        shifted = torch.roll(tensor.reshape(-1), shifts=shifts, dims=0)
        shifted.index_fill_(0, cu_seqlens[1:] - 1, fill_value)
        return shifted.view_as(tensor)

    shifted = torch.roll(tensor, shifts=shifts, dims=dims)
    shifted.select(dims, shifts).fill_(fill_value)
    return shifted


class _MTPPerTokenLoss(torch.autograd.Function):
    """Return the MTP mean loss while applying Megatron's per-token gradient scale."""

    @staticmethod
    def forward(ctx, loss_sum, original_num_tokens, mtp_num_tokens, global_avg_num_tokens):
        ctx.save_for_backward(original_num_tokens, mtp_num_tokens, global_avg_num_tokens)
        return loss_sum / mtp_num_tokens.to(loss_sum.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        original_num_tokens, mtp_num_tokens, global_avg_num_tokens = ctx.saved_tensors
        grad_scale = (
            original_num_tokens.to(grad_output.dtype)
            / mtp_num_tokens.to(grad_output.dtype)
            / global_avg_num_tokens.to(grad_output.dtype)
        )
        return grad_output * grad_scale, None, None, None


class MultiTokenPredictionBlock(nn.Module):
    """Compute auxiliary losses for ``config.mtp_num_layers`` prediction depths.

    Depth 1 predicts token t+2 from the backbone state at t and the token t+1
    embedding; each later depth advances the target by one more token.
    ``share_decoder=True`` reuses one MTP decoder across all depths; otherwise
    each depth owns a decoder. Backbone decoder parameters are not shared.

    ``position_ids_mode`` describes the position-ID layout, not a RoPE algorithm:
    ``text`` uses [batch, sequence] positions (the first axis of 3-D input),
    while ``mrope`` uses [3, batch, sequence] positions for rotary embeddings.
    In mrope mode, 2-D input is expanded to three axes; 4-axis input separates
    the leading text positions from the remaining three rotary-position axes.
    The decoder and causal mask receive the text positions in both modes.
    A rotary module is optional, including for NoPE attention.
    """

    def __init__(
        self,
        config,
        layer_cls,
        norm_cls,
        position_ids_mode: str = "mrope",
        causal_mask_builder: Optional[Callable] = _DEFAULT_MASK_BUILDER,
        share_decoder: bool = True,
    ):
        super().__init__()
        if position_ids_mode not in ("mrope", "text"):
            raise ValueError(f"Unsupported MTP position_ids_mode: {position_ids_mode}")
        self.config = config
        self.position_ids_mode = position_ids_mode
        self.share_decoder = share_decoder
        self.causal_mask_builder = (
            create_causal_mask if causal_mask_builder is _DEFAULT_MASK_BUILDER else causal_mask_builder
        )
        self._mask_embedding_name = _mask_embedding_keyword(self.causal_mask_builder)
        num_physical_layers = int(config.mtp_num_layers > 0) if share_decoder else config.mtp_num_layers
        self.layers = nn.ModuleList([
            layer_cls(config, i, is_mtp=True)
            for i in range(num_physical_layers)
        ])
        if self.layers:
            layer_signature = inspect.signature(self.layers[0].forward)
            self._layer_forward_names = set(layer_signature.parameters)
            self._layer_accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in layer_signature.parameters.values()
            )
        norm_eps = getattr(config, "rms_norm_eps", getattr(config, "layer_norm_epsilon", 1e-5))
        self.pre_fc_norm_embedding = norm_cls(config.hidden_size, eps=norm_eps)
        self.pre_fc_norm_hidden = norm_cls(config.hidden_size, eps=norm_eps)
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.norm = norm_cls(config.hidden_size, eps=norm_eps)

    def _prepare_position_ids(
        self,
        position_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[Cache],
        cache_position: Optional[torch.LongTensor]
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            if self.position_ids_mode == "text":
                position_ids = cache_position.view(1, -1).expand(inputs_embeds.shape[0], -1)
            else:
                position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2 and self.position_ids_mode == "mrope":
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if self.position_ids_mode == "text":
            text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
            position_ids = text_position_ids
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        return position_ids, text_position_ids, cache_position

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        embed_tokens: Optional[nn.Module] = None,
        rotary_emb: Optional[nn.Module] = None,
        output_layer: Optional[nn.Module] = None,
        loss_function: Optional[Callable] = None,
        logits_to_keep: int | torch.Tensor = 0,
        seq_len: Optional[int] = 1,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if not self.layers:
            return None
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        all_mtp_loss = None
        calculate_per_token_loss = hasattr(loss_function, "avg_per_step_token_num")
        if calculate_per_token_loss:
            avg_per_step_token_num = getattr(loss_function, "avg_per_step_token_num", None)
            if avg_per_step_token_num is None:
                raise KeyError("per_token_loss must use PrefetchGradAccDataLoader")
            original_num_tokens = (labels > -1).sum()
            global_avg_num_tokens = avg_per_step_token_num.detach().clone()
            torch.distributed.all_reduce(global_avg_num_tokens, op=torch.distributed.ReduceOp.AVG)

        ps = get_parallel_state()
        use_packing = "cu_seqlens" in kwargs and kwargs["cu_seqlens"] is not None
        cu_seqlens = None
        if use_packing:
            text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
            kwargs.update(generate_ulysses_cu_seqlen_params(
                text_position_ids, need_cpu_tensor=ps.is_ulysses_enable()
            ))
            cu_seqlens = kwargs["cu_seqlens"].squeeze(0).to(device=input_ids.device, dtype=torch.long)

        for step in range(self.config.mtp_num_layers):
            decoder_layer = self.layers[0 if self.share_decoder else step]
            input_ids = shift_tensor(input_ids, cu_seqlens=cu_seqlens)
            labels = shift_tensor(labels, fill_value=-100, cu_seqlens=cu_seqlens)
            if cu_seqlens is not None:
                # The shared causal loss shifts once more; mask starts to protect the preceding sequence.
                labels.view(-1).index_fill_(0, cu_seqlens[1:-1], -100)
            inputs_embeds = embed_tokens(input_ids)

            position_ids, text_position_ids, cache_position = self._prepare_position_ids(
                position_ids, inputs_embeds, past_key_values, cache_position
            )

            if use_packing or self.causal_mask_builder is None:
                causal_mask = None
            else:
                causal_mask = self.causal_mask_builder(
                    config=self.config,
                    **{self._mask_embedding_name: inputs_embeds},
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                )

            if ps.is_ulysses_enable():
                if not use_packing:
                    kwargs.update(generate_ulysses_cu_seqlen_params(text_position_ids))
                position_ids = split_forward_gather_backward_with_cp(position_ids, dim=position_ids.ndim - 1)
                text_position_ids = split_forward_gather_backward_with_cp(text_position_ids, dim=1)
                inputs_embeds = split_forward_gather_backward_with_cp(inputs_embeds, dim=1)

            # NoPE attention has no rotary embedding module.
            position_embeddings = (
                rotary_emb(hidden_states, position_ids) if rotary_emb is not None else None
            )

            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))

            layer_kwargs = dict(
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            if not self._layer_accepts_kwargs:
                layer_kwargs = {k: v for k, v in layer_kwargs.items() if k in self._layer_forward_names}
            hidden_states = decoder_layer(hidden_states, **layer_kwargs)
            if isinstance(hidden_states, (tuple, list)):
                hidden_states = hidden_states[0]
            if not isinstance(hidden_states, torch.Tensor):
                raise ValueError("MTP decoder must return a tensor or a tuple starting with a tensor")

            hidden_states = self.norm(hidden_states)

            if getattr(self.config, "enable_chunk_loss", False) or getattr(self.config, "enable_dynamic_chunk_loss", False):
                mtp_loss = output_layer(hidden_states[:, slice_indices, :], loss_function, labels)
            else:
                mtp_logits = output_layer(hidden_states[:, slice_indices, :])
                if labels is not None:
                    mtp_loss = loss_function(mtp_logits, labels, vocab_size=self.config.vocab_size)

            if all_mtp_loss is None:
                all_mtp_loss = []
            if calculate_per_token_loss:
                # The MTP loss function returns the local loss sum. Match
                # Megatron's N/M correction, then use the main loss denominator D.
                mtp_labels = shift_tensor(labels, shifts=-1, dims=-1, fill_value=-100)
                mtp_num_tokens = (mtp_labels > -1).sum().clamp(min=1)
                mtp_loss = _MTPPerTokenLoss.apply(
                    mtp_loss,
                    original_num_tokens,
                    mtp_num_tokens,
                    global_avg_num_tokens.clamp(min=1),
                )
            all_mtp_loss.append(mtp_loss)

            if ps.is_ulysses_enable():
                gather_sizes = cal_split_sizes(seq_len, ps.get_ulysses_group_size())
                position_ids = gather_forward_split_backward(position_ids, ps.get_ulysses_group(), dim=position_ids.ndim - 1, grad_scale="up", gather_sizes=gather_sizes)
                text_position_ids = gather_forward_split_backward(text_position_ids, ps.get_ulysses_group(), dim=1, grad_scale="up", gather_sizes=gather_sizes)

        if all_mtp_loss is not None and ps.is_cp_enable():
            for i in range(len(all_mtp_loss)):
                all_mtp_loss[i] = gather_forward_split_backward(all_mtp_loss[i].unsqueeze(0), ps.get_cp_group(), dim=0)
                all_mtp_loss[i] = all_mtp_loss[i].sum()
        return all_mtp_loss
