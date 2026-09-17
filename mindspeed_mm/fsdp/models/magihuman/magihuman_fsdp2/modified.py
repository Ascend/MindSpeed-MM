# Copyright (c) 2026 SandAI. All Rights Reserved.
# Copyright 2026 Huawei Technologies Co., Ltd
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

"""FSDP2 / activation-recompute friendly forward rewrites for MagiHuman's DiT.

Mirrors ``mindspeed_mm/fsdp/models/ltx2/ltx2_fsdp2/modified.py``: standalone
functions rebound onto the upstream instance via ``types.MethodType`` in
``MagiHumanForTraining.__init__`` (keeping the vendored upstream source
pristine). Two rewrites are needed:

  * :func:`dit_forward` replaces ``inference/model/dit/dit_module.py::DiTModel.forward``
    (lines 902-950). The ONLY change vs. upstream is that the
    ``ulysses_scheduler().dispatch/undispatch/cp_split_sizes`` wrapping is driven
    by MindSpeed MM's context-parallel groups instead of ``inference.infra.parallelism``
    (a CUDA-coupled distributed stack), which also brings Ring Attention that the
    upstream scheduler has no equivalent for. Both cp dims go through the repo's
    packed-sequence primitives, so the split rule is the shared one: each sample is
    cut per ring rank, the concatenated result is cut again per ulysses rank, and
    remainder tokens go to the lowest ranks (``context_parallel.utils.cal_split_sizes``,
    matching ``UlyssesScheduler._dispatch``). With both parallel sizes at 1
    ``cp_split_sizes`` stays ``None`` and the attention path skips every collective.

  * :func:`block_forward` replaces ``TransFormerLayer.forward`` (dit_module.py:782-811),
    rebound onto each ``self.dit.block.layers[*]``. It is a faithful copy of the
    upstream layer forward; the point is to give every layer a clean standalone
    Python forward (independent of the no-op ``@magi_compile`` decorator on the
    parent ``TransformerBlock``) so FSDP2's recompute_plan has a stable per-layer
    boundary (``features.recompute_plan.apply_modules: [dit.block.layers.{*}]``).

  * :func:`attention_forward` keeps the upstream attention math but forwards the
    packed-sample ``VarlenHandler`` to the NPU full-attention replacement. The
    upstream inference path normally runs one sample and its full-attention call
    does not consume the handler; training with micro-batch > 1 needs it to keep
    concatenated samples isolated.

These are wired in the wrapper's ``__init__`` via ``types.MethodType`` rebinds,
keeping the vendored upstream source pristine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mindspeed_mm.fsdp.distributed.context_parallel.communication import (
    packed_data_gather_forward_split_backward_with_cp,
    packed_data_split_forward_gather_backward_with_cp,
)
from mindspeed_mm.fsdp.distributed.context_parallel.utils import cal_split_sizes_multi
from mindspeed_mm.fsdp.distributed.parallel_state import (
    get_parallel_state,
    is_parallel_state_initialized,
)


@dataclass(frozen=True)
class CPSplitPlan:
    """What the attention path needs to undo the context-parallel token split.

    Upstream's ``TransformerBlock.forward`` (dit_module.py:835-857) has a fixed
    signature and forwards ``cp_split_sizes`` to every layer untouched, so it is
    the only channel that reaches ``Attention.forward`` without patching the
    vendored container as well.

    ``ulysses_gather_size`` is the token count this ring rank owns, i.e. the
    all-to-all gather size that turns a token slice into a head slice.
    ``ring_split_lens`` is ``None`` unless Ring Attention is on; it is the
    ``[ring_size, num_samples]`` table telling the ring how many tokens of each
    packed sample live on every rank, which is what rebuilds ``actual_seq_qlen``
    as KV rotates around the ring.
    """

    ulysses_gather_size: int
    ring_split_lens: torch.Tensor | None


def _cp_split_plan(varlen_handler: Any):
    """Return ``(seq_lens, plan)`` when context parallel is on, else ``None``.

    ``seq_lens`` is the per-sample token count of the whole packed stream, which
    both the split and the gather need. Guarded by
    ``is_parallel_state_initialized`` so the module stays importable and
    unit-testable outside a distributed run.
    """
    if not is_parallel_state_initialized():
        return None
    ps = get_parallel_state()
    if not ps.is_cp_enable():
        return None

    cu_seqlens = varlen_handler.cu_seqlens_q
    seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

    ring_split_lens = None
    tokens_this_ring_rank = sum(seq_lens)
    if ps.is_ring_enable():
        ring_split_lens = cal_split_sizes_multi(seq_lens, ps.get_ring_group_size())
        tokens_this_ring_rank = int(ring_split_lens[ps.get_ring_rank()].sum())

    return seq_lens, CPSplitPlan(
        ulysses_gather_size=tokens_this_ring_rank,
        ring_split_lens=ring_split_lens,
    )


def dit_forward(
    self,
    x: torch.Tensor,
    coords_mapping: torch.Tensor,
    modality_mapping: torch.Tensor,
    varlen_handler: Any,
    local_attn_handler: Any,
) -> torch.Tensor:
    """Drop-in for ``DiTModel.forward`` with the ulysses dispatch wrapping removed.

    Faithful copy of dit_module.py:902-950 EXCEPT the context-parallel dispatch
    lines. The upstream symbols referenced below (``ModalityDispatcher``,
    ``Modality``) live in ``inference.model.dit.dit_module`` and
    ``inference.common`` respectively and are imported lazily so this module is
    importable without the vendored tree present.

    NOTE: equivalent to upstream when the cp world size == 1 (validated).
      Diff against dit_module.py after each upstream re-sync.
    """
    from inference.common import Modality
    from inference.model.dit.dit_module import ModalityDispatcher

    # Context-parallel dispatch: keep only this rank's token slice. Ring splits
    # every packed sample across the ring group, so each rank holds a contiguous
    # chunk of every sample and attention can rotate KV with the per-sample
    # boundaries intact; ulysses then splits that ring-local stream again and
    # trades tokens for heads inside the attention call. With both sizes at 1 the
    # plan stays ``None`` and the attention path skips every collective.
    cp_split_sizes = None
    cp = _cp_split_plan(varlen_handler)
    if cp is not None:
        seq_lens, cp_split_sizes = cp
        x = packed_data_split_forward_gather_backward_with_cp(x, dim=0, seq_lens=seq_lens)
        coords_mapping = packed_data_split_forward_gather_backward_with_cp(
            coords_mapping, dim=0, seq_lens=seq_lens
        )
        modality_mapping = packed_data_split_forward_gather_backward_with_cp(
            modality_mapping, dim=0, seq_lens=seq_lens
        )

    modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
    permute_mapping = modality_dispatcher.permute_mapping
    inv_permute_mapping = modality_dispatcher.inv_permute_mapping
    video_mask = modality_mapping == Modality.VIDEO
    audio_mask = modality_mapping == Modality.AUDIO
    text_mask = modality_mapping == Modality.TEXT

    # The framework's move_to_device casts float batch tensors to param_dtype
    # (bf16); the adapter's RoPE math needs fp32 coords (large / negative grid
    # positions lose precision in bf16). Re-cast model-side (not a dataset change).
    coords_mapping = coords_mapping.float()
    x, rope = self.adapter(x, coords_mapping, video_mask, audio_mask, text_mask)
    x = x.to(self.config.params_dtype)
    x = ModalityDispatcher.permute(x, permute_mapping)
    x = self.block(
        x,
        rope,
        permute_mapping=permute_mapping,
        inv_permute_mapping=inv_permute_mapping,
        varlen_handler=varlen_handler,
        local_attn_handler=local_attn_handler,
        modality_dispatcher=modality_dispatcher,
        cp_split_sizes=cp_split_sizes,
    )
    x = ModalityDispatcher.inv_permute(x, inv_permute_mapping)

    x_video = x[video_mask].to(self.final_norm_video.weight.dtype)
    x_video = self.final_norm_video(x_video)
    x_video = self.final_linear_video(x_video)

    x_audio = x[audio_mask].to(self.final_norm_audio.weight.dtype)
    x_audio = self.final_norm_audio(x_audio)
    x_audio = self.final_linear_audio(x_audio)

    # Output velocity is fp32 (the wrapper computes the flow-matching MSE in fp32).
    # Do NOT use dtype=x.dtype: under FSDP move_to_device makes x bf16 while
    # x_video/x_audio come out fp32 from the fp32 norm/linear chain, and the NPU
    # index_put requires matching dtypes (the single-process smoke fed x as fp32
    # so never hit this).
    x_out = torch.zeros(
        x.shape[0],
        max(self.config.video_in_channels, self.config.audio_in_channels),
        device=x.device,
        dtype=torch.float32,
    )
    x_out[video_mask, : self.config.video_in_channels] = x_video.float()
    x_out[audio_mask, : self.config.audio_in_channels] = x_audio.float()
    # Context-parallel undispatch: rebuild the full token stream so the wrapper
    # computes the loss over the whole packed sequence, exactly as with cp size 1.
    # The gather runs ulysses first and then ring, mirroring the split order. Its
    # ``grad_scale="up"`` multiplies the incoming gradient by the cp size, which
    # compensates FSDP2 averaging the reduce over ``dp_shard_cp`` (that mesh dim
    # includes both cp dims) and keeps the optimizer step identical to cp size 1.
    if cp is not None:
        x_out = packed_data_gather_forward_split_backward_with_cp(
            x_out, dim=0, seq_lens=seq_lens
        )
    return x_out


def attention_forward(
    self,
    hidden_states: torch.Tensor,
    rope: torch.Tensor,
    permute_mapping: torch.Tensor,
    inv_permute_mapping: torch.Tensor,
    varlen_handler: Any,
    local_attn_handler: Any,
    modality_dispatcher: Any,
    cp_split_sizes: Any,
) -> torch.Tensor:
    """Drop-in for upstream ``Attention.forward`` with varlen NPU attention.

    The q/k/v construction, modality permutation, RoPE, optional gating and
    output projection are identical to upstream. For the base model's full
    attention path, the only change is passing ``varlen_handler`` to the NPU
    replacement so multiple packed samples use block-isolated TND attention.
    """
    from inference.model.dit.dit_module import (
        ModalityDispatcher,
        apply_rotary_emb_torch,
        flash_attn_with_cp,
        flex_flash_attn_with_cp,
    )

    hidden_states = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)
    qkv = self.linear_qkv(hidden_states, modality_dispatcher=modality_dispatcher).to(torch.float32)

    q, k, v, g = torch.split(
        qkv,
        [self.q_size, self.kv_size, self.kv_size, self.gating_size],
        dim=1,
    )
    q = q.view(-1, self.config.num_heads_q, self.config.head_dim)
    k = k.view(-1, self.config.num_heads_kv, self.config.head_dim)
    v = v.view(-1, self.config.num_heads_kv, self.config.head_dim)
    g = g.view(k.shape[0], self.config.num_heads_q, -1)

    q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
    k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

    q = ModalityDispatcher.inv_permute(q, inv_permute_mapping).unsqueeze(0)
    k = ModalityDispatcher.inv_permute(k, inv_permute_mapping).unsqueeze(0)
    v = ModalityDispatcher.inv_permute(v, inv_permute_mapping).unsqueeze(0)

    sin_emb, cos_emb = rope.tensor_split(2, -1)
    q = apply_rotary_emb_torch(q, cos_emb, sin_emb)
    k = apply_rotary_emb_torch(k, cos_emb, sin_emb)

    if self.config.use_local_attn:
        self_attn_out = flex_flash_attn_with_cp(
            q,
            k,
            v,
            local_attn_handler.q_ranges,
            local_attn_handler.k_ranges,
            cp_split_sizes,
        )
    else:
        self_attn_out = flash_attn_with_cp(
            q,
            k,
            v,
            cp_split_sizes,
            varlen_handler=varlen_handler,
        )
    self_attn_out = ModalityDispatcher.permute(self_attn_out, permute_mapping)

    if self.config.enable_attn_gating:
        self_attn_out = self_attn_out * torch.sigmoid(g)

    self_attn_out = self_attn_out.view(
        -1, self.config.num_heads_q * self.config.head_dim
    ).to(torch.bfloat16)
    return self.linear_proj(self_attn_out, modality_dispatcher=modality_dispatcher)


def block_forward(
    self,
    hidden_states: torch.Tensor,
    rope: torch.Tensor,
    permute_mapping: torch.Tensor,
    inv_permute_mapping: torch.Tensor,
    varlen_handler: Any,
    local_attn_handler: Any,
    modality_dispatcher: Any,
    cp_split_sizes: Any,
    **_kwargs: Any,  # tolerate framework recompute kwargs (past_key_values / use_cache)
) -> torch.Tensor:
    """Drop-in for ``TransFormerLayer.forward`` (dit_module.py:782-811).

    Rebound onto every ``self.dit.block.layers[*]`` (each a ``TransFormerLayer``)
    so each layer's forward is a clean, standalone Python function — independent
    of whatever the ``@magi_compile`` class decorator on the parent
    ``TransformerBlock`` would otherwise install (it is a no-op on NPU; see
    ``npu_patch.disable_magi_compile``). This gives FSDP2's recompute_plan a
    stable per-layer boundary (``features.recompute_plan.apply_modules:
    [dit.block.layers.{*}]``).

    Faithful copy of upstream — the only "magi_compile path" being dropped is the
    compiler wrapper itself, not anything in the layer body. ``self.post_norm``,
    ``self.attn_post_norm`` and ``self.mlp_post_norm`` only exist on post-norm
    layers (empty for the base model), so they stay guarded by ``self.post_norm``.
    """
    attn_out = self.attention(
        hidden_states,
        rope,
        permute_mapping,
        inv_permute_mapping,
        varlen_handler,
        local_attn_handler,
        modality_dispatcher,
        cp_split_sizes,
    )
    if self.post_norm:
        attn_out = self.attn_post_norm(attn_out, modality_dispatcher=modality_dispatcher)
    hidden_states = hidden_states + attn_out

    mlp_out = self.mlp(hidden_states, modality_dispatcher)
    if self.post_norm:
        mlp_out = self.mlp_post_norm(mlp_out, modality_dispatcher=modality_dispatcher)
    hidden_states = hidden_states + mlp_out
    return hidden_states
