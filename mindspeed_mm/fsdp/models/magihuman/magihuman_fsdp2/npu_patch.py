"""NPU patches for MagiHuman's CUDA-only attention + custom-op stack.

MagiHuman's upstream DiT (``inference/model/dit/dit_module.py``) is CUDA/custom-
kernel heavy. Three things break on Ascend NPU and are neutralized here BEFORE
the upstream module is imported/instantiated (called from
``modeling_magihuman._import_dit`` via :func:`apply_npu_patches`):

  1. ``magi_compile`` decorator on ``TransformerBlock`` (from ``magi_compiler``)
     — proprietary CUDA-graph/offload compiler. Must be a no-op.
  2. ``@magi_register_custom_op`` ops (``magi_compiler.api``) — must become plain
     Python fns (the registration machinery is CUDA-graph aware).
  3. The actual attention math: FA2/FA3 ``flash_attn_func`` and
     ``magi_attention`` flex-attention — replace with
     ``torch_npu.npu_fusion_attention`` (or SDPA fallback).

Each function documents the exact upstream symbol it replaces. The attention
rebind is live (NPU ``npu_fusion_attention``) but stays a *safe no-op* when the
upstream tree isn't vendored yet (the import fails and we return), so plugin
import never fails off-NPU. The ``magi_compiler`` shim and the local-attn path
remain raising/identity stubs (the base model never exercises them).

Reference for the NPU fusion-attention call shape:
  ``mindspeed_mm/fsdp/models/wan2_2/modeling_wan2_2.py::flash_attention`` (lines
  ~282-345) shows the `torch_npu.npu_fusion_attention(q, k, v, head_num, ...)`
  branch used elsewhere in this repo.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any, List

import torch

try:  # pragma: no cover - NPU-only
    import torch_npu
except Exception:
    torch_npu = None

if TYPE_CHECKING:  # annotation only: keep this module's import surface minimal
    from .modified import CPSplitPlan


_PATCHED = False


def apply_npu_patches() -> None:
    """Idempotent entry point. Apply all MagiHuman NPU patches.

    Order matters: stub out ``magi_compiler`` (so the class-body decorator on
    ``TransformerBlock`` is a no-op) BEFORE ``dit_module`` is imported, then
    replace the attention ops.
    """
    global _PATCHED
    if _PATCHED:
        return
    disable_magi_compile()
    patch_attention_ops()
    patch_linear_for_training()
    patch_fused_elementwise_ops()
    patch_swiglu7_deinterleaved()
    _PATCHED = True


# ---------------------------------------------------------------------------
# 1 + 2. Neutralize the magi_compiler stack.
# ---------------------------------------------------------------------------
def disable_magi_compile() -> None:
    """Install a fake ``magi_compiler`` package so upstream imports no-op.

    Upstream ``dit_module.py`` does, at module top:
        from magi_compiler import magi_compile
        from magi_compiler.api import magi_register_custom_op
        from magi_compiler.config import CompileConfig
    and decorates ``TransformerBlock`` with ``@magi_compile(config_patch=...)``
    and the four attention fns with ``@magi_register_custom_op(...)``.

    Makes `magi_compile` an identity class decorator and
      `magi_register_custom_op` an identity function decorator (signatures
      matched against the upstream `magi_compiler` API: config_patch kwarg;
      name/mutates_args/infer_output_meta_fn/is_subgraph_boundary kwargs).
    """
    if "magi_compiler" in sys.modules:
        return

    fake = _make_fake_magi_compiler()
    sys.modules["magi_compiler"] = fake
    sys.modules["magi_compiler.api"] = fake.api
    sys.modules["magi_compiler.config"] = fake.config


def _make_fake_magi_compiler():
    import types as _types

    pkg = _types.ModuleType("magi_compiler")
    api = _types.ModuleType("magi_compiler.api")
    config = _types.ModuleType("magi_compiler.config")

    def magi_compile(*_dargs, **_dkwargs):
        # Used as `@magi_compile(config_patch=...)` -> returns identity class deco.
        def _deco(cls):
            return cls

        return _deco

    def magi_register_custom_op(*_dargs, **_dkwargs):
        # Used as `@magi_register_custom_op(name=..., ...)` -> identity fn deco.
        def _deco(fn):
            return fn

        return _deco

    class CompileConfig:  # minimal stand-in for type hints / config_patch arg
        offload_config = type("OffloadCfg", (), {"gpu_resident_weight_ratio": 1.0})()

    pkg.magi_compile = magi_compile
    pkg.api = api
    pkg.config = config
    api.magi_register_custom_op = magi_register_custom_op
    config.CompileConfig = CompileConfig
    return pkg


# ---------------------------------------------------------------------------
# 3. Replace the attention ops with NPU fusion attention.
# ---------------------------------------------------------------------------
def patch_attention_ops() -> None:
    """Swap the CUDA attention paths in ``dit_module`` for NPU equivalents.

    The base MagiHuman model (``local_attn_layers=[]``) uses ONLY the full
    self-attention path ``flash_attn_with_cp`` (dit_module.py:506-524), which
    internally calls ``torch.ops.infra.flash_attn_func`` (FA2/FA3, gated on the
    CUDA-only ``is_hopper_arch()`` landmine). For first bring-up we set
    ulysses/cp = 1, so the cp all-to-all branch (``get_cp_world_size() > 1``) is
    skipped and we only need the core q,k,v -> out attention.

    These are module-level functions called by name inside ``Attention.forward``
    (dit_module.py:640-644), so rebinding the names on the module object is
    sufficient — no per-instance patch needed.

    Two important properties:
      * ``torch.ops.infra.flash_attn_func`` does NOT exist on NPU: our fake
        ``magi_register_custom_op`` is an identity decorator that never registers
        a torch custom op. So the upstream ``flash_attn_with_cp`` body would
        raise ("no such operator") if it ran — we therefore replace the WHOLE
        ``flash_attn_with_cp`` function, not just the inner op.
      * The replacement is MANDATORY: if the upstream symbol is missing we raise,
        so the CUDA flash-attn path (the ``is_hopper_arch`` landmine) can never
        be reached on NPU. (When the upstream tree isn't vendored yet the import
        fails and we no-op, keeping plugin import safe off-NPU.)
    """
    try:
        mod = importlib.import_module("inference.model.dit.dit_module")
    except Exception:  # upstream not vendored yet; import-safe no-op
        return

    # MANDATORY: the base-model attention entrypoint must exist. Fail loud rather
    # than silently fall through to the CUDA flash-attn / is_hopper_arch path.
    if not hasattr(mod, "flash_attn_with_cp"):
        raise RuntimeError(
            "magihuman npu_patch: upstream dit_module has no `flash_attn_with_cp` "
            "to replace; refusing so the CUDA flash-attn path can never run on NPU."
        )

    mod.flash_attn_with_cp = npu_flash_attn_with_cp
    # Defense-in-depth: route the bare FA2/FA3 entrypoint (the is_hopper_arch /
    # CUDA-only landmine) through NPU too, so no reachable code path can hit the
    # CUDA kernel even if upstream is re-synced to call it directly.
    if hasattr(mod, "flash_attn_func"):
        mod.flash_attn_func = npu_flash_attn_func
    # local-attn / SR path is unused by the base model (local_attn_layers=[]);
    # bind a raising stub so accidental enablement is caught loudly, not silently
    # routed to a missing torch.op or the CUDA kernel.
    if hasattr(mod, "flex_flash_attn_with_cp"):
        mod.flex_flash_attn_with_cp = npu_flex_flash_attn_with_cp


def _npu_attention_bsnd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Core NPU full (bidirectional) self-attention, FA2-equivalent.

    Replaces ``dit_module.flash_attn_func`` (FA2/FA3 ``flash_attn_func(q, k, v)``)
    which runs full, non-causal attention with ``softmax_scale = head_dim**-0.5``.

    Inputs/outputs match the upstream FA2 BSND convention exactly:
      q: [B=1, S, num_heads_q, head_dim]
      k: [B=1, S, num_heads_kv, head_dim]   (GQA: num_heads_kv < num_heads_q)
      v: [B=1, S, num_heads_kv, head_dim]
      returns: [B=1, S, num_heads_q, head_dim]

    GQA is handled natively by ``npu_fusion_attention``: ``head_num`` is the QUERY
    head count and the kernel infers the (smaller) kv head count from the k/v
    tensors, requiring ``num_heads_q % num_heads_kv == 0`` (40 % 8 == 0 here) —
    same convention as the repo's non-ulysses path in
    ``mindspeed_mm/fsdp/ops/flash_attn/flash_attn.py``. If a CANN build rejects
    GQA in BSND, repeat_interleave k/v on the head dim to num_heads_q first
    (mirroring flash_attn.py's ulysses branch).
    """
    if torch_npu is None:  # pragma: no cover - guarded so module imports off-NPU
        raise RuntimeError("magihuman npu attention requires torch_npu (NPU runtime).")

    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    num_heads_q = q.shape[-2]
    head_dim = q.shape[-1]
    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        num_heads_q,
        "BSND",
        pse=None,
        padding_mask=None,
        atten_mask=None,  # full bidirectional attention (no causal mask)
        scale=head_dim**-0.5,
        keep_prob=1.0,  # no dropout
        inner_precise=0,
        sparse_mode=0,
    )[0]
    return out  # [1, S, num_heads_q, head_dim]


def npu_flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """NPU replacement for ``dit_module.flash_attn_func`` (BSND, no squeeze).

    Returns ``[1, S, num_heads_q, head_dim]`` exactly like the upstream FA2/FA3
    ``flash_attn_func`` (the ``.squeeze(0)`` is applied by callers).
    """
    return _npu_attention_bsnd(q, k, v)


def _npu_attention_tnd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    varlen_handler: Any,
) -> torch.Tensor:
    """Block-isolated full attention for multiple packed samples.

    MagiHuman concatenates every sample into one token stream. ``TND`` plus
    cumulative sequence endpoints tells the NPU kernel that attention is full
    *within* each sample but must not cross a sample boundary.
    """
    if torch_npu is None:  # pragma: no cover - NPU-only
        raise RuntimeError("magihuman npu attention requires torch_npu (NPU runtime).")

    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError(f"Expected q in [1, T, N, D], got {tuple(q.shape)}.")
    if k.ndim != 4 or v.ndim != 4 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("Expected k/v in [1, T, N, D] for packed attention.")

    cu_q = varlen_handler.cu_seqlens_q
    cu_k = varlen_handler.cu_seqlens_k
    if cu_q.ndim != 1 or cu_k.ndim != 1 or cu_q.numel() < 3:
        raise ValueError("Packed TND attention requires at least two sample spans.")
    if not torch.equal(cu_q, cu_k):
        raise ValueError("MagiHuman self-attention requires identical q/k boundaries.")
    if int(cu_q[0].item()) != 0 or int(cu_q[-1].item()) != q.shape[1]:
        raise ValueError("Varlen boundaries do not cover the packed token stream.")

    # npu_fusion_attention expects cumulative END positions, without leading 0.
    actual_seq = tuple(int(v) for v in cu_q[1:].detach().cpu().tolist())
    q = q.squeeze(0).to(torch.bfloat16)
    k = k.squeeze(0).to(torch.bfloat16)
    v = v.squeeze(0).to(torch.bfloat16)
    num_heads_q = q.shape[-2]
    head_dim = q.shape[-1]
    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        num_heads_q,
        "TND",
        pse=None,
        padding_mask=None,
        atten_mask=None,
        actual_seq_qlen=actual_seq,
        actual_seq_kvlen=actual_seq,
        scale=head_dim**-0.5,
        keep_prob=1.0,
        inner_precise=0,
        sparse_mode=0,
    )[0]


def _npu_ring_attention_tnd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ring_split_lens: torch.Tensor,
) -> torch.Tensor:
    """Block-isolated full attention across the ring group, TND layout.

    q/k/v hold this ring rank's contiguous chunk of every packed sample, shaped
    ``[1, T_local, N, D]``. ``ring_split_lens`` (``[ring_size, num_samples]``) is
    how the ring rebuilds each peer's ``actual_seq_qlen`` while KV rotates, which
    is what keeps attention from crossing a sample boundary without a mask — the
    same role ``VarlenHandler.cu_seqlens_*`` plays on the single-rank TND path.

    MagiHuman's self-attention is bidirectional, so ``is_causal`` is False and the
    general TND strategy applies (the framework's causal TND path raises).
    Reuses ``ops.flash_attn.do_ring_attention`` so the cp_para wiring stays
    identical to every other model in the repo.
    """
    from mindspeed_mm.fsdp.ops.flash_attn.flash_attn import do_ring_attention

    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError(f"Expected q in [1, T, N, D], got {tuple(q.shape)}.")

    num_heads_q = q.shape[-2]
    head_dim = q.shape[-1]
    # [1, T, N, D] -> [T, N, D]: the ring kernel consumes bare TND.
    q, k, v = (t.squeeze(0).to(torch.bfloat16) for t in (q, k, v))
    return do_ring_attention(
        q,
        k,
        v,
        num_heads_q,
        softmax_scale=head_dim**-0.5,
        is_causal=False,
        fa_layout="TND",
        seq_split_lens=ring_split_lens,
    )  # [T_local, num_heads_q, head_dim]


def npu_flash_attn_with_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_split_sizes: "CPSplitPlan | None",
    varlen_handler: Any | None = None,
) -> torch.Tensor:
    """NPU replacement for ``dit_module.flash_attn_with_cp``.

    Upstream contract (dit_module.py:506-524):
      * inputs q,k,v have a leading batch dim of 1, shape [1, S, num_heads, head_dim]
        (GQA: num_heads_kv=8 for k/v); cast to bf16.
      * when cp world size == 1 it is exactly
        ``torch.ops.infra.flash_attn_func(q, k, v).squeeze(0)`` ->
        returns [S, num_heads_q, head_dim].

    For one sample the validated BSND path is used. For multiple concatenated
    samples, TND varlen attention uses ``VarlenHandler.cu_seqlens_*`` to prevent
    cross-sample attention.

    Context parallel (``cp_split_sizes`` is a :class:`~.modified.CPSplitPlan`):
    q/k/v arrive holding this rank's token slice and every head.

      * Ulysses: an all-to-all turns that into the ring rank's whole token stream
        with a head slice, so attention sees complete sequences and ``cu_seqlens``
        stays valid unchanged; the inverse all-to-all restores
        ``[S_local, num_heads_q, head_dim]`` for the caller. This mirrors upstream's
        ``batch_scatter_head_gather_seqlen`` / ``scatter_seqlen_gather_head`` pair
        using the framework's NPU collectives.
      * Ring: no gather happens — each rank keeps its own token chunk and the ring
        kernel rotates KV around the group instead, so the sequence is never
        materialised in full on any rank.

    With both on (USP) the ulysses all-to-all runs first and ring attention then
    operates on the head slice, matching the order in
    ``ops/flash_attn/flash_attn.py::flash_attention_forward``.
    """
    if cp_split_sizes is None:
        return _attention_dispatch(q, k, v, varlen_handler)

    from mindspeed_mm.fsdp.distributed.context_parallel.communication import all_to_all
    from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state

    parallel_state = get_parallel_state()
    ulysses_enabled = parallel_state.is_ulysses_enable()
    num_heads_q = q.shape[-2]

    if ulysses_enabled:
        cp_group = parallel_state.get_ulysses_group()
        cp_size = parallel_state.get_ulysses_group_size()
        num_heads_kv = k.shape[-2]
        if num_heads_q % cp_size or num_heads_kv % cp_size:
            raise ValueError(
                f"ulysses size {cp_size} must divide both head counts "
                f"(q={num_heads_q}, kv={num_heads_kv})."
            )
        # [1, S_local, H, D] -> [1, S_ring, H // cp, D]
        q, k, v = (
            all_to_all(
                t,
                cp_group,
                scatter_dim=2,
                gather_dim=1,
                gather_size=cp_split_sizes.ulysses_gather_size,
            )
            for t in (q, k, v)
        )

    if cp_split_sizes.ring_split_lens is None:
        out = _attention_dispatch(q, k, v, varlen_handler)  # [S_ring, H // cp, D]
    else:
        out = _npu_ring_attention_tnd(q, k, v, cp_split_sizes.ring_split_lens)

    if ulysses_enabled:
        # [1, S_ring, H // cp, D] -> [1, S_local, H, D]
        out = all_to_all(
            out.unsqueeze(0),
            cp_group,
            scatter_dim=1,
            gather_dim=2,
            gather_size=num_heads_q,
        ).squeeze(0)
    return out


def _attention_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    varlen_handler: Any | None,
) -> torch.Tensor:
    """Pick block-isolated TND attention for packed batches, BSND otherwise."""
    if varlen_handler is not None and varlen_handler.cu_seqlens_q.numel() > 2:
        return _npu_attention_tnd(q, k, v, varlen_handler)
    return _npu_attention_bsnd(q, k, v).squeeze(0)  # [S, num_heads_q, head_dim]


def npu_flex_flash_attn_with_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    cp_split_sizes: List[int],
) -> torch.Tensor:
    """NPU replacement for ``dit_module.flex_flash_attn_with_cp``.

    Only used by local-attn / SR layers (``local_attn_layers``), which are EMPTY
    for the base MagiHuman model — so this is NOT needed for first bring-up.

    NOTE (future work): only required if/when local-attention layers are enabled.
      Implement arbitrary (q_ranges, k_ranges) masked attention on NPU, e.g. by
      constructing a block-diagonal attention mask from the ranges and calling
      torch_npu.npu_fusion_attention with atten_mask, or by looping ranges and
      doing online-softmax LSE correction like the upstream pure-torch fallback
      `_custom_flex_flash_attn_func` (dit_module.py:464-468). Left unimplemented
      on purpose — raise to catch accidental use before it's needed.
    """
    raise NotImplementedError(
        "npu_flex_flash_attn_with_cp: local-attn path not needed for the base "
        "model (local_attn_layers=[]); implement only when SR/local layers are on."
    )


# ---------------------------------------------------------------------------
# 4. Make the linears trainable (inference-only custom Function lacks backward).
# ---------------------------------------------------------------------------
class _DifferentiableBF16Linear:
    """Autograd-traceable drop-in for upstream ``_BF16ComputeLinear``.

    ``.apply`` matches the upstream signature
    ``(input, weight, bias, output_dtype, compute_dtype=bf16)`` and performs the
    same math (``linear(input.bf16, weight.bf16, bias) -> output_dtype``) but as
    plain autograd ops, so gradients flow.
    """

    @staticmethod
    def apply(input, weight, bias, output_dtype, compute_dtype=torch.bfloat16):
        out = torch.nn.functional.linear(
            input.to(compute_dtype),
            weight.to(compute_dtype),
            bias.to(compute_dtype) if bias is not None else None,
        )
        return out.to(output_dtype)


def patch_linear_for_training() -> None:
    """Replace upstream ``_BF16ComputeLinear`` so ``loss.backward()`` works.

    Upstream ``BaseLinear`` / ``NativeMoELinear`` (dit_module.py:318-375) route
    every matmul through ``_BF16ComputeLinear`` (dit_module.py:290), a
    ``torch.autograd.Function`` that implements ONLY ``forward`` (inference-only).
    In training ``loss.backward()`` raises
    ``NotImplementedError: You must implement either the backward or vjp method``.
    The op is just ``linear(input.bf16, weight.bf16, bias) -> output_dtype``; we
    swap it for :class:`_DifferentiableBF16Linear` whose ``.apply`` is a plain
    autograd-traceable ``F.linear``. ``BaseLinear.forward`` /
    ``NativeMoELinear.forward`` resolve ``_BF16ComputeLinear`` from the module
    global at call time, so rebinding the module attribute is sufficient. Import-
    safe no-op when the upstream tree isn't vendored yet.
    """
    try:
        mod = importlib.import_module("inference.model.dit.dit_module")
    except Exception:  # upstream not vendored yet; import-safe no-op
        return
    if hasattr(mod, "_BF16ComputeLinear"):
        mod._BF16ComputeLinear = _DifferentiableBF16Linear


# ---------------------------------------------------------------------------
# 5. Replace RoPE / RMSNorm element-wise stacks with torch_npu fused kernels.
# ---------------------------------------------------------------------------
# Upstream runs both in plain PyTorch inside every one of the 40 DiT layers:
#
#   dit_module.apply_rotary_emb_torch  -> Slice x3, chunk, Neg, Cat x2, Mul x2, Add
#   MultiModalityRMSNorm.rms           -> Pow, Mean, Add, Rsqrt, Mul
#
# Both have single-kernel NPU equivalents that the rest of this repository
# already uses (mindspeed_mm/fsdp/ops/npu_patch/npu_fused_operator.py); MagiHuman
# was the only model not wired up to them.
#
# Measured on Ascend910B3 at the official L=4606 shape, fp32 (the dtype the model
# actually uses here: `linear_qkv(...).to(torch.float32)` and `rms()` casts with
# `x.float()`):
#
#   RoPE      q[1,4606,40,128]  naive 0.880 ms -> fused 0.450 ms  (1.96x), fwd+bwd 1.83x
#             max_abs difference 0.000e+00 (bit-exact)
#   RMSNorm   x[4606,40,128]    naive 0.422 ms -> fused 0.131 ms  (3.24x), bit-exact
#             x[4606,5120]      naive 0.416 ms -> fused 0.135 ms  (3.08x), max_abs 9.5e-07
#
# Note: `npu_fused_operator.apply_transformers_rope_half_npu` guards the fused
# path behind `batch*seq <= 128` and otherwise falls back to the eager stack.
# That guard is not right for this workload -- at seq 4606 the fused kernel is
# 1.9x faster -- so this patch does not reproduce it.

_ORIG_APPLY_ROTARY_EMB = None
_ORIG_RMS = None
_ORIG_FWD_SINGLE = None


def _fused_ops_available() -> bool:
    """Whether the fused NPU kernels can be used (i.e. ``torch_npu`` imported)."""

    return torch_npu is not None


def npu_apply_rotary_emb(x, cos, sin, interleaved: bool = False):
    """``npu_rotary_mul`` drop-in for ``dit_module.apply_rotary_emb_torch``.

    Handles MagiHuman's PARTIAL rotary: the Fourier rope is 96 wide while
    head_dim is 128, so only ``x[..., :96]`` is rotated and the 32-wide tail is
    passed through unchanged -- exactly as upstream does.

    ``npu_rotary_mul`` implements the same ``half`` convention as upstream's
    ``rotate_half`` (chunk into two, ``[-x2, x1]``), so cos/sin are tiled to the
    full rotary width, matching upstream's ``repeat(cos, "... d -> ... 1 (2 d)")``.

    Falls back to the original for interleaved mode or any unexpected rank.
    """
    if not _fused_ops_available() or interleaved:
        return _ORIG_APPLY_ROTARY_EMB(x, cos, sin, interleaved)
    if cos.dim() != 2 or x.dim() != 4:  # only the shape the DiT actually produces
        return _ORIG_APPLY_ROTARY_EMB(x, cos, sin, interleaved)

    ro_dim = cos.shape[-1] * 2
    if ro_dim > x.shape[-1]:
        raise ValueError(f"rotary dim {ro_dim} exceeds head dim {x.shape[-1]}")

    cos_e = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(-2)  # [1, L, 1, ro]
    sin_e = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(-2)

    if ro_dim == x.shape[-1]:
        return torch_npu.npu_rotary_mul(x.contiguous(), cos_e, sin_e)
    rotated = torch_npu.npu_rotary_mul(x[..., :ro_dim].contiguous(), cos_e, sin_e)
    return torch.cat([rotated, x[..., ro_dim:]], dim=-1)


def _npu_rms(self, x: torch.Tensor) -> torch.Tensor:
    """Normalization-only fused path for ``MultiModalityRMSNorm.rms``.

    ``rms()`` returns the normalized tensor WITHOUT the weight, because
    ``forward_multi_experts`` applies a different weight chunk per modality
    afterwards. ``npu_rms_norm`` requires a gamma, so a cached all-ones vector is
    used to get pure normalization.
    """
    # ``aclnnRmsNormGrad`` fails on a zero-row input, which Ulysses CP produces:
    # a rank whose token slice contains no VIDEO/AUDIO/TEXT token at all feeds an
    # empty tensor into that modality's expert. Eager handles it correctly.
    if not _fused_ops_available() or x.numel() == 0:
        return _ORIG_RMS(self, x)
    gamma = getattr(self, "_npu_ones_gamma", None)
    if gamma is None or gamma.shape[-1] != x.shape[-1] or gamma.device != x.device:
        gamma = torch.ones(x.shape[-1], device=x.device, dtype=torch.float32)
        self._npu_ones_gamma = gamma
    return torch_npu.npu_rms_norm(x.float(), gamma, epsilon=self.eps)[0]


def _npu_forward_single_expert(self, x: torch.Tensor, modality_dispatcher=None) -> torch.Tensor:
    """Fused single-expert RMSNorm: normalization and ``weight + 1`` in one kernel."""
    if not _fused_ops_available() or x.numel() == 0:  # see _npu_rms on empty slices
        return _ORIG_FWD_SINGLE(self, x, modality_dispatcher)
    original_dtype = x.dtype
    out = torch_npu.npu_rms_norm(x.float(), (self.weight + 1).float(), epsilon=self.eps)[0]
    return out.to(original_dtype)


def patch_fused_elementwise_ops() -> None:
    """Rebind RoPE and RMSNorm onto fused NPU kernels.

    ``MultiModalityRMSNorm.__init__`` binds ``self.forward`` to one of the two
    expert methods as an INSTANCE attribute, so the class attributes must be
    replaced before any module is constructed. ``apply_npu_patches`` runs from
    ``modeling_magihuman._import_dit``, i.e. at import time, which satisfies that.
    """
    global _ORIG_APPLY_ROTARY_EMB, _ORIG_RMS, _ORIG_FWD_SINGLE

    try:
        mod = importlib.import_module("inference.model.dit.dit_module")
    except Exception:  # upstream not vendored yet; import-safe no-op
        return
    if torch_npu is None:
        return

    if hasattr(mod, "apply_rotary_emb_torch") and _ORIG_APPLY_ROTARY_EMB is None:
        _ORIG_APPLY_ROTARY_EMB = mod.apply_rotary_emb_torch
        mod.apply_rotary_emb_torch = npu_apply_rotary_emb

    norm_cls = getattr(mod, "MultiModalityRMSNorm", None)
    if norm_cls is not None and _ORIG_RMS is None:
        _ORIG_RMS = norm_cls.rms
        _ORIG_FWD_SINGLE = norm_cls.forward_single_expert
        norm_cls.rms = _npu_rms
        norm_cls.forward_single_expert = _npu_forward_single_expert


# ---------------------------------------------------------------------------
# 6. De-interleave the gated MLP so SwiGLU splits contiguous halves.
# ---------------------------------------------------------------------------
# Upstream `swiglu7` splits the up_gate_proj output with a STRIDE-2 interleaved
# slice, `x[..., ::2]` / `x[..., 1::2]`, on an fp32 [L, 2*intermediate] tensor.
# Materializing those views is pathologically slow: each 512-byte cache line
# delivers only half its bytes. Profiling the 8-card full-parameter run at
# L=4606 measured, for one step:
#
#   ViewCopy    [4606,13652]  x72   1767.3 ms   (26.9% of the 6578.8 ms step)
#   StridedSlice[4606,27304]  x252   273.3 ms   ( 4.2%)
#
# ~30 GB/s effective, against >1 TB/s HBM. 72 calls == 36 gated layers x 2 slices
# (layers 0-3 are GELU7 / non-gated, per config `gelu7_layers=[0,1,2,3]`).
#
# Fix: permute the up_gate_proj OUTPUT ROWS once at load, [g0,l0,g1,l1,...] ->
# [g0..gN, l0..lN], then split with `chunk(2, -1)`, which yields contiguous views
# needing no copy. Output channel i maps to (gate i, linear i) either way, so
# `down_proj` is unaffected and the math is exactly equivalent.
#
# Both halves of this change MUST be applied together. Runtime loading and the
# HF/DCP converter share the inverse transforms in ``weight_layout.py``. DCP is
# the optimized internal format; exported HF safetensors are converted back to
# the upstream interleaved format by ``MagiHumanConverter``.

_ORIG_SWIGLU7 = None


def _swiglu7_chunk(x, alpha: float = 1.702, limit: float = 7.0, out_dtype=None):
    """`swiglu7` over contiguous halves. Identical math to upstream's stride-2 form."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu, x_linear = x.chunk(2, dim=-1)
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return (out_glu * (x_linear + 1)).to(out_dtype)


def patch_swiglu7_deinterleaved() -> None:
    """Rebind `dit_module.swiglu7` before any MLP is constructed.

    `MLP.__init__` calls `create_activation_func`, which resolves `swiglu7` as a
    module global at call time, so rebinding here (import time, from
    `_import_dit`) is picked up by every layer built afterwards.
    """
    global _ORIG_SWIGLU7
    try:
        mod = importlib.import_module("inference.model.dit.dit_module")
    except Exception:
        return
    if hasattr(mod, "swiglu7") and _ORIG_SWIGLU7 is None:
        _ORIG_SWIGLU7 = mod.swiglu7
        mod.swiglu7 = _swiglu7_chunk
