"""FSDP2 training wrapper for the MagiHuman 15B single-stream DiT.

MagiHuman jointly denoises a *single packed token sequence* of VIDEO + AUDIO
latents conditioned on TEXT embeddings (sandwich: first/last 4 layers are
modality-specific MoE, middle 32 layers shared). This wrapper adapts the
upstream *inference-only* `DiTModel` (which returns a single velocity tensor and
has NO loss) to MindSpeed-MM's contract: ``model(**batch)`` returns an object
with ``.loss`` (config sets ``features.loss_cfg.loss_type: raw`` so the model
owns the loss).

Structure mirrors ``mindspeed_mm/fsdp/models/ltx2/ltx2_fsdp2/modeling_ltx2.py``:
  * ``@model_register.register("magihuman")`` on the wrapper class.
  * ``from_pretrained`` (real-weights, CPU/fp32).
  * ``_from_config`` (from-scratch / meta-device).
  * ``forward(**batch) -> MagiHumanModelOutput(loss=...)``.
  * monkey-patches applied in ``__init__`` (see ``modified.py`` / ``npu_patch.py``).

The upstream DiT internals (40-layer sandwich, MoE linears, custom attention)
are NOT reimplemented here; they are imported from the vendored upstream package
(``inference/``, see the wiring notes below) and patched for NPU training.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# torch_npu is imported purely to register NPU dispatch (SDPA etc.), mirroring
# ltx2. The explicit attention op replacement lives in ``npu_patch.py`` because
# MagiHuman's upstream calls CUDA flash-attn / magi_attention directly (unlike
# ltx2 which uses plain SDPA). Guarded so the module still imports off-NPU.
try:  # pragma: no cover - NPU-only
    import torch_npu  # 注册 NPU 算子（副作用导入）
except Exception:
    torch_npu = None

from mindspeed_mm.fsdp.models.base_model import BaseModel
from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.utils.register import model_register


# ---------------------------------------------------------------------------
# Vendored-upstream wiring (lazy).
#
# The upstream MagiHuman code uses absolute imports rooted at the package name
# ``inference`` (e.g. ``from inference.common import Modality, VarlenHandler``).
# Following the ltx2 ``sys.modules["ltx_core"] = ...`` pattern, we vendor the
# upstream ``inference/`` tree into this plugin dir and alias it so those
# absolute imports resolve. The alias + heavy import are LAZY (done at build
# time, not import time) so that:
#   (1) the @model_register decorator below still fires at plugin-import time
#       even before the operator has vendored the upstream tree, and
#   (2) we never import the CUDA-only `magi_compiler` / flash-attn stack until
#       npu_patch has had a chance to neutralize it.
#
# The upstream tree is vendored as a setup step (NOT committed — kept pristine
# and re-syncable, exactly like ltx2's untracked ltx_core/); see
# examples/magihuman/README.md 环境安装 for the clone/checkout/cp/sed commands.
# Minimum subtree needed for DiTModel:
#     inference/model/dit/dit_module.py   (DiTModel + all nn.Modules)
#     inference/common/{__init__,sequence_schema,arch,config}.py
#                                         (Modality, VarlenHandler, is_hopper_arch)
#     inference/infra/parallelism/        (ulysses_scheduler, all_to_all_primitive)
# Do NOT vendor inference/model/dit/dit_model.py::get_dit — it pulls the CUDA
# checkpoint/distributed stack; we build DiTModel directly (see _import_dit).
# ---------------------------------------------------------------------------
_UPSTREAM_PKG = "inference"
_VENDORED_PKG = "mindspeed_mm.fsdp.models.magihuman.inference"


def _alias_upstream() -> None:
    """Alias the vendored upstream tree under its original top-level name."""
    if _UPSTREAM_PKG not in sys.modules:
        sys.modules[_UPSTREAM_PKG] = importlib.import_module(_VENDORED_PKG)


def _import_dit():
    """Lazily import the upstream `DiTModel` class (after NPU patching).

    Returns the `DiTModel` type. Imported here (not at module top) so plugin
    registration does not depend on the upstream tree being present yet.
    """
    _alias_upstream()

    # Neutralize CUDA-only build-time machinery BEFORE importing dit_module so
    # the class bodies / decorators don't blow up on NPU:
    #     * inference/model/dit/dit_module.py::magi_compile decorator on
    #       `TransformerBlock` (from magi_compiler) — proprietary CUDA-graph
    #       compiler; must be made a no-op. See npu_patch.disable_magi_compile().
    #     * the three `@magi_register_custom_op` attention ops (flash_attn_func,
    #       flex_flash_attn_func, flash_attn_with_cp / flex_flash_attn_with_cp)
    #       — replace FA2/FA3/magi_attention with torch_npu.npu_fusion_attention.
    #       See npu_patch.patch_attention_ops().
    from mindspeed_mm.fsdp.models.magihuman.magihuman_fsdp2 import npu_patch

    npu_patch.apply_npu_patches()  # no-op-safe when torch_npu is unavailable

    from inference.model.dit.dit_module import DiTModel  # 延迟导入：构建时才需要 vendored 上游

    return DiTModel


@dataclass
class MagiHumanModelOutput:
    """Return type read by train_engine (only ``.loss`` is consumed)."""

    loss: torch.Tensor
    pred: torch.Tensor | None = None


class _DiTConfig:
    """Lightweight attribute bag matching the fields upstream `DiTModel.__init__`
    and `TransFormerLayer.__init__` read off `model_config`.

    Upstream uses a pydantic `inference.common.config.ModelConfig`; here we build
    a plain object from the YAML ``model.transformer`` sub-dict so we don't drag
    in pydantic / the full inference config stack. Defaults are the verified 15B
    values from inference/common/config.py::ModelConfig.
    """

    # Verified defaults (inference/common/config.py::ModelConfig).
    num_layers = 40
    hidden_size = 5120
    head_dim = 128
    num_query_groups = 8  # GQA kv groups
    video_in_channels = 48 * 4  # 192 (patchified Wan2.2 z_dim=48, patch (1,2,2))
    audio_in_channels = 64
    text_in_channels = 3584
    checkpoint_qk_layernorm_rope = False
    params_dtype = torch.float32
    mm_layers = (0, 1, 2, 3, 36, 37, 38, 39)  # modality-specific (num_modality=3)
    local_attn_layers = ()  # base model: empty -> full-attention path only
    enable_attn_gating = True
    gelu7_layers = (0, 1, 2, 3)
    post_norm_layers = ()
    # Computed (upstream post_override_config): hidden_size // head_dim, num_query_groups.
    num_heads_q = 0
    num_heads_kv = 0

    def __init__(self, overrides: dict[str, Any]):
        for key, value in overrides.items():
            setattr(self, key, value)
        # Replicate upstream computed fields (config.py:222-223).
        self.num_heads_q = self.hidden_size // self.head_dim
        self.num_heads_kv = self.num_query_groups
        # Upstream reads list-membership (`idx in config.mm_layers`); tuples work.


# Modality IntEnum values (= inference.common.sequence_schema.Modality:
# VIDEO=0, AUDIO=1, TEXT=2; verified against the vendored upstream).
_MOD_VIDEO = 0
_MOD_AUDIO = 1


def _magihuman_shard_files(ckpt_path: str):
    """Resolve the list of weight-shard files for a MagiHuman checkpoint.

    Handles an HF-sharded directory (``model.safetensors.index.json``), a directory
    of ``*.safetensors`` / ``*.pt`` / ``*.bin``, or a single weight file.
    """
    p = Path(ckpt_path)
    if p.is_dir():
        index = p / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            return [p / s for s in sorted(set(weight_map.values()))]
        safes = sorted(p.glob("*.safetensors"))
        if safes:
            return safes
        bins = sorted(p.glob("*.pt")) + sorted(p.glob("*.bin"))
        if bins:
            return bins
        raise FileNotFoundError(f"No safetensors/pt/bin weights under {p}")
    return [p]


def _load_magihuman_weights_into(dit: torch.nn.Module, ckpt_path: str):
    """Copy MagiHuman base weights into ``dit`` IN PLACE, shard by shard.

    Loads each shard, copies matching keys into the live module tensors, then frees
    the shard — peak host memory stays ~(model + one shard), not 2x the model (which
    matters with one CPU build per DDP rank). Returns ``(missing, unexpected)`` key
    lists. CPU-only; no CUDA/distributed coupling.
    """
    from safetensors.torch import load_file

    model_sd = dit.state_dict()  # live tensors (shared storage) -> copy_ writes params
    model_keys = set(model_sd.keys())
    loaded: set = set()
    unexpected: list = []
    for shard in _magihuman_shard_files(ckpt_path):
        shard_sd = load_file(str(shard), device="cpu") if str(shard).endswith(".safetensors") \
            else torch.load(str(shard), map_location="cpu", weights_only=True)
        for k, v in shard_sd.items():
            if k in model_keys:
                model_sd[k].copy_(v.to(model_sd[k].dtype))
                loaded.add(k)
            else:
                unexpected.append(k)
        del shard_sd
    missing = sorted(model_keys - loaded)
    return missing, unexpected


@model_register.register("magihuman")
class MagiHumanForTraining(torch.nn.Module, BaseModel):
    """FSDP2 training wrapper for MagiHuman's DiT.

    Adapts the upstream single-stream `DiTModel`
        forward(x, coords_mapping, modality_mapping, varlen_handler,
                local_attn_handler) -> x_out [L, 192]
    to ``model(**batch) -> MagiHumanModelOutput(loss=...)`` by computing the
    flow-matching MSE between the predicted velocity ``x_out`` (video+audio rows
    only) and the target velocity supplied in the batch by the dataset's
    ``collate_fn`` (text rows are condition-only and never denoised).
    """

    def __init__(self, dit: torch.nn.Module):
        super().__init__()
        self.dit = dit

        # The public checkpoint is rooted at ``block.*`` while this wrapper
        # owns the DiT under ``dit``.  The generic online HF loader already
        # honors this repository-wide key-mapping convention.
        self._checkpoint_conversion_mapping = {r"^(.*)$": r"dit.\1"}

        # Per-modality output channel counts for the channel-aware flow-matching
        # loss (audio velocity occupies only cols[:audio_ch]; cols[audio_ch:] are
        # zeros upstream). Read off the built DiT config with verified 15B defaults.
        cfg = getattr(dit, "model_config", None)
        self._video_ch = int(getattr(cfg, "video_in_channels", 192))
        self._audio_ch = int(getattr(cfg, "audio_in_channels", 64))

        # FSDP2 / activation-recompute friendly forward rewrites, applied by
        # MethodType rebind (ltx2 pattern). Kept in modified.py so upstream stays
        # pristine. dit_forward strips the ulysses dispatch (identity at cp=1) from
        # DiTModel.forward; block_forward is a clean standalone copy of
        # TransFormerLayer.forward bound onto each dit.block.layers[*] (a stable
        # per-layer recompute boundary, independent of the no-op @magi_compile
        # decorator on the parent TransformerBlock). Guarded so we only rebind
        # what the built DiT actually exposes.
        from mindspeed_mm.fsdp.models.magihuman.magihuman_fsdp2.modified import (
            attention_forward,
            block_forward,
            dit_forward,
        )

        self.dit.forward = types.MethodType(dit_forward, self.dit)
        block = getattr(self.dit, "block", None)
        layers = getattr(block, "layers", None)
        if layers is not None:
            for layer in layers:
                layer.forward = types.MethodType(block_forward, layer)
                layer.attention.forward = types.MethodType(
                    attention_forward, layer.attention
                )

    # ------------------------------------------------------------------ build
    @staticmethod
    def _to_mapping(obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return {}

    @classmethod
    def _build_dit_from_config(cls, config: ModelArguments) -> torch.nn.Module:
        """Instantiate an un-weighted `DiTModel` on CPU/meta from the YAML
        ``model.transformer`` sub-dict.
        """
        DiTModel = _import_dit()
        transformer_cfg = cls._to_mapping(getattr(config, "transformer", {}))
        model_config = _DiTConfig(transformer_cfg)
        return DiTModel(model_config=model_config)

    @classmethod
    def _from_config(cls, config: ModelArguments) -> "MagiHumanForTraining":
        """Meta-device / from-scratch path (`init_model_with_meta_device: true`)."""
        dit = cls._build_dit_from_config(config)
        if bool(getattr(config, "enable_gradient_checkpointing", False)):
            # Upstream DiTModel has no set_gradient_checkpointing; recompute is
            # driven by features.recompute_plan.apply_modules in the YAML instead.
            # No-op here (FSDP2 wraps + recomputes the block.layers.{*} modules).
            pass
        return cls(dit=dit)

    @classmethod
    def from_pretrained(cls, config: ModelArguments) -> "MagiHumanForTraining":
        """Build on CPU/fp32 and load the MagiHuman base checkpoint."""
        ckpt_path = getattr(config, "checkpoint_path", None) or config.model_name_or_path
        if ckpt_path is None:
            raise ValueError("`model_name_or_path` or `checkpoint_path` must be provided for MagiHuman.")
        ckpt_path = str(Path(ckpt_path).expanduser())

        dit = cls._build_dit_from_config(config)

        # Load the base weights onto the freshly built DiTModel. Upstream loads via
        # inference.infra.checkpoint.load_model_checkpoint (CUDA/distributed-coupled);
        # we instead load the raw state_dict on CPU. The vendored DiTModel is the same
        # class the checkpoint was saved from, so every key must line up.
        # NativeMoELinear / MultiModalityRMSNorm store per-modality weights as one
        # fused Parameter, matching the pinned upstream checkpoint layout.
        missing, unexpected = _load_magihuman_weights_into(dit, ckpt_path)
        n_model = len(dit.state_dict())
        logger.info(
            "MagiHuman weights: loaded %d/%d params from %s (missing=%d, unexpected=%d)",
            n_model - len(missing), n_model, ckpt_path, len(missing), len(unexpected),
        )
        if missing:
            logger.warning("MagiHuman load: %d MISSING keys, e.g. %s", len(missing), list(missing)[:6])
        if unexpected:
            logger.warning("MagiHuman load: %d UNEXPECTED ckpt keys, e.g. %s", len(unexpected), list(unexpected)[:6])
        if missing or unexpected:
            raise RuntimeError(
                "MagiHuman checkpoint does not exactly match the pinned DiT structure: "
                f"missing={len(missing)}, unexpected={len(unexpected)}."
            )

        # Convert the upstream interleaved gated-MLP rows to the optimized
        # contiguous-halves training layout. This must run HERE: weights are
        # loaded, the model is on CPU, and FSDP2 has not sharded it yet.
        from .weight_layout import deinterleave_gated_mlp_weights_

        converted = deinterleave_gated_mlp_weights_(dit)
        logger.info(
            "MagiHuman: converted %d gated MLP layers to the internal contiguous layout",
            converted,
        )

        if bool(getattr(config, "enable_gradient_checkpointing", False)):
            pass  # see _from_config note

        return cls(dit=dit)

    # ---------------------------------------------------------------- forward
    @staticmethod
    def _build_varlen_handler(
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
        varlen_handler: Any | None,
    ) -> Any:
        """Rebuild upstream's handler from flat batch tensors.

        ``move_to_device`` only preserves/moves top-level tensors, so the
        dataset cannot safely return a nested ``VarlenHandler`` dataclass.
        Keeping the cumulative boundaries in the batch also makes the packed
        sample contract visible to the attention implementation.
        """
        if varlen_handler is not None:
            return varlen_handler

        if cu_seqlens is None:
            # Backward-compatible single-sample path for direct model calls.
            cu_seqlens = torch.tensor(
                [0, x.shape[0]], dtype=torch.int32, device=x.device
            )
        else:
            cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32)

        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError(
                "cu_seqlens must be a 1-D tensor with at least [0, L]."
            )
        if int(cu_seqlens[0].item()) != 0:
            raise ValueError("cu_seqlens must start at 0.")
        if int(cu_seqlens[-1].item()) != x.shape[0]:
            raise ValueError(
                f"cu_seqlens ends at {int(cu_seqlens[-1].item())}, "
                f"but packed x has length {x.shape[0]}."
            )

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if bool((lengths <= 0).any().item()):
            raise ValueError("cu_seqlens must describe non-empty samples.")

        inferred_max = int(lengths.max().item())
        if max_seqlen is None:
            resolved_max = inferred_max
        elif isinstance(max_seqlen, torch.Tensor):
            resolved_max = int(max_seqlen.item())
        else:
            resolved_max = int(max_seqlen)
        if resolved_max != inferred_max:
            raise ValueError(
                f"max_seqlen={resolved_max} does not match packed boundaries "
                f"(expected {inferred_max})."
            )

        from inference.common import VarlenHandler

        return VarlenHandler(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=resolved_max,
            max_seqlen_k=resolved_max,
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        target_velocity: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | int | None = None,
        varlen_handler: Any | None = None,
        local_attn_handler: Any | None = None,
        **_: Any,
    ) -> MagiHumanModelOutput:
        """Run one packed denoising step and return the flow-matching loss.

        Args (produced by `MagiHumanPrecomputedDataset.collate_fn`, which is
        net-new code — see plan §"forward/loss wiring"):
          x:                [L, 3584] packed noised token sequence at width
                              text_in_channels(=3584): video rows fill cols[:192],
                              audio rows fill cols[:64], text rows fill cols[:3584].
                              Upstream Adapter slices x[mask, :ch] per modality
                              (dit_module.py:735) — only pred/target_velocity are
                              [L, 192]. The dataset/collator builds
                              x_t = sigma-interp(clean_latent, noise). The
                              upstream model forward has no timestep argument;
                              its inference scheduler keeps time outside DiT.
          coords_mapping:   [L, 9]    (t,row,col, T,H,W, ref_T,ref_H,ref_W) RoPE coords
          modality_mapping: [L]       Modality IntEnum (VIDEO=0/AUDIO=1/TEXT=2)
          target_velocity:  [L, 192]  flow-matching target (e.g. noise - x0),
                              meaningful on video+audio rows only.
          loss_mask:        [L]       True on denoised (video+audio) rows; text=False.
          cu_seqlens:       [B+1]     cumulative sample boundaries in packed x.
          max_seqlen:       []        longest packed sample in the batch.
          varlen_handler:              optional direct-call override; normal
                              dataloader calls rebuild it from the two flat
                              tensors above after device movement.
          local_attn_handler:          FFA ranges; unused by the base model
                              because ``local_attn_layers`` is empty.

        Upstream `DiTModel.forward` (dit_module.py:902-950) returns a single
        x_out [L, 192]: video rows -> final_norm_video/final_linear_video (192ch),
        audio rows -> final_norm_audio/final_linear_audio (64ch), text rows zero.
        """
        varlen_handler = self._build_varlen_handler(
            x, cu_seqlens, max_seqlen, varlen_handler
        )

        # Upstream forward signature: self.dit(x, coords_mapping, modality_mapping,
        # varlen_handler, local_attn_handler). modified.py neutralizes the
        # ulysses dispatch/undispatch identity wrapping and the magi_compile path,
        # so this call is a 1:1 passthrough.
        pred = self.dit(
            x,
            coords_mapping,
            modality_mapping,
            varlen_handler,
            local_attn_handler,
        )
        pred = pred.float()

        loss = self._flow_matching_loss(
            pred,
            target_velocity.float(),
            modality_mapping,
            loss_mask,
        )
        return MagiHumanModelOutput(loss=loss.float(), pred=pred)

    def _flow_matching_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        modality_mapping: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Channel-aware flow-matching MSE, computed per modality then summed
        (mirrors ltx2's separate video/audio loss).

        Avoids audio-loss dilution: the audio velocity occupies only
        ``cols[:audio_ch]`` (cols[audio_ch:video_ch] are hard zeros in both pred
        and target upstream), so averaging over all ``video_ch`` columns would
        shrink the audio term by ``audio_ch/video_ch``. We therefore restrict each
        modality's MSE to its own valid channels and rows. Text rows are excluded
        (condition-only, never denoised).
        """
        diff = (pred - target).pow(2)  # [L, video_ch]
        video_rows = modality_mapping == _MOD_VIDEO
        audio_rows = modality_mapping == _MOD_AUDIO
        if loss_mask is not None:
            if loss_mask.shape != modality_mapping.shape:
                raise ValueError(
                    f"loss_mask shape {tuple(loss_mask.shape)} must match "
                    f"modality_mapping shape {tuple(modality_mapping.shape)}."
                )
            valid_rows = loss_mask.to(device=modality_mapping.device, dtype=torch.bool)
            video_rows = video_rows & valid_rows
            audio_rows = audio_rows & valid_rows
        terms = []
        if bool(video_rows.any()):
            terms.append(diff[video_rows][:, : self._video_ch].mean())
        if bool(audio_rows.any()):
            terms.append(diff[audio_rows][:, : self._audio_ch].mean())
        if not terms:  # degenerate batch with no denoised rows
            return diff.sum() * 0.0
        return torch.stack(terms).sum()
