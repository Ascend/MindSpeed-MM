"""MagiHuman precomputed dataset for the FSDP2 backend (issue #357, guide §4).

Loads CLEAN precomputed latents (video + audio) and text-condition embeddings,
then in ``collate_fn`` does the flow-matching noising OUTSIDE the model (exactly
like ltx2 — MagiHuman has no internal noiser) and packs ONE single-stream token
sequence the upstream DiT consumes.

Registered as ``magihuman_precomputed`` to match
``examples/magihuman/magihuman_config_t2av.yaml`` (``data.dataset_param.dataset_type``).

Structure mirrors ``mindspeed_mm/fsdp/data/datasets/ltx2/ltx2_precomputed_dataset.py``
(the ``@data_register`` precomputed dataset + bound ``collate_fn`` + DP-aware
trimming). The key STRUCTURAL difference vs ltx2: the MagiHuman DiT forward takes
a SINGLE packed sequence ``x [L, 3584]`` (not a batched ``[B, S, C]``), so
``collate_fn`` CONCATENATES samples along the token axis (dim 0) instead of
stacking — naturally handling ragged per-sample lengths.

batch keys emitted by ``collate_fn`` (all FLAT top-level tensors, per guide §4.3
``move_to_device`` only moves top-level tensors / tensor-lists / scalars / None):

    x                [L, 3584] float  packed NOISED tokens (video[:192], audio[:64],
                                      text[:3584]); video+audio are noised, text is
                                      the clean condition (never noised).
    coords_mapping   [L, 9]    float32 (t,h,w, T,H,W, ref_T,ref_H,ref_W) RoPE coords
    modality_mapping [L]       int64  Modality ints VIDEO=0 / AUDIO=1 / TEXT=2
    target_velocity  [L, 192]  float  flow-matching target (eps - x0); audio in [:64]
    loss_mask        [L]       bool   True on video+audio rows, False on text
    cu_seqlens       [B+1]     int32  cumulative packed-sample boundaries
    max_seqlen       []        int32  longest packed sample in the batch

``x`` and ``target_velocity`` flatten each video token in a different order.
Upstream ``img2tokens`` packs a patch with ``UnfoldNd(kernel=(1, 2, 2))``, which
lays it out as ``(C, pT, pH, pW)``, while ``depack_token_sequence`` reads the model
output with ``rearrange(x, "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)")``,
i.e. as ``(pT, pH, pW, C)``. The pretrained input embedder and output head encode
exactly that asymmetry, so the noised input keeps the ``(C, pT, pH, pW)`` latent
while the video target is reordered by ``_channels_last_within_patch``. Random
synthetic latents are invariant to that reordering, so only real data exposes a
mismatch.

The dataset emits only flat tensors because ``move_to_device`` does not move
nested dataclasses (guide §4.3). The model wrapper reconstructs upstream's
``VarlenHandler`` from ``cu_seqlens`` / ``max_seqlen`` after device movement.
``local_attn_handler`` is not emitted because the base model has no local-attn
layers.

Importable off-NPU: no torch_npu / no vendored-upstream imports are needed here.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from mindspeed_mm.fsdp import envs
from mindspeed_mm.fsdp.utils.register import data_register

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modality ints. MUST stay consistent with the model wrapper, which hardcodes
# the same values (modeling_magihuman.py: ``_MOD_VIDEO = 0``, ``_MOD_AUDIO = 1``);
# the loss selects video/audio rows by these ints. Verified against the vendored
# ``inference.common.sequence_schema.Modality`` IntEnum (VIDEO=0, AUDIO=1,
# TEXT=2). The vendored tree is only present/aliased at model-build time, so we
# keep plain ints here to stay importable off-NPU.
# ---------------------------------------------------------------------------
_MOD_VIDEO = 0
_MOD_AUDIO = 1
_MOD_TEXT = 2

# Offset applied to ``seed + sample_index`` for the noising RNG so that it never
# overlaps the stream ``_make_synthetic_sample`` draws the clean latents from.
_NOISE_STREAM_OFFSET = 1_000_003

# Upstream audio RoPE "v2" magic: ref_T = (Na - 1) // AUDIO_REF_DOWNSAMPLE + 1.
# Lifted from the validated packed-input builder (magihuman_fwd_smoke.py).
_AUDIO_REF_DOWNSAMPLE = 4

# Upstream spatial patch size (``DataProxyConfig.patch_size``). With
# ``args.t_patch_size`` it gives the patch volume the video target is reordered in.
_SPATIAL_PATCH = 2


@dataclass
class MagiHumanPrecomputedBasicArgs:
    """Resolved dataset args (YAML ``basic_parameters`` + ``magihuman_dataset_custom``)."""

    # --- real precomputed layout (mirrors ltx2: {dataset_dir}/.precomputed/<sub>/*.pt) ---
    dataset_dir: str = "/data/magihuman"
    latents_dir: str = "latents"
    conditions_dir: str = "conditions"
    audio_latents_dir: str = "audio_latents"
    with_audio: bool = True
    max_samples: int | None = None
    seed: int = 42

    # --- per-modality channel widths (verified 15B defaults; see _DiTConfig) ---
    video_channels: int = 192  # video_in_channels (48 * 4)
    audio_channels: int = 64  # audio_in_channels
    text_channels: int = 3584  # text_in_channels == packed sequence width
    t_patch_size: int = 1

    # --- flow-matching sigma sampler (per-sample timestep) ---
    timestep_sampling_mode: Literal["uniform", "logit_normal"] = "logit_normal"
    timestep_sampling_params: dict[str, float] = field(default_factory=dict)

    # --- synthetic mode (E2E with NO real data; small + configurable) ---
    synthetic: bool = False
    synthetic_samples: int = 8
    synthetic_video_grid: Any = (1, 2, 2)  # patchified (T, H, W) -> Nv = T*H*W
    synthetic_audio_tokens: int = 2
    synthetic_text_tokens: int = 3


def _get_coords(
    shape: tuple[int, int, int],
    ref_feat_shape: tuple[int, int, int],
    offset_thw: tuple[int, int, int] = (0, 0, 0),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build [T*H*W, 9] = (t, h, w, T, H, W, ref_T, ref_H, ref_W).

    Exact reimplementation of upstream ``data_proxy.get_coords`` (CPU/float32),
    lifted from the validated smoke builder. Rows are emitted in row-major
    ``meshgrid(..., indexing="ij")`` order over (t, h, w) — patchified latent
    tokens MUST be flattened in this SAME order so token i aligns with row i.

    RoPE invariant (ElementWiseFourierEmbed): scale = (ref - 1) / (size - 1), and
    any axis with size == 1 is forced to scale 1 — so every size==1 axis must have
    ref == 1, else a real divide-by-zero -> inf -> assert. All callers below honor
    this (video uses ref == size; audio/text size==1 axes have ref==1).
    """
    ori_t, ori_h, ori_w = (int(shape[0]), int(shape[1]), int(shape[2]))
    ref_t, ref_h, ref_w = (int(ref_feat_shape[0]), int(ref_feat_shape[1]), int(ref_feat_shape[2]))
    off_t, off_h, off_w = offset_thw

    time_rng = torch.arange(ori_t, dtype=dtype) + off_t
    height_rng = torch.arange(ori_h, dtype=dtype) + off_h
    width_rng = torch.arange(ori_w, dtype=dtype) + off_w

    time_grid, height_grid, width_grid = torch.meshgrid(time_rng, height_rng, width_rng, indexing="ij")
    coords_flat = torch.stack([time_grid, height_grid, width_grid], dim=-1).reshape(-1, 3)

    meta = torch.tensor([ori_t, ori_h, ori_w, ref_t, ref_h, ref_w], dtype=dtype)
    meta_expanded = meta.expand(coords_flat.size(0), -1)
    # torch.cat returns fresh contiguous storage -> safe for DataLoader pin_memory
    # (the meta.expand stride-0 view is consumed here, never returned directly).
    return torch.cat([coords_flat, meta_expanded], dim=-1)  # [N, 9]


def _channels_last_within_patch(tok: torch.Tensor, t_patch: int, s_patch: int) -> torch.Tensor:
    """Reorder each video token from ``(C, pT, pH, pW)`` to ``(pT, pH, pW, C)``.

    ``tok`` is ``[N, C * pT * pH * pW]`` as upstream ``img2tokens`` flattens it. The
    result follows the ``(pT pH pW C)`` pattern that ``depack_token_sequence`` uses to
    read the DiT output. A pure permutation, so every value is copied bit-for-bit.
    """
    n, width = tok.shape
    patch_volume = t_patch * s_patch * s_patch
    if width % patch_volume:
        raise ValueError(f"video width {width} is not divisible by patch volume {patch_volume}.")
    channels = width // patch_volume
    return (
        tok.reshape(n, channels, t_patch, s_patch, s_patch)
        .permute(0, 2, 3, 4, 1)
        .reshape(n, width)
        .contiguous()
    )


@data_register.register("magihuman_precomputed")
class MagiHumanPrecomputedDataset(Dataset):
    """Load MagiHuman precomputed clean latents/conditions; noise+pack in collate.

    Two data sources, switchable via ``dataset_param`` (see ``_resolve_basic_param``):
      * real precomputed: reads ``{dataset_dir}/.precomputed/{latents,conditions,
        audio_latents}/*.pt`` (mirrors the ltx2 file layout).
      * synthetic: ``synthetic: true`` generates N small random CLEAN samples so
        a 2-card FSDP single-step E2E runs with NO real data.
    """

    def __init__(self, basic_param, preprocess_param=None, dataset_param=None, **kwargs):
        _ = (preprocess_param, kwargs)
        resolved_basic_param = self._resolve_basic_param(basic_param, dataset_param)
        self.args = MagiHumanPrecomputedBasicArgs(**resolved_basic_param)
        self.args.synthetic_video_grid = self._as_thw(self.args.synthetic_video_grid)

        dataset_dir = Path(self.args.dataset_dir).expanduser()
        self.synthetic = bool(self.args.synthetic)

        if self.synthetic:
            self._init_synthetic()
        else:
            self._init_real(dataset_dir)

    # --------------------------------------------------------------- arg plumbing
    @staticmethod
    def _to_dict(obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return {}

    @classmethod
    def _resolve_basic_param(cls, basic_param: Any, dataset_param: Any) -> dict[str, Any]:
        """Merge ``dataset_param.magihuman_dataset_custom`` (low) with ``basic_param`` (high)."""
        merged: dict[str, Any] = {}
        dataset_param_dict = cls._to_dict(dataset_param)

        # Preferred location: data.dataset_param.magihuman_dataset_custom (YAML anchor).
        dataset_custom = dataset_param_dict.get("magihuman_dataset_custom", {})
        if isinstance(dataset_custom, dict):
            merged.update(dataset_custom)

        # Highest priority: basic_parameters.
        if isinstance(basic_param, dict):
            merged.update(basic_param)

        valid_keys = {f.name for f in fields(MagiHumanPrecomputedBasicArgs)}
        return {k: v for k, v in merged.items() if k in valid_keys}

    @staticmethod
    def _as_thw(value: Any) -> tuple[int, int, int]:
        seq = list(value)
        if len(seq) != 3:
            raise ValueError(f"Expected a (T, H, W) triple, got: {value!r}")
        return (int(seq[0]), int(seq[1]), int(seq[2]))

    # ------------------------------------------------------------- sample listing
    def _init_synthetic(self) -> None:
        n = int(self.args.synthetic_samples)
        if n <= 0:
            raise ValueError(f"synthetic_samples must be > 0, got {n}.")
        items = list(range(n))
        self.samples: list[Any] = self._trim_for_data_parallel(items)

    def _init_real(self, dataset_dir: Path) -> None:
        root = dataset_dir.resolve()
        if (root / ".precomputed").is_dir():
            root = root / ".precomputed"
        self.latents_root = root / self.args.latents_dir
        self.conditions_root = root / self.args.conditions_dir
        self.audio_latents_root = root / self.args.audio_latents_dir if self.args.with_audio else None

        if not self.latents_root.is_dir():
            raise FileNotFoundError(f"Latents directory not found: {self.latents_root}")
        if not self.conditions_root.is_dir():
            raise FileNotFoundError(f"Conditions directory not found: {self.conditions_root}")
        if self.args.with_audio and (self.audio_latents_root is None or not self.audio_latents_root.is_dir()):
            raise FileNotFoundError(f"Audio latents directory not found: {self.audio_latents_root}")

        latent_files = sorted(self.latents_root.rglob("*.pt"))
        if not latent_files:
            raise ValueError(f"No latent .pt files found under: {self.latents_root}")

        pairs: list[tuple[Path, Path, Path | None]] = []
        for latent_path in latent_files:
            rel = latent_path.relative_to(self.latents_root)
            cond_path = self.conditions_root / rel
            if not cond_path.exists() and latent_path.name.startswith("latent_"):
                cond_path = self.conditions_root / rel.with_name(f"condition_{latent_path.stem[7:]}.pt")
            if not cond_path.exists():
                continue

            audio_latent_path: Path | None = None
            if self.args.with_audio:
                audio_latent_path = self.audio_latents_root / rel
                if not audio_latent_path.exists() and latent_path.name.startswith("latent_"):
                    audio_latent_path = self.audio_latents_root / rel.with_name(f"audio_{latent_path.stem[7:]}.pt")
                if not audio_latent_path.exists():
                    continue
            pairs.append((latent_path, cond_path, audio_latent_path))

        if not pairs:
            raise ValueError("No matched (latents, conditions[, audio_latents]) file triplets were found.")

        if self.args.max_samples is not None:
            pairs = pairs[: self.args.max_samples]
        self.samples = self._trim_for_data_parallel(pairs)

    @staticmethod
    def _get_world_size() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
        # Dataset construction can precede process-group init; the launcher's
        # WORLD_SIZE (default 1) is then the only source of truth.
        return envs.WORLD_SIZE

    @classmethod
    def _trim_for_data_parallel(cls, items: list[Any]) -> list[Any]:
        """Drop tail samples so per-rank counts stay aligned with drop_last=True."""
        world_size = max(cls._get_world_size(), 1)
        if world_size <= 1:
            return items
        usable = (len(items) // world_size) * world_size
        if usable == 0:
            raise ValueError(
                f"Dataset has {len(items)} samples, smaller than world size {world_size}. "
                "Cannot shard evenly for distributed training."
            )
        if usable != len(items):
            logger.warning(
                "Trimming %d tail samples from MagiHuman dataset for even DP sharding: %d -> %d (world_size=%d)",
                len(items) - usable,
                len(items),
                usable,
                world_size,
            )
            return items[:usable]
        return items

    # --------------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return ONE CLEAN sample (no noising here; noising happens in collate_fn).

        Returns a flat dict:
            video_latent    [Nv, 192]   clean patchified video latent tokens
            video_grid_thw  (T, H, W)   patchified grid dims, T*H*W == Nv
            text_embed      [Nt, 3584]  clean text condition embeddings
            audio_latent    [Na, 64]    clean patchified audio latent tokens (if with_audio)
            sample_index    int         dataset index; seeds the noising RNG
        """
        sample = self._make_synthetic_sample(idx) if self.synthetic else self._load_real_sample(idx)
        sample["sample_index"] = int(idx)
        return sample

    def _make_synthetic_sample(self, idx: int) -> dict[str, Any]:
        gen = torch.Generator().manual_seed(int(self.args.seed) + int(idx))
        grid = self.args.synthetic_video_grid
        n_video = grid[0] * grid[1] * grid[2]
        sample: dict[str, Any] = {
            "video_latent": torch.randn(n_video, self.args.video_channels, generator=gen, dtype=torch.float32),
            "video_grid_thw": grid,
            "text_embed": torch.randn(
                int(self.args.synthetic_text_tokens), self.args.text_channels, generator=gen, dtype=torch.float32
            ),
        }
        if self.args.with_audio:
            sample["audio_latent"] = torch.randn(
                int(self.args.synthetic_audio_tokens), self.args.audio_channels, generator=gen, dtype=torch.float32
            )
        return sample

    def _load_real_sample(self, idx: int) -> dict[str, Any]:
        latent_path, cond_path, audio_path = self.samples[idx]
        latent_data = self._safe_torch_load(latent_path)
        cond_data = self._safe_torch_load(cond_path)

        video_latent = self._require(latent_data, ("video_latent", "latents"), latent_path).float()
        grid_thw = self._require(latent_data, ("video_grid_thw", "grid_thw"), latent_path)
        text_embed = self._require(cond_data, ("prompt_embeds", "text_embed"), cond_path).float()

        sample: dict[str, Any] = {
            "video_latent": video_latent,
            "video_grid_thw": self._as_thw(self._to_list(grid_thw)),
            "text_embed": text_embed,
        }
        if self.args.with_audio:
            if audio_path is None:
                raise ValueError("with_audio=True but audio latent path is missing for this sample.")
            audio_data = self._safe_torch_load(audio_path)
            sample["audio_latent"] = self._require(audio_data, ("audio_latent", "latents"), audio_path).float()
        return sample

    @staticmethod
    def _to_list(value: Any) -> list:
        if isinstance(value, torch.Tensor):
            return value.reshape(-1).tolist()
        return list(value)

    @staticmethod
    def _require(data: dict[str, Any], keys: tuple[str, ...], path: Path) -> Any:
        if not isinstance(data, dict):
            raise TypeError(f"Expected a dict in {path}, got {type(data).__name__}.")
        for key in keys:
            if key in data:
                return data[key]
        raise KeyError(f"{path} must contain one of {keys}; found keys: {sorted(data.keys())}.")

    @staticmethod
    def _safe_torch_load(path: Path) -> dict:
        """Load a precomputed ``.pt`` with ``weights_only=True``.

        Precomputed latents/embeddings are plain tensor dicts, so the safe
        loader always suffices. There is deliberately no fallback to the
        unrestricted loader: a file that needs one is either corrupt or
        untrusted, and both cases must fail loudly rather than deserialize
        arbitrary objects.
        """
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as e:
            raise RuntimeError(
                f"Failed to load {path} with weights_only=True. Precomputed "
                f"MagiHuman data must hold only tensors and Python native "
                f"scalars/containers (no numpy arrays, Path or custom objects) "
                f"in every key; regenerate the file."
            ) from e

    # ----------------------------------------------------------- noising + packing
    def _noise_generator(self, idx: int) -> torch.Generator:
        """Return the per-sample RNG used for sigma and the flow-matching noise.

        Seeding by dataset index (instead of drawing from the global RNG) keeps
        the noising of sample ``i`` identical no matter how many ranks, dataloader
        workers or gradient-accumulation micro-steps the run uses, which is what
        makes two configurations comparable step by step.

        ``_NOISE_STREAM_OFFSET`` keeps this stream disjoint from the one
        ``_make_synthetic_sample`` uses; sharing it would make the noise equal to
        the very latent it is supposed to noise.
        """
        return torch.Generator().manual_seed(int(self.args.seed) + _NOISE_STREAM_OFFSET + int(idx))

    def _sample_sigma(self, generator: torch.Generator) -> torch.Tensor:
        """Sample one flow-matching sigma in (0, 1) for the whole packed sample.

        MagiHuman's upstream DiT accepts ``x_t`` but has no timestep argument;
        sigma is used here only to construct ``x_t`` and the velocity target.
        The upstream release is inference-only, so this sampling distribution
        is migration-side configuration rather than copied training code.
        """
        mode = self.args.timestep_sampling_mode
        params = self.args.timestep_sampling_params or {}
        if mode == "uniform":
            lo = float(params.get("min_value", 0.0))
            hi = float(params.get("max_value", 1.0))
            if hi <= lo:
                raise ValueError(f"uniform sampler needs max_value ({hi}) > min_value ({lo}).")
            return torch.rand((), generator=generator, dtype=torch.float32) * (hi - lo) + lo
        if mode == "logit_normal":
            loc = float(params.get("loc", 0.0))
            scale = float(params.get("scale", 1.0))
            return torch.sigmoid(torch.randn((), generator=generator, dtype=torch.float32) * scale + loc)
        raise ValueError(f"Unsupported timestep_sampling_mode: {mode}. Expected 'uniform' or 'logit_normal'.")

    def _build_coords_mapping(self, video_grid: tuple[int, int, int], n_audio: int, n_text: int) -> torch.Tensor:
        """Packed [L, 9] coords in the SAME row order as x / modality (video, audio, text)."""
        # video ("extra" interpolation): ref == size -> every video RoPE scale == 1.
        parts = [_get_coords(shape=video_grid, ref_feat_shape=video_grid)]
        if n_audio > 0:
            # audio ("v2"): magic ref_T then // t_patch_size; h,w axes are size==1 & ref==1.
            magic_ref_t = (n_audio - 1) // _AUDIO_REF_DOWNSAMPLE + 1
            ref_t = max(magic_ref_t // int(self.args.t_patch_size), 1)
            parts.append(_get_coords(shape=(n_audio, 1, 1), ref_feat_shape=(ref_t, 1, 1)))
        # text ("v2"): ref=(1,1,1), negative time offset -Nt -> time indices [-Nt, ..., -1].
        parts.append(_get_coords(shape=(n_text, 1, 1), ref_feat_shape=(1, 1, 1), offset_thw=(-n_text, 0, 0)))
        return torch.cat(parts, dim=0).to(torch.float32)

    def _build_packed_sample(self, sample: dict[str, Any], idx: int) -> dict[str, torch.Tensor]:
        """Noise (flow-matching) + pack ONE clean sample into the model's flat tensors."""
        video = sample["video_latent"].float()  # [Nv, 192]
        grid = self._as_thw(sample["video_grid_thw"])
        text = sample["text_embed"].float()  # [Nt, 3584]
        audio = sample.get("audio_latent")
        audio = audio.float() if isinstance(audio, torch.Tensor) else None

        n_video = video.shape[0]
        n_text = text.shape[0]
        n_audio = audio.shape[0] if audio is not None else 0
        if grid[0] * grid[1] * grid[2] != n_video:
            raise ValueError(
                f"video_grid_thw {grid} (prod={grid[0] * grid[1] * grid[2]}) != n_video {n_video}."
            )
        if video.shape[1] != self.args.video_channels:
            raise ValueError(f"video_latent width {video.shape[1]} != video_channels {self.args.video_channels}.")
        if text.shape[1] != self.args.text_channels:
            raise ValueError(f"text_embed width {text.shape[1]} != text_channels {self.args.text_channels}.")
        if audio is not None and audio.shape[1] != self.args.audio_channels:
            raise ValueError(f"audio_latent width {audio.shape[1]} != audio_channels {self.args.audio_channels}.")

        total_len = n_video + n_audio + n_text
        generator = self._noise_generator(idx)
        sigma = self._sample_sigma(generator)

        # x_t = (1 - sigma) * x0 + sigma * eps ; target = eps - x0 (flow matching).
        x = torch.zeros(total_len, self.args.text_channels, dtype=torch.float32)
        target_velocity = torch.zeros(total_len, self.args.video_channels, dtype=torch.float32)

        eps_v = torch.randn(video.shape, generator=generator, dtype=torch.float32)
        x[:n_video, : self.args.video_channels] = (1.0 - sigma) * video + sigma * eps_v
        # Only the target is reordered: the input embedder reads (C, pT, pH, pW) tokens
        # while the output head emits (pT, pH, pW, C) ones (see the module docstring).
        target_velocity[:n_video, : self.args.video_channels] = _channels_last_within_patch(
            eps_v - video, int(self.args.t_patch_size), _SPATIAL_PATCH
        )

        if audio is not None and n_audio > 0:
            eps_a = torch.randn(audio.shape, generator=generator, dtype=torch.float32)
            x[n_video : n_video + n_audio, : self.args.audio_channels] = (1.0 - sigma) * audio + sigma * eps_a
            target_velocity[n_video : n_video + n_audio, : self.args.audio_channels] = eps_a - audio

        # text rows hold the CLEAN condition (never noised); excluded from loss.
        x[n_video + n_audio : total_len, : self.args.text_channels] = text

        modality_mapping = torch.cat(
            [
                torch.full((n_video,), _MOD_VIDEO, dtype=torch.int64),
                torch.full((n_audio,), _MOD_AUDIO, dtype=torch.int64),
                torch.full((n_text,), _MOD_TEXT, dtype=torch.int64),
            ],
            dim=0,
        )

        loss_mask = torch.zeros(total_len, dtype=torch.bool)
        loss_mask[: n_video + n_audio] = True

        coords_mapping = self._build_coords_mapping(grid, n_audio, n_text)
        if coords_mapping.shape[0] != total_len:
            raise ValueError(f"coords rows {coords_mapping.shape[0]} != packed length {total_len}.")

        return {
            "x": x,
            "coords_mapping": coords_mapping,
            "modality_mapping": modality_mapping,
            "target_velocity": target_velocity,
            "loss_mask": loss_mask,
        }

    def collate_fn(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Noise + pack samples and emit boundaries for varlen attention.

        MagiHuman's DiT consumes a SINGLE packed sequence ``x [L, 3584]`` (not a
        batched ``[B, S, C]``), so tensors are concatenated along the token axis.
        ``cu_seqlens`` preserves sample boundaries; the model wrapper rebuilds
        upstream's ``VarlenHandler`` and the NPU attention patch uses TND varlen
        attention when the batch contains more than one sample.
        """
        if not features:
            raise ValueError("collate_fn received an empty feature list.")

        packed = [self._build_packed_sample(feat, feat["sample_index"]) for feat in features]
        keys = ("x", "coords_mapping", "modality_mapping", "target_velocity", "loss_mask")
        # torch.cat yields fresh contiguous storage -> DataLoader pin_memory safe.
        batch = {key: torch.cat([p[key] for p in packed], dim=0).contiguous() for key in keys}

        lengths = torch.tensor([p["x"].shape[0] for p in packed], dtype=torch.int32)
        batch["cu_seqlens"] = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                lengths.cumsum(dim=0, dtype=torch.int32),
            ],
            dim=0,
        ).contiguous()
        batch["max_seqlen"] = lengths.max()
        return batch
