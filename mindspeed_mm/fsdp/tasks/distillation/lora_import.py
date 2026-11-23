# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Strict, configurable LoRA import before FSDP wrapping.

H3 LightX2V layout uses separate Q/K/V and up/gate order. The MindSpeed-MM
H3 DiT uses head-interleaved fused QKV and gate/up order. Conversion preserves
the complete low-rank increment; it performs no SVD or rank truncation.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import torch


@dataclass
class LoraImportConfig:
    path: str
    source_format: str = "auto"  # peft/diffsynth use native module paths
    alpha: float | None = None
    alpha_pattern: dict[str, float] = field(default_factory=dict)
    scaling: str | None = None  # alpha_over_rank, alpha_over_sqrt_rank, unit
    head_dim: int = 128
    source_adapter: str = "default"
    strip_prefixes: tuple[str, ...] = (
        "base_model.model.",
        "pipe.dit.",
        "transformer.",
        "diffusion_model.",
    )
    module_map: dict[str, str] = field(default_factory=dict)
    dropout: float = 0.0

    def __post_init__(self):
        if not isinstance(self.head_dim, int) or isinstance(self.head_dim, bool) or self.head_dim < 1:
            raise ValueError("head_dim must be a positive integer")
        if not isinstance(self.source_adapter, str) or not self.source_adapter:
            raise ValueError("source_adapter must be a nonempty name")
        if not all(isinstance(prefix, str) and prefix for prefix in self.strip_prefixes):
            raise ValueError("strip_prefixes must contain nonempty strings")
        if not all(
            isinstance(name, str) and name and isinstance(target, str) and target
            for name, target in self.module_map.items()
        ):
            raise ValueError("module_map must map nonempty module names")


@dataclass
class PreparedLora:
    # Canonical PEFT keys, with scale folded into B and destination alpha/r=1.
    state_dict: dict[str, torch.Tensor]
    ranks: dict[str, int]
    source_format: str
    source_modules: tuple[str, ...]
    head_dim: int | None = None
    destination_alphas: dict[str, float] | None = None


def _strip_prefix(name, prefixes):
    while True:
        prefix = next((p for p in prefixes if p and name.startswith(p)), None)
        if prefix is None:
            return name
        name = name[len(prefix) :]


def _pattern_value(pattern, name, prefixes):
    values = [
        value
        for key, value in pattern.items()
        if name == _strip_prefix(key, prefixes) or name.endswith("." + _strip_prefix(key, prefixes))
    ]
    if not values:
        return None
    if len(set(values)) != 1:
        raise ValueError(f"Ambiguous LoRA pattern for {name}")
    return values[0]


def _positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def _read_file(config):
    path = Path(config.path).expanduser()
    if path.is_dir():
        candidates = [
            path / name
            for name in (
                "adapter_model.safetensors",
                "pytorch_lora_weights.safetensors",
                "adapter_model.bin",
            )
        ]
        candidates = [p for p in candidates if p.is_file()]
        if len(candidates) != 1:
            raise ValueError(
                "LoRA directory must contain one recognized weight file; otherwise configure the file path"
            )
        path = candidates[0]
    metadata = {}
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
        state = load_file(str(path), device="cpu")
    elif path.suffix in (".bin", ".pt", ".pth"):
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"Unsupported LoRA file extension: {path.suffix}")
    sidecar = path.with_suffix(".adapter_config.json")
    directory_config = path.parent / "adapter_config.json"
    if sidecar.is_file():
        settings = json.loads(sidecar.read_text(encoding="utf-8"))
    elif directory_config.is_file():
        settings = json.loads(directory_config.read_text(encoding="utf-8"))
    elif "flashgen_lora_config" in metadata:
        settings = json.loads(metadata["flashgen_lora_config"])
    else:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("LoRA config metadata must be an object")
    if "lora_alpha" not in settings and "lora_alpha" in metadata:
        settings["lora_alpha"] = float(metadata["lora_alpha"])
    return state, settings


def _source_pairs(state, config):
    if not isinstance(state, dict) or not state:
        raise ValueError("LoRA weights must be a nonempty tensor dictionary")
    pairs, alphas = {}, {}
    aliases = (
        (".lora.down.weight", "A"),
        (".lora.up.weight", "B"),
        (".lora_down.weight", "A"),
        (".lora_up.weight", "B"),
    )
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("LoRA weights must contain string keys and tensor values only")
        name = _strip_prefix(key, config.strip_prefixes)
        if name.endswith(".alpha"):
            module = name[:-6]
            if module in alphas or value.numel() != 1:
                raise ValueError(f"Duplicate or invalid alpha tensor: {key}")
            alphas[module] = _positive(value.item(), key)
            continue
        standard = None
        for side in "AB":
            module, marker, suffix = name.rpartition(".lora_" + side)
            if not marker or not module:
                continue
            if suffix == ".weight":
                standard = module, side
                break
            if suffix.startswith(".") and suffix.endswith(".weight"):
                adapter = suffix[1 : -len(".weight")]
                if not adapter or "." in adapter:
                    continue
                if adapter != config.source_adapter:
                    raise ValueError(f"Unexpected source adapter in {key}; select source_adapter explicitly")
                standard = module, side
                break
        if standard is not None:
            module, side = standard
        else:
            alias = next(
                ((suffix, side) for suffix, side in aliases if name.endswith(suffix)),
                None,
            )
            if alias is None:
                raise ValueError(f"Unsupported LoRA tensor: {key}")
            suffix, side = alias
            module = name[: -len(suffix)]
            if not module:
                raise ValueError(f"Missing LoRA module name: {key}")
        pair = pairs.setdefault(module, {})
        if side in pair:
            raise ValueError(f"Duplicate normalized LoRA matrix: {module}.{side}")
        if value.ndim != 2 or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"LoRA matrix must be finite floating point and two-dimensional: {key}")
        pair[side] = value.detach().cpu().float()
    if not pairs or set(alphas) - set(pairs):
        raise ValueError("No LoRA pairs, or alpha tensors without matching matrices")
    for name, pair in pairs.items():
        if set(pair) != {"A", "B"} or min(pair["A"].shape) < 1 or min(pair["B"].shape) < 1:
            raise ValueError(f"Incomplete or empty LoRA pair: {name}")
        if pair["A"].shape[0] != pair["B"].shape[1]:
            raise ValueError(f"LoRA A/B rank mismatch: {name}")
    return pairs, alphas


def _scale_pairs(pairs, alphas, config, metadata):
    if metadata.get("peft_type", "LORA") != "LORA" or metadata.get("use_dora", False):
        raise ValueError("Only standard LoRA is supported; DoRA and other adapter types need their own converter")
    if metadata.get("fan_in_fan_out", False) or metadata.get("bias", "none") != "none":
        raise ValueError("Transposed base layers and adapter bias need an explicit converter")
    if not isinstance(metadata.get("use_rslora", False), bool):
        raise ValueError("use_rslora metadata must be boolean")
    declared_scaling = "alpha_over_sqrt_rank" if metadata.get("use_rslora", False) else "alpha_over_rank"
    scaling = config.scaling or declared_scaling
    if scaling not in ("alpha_over_rank", "alpha_over_sqrt_rank", "unit"):
        raise ValueError(f"Unsupported LoRA scaling: {scaling}")
    if "use_rslora" in metadata and scaling != declared_scaling:
        raise ValueError("Configured scaling conflicts with use_rslora metadata")
    for pattern in (
        config.alpha_pattern,
        metadata.get("alpha_pattern", {}),
        metadata.get("rank_pattern", {}),
    ):
        unused = [
            key
            for key in pattern
            if not any(
                name == _strip_prefix(key, config.strip_prefixes)
                or name.endswith("." + _strip_prefix(key, config.strip_prefixes))
                for name in pairs
            )
        ]
        if unused:
            raise ValueError(f"Unused LoRA pattern keys: {sorted(unused)}")
    for name, pair in pairs.items():
        rank = pair["A"].shape[0]
        declared_rank = _pattern_value(metadata.get("rank_pattern", {}), name, config.strip_prefixes)
        if declared_rank is None:
            declared_rank = metadata.get("r")
        if declared_rank is not None and declared_rank != rank:
            raise ValueError(f"Rank metadata conflicts with matrices at {name}")
        file_alpha = _pattern_value(metadata.get("alpha_pattern", {}), name, config.strip_prefixes)
        if file_alpha is None:
            file_alpha = metadata.get("lora_alpha")
        if name in alphas:
            if file_alpha is not None and float(file_alpha) != alphas[name]:
                raise ValueError(f"Alpha tensor conflicts with metadata at {name}")
            file_alpha = alphas[name]
        user_alpha = _pattern_value(config.alpha_pattern, name, config.strip_prefixes)
        if user_alpha is None:
            user_alpha = config.alpha
        if file_alpha is not None and user_alpha is not None and float(file_alpha) != float(user_alpha):
            raise ValueError(f"Configured alpha conflicts with checkpoint at {name}")
        alpha = file_alpha if user_alpha is None else user_alpha
        if scaling == "unit":
            if alpha is not None and float(alpha) != rank:
                raise ValueError(f"Unit scaling conflicts with alpha at {name}")
            scale = 1.0
        else:
            if alpha is None:
                raise ValueError(f"Missing alpha for {name}; configure alpha/scaling instead of guessing")
            scale = _positive(alpha, f"alpha for {name}") / (
                math.sqrt(rank) if scaling == "alpha_over_sqrt_rank" else rank
            )
        pair["B"] = pair["B"] * scale
        if not torch.isfinite(pair["B"]).all():
            raise ValueError(f"LoRA scaling overflows at {name}")


def prepare_lora(config, *, state_dict=None, metadata=None):
    """Normalize and convert on CPU, without mutating a model."""
    if state_dict is None:
        state_dict, file_metadata = _read_file(config)
        if metadata is not None:
            raise ValueError("Metadata override is only accepted with an in-memory state_dict")
        metadata = file_metadata
    pairs, alphas = _source_pairs(state_dict, config)
    source_names = tuple(sorted(pairs))
    source_format = config.source_format
    native = any(name.startswith(("transformer_blocks.", "token_refiner.refiner_blocks.")) for name in pairs)
    if source_format == "auto":
        source_format = "lightx2v" if native else "peft"
    if source_format not in ("lightx2v", "peft", "diffsynth"):
        raise ValueError(f"Unsupported LoRA format: {source_format}")
    if native and source_format != "lightx2v":
        raise ValueError("Native H3 module paths require lightx2v conversion")
    _scale_pairs(pairs, alphas, config, metadata or {})
    if source_format == "lightx2v":
        from mindspeed_mm.fsdp.models.minimax_h3.lora_converter import convert_native_h3_lora

        pairs = convert_native_h3_lora(pairs, config.head_dim)
    mapped, used_mapping = {}, set()
    for name, pair in pairs.items():
        target = config.module_map.get(name, name)
        if name in config.module_map:
            used_mapping.add(name)
        if target in mapped:
            raise ValueError(f"Multiple source modules map to {target}")
        mapped[target] = pair
    if set(config.module_map) != used_mapping:
        raise ValueError(f"Unused module_map keys: {sorted(set(config.module_map) - used_mapping)}")
    state = {
        f"{name}.lora_{side}.default.weight": tensor for name, pair in mapped.items() for side, tensor in pair.items()
    }
    return PreparedLora(
        state,
        {name: pair["A"].shape[0] for name, pair in mapped.items()},
        source_format,
        source_names,
        config.head_dim if source_format == "lightx2v" else None,
    )


def attach_prepared_lora(model, prepared, *, dropout=0.0, allow_meta=False):
    """Install adapter structure before sharding; optionally keep it on meta."""
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    from torch.distributed.tensor import DTensor

    if not prepared.ranks:
        raise ValueError("Cannot inject an empty LoRA plan")
    expected_keys = {f"{name}.lora_{side}.default.weight" for name in prepared.ranks for side in "AB"}
    if set(prepared.state_dict) != expected_keys:
        raise ValueError("Prepared LoRA plan has missing or unexpected tensors")
    for param in model.parameters():
        if (param.is_meta and not allow_meta) or isinstance(param, DTensor):
            raise ValueError("Import LoRA after base weight materialization and before FSDP")
    if any(hasattr(module, "lora_A") for module in model.modules()):
        raise ValueError("Model already contains LoRA adapters")
    modules = dict(model.named_modules())
    for name, rank in prepared.ranks.items():
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"LoRA target is not a model Linear: {name}")
        a, b = (prepared.state_dict[f"{name}.lora_{side}.default.weight"] for side in "AB")
        if a.shape != (rank, module.in_features) or b.shape != (
            module.out_features,
            rank,
        ):
            raise ValueError(f"LoRA dimensions do not match target {name}")
        if prepared.head_dim is not None and name.endswith(".attn.qkv_proj"):
            attention = modules[name.rsplit(".", 1)[0]]
            if getattr(attention, "head_dim", None) != prepared.head_dim:
                raise ValueError(f"Converted QKV head_dim does not match model at {name}")
    from peft import LoraConfig, inject_adapter_in_model

    rank = next(iter(prepared.ranks.values()))
    config = LoraConfig(
        r=rank,
        lora_alpha=(prepared.destination_alphas or prepared.ranks)[next(iter(prepared.ranks))],
        target_modules=list(prepared.ranks),
        rank_pattern=prepared.ranks,
        alpha_pattern=prepared.destination_alphas or prepared.ranks,
        lora_dropout=dropout,
        bias="none",
    )
    model = inject_adapter_in_model(config, model)
    params = dict(model.named_parameters())
    if not set(prepared.state_dict).issubset(params):
        raise RuntimeError("Injected PEFT parameter names do not match the verified import plan")
    for name in prepared.state_dict:
        params[name].data = params[name].data.float()
    return model


def load_prepared_lora_weights(model, prepared):
    """Copy adapter values into materialized plain tensors or FSDP DTensors."""
    from torch.distributed.tensor import DTensor

    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in prepared.state_dict.items():
            param = params[name]
            if param.is_meta:
                raise ValueError("Materialize the sharded model before loading adapter values")
            value = value.to(dtype=param.dtype)
            if isinstance(param, DTensor):
                from mindspeed_mm.fsdp.utils.utils import tensor_to_dtensor_local

                local = tensor_to_dtensor_local(value, param.device_mesh, param.placements)
                param.to_local().copy_(local.to(param.device))
            else:
                value = value.to(param.device)
                param.copy_(value)


def inject_prepared_lora(model, prepared, *, dropout=0.0):
    model = attach_prepared_lora(model, prepared, dropout=dropout)
    load_prepared_lora_weights(model, prepared)
    return model


def fresh_lora_plan(model, *, rank, alpha, targets, seed):
    """Deterministic CPU adapter initialization without materializing the base."""
    generator = torch.Generator("cpu").manual_seed(seed)
    values, ranks = {}, {}
    for name, module in sorted(model.named_modules()):
        if not isinstance(module, torch.nn.Linear) or not any(
            name == target or name.endswith("." + target) for target in targets
        ):
            continue
        a = torch.empty(rank, module.in_features, dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        values[f"{name}.lora_A.default.weight"] = a
        values[f"{name}.lora_B.default.weight"] = torch.zeros(module.out_features, rank)
        ranks[name] = rank
    if not ranks:
        raise ValueError("Fresh LoRA target modules did not match model Linear layers")
    return PreparedLora(values, ranks, "fresh", tuple(ranks), destination_alphas={name: alpha for name in ranks})


def load_lora_for_distillation(model, config):
    """Configured one-shot import; caller owns role selection and base loading."""
    prepared = prepare_lora(config)
    return inject_prepared_lora(model, prepared, dropout=config.dropout), prepared
