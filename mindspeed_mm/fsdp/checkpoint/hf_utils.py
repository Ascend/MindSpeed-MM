from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import re
from typing import Dict, Generator, List, Optional, Set, Tuple

from pydantic import FilePath
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.fsdp.checkpoint.utils import remove_base_layer_keys
from mindspeed_mm.fsdp.utils.utils import tensor_to_dtensor

logger = logging.getLogger(__name__)


@dataclass
class HFWeightFileStream:
    """Yields ``(key, full_tensor)`` one tensor at a time from a safetensors file.

    Uses ``safe_open`` so only the current tensor is materialized in host
    memory; the consumer copies it into the model and drops it before the next
    is read. Peak host memory is therefore about one tensor, not the whole
    shard file.
    """

    filepath: str

    def __iter__(self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        with safe_open(self.filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def _lora_base_key_map(param_names: Set[str]) -> Dict[str, str]:
    """Ref DCP lora base key map.
    """
    base_to_bare = remove_base_layer_keys({name: None for name in param_names})
    return {bare: base for base, bare in base_to_bare.items()}


def locate_hf_weight_files(weights_path: str) -> List[HFWeightFileStream]:
    """Resolve the safetensors files under *weights_path* into stream readers.

    Supports both standard HF layouts:
      - single ``model.safetensors``
      - sharded ``model-*-of-*.safetensors`` described by ``model.safetensors.index.json``
    """
    single = os.path.join(weights_path, SAFE_WEIGHTS_NAME)
    if os.path.isfile(single):
        return [HFWeightFileStream(single)]

    index = os.path.join(weights_path, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.isfile(index):
        with open(index, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        streams = []
        for name in sorted(set(weight_map.values())):
            path = os.path.join(weights_path, name)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Shard '{name}' referenced by {index} is missing: {path}"
                )
            streams.append(HFWeightFileStream(path))
        return streams

    raise ValueError(f"No HF safetensors weights found under {weights_path}.")


def convert_weight_key(key: str, model: torch.nn.Module) -> str:
    mapping = getattr(model, "_checkpoint_conversion_mapping", None)
    if not mapping:
        return key
    for pattern, replacement in mapping.items():
        replacement = re.sub(r"\(.*\)", "", replacement.lstrip("^"))
        new_key, n_subs = re.subn(pattern, replacement, key)
        if n_subs > 0:
            return new_key
    return key


def _resolve_leaf(model: torch.nn.Module, name: str) -> Tuple[torch.nn.Module, str]:
    """Walk the dotted FQN *name* down to the leaf module that owns it.

    For ``model.language_model.layers.0.self_attn.q_proj.weight`` it returns
    the ``q_proj`` Linear module and ``"weight"``.
    """
    module = model
    pieces = name.split(".")
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot resolve '{name}': submodule '{piece}' not found.")
        module = getattr(module, piece)
    return module, pieces[-1]


def write_full_tensor(model: torch.nn.Module, name: str, full_tensor: torch.Tensor) -> None:
    """Write *full_tensor* into the model at *name* (parameter or buffer).

    For a sharded parameter (a DTensor created by ``fully_shard``), the target
    already carries its ``device_mesh`` / ``placements``; we move the full
    tensor onto the mesh device and let ``tensor_to_dtensor`` carve out this
    rank's shard (a local Replicate -> Shard redistribute, no comm). For a
    plain tensor (replicated parameter or buffer), we just move dtype/device
    and copy. The write is always in place since FSDP2 holds the parameter
    object.
    """
    leaf_module, local_name = _resolve_leaf(model, name)

    if local_name in leaf_module._parameters:
        target = leaf_module._parameters[local_name].data
    elif local_name in leaf_module._buffers:
        target = leaf_module._buffers[local_name]
    else:
        raise ValueError(f"'{name}' is neither a parameter nor a buffer of the model.")

    if hasattr(target, "device_mesh"):  # sharded -> a DTensor
        full_tensor = full_tensor.to(device=target.to_local().device, dtype=target.dtype)
        shard = tensor_to_dtensor(full_tensor, target.device_mesh, target.placements)
        target.copy_(shard)
    else:  # plain tensor (replicated on every rank)
        target.copy_(full_tensor.to(device=target.device, dtype=target.dtype))


def _log_unexpected_keys(unexpected_keys: Set[str]) -> None:
    if not unexpected_keys:
        return
    samples = sorted(unexpected_keys)[:5]
    suffix = "" if len(unexpected_keys) <= 5 else f" (showing 5 of {len(unexpected_keys)})"
    print_rank(
        logger.info,
        f"HF checkpoint had {len(unexpected_keys)} key(s) not present in the model. Examples{suffix}: {samples}",
    )


def post_process_after_load(
    missing_param_keys: set,
    load_strict: bool = False,
) -> None:
    # filter LoRA: lora_A/lora_B are intentionally absent from a base checkpoint
    missing = {k for k in missing_param_keys if "lora_" not in k}
    if missing:
        # load_strict : true -> error on any model key the tensor did not provide with raise.
        if load_strict:
            raise RuntimeError(
                f"{len(missing)} parameter key(s) absent from the HF checkpoint "
                f"(load_strict=True): {sorted(missing)}"
            )
        logger.warning(
            "%d parameter key(s) absent from the HF checkpoint and left "
            "uninitialized: %s. Training will likely produce NaNs. "
            "Pre-load these via the HF file or extend post_process_after_load "
            "to call _init_weights on them.",
            len(missing),
            sorted(missing),
        )


def looks_like_hf_weight_dir(path: Optional[str]) -> bool:
    """Whether *path* is a directory holding HF safetensors weights.

    Used by ``load_format='auto'`` to tell an HF directory apart from a DCP
    checkpoint directory -- the latter has ``latest_checkpointed_iteration.txt``
    and ``.distcp`` files instead, never the standard HF weight filenames.
    """
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, SAFE_WEIGHTS_NAME)) or os.path.isfile(
        os.path.join(path, SAFE_WEIGHTS_INDEX_NAME)
    )


@torch.no_grad()
def load_hf_weights(
    model: torch.nn.Module,
    hf_dir: str,
    enable_lora: bool = False,
    load_strict: bool = False,
    weight_transform=None,
) -> None:
    """Load HF safetensors weights into *model* directly, no offline conversion.

    Every rank opens the same files and slices each full tensor into its own
    DTensor shard locally -- no cross-rank communication. Suitable for small/
    medium models or shared filesystems; for very large models use the
    rank0-read-and-broadcast variant (added in a follow-up).

    Assumes ``model`` has already been laid out by ``fully_shard`` and brought
    out of meta via ``to_empty_if_needed`` -- i.e. parameter slots are empty
    DTensors on the real device and buffer values have been preserved.
    """
    param_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    lora_base_map = _lora_base_key_map(param_names) if enable_lora else {}
    unexpected_keys: Set[str] = set()

    for shard_stream in locate_hf_weight_files(hf_dir):
        for raw_key, full_tensor in shard_stream:
            key = convert_weight_key(raw_key, model)
            if weight_transform is not None:
                key, full_tensor = weight_transform.hf_to_dcp(key, full_tensor)
            key = lora_base_map.get(key, key)  # bare -> base_layer for LoRA targets
            if key in param_names:
                param_names.discard(key)
                write_full_tensor(model, key, full_tensor)
            elif key in buffer_names:
                # Persistent buffer override from the file. Non-persistent
                # buffers (e.g. RoPE inv_freq) never appear here; they were
                # built in model construction and preserved by to_empty_if_needed.
                write_full_tensor(model, key, full_tensor)
            else:
                unexpected_keys.add(key)

    _log_unexpected_keys(unexpected_keys)
    post_process_after_load(missing_param_keys=param_names, load_strict=load_strict)


def find_safetensors_index(directory: Path) -> Optional[FilePath]:
    """Find the .safetensors.index.json file in the given directory."""
    if not directory.is_dir():
        return None
    for file in directory.iterdir():
        if file.is_file() and file.name.endswith(".safetensors.index.json"):
            return file
    return None


def shard_index_from_filename(filename: str) -> int:
    """Parse shard index from ``model-00003-of-00014.safetensors`` style names."""
    return int(filename.split("-")[1])


def parse_fqn_to_filename_mapping_from_json(safetensor_idx_path: str) -> Dict[str, int]:
    """Load ``weight_map`` from a HuggingFace ``model.safetensors.index.json`` file."""
    with open(safetensor_idx_path) as f:
        return json.load(f)["weight_map"]


@torch.no_grad()
def get_model_save_state(
    model: torch.nn.Module,
    fqn_to_filename_mapping: Optional[Dict[str, int]],
    save_ckpt_dtype: Optional[torch.dtype] = None,
    enable_lora: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build a state dict suitable for HuggingFace safetensors saving.
    """
    from mindspeed_mm.fsdp.checkpoint.dcp_checkpointer import LoraModelState, ModelState

    model_state_cls = LoraModelState if enable_lora else ModelState
    save_state = model_state_cls(model, save_ckpt_dtype).state_dict()

    if fqn_to_filename_mapping is not None:
        filtered_state = {}
        for k, v in save_state.items():
            if k in fqn_to_filename_mapping:
                filtered_state[k] = v
            else:
                logger.info_rank0(f"Skipping weight not in HF weight_map: {k}")
        save_state = filtered_state

    return save_state


@lru_cache
def get_dtype_size(dtype: "torch.dtype") -> int:
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    _SIZE = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
        _float8_e4m3fn: 1,
        _float8_e5m2: 1,
    }
    return _SIZE[dtype]


def save_state_dict(
    state_dict: Dict[str, "torch.Tensor"],
    path_to_save: "os.PathLike"
) -> None:
    """
    HuggingFace checkpoint save function.
    """
    save_file(state_dict, path_to_save, metadata={"format": "pt"})
