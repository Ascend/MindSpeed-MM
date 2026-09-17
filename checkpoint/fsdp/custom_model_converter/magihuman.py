"""HF/DCP checkpoint conversion for MagiHuman's optimized gated-MLP layout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader

from checkpoint.common.converter import DcpConverter
from checkpoint.common.dcp_utils import extract_metadata, load_metadata, partial_load_dcp_state_dict
from checkpoint.common.hf_to_dcp import hf_to_dcp_sharded
from checkpoint.common.merge_dcp_to_hf import find_safetensors_index
from checkpoint.common.permissions import set_directory_permissions
from mindspeed_mm.fsdp.models.magihuman.magihuman_fsdp2.weight_layout import (
    EXPECTED_GATED_UP_GATE_WEIGHTS,
    convert_state_dict_up_gate_layout_,
)


class MagiHumanConverter(DcpConverter):
    """Convert MagiHuman checkpoints between public HF and internal DCP layouts.

    HF uses upstream GPT-OSS-style interleaved rows in gated
    ``up_gate_proj.weight`` tensors. FSDP2 training uses contiguous gate/linear
    halves to avoid stride-2 ViewCopy kernels. The two transforms are exact
    inverses; all other tensors and the original seven-shard HF weight map are
    preserved.
    """

    dcp_prefix = "dit."
    _safetensors_dtypes = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
    }

    def hf_to_dcp(
        self,
        hf_dir: str,
        dcp_dir: str,
        dcp_prefix: str = dcp_prefix,
    ) -> None:
        """Convert upstream HF safetensors into resume-ready internal DCP."""

        converted = 0

        def state_dict_convert_func(state_dict):
            nonlocal converted
            converted += convert_state_dict_up_gate_layout_(state_dict, to_internal=True)
            return {f"{dcp_prefix}{key}": value for key, value in state_dict.items()}

        hf_to_dcp_sharded(
            hf_dir=hf_dir,
            dcp_dir=dcp_dir,
            state_dict_convert_func=state_dict_convert_func,
        )
        self._validate_converted_count(converted, "HF -> DCP")

    def dcp_to_hf(
        self,
        load_dir: str,
        save_dir: str,
        model_assets_dir: str,
        dcp_prefix: str = dcp_prefix,
    ) -> None:
        """Export a training DCP checkpoint in the exact upstream HF layout."""

        load_path = Path(load_dir)
        save_path = Path(save_dir)
        assets_path = Path(model_assets_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self._copy_non_weight_assets(assets_path, save_path)

        index_file = find_safetensors_index(assets_path)
        if index_file is None:
            raise FileNotFoundError(
                f"MagiHuman requires model.safetensors.index.json under {assets_path}"
            )
        with open(index_file, "r", encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]

        file_to_hf_keys: dict[str, list[str]] = {}
        for hf_key, filename in weight_map.items():
            file_to_hf_keys.setdefault(filename, []).append(hf_key)

        storage_reader = FileSystemReader(str(load_path))
        metadata = load_metadata(storage_reader)
        converted = 0
        for filename, hf_keys in sorted(file_to_hf_keys.items()):
            # DCP metadata contains the outer state key (`model.`), while the
            # partial loader returns that outer mapping as a nested dictionary.
            selected_keys = [f"model.{dcp_prefix}{key}" for key in hf_keys]
            partial_metadata = extract_metadata(selected_keys, metadata)
            state_dict = partial_load_dcp_state_dict(partial_metadata, storage_reader)
            state_dict = state_dict["model"] if "model" in state_dict else state_dict

            converted += convert_state_dict_up_gate_layout_(state_dict, to_internal=False)
            hf_state_dict = {}
            with safe_open(
                assets_path / filename, framework="pt", device="cpu"
            ) as reference_file:
                for key, value in state_dict.items():
                    if not key.startswith(dcp_prefix):
                        raise KeyError(
                            f"unexpected MagiHuman DCP key {key!r}; "
                            f"expected prefix {dcp_prefix!r}"
                        )
                    hf_key = key.removeprefix(dcp_prefix)
                    reference_slice = reference_file.get_slice(hf_key)
                    reference_shape = tuple(reference_slice.get_shape())
                    if tuple(value.shape) != reference_shape:
                        raise RuntimeError(
                            f"MagiHuman tensor {hf_key} shape mismatch: "
                            f"DCP={tuple(value.shape)}, HF reference={reference_shape}"
                        )
                    reference_dtype_name = reference_slice.get_dtype()
                    if reference_dtype_name not in self._safetensors_dtypes:
                        raise TypeError(
                            f"unsupported reference dtype {reference_dtype_name!r} "
                            f"for MagiHuman tensor {hf_key}"
                        )
                    reference_dtype = self._safetensors_dtypes[reference_dtype_name]
                    hf_state_dict[hf_key] = value.to(dtype=reference_dtype).contiguous()

            expected_keys = set(hf_keys)
            actual_keys = set(hf_state_dict)
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                unexpected = sorted(actual_keys - expected_keys)
                raise RuntimeError(
                    f"MagiHuman shard {filename} key mismatch: "
                    f"missing={missing[:4]}, unexpected={unexpected[:4]}"
                )
            save_file(hf_state_dict, save_path / filename, metadata={"format": "pt"})

        self._validate_converted_count(converted, "DCP -> HF")
        set_directory_permissions(save_path)

    @staticmethod
    def _copy_non_weight_assets(source: Path, destination: Path) -> None:
        """Copy index/config assets without copying the source weight shards."""

        for item in source.iterdir():
            if item.is_file() and item.suffix == ".safetensors":
                continue
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            elif item.is_file():
                shutil.copy2(item, target)

    @staticmethod
    def _validate_converted_count(converted: int, direction: str) -> None:
        if converted != EXPECTED_GATED_UP_GATE_WEIGHTS:
            raise RuntimeError(
                f"MagiHuman {direction} converted {converted} gated up-gate weights; "
                f"expected {EXPECTED_GATED_UP_GATE_WEIGHTS}"
            )
