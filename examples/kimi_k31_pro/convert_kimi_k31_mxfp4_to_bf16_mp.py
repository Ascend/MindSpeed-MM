#!/usr/bin/env python3
"""Dequantize a native Kimi-K31 MXFP4 checkpoint to BF16 weights (multiprocess)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_INDEX = "model.safetensors.index.json"
OUTPUT_INDEX = "model.safetensors.index.json"
MANIFEST_DIR = ".conversion-manifests"
GROUP_SIZE = 32

FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
FP4_PAIR_TABLE = torch.stack(
    (
        FP4_TABLE[torch.arange(256, dtype=torch.uint8).bitwise_and(0x0F).long()],
        FP4_TABLE[torch.arange(256, dtype=torch.uint8).bitwise_right_shift(4).long()],
    ),
    dim=-1,
)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight_packed"):
        raise ValueError(f"not an MXFP4 packed weight: {weight_name}")
    return weight_name.removesuffix(".weight_packed") + ".weight_scale"


def output_name(weight_name: str) -> str:
    return weight_name.removesuffix("_packed")


def decode_mxfp4(
    packed_weight: torch.Tensor,
    exponent_scale: torch.Tensor,
) -> torch.Tensor:
    """Decode packed E2M1 values with one E8M0 scale per 32 values."""
    if packed_weight.dtype != torch.uint8 or packed_weight.ndim != 2:
        raise ValueError(
            "expected a 2D uint8 MXFP4 packed weight, got "
            f"dtype={packed_weight.dtype} shape={tuple(packed_weight.shape)}"
        )
    if exponent_scale.dtype != torch.uint8 or exponent_scale.ndim != 2:
        raise ValueError(
            "expected a 2D uint8 E8M0 scale, got "
            f"dtype={exponent_scale.dtype} shape={tuple(exponent_scale.shape)}"
        )

    out_features, packed_in_features = packed_weight.shape
    in_features = packed_in_features * 2
    if in_features % GROUP_SIZE:
        raise ValueError(
            f"input size {in_features} is not divisible by group size {GROUP_SIZE}"
        )
    expected_scale_shape = (out_features, in_features // GROUP_SIZE)
    if tuple(exponent_scale.shape) != expected_scale_shape:
        raise ValueError(
            "unexpected MXFP4 scale shape: "
            f"weight={tuple(packed_weight.shape)} "
            f"scale={tuple(exponent_scale.shape)} "
            f"expected={expected_scale_shape}"
        )

    values = FP4_PAIR_TABLE[packed_weight.long()].reshape(out_features, in_features)
    scales = torch.exp2(exponent_scale.float() - 127.0)
    decoded = values.reshape(out_features, -1, GROUP_SIZE) * scales.unsqueeze(-1)
    return decoded.reshape(out_features, in_features).to(torch.bfloat16)


def validate_quantization_config(config: dict[str, object]) -> None:
    text_config = config.get("text_config")
    quant_config = config.get("quantization_config")
    if quant_config is None and isinstance(text_config, dict):
        quant_config = text_config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise ValueError("config.json does not contain a quantization_config")

    if quant_config.get("quant_method") != "compressed-tensors":
        raise ValueError("expected quant_method=compressed-tensors")
    if quant_config.get("format") != "mxfp4-pack-quantized":
        raise ValueError("expected format=mxfp4-pack-quantized")

    groups = quant_config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("quantization_config.config_groups is missing")
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        if not isinstance(weights, dict):
            continue
        if (
            weights.get("group_size") == GROUP_SIZE
            and weights.get("num_bits") == 4
            and weights.get("scale_dtype") == "torch.uint8"
            and weights.get("symmetric") is True
            and weights.get("type") == "float"
        ):
            return
    raise ValueError("no supported group_size=32 MXFP4 weight group found")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")
    os.replace(temporary, path)


def copy_sidecars(source: Path, output: Path) -> None:
    skip_names = {
        SOURCE_INDEX,
        MANIFEST_DIR,
        "k3_bf16_conversion_manifest.json",
    }
    for item in source.iterdir():
        if item.name in skip_names or item.name == ".git":
            continue
        if item.is_file() and item.suffix == ".safetensors":
            continue
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)

    config_path = output / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    config.pop("quantization_config", None)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config.pop("quantization_config", None)
    write_json(config_path, config)


class SourceShardWriter:
    def __init__(
        self,
        output: Path,
        source_id: int,
        max_shard_bytes: int,
    ) -> None:
        self.output = output
        self.source_id = source_id
        self.max_shard_bytes = max_shard_bytes
        self.tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.current_bytes = 0
        self.total_bytes = 0
        self.parts: list[dict[str, object]] = []
        self.weight_map: dict[str, str] = {}

    def add(self, name: str, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous().cpu()
        nbytes = tensor_nbytes(tensor)
        if self.tensors and self.current_bytes + nbytes > self.max_shard_bytes:
            self.flush()
        self.tensors[name] = tensor
        self.current_bytes += nbytes
        self.total_bytes += nbytes

    def flush(self) -> None:
        if not self.tensors:
            return
        part_id = len(self.parts) + 1
        filename = (
            f"model-bf16-src{self.source_id:05d}-part{part_id:05d}.safetensors"
        )
        destination = self.output / filename
        temporary = destination.with_name(destination.name + ".tmp")
        save_file(dict(self.tensors), temporary, metadata={"format": "pt"})
        os.replace(temporary, destination)

        for name in self.tensors:
            self.weight_map[name] = filename
        self.parts.append(
            {
                "file": filename,
                "file_size": destination.stat().st_size,
                "tensor_bytes": self.current_bytes,
                "tensor_count": len(self.tensors),
            }
        )
        self.tensors.clear()
        self.current_bytes = 0

    def finish(self) -> None:
        self.flush()


def validate_pairs(weight_map: dict[str, str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    output_names = set(weight_map)
    for name, shard_name in weight_map.items():
        if not name.endswith(".weight_packed"):
            continue
        paired_scale = scale_name(name)
        scale_shard = weight_map.get(paired_scale)
        if scale_shard is None:
            raise ValueError(f"missing MXFP4 scale for {name}")
        if scale_shard != shard_name:
            raise ValueError(f"MXFP4 pair crosses source shards: {name}")
        if output_name(name) in output_names:
            raise ValueError(f"output tensor name already exists: {output_name(name)}")
        pairs[name] = paired_scale
    if not pairs:
        raise ValueError("no .weight_packed MXFP4 tensors found")
    return pairs


def source_manifest_is_valid(
    manifest: dict[str, object],
    source_path: Path,
    output: Path,
) -> bool:
    if manifest.get("source_file") != source_path.name:
        return False
    if manifest.get("source_size") != source_path.stat().st_size:
        return False
    if manifest.get("source_mtime_ns") != source_path.stat().st_mtime_ns:
        return False
    parts = manifest.get("parts")
    if not isinstance(parts, list):
        return False
    for part in parts:
        if not isinstance(part, dict) or not isinstance(part.get("file"), str):
            return False
        part_path = output / part["file"]
        if not part_path.is_file() or part_path.stat().st_size != part.get("file_size"):
            return False
    return isinstance(manifest.get("weight_map"), dict)


def clear_source_outputs(output: Path, source_id: int) -> None:
    prefix = f"model-bf16-src{source_id:05d}-part"
    for path in output.glob(prefix + "*.safetensors*"):
        path.unlink()


def convert_source_shard(
    source_path: Path,
    source_id: int,
    keys: list[str],
    pairs: dict[str, str],
    output: Path,
    max_shard_bytes: int,
) -> dict[str, object]:
    writer = SourceShardWriter(output, source_id, max_shard_bytes)
    scales_to_skip = set(pairs.values())
    converted_count = 0
    skipped_scale_count = 0
    started = time.time()

    with safe_open(source_path, framework="pt", device="cpu") as shard:
        for name in keys:
            if name in scales_to_skip:
                skipped_scale_count += 1
                continue
            if name in pairs:
                tensor = decode_mxfp4(
                    shard.get_tensor(name),
                    shard.get_tensor(pairs[name]),
                )
                writer.add(output_name(name), tensor)
                converted_count += 1
            else:
                writer.add(name, shard.get_tensor(name))
    writer.finish()

    stat = source_path.stat()
    return {
        "converter_version": 1,
        "source_file": source_path.name,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "elapsed_s": time.time() - started,
        "converted_count": converted_count,
        "skipped_scale_count": skipped_scale_count,
        "total_size": writer.total_bytes,
        "parts": writer.parts,
        "weight_map": writer.weight_map,
    }


def expected_output_names(
    source_names: set[str],
    pairs: dict[str, str],
) -> set[str]:
    scales = set(pairs.values())
    return {
        output_name(name) if name in pairs else name
        for name in source_names
        if name not in scales
    }


def _convert_one_shard_job(
    source: str,
    output: str,
    source_id: int,
    shard_name: str,
    keys: list[str],
    pairs: dict[str, str],
    max_shard_bytes: int,
) -> tuple[int, str, dict[str, object]]:
    """Worker entry: convert one source shard and persist its manifest."""
    source_path = Path(source) / shard_name
    output_path = Path(output)
    manifest_path = output_path / MANIFEST_DIR / f"source-{source_id:05d}.json"
    clear_source_outputs(output_path, source_id)
    manifest = convert_source_shard(
        source_path,
        source_id,
        keys,
        pairs,
        output_path,
        max_shard_bytes,
    )
    write_json(manifest_path, manifest)
    return source_id, shard_name, manifest


def convert_checkpoint(
    source: Path,
    output: Path,
    max_shard_bytes: int,
    overwrite: bool,
    resume: bool,
    workers: int,
) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("source and output directories must differ")
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    with (source / SOURCE_INDEX).open() as f:
        source_index = json.load(f)
    source_weight_map: dict[str, str] = source_index["weight_map"]
    pairs = validate_pairs(source_weight_map)
    with (source / "config.json").open() as f:
        source_config = json.load(f)
    validate_quantization_config(source_config)

    if output.exists() and overwrite:
        shutil.rmtree(output)
    elif output.exists() and not resume:
        raise FileExistsError(
            f"{output} exists; pass --resume or --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifests = output / MANIFEST_DIR
    manifests.mkdir(exist_ok=True)
    copy_sidecars(source, output)

    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard_name in source_weight_map.items():
        keys_by_shard[shard_name].append(name)

    source_shards = sorted(keys_by_shard)
    shard_count = len(source_shards)
    manifests_by_id: dict[int, dict[str, object]] = {}
    pending: list[tuple[int, str, list[str]]] = []
    started = time.time()

    for source_id, shard_name in enumerate(source_shards, start=1):
        source_path = source / shard_name
        manifest_path = manifests / f"source-{source_id:05d}.json"
        manifest: dict[str, object] | None = None
        if resume and manifest_path.is_file():
            with manifest_path.open() as f:
                candidate = json.load(f)
            if source_manifest_is_valid(candidate, source_path, output):
                manifest = candidate

        if manifest is not None:
            manifests_by_id[source_id] = manifest
            print(
                f"[{source_id}/{shard_count}] {shard_name}: "
                f"converted={manifest['converted_count']} "
                f"parts={len(manifest['parts'])} (resumed)",
                flush=True,
            )
        else:
            pending.append((source_id, shard_name, keys_by_shard[shard_name]))

    pool_workers = min(workers, len(pending)) if pending else 1
    if pending and pool_workers == 1:
        for source_id, shard_name, keys in pending:
            _, _, manifest = _convert_one_shard_job(
                str(source),
                str(output),
                source_id,
                shard_name,
                keys,
                pairs,
                max_shard_bytes,
            )
            manifests_by_id[source_id] = manifest
            print(
                f"[{source_id}/{shard_count}] {shard_name}: "
                f"converted={manifest['converted_count']} "
                f"parts={len(manifest['parts'])}",
                flush=True,
            )
    elif pending:
        print(
            f"converting {len(pending)} shards with workers={pool_workers}",
            flush=True,
        )
        ctx = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=pool_workers,
            mp_context=ctx,
        ) as executor:
            futures = {
                executor.submit(
                    _convert_one_shard_job,
                    str(source),
                    str(output),
                    source_id,
                    shard_name,
                    keys,
                    pairs,
                    max_shard_bytes,
                ): (source_id, shard_name)
                for source_id, shard_name, keys in pending
            }
            for future in as_completed(futures):
                source_id, shard_name, manifest = future.result()
                manifests_by_id[source_id] = manifest
                print(
                    f"[{source_id}/{shard_count}] {shard_name}: "
                    f"converted={manifest['converted_count']} "
                    f"parts={len(manifest['parts'])}",
                    flush=True,
                )

    all_manifests = [manifests_by_id[i] for i in range(1, shard_count + 1)]

    output_weight_map: dict[str, str] = {}
    total_size = 0
    converted_count = 0
    skipped_scale_count = 0
    for manifest in all_manifests:
        manifest_weight_map = manifest["weight_map"]
        if not isinstance(manifest_weight_map, dict):
            raise ValueError("invalid source manifest weight_map")
        duplicates = output_weight_map.keys() & manifest_weight_map.keys()
        if duplicates:
            raise ValueError(f"duplicate output tensors: {sorted(duplicates)[:5]}")
        output_weight_map.update(manifest_weight_map)
        total_size += int(manifest["total_size"])
        converted_count += int(manifest["converted_count"])
        skipped_scale_count += int(manifest["skipped_scale_count"])

    expected_names = expected_output_names(set(source_weight_map), pairs)
    actual_names = set(output_weight_map)
    if actual_names != expected_names:
        raise ValueError(
            "output tensor set mismatch: "
            f"missing={sorted(expected_names - actual_names)[:5]} "
            f"unexpected={sorted(actual_names - expected_names)[:5]}"
        )

    write_json(
        output / OUTPUT_INDEX,
        {
            "metadata": {"total_size": total_size},
            "weight_map": output_weight_map,
        },
    )
    report = {
        "converter_version": 1,
        "source": str(source),
        "output": str(output),
        "workers": workers,
        "source_tensor_count": len(source_weight_map),
        "source_metadata_total_size": source_index.get("metadata", {}).get(
            "total_size"
        ),
        "converted_mxfp4_tensor_count": converted_count,
        "skipped_mxfp4_scale_count": skipped_scale_count,
        "output_tensor_count": len(output_weight_map),
        "output_total_size": total_size,
        "output_part_count": sum(len(manifest["parts"]) for manifest in all_manifests),
        "elapsed_s": time.time() - started,
        "accuracy_note": (
            "MXFP4 dequantization is lossy and does not reconstruct the original "
            "pre-quantization BF16 weights."
        ),
    }
    write_json(output / "k3_bf16_conversion_manifest.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dequantize Kimi-K3.1 compressed-tensors MXFP4 weights to BF16 "
            "(multiprocess over source shards)."
        )
    )
    parser.add_argument("--source", default="/models/tmp/tmp/Kimi-K31")
    parser.add_argument(
        "--output",
        default="/models/tmp/tmp/k3work/Kimi-K31-BF16-dequantized",
    )
    parser.add_argument("--max-shard-gb", type=float, default=4.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel processes over source shards (default: 4).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_shard_gb <= 0:
        raise ValueError("--max-shard-gb must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    report = convert_checkpoint(
        Path(args.source),
        Path(args.output),
        int(args.max_shard_gb * 1024**3),
        overwrite=args.overwrite,
        resume=args.resume,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
