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

from collections import defaultdict
from collections.abc import MutableMapping
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import logging
import math
import os
from pathlib import Path
import re
import struct
from typing import Dict, Generator, List, Optional, Set, Tuple

from pydantic import FilePath
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from mindspeed_mm.fsdp import envs
from mindspeed_mm.fsdp.checkpoint.utils import remove_base_layer_keys
from mindspeed_mm.fsdp.utils.utils import tensor_to_dtensor_local
from mindspeed_mm.fsdp.log import print_rank

logger = logging.getLogger(__name__)


DTYPES = {
    'BOOL': torch.bool, 'U8': torch.uint8, 'I8': torch.int8, 'I16': torch.int16,
    'I32': torch.int32, 'I64': torch.int64, 'F16': torch.float16,
    'BF16': torch.bfloat16, 'F32': torch.float32, 'F64': torch.float64,
    'C64': torch.complex64, 'C128': torch.complex128,
}


def identity(path):
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def read_tensor(src, entry, pool=None):
    """Read one CPU tensor, optionally using a caller-owned IO thread pool."""
    path = src / entry['file']
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = bytearray(entry['length'])

        def read_range(start):
            view = memoryview(buf)[start:min(start + (4 << 20), len(buf))]
            pos = entry['offset'] + start
            while view:
                n = os.preadv(fd, [view], pos)
                if n == 0:
                    raise EOFError(f'Short read: {path}')
                pos += n
                view = view[n:]

        read_map = map if pool is None else pool.map
        list(read_map(read_range, range(0, len(buf), 4 << 20)))
        return torch.frombuffer(buf, dtype=DTYPES[entry['dtype']]).reshape(entry['shape'])
    finally:
        os.close(fd)


class LazyExpertWeights(MutableMapping):
    """Resolve expert keys on demand without caching source tensor payloads."""

    def __init__(self, entries, src, pool):
        self.entries, self.src, self.pool = dict(entries), src, pool

    def __contains__(self, key):
        # Mapping's default membership would call __getitem__ and read all keys.
        return key in self.entries

    def __getitem__(self, key):
        return read_tensor(self.src, self.entries[key], self.pool)

    def __delitem__(self, key):
        del self.entries[key]

    def __setitem__(self, key, value):
        raise TypeError('Expert sources are read-only')

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_safetensors_header(path):
    """Return the tensor entries and payload start offset without reading weights."""
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f'Truncated safetensors header: {path}')
        size = struct.unpack('<Q', raw)[0]
        if size > 100_000_000 or size + 8 > path.stat().st_size:
            raise ValueError(f'Invalid safetensors header: {path}')
        header = json.loads(stream.read(size))
    return {k: v for k, v in header.items() if k != '__metadata__'}, size + 8


def _read_hf_index(src):
    if (src / SAFE_WEIGHTS_INDEX_NAME).is_file():
        return _read_json(src / SAFE_WEIGHTS_INDEX_NAME)
    if (src / SAFE_WEIGHTS_NAME).is_file():
        header, _ = read_safetensors_header(src / SAFE_WEIGHTS_NAME)
        return {'weight_map': {key: SAFE_WEIGHTS_NAME for key in header}}
    raise FileNotFoundError(f'No HF safetensors index or model.safetensors in {src}')


def read_hf_tensor_shape(src, key, cache):
    """Resolve original shapes lazily; cache only the index and per-file shapes."""
    if 'weight_map' not in cache:
        cache['weight_map'] = _read_hf_index(src)['weight_map']
    filename = cache['weight_map'].get(key)
    if not filename or not filename.endswith('.safetensors'):
        raise ValueError(f'Cannot find original HF tensor shape: {key}')
    if filename not in cache:
        path = (src / filename).resolve()
        if not path.is_relative_to(src) or not path.is_file():
            raise ValueError(f'Cannot restore HF shape without source shard: {path}')
        header, _ = read_safetensors_header(path)
        cache[filename] = {name: entry['shape'] for name, entry in header.items()}
    if key not in cache[filename]:
        raise ValueError(f'Cannot find original HF tensor shape: {key}')
    return list(cache[filename][key])


def inspect_source(src):
    """Inspect all source metadata without materializing model tensors."""
    index = _read_hf_index(src)
    groups = defaultdict(set)
    for key, filename in index['weight_map'].items():
        path = (src / filename).resolve()
        if not path.is_relative_to(src) or path == src:
            raise ValueError(f'Source file outside HF directory: {filename}')
        groups[filename].add(key)
    if not groups:
        raise ValueError('Empty HF checkpoint')
    entries, files, total = {}, {}, 0
    for filename, expected in sorted(groups.items()):
        path = src / filename
        before = identity(path)
        header, base = read_safetensors_header(path)
        if set(header) != expected:
            raise ValueError(f'{filename}: header/index key mismatch')
        cursor = 0
        for key, info in sorted(header.items(), key=lambda item: item[1]['data_offsets'][0]):
            dtype, shape = info['dtype'], info['shape']
            if dtype not in DTYPES or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(f'{key}: unsupported dtype or shape')
            start, end = info['data_offsets']
            length = math.prod(shape) * DTYPES[dtype].itemsize
            if type(start) is not int or type(end) is not int or start != cursor or end - start != length:
                raise ValueError(f'{key}: invalid safetensors offsets/shape')
            entries[key] = dict(file=filename, offset=base + start, length=length, shape=shape, dtype=dtype)
            cursor = end
            total += length
        if base + cursor != before[0] or identity(path) != before:
            raise ValueError(f'{filename}: source changed or payload size mismatch')
        files[filename] = before
    declared = index.get('metadata', {}).get('total_size')
    if declared is not None and declared != total:
        raise ValueError('HF index total_size mismatch')
    return entries, files


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


# Diffusers-style weight file names, used as a fallback by the HF locator so a
# single locator covers both ecosystems (e.g. Wan2.2 diffusers checkpoints).
DIFFUSERS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
DIFFUSERS_WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"


def locate_hf_weight_files(weights_path: str) -> List[HFWeightFileStream]:
    """Resolve the safetensors files under *weights_path* into stream readers.

    Supports both HF and diffusers layouts:
      - single ``model.safetensors`` / ``diffusion_pytorch_model.safetensors``
      - sharded files described by the corresponding ``*.index.json``
      - direct path to a .safetensors file
    """
    if os.path.isfile(weights_path) and weights_path.endswith(".safetensors"):
        return [HFWeightFileStream(weights_path)]

    for weights_name, index_name in (
        (SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME),
        (DIFFUSERS_WEIGHTS_NAME, DIFFUSERS_WEIGHTS_INDEX_NAME),
    ):
        single = os.path.join(weights_path, weights_name)
        if os.path.isfile(single):
            return [HFWeightFileStream(single)]

        index = os.path.join(weights_path, index_name)
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

    raise ValueError(f"No HF safetensors or diffusers weights found under {weights_path}.")


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


def _target_tensor(model: torch.nn.Module, name: str) -> torch.Tensor:
    leaf_module, local_name = _resolve_leaf(model, name)
    if local_name in leaf_module._parameters:
        return leaf_module._parameters[local_name].data
    elif local_name in leaf_module._buffers:
        return leaf_module._buffers[local_name]
    raise ValueError(f"'{name}' is neither a parameter nor a buffer of the model.")


def write_full_tensor(model: torch.nn.Module, name: str, full_tensor: torch.Tensor) -> None:
    """Write a full parameter/buffer, slicing on CPU before copying to its device."""
    target = _target_tensor(model, name)

    if hasattr(target, "device_mesh"):  # sharded -> a DTensor
        # Slice on CPU first, then move only this rank's shard to the NPU.
        full_tensor = tensor_to_dtensor_local(full_tensor, target.device_mesh, target.placements)
        # Return the local tensor shard stored on the current rank.
        target.to_local().copy_(full_tensor.to(device=target.device, dtype=target.dtype))
    else:  # plain tensor (replicated on every rank)
        target.copy_(full_tensor.to(device=target.device, dtype=target.dtype))


def supports_hf_group_loading(weight_transform) -> bool:
    """Opt in only transforms implementing the offline expert-group protocol."""
    return all(callable(getattr(weight_transform, name, None))
               for name in ('describe_hf_tensors', 'convert_expert_group'))


def _local_tensor_region(target: torch.Tensor) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return this rank's rectangular region using the actual DTensor mesh.

    Partial and strided/interleaved shards cannot be represented by one region
    and must fail rather than silently load the wrong experts.
    """
    if isinstance(target, DTensor):
        mesh = target.device_mesh
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            raise ValueError('Current rank is outside the target parameter mesh')
        offsets, sizes = _tensor_region(target.shape, mesh, target.placements, coordinate)
        if tuple(sizes) != tuple(target.to_local().shape):
            raise ValueError('DTensor local shape disagrees with its mesh/placements')
        return offsets, sizes
    return (0,) * target.ndim, tuple(target.shape)


def _tensor_region(shape, mesh, placements, coordinate):
    """Compute a contiguous region for an arbitrary coordinate in a real mesh."""
    offsets, sizes = [0] * len(shape), list(shape)
    for axis, placement in enumerate(placements):
        if type(placement) is Replicate:
            continue
        if type(placement) is not Shard:
            raise ValueError(f'Local HF loading requires contiguous Shard/Replicate: {placement}')
        dim = placement.dim % len(shape)
        # Match torch.chunk without depending on version-specific private APIs.
        width = (sizes[dim] + mesh.size(axis) - 1) // mesh.size(axis)
        offset = min(coordinate[axis] * width, sizes[dim])
        offsets[dim] += offset
        sizes[dim] = min(width, sizes[dim] - offset)
    return tuple(offsets), tuple(sizes)


def _expert_read_group(target):
    """Use an existing FSDP/replication axis whose members own the same experts.

    Multiple non-expert axes need a multidimensional distribution protocol;
    those layouts retain independent, bounded local reads for compatibility.
    """
    if not isinstance(target, DTensor):
        return None
    axes = [axis for axis, placement in enumerate(target.placements)
            if target.device_mesh.size(axis) > 1 and
            (type(placement) is Replicate or
             (type(placement) is Shard and placement.dim % target.ndim != 0))]
    return target.device_mesh.get_group(axes[0]) if len(axes) == 1 else None


def _check_hf_group_error(error, group, status):
    """Make local IO/allocation/copy failures fail every member before more P2P."""
    status.fill_(int(error is not None))
    dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
    if status.item():
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
        raise RuntimeError('HF expert group loading failed: ' + next(e for e in errors if e is not None))


def _load_expert_projection(target, item, entries, source, pool, weight_transform, region):
    """Read/convert once per EP group's FSDP axis, sending one local slice at a time."""
    offsets, sizes = region
    local_target = target.to_local() if isinstance(target, DTensor) else target
    group = _expert_read_group(target)
    rank = dist.get_rank() if group is not None else 0
    peers = [(rank, offsets, sizes)]
    if group is not None:
        mesh = target.device_mesh
        peers = []
        for peer in dist.get_process_group_ranks(group):
            coordinate = (mesh.mesh == peer).nonzero().flatten().tolist()
            peer_offsets, peer_sizes = _tensor_region(target.shape, mesh, target.placements, coordinate)
            if (peer_offsets[0], peer_sizes[0]) != (offsets[0], sizes[0]):
                raise ValueError('HF reader group members must own the same expert range')
            peers.append((peer, peer_offsets, peer_sizes))
    leader = peers[0][0]
    max_numel = max(math.prod(peer_sizes[1:]) for _, _, peer_sizes in peers)
    if not sizes[0] or not max_numel:
        return

    status = torch.zeros(1, dtype=torch.int32, device=local_target.device) if group is not None else None
    staging, error = None, None
    if group is not None:
        try:
            count = max_numel if rank == leader else math.prod(sizes[1:])
            staging = torch.empty(count, dtype=local_target.dtype, device=local_target.device)
        except Exception as exc:
            error = f'{item["target"]}: staging allocation: {type(exc).__name__}: {exc}'
        _check_hf_group_error(error, group, status)

    for local_index in range(sizes[0]):
        value, error = None, None
        if rank == leader:
            try:
                keys = item['keys'][offsets[0] + local_index]
                weights = LazyExpertWeights({key: entries[key] for key in keys}, source, pool)
                value = torch.empty((1, *item['shape'][1:]), dtype=DTYPES[item['dtype']], device='cpu')
                weight_transform.convert_expert_group(item['target'], tuple(keys), weights, value)
                if weights:
                    raise ValueError(f'Unconsumed expert sources: {item["target"]}')
                del weights
            except Exception as exc:
                if group is None:
                    raise
                error = f'{item["target"]}: expert {offsets[0] + local_index}: {type(exc).__name__}: {exc}'
        if group is not None:
            _check_hf_group_error(error, group, status)

        copy_error = None
        for peer, peer_offsets, peer_sizes in peers:
            count = math.prod(peer_sizes[1:])
            if not count:
                continue  # All members compute the same empty destination list.
            error, local = None, None
            if rank == leader:
                try:
                    local = value
                    for dim in range(1, len(peer_sizes)):
                        local = local.narrow(dim, peer_offsets[dim], peer_sizes[dim])
                    if group is not None:
                        # copy_ performs dtype conversion and a blocking CPU->device copy;
                        # no list of all device shards or full device projection is built.
                        staging[:count].view(1, *peer_sizes[1:]).copy_(local)
                except Exception as exc:
                    if group is None:
                        raise
                    error = f'{item["target"]}: preparing rank {peer}: {type(exc).__name__}: {exc}'
            if group is not None:
                _check_hf_group_error(error, group, status)
                if peer != leader:
                    if rank == leader:
                        dist.send(staging[:count], dst=peer, group=group)
                    elif rank == peer:
                        dist.recv(staging, src=leader, group=group)
            if rank == peer:
                try:
                    received = staging[:count].view(1, *peer_sizes[1:]) if group is not None else local
                    local_target[local_index:local_index + 1].copy_(received)
                    del received
                except Exception as exc:
                    if group is None:
                        raise
                    copy_error = f'{item["target"]}: writing rank {rank}: {type(exc).__name__}: {exc}'
            del local
        if group is not None:
            _check_hf_group_error(copy_error, group, status)
        del value


def _load_hf_weights_local(model, hf_dir, enable_lora, load_strict, weight_transform):
    """Reuse offline descriptors and fusion, reading only local expert sources.

    Within the local EP range, a FSDP group leader fuses one expert projection
    at a time and sends local slices to its peers. This bounds the fusion buffer
    even for EP=1 or replicated tensors. Ordinary parameters retain the existing
    full-tensor conversion and CPU slicing behavior.
    """
    source = Path(hf_dir).resolve()
    entries, source_ids = inspect_source(source)
    description = weight_transform.describe_hf_tensors(entries)
    param_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    lora_base_map = _lora_base_key_map(param_names) if enable_lora else {}
    unexpected_keys, assigned, targets = set(), set(), set()
    prepared = []
    # Validate the full descriptor and target layout before reading payloads.
    descriptors = [(item, False) for item in description['ordinary']]
    descriptors.extend((item, True) for group in description['expert_groups']
                       for item in group['tensors'])
    for item, expert in descriptors:
        keys = [key for row in item['keys'] for key in row] if expert else [item['key']]
        if expert and (len(item['keys']) != item['shape'][0] or any(not row for row in item['keys'])):
            raise ValueError(f'Expert descriptor shape/key mismatch: {item["target"]}')
        for key in keys:
            if key not in entries or key in assigned or entries[key]['dtype'] != item['dtype']:
                raise ValueError(f'Missing, duplicate or inconsistent HF source: {key}')
            assigned.add(key)
        name = lora_base_map.get(item['target'], item['target'])
        if name in targets:
            raise ValueError(f'Duplicate HF target: {name}')
        targets.add(name)
        if name not in param_names and name not in buffer_names:
            unexpected_keys.add(name)
            continue
        target = _target_tensor(model, name)
        if tuple(target.shape) != tuple(item['shape']):
            raise ValueError(f'HF/model shape mismatch: {name}: {item["shape"]} != {list(target.shape)}')
        region = _local_tensor_region(target) if expert else None
        prepared.append((item, expert, name, target, region))
    if assigned != set(entries):
        raise ValueError(f'Unconverted HF sources: {sorted(set(entries) - assigned)[:6]}')

    shared = sum(expert and _expert_read_group(target) is not None
                 for _, expert, _, target, _ in prepared)
    total_experts = sum(expert for _, expert, _, _, _ in prepared)
    print_rank(logger.info, f'HF expert projections: {shared} using group readers, '
               f'{total_experts - shared} using independent bounded reads')
    with ThreadPoolExecutor(max_workers=1) as pool:
        for item, expert, name, target, region in tqdm(
            prepared, desc='Loading local HF parameters', disable=envs.get('LOCAL_RANK', -1) > 0,
        ):
            if not expert:
                converted = weight_transform.hf_to_dcp(item['key'], read_tensor(source, entries[item['key']], pool))
                if (converted is None or converted[0] != item['target']
                        or tuple(converted[1].shape) != tuple(item['shape'])
                        or converted[1].dtype != DTYPES[item['dtype']]):
                    raise ValueError(f'Unexpected ordinary HF conversion: {name}')
                write_full_tensor(model, name, converted[1])
                del converted
            else:
                _load_expert_projection(target, item, entries, source, pool, weight_transform, region)
            param_names.discard(name)
    for filename, expected in source_ids.items():
        if identity(source / filename) != expected:
            raise ValueError(f'HF source changed during loading: {filename}')
    # describe_hf_tensors validates the complete expert set. The streaming
    # finalize hook tracks full projections and does not apply to group loads.
    _log_unexpected_keys(unexpected_keys)
    post_process_after_load(missing_param_keys=param_names, load_strict=load_strict)


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

    Transforms implementing the expert-group protocol use a reader per FSDP
    group and distribute local expert slices. Other models retain the existing
    full-tensor streaming path without new communication.

    Assumes ``model`` has already been laid out by ``fully_shard`` and brought
    out of meta via ``to_empty_if_needed`` -- i.e. parameter slots are empty
    DTensors on the real device and buffer values have been preserved.
    """
    if supports_hf_group_loading(weight_transform):
        return _load_hf_weights_local(model, hf_dir, enable_lora, load_strict, weight_transform)

    param_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    lora_base_map = _lora_base_key_map(param_names) if enable_lora else {}
    unexpected_keys: Set[str] = set()

    for shard_stream in tqdm(
        locate_hf_weight_files(hf_dir),
        desc="Loading checkpoint shards",
        disable=envs.get("LOCAL_RANK", -1) > 0,
    ):
        for raw_key, full_tensor in shard_stream:
            key = convert_weight_key(raw_key, model)
            if weight_transform is not None:
                converted = weight_transform.hf_to_dcp(key, full_tensor)
                if converted is None:
                    continue
                key, full_tensor = converted
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

    finalize = getattr(weight_transform, 'finalize_hf_to_dcp', None)
    if finalize is not None:
        finalize()
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
    save_ckpt_dtype: Optional[torch.dtype] = None,
    enable_lora: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build a state dict suitable for HuggingFace safetensors saving.
    """
    # dcp_checkpointer imports hf_utils through broadcast_utils; defer this dependency.
    from mindspeed_mm.fsdp.checkpoint.dcp_checkpointer import LoraModelState, ModelState

    model_state_cls = LoraModelState if enable_lora else ModelState
    return model_state_cls(model, save_ckpt_dtype).state_dict()


@torch.no_grad()
def merge_lora_weights(
    state_dict: Dict[str, torch.Tensor],
    scaling: float,
) -> Dict[str, torch.Tensor]:
    """Merge the default LoRA adapter into its base weights.

    For a base linear weight ``W`` with shape ``[out_features, in_features]``,
    PEFT stores ``lora_A`` with shape ``[rank, in_features]`` and ``lora_B``
    with shape ``[out_features, rank]``.  The adapter update therefore has the
    same shape as ``W`` and is computed as ``lora_B @ lora_A``.

    FSDP2 can represent ``lora_A`` as a sharded DTensor.  The matrix product
    must use the complete ``[rank, in_features]`` matrix (the rank dimension
    is needed to contract with ``lora_B``), so ``lora_A`` is redistributed to
    replicated placement before calculating the update.
    """
    suffix_a = ".lora_A.default.weight"
    suffix_b = ".lora_B.default.weight"

    for lora_a_key in [key for key in state_dict if key.endswith(suffix_a)]:
        prefix = lora_a_key.removesuffix(suffix_a)
        lora_b_key = f"{prefix}{suffix_b}"
        base_key = f"{prefix}.weight"

        base = state_dict[base_key]
        lora_a = state_dict[lora_a_key]
        lora_b = state_dict[lora_b_key]
        if isinstance(lora_a, DTensor):
            # A sharded lora_A contains only part of the rank dimension on
            # each rank; aggregate it before the B @ A multiplication.
            lora_a = lora_a.redistribute(
                placements=[Replicate() for _ in range(lora_a.device_mesh.ndim)]
            )
        delta = torch.matmul(lora_b, lora_a)

        state_dict[base_key] = base + scaling * delta
        del state_dict[lora_a_key]
        del state_dict[lora_b_key]

    return state_dict


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
