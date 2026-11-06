#!/usr/bin/env python3
"""HF Kimi-K3.1 -> native, physically sharded FSDP2+EP DCP, on CPU.

Self-contained CPU converter with built-in HF validation and model-specific
name/shape adaptation. No full-DCP intermediate or distributed/NPU initialization.
Each expert job holds only one EP group, not all experts in a layer. A committed
file has full bytewise write/readback verification and a SHA256 resume receipt.
Only a complete conversion publishes release/.metadata and the release tracker.
"""
from __future__ import annotations
import argparse
import copy
import re
import shutil
import struct
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import time
import uuid

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from torch.distributed.checkpoint.metadata import (
    Metadata, MetadataIndex, TensorProperties, TensorStorageMetadata, ChunkStorageMetadata,
)
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.utils import _create_file_view


DTYPES = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16}
# HF mapping and source validation (CPU-only).
NUM_LAYERS = 93
NUM_EXPERTS = 896
NUM_VISION_BLOCKS = 27
INDEX_NAME = "model.safetensors.index.json"
LAYER_KEY_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.")


_LAYER_NAMES = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attention_res_norm.weight", "self_attention_res_proj.weight",
    "mlp_res_norm.weight", "mlp_res_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "self_attn.A_log", "self_attn.dt_bias",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight",
    "self_attn.f_a_proj.weight", "self_attn.f_b_proj.weight", "self_attn.b_proj.weight",
    "self_attn.o_norm.weight", "self_attn.g_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight", "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.indexer.wq_b.weight", "self_attn.indexer.wk.weight",
    "self_attn.indexer.weights_proj.weight", "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
    "block_sparse_moe.gate.weight", "block_sparse_moe.gate.e_score_correction_bias",
    "block_sparse_moe.routed_expert_up_proj.weight",
    "block_sparse_moe.routed_expert_down_proj.weight",
    "block_sparse_moe.routed_expert_norm.weight",
    "block_sparse_moe.shared_experts.gate_proj.weight",
    "block_sparse_moe.shared_experts.up_proj.weight",
    "block_sparse_moe.shared_experts.down_proj.weight",
)


_NAMES = (
    "language_model.model.embed_tokens.weight", "language_model.lm_head.weight",
    "language_model.model.norm.weight", "language_model.model.output_attn_res_norm.weight",
    "language_model.model.output_attn_res_proj.weight",
    "mm_projector.post_norm.weight", "mm_projector.proj.0.weight", "mm_projector.proj.2.weight",
    "vision_tower.encoder.final_layernorm.weight", "vision_tower.patch_embed.proj.weight",
    "vision_tower.patch_embed.pos_emb.weight",
)


_VISION_NAMES = ("wqkv.weight", "wo.weight", "norm0.weight", "norm1.weight",
                 "mlp.fc0.weight", "mlp.fc1.weight")


WEIGHT_MAPPING = {key: key for key in _NAMES}


WEIGHT_MAPPING.update({f"language_model.model.layers.{{i}}.{name}":
                       f"language_model.model.layers.{{i}}.{name}" for name in _LAYER_NAMES})


WEIGHT_MAPPING.update({f"vision_tower.encoder.blocks.{{b}}.{name}":
                       f"vision_tower.encoder.blocks.{{b}}.{name}" for name in _VISION_NAMES})


EXPERT_W_MAPPING = {"w1": "gate_up_proj", "w2": "down_proj", "w3": "gate_up_proj"}


EXPERT_SUFFIXES = ("weight",)


EXPERT_KEY_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$"
)


def build_mapping(num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS,
                  num_vision_blocks=NUM_VISION_BLOCKS):
    mapping = {}
    for old, new in WEIGHT_MAPPING.items():
        if "{i}" in old:
            mapping.update((old.format(i=i), new.format(i=i)) for i in range(num_layers))
        elif "{b}" in old:
            mapping.update((old.format(b=b), new.format(b=b)) for b in range(num_vision_blocks))
        else:
            mapping[old] = new
    # Expert conversion is many-to-one and cannot be represented as a rename.
    return mapping


def to_train_key(key, mapping):
    # Never silently pass an unknown / quantized key to a BF16 training model.
    if key not in mapping:
        raise ValueError(f"Unsupported non-expert K3.1 key: {key}. "
                         "Per-expert weights must go through tensor fusion; "
                         "K3 weight_packed/weight_scale is not supported.")
    return mapping[key]


def _expert_key(layer, projection):
    return f"language_model.model.layers.{layer}.block_sparse_moe.experts.{projection}"


def _expert_filename(layer, projection):
    return f"experts-layer{layer:05d}-{projection}.safetensors"


def _target_config(source, num_layers=None, num_experts=None):
    target = copy.deepcopy(source)
    text = target["text_config"]
    for option, name, requested in (("num_layers", "num_hidden_layers", num_layers),
                                    ("num_experts", "num_experts", num_experts)):
        if requested is not None:
            if isinstance(requested, bool) or not isinstance(requested, int) or not 1 <= requested <= text[name]:
                raise ValueError(f"{option} must be between 1 and {text[name]}, got {requested}")
            text[name] = requested
    layers, experts = text["num_hidden_layers"], text["num_experts"]
    for name in ("linear_attn_config", "dsa_config"):
        for key, values in (text.get(name) or {}).items():
            if key.endswith("_layers") and isinstance(values, list):
                text[name][key] = [i for i in values if 1 <= i <= layers]
    if "num_experts_per_token" in text:
        text["num_experts_per_token"] = min(text["num_experts_per_token"], experts)
    group_count = text.get("num_expert_group", 1)
    if experts % group_count or group_count > experts:
        raise ValueError(f"num_experts={experts} must be divisible by num_expert_group={group_count}")
    return target


def _select_index(index, config):
    text = config["text_config"]
    selected = {}
    for key, filename in index["weight_map"].items():
        layer = LAYER_KEY_RE.match(key)
        if layer and int(layer.group(1)) >= text["num_hidden_layers"]:
            continue
        expert = EXPERT_KEY_RE.fullmatch(key)
        if expert and int(expert.group(2)) >= text["num_experts"]:
            continue
        selected[key] = filename
    return {**index, "metadata": dict(index.get("metadata", {})), "weight_map": selected}


def _build_expert_index(index, text_config):
    experts = defaultdict(lambda: defaultdict(dict))
    for key, filename in index["weight_map"].items():
        match = EXPERT_KEY_RE.fullmatch(key)
        if match:
            layer, expert, part = match.groups()
            experts[int(layer)][int(expert)][part] = (filename, key)
    expected_layers = {i for i in range(text_config["first_k_dense_replace"],
                                       text_config["num_hidden_layers"])
                       if i % text_config.get("moe_layer_freq", 1) == 0}
    if set(experts) != expected_layers:
        raise ValueError(f"Missing/unexpected MoE layers: {set(experts) ^ expected_layers}")
    for layer, layer_experts in experts.items():
        if set(layer_experts) != set(range(text_config["num_experts"])):
            raise ValueError(f"Layer {layer}: missing/unexpected expert IDs")
        for expert, parts in layer_experts.items():
            if set(parts) != {"w1", "w2", "w3"}:
                raise ValueError(f"Layer {layer}, expert {expert}: missing w1/w2/w3")
    return experts


def _converted_weight_map(index, mapping):
    converted = {}
    for key, filename in index["weight_map"].items():
        match = EXPERT_KEY_RE.fullmatch(key)
        if match:
            layer, _, part = match.groups()
            projection = EXPERT_W_MAPPING[part]
            converted[_expert_key(int(layer), projection)] = _expert_filename(int(layer), projection)
        else:
            target = to_train_key(key, mapping)
            if target in converted:
                raise ValueError(f"Duplicate target key: {target}")
            converted[target] = filename
    return dict(sorted(converted.items()))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_header(path):
    # Reading just the header avoids loading multi-GB tensors for validation.
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > 100_000_000 or size + 8 > path.stat().st_size:
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(size))
    return {k: v for k, v in header.items() if k != "__metadata__"}, size + 8


def _a_log_padding_bytes(path, key, info, data_start, text):
    heads = (text.get("linear_attn_config") or {}).get("num_heads")
    if not key.endswith(".self_attn.A_log") or heads is None:
        return 0
    if info["shape"] == [heads]:
        return 0
    padded = 1 << (heads - 1).bit_length()
    if info["shape"] != [padded]:
        raise ValueError(f"{key}: unexpected A_log shape {info['shape']}; expected [{heads}]")
    width = 4 if info["dtype"] == "F32" else 2
    removed = (padded - heads) * width
    with path.open("rb") as stream:
        stream.seek(data_start + info["data_offsets"][0] + heads * width)
        tail = stream.read(removed)
    # Accept positive/negative floating zero, reject all meaningful tail data.
    mask = (1 << (8 * width - 1)) - 1
    if len(tail) != removed or any(int.from_bytes(tail[i:i+width], "little") & mask
                                   for i in range(0, removed, width)):
        raise ValueError(f"{key}: A_log padding contains nonzero data; refusing to truncate")
    return removed


def _removed_tensor_bytes(path, key, info, data_start, text):
    removed = _a_log_padding_bytes(path, key, info, data_start, text)
    if key.endswith((".block_sparse_moe.gate.weight", ".block_sparse_moe.gate.e_score_correction_bias")):
        shape = info["shape"]
        expected_tail = [text["hidden_size"]] if key.endswith(".gate.weight") else []
        if shape[1:] != expected_tail or shape[0] < text["num_experts"]:
            raise ValueError(f"{key}: incompatible router shape {shape}")
        width = 4 if info["dtype"] == "F32" else 2
        removed += (shape[0]-text["num_experts"]) * math.prod(shape[1:]) * width
    return removed


def _adapt_non_expert(key, tensor, text):
    if key.endswith((".block_sparse_moe.gate.weight", ".block_sparse_moe.gate.e_score_correction_bias")):
        if tensor.shape[0] < text["num_experts"]:
            raise ValueError(f"{key}: too few router rows")
        if tensor.shape[0] > text["num_experts"]:
            return tensor[:text["num_experts"]].clone()
    heads = (text.get("linear_attn_config") or {}).get("num_heads")
    if key.endswith(".self_attn.A_log") and heads is not None and tensor.shape != (heads,):
        padded = 1 << (heads - 1).bit_length()
        if tensor.shape != (padded,) or tensor[heads:].count_nonzero().item():
            raise ValueError(f"{key}: incompatible/nonzero A_log padding")
        return tensor[:heads].clone()
    return tensor


def inspect_checkpoint(src_dir, num_layers=None, num_experts=None):
    """Validate selected tensors and their source shard headers; return the plan."""
    src_dir = Path(src_dir).resolve()
    source_index = _read_json(src_dir / INDEX_NAME)
    config = _target_config(_read_json(src_dir / "config.json"), num_layers, num_experts)
    index = _select_index(source_index, config)
    weight_map = index["weight_map"]
    if not weight_map:
        raise ValueError("Empty weight_map")
    text = config["text_config"]
    mapping = build_mapping(text["num_hidden_layers"], text["num_experts"],
                            config["vision_config"]["vt_num_hidden_layers"])
    experts = _build_expert_index(index, text)
    expert_dtypes = {}
    expert_hidden = text.get("routed_expert_hidden_size") or text["hidden_size"]
    intermediate = text["moe_intermediate_size"]
    groups = defaultdict(set)
    targets = set()
    for key, filename in weight_map.items():
        if EXPERT_KEY_RE.fullmatch(key) is None:
            target = to_train_key(key, mapping)
            if target in targets:
                raise ValueError(f"Duplicate target key: {target}")
            targets.add(target)
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Invalid shard filename: {filename}")
        groups[filename].add(key)
    actual_files = {p.name for p in src_dir.glob("*.safetensors")}
    source_files = set(source_index["weight_map"].values())
    if actual_files != source_files:
        raise ValueError(f"Shard/index mismatch: missing={sorted(set(groups)-actual_files)[:5]}, "
                         f"unindexed={sorted(actual_files-source_files)[:5]}")
    source_groups = defaultdict(set)
    for key, filename in source_index["weight_map"].items():
        if filename in groups:
            source_groups[filename].add(key)
    total_size = 0
    removed_padding = 0
    dtypes = Counter()
    for filename, expected in sorted(groups.items()):
        path = src_dir / filename
        header, data_start = _read_header(path)
        if set(header) != source_groups[filename]:
            raise ValueError(f"{filename}: header/index key mismatch; "
                             f"missing={sorted(source_groups[filename]-header.keys())[:5]}, "
                             f"extra={sorted(header.keys()-source_groups[filename])[:5]}")
        offset = 0
        for key, info in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
            dtype = info["dtype"]
            if dtype not in {"BF16", "F32", "F16"}:
                raise ValueError(f"{key}: unsupported training dtype {dtype}")
            start, end = info["data_offsets"]
            nbytes = math.prod(info["shape"]) * (4 if dtype == "F32" else 2)
            if start != offset or end - start != nbytes:
                raise ValueError(f"{filename}: invalid tensor offsets/shape for {key}")
            offset = end
            if key not in expected:
                continue
            removed_padding += _removed_tensor_bytes(path, key, info, data_start, text)
            match = EXPERT_KEY_RE.fullmatch(key)
            if match:
                layer, _, part = match.groups()
                shape = ([expert_hidden, intermediate] if part == "w2"
                         else [intermediate, expert_hidden])
                if info["shape"] != shape:
                    raise ValueError(f"{key}: expected {shape}, got {info['shape']}")
                if expert_dtypes.setdefault(layer, dtype) != dtype:
                    raise ValueError(f"Layer {layer}: mixed expert dtypes cannot be fused losslessly")
            total_size += nbytes
            dtypes[dtype] += 1
        if data_start + offset != path.stat().st_size:
            raise ValueError(f"{filename}: tensor payload size mismatch")
    declared = index.get("metadata", {}).get("total_size")
    if len(weight_map) == len(source_index["weight_map"]) and declared is not None and declared != total_size:
        raise ValueError(f"total_size mismatch: index={declared}, headers={total_size}")
    index["metadata"]["total_size"] = total_size
    print(f"Target selection: first {text['num_hidden_layers']} layers, "
          f"first {text['num_experts']} routed experts per MoE layer", flush=True)
    print(f"Validated {len(weight_map)} tensors, {len(groups)} shards, "
          f"{total_size} bytes; dtypes={dict(dtypes)}", flush=True)
    print(f"Fused target: {len(_converted_weight_map(index, mapping))} model tensors; "
          f"{len(experts)} MoE layers; gate_up_proj="
          f"[{text['num_experts']}, {expert_hidden}, {2*intermediate}], "
          f"down_proj=[{text['num_experts']}, {intermediate}, {expert_hidden}]", flush=True)
    print(f"A_log padding/router rows removed: {removed_padding} bytes; "
          f"target tensor bytes={total_size-removed_padding}", flush=True)
    return index, mapping, groups


def copy_non_weight_files(src_dir, dst_dir, config=None):
    # Preserve the supplied config/tokenizer/processor, without copying manifests,
    # hidden temporary files, other weights, or a misleading HF index into DCP.
    for path in Path(src_dir).iterdir():
        if (path.is_file() and not path.name.startswith(".") and path.name != INDEX_NAME
                and path.suffix in {".json", ".py", ".model", ".txt", ".jinja", ".tiktoken"}
                and "manifest" not in path.name.lower()):
            shutil.copy2(path, Path(dst_dir) / path.name)
    if config is not None:
        _write_json(Path(dst_dir) / "config.json", config)
        source_text = _read_json(Path(src_dir) / "config.json")["text_config"]
        _write_json(Path(dst_dir) / "conversion_config.json", {
            "source_dir": str(Path(src_dir).resolve()),
            "source_num_layers": source_text["num_hidden_layers"],
            "source_num_experts": source_text["num_experts"],
            "num_layers": config["text_config"]["num_hidden_layers"],
            "num_experts": config["text_config"]["num_experts"],
            "layer_selection": "first N layers, starting at 0",
            "expert_selection": "first E routed experts in each MoE layer, starting at 0",
        })


# Native FSDP2/EP DCP writer.
def digest_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def syncdir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with tmp.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        syncdir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, obj):
    atomic(path, (json.dumps(obj, indent=2, sort_keys=True) + '\n').encode())


@contextmanager
def lock(dst):
    import fcntl
    with (dst / '.conversion.lock').open('a+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def identity(path):
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def parts(size, n):
    width = (size + n - 1) // n
    return [(i, i * width, min(width, size - i * width)) for i in range(n) if i * width < size]


def chunks(shape, fsdp, ep=1, expert=False):
    if not shape:
        return [{'offsets': [], 'sizes': [], 'rank': 0}]
    result = []
    if expert:
        for r1, o1, n1 in parts(shape[1], fsdp // ep):
            for r0, o0, n0 in parts(shape[0], ep):
                result.append(dict(offsets=[o0, o1, 0], sizes=[n0, n1, shape[2]], rank=r1 * ep + r0))
    else:
        for rank, off, size in parts(shape[0], fsdp):
            result.append(dict(offsets=[off] + [0] * (len(shape) - 1),
                               sizes=[size] + shape[1:], rank=rank))
    if sum(math.prod(c['sizes']) for c in result) != math.prod(shape):
        raise ValueError('Chunk coverage error')
    return result


def plan(src, fsdp, ep):
    if fsdp <= 0 or ep <= 0 or fsdp % ep:
        raise ValueError('fsdp-shards must be positive and divisible by ep-size')
    config = _read_json(src / 'config.json')
    text = config['text_config']
    if text['num_experts'] % ep:
        raise ValueError('num_experts must be divisible by ep-size for the training EP mesh')
    index, mapping, groups = inspect_checkpoint(src)
    specs, files, entries = {}, {}, {}
    for filename in sorted(groups):
        path = src / filename
        header, base = _read_header(path)
        files[filename] = identity(path)
        for key in groups[filename]:
            info = header[key]
            entries[key] = dict(file=filename, offset=base + info['data_offsets'][0],
                                length=info['data_offsets'][1]-info['data_offsets'][0],
                                shape=info['shape'], dtype=info['dtype'])
    jobs = []
    for filename in sorted(groups):
        tensors = []
        for key in sorted(groups[filename]):
            if EXPERT_KEY_RE.fullmatch(key):
                continue
            entry = entries[key]
            shape = list(entry['shape'])
            if key.endswith('.self_attn.A_log'):
                shape = [text['linear_attn_config']['num_heads']]
            if key.endswith(('.block_sparse_moe.gate.weight', '.block_sparse_moe.gate.e_score_correction_bias')):
                shape[0] = text['num_experts']
            name = 'model.' + to_train_key(key, mapping)
            specs[name] = dict(shape=shape, dtype=entry['dtype'], chunks=chunks(shape, fsdp))
            tensors.append(dict(name=name, key=key, source=entry, **specs[name]))
        if tensors:
            jobs.append(dict(kind='ordinary', tensors=tensors))
    hidden = text.get('routed_expert_hidden_size') or text['hidden_size']
    inter, count = text['moe_intermediate_size'], text['num_experts']
    experts = _build_expert_index(index, text)
    for layer, layer_experts in sorted(experts.items()):
        dtype = entries[layer_experts[0]['w1'][1]]['dtype']
        names = ['model.' + _expert_key(layer, p) for p in ('gate_up_proj', 'down_proj')]
        for name, shape in zip(names, ([count, hidden, 2 * inter], [count, inter, hidden])):
            specs[name] = dict(shape=shape, dtype=dtype, chunks=chunks(shape, fsdp, ep, True))
        for ep_rank, start, length in parts(count, ep):
            selected = [[entries[layer_experts[e][p][1]] for p in ('w1', 'w2', 'w3')]
                        for e in range(start, start + length)]
            tensors = [dict(name=name, **{**specs[name], 'chunks': [c for c in specs[name]['chunks']
                                                                  if c['offsets'][0] == start]}) for name in names]
            jobs.append(dict(kind='expert', start=start, sources=selected, tensors=tensors))
    for i, job in enumerate(jobs):
        job['id'] = i
        job['bytes'] = sum(math.prod(c['sizes']) * (4 if t['dtype'] == 'F32' else 2)
                           for t in job['tensors'] for c in t['chunks'])
    total = sum(j['bytes'] for j in jobs)
    return config, files, specs, jobs, total


def read_tensor(src, entry, pool):
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
        list(pool.map(read_range, range(0, len(buf), 4 << 20)))
        return torch.frombuffer(buf, dtype=DTYPES[entry['dtype']]).reshape(entry['shape'])
    finally:
        os.close(fd)


def byte_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)))


def emit(dest, tmp, name, spec, local):
    # Clone prevents torch.save from serializing the backing full/group storage.
    local = local.clone(memory_format=torch.contiguous_format)
    offset = dest.tell()
    torch.save(local, dest)
    length = dest.tell() - offset
    dest.flush()
    with tmp.open('rb') as f:
        loaded = torch.load(_create_file_view(f, offset, length), weights_only=True, map_location='cpu')
    if not byte_equal(local, loaded):
        raise ValueError(f'Write/readback mismatch: {name}')
    return dict(name=name, **spec, offset=offset, length=length)


def convert_job(src_s, release_s, job, text, source_ids, fingerprint, read_threads):
    torch.set_num_threads(1)
    started = time.monotonic()
    src, release = Path(src_s), Path(release_s)
    filename = f'__hf_{job["id"]:06d}.distcp'
    dst = release / filename
    receipt_path = release / '.receipts' / f'{job["id"]:06d}.json'
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    tmp = dst.with_name(dst.name + '.tmp.' + uuid.uuid4().hex)
    def check_sources():
        for file, expected in source_ids.items():
            if identity(src / file) != expected:
                raise ValueError(f'Source changed: {file}')
    check_sources()
    out = []
    try:
        with ThreadPoolExecutor(max_workers=read_threads) as pool, tmp.open('xb') as dest:
            if job['kind'] == 'ordinary':
                for t in job['tensors']:
                    value = _adapt_non_expert(t['key'], read_tensor(src, t['source'], pool), text)
                    if list(value.shape) != t['shape'] or value.dtype != DTYPES[t['dtype']]:
                        raise ValueError(f'Unexpected ordinary tensor: {t["name"]}')
                    for c in t['chunks']:
                        local = value
                        for dim, (off, size) in enumerate(zip(c['offsets'], c['sizes'])):
                            local = local.narrow(dim, off, size)
                        out.append(emit(dest, tmp, t['name'], c, local))
                    del value, local
            else:
                count = len(job['sources'])
                gate, down = [torch.empty([count] + t['shape'][1:], dtype=DTYPES[t['dtype']])
                              for t in job['tensors']]
                inter = down.shape[1]
                for e, sources in enumerate(job['sources']):
                    for p, entry in enumerate(sources):
                        value = read_tensor(src, entry, pool)
                        if p == 1:
                            down[e].copy_(value.t())
                        else:
                            begin = 0 if p == 0 else inter
                            gate[e, :, begin:begin+inter].copy_(value.t())
                        del value
                for value, t in zip((gate, down), job['tensors']):
                    for c in t['chunks']:
                        local = value[:, c['offsets'][1]:c['offsets'][1]+c['sizes'][1], :]
                        out.append(emit(dest, tmp, t['name'], c, local))
                del gate, down, value, local
            dest.flush()
            os.fsync(dest.fileno())
        check_sources()
        size, digest = tmp.stat().st_size, sha(tmp)
        os.replace(tmp, dst)
        syncdir(release)
        receipt = dict(id=job['id'], fingerprint=fingerprint, job_sha256=digest_json(job),
                       file=filename, file_size=size, file_sha256=digest, chunks=out,
                       logical_bytes=job['bytes'], elapsed_s=time.monotonic()-started, verify='full')
        receipt['receipt_sha256'] = digest_json(receipt)
        write_json(receipt_path, receipt)
        return receipt
    finally:
        tmp.unlink(missing_ok=True)


def validate_receipt(release, job, fingerprint, verify_hash):
    r = json.loads((release / '.receipts' / f'{job["id"]:06d}.json').read_text())
    body = {k: v for k, v in r.items() if k != 'receipt_sha256'}
    if (r['receipt_sha256'] != digest_json(body) or r['fingerprint'] != fingerprint
            or r['job_sha256'] != digest_json(job) or r['verify'] != 'full'
            or r['id'] != job['id'] or r['logical_bytes'] != job['bytes']
            or r['file'] != f'__hf_{job["id"]:06d}.distcp'):
        raise ValueError(f'Invalid receipt: {job["id"]}')
    path = release / r['file']
    if path.stat().st_size != r['file_size'] or (verify_hash and sha(path) != r['file_sha256']):
        raise ValueError(f'Output corrupted: {path}')
    expected = [dict(name=t['name'], **c) for t in job['tensors'] for c in t['chunks']]
    if len(expected) != len(r['chunks']):
        raise ValueError('Receipt chunk count mismatch')
    cursor = 0
    for e, c in zip(expected, r['chunks']):
        if any(c[k] != v for k, v in e.items()) or c['offset'] != cursor or c['length'] <= 0:
            raise ValueError('Receipt chunk layout mismatch')
        cursor += c['length']
    if cursor != r['file_size']:
        raise ValueError('Receipt file extent mismatch')
    return r


def assemble(specs, receipts):
    md = Metadata(state_dict_metadata={}, planner_data={}, storage_data={})
    storage = {}
    for r in receipts:
        for c in r['chunks']:
            key = (c['name'], tuple(c['offsets']))
            if key in storage:
                raise ValueError('Duplicate physical chunk')
            storage[key] = _StorageInfo(relative_path=r['file'], offset=c['offset'], length=c['length'])
    for name, t in specs.items():
        cs = [ChunkStorageMetadata(torch.Size(c['offsets']), torch.Size(c['sizes'])) for c in t['chunks']]
        md.state_dict_metadata[name] = TensorStorageMetadata(
            TensorProperties(dtype=DTYPES[t['dtype']]), torch.Size(t['shape']), cs)
        md.planner_data[name] = ('model', name.removeprefix('model.'))
        for i, c in enumerate(cs):
            md.storage_data[MetadataIndex(name, c.offsets, i)] = storage.pop((name, tuple(c.offsets)))
    if storage:
        raise ValueError('Unexpected saved chunks')
    return md


def run(args):
    src, dst = args.src_dir.resolve(), args.dst_dir.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError('Source/destination must be separate directories')
    if dst.exists() and (not dst.is_dir() or (any(dst.iterdir()) and not args.resume)):
        raise ValueError('Destination must be a directory; nonempty destination requires --resume')
    config, files, specs, jobs, total = plan(src, args.fsdp_shards, args.ep_size)
    recipe = dict(version=2, source=str(src), index_sha256=sha(src / INDEX_NAME),
                  config_sha256=sha(src / 'config.json'), files=files,
                  converter_sha256=sha(Path(__file__)),
                  torch_version=str(torch.__version__), fsdp=args.fsdp_shards, ep=args.ep_size,
                  plan_sha256=digest_json(jobs))
    fingerprint = digest_json(recipe)
    summary = dict(source=str(src), output=str(dst), shard_count=args.fsdp_shards,
                   ep_size=args.ep_size, expert_fsdp_size=args.fsdp_shards//args.ep_size,
                   num_layers=config['text_config']['num_hidden_layers'],
                   num_experts=config['text_config']['num_experts'], model_tensors=len(specs),
                   jobs=len(jobs), total_logical_bytes=total,
                   tensor_chunks=sum(len(t['chunks']) for t in specs.values()), fingerprint=fingerprint)
    print(json.dumps(summary), flush=True)
    dst.mkdir(parents=True, exist_ok=True)
    with lock(dst):
        release = dst / 'release'
        manifest = dst / 'conversion_recipe.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != recipe:
                raise ValueError('Resume recipe/source identity mismatch')
        else:
            if any(p.name != '.conversion.lock' for p in dst.iterdir()):
                raise ValueError('Existing checkpoint without matching conversion recipe')
            write_json(manifest, recipe)
        release.mkdir(exist_ok=True)
        (release / '.receipts').mkdir(exist_ok=True)
        if not (release / '.metadata').exists():
            atomic(release / '.incomplete', b'Conversion in progress; do not load.\n')
        started, receipts, done_bytes = time.monotonic(), {}, 0
        def progress(phase):
            status = dict(summary, phase=phase, completed_jobs=len(receipts),
                          completed_logical_bytes=done_bytes, elapsed_s=time.monotonic()-started,
                          updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
            write_json(dst / 'progress.json', status)
            print(json.dumps(status), flush=True)
        progress('resume_validation' if args.resume else 'convert')
        pending = []
        for job in jobs:
            if (release / '.receipts' / f'{job["id"]:06d}.json').exists():
                r = validate_receipt(release, job, fingerprint, True)
                receipts[job['id']] = r
                done_bytes += job['bytes']
                progress('resume_validation')
            else:
                pending.append(job)
        if (release / '.metadata').exists() and pending:
            raise ValueError('Published metadata but incomplete receipts')
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {}
            try:
                for job in pending:
                    entries = ([t['source'] for t in job['tensors']] if job['kind'] == 'ordinary'
                               else [e for row in job['sources'] for e in row])
                    subset = {e['file']: files[e['file']] for e in entries}
                    f = pool.submit(convert_job, str(src), str(release), job, config['text_config'],
                                    subset, fingerprint, args.read_threads)
                    futures[f] = job
                for f in as_completed(futures):
                    r = f.result()
                    receipts[r['id']] = r
                    done_bytes += r['logical_bytes']
                    progress('convert')
            except BaseException:
                for f in futures:
                    f.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                progress('failed')
                raise
        for file, expected in files.items():
            if identity(src / file) != expected:
                raise ValueError(f'Source changed before publish: {file}')
        if sha(src / INDEX_NAME) != recipe['index_sha256'] or sha(src / 'config.json') != recipe['config_sha256']:
            raise ValueError('Source index/config changed')
        ordered = [validate_receipt(release, job, fingerprint, False) for job in jobs]
        md = assemble(specs, ordered)
        data = pickle.dumps(md, protocol=pickle.HIGHEST_PROTOCOL)
        copy_non_weight_files(src, dst, config=config)
        report = dict(summary, schema_version=1, completed=True, verify='full',
                      output_files=len(ordered), output_bytes=sum(r['file_size'] for r in ordered),
                      output_metadata_sha256=hashlib.sha256(data).hexdigest(),
                      config=dict(fsdp=args.fsdp_shards, ep=args.ep_size,
                                  expert_fsdp=args.fsdp_shards//args.ep_size))
        write_json(release / 'reshard_manifest.json', report)
        # Native DCP metadata is the commit point; tracker comes strictly last.
        (release / '.incomplete').unlink(missing_ok=True)
        syncdir(release)
        atomic(release / '.metadata', data)
        atomic(dst / 'latest_checkpointed_iteration.txt', b'release\n')
        progress('complete')
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src-dir', type=Path, required=True)
    p.add_argument('--dst-dir', type=Path, required=True)
    p.add_argument('--fsdp-shards', type=int, default=128)
    p.add_argument('--ep-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--read-threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.workers <= 0 or args.read_threads <= 0:
        p.error('workers and read-threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
