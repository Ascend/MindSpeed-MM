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

"""CPU-only physical FSDP2 + EP checkpoint layout and native DCP storage.

Model-specific key mapping, projection fusion and source IO belong to callers.
No process group or model construction is needed for offline conversion.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import uuid

import torch
from torch.distributed.checkpoint.metadata import (
    Metadata, MetadataIndex, TensorProperties, TensorStorageMetadata, ChunkStorageMetadata,
)
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.utils import _create_file_view

from mindspeed_mm.fsdp.checkpoint.hf_utils import DTYPES


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


def byte_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)))


def emit(dest, tmp, name, spec, local):
    """Write one CPU chunk, read it back byte-for-byte, and return its extent."""
    if local.device.type != 'cpu' or local.layout != torch.strided:
        raise ValueError('Offline DCP conversion requires dense CPU tensors')
    if list(local.shape) != list(spec['sizes']):
        raise ValueError(f'Chunk shape mismatch: {name}')
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


def commit_file(release, tmp, job, fingerprint, records, *, file_prefix='__chunk_'):
    """Commit a flushed/fsynced chunk file before publishing its integrity receipt."""
    filename = f'{file_prefix}{job["id"]:06d}.distcp'
    size, digest = tmp.stat().st_size, sha(tmp)
    os.replace(tmp, release / filename)
    syncdir(release)
    receipt = dict(id=job['id'], fingerprint=fingerprint, job_sha256=digest_json(job),
                   file=filename, file_size=size, file_sha256=digest, chunks=records,
                   logical_bytes=job['bytes'], verify='full')
    receipt['receipt_sha256'] = digest_json(receipt)
    write_json(release / '.receipts' / f'{job["id"]:06d}.json', receipt)
    return receipt


def publish(release, metadata, report):
    """Publish validated metadata first and the release tracker last."""
    data = pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)
    report = dict(report, completed=True, output_metadata_sha256=hashlib.sha256(data).hexdigest())
    atomic(release / '.metadata', data)
    write_json(release / 'reshard_manifest.json', report)
    (release / '.incomplete').unlink(missing_ok=True)
    syncdir(release)
    atomic(release.parent / 'latest_checkpointed_iteration.txt', b'release\n')
    return report


def validate_receipt(release, job, fingerprint, verify_hash=True, file_prefix='__chunk_'):
    """Validate a committed file against its job, recipe, layout and SHA256."""
    r = json.loads((release / '.receipts' / f'{job["id"]:06d}.json').read_text())
    body = {k: v for k, v in r.items() if k != 'receipt_sha256'}
    if (r['receipt_sha256'] != digest_json(body) or r['fingerprint'] != fingerprint
            or r['job_sha256'] != digest_json(job) or r['verify'] != 'full'
            or r['id'] != job['id'] or r['logical_bytes'] != job['bytes']
            or r['file'] != f'{file_prefix}{job["id"]:06d}.distcp'):
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


def parts(size, n):
    """Return nonempty (rank, offset, length) partitions using DTensor's ceil split."""
    if type(size) is not int or size < 0 or type(n) is not int or n <= 0:
        raise ValueError('size must be a nonnegative integer; shard count must be positive')
    width = (size + n - 1) // n
    return [(i, i * width, min(width, size - i * width))
            for i in range(n) if i * width < size]


@dataclass(frozen=True)
class FSDPEPLayout:
    """Physical layout independent of any model or HF key naming scheme.

    Ordinary tensors use ``fsdp_shards`` ranks. Experts use mesh
    ``[fsdp_shards // ep_size, ep_size]`` with placements
    ``(Shard(expert_shard_dim), Shard(expert_dim))``. Rank order is
    ``expert_fsdp_rank * ep_size + ep_rank``. Replicated ranks are omitted.
    """

    fsdp_shards: int
    ep_size: int = 1
    ordinary_shard_dim: int = 0
    expert_dim: int = 0
    expert_shard_dim: int = 1

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (self.fsdp_shards, self.ep_size)):
            raise ValueError('fsdp_shards and ep_size must be positive integers')
        if self.fsdp_shards % self.ep_size:
            raise ValueError('fsdp_shards must be divisible by ep_size')
        axes = (self.ordinary_shard_dim, self.expert_dim, self.expert_shard_dim)
        if any(type(d) is not int or d < 0 for d in axes):
            raise ValueError('Shard dimensions must be nonnegative integers')
        if self.expert_dim == self.expert_shard_dim:
            raise ValueError('Expert and expert FSDP dimensions must differ')

    @property
    def expert_fsdp_size(self):
        return self.fsdp_shards // self.ep_size

    def chunks(self, shape, *, expert=False):
        """Plan global offsets/sizes; scalars have one chunk, empty tensors are rejected."""
        shape = list(shape)
        if any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('Tensor dimensions must be positive integers; empty tensors are unsupported')
        if not shape and not expert:
            return [dict(offsets=[], sizes=[], rank=0)]
        axes = (self.expert_dim, self.expert_shard_dim) if expert else (self.ordinary_shard_dim,)
        if any(d >= len(shape) for d in axes):
            raise ValueError(f'Shard dimension outside tensor shape: {shape}')
        if expert and shape[self.expert_dim] % self.ep_size:
            raise ValueError('Expert count must be divisible by ep_size')
        result = []
        if expert:
            for rf, of, nf in parts(shape[self.expert_shard_dim], self.expert_fsdp_size):
                for re, oe, ne in parts(shape[self.expert_dim], self.ep_size):
                    offsets, sizes = [0] * len(shape), shape.copy()
                    offsets[self.expert_dim], sizes[self.expert_dim] = oe, ne
                    offsets[self.expert_shard_dim], sizes[self.expert_shard_dim] = of, nf
                    result.append(dict(offsets=offsets, sizes=sizes, rank=rf * self.ep_size + re))
        else:
            for rank, offset, length in parts(shape[self.ordinary_shard_dim], self.fsdp_shards):
                offsets, sizes = [0] * len(shape), shape.copy()
                offsets[self.ordinary_shard_dim], sizes[self.ordinary_shard_dim] = offset, length
                result.append(dict(offsets=offsets, sizes=sizes, rank=rank))
        return result


def chunks(shape, fsdp, ep=1, expert=False):
    """Convenience wrapper for the default ordinary dim0 / expert dim1+dim0 layout."""
    return FSDPEPLayout(fsdp, ep).chunks(shape, expert=expert)


def _validate_spec(name, spec):
    shape, planned = list(spec['shape']), spec['chunks']
    if spec['dtype'] not in DTYPES:
        raise ValueError(f'Unsupported dtype for {name}: {spec["dtype"]}')
    if not planned or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError(f'Invalid or empty tensor specification: {name}')
    active, seen, volume = [], set(), 0
    for c in sorted(planned, key=lambda c: tuple(c['offsets'])):
        offsets, sizes = list(c['offsets']), list(c['sizes'])
        if (len(offsets) != len(shape) or len(sizes) != len(shape)
                or any(type(o) is not int or type(s) is not int or o < 0 or s <= 0 or o + s > n
                       for o, s, n in zip(offsets, sizes, shape))):
            raise ValueError(f'Chunk outside tensor bounds: {name}')
        if tuple(offsets) in seen:
            raise ValueError(f'Duplicate chunk offsets: {name}')
        seen.add(tuple(offsets))
        if shape:
            active = [p for p in active if p['offsets'][0] + p['sizes'][0] > offsets[0]]
            for p in active:
                if all(o < po + ps and po < o + s
                       for o, s, po, ps in zip(offsets, sizes, p['offsets'], p['sizes'])):
                    raise ValueError(f'Overlapping chunks: {name}')
            active.append(c)
        volume += math.prod(sizes)
    if volume != math.prod(shape):
        raise ValueError(f'Incomplete chunk coverage: {name}')


def assemble(specs, receipts, state_dict_key='model'):
    """Build native DCP metadata from complete tensor plans and validated file receipts.

    ``specs`` uses flattened FQNs; ``state_dict_key=None`` writes a flat state dict.
    Receipt integrity/hash validation is the caller's responsibility before assembly.
    """
    if not specs:
        raise ValueError('Cannot publish an empty checkpoint')
    if state_dict_key is not None and (not isinstance(state_dict_key, str)
                                      or not state_dict_key or '.' in state_dict_key):
        raise ValueError('state_dict_key must be a single nonempty key or None')
    md = Metadata(state_dict_metadata={}, planner_data={}, storage_data={})
    storage = {}
    for r in receipts:
        for c in r['chunks']:
            key = (c['name'], tuple(c['offsets']))
            if key in storage:
                raise ValueError('Duplicate physical chunk')
            if c['offset'] < 0 or c['length'] <= 0 or c['offset'] + c['length'] > r['file_size']:
                raise ValueError('Physical chunk outside file bounds')
            storage[key] = (c, _StorageInfo(relative_path=r['file'], offset=c['offset'], length=c['length']))
    prefix = state_dict_key + '.' if state_dict_key is not None else ''
    for name, t in specs.items():
        if not isinstance(name, str) or not name.startswith(prefix) or not name[len(prefix):]:
            raise ValueError(f'Invalid flattened parameter name: {name}')
        _validate_spec(name, t)
        cs = [ChunkStorageMetadata(torch.Size(c['offsets']), torch.Size(c['sizes'])) for c in t['chunks']]
        md.state_dict_metadata[name] = TensorStorageMetadata(
            TensorProperties(dtype=DTYPES[t['dtype']]), torch.Size(t['shape']), cs)
        md.planner_data[name] = (state_dict_key, name[len(prefix):]) if prefix else (name,)
        for i, c in enumerate(cs):
            saved = storage.pop((name, tuple(c.offsets)), None)
            if saved is None or list(saved[0]['sizes']) != list(c.sizes):
                raise ValueError(f'Missing or mismatched saved chunk: {name}, {list(c.offsets)}')
            md.storage_data[MetadataIndex(name, c.offsets, i)] = saved[1]
    if storage:
        raise ValueError('Unexpected saved chunks')
    return md


def save_sharded_tensors(tensors, dst_dir, *, fsdp_shards, ep_size=1,
                         expert_keys=(), state_dict_key='model'):
    """Save mapped CPU tensors as a physically sharded, model-only release DCP.

    ``tensors`` is a mapping or a lazy iterable of (model_key, CPU tensor) pairs.
    Keys exclude the outer ``state_dict_key`` prefix. Mark expert tensors explicitly
    with ``expert_keys``; no model-specific name guessing is performed. One tensor
    is consumed at a time. For large fused experts, adapters can instead use
    ``FSDPEPLayout.chunks``, ``emit``, ``validate_receipt`` and ``assemble`` to
    generate one EP group at a time, as the Kimi adapter does.

    Output must be empty/new. This convenience API does not resume interrupted
    generators; adapters own source identity and resumable job scheduling.
    """
    layout = FSDPEPLayout(fsdp_shards, ep_size)
    if state_dict_key is not None and (not isinstance(state_dict_key, str)
                                      or not state_dict_key or '.' in state_dict_key):
        raise ValueError('state_dict_key must be a single nonempty key or None')
    experts = set(expert_keys)
    if any(not isinstance(k, str) or not k for k in experts):
        raise ValueError('expert_keys must contain nonempty model keys')
    dst = Path(dst_dir).resolve()
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise ValueError('Destination must be empty or nonexistent')
    dst.mkdir(parents=True, exist_ok=True)
    with lock(dst):
        if any(p.name != '.conversion.lock' for p in dst.iterdir()):
            raise ValueError('Destination changed before acquiring lock')
        release = dst / 'release'
        release.mkdir()
        (release / '.receipts').mkdir()
        atomic(release / '.incomplete', b'Conversion in progress; do not load.\n')
        recipe = dict(version=1, fsdp=fsdp_shards, ep=ep_size, state_dict_key=state_dict_key,
                      expert_keys=sorted(experts), sharding_sha256=sha(Path(__file__)),
                      torch_version=str(torch.__version__))
        fingerprint = digest_json(recipe)
        write_json(dst / 'conversion_recipe.json', recipe)
        specs, receipts, seen = {}, [], set()
        pairs = tensors.items() if isinstance(tensors, Mapping) else iter(tensors)
        dtype_names = {v: k for k, v in DTYPES.items()}
        for index, (key, tensor) in enumerate(pairs):
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError(f'Invalid or duplicate model key: {key}')
            seen.add(key)
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                    or tensor.layout != torch.strided or tensor.dtype not in dtype_names):
                raise ValueError(f'Expected a supported dense CPU tensor: {key}')
            name = f'{state_dict_key}.{key}' if state_dict_key is not None else key
            planned = layout.chunks(list(tensor.shape), expert=key in experts)
            spec = dict(shape=list(tensor.shape), dtype=dtype_names[tensor.dtype], chunks=planned)
            specs[name] = spec
            job = dict(id=index, tensors=[dict(name=name, **spec)],
                       bytes=tensor.numel() * tensor.element_size())
            filename = f'__chunk_{index:06d}.distcp'
            tmp = release / (filename + '.tmp.' + uuid.uuid4().hex)
            out = []
            try:
                with tmp.open('xb') as dest:
                    for c in planned:
                        local = tensor.detach()
                        for dim, (offset, length) in enumerate(zip(c['offsets'], c['sizes'])):
                            local = local.narrow(dim, offset, length)
                        out.append(emit(dest, tmp, name, c, local))
                    dest.flush()
                    os.fsync(dest.fileno())
                commit_file(release, tmp, job, fingerprint, out)
                receipts.append(validate_receipt(release, job, fingerprint, verify_hash=False))
            finally:
                tmp.unlink(missing_ok=True)
            del tensor, local
        if experts - seen:
            raise ValueError(f'Expert keys missing from input: {sorted(experts - seen)}')
        md = assemble(specs, receipts, state_dict_key=state_dict_key)
        report = dict(shard_count=fsdp_shards, ep_size=ep_size,
                      expert_fsdp_size=layout.expert_fsdp_size, model_tensors=len(specs),
                      tensor_chunks=len(md.storage_data), output_files=len(receipts),
                      logical_bytes=sum(r['logical_bytes'] for r in receipts), verify='full')
        return publish(release, md, report)
