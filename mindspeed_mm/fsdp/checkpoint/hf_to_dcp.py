#!/usr/bin/env python3
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

"""Convert HF weights to a native, physically sharded FSDP2+EP release DCP.

Model semantics come from WeightTransformPipeline and source IO from hf_utils;
this module owns EP-group scheduling, recovery and checkpoint publication.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import uuid

import torch

from mindspeed_mm.fsdp.checkpoint import convert as transforms
from mindspeed_mm.fsdp.checkpoint import hf_utils, sharded_dcp

_WORKER_PIPELINE = None


def plan(src, model_id, fsdp, ep):
    """Translate a model's tensor/group descriptors into reusable physical jobs."""
    src = Path(src).resolve()
    layout = sharded_dcp.FSDPEPLayout(fsdp, ep)
    pipeline = transforms.build_weight_transform(model_id, str(src))
    if not hf_utils.supports_hf_group_loading(pipeline):
        raise ValueError(f'Model {model_id!r} has no offline conversion descriptor')
    entries, files = hf_utils.inspect_source(src)
    description = pipeline.describe_hf_tensors(entries)
    specs, jobs, ordinary, assigned = {}, [], defaultdict(list), set()

    def use_keys(keys, dtype):
        for key in keys:
            if key not in entries or key in assigned:
                raise ValueError(f'Missing or multiply assigned HF key: {key}')
            if entries[key]['dtype'] != dtype:
                raise ValueError(f'Conversion would change dtype: {key}')
            assigned.add(key)

    def add_spec(tensor, expert=False):
        target = tensor['target']
        if not isinstance(target, str) or not target:
            raise ValueError('Invalid model target key')
        name = 'model.' + target
        if name in specs:
            raise ValueError(f'Duplicate target key: {target}')
        spec = dict(shape=list(tensor['shape']), dtype=tensor['dtype'],
                    chunks=layout.chunks(tensor['shape'], expert=expert))
        specs[name] = spec
        return dict(name=name, model_key=target, **spec)

    for tensor in description['ordinary']:
        key = tensor['key']
        use_keys([key], tensor['dtype'])
        ordinary[entries[key]['file']].append(dict(key=key, source=entries[key], **add_spec(tensor)))
    for filename in sorted(ordinary):
        jobs.append(dict(kind='ordinary', tensors=sorted(ordinary[filename], key=lambda t: t['key'])))
    for group in description['expert_groups']:
        tensors = group['tensors']
        if not tensors:
            raise ValueError('Empty expert group')
        count = tensors[0]['shape'][0]
        group_specs = []
        for tensor in tensors:
            if tensor['shape'][0] != count or len(tensor['keys']) != count:
                raise ValueError('Expert keys and global expert dimension disagree')
            if any(not row for row in tensor['keys']):
                raise ValueError('Expert without source keys')
            use_keys([key for row in tensor['keys'] for key in row], tensor['dtype'])
            group_specs.append(add_spec(tensor, expert=True))
        for _, start, length in sharded_dcp.parts(count, ep):
            job_tensors, sources = [], {}
            for tensor, full_spec in zip(tensors, group_specs):
                keys = [key for row in tensor['keys'][start:start + length] for key in row]
                sources.update((key, entries[key]) for key in keys)
                spec = {**full_spec, 'chunks': [c for c in full_spec['chunks'] if c['offsets'][0] == start]}
                job_tensors.append(dict(hf_keys=keys, **spec))
            jobs.append(dict(kind='expert', start=start, expert_count=length,
                             sources=sources, tensors=job_tensors))
    if assigned != set(entries):
        raise ValueError(f'Unconverted HF keys: {sorted(set(entries) - assigned)[:10]}')
    for index, job in enumerate(jobs):
        job['id'] = index
        job['bytes'] = sum(math.prod(c['sizes']) * hf_utils.DTYPES[t['dtype']].itemsize
                           for t in job['tensors'] for c in t['chunks'])
    return pipeline, files, specs, jobs, sum(j['bytes'] for j in jobs)


def _init_worker(model_id, src):
    global _WORKER_PIPELINE
    torch.set_num_threads(1)
    _WORKER_PIPELINE = transforms.build_weight_transform(model_id, src)


def convert_job(src_s, release_s, job, source_ids, fingerprint, read_threads):
    pipeline = _WORKER_PIPELINE
    if pipeline is None:
        raise RuntimeError('Conversion worker was not initialized')
    src, release = Path(src_s), Path(release_s)
    filename = f'__hf_{job["id"]:06d}.distcp'
    dst = release / filename
    receipt_path = release / '.receipts' / f'{job["id"]:06d}.json'
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    tmp = dst.with_name(dst.name + '.tmp.' + uuid.uuid4().hex)

    def check_sources():
        for file, expected in source_ids.items():
            if hf_utils.identity(src / file) != expected:
                raise ValueError(f'Source changed: {file}')

    check_sources()
    out = []
    try:
        with ThreadPoolExecutor(max_workers=read_threads) as pool, tmp.open('xb') as dest:
            weights = hf_utils.LazyExpertWeights(job['sources'], src, pool) if job['kind'] == 'expert' else None
            for tensor in job['tensors']:
                dtype = hf_utils.DTYPES[tensor['dtype']]
                if job['kind'] == 'ordinary':
                    converted = pipeline.hf_to_dcp(tensor['key'], hf_utils.read_tensor(src, tensor['source'], pool))
                    if converted is None or converted[0] != tensor['model_key']:
                        raise ValueError(f'Unexpected ordinary conversion: {tensor["name"]}')
                    value = converted[1]
                    del converted
                    expected_shape = tensor['shape']
                else:
                    expected_shape = [job['expert_count']] + tensor['shape'][1:]
                    value = torch.empty(expected_shape, dtype=dtype, device='cpu')
                    pipeline.convert_expert_group(tensor['model_key'], tuple(tensor['hf_keys']), weights, value)
                if list(value.shape) != expected_shape or value.dtype != dtype:
                    raise ValueError(f'Unexpected converted shape/dtype: {tensor["name"]}')
                for chunk in tensor['chunks']:
                    local = value
                    for dim, (offset, length) in enumerate(zip(chunk['offsets'], chunk['sizes'])):
                        if job['kind'] == 'expert' and dim == 0:
                            offset -= job['start']
                        local = local.narrow(dim, offset, length)
                    out.append(sharded_dcp.emit(dest, tmp, tensor['name'], chunk, local))
                del value, local
            if weights:
                raise ValueError('Unconsumed expert sources')
            dest.flush()
            os.fsync(dest.fileno())
        check_sources()
        return sharded_dcp.commit_file(release, tmp, job, fingerprint, out, file_prefix='__hf_')
    finally:
        tmp.unlink(missing_ok=True)


def _copy_assets(src, dst):
    control_files = {hf_utils.SAFE_WEIGHTS_INDEX_NAME, 'conversion_recipe.json', 'progress.json', 'verification_report.json',
                     'conversion_config.json', 'latest_checkpointed_iteration.txt'}
    for path in src.iterdir():
        if (path.is_file() and not path.name.startswith('.') and path.name not in control_files
                and path.suffix in {'.json', '.py', '.model', '.txt', '.jinja', '.tiktoken'}
                and 'manifest' not in path.name.lower()):
            shutil.copy2(path, dst / path.name)


def run(args):
    if args.workers <= 0 or args.read_threads <= 0:
        raise ValueError('workers and read_threads must be positive')
    src, dst = Path(args.src_dir).resolve(), Path(args.dst_dir).resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError('Source/destination must be separate directories')
    if dst.exists() and (not dst.is_dir() or (any(dst.iterdir()) and not args.resume)):
        raise ValueError('Destination must be a directory; nonempty destination requires --resume')
    index = src / hf_utils.SAFE_WEIGHTS_INDEX_NAME
    config = src / 'config.json'
    index_before = sharded_dcp.sha(index) if index.exists() else None
    config_before = sharded_dcp.sha(config)
    pipeline, files, specs, jobs, total = plan(src, args.model_id, args.fsdp_shards, args.ep_size)
    index_after = sharded_dcp.sha(index) if index.exists() else None
    if (index_before != index_after or config_before != sharded_dcp.sha(config)
            or pipeline.config != hf_utils._read_json(config)):
        raise ValueError('Source index/config changed during planning')
    modules = [Path(__file__), Path(transforms.__file__), Path(sharded_dcp.__file__), Path(hf_utils.__file__),
               Path(__file__).resolve().parents[1] / 'utils/constants.py']
    recipe = dict(version=5, source=str(src), model_id=args.model_id,
                  index_sha256=index_before, config_sha256=config_before, files=files,
                  code_sha256={p.name: sharded_dcp.sha(p) for p in modules},
                  torch_version=str(torch.__version__), fsdp=args.fsdp_shards, ep=args.ep_size,
                  plan_sha256=sharded_dcp.digest_json(jobs))
    fingerprint = sharded_dcp.digest_json(recipe)
    summary = dict(source=str(src), output=str(dst), model_id=args.model_id,
                   shard_count=args.fsdp_shards, ep_size=args.ep_size,
                   expert_fsdp_size=args.fsdp_shards // args.ep_size, model_tensors=len(specs),
                   jobs=len(jobs), total_logical_bytes=total,
                   tensor_chunks=sum(len(t['chunks']) for t in specs.values()), fingerprint=fingerprint)
    dst.mkdir(parents=True, exist_ok=True)
    with sharded_dcp.lock(dst):
        release, manifest = dst / 'release', dst / 'conversion_recipe.json'
        if manifest.exists():
            if hf_utils._read_json(manifest) != recipe:
                raise ValueError('Resume recipe/source identity mismatch')
        else:
            if any(p.name != '.conversion.lock' for p in dst.iterdir()):
                raise ValueError('Existing checkpoint without matching conversion recipe')
            sharded_dcp.write_json(manifest, recipe)
        release.mkdir(exist_ok=True)
        (release / '.receipts').mkdir(exist_ok=True)
        if not (release / '.metadata').exists():
            sharded_dcp.atomic(release / '.incomplete', b'Conversion in progress; do not load.\n')
        started, receipts, done_bytes = time.monotonic(), {}, 0

        def progress(phase, *, log=False):
            status = dict(summary, phase=phase, completed_jobs=len(receipts),
                          completed_logical_bytes=done_bytes, elapsed_s=time.monotonic()-started,
                          updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
            sharded_dcp.write_json(dst / 'progress.json', status)
            if log:
                print(json.dumps(status), flush=True)

        progress('resume_validation' if args.resume else 'convert', log=True)
        pending = []
        for job in jobs:
            if (release / '.receipts' / f'{job["id"]:06d}.json').exists():
                r = sharded_dcp.validate_receipt(release, job, fingerprint, file_prefix='__hf_')
                receipts[job['id']] = r
                done_bytes += job['bytes']
                progress('resume_validation')
            else:
                pending.append(job)
        if (release / '.metadata').exists() and pending:
            raise ValueError('Published metadata but incomplete receipts')
        if args.resume and pending:
            progress('convert', log=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_init_worker, initargs=(args.model_id, str(src))) as pool:
            futures = []
            try:
                for job in pending:
                    entries = [t['source'] for t in job['tensors']] if job['kind'] == 'ordinary' else job['sources'].values()
                    subset = {e['file']: files[e['file']] for e in entries}
                    futures.append(pool.submit(
                        convert_job, str(src), str(release), job, subset, fingerprint, args.read_threads))
                for f in as_completed(futures):
                    r = f.result()
                    receipts[r['id']] = r
                    done_bytes += r['logical_bytes']
                    progress('convert')
            except BaseException:
                for f in futures:
                    f.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                progress('failed', log=True)
                raise
        for file, expected in files.items():
            if hf_utils.identity(src / file) != expected:
                raise ValueError(f'Source changed before publish: {file}')
        index_sha = sharded_dcp.sha(index) if index.exists() else None
        if index_sha != recipe['index_sha256'] or sharded_dcp.sha(config) != recipe['config_sha256']:
            raise ValueError('Source index/config changed')
        ordered = [sharded_dcp.validate_receipt(release, job, fingerprint, verify_hash=False, file_prefix='__hf_')
                   for job in jobs]
        md = sharded_dcp.assemble(specs, ordered)
        _copy_assets(src, dst)
        report = sharded_dcp.publish(release, md, dict(
            summary, schema_version=1, verify='full',
            output_files=len(ordered), output_bytes=sum(r['file_size'] for r in ordered)))
        progress('complete', log=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--src-dir', type=Path, required=True)
    parser.add_argument('--dst-dir', type=Path, required=True)
    parser.add_argument('--fsdp-shards', type=int, default=128)
    parser.add_argument('--ep-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--read-threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
