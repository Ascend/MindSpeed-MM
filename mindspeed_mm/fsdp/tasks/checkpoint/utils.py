import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from pydantic import FilePath
from safetensors.torch import load_file, save_file
import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

from mindspeed_mm.fsdp.checkpoint.dcp_utils import (
    extract_metadata,
    load_metadata,
    merge_meta_info,
    partial_load_dcp_state_dict,
    partial_save_dcp_state_dict,
    save_metadata,
)
from mindspeed_mm.fsdp.checkpoint.hf_utils import find_safetensors_index
from mindspeed_mm.fsdp.utils.constants import DCP_CHECKPOINT_VERSION, DIR_MODE, FILE_MODE, LATEST_TXT


def set_directory_permissions(target_dir: str | Path) -> None:
    target_dir = Path(target_dir)
    # Set root directory permissions
    target_dir.chmod(DIR_MODE)

    # Traverse the directory and set appropriate permissions for all files and subdirectories
    try:
        for root, dirs, files in os.walk(target_dir):
            # Set subdirectory permissions using pathlib operations
            root_path = Path(root)
            for directory in dirs:
                (root_path / directory).chmod(DIR_MODE)

            # Set file permissions using pathlib operations
            for file in files:
                (root_path / file).chmod(FILE_MODE)
    except OSError as e:
        raise OSError(f"Error occurred while setting permissions: {target_dir}") from e


def _process_single_file(idx, safe_path, save_path, weight_transform, add_checkpoint_version):
    """
    Process a single safetensors file: load, convert, and save as DCP shard.
    This function is designed to be called in a worker thread.

    Args:
        idx: File index, used as part_idx for DCP storage prefix.
        safe_path: Path to the safetensors file to load.
        save_path: Directory path to save the DCP shard.
        weight_transform: Weight transform pipeline selected by model ID.
        add_checkpoint_version: If True, add checkpoint_version to the save dict.
    """
    save_path = Path(save_path)

    # Each worker creates its own storage_writer to avoid thread-safety issues
    storage_writer = FileSystemWriter(save_path)

    # Load and convert
    state_dict = load_file(str(safe_path), device="cpu")
    converted_state_dict = {}
    for key, tensor in state_dict.items():
        key, tensor = weight_transform.hf_to_dcp(key, tensor)
        converted_state_dict[key] = tensor

    save_dict = {"model": converted_state_dict}
    if add_checkpoint_version:
        save_dict["checkpoint_version"] = DCP_CHECKPOINT_VERSION

    # Save as DCP shard
    global_meta, all_write = partial_save_dcp_state_dict(save_dict, storage_writer, part_idx=idx)

    return idx, global_meta, all_write


def hf_to_dcp_sharded(
    hf_dir: str,
    dcp_dir: str,
    weight_transform,
    num_workers: int = 0,
):
    """
    By default, DCP shards are split following the same sharding logic as the original Hugging Face (HF) checkpoint weights.

    Args:
        hf_dir: Path to HF format checkpoint directory
        dcp_dir: Path to save DCP format model
        weight_transform: Weight transform pipeline selected by model ID.
        num_workers: Number of parallel workers. Default is 0 (serial execution).
    """
    iter_name = "release"
    save_root_dir = Path(dcp_dir)
    save_path = save_root_dir.joinpath(iter_name)
    save_path.mkdir(exist_ok=True, parents=True)
    save_root_dir.joinpath(LATEST_TXT).write_text("release")

    storage_writer = FileSystemWriter(save_path)
    files = sorted(list(Path(hf_dir).glob("*.safetensors")))

    # Prepare task arguments
    tasks = []
    for i, safe_path in enumerate(files):
        add_checkpoint_version = (i == 0)
        tasks.append((i, str(safe_path), str(save_path), weight_transform, add_checkpoint_version))

    if num_workers >= 1 and len(files) > 1:
        # Parallel execution using threads (threads share memory, so closures work)
        meta_infos = [None] * len(files)
        all_writes = [None] * len(files)
        failed_tasks = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_file, *task) for task in tasks}

            for future in tqdm(as_completed(futures), total=len(files), desc="Processing files (parallel)"):
                try:
                    idx, global_meta, all_write = future.result()
                    meta_infos[idx] = global_meta
                    all_writes[idx] = all_write
                except Exception as e:
                    failed_tasks.append((future, e))
                    print(f"Warning: Task failed with error: {e}")

        if failed_tasks:
            print(f"Error: {len(failed_tasks)} task(s) failed. Check the warnings above for details.")
            raise RuntimeError(f"{len(failed_tasks)} task(s) failed during parallel processing")
    else:
        # Serial execution: reuse _process_single_file in main thread
        meta_infos = []
        all_writes = []
        for task in tqdm(tasks, desc="Processing files"):
            idx, global_meta, all_write = _process_single_file(*task)
            meta_infos.append(global_meta)
            all_writes.append(all_write)

    merged_meta = merge_meta_info(meta_infos)
    save_metadata(merged_meta, all_writes, storage_writer)
    set_directory_permissions(Path(dcp_dir))


def get_single_safetensors_filename(directory: Path) -> str:
    """Return the single HF safetensors filename when no index file exists."""
    if directory.is_dir():
        safetensor_files = sorted(
            file.name
            for file in directory.iterdir()
            if file.is_file()
            and file.name.endswith(".safetensors")
            and not file.name.endswith(".safetensors.index.json")
        )
        if len(safetensor_files) == 1:
            return safetensor_files[0]
    return "model.safetensors"


def _process_single_dcp_shard(
    idx,
    safetensor_file,
    selected_keys,
    load_dir,
    save_dir,
    metadata,
    weight_transform,
    to_bf16,
    hf_metadata,
):
    """
    Process a single HF safetensors shard: partial-load DCP weights, convert, and save.
    This function is designed to be called in a worker thread.

    Args:
        idx: File index, used to restore task ordering in logs/errors.
        safetensor_file: Target safetensors filename.
        selected_keys: DCP keys needed by this target safetensors shard.
        load_dir: Directory path to load the DCP checkpoint.
        save_dir: Directory path to save the HF shard.
        metadata: Full DCP metadata. It is read-only in this worker.
        weight_transform: Weight transform pipeline selected by model ID.
        to_bf16: Whether to convert weights to BF16.
        hf_metadata: Metadata written into the target safetensors file.
    """
    load_dir = Path(load_dir)
    save_dir = Path(save_dir)

    # Each worker creates its own storage_reader to avoid thread-safety issues.
    storage_reader = FileSystemReader(str(load_dir))

    partial_metadata = extract_metadata(selected_keys, metadata)
    partial_state_dict = partial_load_dcp_state_dict(partial_metadata, storage_reader)
    partial_state_dict = partial_state_dict["model"] if "model" in partial_state_dict else partial_state_dict

    converted_state_dict = {}
    for key, tensor in partial_state_dict.items():
        key, tensor = weight_transform.dcp_to_hf(key, tensor)
        if to_bf16:
            tensor = tensor.to(dtype=torch.bfloat16)
        converted_state_dict[key] = tensor

    save_file(converted_state_dict, save_dir / safetensor_file, metadata=hf_metadata)

    return idx, safetensor_file


def merge_dcp_to_hf_sharded(
    load_dir: str | Path,
    save_dir: str | Path,
    model_assets_dir: str | Path,
    select_key_convert_func: Optional[callable],
    weight_transform,
    trust_remote_code: bool = True,
    to_bf16: bool = False,
    num_workers: int = 0,
):
    """
    Load DCP weights in shards and save them as sharded checkpoints in Hugging Face (HF) format.

    Args:
        load_dir: Path to DCP checkpoint directory.
        save_dir: Path to save HF format model.
        model_assets_dir: Path to model assets (config, tokenizer, index, etc.).
        select_key_convert_func: Optional function to map HF keys to DCP keys for selection.
        weight_transform: Weight transform pipeline selected by model ID.
        trust_remote_code: Whether to trust remote code when loading HF assets.
        to_bf16: Whether to convert weights to BF16.
        num_workers: Number of parallel workers. Default is 0 (serial execution).
    """
    load_dir = Path(load_dir)
    save_dir = Path(save_dir)
    model_assets_dir = Path(model_assets_dir)

    config = AutoConfig.from_pretrained(model_assets_dir, trust_remote_code=trust_remote_code)
    processor = AutoProcessor.from_pretrained(model_assets_dir, trust_remote_code=trust_remote_code)
    config.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    storage_reader = FileSystemReader(str(load_dir))
    metadata = load_metadata(storage_reader)
    hf_metadata = {"format": "pt"}

    index_file: Optional[FilePath] = find_safetensors_index(Path(model_assets_dir))
    if index_file is not None:
        shutil.copy2(index_file, save_dir)
        with open(index_file, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        file_to_selected_keys = {
            safetensor_file: [
                select_key_convert_func(k) if select_key_convert_func else k
                for k, v in weight_map.items()
                if v == safetensor_file
            ]
            for safetensor_file in set(weight_map.values())
        }
    else:
        safetensor_file = get_single_safetensors_filename(Path(model_assets_dir))
        file_to_selected_keys = {safetensor_file: list(metadata.state_dict_metadata.keys())}

    # Prepare task arguments
    tasks = []
    for i, (safetensor_file, selected_keys) in enumerate(file_to_selected_keys.items()):
        tasks.append((
            i,
            safetensor_file,
            selected_keys,
            load_dir,
            save_dir,
            metadata,
            weight_transform,
            to_bf16,
            hf_metadata,
        ))

    if num_workers >= 1 and len(tasks) > 1:
        # Parallel execution using threads (threads share memory, so closures work)
        failed_tasks = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_dcp_shard, *task) for task in tasks}

            for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing files (parallel)"):
                try:
                    future.result()
                except Exception as e:
                    failed_tasks.append((future, e))
                    print(f"Warning: Task failed with error: {e}")

        if failed_tasks:
            print(f"Error: {len(failed_tasks)} task(s) failed. Check the warnings above for details.")
            raise RuntimeError(f"{len(failed_tasks)} task(s) failed during parallel processing")
    else:
        # Serial execution: reuse _process_single_dcp_shard in main thread
        for task in tqdm(tasks, desc="Processing files"):
            _process_single_dcp_shard(*task)

    set_directory_permissions(save_dir)
