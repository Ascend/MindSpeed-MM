import argparse
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from torch.distributed.checkpoint import FileSystemWriter
from safetensors.torch import load_file

from checkpoint.common.constant import LATEST_TXT, DCP_CHECKPOINT_VERSION
from checkpoint.common.dcp_utils import partial_save_dcp_state_dict, merge_meta_info, save_metadata
from checkpoint.vlm_model.hf_to_mm import load_from_hf, save_by_dcp
from checkpoint.common.permissions import set_directory_permissions


def hf_to_dcp(
    hf_dir: str,
    dcp_dir: str,
    prefix: Optional[str]
):
    state_dict = load_from_hf(Path(hf_dir))
    state_dict = {f"{prefix}{k}": v for k, v in state_dict.items()}
    save_by_dcp(state_dict, Path(dcp_dir))


def _process_single_file(idx, safe_path, save_path, state_dict_convert_func, add_checkpoint_version):
    """
    Process a single safetensors file: load, convert, and save as DCP shard.
    This function is designed to be called in a worker thread.

    Args:
        idx: File index, used as part_idx for DCP storage prefix.
        safe_path: Path to the safetensors file to load.
        save_path: Directory path to save the DCP shard.
        state_dict_convert_func: Optional function to convert the loaded state dict.
        add_checkpoint_version: If True, add checkpoint_version to the save dict.
    """
    save_path = Path(save_path)

    # Each worker creates its own storage_writer to avoid thread-safety issues
    storage_writer = FileSystemWriter(save_path)

    # Load and convert
    state_dict = load_file(str(safe_path), device="cpu")
    if state_dict_convert_func:
        state_dict = state_dict_convert_func(state_dict)

    save_dict = {"model": state_dict}
    if add_checkpoint_version:
        save_dict["checkpoint_version"] = DCP_CHECKPOINT_VERSION

    # Save as DCP shard
    global_meta, all_write = partial_save_dcp_state_dict(save_dict, storage_writer, part_idx=idx)

    return idx, global_meta, all_write


def hf_to_dcp_sharded(
    hf_dir: str,
    dcp_dir: str,
    state_dict_convert_func: Optional[callable],
    num_workers: int = 0,
):
    """
    By default, DCP shards are split following the same sharding logic as the original Hugging Face (HF) checkpoint weights.

    Args:
        hf_dir: Path to HF format checkpoint directory
        dcp_dir: Path to save DCP format model
        state_dict_convert_func: Optional function to convert state dict
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
        tasks.append((i, str(safe_path), str(save_path), state_dict_convert_func, add_checkpoint_version))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=str, required=True, help="Path to HF format checkpoint directory")
    parser.add_argument("--dcp-dir", type=str, required=True, help="Path to save torch_dcp format model")
    parser.add_argument("--prefix", type=str, default="", help="Key prefix for state dict (e.g., 'model.')")
    parser.add_argument("--sharded", action="store_true", help="Enable sharded conversion to reduce memory usage (process one shard at a time)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of parallel workers for sharded conversion. Default is 0 (serial execution).")

    args = parser.parse_args()
    if args.sharded:
        hf_to_dcp_sharded(
            args.hf_dir,
            args.dcp_dir,
            state_dict_convert_func=lambda sd: {f"{args.prefix}{k}": v for k, v in sd.items()},
            num_workers=args.num_workers
        )
    else:
        hf_to_dcp(
            args.hf_dir,
            args.dcp_dir,
            prefix=args.prefix
        )
