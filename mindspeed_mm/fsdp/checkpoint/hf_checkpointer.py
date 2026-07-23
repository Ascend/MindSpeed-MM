import gc
import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import FilePath
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoProcessor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.fsdp.checkpoint.broadcast_utils import rank0_load_and_broadcast_hf_weights
from mindspeed_mm.fsdp.checkpoint.checkpointer import CheckpointerBase
from mindspeed_mm.fsdp.checkpoint.convert import WEIGHT_TRANSFORM_PIPELINES
from mindspeed_mm.fsdp.checkpoint.utils import get_checkpoint_name
from mindspeed_mm.fsdp.checkpoint.hf_utils import (
    load_hf_weights,
    get_model_save_state,
    get_dtype_size,
    find_safetensors_index,
    save_state_dict,
)
from mindspeed_mm.fsdp.utils.constants import FILE_MODE
from mindspeed_mm.fsdp.utils.device import empty_cache, get_device_type, synchronize

logger = logging.getLogger(__name__)

# Controls whether non-zero ranks sleep or compute while rank 0 saves HuggingFace checkpoint shards.
HF_SAVE_WAIT_MODE = os.getenv("HF_SAVE_WAIT_MODE", "sleep")


class HuggingFaceCheckpointer(CheckpointerBase):
    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        iteration: int = None,
        save_ckpt_dtype: Optional[torch.dtype] = None,
        model_assets_dir: str = None,
        model_id=None,
        **kwargs,
    ) -> None:
        """
        Save model weights to a huggingface checkpoint.
        args:
            path: path to save checkpoint
            state: state to save, "model" is required
            iteration: global steps, used to create the iter_xxxxxxx checkpoint directory
            save_ckpt_dtype: optional dtype to cast floating-point model weights before saving
            model_assets_dir: original huggingface checkpoint directory used to copy config, processor and shard mapping
            model_id: model id used to select model-specific weight transform pipeline
        return:
            None
        """
        if iteration is None:
            raise ValueError("iteration must be provided when saving a huggingface checkpoint.")
        if model_assets_dir is None or not os.path.exists(model_assets_dir):
            raise ValueError("model_assets_dir cannot be none and must be an existing directory")

        checkpoint_dir = get_checkpoint_name(path, iteration, release=False)
        safetensor_idx_path: Optional[FilePath] = find_safetensors_index(Path(model_assets_dir))
        fqn_to_filename_mapping = None
        if safetensor_idx_path is not None:
            with open(safetensor_idx_path, "r", encoding="utf-8") as f:
                fqn_to_filename_mapping = json.load(f)["weight_map"]
        save_state = get_model_save_state(state["model"], fqn_to_filename_mapping, save_ckpt_dtype)

        # Transform state dict from DCP to HF format
        transform_cls = WEIGHT_TRANSFORM_PIPELINES.get(model_id, None)
        weight_transform = transform_cls() if transform_cls is not None else None
        if weight_transform is not None:
            new_state = {}
            for key, tensor in save_state.items():
                key, tensor = weight_transform.dcp_to_hf(key, tensor)
                new_state[key] = tensor
            save_state = new_state

        if fqn_to_filename_mapping is not None:
            original_mapping_size = len(fqn_to_filename_mapping)
            fqn_to_filename_mapping = {k: v for k, v in fqn_to_filename_mapping.items() if k in save_state}
            if len(fqn_to_filename_mapping) < original_mapping_size:
                logger.info_rank0(
                    f"Filtered fqn_to_index_mapping from {original_mapping_size} to {len(fqn_to_filename_mapping)} keys "
                    f"to match actual model weights."
                )

        is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0
        weights_name = SAFE_WEIGHTS_NAME
        weight_map = {}
        shard_files = OrderedDict()
        if fqn_to_filename_mapping:
            for name in save_state.keys():
                fname = fqn_to_filename_mapping[name]
                weight_map[name] = fname
                shard_files.setdefault(fname, []).append(name)
        else:
            logger.warning_rank0(
                "fqn_to_filename_mapping is None, saved safetensor will be a single file instead of sharded."
            )
            for name in save_state.keys():
                weight_map[name] = weights_name
            shard_files[weights_name] = list(save_state.keys())
        num_shards = len(shard_files)
        sync_markers = [Path(checkpoint_dir) / f".hf-save-{i:05d}.done" for i in range(num_shards)]

        if is_rank_0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            if dist.is_initialized():
                for marker in sync_markers:
                    marker.unlink(missing_ok=True)

        logger.info_rank0("Starting HuggingFace safetensors save from model...")
        if dist.is_initialized():
            dist.barrier()

        total_size = 0
        start_time = time.time()
        for shard_index, (fname, names) in enumerate(shard_files.items()):
            full_state_dict = OrderedDict()
            for name in names:
                tensor = save_state[name]
                if hasattr(tensor.data, "full_tensor"):  # FSDP2 dtensor
                    tensor = tensor.data.full_tensor()
                else:
                    tensor = tensor.data
                if is_rank_0:
                    full_state_dict[name] = tensor.detach().cpu()
                    total_size += tensor.numel() * get_dtype_size(tensor.dtype)
                del tensor

            # Avoid launching an HCCL collective while rank 0 writes the shard.
            if is_rank_0:
                save_state_dict(full_state_dict, os.path.join(checkpoint_dir, fname))
                del full_state_dict
                gc.collect()

            empty_cache()
            if dist.is_initialized():
                synchronize()
                if is_rank_0:
                    sync_markers[shard_index].touch(mode=FILE_MODE)
                else:
                    # Keep non-zero ranks computationally active to prevent termination by idle-worker enforcement.
                    if HF_SAVE_WAIT_MODE == "compute":
                        dummy_tensor = torch.ones((1024, 1024), dtype=torch.float16, device=get_device_type())
                        while not sync_markers[shard_index].is_file():
                            torch.matmul(dummy_tensor, dummy_tensor)
                            synchronize()
                        del dummy_tensor
                        empty_cache()
                    else:
                        while not sync_markers[shard_index].is_file():
                            time.sleep(1)

        elapsed_time = time.time() - start_time
        logger.info_rank0(f"HuggingFace safetensors save from live model took {elapsed_time:.2f}s")
        # Write index.json and model assets on rank 0
        if is_rank_0:
            if num_shards > 1:
                index = {
                    "metadata": {"total_size": total_size},
                    "weight_map": weight_map,
                }
                index_file = SAFE_WEIGHTS_INDEX_NAME
                with open(os.path.join(checkpoint_dir, index_file), "w", encoding="utf-8") as f:
                    f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")
                logger.info(f"Model weight splits saved in {checkpoint_dir}.")
            else:
                logger.info(f"Model weights saved at {os.path.join(checkpoint_dir, weights_name)}.")

            config = AutoConfig.from_pretrained(model_assets_dir, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_assets_dir, trust_remote_code=True)
            config.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)

            logger.info(f"HuggingFace checkpoint saved at {checkpoint_dir} successfully!")

        if dist.is_initialized():
            dist.barrier()
            if is_rank_0:
                for marker in sync_markers:
                    marker.unlink(missing_ok=True)

        return

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        load_rank0_and_broadcast: bool = False,
        load_strict: bool = False,
        enable_lora: bool = False,
        model_id=None,
        **kwargs,
    ) -> bool:
        """
        Load training state from huggingface checkpoint
        args:
            path: path to load checkpoint
            state: state to load, "model" are required,  "optimizer" and "extra_state" are optional
            load_rank0_and_broadcast: rank 0 loads the checkpoint, then broadcasts to other ranks
            load_strict: strictly enforce that the checkpoint keys match the model's state_dict
            enable_lora: whether to enable LoRA checkpoint loading logic
            model_id: model id used to select model-specific weight transform pipeline
        return:
            release (bool): whether the loaded checkpoint is a "release" checkpoint
        """
        if state is None or "model" not in state:
            raise ValueError("State dict and model must be provided to load a huggingface checkpoint.")

        cfg = {}
        cfg_path = os.path.join(path, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        tcfg = cfg.get("text_config", cfg)
        method = "rank0-broadcast" if load_rank0_and_broadcast else "every-rank-read"
        print_rank(
            logger.info,
            f"Loading HF safetensors -> FSDP DTensors via {method} (online): "
            f"dir={path} "
            f"arch={(cfg.get('architectures') or ['?'])[0]} "
            f"layers={tcfg.get('num_hidden_layers', '?')} "
            f"tie_emb={cfg.get('tie_word_embeddings', tcfg.get('tie_word_embeddings', '?'))}",
        )

        transform_cls = WEIGHT_TRANSFORM_PIPELINES.get(model_id, None)
        weight_transform = transform_cls() if transform_cls is not None else None

        original_num_threads = torch.get_num_threads()
        local_world_size = max(1, int(os.getenv("LOCAL_WORLD_SIZE", "1")))
        load_num_threads = max(1, (os.cpu_count() or 1) // local_world_size)

        try:
            # Increase CPU parallelism temporarily for HF weight loading.
            torch.set_num_threads(load_num_threads)
            if original_num_threads != load_num_threads:
                print_rank(
                    logger.info,
                    f"Changed PyTorch CPU thread count from {original_num_threads} to "
                    f"{load_num_threads} for HF weight loading.",
                )
            if load_rank0_and_broadcast:
                rank0_load_and_broadcast_hf_weights(
                    model=state["model"],
                    hf_dir=path,
                    enable_lora=enable_lora,
                    load_strict=load_strict,
                    weight_transform=weight_transform,
                )
            else:
                load_hf_weights(
                    model=state["model"],
                    hf_dir=path,
                    enable_lora=enable_lora,
                    load_strict=load_strict,
                    weight_transform=weight_transform,
                )
        finally:
            # Restore the thread count for subsequent training.
            torch.set_num_threads(original_num_threads)
            if original_num_threads != load_num_threads:
                print_rank(
                    logger.info,
                    f"Restored PyTorch CPU thread count from {load_num_threads} to "
                    f"{original_num_threads} after HF weight loading.",
                )

        return True
