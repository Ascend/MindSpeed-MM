import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from pydantic import validate_arguments, DirectoryPath, FilePath
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner

from transformers import AutoConfig, AutoProcessor
from safetensors.torch import save_file

from checkpoint.common.permissions import set_directory_permissions


@validate_arguments
def load_dcp_state_dict(dcp_checkpoint_dir: DirectoryPath) -> STATE_DICT_TYPE:
    sd: STATE_DICT_TYPE = {}
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(str(dcp_checkpoint_dir)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return sd['model'] if 'model' in sd else sd


def find_safetensors_index(directory: Path) -> Optional[FilePath]:
    """Find the .safetensors.index.json file in the given directory."""
    if not directory.is_dir():
        return None
    for file in directory.iterdir():
        if file.is_file() and file.name.endswith(".safetensors.index.json"):
            return file
    return None


@validate_arguments
def save_hf_weights(
    save_path: Path,
    model_assets_dir: DirectoryPath,
    state_dict: Dict,
    prefix: str = "",
):
    save_path.mkdir(parents=True, exist_ok=True)

    index_file: Optional[FilePath] = find_safetensors_index(Path(model_assets_dir))
    if index_file is None:
        raise FileNotFoundError(f"Could not find safetensors index file in directory {model_assets_dir}")

    # Copy index file
    shutil.copy2(index_file, save_path)

    with open(index_file, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    state_dicts = []
    for key, value in weight_map.items():
        index = int(value.split("-")[1])
        while index > len(state_dicts):
            state_dicts.append({})
        full_key = f"{prefix}{key}"
        if full_key in state_dict:
            state_dicts[index - 1][key] = state_dict[full_key]
        else:
            print(f"Missing key: '{full_key}' in state_dict")

    metadata = {"format": "pt"}
    for idx, sd in enumerate(state_dicts, start=1):
        name = f"model-{idx:05d}-of-{len(state_dicts):05d}.safetensors"
        save_file(sd, save_path / name, metadata=metadata)

    set_directory_permissions(save_path)


@validate_arguments
def merge_dcp_to_hf(
    load_dir: DirectoryPath,
    save_dir: str | Path,
    model_assets_dir: DirectoryPath,
    prefix: str = "",
):
    """
    Load model in torch DCP format and save in Hugging Face format.
    """
    state_dict = load_dcp_state_dict(load_dir)

    config = AutoConfig.from_pretrained(str(model_assets_dir))
    processor = AutoProcessor.from_pretrained(str(model_assets_dir), trust_remote_code=True)
    
    save_path = Path(save_dir)
    config.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    save_hf_weights(
        save_path=save_path,
        model_assets_dir=str(model_assets_dir),
        state_dict=state_dict,
        prefix=prefix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dir", type=str, required=True, help="Path to DCP checkpoint directory")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save HF format model")
    parser.add_argument("--model-assets-dir", type=str, required=True, help="Path to model assets (config, tokenizer, etc.)")
    parser.add_argument("--prefix", type=str, default="", help="Key prefix for state dict (e.g., 'model.')")

    args = parser.parse_args()

    print(f"Merge Args: {args}")
    merge_dcp_to_hf(
        load_dir=args.load_dir,
        save_dir=args.save_dir,
        model_assets_dir=args.model_assets_dir,
        prefix=args.prefix,
    )
    print(f"Merge to HF format success! Saved to: {args.save_dir}")