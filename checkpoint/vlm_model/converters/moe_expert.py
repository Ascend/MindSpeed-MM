"""
Models such as InternVL3.5/Qwen3Omni directly use the Qwen3MOE structure, implemented in the same way as Mixtral (where
 each expert has a separate weight). This approach has issues:
 1. With fsdp2, zero2/3, uneven expert activation during data parallel processing can cause gradient reduce_scatter to stall
 2. It is not easy to adapt to optimisations for fused operators such as moe group gemm
Therefore, this file introduces the ability to merge multiple expert weights and split them again after merged training,
supporting InternVL3.5/Qwen3OmniMoe

Following command-line interfaces are provided:
```bash
mm-convert moe_expert --style merge --hf_dir "OpenGVLab/InternVL3_5-30B-A3B-Instruct" --save_dir "merged_weight"
mm-convert moe_expert --style split --hf_dir "merged_weight" --save_dir "splited_weight"
"""

import json
from pathlib import Path
from typing import Dict, Literal

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from pydantic import DirectoryPath
from safetensors.torch import save_file
from tqdm import tqdm, trange
from transformers import AutoConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from checkpoint.common.converter import Commandable, DcpConverter
from checkpoint.common.merge_dcp_to_hf import load_dcp_state_dict
from checkpoint.common.mm_types import STATE_DICT_T
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.vlm_model.hf_to_mm import load_from_hf, save_by_dcp
from checkpoint.vlm_model.mm_to_hf import copy_files_except_suffix


def merge_moe_expert_weights(state_dict: STATE_DICT_T, num_hidden_layers: int, num_experts: int,
                             weight_path: str) -> None:
    """Process weights for each layer and expert. state_dict will be modified in place"""
    for layer in trange(num_hidden_layers, desc="merge moe experts weight"):
        gate_up_proj_weights = []
        down_proj_weights = []
        for expert in range(num_experts):
            gate_proj = (weight_path + ".gate_proj.weight").format(layer=layer, expert=expert)
            up_proj = (weight_path + ".up_proj.weight").format(layer=layer, expert=expert)
            down_proj = (weight_path + ".down_proj.weight").format(layer=layer, expert=expert)
            gate_proj_weight = state_dict.pop(gate_proj)  # intermediate_size*hidden_size
            up_proj_weight = state_dict.pop(up_proj)  # intermediate_size*hidden_size
            down_proj_weight = state_dict.pop(down_proj)  # intermediate_size*hidden_size
            gate_up_proj_weight = torch.concat([gate_proj_weight, up_proj_weight])
            # gate_proj/up_proj/down_proj (which use nn.Linear) need transpose to align with gate_up_proj
            # (which use nn.Parameter)
            gate_up_proj_weights.append(gate_up_proj_weight.T)
            down_proj_weights.append(down_proj_weight.T)
        new_gate_up_proj_weight = torch.stack(gate_up_proj_weights)
        new_down_proj_weight = torch.stack(down_proj_weights)
        new_gate_up_proj = (weight_path.replace('.{expert}', '') + ".gate_up_proj").format(layer=layer)
        new_down_proj = (weight_path.replace('.{expert}', '') + ".down_proj").format(layer=layer)
        state_dict[new_gate_up_proj] = new_gate_up_proj_weight
        state_dict[new_down_proj] = new_down_proj_weight


def split_moe_expert_weights(state_dict: STATE_DICT_T, num_hidden_layers: int, num_experts: int,
                             weight_path: str) -> None:
    """Split merged expert weights back into individual expert weights. state_dict will be modified in place."""
    for layer in trange(num_hidden_layers, desc="split moe experts weight"):
        # Get merged gate_up_proj and down_proj weights
        gate_up_proj = (weight_path.replace('.{expert}', '') + ".gate_up_proj").format(layer=layer)
        down_proj = (weight_path.replace('.{expert}', '') + ".down_proj").format(layer=layer)

        if gate_up_proj not in state_dict or down_proj not in state_dict:
            raise ValueError(f"No {gate_up_proj} or {down_proj} in state_dict!")

        merged_gate_up_proj = state_dict.pop(gate_up_proj)
        merged_down_proj = state_dict.pop(down_proj)

        # Split merged_gate_up_proj into individual gate_proj and up_proj for each expert
        gate_up_experts = merged_gate_up_proj.unbind()
        down_experts = merged_down_proj.unbind()
        for expert, (gate_up_weight, down_weight) in enumerate(zip(gate_up_experts, down_experts)):
            # Split gate_proj and up_proj for the current expert
            gate_proj_weight, up_proj_weight = torch.chunk(gate_up_weight.T, 2)

            # Add individual expert weights to state_dict
            gate_proj = (weight_path + ".gate_proj.weight").format(layer=layer, expert=expert)
            up_proj = (weight_path + ".up_proj.weight").format(layer=layer, expert=expert)
            down_proj = (weight_path + ".down_proj.weight").format(layer=layer, expert=expert)

            state_dict[gate_proj] = gate_proj_weight
            state_dict[up_proj] = up_proj_weight
            state_dict[down_proj] = down_weight.T


def save_sharded_state_dict(state_dict: STATE_DICT_T, save_dir: Path, max_shard_size: str, metadata: Dict) -> None:
    """Save the sharded state_dict to files."""
    filename_pattern = SAFE_WEIGHTS_NAME.replace(".safetensors", "{suffix}.safetensors")
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern=filename_pattern,
        max_shard_size=max_shard_size
    )
    index = {
        "metadata": state_dict_split.metadata,
        "weight_map": state_dict_split.tensor_to_filename,
    }

    index_file = save_dir / SAFE_WEIGHTS_INDEX_NAME
    index_file.write_text(json.dumps(index, indent=2, sort_keys=True))
    for shard_file, tensors in tqdm(state_dict_split.filename_to_tensors.items(), desc="save sharded safetensors"):
        shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
        save_file(shard, save_dir / shard_file, metadata=metadata)


def get_expert_config_from_pretrained(save_dir: Path) -> tuple[int, int, str]:
    config = AutoConfig.from_pretrained(save_dir, trust_remote_code=True)
    if config.__class__.__name__ == "InternVLChatConfig":
        num_hidden_layers = config.llm_config.num_hidden_layers
        num_experts = config.llm_config.num_experts
        weight_path = "language_model.model.layers.{layer}.mlp.experts.{expert}"
    elif config.__class__.__name__ == "Qwen3OmniMoeConfig":
        num_hidden_layers = config.thinker_config.text_config.num_hidden_layers
        num_experts = config.thinker_config.text_config.num_experts
        weight_path = "thinker.model.layers.{layer}.mlp.experts.{expert}"
    else:
        # new model such as qwen3omni moe/qwen3 moe only need define num_hidden_layers/num_experts/weight_path here.
        raise ValueError(f"Not supported model config: {config}")
    return num_experts, num_hidden_layers, weight_path


def moe_expert(style: Literal["merge", "split"], hf_dir: DirectoryPath, save_dir: Path, max_shard_size: str = "4GB"):
    """merge Mixtral style moe (one parameter per expert) all experts parameter to one parameter, or vice versa
    usage:
    mm-convert moe_expert --style merge --hf_dir "OpenGVLab/InternVL3_5-30B-A3B-Instruct" --save_dir "merged_weight"
    mm-convert moe_expert --style split --hf_dir "merge" --save_dir "splited_weight"

    Args:
        style (Literal["merge", "split"]): merge or split experts weight
        hf_dir (DirectoryPath): pretrained weight directory from huggingface
        save_dir (Path): merged or splited experts weight saved directory
        max_shard_size (str, optional): shard file by max size. Defaults to "4GB".
    """
    copy_files_except_suffix(hf_dir, save_dir, except_suffix='.safetensors')

    num_experts, num_hidden_layers, weight_path = get_expert_config_from_pretrained(save_dir)
    state_dict = load_from_hf(hf_dir)
    if style == "merge":
        merge_moe_expert_weights(state_dict, num_hidden_layers, num_experts, weight_path=weight_path)
    elif style == "split":
        split_moe_expert_weights(state_dict, num_hidden_layers, num_experts, weight_path=weight_path)
    else:
        raise ValueError(f"Only support style `merge` or `split`, given: {style}")
    save_sharded_state_dict(state_dict, save_dir, max_shard_size, metadata={"format": "pt"})


class ExpertMergeDcpConverter(DcpConverter):
    @staticmethod
    def hf_to_dcp(hf_dir: DirectoryPath, save_dir: Path):
        num_experts, num_hidden_layers, weight_path = get_expert_config_from_pretrained(hf_dir)
        state_dict = load_from_hf(hf_dir)
        merge_moe_expert_weights(state_dict, num_hidden_layers, num_experts, weight_path=weight_path)
        weight_names = list(state_dict.keys())
        for weight_name in weight_names:
            state_dict[f"model.{weight_name}"] = state_dict.pop(weight_name)
        save_by_dcp(state_dict, save_dir)
        set_directory_permissions(save_dir)

    @staticmethod
    def dcp_to_hf(hf_dir: DirectoryPath, dcp_dir: DirectoryPath, save_dir: Path):
        copy_files_except_suffix(hf_dir, save_dir, except_suffix='.safetensors')
        state_dict = load_dcp_state_dict(dcp_dir)
        weight_names = list(state_dict.keys())
        for weight_name in weight_names:
            state_dict[weight_name.removeprefix("model.")] = state_dict.pop(weight_name)
        num_experts, num_hidden_layers, weight_path = get_expert_config_from_pretrained(save_dir)
        split_moe_expert_weights(state_dict, num_hidden_layers, num_experts, weight_path=weight_path)
        save_sharded_state_dict(state_dict, save_dir, max_shard_size="4GB", metadata={"format": "pt"})
        set_directory_permissions(save_dir)


# register to command line interface
Commandable.add_command(moe_expert)
