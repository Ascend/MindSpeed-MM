#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : deepseekvl_hf_to_mm.py
@Time    : 2025/04/17
@Desc    : deepseekvl huggingface模型转换成mindspeed-mm模型

"""
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple

import torch
from tqdm import tqdm

from checkpoint.common.constant import LATEST_TXT
from checkpoint.common.types import STATE_DICT_T
from checkpoint.vlm_model.config import ConvertMMConfig
from checkpoint.vlm_model.hf_to_mm import split_by_tp, split_by_ep, merge_pp_index
from checkpoint.vlm_model.operator import RowSplit, GLUSplit, ColSplit, RowSplit

deepseekvl_tp_patterns = {
    r"text_decoder.output_layer.weight": RowSplit,
    r"text_decoder.embedding.word_embeddings.weight": RowSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight": GLUSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.linear_fc2.weight": ColSplit,
    r"text_decoder.decoder.layers.(\d+).self_attention.linear_qb.weight": RowSplit,
    r"text_decoder.decoder.layers.(\d+).self_attention.linear_kvb.weight": RowSplit,
    r"text_decoder.decoder.layers.(\d+).self_attention.linear_kvb.bias": RowSplit,
    r"text_decoder.decoder.layers.(\d+).self_attention.linear_proj.weight": ColSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1.weight": GLUSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc2.weight": ColSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc1.weight": GLUSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc2.weight": ColSplit
}


def load_from_hf(load_dir, trust_remote_code):
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(
        load_dir,
        device_map="cpu",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code
    )
    return hf_model.state_dict()


def merge_llm_state_dict(vl_state_dict, llm_state_dict):
    for key in list(vl_state_dict.keys()):
        if key.startswith("vision"):
            continue
        vl_state_dict.pop(key)

    for key in list(llm_state_dict.keys()):
        llm_state_dict[f"language.{key}"] = llm_state_dict.pop(key)

    merged_state_dict = {**vl_state_dict, **llm_state_dict}
    return merged_state_dict


def convert_hf_to_mm(_state_dict: Dict[str, torch.Tensor],
                     _num_layers: int,
                     _num_experts: int) -> Dict[str, torch.Tensor]:
    new_params = {}
    for key, value in tqdm(_state_dict.items(), desc="convert weights"):
        new_key = None
        # visual 权重转换部分
        if key.startswith("vision"):
            new_key = key.replace("vision", "image_encoder.encoder")
        # projector 部分
        elif key.startswith("projector"):
            new_key = key.replace("projector", "image_encoder.projector")

        elif key.startswith("language"):
            new_key = key.replace('language', 'text_decoder')
            new_key = new_key.replace('model.embed_tokens', 'embedding.word_embeddings')
            new_key = new_key.replace('model.layers', 'decoder.layers')

            new_key = new_key.replace('self_attn.q_proj', 'self_attention.q_proj')
            new_key = new_key.replace('self_attn.q_a_proj', 'self_attention.q_proj')
            new_key = new_key.replace('self_attn.q_b_proj', 'self_attention.linear_qb')
            new_key = new_key.replace('self_attn.kv_a_proj_with_mqa', 'self_attention.kv_a_proj_with_mqa')
            new_key = new_key.replace('self_attn.q_a_layernorm', 'self_attention.q_layernorm')
            new_key = new_key.replace('self_attn.kv_a_layernorm', 'self_attention.k_layernorm')
            new_key = new_key.replace('self_attn.kv_b_proj', 'self_attention.linear_kvb')
            new_key = new_key.replace('self_attn.o_proj', 'self_attention.linear_proj')

            new_key = new_key.replace('gate_proj', 'linear_fc1_gate')
            new_key = new_key.replace('up_proj', 'linear_fc1_up')
            new_key = new_key.replace('down_proj', 'linear_fc2')
            new_key = new_key.replace('post_attention_layernorm', 'pre_mlp_layernorm')

            new_key = new_key.replace('mlp.experts', 'mlp.experts.local_experts')
            new_key = new_key.replace('mlp.gate', 'mlp.router')

            new_key = new_key.replace('e_score_correction_bias', 'expert_bias')

            new_key = new_key.replace('model.norm', 'decoder.final_layernorm')
            new_key = new_key.replace('lm_head', 'output_layer')
        else:
            new_key = key

        if new_key is not None:
            print(f"mapping {key} to {new_key}")
            new_params[new_key] = value

    # 合并gate up 权重
    for i in range(_num_layers):
        # 合并self attn的mlp
        gate_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_gate.weight'
        up_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_up.weight'
        fc1_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1.weight'

        # 如果权重名字在新字典中，则获取对应权重值
        if gate_name in new_params.keys() and up_name in new_params.keys():
            gate_proj_weight, up_proj_weight = new_params[gate_name], new_params[up_name]
            new_params.pop(gate_name)
            new_params.pop(up_name)

            # 将 gate 和 up 沿着第0维度进行拼接
            linear_fc1 = torch.cat([gate_proj_weight, up_proj_weight], dim=0)
            new_params[fc1_name] = linear_fc1

        # 合并moe的shared experts的mlp
        gate_name = f'text_decoder.decoder.layers.{i}.mlp.shared_experts.linear_fc1_gate.weight'
        up_name = f'text_decoder.decoder.layers.{i}.mlp.shared_experts.linear_fc1_up.weight'
        fc1_name = f'text_decoder.decoder.layers.{i}.mlp.shared_experts.linear_fc1.weight'

        # 如果权重名字在新字典中，则获取对应权重值
        if gate_name in new_params.keys() and up_name in new_params.keys():
            gate_proj_weight, up_proj_weight = new_params[gate_name], new_params[up_name]
            new_params.pop(gate_name)
            new_params.pop(up_name)

            # 将 gate 和 up 沿着第0维度进行拼接
            linear_fc1 = torch.cat([gate_proj_weight, up_proj_weight], dim=0)
            new_params[fc1_name] = linear_fc1

            # 合并moe的experts的mlp
            for j in range(_num_experts):
                gate_name = f'text_decoder.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc1_gate.weight'
                up_name = f'text_decoder.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc1_up.weight'
                fc1_name = f'text_decoder.decoder.layers.{i}.mlp.experts.local_experts.{j}.linear_fc1.weight'

                gate_proj_weight, up_proj_weight = new_params[gate_name], new_params[up_name]
                new_params.pop(gate_name)
                new_params.pop(up_name)

                linear_fc1 = torch.cat([gate_proj_weight, up_proj_weight], dim=0)
                new_params[fc1_name] = linear_fc1

    # 合并 kv_a_proj_with_mqa + q_proj = linear_qkv
    for i in range(_num_layers):
        q_proj_name = f'text_decoder.decoder.layers.{i}.self_attention.q_proj.weight'
        kv_a_proj_name = f'text_decoder.decoder.layers.{i}.self_attention.kv_a_proj_with_mqa.weight'
        linear_qkv_name = f'text_decoder.decoder.layers.{i}.self_attention.linear_qkv.weight'

        if q_proj_name in new_params.keys() and kv_a_proj_name in new_params.keys():
            q_proj_weight = new_params[q_proj_name]
            kv_a_proj_weight = new_params[kv_a_proj_name]
            new_params.pop(q_proj_name)
            new_params.pop(kv_a_proj_name)

            # 将q_proj 和 kv_a_proj 沿着0维concat
            linear_qkv = torch.cat([q_proj_weight, kv_a_proj_weight], dim=0)
            new_params[linear_qkv_name] = linear_qkv

    return new_params


def save_by_rank(state_dicts: List[Dict[str, torch.Tensor]],
                 save_root_dir: Path,
                 iteration: Optional[Union[str, int]] = 'release',
                 ep_size: int = 1,
                 tp_rank: int = 0,
                 ep_rank: int = 1):
    for pp_rank, state_dict in enumerate(tqdm(state_dicts, desc="pp step")):
        # megatron格式权重的命名方式为 "mp_rank_{tp_rank}_{pp_rank}_{ep_rank}"
        name_parts = ["mp", "rank", f"{tp_rank:02d}"]
        if len(state_dicts) > 1:
            name_parts.append(f"{pp_rank:03d}")
        if ep_size > 1:
            name_parts.append(f"{ep_rank:03d}")
        iter_name = iteration if isinstance(iteration, str) else f"iter_{iteration:07d}"
        save_path = save_root_dir.joinpath(iter_name, "_".join(name_parts))
        save_path.mkdir(exist_ok=True, parents=True)
        torch.save({'model': state_dict}, save_path.joinpath('model_optim_rng.pt'))
    save_root_dir.joinpath(LATEST_TXT).write_text(str(iteration))


def split_model_by_pipeline(state_dict: STATE_DICT_T, pp_split: List[Tuple[int, int]]) -> List[STATE_DICT_T]:
    if len(pp_split) <= 1:
        return [state_dict]

    pp_size = len(pp_split)
    vit_range = [0, 0]
    llm_range = [pp_size - 1, pp_size - 1]
    for pp_rank, (vit_num, llm_num) in enumerate(pp_split):
        if vit_num > 0 and pp_rank > vit_range[1]:
            vit_range[1] = pp_rank
        if llm_num > 0 and pp_rank < llm_range[0]:
            llm_range[0] = pp_rank
    vit_start_idx = 0
    llm_start_idx = 0
    return_dicts = []
    copy_dict = deepcopy(state_dict)
    for pp_rank, (vit_num, llm_num) in enumerate(pp_split):
        vit_end_idx = vit_start_idx + vit_num
        llm_end_idx = llm_start_idx + llm_num
        new_dict = {}
        for key, value in state_dict.items():
            if key.startswith('image_encoder.encoder.patch_embed.'):
                if pp_rank == vit_range[0]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.encoder.pos_embed'):
                if pp_rank == vit_range[0]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.encoder.blocks.'):
                layer_idx = int(key.split('.')[3])
                if vit_start_idx <= layer_idx < vit_end_idx and vit_range[0] <= pp_rank <= vit_range[1]:
                    new_idx = layer_idx - vit_start_idx
                    new_key = key.replace(f'{layer_idx}', f'{new_idx}', 1)
                    new_dict[new_key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.encoder.norm'):
                if pp_rank == vit_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.encoder.attn_pool'):
                if pp_rank == vit_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_newline'):
                new_dict[key] = value
            elif key.startswith('view_seperator'):
                new_dict[key] = value
            elif key.startswith('image_encoder.projector.'):
                if pp_rank == vit_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.embedding.'):
                if pp_rank == llm_range[0]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.layers.'):
                layer_idx = int(key.split('.')[3])
                if llm_start_idx <= layer_idx < llm_end_idx and llm_range[0] <= pp_rank <= llm_range[1]:
                    new_idx = layer_idx - llm_start_idx
                    new_key = key.replace(f'{layer_idx}', f'{new_idx}', 1)
                    new_dict[new_key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.final_layernorm.'):
                if pp_rank == llm_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.output_layer.'):
                if pp_rank == llm_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
        vit_start_idx = vit_end_idx
        llm_start_idx = llm_end_idx
        return_dicts.append(new_dict)
    return return_dicts


def main(convert_config: ConvertMMConfig):
    from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
    config = convert_config.hf_config.config
    load_dir = convert_config.hf_config.hf_dir
    parallel_config = convert_config.parallel_config
    trust_remote_code = convert_config.trust_remote_code
    llm_hf_config = convert_config.llm_hf_config

    llm_num_layers = config.language_config.num_hidden_layers
    num_experts = config.language_config.n_routed_experts

    # 加载原始权重字典
    state_dict = load_from_hf(load_dir, trust_remote_code=trust_remote_code)

    # 如果有llm权重，则合并llm权重
    if llm_hf_config is not None:
        llm_state_dict = load_from_hf(llm_hf_config.hf_dir, trust_remote_code=trust_remote_code)
        state_dict = merge_llm_state_dict(state_dict, llm_state_dict)

        llm_num_layers = llm_hf_config.config.num_hidden_layers
        num_experts = llm_hf_config.config.n_routed_experts

    # hf权重映射到mm
    state_dict = convert_hf_to_mm(state_dict, llm_num_layers, num_experts)

    # 权重字典按ep域切分
    ep_state_dicts = split_by_ep(state_dict, parallel_config.ep_size, _num_experts=num_experts)

    # 权重字典按tp域切分
    ep_tp_state_dicts = []
    for ep_state_dict in ep_state_dicts:
        tp_state_dicts = split_by_tp(ep_state_dict, deepseekvl_tp_patterns, tp_size=parallel_config.tp_size)
        ep_tp_state_dicts.append(tp_state_dicts)

    # pp索引生成
    pp_split = merge_pp_index(parallel_config.vit_pp_layers, parallel_config.llm_pp_layers)

    for ep_rank, tp_state_dicts in enumerate(tqdm(ep_tp_state_dicts, desc="ep step")):
        for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
            pp_state_dicts = split_model_by_pipeline(tp_state_dict, pp_split)
            save_by_rank(pp_state_dicts, convert_config.mm_dir, ep_size=parallel_config.ep_size, ep_rank=ep_rank,
                         tp_rank=tp_rank)
