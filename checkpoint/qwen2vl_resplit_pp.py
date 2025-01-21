#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright:   Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
@File    : qwen2vl_resplit_pp.py
@Time    : 2025/01/14
@Desc    : mindspeed-mm训练出的模型重新切分成新的pp配置
"""
from tqdm import tqdm

from checkpoint.qwen2vl_hf_to_mm import split_model_by_pipeline, save_by_pp, merge_pp_index
from checkpoint.qwen2vl_mm_to_hf import load_from_mm
from checkpoint.utils import ConvertPPConfig


def main(cfg: ConvertPPConfig):
    source = cfg.source_parallel_config
    target = cfg.target_parallel_config
    tp_state_dicts = load_from_mm(cfg.source_dir, source.vit_pp_layers, source.llm_pp_layers, source.tp_size)
    pp_split = merge_pp_index(target.vit_pp_layers, target.llm_pp_layers)

    for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
        pp_state_dicts = split_model_by_pipeline(tp_state_dict, pp_split)
        save_by_pp(pp_state_dicts, cfg.target_dir, tp_rank=tp_rank)
