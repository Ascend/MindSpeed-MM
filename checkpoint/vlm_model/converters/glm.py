#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : glm_hf_to_mm.py
@Time    : 2025/07/01
@Desc    : glm4.1v huggingface模型转换成mindspeed-mm模型

"""
import os
from copy import deepcopy
from typing import Callable, Any, List, Dict, cast
import torch
from tqdm import tqdm

from checkpoint.common.converter import Converter
from checkpoint.common.types import STATE_DICT_T
from checkpoint.vlm_model.operator import TieOp
from checkpoint.common.constant import SAFE_MODE
from checkpoint.vlm_model.operator import Operator
from checkpoint.vlm_model.hf_to_mm import load_from_hf, convert_hf_to_mm, \
    PPStageSchema, save_by_vpp, merge_vpp_index, partition_state_dict_by_pp
from checkpoint.vlm_model.config import ConvertMMConfig, ConvertVppMMConfig, ConvertHFConfig, ConvertResplitConfig
from checkpoint.vlm_model.operator import (
    Operator, UpGateMergeOp, QKVMergeOp, RelocateOp, RenameOp, RowSplit, GLUSplit, ColSplit
)

glm_text_schema = PPStageSchema(
    firsts=['text_decoder.embedding.'],
    lasts=['text_decoder.decoder.final_layernorm.', 'text_decoder.output_layer.'],
    middle='text_decoder.decoder.layers.'
)
glm_vision_schema = PPStageSchema(
    firsts=['image_encoder.encoder.patch_embed.', "image_encoder.encoder.post_conv_layernorm.", "image_encoder.encoder.embeddings.position_embedding"],
    lasts=['image_encoder.projector.', "image_encoder.encoder.blocks.final_layernorm", "image_encoder.encoder.downsample"],
    middle='image_encoder.encoder.blocks.layers.'
)


def create_glm_ops(vit_embed_dim: int, vit_num_heads: int, llm_num_query_groups: int,
                       llm_q_size: int, llm_kv_size: int) -> List[Operator]:
    """glm4v 权重转换逻辑"""
    ops = [
        UpGateMergeOp(raw_names=[r"model.visual.blocks.(\d+).mlp.gate_proj.weight", r"model.visual.blocks.(\d+).mlp.up_proj.weight"],
                      new_name=r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight"),
        RenameOp(
            (
                (r'model.visual.patch_embed.proj', r'image_encoder.encoder.patch_embed.proj'),
                (r'model.visual.post_conv_layernorm', r'image_encoder.encoder.post_conv_layernorm'),
                (r'model.visual.embeddings.position_embedding', r'image_encoder.encoder.embeddings.position_embedding'),
                (r'model.visual.blocks.(\d+).attn.proj', r'image_encoder.encoder.blocks.layers.(\d+).self_attention.proj'),
                (r'model.visual.blocks.(\d+).attn.qkv', r'image_encoder.encoder.blocks.layers.(\d+).self_attention.qkv'),
                (r'model.visual.blocks.(\d+).mlp.down_proj', r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2'),
                (r'model.visual.blocks.(\d+).norm1', r'image_encoder.encoder.blocks.layers.(\d+).input_layernorm'),
                (r'model.visual.blocks.(\d+).norm2', r'image_encoder.encoder.blocks.layers.(\d+).pre_mlp_layernorm'),
                (r'model.visual.post_layernorm', r'image_encoder.encoder.blocks.final_layernorm'),
                (r'model.visual.downsample', r'image_encoder.encoder.downsample'),
                (r'model.visual.merger.proj', r'image_encoder.projector.proj'),
                (r'model.visual.merger.post_projection_norm', r'image_encoder.projector.post_projection_norm'),
                (r'model.visual.merger.gate_proj', r'image_encoder.projector.gate_proj'),
                (r'model.visual.merger.up_proj', r'image_encoder.projector.up_proj'),
                (r'model.visual.merger.down_proj', r'image_encoder.projector.down_proj'),
                (r'model.language_model.embed_tokens', r'text_decoder.embedding.word_embeddings'),
                (r'model.language_model.layers.(\d+).input_layernorm', r'text_decoder.decoder.layers.(\d+).input_layernorm'),
                (r'model.language_model.layers.(\d+).mlp.down_proj', r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2'),
                (r'model.language_model.layers.(\d+).mlp.gate_up_proj', r'text_decoder.decoder.layers.(\d+).mlp.linear_fc1'),
                (r'model.language_model.layers.(\d+).post_attention_layernorm', r'text_decoder.decoder.layers.(\d+).pre_mlp_layernorm'),
                (r'model.language_model.layers.(\d+).post_self_attn_layernorm', r'text_decoder.decoder.layers.(\d+).post_self_attn_layernorm'),
                (r'model.language_model.layers.(\d+).post_mlp_layernorm', r'text_decoder.decoder.layers.(\d+).post_mlp_layernorm'),
                (r'model.language_model.norm', r'text_decoder.decoder.final_layernorm'),
                (r'lm_head', r'text_decoder.output_layer'),
                (r"model.language_model.layers.(\d+).self_attn.q_proj", r"text_decoder.decoder.layers.(\d+).self_attention.q_proj"),
                (r"model.language_model.layers.(\d+).self_attn.k_proj", r"text_decoder.decoder.layers.(\d+).self_attention.k_proj"),
                (r"model.language_model.layers.(\d+).self_attn.v_proj", r"text_decoder.decoder.layers.(\d+).self_attention.v_proj"),
                (r'model.language_model.layers.(\d+).self_attn.o_proj', r'text_decoder.decoder.layers.(\d+).self_attention.o_proj'),
            )
        ),
    ]
    return ops


glm_tp_patterns = {}


class GlmConverter(Converter):
    """GLM4V 模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(config: Any):
        from transformers.models.glm4v import Glm4vConfig
        config = cast(Glm4vConfig, config)
        llm_head_hidden_size = config.hidden_size // config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.num_attention_heads // config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        ops = create_glm_ops(config.vision_config.hidden_size,
                                 config.vision_config.num_heads,
                                 config.num_key_value_heads,
                                 llm_q_size,
                                 llm_kv_size
                                 )
        return ops, config

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfig):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops, _ = GlmConverter._create_ops(cfg.hf_config.config)
        convert_hf_to_mm(cfg, cfg.hf_config.config, ops, glm_tp_patterns, [glm_vision_schema, glm_text_schema])
        # 安全管控权限
        os.chmod(cfg.mm_dir, SAFE_MODE)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        pass

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass