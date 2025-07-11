import os
from typing import cast, List

from checkpoint.common.constant import SAFE_MODE
from checkpoint.common.converter import Converter
from checkpoint.vlm_model import hf_to_mm
from checkpoint.vlm_model.config import ConvertVppMMConfig, ConvertHFConfig, ConvertResplitConfig
from checkpoint.vlm_model.converters.qwen2_5vl import create_qwen2_5_vl_ops, qwen2_5_vl_tp_patterns
from checkpoint.vlm_model.hf_to_mm import vision_schema, text_schema
from checkpoint.vlm_model.operator import Operator, RenameOp, ExpertUpGateMergeOp, GLUSplit, ColSplit


def create_qwen3_vl_ops(vit_embed_dim: int, vit_num_heads: int, llm_num_query_groups: int, llm_q_size: int,
                        llm_kv_size: int) -> List[Operator]:
    """qwen3vl 在qwen2.5vl的基础上增加了moe相关的专家参数，需要增加映射逻辑"""
    ops = [
              RenameOp(
                  (
                      # 处理 MoE专家相关
                      (r'model.layers.(\d+).mlp.gate.weight', r'text_decoder.decoder.layers.(\d+).mlp.router.weight'),
                      (r'model.layers.(\d+).mlp.experts.(\d+)\.down_proj',
                       r'text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc2'),
                      # 处理 q, k layernorm
                      (r'model.layers.(\d+).self_attn.q_norm.weight',
                       r'text_decoder.decoder.layers.(\d+).self_attention.q_layernorm.weight'),
                      (r'model.layers.(\d+).self_attn.k_norm.weight',
                       r'text_decoder.decoder.layers.(\d+).self_attention.k_layernorm.weight')
                  )
              ),
              ExpertUpGateMergeOp(
                  raw_names=[r"model.layers.(\d+).mlp.experts.(\d+).gate_proj.weight",
                             r"model.layers.(\d+).mlp.experts.(\d+).up_proj.weight"],
                  new_name=r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1.weight"
              )
          ] + create_qwen2_5_vl_ops(vit_embed_dim, vit_num_heads, llm_num_query_groups, llm_q_size, llm_kv_size)
    return ops


# qwen3vl的tp切分在qwen2.5vl的tp切分基础上，增加了expert的tp切分逻辑
qwen3_vl_tp_patterns = {
    **qwen2_5_vl_tp_patterns,
    **{
        r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1.weight": GLUSplit,
        r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc2.weight": ColSplit,
    }
}


class Qwen3_VLConverter(Converter):
    """Qwen3VL模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(config, llm_config=None) -> List[Operator]:
        from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
        config = cast(Qwen2_5_VLConfig, config)
        num_key_value_heads = llm_config.config.num_key_value_heads if llm_config is not None else config.num_key_value_heads
        llm_head_hidden_size = config.hidden_size // config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.num_attention_heads // config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        return create_qwen3_vl_ops(config.vision_config.hidden_size,
                                   config.vision_config.num_heads,
                                   num_key_value_heads,
                                   llm_q_size,
                                   llm_kv_size
                                   )

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfig):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops = Qwen3_VLConverter._create_ops(cfg.hf_config.config)
        hf_to_mm.convert_hf_to_mm(cfg, cfg.hf_config.config, ops, qwen3_vl_tp_patterns, [vision_schema, text_schema])
        # 安全管控权限
        os.chmod(cfg.mm_dir, SAFE_MODE)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        pass

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass
