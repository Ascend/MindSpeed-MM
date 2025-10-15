from typing import Any, cast, List

from tqdm import tqdm

from checkpoint.common.converter import Converter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.vlm_model import hf_to_mm, mm_to_hf
from checkpoint.vlm_model.config import ConvertVppMMConfig, ConvertHFConfig, ConvertResplitConfig, ConvertTorchDCPConfig
from checkpoint.vlm_model.converters.qwen2vl import create_qwen2vl_ops, qwen2vl_tp_patterns
from checkpoint.vlm_model.hf_to_mm import vision_schema, text_schema, split_by_tp, merge_vpp_index, \
    partition_state_dict_by_pp, save_by_vpp
from checkpoint.vlm_model.mm_to_hf import load_from_mm, merge_by_tp
from checkpoint.vlm_model.operator import (
    Operator, UpGateMergeOp, RenameOp, GLUSplit
)


def create_qwen2_5_vl_ops(vit_embed_dim: int, vit_num_heads: int, llm_num_query_groups: int,
                          llm_q_size: int, llm_kv_size: int) -> List[Operator]:
    """qwen2.5vl在qwen2vl的基础上vit的mlp变成了glu模式、需要增加合并处理逻辑"""
    ops = [
              UpGateMergeOp(
                  raw_names=[r"visual.blocks.(\d+).mlp.gate_proj.weight", r"visual.blocks.(\d+).mlp.up_proj.weight"],
                  new_name=r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight"),
              UpGateMergeOp(
                  raw_names=[r"visual.blocks.(\d+).mlp.gate_proj.bias", r"visual.blocks.(\d+).mlp.up_proj.bias"],
                  new_name=r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias"),
              RenameOp(
                  patterns=((r'visual.blocks.(\d+).mlp.down_proj',
                             r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2'),))
          ] + create_qwen2vl_ops(vit_embed_dim, vit_num_heads, llm_num_query_groups, llm_q_size, llm_kv_size)
    return ops


#  qwen2.5vl的tp切分在qwen2vl的tp切分基础上，修改了vit中mlp的tp切分逻辑，适应glu结构
qwen2_5_vl_tp_patterns = {**qwen2vl_tp_patterns,
                          **{r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias": GLUSplit,
                             r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight": GLUSplit}
                          }


class ConvertVppMMConfigQwen2_5(ConvertVppMMConfig):

    def model_post_init(self, _context):
        from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
        config = cast(Qwen2_5_VLConfig, self.hf_config.config)
        self.common_model_config.num_key_value_heads = config.num_key_value_heads
        self.common_model_config.llm_num_layers = config.num_hidden_layers
        self.common_model_config.vit_num_layers = config.vision_config.depth
        self.common_model_config.tie_word_embeddings = config.tie_word_embeddings


class ConvertTorchDCPMMConfigQwen2_5(ConvertTorchDCPConfig):
    def model_post_init(self, _context):
        from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
        config = cast(Qwen2_5_VLConfig, self.hf_config.config)
        self.common_model_config.num_key_value_heads = config.num_key_value_heads
        self.common_model_config.llm_num_layers = config.num_hidden_layers
        self.common_model_config.vit_num_layers = config.vision_config.depth
        self.common_model_config.tie_word_embeddings = config.tie_word_embeddings


class Qwen2_5_VLConverter(Converter):
    """Qwen2.5VL模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(config: Any) -> List[Operator]:
        from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
        config = cast(Qwen2_5_VLConfig, config)
        # qwen2.5vl和qwen2vl的差异主要在权重转换的算子以及tp转换时的模式
        llm_head_hidden_size = config.hidden_size // config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.num_attention_heads // config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        ops = create_qwen2_5_vl_ops(config.vision_config.hidden_size,
                                    config.vision_config.num_heads,
                                    config.num_key_value_heads,
                                    llm_q_size,
                                    llm_kv_size
                                    )
        return ops

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfigQwen2_5):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops = Qwen2_5_VLConverter._create_ops(cfg.hf_config.config)

        hf_to_mm.convert_hf_to_mm(cfg, ops, qwen2_5_vl_tp_patterns, [vision_schema, text_schema])
        # 安全管控权限
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        ops = Qwen2_5_VLConverter._create_ops(cfg.hf_config.config)
        mm_to_hf.convert_mm_to_hf(cfg, ops, qwen2_5_vl_tp_patterns)
        # 安全管控权限
        set_directory_permissions(cfg.save_hf_dir)

    @staticmethod
    def hf_to_mm_dcp(cfg: ConvertTorchDCPMMConfigQwen2_5):
        ops = Qwen2_5_VLConverter._create_ops(cfg.hf_config.config)
        hf_to_mm.convert_hf_to_mm_dcp(cfg, ops)
        # set directory permission for security control 
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        source = cfg.source_parallel_config
        target = cfg.target_parallel_config
        tp_state_dicts = load_from_mm(cfg.source_dir, source.vit_pp_layers, source.llm_pp_layers, source.tp_size)
        state_dict = merge_by_tp(tp_state_dicts=tp_state_dicts, patterns=qwen2_5_vl_tp_patterns)
        tp_state_dicts = split_by_tp(state_dict=state_dict, patterns=qwen2_5_vl_tp_patterns, tp_size=target.tp_size)
        pp_ranges = merge_vpp_index([target.vit_pp_layers], [target.llm_pp_layers], [[]])
        for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
            pp_state_dicts = partition_state_dict_by_pp(tp_state_dict, pp_ranges, [vision_schema, text_schema])
            save_by_vpp(pp_state_dicts, cfg.target_dir,
                        pp_and_vpp_size=(target.pp_size, 1),
                        tp_rank=tp_rank)
        # 安全管控权限
        set_directory_permissions(cfg.target_dir)
