import copy

import pytest
import torch
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention as Qwen3VLTextAttentionGoden
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention as Qwen3VLVisionAttentionGoden

from mindspeed_mm.models.transformers.qwen3vl.modules import Qwen3VLTextAttention, Qwen3VLVisionAttention
from mindspeed_mm.models.transformers.qwen3vl.utils import set_seq_len
from tests.ut.utils import judge_expression


@pytest.fixture()
def text_config():
    text_config = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "decoder_sparse_step": 1,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "max_position_embeddings": 128000,
        "mlp_only_layers": [],
        "model_type": "qwen3_vl_moe_text",
        "moe_intermediate_size": 768,
        "norm_topk_prob": True,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 4,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [
            24,
            20,
            20
        ],
        "rope_type": "default"
        },
        "rope_theta": 1000000,
        "use_cache": True,
        "vocab_size": 151936
    }
    config = Qwen3VLTextConfig(**text_config)
    setattr(config, "attn_layout", "BNSD")
    return config


@pytest.fixture()
def vision_config():
    vision_config = {
        "deepstack_visual_indexes": [
        8,
        16,
        24
        ],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "in_chans": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_vl_moe",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    }
    config = Qwen3VLVisionConfig(**vision_config)
    setattr(config, "attn_layout", "BNSD")
    return config


@pytest.fixture()
def setup_text_attention(text_config):
    goden_config = copy.deepcopy(text_config)
    config = copy.deepcopy(text_config)
    attention_goden = Qwen3VLTextAttentionGoden(goden_config, layer_idx=0).npu()
    attention = Qwen3VLTextAttention(config, layer_idx=0).npu()
    attention.load_state_dict(attention_goden.state_dict())
    return attention_goden, attention


@pytest.fixture()
def setup_vision_attention(vision_config):
    goden_config = copy.deepcopy(vision_config)
    config = copy.deepcopy(vision_config)
    attention_goden = Qwen3VLVisionAttentionGoden(goden_config).npu()
    attention = Qwen3VLVisionAttention(config).npu()
    attention.load_state_dict(attention_goden.state_dict())
    return attention_goden, attention


@pytest.fixture()
def text_inputs(text_config):
    text_config._attn_implementation = "eager"  # for create mask
    hidden_states = torch.randn([2, 488, 2048], device="npu")
    position_embeddings = (torch.randn([2, 488, 128], device="npu"), torch.randn([2, 488, 128], device="npu"))
    attention_mask = torch.tensor([[1] * 400 + [0] * 88, [1] * 468 + [0] * 20], dtype=torch.int64, device="npu")
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    cu_seqlens = cu_seqlens[1:] if len(cu_seqlens) > 1 else cu_seqlens
    cu_seqlens = tuple(cu_seqlens.cpu().numpy().tolist())
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    set_seq_len("total", hidden_states.shape[1])

    return {
        "hidden_states": hidden_states, 
        "position_embeddings": position_embeddings, 
        "attention_mask": attention_mask,
        "cu_seqlens": cu_seqlens,
        "indices": indices
    }


@pytest.fixture()
def text_attn_mask(text_config, text_inputs):
    text_config._attn_implementation = "eager"  # for create mask
    cache_position = torch.tensor([i for i in range(488)], dtype=torch.int64, device='npu')
    attention_mask = create_causal_mask(
        config=text_config,
        input_embeds=text_inputs["hidden_states"],
        attention_mask=text_inputs["attention_mask"],
        cache_position=cache_position,
        past_key_values=None,
        position_ids=None
    ).npu()
    return attention_mask


@pytest.fixture()
def vision_inputs_goden():
    hidden_states = torch.randn([2048, 1152], device="npu")
    cu_seqlens = torch.tensor([0, 1008, 2048], device='npu', dtype=torch.int32)
    position_embeddings = (torch.randn([2048, 72], device='npu'), torch.randn([2048, 72], device='npu'))
    return {
        "hidden_states": hidden_states,
        "cu_seqlens": cu_seqlens,
        "position_embeddings": position_embeddings
    }


@pytest.fixture()
def vision_inputs(vision_inputs_goden):
    cu_seqlens = vision_inputs_goden["cu_seqlens"]
    cu_seqlens = cu_seqlens[1:] if len(cu_seqlens) > 1 else cu_seqlens
    cu_seqlens = tuple(cu_seqlens.cpu().numpy().tolist())
    return {
        "hidden_states": vision_inputs_goden["hidden_states"],
        "cu_seqlens": cu_seqlens,
        "position_embeddings": vision_inputs_goden["position_embeddings"]
    }


def test_text_attention_eager_mode(setup_text_attention, text_inputs, text_attn_mask):
    """test text attention use eager attention"""
    goden_attn, attn = setup_text_attention
    goden_attn.config._attn_implementation = "eager"
    attn.config._attn_implementation = "eager"
    text_inputs.update({"attention_mask": text_attn_mask})
    goden_output_eager = goden_attn(**text_inputs)[0]
    output_eager = attn(**text_inputs)
    judge_expression(torch.all(goden_output_eager == output_eager))


def test_text_attention_sdpa_mode(setup_text_attention, text_inputs, text_attn_mask):
    """test text attention use sdpa"""
    goden_attn, attn = setup_text_attention
    goden_attn.config._attn_implementation = "sdpa"
    attn.config._attn_implementation = "sdpa"
    text_inputs.update({"attention_mask": ~text_attn_mask.bool()})
    goden_output_sdpa = goden_attn(**text_inputs)[0]
    output_sdpa = attn(**text_inputs)
    judge_expression(torch.all(goden_output_sdpa == output_sdpa))


def test_text_attention_varlen_mode(setup_text_attention, text_inputs):
    """test text attention use varlen fa"""
    goden_attn, attn = setup_text_attention
    goden_attn.config._attn_implementation = "flash_attention_2"  # transformers flash_attention_2 include varlen_fa
    attn.config._attn_implementation = "flash_attention_2"
    attn.config.attn_layout = "TND"
    goden_output_varlen = goden_attn(**text_inputs)[0]
    output_varlen = attn(**text_inputs)
    judge_expression(torch.all(goden_output_varlen == output_varlen))


def test_vision_attention_eager_mode(setup_vision_attention, vision_inputs_goden, vision_inputs):
    """test vision attention use eager fa"""
    goden_attn, attn = setup_vision_attention
    goden_attn.config._attn_implementation = "eager"
    attn.config._attn_implementation = "eager"
    goden_output_eager = goden_attn(**vision_inputs_goden)
    output_eager = attn(**vision_inputs)
    judge_expression(torch.all(goden_output_eager == output_eager))


def test_vision_attention_sdpa_mode(setup_vision_attention, vision_inputs_goden, vision_inputs):
    """test vision attention use sdpa"""
    goden_attn, attn = setup_vision_attention
    goden_attn.config._attn_implementation = "sdpa"
    attn.config._attn_implementation = "sdpa"
    goden_output_sdpa = goden_attn(**vision_inputs_goden)
    output_sdpa = attn(**vision_inputs)
    judge_expression(torch.all(goden_output_sdpa == output_sdpa))


def test_vision_attention_varlen_mode(setup_vision_attention, vision_inputs_goden, vision_inputs):
    """test vision attention use varlen"""
    goden_attn, attn = setup_vision_attention
    goden_attn.config._attn_implementation = "flash_attention_2"
    attn.config._attn_implementation = "flash_attention_2"
    attn.config.attn_layout = "TND"
    goden_output_varlen = goden_attn(**vision_inputs_goden)
    output_varlen = attn(**vision_inputs)
    judge_expression(torch.all(goden_output_varlen == output_varlen))
