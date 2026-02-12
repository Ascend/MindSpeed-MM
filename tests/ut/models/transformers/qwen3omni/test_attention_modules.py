import copy

import pytest
import torch
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeVisionEncoderConfig,
    Qwen3OmniMoeTextConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioAttention as Qwen3OmniMoeAudioAttentionGoden
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionAttention as Qwen3OmniMoeVisionAttentionGoden
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextAttention as Qwen3OmniMoeThinkerTextAttentionGoden

from mindspeed_mm.models.transformers.qwen3omni.modules import (
    Qwen3OmniMoeAudioAttention,
    Qwen3OmniMoeVisionAttention,
    Qwen3OmniMoeThinkerTextAttention,
)
from mindspeed_mm.models.transformers.qwen3vl.utils import set_seq_len
from tests.ut.utils import judge_expression


@pytest.fixture()
def audio_config():
    audio_config = {
        "_name_or_path": "",
        "activation_dropout": 0,
        "activation_function": "gelu",
        "add_cross_attention": False,
        "attention_dropout": 0,
        "chunk_size_feed_forward": 0,
        "conv_chunksize": 500,
        "d_model": 1280,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "downsample_hidden_size": 480,
        "dropout": 0,
        "early_stopping": False,
        "encoder_attention_heads": 20,
        "encoder_ffn_dim": 5120,
        "encoder_layers": 32,
        "encoder_no_repeat_ngram_size": 0,
        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
        },
        "initializer_range": 0.02,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
        },
        "length_penalty": 1.0,
        "max_length": 20,
        "max_source_positions": 1500,
        "min_length": 0,
        "model_type": "qwen3_omni_moe_audio_encoder",
        "n_window": 50,
        "n_window_infer": 800,
        "no_repeat_ngram_size": 0,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 32,
        "num_mel_bins": 128,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_dim": 2048,
        "output_hidden_states": False,
        "output_scores": False,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "scale_embedding": False,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "top_k": 50,
        "top_p": 1.0,
        "torchscript": False,
        "typical_p": 1.0,
        "use_bfloat16": False
    }
    config = Qwen3OmniMoeAudioEncoderConfig(**audio_config)
    setattr(config, "attn_layout", "BNSD")
    return config


@pytest.fixture()
def vision_config():
    vision_config = {
        "_name_or_path": "",
        "add_cross_attention": False,
        "apply_vit_abs_pos_embed": True,
        "chunk_size_feed_forward": 0,
        "deepstack_visual_indexes": [
            8,
            16,
            24
        ],
        "depth": 27,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
        },
        "image_size": 768,
        "in_channels": 3,
        "in_chans": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
        },
        "length_penalty": 1.0,
        "max_length": 20,
        "min_length": 0,
        "model_type": "qwen3_omni_moe_vision_encoder",
        "no_repeat_ngram_size": 0,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_heads": 16,
        "num_return_sequences": 1,
        "out_hidden_size": 2048,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "patch_size": 16,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "spatial_merge_size": 2,
        "spatial_patch_size": 16,
        "temperature": 1.0,
        "temporal_patch_size": 2,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokens_per_second": 2,
        "top_k": 50,
        "top_p": 1.0,
        "torchscript": False,
        "typical_p": 1.0,
        "use_bfloat16": False
    }
    config = Qwen3OmniMoeVisionEncoderConfig(**vision_config)
    setattr(config, "attn_layout", "BNSD")
    return config


@pytest.fixture()
def text_config():
    text_config = {
        "_name_or_path": "",
        "add_cross_attention": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "chunk_size_feed_forward": 0,
        "decoder_sparse_step": 1,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
        },
        "initializer_range": 0.02,
        "intermediate_size": 768,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
        },
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 65536,
        "min_length": 0,
        "mlp_only_layers": [],
        "model_type": "qwen3_omni_moe_text",
        "moe_intermediate_size": 768,
        "no_repeat_ngram_size": 0,
        "norm_topk_prob": True,
        "num_attention_heads": 32,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_router_logits": False,
        "output_scores": False,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "interleaved": True,
            "mrope_interleaved": True,
            "mrope_section": [
            24,
            20,
            20
            ],
            "rope_type": "default",
            "type": "default"
        },
        "rope_theta": 1000000,
        "router_aux_loss_coef": 0.001,
        "shared_expert_intermediate_size": 0,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": False,
        "top_k": 50,
        "top_p": 1.0,
        "torchscript": False,
        "typical_p": 1.0,
        "use_bfloat16": False,
        "use_cache": True,
        "use_qk_norm": True,
        "use_sliding_window": False,
        "vocab_size": 152064
    }
    config = Qwen3OmniMoeTextConfig(**text_config)
    setattr(config, "attn_layout", "BNSD")
    return config


@pytest.fixture()
def setup_audio_attention(audio_config):
    goden_config = copy.deepcopy(audio_config)
    config = copy.deepcopy(audio_config)
    attention_goden = Qwen3OmniMoeAudioAttentionGoden(goden_config).npu()
    attention = Qwen3OmniMoeAudioAttention(config).npu()
    attention.load_state_dict(attention_goden.state_dict())
    return attention_goden, attention


@pytest.fixture()
def setup_vision_attention(vision_config):
    goden_config = copy.deepcopy(vision_config)
    config = copy.deepcopy(vision_config)
    attention_goden = Qwen3OmniMoeVisionAttentionGoden(goden_config).npu()
    attention = Qwen3OmniMoeVisionAttention(config).npu()
    attention.load_state_dict(attention_goden.state_dict())
    return attention_goden, attention


@pytest.fixture()
def setup_text_attention(text_config):
    goden_config = copy.deepcopy(text_config)
    config = copy.deepcopy(text_config)
    attention_goden = Qwen3OmniMoeThinkerTextAttentionGoden(goden_config, layer_idx=0).npu()
    attention = Qwen3OmniMoeThinkerTextAttention(config, layer_idx=0).npu()
    attention.load_state_dict(attention_goden.state_dict())
    return attention_goden, attention


"""
Qwen3Omni Thinker Audio Attention Module Test Suite
"""


@pytest.fixture()
def audio_inputs_goden():
    hidden_states = torch.randn([2048, 1280], device="npu")
    cu_seqlens = torch.tensor([0, 1008, 2048], device='npu', dtype=torch.int32)
    return {
        "hidden_states": hidden_states,
        "cu_seqlens": cu_seqlens
    }


@pytest.fixture()
def audio_inputs(audio_inputs_goden):
    cu_seqlens = audio_inputs_goden["cu_seqlens"]
    cu_seqlens = cu_seqlens[1:] if len(cu_seqlens) > 1 else cu_seqlens
    cu_seqlens = tuple(cu_seqlens.cpu().numpy().tolist())
    return {
        "hidden_states": audio_inputs_goden["hidden_states"],
        "cu_seqlens": cu_seqlens
    }


def test_audio_attention_eager_mode(setup_audio_attention, audio_inputs_goden, audio_inputs):
    """test audio attention use eager fa"""
    goden_attn, attn = setup_audio_attention
    goden_attn.config._attn_implementation = "eager"
    attn.config._attn_implementation = "eager"
    goden_output_eager = goden_attn(**audio_inputs_goden)
    output_eager = attn(**audio_inputs)
    judge_expression(torch.all(goden_output_eager == output_eager))


def test_audio_attention_sdpa_mode(setup_audio_attention, audio_inputs_goden, audio_inputs):
    """test audio attention use sdpa"""
    goden_attn, attn = setup_audio_attention
    goden_attn.config._attn_implementation = "sdpa"
    attn.config._attn_implementation = "sdpa"
    goden_output_sdpa = goden_attn(**audio_inputs_goden)
    output_sdpa = attn(**audio_inputs)
    judge_expression(torch.all(goden_output_sdpa == output_sdpa))


def test_audio_attention_fa2_mode_tnd_layout(setup_audio_attention, audio_inputs_goden, audio_inputs):
    """test audio attention use fa2 with tnd layout"""
    goden_attn, attn = setup_audio_attention
    goden_attn.config._attn_implementation = "flash_attention_2"
    attn.config._attn_implementation = "flash_attention_2"
    attn.config.attn_layout = "TND"
    goden_output_varlen = goden_attn(**audio_inputs_goden)
    output_varlen = attn(**audio_inputs)
    judge_expression(torch.all(goden_output_varlen == output_varlen))


"""
Qwen3Omni Thinker Vision Attention Module Test Suite
"""


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


def test_vision_attention_fa2_mode_tnd_layout(setup_vision_attention, vision_inputs_goden, vision_inputs):
    """test vision attention use fa2 with tnd layout"""
    goden_attn, attn = setup_vision_attention
    goden_attn.config._attn_implementation = "flash_attention_2"
    attn.config._attn_implementation = "flash_attention_2"
    attn.config.attn_layout = "TND"
    goden_output_varlen = goden_attn(**vision_inputs_goden)
    output_varlen = attn(**vision_inputs)
    judge_expression(torch.all(goden_output_varlen == output_varlen))


"""
Qwen3Omni Thinker Text Attention Module Test Suite

Validates the consistency between the mindspeed_mm implementation of the Qwen3Omni thinker text attention module 
and the original transformers implementation, ensuring that the mathematical behavior of the model remains 
unchanged after optimization and refactoring.

Includes three attention implementation modes:
1. Eager mode: Standard PyTorch attention implementation
2. SDPA mode: Scaled Dot-Product Attention (optimized implementation)
3. Flash Attention 2: With TND layouts
4. Layout Comparison: Between bnsd and bsnd when using npu_fusion_attention
"""


@pytest.fixture()
def text_inputs(text_config):
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


def test_text_attention_fa2_mode_tnd_layout(setup_text_attention, text_inputs):
    """test text attention use fa2 with tnd layout"""
    goden_attn, attn = setup_text_attention
    goden_attn.config._attn_implementation = "flash_attention_2"
    attn.config._attn_implementation = "flash_attention_2"
    attn.config.attn_layout = "TND"
    goden_output_varlen = goden_attn(**text_inputs)[0]
    output_varlen = attn(**text_inputs)
    judge_expression(torch.all(goden_output_varlen == output_varlen))


def test_text_attention_fa2_mode_bsnd_and_bnsd(setup_text_attention, text_inputs):
    """test text attention use fa2 with bnsd and bsnd layout"""
    goden_attn, attn = setup_text_attention
    attn.config._attn_implementation = "flash_attention_2"
    attn.config.attn_layout = "BNSD"
    output_varlen = attn(**text_inputs)
    attn.config.attn_layout = "BSND"
    goden_output_varlen = attn(**text_inputs)
    judge_expression(torch.all(goden_output_varlen == output_varlen))