import pytest
import mindspeed.megatron_adaptor
import torch
import torch_npu

from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import apply_rotary_pos_emb_vision, apply_multimodal_rotary_pos_emb
from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import VisionRotaryEmbedding
from tests.ut.utils import judge_expression
from tests.ut.utils import TestConfig

VISION_TENSOR = "/home/ci_resource/data/qwen2vl/rope_test/vision_tensor.pt"
VISION_FREQS = "/home/ci_resource/data/qwen2vl/rope_test/vision_freqs.pt"
LLM_QUERY = "/home/ci_resource/data/qwen2vl/rope_test/test_q.pt"
LLM_KEY = "/home/ci_resource/data/qwen2vl/rope_test/test_k.pt"
LLM_COS = "/home/ci_resource/data/qwen2vl/rope_test/cos.pt"
LLM_SIN = "/home/ci_resource/data/qwen2vl/rope_test/sin.pt"

MROPE_SECTION = [16, 24, 24]
VISION_OUTPUT = -5.0529327392578125
VISION_FUSION_OUTPUT = -5.0529327392578125
QUERY_EMBED_OUTPUT = -9.803374290466309
KEY_EMBED_OUTPUT = -9.747139930725098
QUERY_EMBED_FUSION_OUTPUT = -9.803374290466309
KEY_EMBED_FUSION_OUTPUT = -9.747139930725098
MAX_GRID_SIZE = 36
VISION_ROPE_OUTPUT_SUM = 1706.948974609375
VISION_ROPE_OUTPUT_SHAPE = [36, 20]


class TestQwen2vlROPE:
    """
    Qwen2VL MROPE test case.
    """


    def test_apply_rotary_pos_emb_vision(self):
        """
        Test apply_rotary_pos_emb_vision function.
        """
        vision_tensor = torch.load(VISION_TENSOR, map_location="cpu")
        cos = torch.load(VISION_FREQS, map_location="cpu").cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(1).float()
        sin = torch.load(VISION_FREQS, map_location="cpu").sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(1).float()
        vision_freqs = torch.concat((cos.npu(), sin.npu()), dim=0)
        use_fused_rope = False
        output = apply_rotary_pos_emb_vision(vision_tensor.npu().transpose(0, 1), vision_freqs, use_fused_rope=use_fused_rope)
        judge_expression(output.min().item() == VISION_OUTPUT)

    def test_apply_rotary_pos_emb_vision_fusion(self):
        """
        Test apply_rotary_pos_emb_vision function based on npu_rotary_mul fusion operator.
        """
        vision_tensor = torch.load(VISION_TENSOR, map_location="cpu")
        cos = torch.load(VISION_FREQS, map_location="cpu").cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(1).float()
        sin = torch.load(VISION_FREQS, map_location="cpu").sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(1).float()
        vision_freqs = torch.concat((cos.npu(), sin.npu()), dim=0)
        use_fused_rope = True
        output = apply_rotary_pos_emb_vision(vision_tensor.npu().transpose(0, 1), vision_freqs, use_fused_rope=use_fused_rope)
        judge_expression(output.min().item() == VISION_FUSION_OUTPUT)

    def test_apply_multimodal_rotary_pos_emb(self):
        """
        Test apply_multimodal_rotary_pos_emb function.
        """
        test_q = torch.load(LLM_QUERY, map_location="cpu").permute(2, 0, 1, 3).contiguous()
        test_k = torch.load(LLM_KEY, map_location="cpu").permute(2, 0, 1, 3).contiguous()
        sin = torch.load(LLM_SIN, map_location="cpu").permute(2, 0, 1, 3).contiguous()
        cos = torch.load(LLM_COS, map_location="cpu").permute(2, 0, 1, 3).contiguous()
        mrope_section = MROPE_SECTION * 2
        use_fused_rope = False
        cos = torch.cat([m[:, i % 3, :, :] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[:, i % 3, :, :] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        q_embed, k_embed = apply_multimodal_rotary_pos_emb(test_q.npu(), test_k.npu(), cos.npu(), sin.npu(), use_fused_rope=use_fused_rope)
        judge_expression(q_embed.min().item() == QUERY_EMBED_OUTPUT)
        judge_expression(k_embed.min().item() == KEY_EMBED_OUTPUT)

    def test_apply_multimodal_rotary_pos_emb_fusion(self):
        """
        Test apply_multimodal_rotary_pos_emb function based on npu_rotary_mul fusion operator.
        """
        test_q = torch.load(LLM_QUERY, map_location="cpu").permute(2, 0, 1, 3).contiguous()
        test_k = torch.load(LLM_KEY, map_location="cpu").permute(2, 0, 1, 3).contiguous()
        sin = torch.load(LLM_SIN, map_location="cpu").permute(2, 0, 1, 3).contiguous()
        cos = torch.load(LLM_COS, map_location="cpu").permute(2, 0, 1, 3).contiguous()
        mrope_section = MROPE_SECTION * 2
        use_fused_rope = True
        cos = torch.cat([m[:, i % 3, :, :] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[:, i % 3, :, :] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        q_embed, k_embed = apply_multimodal_rotary_pos_emb(test_q.npu(), test_k.npu(), cos.npu(), sin.npu(), use_fused_rope=use_fused_rope)
        judge_expression(q_embed.min().item() == QUERY_EMBED_FUSION_OUTPUT)
        judge_expression(k_embed.min().item() == KEY_EMBED_FUSION_OUTPUT)

    def test_vision_rotary_embedding(self):
        vision_config = {
            "hidden_size": 1280,
            "num_attention_heads": 16,
        }
        config = TestConfig(vision_config)
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        rotary_pos_emb = rotary_pos_emb.npu()
        max_grid_size = MAX_GRID_SIZE
        rotary_pos_emb_output = rotary_pos_emb(max_grid_size)
        judge_expression(rotary_pos_emb_output.shape == torch.Size(VISION_ROPE_OUTPUT_SHAPE))
        judge_expression(rotary_pos_emb_output.sum().item() == VISION_ROPE_OUTPUT_SUM)
