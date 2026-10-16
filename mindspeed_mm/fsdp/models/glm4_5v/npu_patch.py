import torch
from transformers.models.glm4v_moe import modeling_glm4v_moe


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    import torch_npu

    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin).squeeze(0).to(orig_q_dtype)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin).squeeze(0).to(orig_k_dtype)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    import torch_npu

    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def apply_glm4v_npu_patch():
    modeling_glm4v_moe.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision
    modeling_glm4v_moe.apply_multimodal_rotary_pos_emb = apply_multimodal_rotary_pos_emb
