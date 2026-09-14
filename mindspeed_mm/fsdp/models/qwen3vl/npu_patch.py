from . import modeling_qwen3_vl


def apply_qwen3vl_npu_patch():
    from mindspeed_mm.fsdp.ops.npu_patch import npu_fused_operator
    # Patches for Qwen3VL Model
    modeling_qwen3_vl.apply_rotary_pos_emb_vision = npu_fused_operator.apply_transformers_vision_rope_half_npu
    modeling_qwen3_vl.apply_rotary_pos_emb = npu_fused_operator.apply_transformers_rope_half_npu
    modeling_qwen3_vl.Qwen3VLTextRMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
