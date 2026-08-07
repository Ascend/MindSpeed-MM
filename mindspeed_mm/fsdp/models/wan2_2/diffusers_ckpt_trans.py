# Copyright 2025 Bytedance Ltd. and/or its affiliates
"""Diffusers -> native WanModel key mapping tables.
"""

# Exact key renames (old_key -> new_key)
DIFFUSERS_CONVERT_MAPPING = {
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
}

# Sub-string replacements applied to every key
DIFFUSERS_STR_REPLACE_MAPPING = {
    "attn1.norm_q": "self_attn.norm_q",
    "attn1.norm_k": "self_attn.norm_k",
    "attn2.norm_q": "cross_attn.norm_q",
    "attn2.norm_k": "cross_attn.norm_k",
    "attn1.to_q.": "self_attn.q.",
    "attn1.to_k.": "self_attn.k.",
    "attn1.to_v.": "self_attn.v.",
    "attn1.to_out.0.": "self_attn.o.",
    "attn2.to_q.": "cross_attn.q.",
    "attn2.to_k.": "cross_attn.k.",
    "attn2.to_v.": "cross_attn.v.",
    "attn2.to_out.0.": "cross_attn.o.",
    ".ffn.net.0.proj.": ".ffn.0.",
    ".ffn.net.2.": ".ffn.2.",
    "scale_shift_table": "modulation",
    ".norm2.": ".norm3.",
}


from mindspeed_mm.fsdp.checkpoint.convert import (
    DiffusersKeyMapTransformPipeline,
    WEIGHT_TRANSFORM_PIPELINES,
)


class Wan22DiffusersTransformPipeline(DiffusersKeyMapTransformPipeline):
    """Wan2.2 diffusers -> native transform pipeline, driven by the tables above."""

    CONVERT_MAPPING = DIFFUSERS_CONVERT_MAPPING
    STR_REPLACE_MAPPING = DIFFUSERS_STR_REPLACE_MAPPING


WEIGHT_TRANSFORM_PIPELINES["wan2_2"] = Wan22DiffusersTransformPipeline
