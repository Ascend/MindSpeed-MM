# The FLOPs calculation method is based on veomni, with some bugs fixed.

# Copyright 2025 Bytedance Ltd. and/or its affiliates

import argparse

from transformers import AutoConfig, PretrainedConfig


class Qwen3VLFlopsCounter:
    """Estimate training FLOPs for dense and MoE Qwen3-VL models."""

    def __init__(self, config: PretrainedConfig):
        self.config = config

    @staticmethod
    def _get_moe_layer_count(config):
        """Return the number of decoder layers that use routed experts."""
        num_experts = getattr(config, "num_experts", 0) or 0
        if num_experts <= 0:
            return 0

        sparse_step = getattr(config, "decoder_sparse_step", 1) or 1
        mlp_only_layers = set(getattr(config, "mlp_only_layers", None) or ())
        return sum(
            layer_idx not in mlp_only_layers and (layer_idx + 1) % sparse_step == 0
            for layer_idx in range(config.num_hidden_layers)
        )

    @staticmethod
    def _compute_attn_params(config):
        """
        Compute Qwen3-VL attention linear param count and layer info.

        Qwen3VLTextAttention projections:
            q_proj:  hidden_size -> num_attention_heads * head_dim
            k_proj:  hidden_size -> num_key_value_heads * head_dim
            v_proj:  hidden_size -> num_key_value_heads * head_dim
            o_proj:  num_attention_heads * head_dim -> hidden_size

        Note: Qwen3-VL may use head_dim != hidden_size // num_attention_heads,
        so head_dim is read from the config when present.
        """
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or hidden_size // num_attention_heads

        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # q_proj + k_proj + v_proj + o_proj
        attn_linear_N = hidden_size * (q_size + k_size + v_size + q_size)
        attn_linear_N *= config.num_hidden_layers

        return attn_linear_N, head_dim, num_attention_heads

    @staticmethod
    def _estimate_text_flops(config, tokens_sum, batch_seqlens):
        """
        Estimate the FLOPS of the Qwen3-VL text model (dense/MoE MLP + causal attention).

        Dense MLP per layer (SwiGLU, 3 projections):
            gate_proj:  hidden_size -> intermediate_size
            up_proj:    hidden_size -> intermediate_size
            down_proj:  intermediate_size -> hidden_size

        MoE layer:
            TopkGate router:  hidden_size -> num_experts
            Routed experts (top-k activated, each SwiGLU):
                gate_proj:  hidden_size -> moe_intermediate_size
                up_proj:    hidden_size -> moe_intermediate_size
                down_proj:  moe_intermediate_size -> hidden_size
                -> 3 projections * num_experts_per_tok active experts

        Attention per layer:
            q_proj:  hidden_size -> num_attention_heads * head_dim
            k_proj:  hidden_size -> num_key_value_heads * head_dim
            v_proj:  hidden_size -> num_key_value_heads * head_dim
            o_proj:  num_attention_heads * head_dim -> hidden_size

        LM head:
            lm_head:  hidden_size -> vocab_size

        Causal attention FLOPs:
            Q@K and attn@V use the lower triangular attention area.
            Forward coefficient 2 + backward coefficient 5 = 7.
        """
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers

        # attention linear projection params
        attn_linear_N, head_dim, num_attention_heads = Qwen3VLFlopsCounter._compute_attn_params(config)

        # MLP params: MoE or dense depending on config
        moe_layer_count = Qwen3VLFlopsCounter._get_moe_layer_count(config)
        dense_layer_count = num_hidden_layers - moe_layer_count

        # dense MLP per layer: gate_proj + up_proj + down_proj (SwiGLU)
        dense_mlp_N = hidden_size * config.intermediate_size * 3
        mlp_N = dense_mlp_N * dense_layer_count
        if moe_layer_count:
            # MoE per layer: router gate + routed expert MLPs (top-k)
            router_N = hidden_size * config.num_experts
            active_expert_N = hidden_size * config.moe_intermediate_size * config.num_experts_per_tok * 3
            mlp_N += (router_N + active_expert_N) * moe_layer_count

        # Notice: only lm_head, embedding is a table lookup
        lm_head_N = hidden_size * vocab_size
        linear_N = attn_linear_N + mlp_N + lm_head_N
        # linear projection flops: 6 (fwd + bwd) * params * tokens
        linear_flops = 6 * linear_N * tokens_sum

        # quadratic attention flops (Q@K and attn@V)
        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)

        # Notice: This is the attention flops calculation for the causal scenario,
        # which differs from veomni.
        # Forward pass: 2 * seqlen_square_sum * head_dim * num_heads
        # Backward pass: (4 + 1) * seqlen_square_sum * head_dim * num_heads.
        # This is because the FA operator recomputes the kv matrix during backward.
        # This implementation applies to both GPU and NPU.
        attention_flops = (
            7 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        )
        return linear_flops + attention_flops

    @staticmethod
    def _estimate_qwen3_vit_flop(images_seqlens, config, vit_freeze=False):
        """
        Estimate the FLOPS of the vision encoder
        """
        if config is None:
            return 0

        tokens_sum = sum(images_seqlens)
        num_heads = config.num_heads
        depth = config.depth
        dim = config.hidden_size
        mlp_hidden_dim = config.intermediate_size
        out_hidden_size = config.out_hidden_size

        head_dim = dim // num_heads
        spatial_merge_size = config.spatial_merge_size
        merger_hidden_size = dim * spatial_merge_size**2

        # every vision token's patch_embed comes from a conv of (C, T, H, W) -> (dim,)
        patch_embed_N = (
            dim
            * config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size
        )
        # Qwen3 VL vision mlp does not use GLU, thus 2.
        mlp_N = dim * mlp_hidden_dim * 2
        attn_linear_N = dim * (4 * dim)  # qkv and output proj
        merger_N = (out_hidden_size + merger_hidden_size) * merger_hidden_size

        # Qwen3 VL uses deep stack, one merger for every deepstack layer
        deepstack_merger_N = merger_N * len(getattr(config, "deepstack_visual_indexes", ()) or ())
        # non-attn all_layer parm
        dense_N = patch_embed_N + (mlp_N + attn_linear_N) * depth + deepstack_merger_N + merger_N

        if vit_freeze:
            # Frozen ViT only performs the forward pass.
            dense_rate = 2
            attn_rate = 4
        else:
            # Trainable ViT performs forward and backward, including FA KV recomputation.
            dense_rate = 6
            attn_rate = 14

        # non-attn all_layer & all_token flops
        linear_flops = dense_rate * dense_N * tokens_sum

        # In Qwen3 VL, full attention is used in all vision layers.
        full_attn_layer_num = depth

        # full attn layer & all_token flops
        seqlen_square_sum = sum(seqlen * seqlen for seqlen in images_seqlens)

        # Notice: This is the attention flops calculation for the full scenario.
        # Forward pass: 4 * seqlen_square_sum * head_dim * num_heads.
        # Backward pass: (8 + 2) * seqlen_square_sum * head_dim * num_heads.
        # This is because the FA operator recomputes the kv matrix during backward.
        # The backward pass is skipped when the ViT is frozen.
        attention_flops = attn_rate * seqlen_square_sum * head_dim * num_heads * full_attn_layer_num
        return linear_flops + attention_flops

    def _estimate_qwen3_vl_flops(self, tokens_sum, batch_seqlens, **kwargs):
        text_flops = self._estimate_text_flops(self.config.text_config, tokens_sum, batch_seqlens)

        # vit flops (Qwen3-VL ViT)
        images_seqlens = kwargs.get("images_seqlens")
        if images_seqlens:
            vit_flops = self._estimate_qwen3_vit_flop(
                images_seqlens,
                self.config.vision_config,
                kwargs.get("vit_freeze", False),
            )
        else:
            vit_flops = 0

        # all_layer & all_token flops
        return text_flops + vit_flops

    def estimate_flops(self, batch_seqlens, step_time, **kwargs):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
        """
        tokens_sum = sum(batch_seqlens)

        estimated_flops = self._estimate_qwen3_vl_flops(tokens_sum, batch_seqlens, **kwargs) / step_time

        return estimated_flops


def get_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL dense/MoE FLOPs Calculation Tool")
    parser.add_argument(
        "--vit_seqlens",
        type=int,
        default=[],
        nargs="+",
        help="seqlen in vit, only used when the model has a vision module",
    )
    parser.add_argument("--llm_seqlens", type=int, default=[16384], nargs="+", help="seqlen in language_model")
    parser.add_argument(
        "--hf_path",
        type=str,
        default="/home/weights/Qwen3-VL-30B-A3B-Instruct/",
        help="HuggingFace config path",
    )
    parser.add_argument("--device_num", type=int, default=1, help="Device num")
    parser.add_argument("--gbs", type=int, default=1, help="global batchsize")
    parser.add_argument("--step_time", type=float, help="Step time (s)")
    parser.add_argument("--vit_freeze", action="store_true", help="Whether to freeze the vision model")
    parser.add_argument("--hardware_flops", type=float, default=None, help="Hardware FLOPs")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    flopcounter = Qwen3VLFlopsCounter(config=AutoConfig.from_pretrained(args.hf_path))
    flops = flopcounter.estimate_flops(
        batch_seqlens=args.llm_seqlens,
        images_seqlens=args.vit_seqlens,
        vit_freeze=args.vit_freeze,
        step_time=args.step_time,
    )
    flops = flops * args.gbs / args.device_num
    print(f"flops: {flops:.4e}")

    if args.hardware_flops is not None:
        if args.hardware_flops > 0:
            mfu = flops / args.hardware_flops
            print(f"MFU is: {mfu * 100:.2f}%")
        else:
            raise ValueError("Hardware FLOPs must be a positive value.")


"""
e.g.:
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Dense or MoE Qwen3-VL with a frozen vision tower:
python mindspeed_mm/fsdp/tools/flops_tool/flops_qwen3_vl.py \
    --vit_seqlens 1024 \
    --llm_seqlens 16384 \
    --hf_path /home/weights/Qwen3-VL-30B-A3B-Instruct/ \
    --device_num 16 \
    --gbs 16 \
    --step_time 6.9 \
    --vit_freeze
"""
