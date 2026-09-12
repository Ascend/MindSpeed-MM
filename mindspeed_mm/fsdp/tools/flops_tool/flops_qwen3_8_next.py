import argparse
from transformers import PretrainedConfig, AutoConfig


class Qwen38NextFlopsCounter:
    def __init__(self, config: PretrainedConfig):
        self.config = config

    def _estimate_qwen3_vit_flop(self, images_seqlens, config):
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

        spatial_merge_size = config.spatial_merge_size

        head_dim = dim // num_heads

        # every vision token's patch_embed comes from a conv of (C, T, H, W) -> (dim,)
        patch_embed_N = dim * config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
        # Qwen3 VL vision mlp does not use GLU, thus 2.
        mlp_N = dim * mlp_hidden_dim * 2
        attn_linear_N = dim * (4 * dim)  # qkv and output proj
        merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (dim * (spatial_merge_size**2))

        # Qwen3 VL uses deep stack, one merger for every deepstack layer
        deepstack_merger_N = merger_N * len(config.deepstack_visual_indexes)
        # non-attn all_layer parm
        dense_N = patch_embed_N + (mlp_N + attn_linear_N) * depth + deepstack_merger_N + merger_N

        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # In Qwen3 VL, full attention is used in all vision layers.
        full_attn_layer_num = depth

        # full attn layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in images_seqlens:
            seqlen_square_sum += seqlen * seqlen

        # Notice: This is the attention flops calculation for the full scenario,
        # which differs from veomni.
        # Forward pass: 4 * seqlen_square_sum * head_dim * num_heads
        # Backward pass: (8 + 2) * seqlen_square_sum * head_dim * num_heads.
        # This is because the FA operator recomputes the kv matrix during backward.
        # This implementation applies to both GPU and NPU.
        attn_qkv_flops = 14 * seqlen_square_sum * head_dim * num_heads * full_attn_layer_num

        vit_flops = dense_N_flops + attn_qkv_flops

        return vit_flops

    @staticmethod
    def _get_layer_counts(config):
        num_qsa_layers = sum(
            layer_type == "full_attention"
            for layer_type in config.layer_types
        )
        num_gdn_layers = sum(
            layer_type == "linear_attention"
            for layer_type in config.layer_types
        )
        return num_qsa_layers, num_gdn_layers

    @staticmethod
    def _compute_ple_flops(config, tokens_sum):
        """
        PLE parameter FLOPs excluding the N-gram embedding table itself.

        Qwen4ExpTextPLELayer:
          key_proj: ple_dim -> hc_count * hidden_size
          value_proj: ple_dim -> hidden_size
          depthwise Conv1D: channels = hc_count * hidden_size, kernel = conv_kernel_size
        N-gram table: embedding lookup only -> 0 FLOPs in MFU convention.
        """
        ple_layer_ids = getattr(config, "ple_layer_ids", [])
        if not ple_layer_ids:
            return 0

        hidden_size = config.hidden_size
        hc_hidden_size = config.hc_count * hidden_size

        ple_dim = getattr(config, "ple_embed_dim", hidden_size)
        key_proj_N = ple_dim * hc_hidden_size
        value_proj_N = ple_dim * hidden_size
        conv_kernel = getattr(config, "ple_conv_kernel_size", getattr(config, "linear_conv_kernel_dim", 4))
        conv_N = hc_hidden_size * conv_kernel

        per_ple_layer_N = (key_proj_N + value_proj_N+ conv_N)

        return 6 * tokens_sum * per_ple_layer_N * len(ple_layer_ids)

    @staticmethod
    def _compute_lm_head_flops(config, tokens_sum):
        """
        Untied LM head: hidden_size -> vocab_size
        """
        tie_word_embeddings = getattr(
            config,
            "tie_word_embeddings",
            False,
        )
        if tie_word_embeddings:
            # Even when tied, the matrix multiplication still executes.
            # For FLOPs it must still be counted.
            pass

        return 6 * tokens_sum * config.hidden_size * config.vocab_size

    @staticmethod
    def _compute_hc_flops(config, tokens_sum):
        """
        Qwen4ExpTextGatedResidual:

          hc_hidden = hc_count * hidden_size

          input_mix_weight_down: hc_hidden -> hc_lowrank
          input_mix_weight_up: hc_lowrank -> hc_hidden
          block_inject_weight: hc_hidden -> hc_count

        Each decoder layer contains:
          - attn_hyper_connection
          - mlp_hyper_connection

        Final model mixer uses use_combine=False and therefore has no block_inject_weight.
        """
        hidden_size = config.hidden_size
        hc_count = config.hc_count
        hc_lowrank = config.hc_lowrank
        hc_hidden_size = hc_count * hidden_size

        one_hc_N = (
            hc_hidden_size * hc_lowrank
            + hc_lowrank * hc_hidden_size
            + hc_hidden_size * hc_count
        )
        decoder_hc_N = 2 * one_hc_N * config.num_hidden_layers

        final_hc_N = (
            hc_hidden_size * hc_lowrank
            + hc_lowrank * hc_hidden_size
        )

        return 6 * tokens_sum * (decoder_hc_N + final_hc_N)

    @staticmethod
    def _sum_floor_div(sequence_length, divisor):
        """
        sum_{t=1..S} floor(t / divisor), Exact O(1) formula.
        """
        q, r = divmod(sequence_length, divisor)
        return (divisor * q * (q - 1) // 2 + q * (r + 1))

    @classmethod
    def _compute_qsa_indexer_flops(cls, config, seqlens, num_qsa_layers):
        """
        QSA indexer scoring FLOPs.
        For each query position t: num_blocks(t) = floor(visible_tokens / compress_ratio)

        Each block is scored by: indexer_n_heads query heads against one/shared block key
        QK forward: 2 * n_heads * head_dim * num_blocks
        Backward of a matmul contributes two corresponding matmuls, giving: 6 * n_heads * head_dim * num_blocks
        Block mean pooling / RMSNorm / ReLU / reduce / TopK are not included in Tensor-MFU FLOPs.
        """
        ratio = config.indexer_compress_ratio
        index_heads = config.indexer_n_heads
        index_head_dim = config.indexer_head_dim

        total_block_queries = 0
        for seq_len in seqlens:
            total_block_queries += (
                cls._sum_floor_div(
                    seq_len,
                    ratio,
                )
            )
        return (
            6
            * index_heads
            * index_head_dim
            * total_block_queries
            * num_qsa_layers
        )

    @staticmethod
    def _qsa_selected_tokens_sum(sequence_length, token_budget, compress_ratio):
        """
        Exact sum of selected token count over all causal queries.

        HF QSA selector uses:
            block_topk = token_budget // compress_ratio

            selected(t) =
                compress_ratio *
                min(floor(t / compress_ratio), block_topk)
                + t % compress_ratio

        The incomplete tail is always retained.
        """
        block_topk = token_budget // compress_ratio
        saturation_len = block_topk * compress_ratio

        total = 0

        # The prefix before TopK budget saturates is small for Qwen3.8
        # (2048), so this loop is negligible and keeps the logic exact.
        prefix_len = min(sequence_length, saturation_len)

        for t in range(1, prefix_len + 1):
            full_blocks = t // compress_ratio
            tail = t % compress_ratio
            total += (
                compress_ratio
                * min(full_blocks, block_topk)
                + tail
            )

        if sequence_length > saturation_len:
            remain = sequence_length - saturation_len

            # For t = saturation_len + 1 ... S:
            # selected = token_budget + (t % ratio)
            total += remain * token_budget

            start = saturation_len + 1
            end = sequence_length

            # Add sum(t % ratio), exact but still O(ratio) conceptually.
            # Use short loop because ratio=4 for Qwen3.8.
            for residue in range(compress_ratio):
                first = start + ((residue - start) % compress_ratio)
                if first <= end:
                    count = (end - first) // compress_ratio + 1
                    total += residue * count

        return total

    @classmethod
    def _compute_qsa_attention_flops(cls, config, seqlens, num_qsa_layers):
        """
        QSA main Attention QK^T + AV FLOPs. For each query, only selected KV tokens participate.

        Forward:
            QK: 2 * Hq * Dh * selected
            AV: 2 * Hq * Dh * selected
            => 4 * Hq * Dh * selected
        """
        num_heads = config.num_attention_heads
        head_dim = config.head_dim

        pair_count = 0
        for seq_len in seqlens:
            pair_count += cls._qsa_selected_tokens_sum(
                seq_len,
                config.indexer_budget,
                config.indexer_compress_ratio,
            )
        # MFU convention aligned with FlashAttention:
        # forward QK+AV: 4
        # backward: 8
        # QK recompute in backward: 2
        # total factor = 14
        return (
            14
            * pair_count
            * head_dim
            * num_heads
            * num_qsa_layers
        )


    @staticmethod
    def _compute_gdn_chunk_flops(config, tokens_sum, num_gdn_layers, chunk_size=64):
        """
        Compute FLOPs for the GatedDeltaNet chunked implementation across all GDN layers.
        Refer to `torch_chunk_gated_delta_rule` function in Qwen35 open source code.
        In training mode, GDN is implemented using the chunked method rather than recurrent method.
        Here are 8 types of matmul calculations involved in the chunked GDN:
            KKT: B * N * S * 2 * d_v * chunk_size
            value: B * N * S * 2 * d_v * chunk_size
            k_cumdecay: B * N * S * 2 * d_k * chunk_size
            chunk_attn: B * N * S * 2 * d_k * chunk_size
            chunk_v_prime: B * N * S * 2 * d_k  * d_v
            chunk_attn_inter: B * N * S * 2 * d_k * d_v
            chunk_core_attn_out: B * N * S * 2 * d_v * chunk_size
            chunk_last_recurrent_state: B * N * S * 2 * d_k  * d_v
        where B is batch_size, S is the sequence length, N = linear_num_value_heads,
        d_v = linear_value_head_dim, d_k = linear_key_head_dim.

        Following the same convention as quadratic attention (Q@K + attn@V):
            fwd: 2 * B * N * S * (3 * d_v * chunk_size + 2 * d_k * chunk_size + 3 * d_v * d_k)
            fwd + bwd (3x): 2 * 3 * B * N * S * (3 * d_v * chunk_size + 2 * d_k * chunk_size + 3 * d_v * d_k)

        """
        gdn_flops = (
            2
            * 3
            * tokens_sum
            * config.linear_num_value_heads
            * (
                config.linear_value_head_dim * chunk_size * 3 + \
                config.linear_key_head_dim * chunk_size * 2 + \
                config.linear_value_head_dim * config.linear_key_head_dim * 3
              )
            )
        return gdn_flops * num_gdn_layers

    @classmethod
    def _compute_hybrid_attn_flops(cls, config, seqlens, tokens_sum):
        """
        Compute parameter MAC weights of:
          - QSA full-attention projections
          - QSA indexer projection
          - GatedDeltaNet projections
          - GatedDeltaNet depthwise Conv1D

        The actual QSA QK/AV FLOPs and GDN recurrence FLOPs are NOT
        included here.

        QSA main attention:
          q_proj: H -> 2 * num_attention_heads * head_dim. The second half is the output gate.
          k_proj: H -> num_key_value_heads * head_dim
          v_proj: H -> num_key_value_heads * head_dim
          o_proj: num_attention_heads * head_dim -> H

        QSA indexer:
          index_qk_proj: H -> (indexer_n_heads + indexer_kv_heads) * indexer_head_dim

        GDN:
          in_proj_qkv: H -> 2 * key_size + value_size
          in_proj_z: H -> value_size
          in_proj_b: H -> num_value_heads
          in_proj_a: H -> num_value_heads
          out_proj: value_size -> H
          conv1d: depthwise conv over (2 * key_size + value_size) channels
        """
        hidden_size = config.hidden_size
        num_qsa_layers, num_gdn_layers = cls._get_layer_counts(config)

        # -------------------------
        # QSA
        # -------------------------
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.head_dim

        q_size = num_attention_heads * head_dim
        kv_size = num_key_value_heads * head_dim

        # q_proj outputs [query, gate], hence 2 * q_size.
        main_qsa_linear_N = (
            hidden_size * (2 * q_size)
            + hidden_size * kv_size
            + hidden_size * kv_size
            + q_size * hidden_size
        )

        index_qk_size = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim
        indexer_linear_N = hidden_size * index_qk_size
        qsa_linear_N = main_qsa_linear_N + indexer_linear_N
        qsa_proj_flops = 6 * tokens_sum * qsa_linear_N * num_qsa_layers

        # -------------------------
        # GatedDeltaNet
        # -------------------------
        linear_k_size = config.linear_num_key_heads * config.linear_key_head_dim
        linear_v_size = config.linear_num_value_heads * config.linear_value_head_dim
        # in_proj: qkv + z + b + a + out
        linear_attn_size = 2 * linear_k_size + 2 * linear_v_size + 2 * config.linear_num_value_heads + linear_v_size
        # depthwise conv1d: each of (2 * linear_k_size + linear_v_size) channels has its own kernel
        conv_N = config.linear_conv_kernel_dim * (2 * linear_k_size + linear_v_size)
        linear_attn_linear_N = (hidden_size * linear_attn_size + conv_N)
        gdn_proj_flops = 6 * tokens_sum * linear_attn_linear_N * num_gdn_layers

        return (
            qsa_proj_flops
            + gdn_proj_flops
            + cls._compute_gdn_chunk_flops(config, tokens_sum, num_gdn_layers)
            + cls._compute_qsa_indexer_flops(config, seqlens, num_qsa_layers)
            + cls._compute_qsa_attention_flops(config, seqlens, num_qsa_layers)
        )

    @staticmethod
    def _compute_moe_flops(config, tokens_sum):
        """
        Qwen4ExpSparseMoeBlock consists of:
          - top-k routed experts
          - one shared expert
          - router
          - shared_expert_gate

        Each SwiGLU expert:
          gate_proj: H -> I
          up_proj:   H -> I
          down_proj: I -> H

        Therefore:
          expert params = 3 * H * I
        """
        num_hidden_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        expert_intermediate_size = config.moe_intermediate_size
        shared_intermediate_size = config.shared_expert_intermediate_size
        num_experts = config.num_experts
        top_k = config.num_experts_per_tok
        routed_expert_active_N = top_k * 3 * hidden_size * expert_intermediate_size
        shared_expert_N = 3 * hidden_size * shared_intermediate_size
        router_N = hidden_size * num_experts
        # shared_expert_gate: Linear(H, 1, bias=False)
        shared_expert_gate_N = hidden_size

        moe_active_N = (
            routed_expert_active_N
            + shared_expert_N
            + router_N
            + shared_expert_gate_N
        )

        return 6 * tokens_sum * moe_active_N * num_hidden_layers

    @staticmethod
    def _compute_dense_mlp_flops(config, tokens_sum):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        num_hidden_layers = config.num_hidden_layers
        # dense MLP per layer: gate_proj + up_proj + down_proj (SwiGLU)
        return 6 * tokens_sum * hidden_size * 3 * intermediate_size * num_hidden_layers

    def _estimate_qwen3_8_family_flops(self, tokens_sum, batch_seqlens, **kargs):
        """
        Estimate Qwen3.8-Flash-Next training FLOPs.
        """
        # vit flops (Qwen3-VL ViT)
        images_seqlens = kargs.get("images_seqlens", None)
        if images_seqlens and sum(images_seqlens) > 0:
            vit_flops = self._estimate_qwen3_vit_flop(images_seqlens, self.config.vision_config)
        else:
            vit_flops = 0

        # llm flops
        text_config = self.config.text_config
        is_moe = hasattr(text_config, "num_experts")
        mlp_flops = self._compute_moe_flops(text_config, tokens_sum) if is_moe else self. _compute_dense_mlp_flops(text_config, tokens_sum)

        llm_flops = (
            self._compute_ple_flops(text_config, tokens_sum)
            + self._compute_hc_flops(text_config, tokens_sum)
            + self._compute_lm_head_flops(text_config, tokens_sum)
            + self._compute_hybrid_attn_flops(text_config, batch_seqlens, tokens_sum)
            + mlp_flops
        )

        return vit_flops + llm_flops

    def estimate_flops(self, batch_seqlens, step_time, **kwargs):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        estimated_flops = self._estimate_qwen3_8_family_flops(tokens_sum, batch_seqlens, **kwargs) / step_time
        return estimated_flops


def get_args():
    parser = argparse.ArgumentParser(description="Qwen3.8-Flash-Next FLOPs counter")
    parser.add_argument("--vit_seqlens", type=int, default=0, nargs="+", help="seqlen in vit")
    parser.add_argument("--llm_seqlens", type=int, default=16384, nargs="+", help="seqlen in language_model")
    parser.add_argument("--hf_path", type=str, default="/home/weights/Qwen3.8-Flash-Next/", help="HuggingFace config path")
    parser.add_argument("--device_num", type=int, default=1, help="Device num")
    parser.add_argument("--gbs", type=int, default=1, help="global batchsize")
    parser.add_argument("--step_time", type=float, help="Step time (s)")
    parser.add_argument('--hardware_flops', type=float, default=None, help='Hardware FLOPs')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    flopcounter = Qwen38NextFlopsCounter(config=AutoConfig.from_pretrained(args.hf_path))
    flops = flopcounter.estimate_flops(
        batch_seqlens=args.llm_seqlens, images_seqlens=args.vit_seqlens, step_time=args.step_time
    )
    flops = flops * args.gbs / args.device_num
    print(f"flops: {flops:.4e}")

    if args.hardware_flops is not None:
        if args.hardware_flops > 0:
            mfu = flops / args.hardware_flops
            print(f"MFU is: {mfu*100:.2f}%")
        else:
            raise ValueError("Hardware FLOPs must be a positive value.")

"""
e.g.:
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python mindspeed_mm/fsdp/tools/flops_tool/flops_qwen3_8_next.py \
    --vit_seqlens 1024 \
    --llm_seqlens 16384 \
    --hf_path /home/weights/Qwen3.8-Flash-Next/ \
    --device_num 1 \
    --gbs 1 \
    --step_time 6.9
"""
