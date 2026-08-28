# Ulysses SP (Ulysses + RingAttention)

## Problem Analysis

In multimodal model training, the importance of long-sequence training is increasingly evident. In generative AI domains such as video generation, reasoning over long contexts in both spatial and temporal dimensions is required. Existing parallelism methods such as data, tensor, and pipeline parallelism cannot address the challenge of scaling along the sequence dimension.

## Solution

The Ulysses long-sequence parallelism solution is introduced to address the scaling problem of the sequence dimension.

### Implementation Logic

Ulysses partitions individual samples along the sequence dimension across the participating compute devices. Then, before the attention computation, it performs an all-to-all communication operation on the partitioned queries (Q), keys (K), and values (V), so that each compute device receives the complete sequence but only for a non-overlapping subset of the attention heads. This allows the participating compute devices to compute different attention heads in parallel. Finally, Ulysses uses another all-to-all operation to gather results across the attention heads while simultaneously re-partitioning along the sequence dimension.

## How to Enable

- Use Case: When the video resolution or frame count is set very high, a single card cannot complete the DiT computation during training, necessitating the enablement of DiT-RingAttention.

- Enabling Method: Modify the following variables in the startup script `pretrain.sh`.

```shell
CP=8

GPT_ARGS="
    --context-parallel-size ${CP} \
    --context-parallel-algo hybrid_cp_algo \
    --use-cp-send-recv-overlap \
    --ulysses-degree-in-cp [int] \
    --megatron-cp-in-bnsd \
    --attention-mask-type [str] \
...
"
```

- `--use-cp-send-recv-overlap` is an optional parameter. It is recommended to enable it, as doing so activates the send-receive overlap feature.
- Ensure that `--context-parallel-size` is divisible by `--ulysses-degree-in-cp` and greater than 1.
  - For example, when `--context-parallel-size` is set to 8, `--ulysses-degree-in-cp` can be set to 2 or 4.
  - Also, ensure that `--ulysses-degree-in-cp` is divisible by the number of attention heads.
- `--megatron-cp-in-bnsd` is an optional parameter. It is recommended to enable it. Since the default `fa_layout` is `sbh`, enabling it supports computation in the [B, N, S, D] format, which can improve performance.
- `--attention-mask-type` sets the type of mask used during attention computation. The available options are `general` and `causal`, where `general` indicates full attention, and `causal` indicates causal attention.

## Effectiveness

By partitioning the input sequence across multiple compute devices, memory consumption per device is reduced. Compared to not enabling sequence parallelism, per-step time increases; however, computational efficiency improves compared to recomputation.

## Acknowledgements

1. GitHub Repository:
<https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses>
