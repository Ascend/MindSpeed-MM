# DiT Ring Attention Sequence Parallelism

## Problem Analysis

In multimodal model training, the importance of long-sequence training is increasingly evident. In generative AI domains such as video generation, reasoning over long contexts is required in both spatial and temporal dimensions. Existing parallelism methods like data, tensor, and pipeline parallelism cannot partition along the sequence dimension. As the sequence length (S) grows, the training memory overhead increases at a rate of $O$($S^2$). Therefore, dedicated optimizations for long-sequence scenarios are necessary to meet the training demands of long-sequence tasks.

## Solution

The Ring Attention long-sequence parallelism scheme is introduced to address the challenge of scaling the sequence dimension. For specific details, refer to [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/pdf/2310.01889).

The Double Ring Attention algorithm is also supported, further accelerating the original Ring Attention implementation. For algorithm details, refer to [LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism](https://arxiv.org/pdf/2406.18485).

## How to Enable

- Use case: When the video resolution/number of frames is set very large, a single card cannot complete the DiT computation during training, and DiT-RingAttention needs to be enabled.
- Enabling method: Modify the following variables in the startup script `pretrain.sh`.

```shell
CP=8

GPT_ARGS="
    --context-parallel-size ${CP} \
    --context-parallel-algo megatron_cp_algo \
    --use-cp-send-recv-overlap \
    --cp-window-size [int] \
    --megatron-cp-in-bnsd \
    --attention-mask-type [str] \
...
"
```

- ``--use-cp-send-recv-overlap`` is an optional parameter. It is recommended to enable it, as it activates the send/receive overlap feature.
- ``--cp-window-size [int]`` is an optional parameter that sets the inner window size of the double-layer Ring Attention algorithm. Ensure that `cp_size` is divisible by this parameter.
  - The default value is 1, which means the original Ring Attention algorithm is used.
  - A value greater than 1 enables the Double Ring Attention algorithm, optimizing the performance of the original Ring Attention.
- ``--megatron-cp-in-bnsd`` is an optional parameter, and it is recommended to enable it. Since the default `fa_layout` is `sbh`, enabling this supports computation in the [B, N, S, D] format, which can improve performance.
- `--attention-mask-type` sets the type of mask used during attention computation. Optional values are `general` and `causal`, where `general` represents full attention and `causal` represents causal attention.

## Effectiveness

By partitioning the input sequence across multiple compute devices, memory consumption per device is reduced. Compared to not enabling sequence parallelism, per-step time increases; however, computational efficiency improves compared to recomputation.

## Notes

1. Enabling Context Parallelism (CP) requires enabling Flash Attention simultaneously; otherwise, the feature is not supported.
2. At a sequence length of 8k, the shortened computation time may cause the send/receive time from CP partitioning to exceed the computation time, leading to performance degradation. For optimal results, it is recommended to configure `seq-length/context-parallel-size > 8k` for optimal results. The general formula is: `S/(Talpha) >= 1/(Wbeta)`, where `S = seq-length/ context-parallel-size`, `T` represents the theoretical computing power of the chip, `alpha` represents computational efficiency, `W` represents theoretical communication bandwidth, and `beta` represents bandwidth utilization.
3. When the inner window (`--cp-window-size`) is increased, the degree of concurrency between communication and computation improves. However, due to potential on-chip memory bandwidth contention, overall efficiency may decrease. Tuning based on the specific deployment scenario is required.
