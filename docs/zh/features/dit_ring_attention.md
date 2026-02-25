# DiT Ring Attention序列并行

## 问题分析

多模态模型的训练中，长序列训练的重要性正逐步体现。在生成性AI领域如视频生成任务中，都需要在空间和时间层面对长上下文进行推理。现有的数据、张量和流水线等并行方法无法在序列维度进行切分。当序列维度(S)增长时，训练内存开销会以 $O$($S^2$) 的速度增长。因此需要针对长序列场景进行特定的优化解决长训练场景的训练需求。

## 解决方案

支持Ring Attention长序列并行方案，以此解决序列维度扩展问题。具体细节参见原文：

> Ring Attention with Blockwise Transformers for Near-Infinite Context ([https://arxiv.org/pdf/2310.01889](https://arxiv.org/pdf/2310.01889))

支持Double Ring Attention算法，进一步加速原始Ring Attention实现。算法细节参见原文：

> LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism ([https://arxiv.org/pdf/2406.18485](https://arxiv.org/pdf/2406.18485))

## 使用方法

- 使用场景：视频分辨率/帧数设置的很大时，训练过程中，单卡无法完成DiT的计算，需要开启DiT-RingAttention
- 使能方式：在启动脚本 pretrain.sh 中修改如下变量

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
```

- ``--use-cp-send-recv-overlap``为可选参数，建议开启，开启后支持send receive overlap功能
- ``--cp-window-size [int]``为可选参数，设置算法中双层Ring Attention的内层窗口大小，需要确保cp_size能被该参数整除
  - 缺省值为1，即使用原始的Ring Attention算法
  - 大于1时，即使用Double Ring Attention算法，优化原始Ring Attention性能
- ``--megatron-cp-in-bnsd``为可选参数，建议开启，因默认 `fa_layout`为"sbh",开启后可支持[B, N, S, D]格式计算，可提高性能。
- `--attention-mask-type`设置attention计算时mask的类型，可选参数为 `general`和 `causal`，其中 `general`表示全attention，`causal`表示causal attention

## 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗，相比不开启序列并行单步耗时增加，相比重计算计算效率提升。

## 注意事项

1. 开启Context Parallel时需要同时开启Flash Attention特性，否则特性不支持。
2. 在8k的序列长度情况下，由于计算的时间缩短，cp功能分割之后的send receive的时间反而会长于计算时间，造成性能的下降，所以建议配置seq-length / context-parallel-size> 8k以获取最佳效果。具体公式参考：S/(Talpha) >= 1/(Wbeta)，其中，S=seq-length / context-parallel-size， T表示芯片的理论算力，alpha表示计算效率，W表示理论通信带宽，beta表示带宽利用率。
3. 内层窗口 `--cp-window-size`增大时，通信与计算并发程度更高，但是计算、通信并发时可能由于片上内存带宽抢占，整体效率下降，需要结合实际场景进行调试。
