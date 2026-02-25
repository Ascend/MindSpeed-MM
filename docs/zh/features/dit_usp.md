# Ulysses SP混合序列并行（Ulysses + RingAttention）

## 问题分析

多模态模型的训练中，长序列训练的重要性正逐步体现。在生成性AI领域如视频生成任务中，都需要在空间和时间层面对长上下文进行推理。现有的数据、张量和流水线等并行方法无法解决序列维度的扩展问题。

## 解决方案

支持 Ulysses长序列并行方案，以此解决序列维度扩展问题。

### 解决思路

Ulysses 将各个样本在序列维度上分割给参与的计算设备。然后，在 attention 计算之前，它对已分割的查询(Q)、键(K)和值(V)执行 all-to-all 通信操作，以使每个计算设备接收完整的序列，但仅用于注意力头的非重叠子集。这使得参与的计算设备可以并行计算不同的注意力头。最后，Ulysses 还使用另一个 all-to-all 来在注意力头上收集结果，同时重新在序列维度上进行分区。

## 使用方法

- 使用场景：视频分辨率/帧数设置的很大时，训练过程中，单卡无法完成DiT的计算，需要开启DiT-RingAttention

- 使能方式：在启动脚本pretrain.sh中修改如下变量

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
```

- ```--use-cp-send-recv-overlap```为可选参数，建议开启，开启后支持send receive overlap功能
- 需要确保```--context-parallel-size```可以被```--ulysses-degree-in-cp```整除且大于1
  - 例如当设置```--context-parallel-size```为8时，可以设置```--ulysses-degree-in-cp```为2或```--ulysses-degree-in-cp```为4
  - 同时需要确保```--ulysses-degree-in-cp```可以被num-attention-heads数整除
- ```--megatron-cp-in-bnsd```为可选参数，建议开启，因默认`fa_layout`为"sbh",开启后可支持[B, N, S, D]格式计算，可提高性能。
- `--attention-mask-type`设置attention计算时mask的类型，可选参数为`general`和`causal`，其中`general`表示全attention，`causal`表示causal attention

## 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗，相比不开启序列并行单步耗时增加，相比重计算计算效率提升。

## 鸣谢

1.GitHub项目地址：
<https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses>
