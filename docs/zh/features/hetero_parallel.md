# Hetero Parallel

## 技术背景

当前基于Megatron训练的主流范式（DP/PP/TP为主，SP/CP/EP为辅）适用于LLM这类同构模型（模型只有一个，规则化的模型结构排布，且每层或每几层就可看作一个重复的基本单元），多模态大模型（MLLM，Omni）这类异构模型一般由多个模态编码器、骨干网络、解码器组成，模型结构差异大。使用Megatron-LM训练多模态模型时，将MLLM视为一个整体，Encoder，LLM，Generator使用相同的DP、TP等分布式策略； Encoder和Generator作为额外的PP stage被整合进主干LLM，会导致出现2种问题：**模型异构、数据异构**。

- 模型异构：不同编码器（vision/audio encoder）、骨干（LLM）的计算量、模型大小不同，带来存算不均衡问题，导致计算空泡。具体表现为`LLM bound`和`encoder bound`现象。
- 数据异构：训练样本中不同模态数据（文本、语音、视觉）token数量差异大且动态变化（动态分辨率），从而导致不同编码器、骨干网络计算量差异，带来负载不均衡问题。具体有`Intra-microbatch`和`inter-microbatch`不均衡。

针对模型异构和数据异构，MindSpeed MM分别设计了hetero-parallel和(在线数据重排方案)[https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/features/online_data_rearrange.md]。


## hetero-parallel

### 方案介绍

hetero-parallel（异构并行）通过解耦多模态模型的并行方案配置，让每个子模块能够独立进行各自模块的并行配置，从而达到解决存算失衡的问题。区别于`dist-train`方案，hetero-parallel使用了编码器与骨干网络混合部署，解决独立部署导致的计算空泡和资源浪费。

对实现的简要描述如下：

- 在线`parallel_state`转换器：通过存储不同子模块运行配置的snapshot，运行时动态修改当前运行模块的mpu状态，从而实现并行配置的动态转换；
- 数据分发util：encoder-->LLM的数据流分发模块，实现数据流正确，通信掩盖实现中；
- 模型hook：在模型fw、bw前后挂载hook实现不同模块数据流的转化以及mpu状态的切换；
- 异构pp：实现异构pp的`forward_backward_func_list`调度。

推荐以下两种使用场景：

#### 异构DP/TP/CP

根据目前模型负载（QwenVL系列等），编码器通常规模较小，静态显存开销小，LLM部分模型大，静态开销大，针对这种场景，encoder可以开启DP/CP，LLM开启DP/TP/CP。例如Qwen2.5Omni 7B模型一个短序列场景的配置为 ViT、Audio: DP8，LLM: TP4DP2 可以获得最佳性能。

#### 异构PP

针对需要开启PP并行的场景（小mbs大gas，并且LLM模型参数量大），可以使用异构PP，ViT、Audio encoder使用大dp，LLM开启PP。支持encoder和LLM使用不同的mbs，推荐encoder使用大mbs（encoder mbs ~= 4-8 LLM mbs）以达到更好的性能。

### 使能方式


1. 训练启动脚本添加如下参数

```shell
GPT_ARGS="
    ...
    --hetero-parallel \
    --hetero-encoder-mbs-scale {num} \   # 将图像/音频编码器的mbs调整为文本解码器的num倍，提升计算效率g
"
```

2. 在对应`model.json`中需要异构并行的子模块添加`tp/pp/cp、mbs`等参数, 注意，骨干网络不再支持通过shell脚本来initial 并行策略，并且shell脚本中的并行策略都要设为1。

```txt
{
   ...
    "image_encoder": {
        "vision_encoder": {
            ...
            "tp":1,
            "pp":1,
            "cp":1
       },
    },
   "audio_encoder": {
        ...
        "tp":1,
        "pp":1,
        "cp":1
    },
    "text_decoder": {
        ...
        "tp":1,
        "pp":1,
        "cp":1
    },
   ...
}
```

```shell
TP=1
PP=1
CP=1
```

### 适用范围
当前仅支持Megatron后端使用pretrain_vlm训练的模型，FSDP2后端不支持。
