# Online Data Rearrange

## 技术背景

多模态大模型的训练样本是一个由文本、图片、音频等多模态token交叉排列构成的序列，不同模态样本的token数量差异大且动态变化（动态分辨率），从而导致不同编码器、骨干网络计算量差异，带来负载不均衡问题。具体有`intra-microbatch`(DP间）和`inter-microbatch`（DP内）不均衡。

针对数据异构问题，MindSpeed MM分别设计了packing+在线数据重排方案。

## packing+在线数据重排

### 方案介绍

- 目标：LLM DP间计算量均衡（计算量定义为sub_seq ** 2和），
- 条件：最大序列长度max_seq_len为条件进行序列packing拼接

实现流程如下：

1. 数据集按照条件和目标进行数据组装;
2. dataloader数据读取后，按照encoder间DP负载均衡进行数据索引重排，
3. 根据索引位置使用all_to_all通信对数据进行重排；
4. encoder执行负载均衡计算；
5. 按照原始索引执行all_to_all通信，使embed数据按照LLM负载均衡；
6. LLM执行负载均衡计算。

### 使能方式

1. 训练启动脚本添加参数`--use-data-balance`，表示开启在线数据负载均衡。

```shell
GPT_ARGS="
    ...
    --use-data-balance \
"
```

注意：当前仅支持ViT负载均衡。