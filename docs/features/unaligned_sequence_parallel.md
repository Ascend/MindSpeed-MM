# Qwen2VL-72B支持非均匀Sequence Parallel切分 

## 问题分析

SP（Sequence Parallel）并行算法是一种针对长序列数据处理的并行化技术，在处理长序列时具有显著优势。多模态模型存在大量序列长度非均匀场景，需要进行相应的适配。

## 解决方案

Sequence Parallel主要作用与TransformerLayer中的Dropout和LayerNorm模块，在序列维度对数据进行非均匀切分。
![alt text](../../sources/images/sp.png)


## 使用方法
(当前仅支持qwen2vl)
1. examples/qwen2vl/finetune_qwen2vl_72b.sh中开启`TP`并在GPT_ARGS中添加如下参数  
```shell
    --sequence-parallel
    # add only if unaligned SP is required
    --unaligned-linear
```