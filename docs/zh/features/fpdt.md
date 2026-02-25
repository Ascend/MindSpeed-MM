# FPDT (fully pipelined distributed transformer) (Ulysses + offload)

## 问题分析
传统 Transformer 模型在多模态和长序列推理中存在流水线阻塞、通信延迟和计算资源浪费的问题，严重影响推理效率与吞吐。原生 Ulysses 模型在推理/预训练过程中仍存在诸多性能瓶颈，它的切分sequence的方式颗粒度较粗，导致存在较多的空泡现象，计算资源利用率不高。同时，其模块之间的同步依赖强，通信与计算无法有效重叠，容易引发阻塞。

## 解决方案
FPDT 通过引入更细粒度的sequence划分、计算通信掩盖机制，CPU-NPU间load/offload操作，缓解了性能和内存瓶颈。

## 解决思路
在原生Ulysses切分sequence逻辑的基础上再将切分后的sequence拆解成多个chunks，结合计算流与通信流并行调度，实现模块内并发，有效提升资源利用率。

## 使用方法
- 使用场景：视频分辨率/帧数设置的很大时，训练过程中，单卡进行计算过程会报OOM，需要开启FPDT
- 使能方式：在启动脚本pretrain_model.json中修改如下变量

  - FPDT (Ulysses Offload) 使能方式: --FPDT; 
  - chunk 数量使能方式: --FPDT-chunk-number; 
  - 开启offload特性: --FPDT-with-offload。

```json
...
"predictor":{
  ...
  ...
  "FPDT":true,
  "FPDT_chunk_number":4,
  "FPDT_with_offload":true

}
...
```
- 当开启CP > 1时，开启```FPDT```同时开启```FPDT_chunk_number```可使能FPDT
- 需要确保```FPDT_chunk_number```可以被per_gpu_seq_len数整除

## 使用效果
根据模型不同、参数量不同，效果各有差异，可以针对FPDT_chunk_number、FPDT_with_offload指标进行调优，均有收益。

## 鸣谢
1.GitHub项目tutorial：
https://www.deepspeed.ai/tutorials/ulysses-offload/

2.GitHub代码：
https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/fpdt_layer.py

