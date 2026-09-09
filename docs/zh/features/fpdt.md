# FPDT（Fully Pipelined Distributed Transformer，Ulysses + Offload）

## 适用后端

MCORE

## 问题分析

传统Transformer模型在多模态和长序列推理中存在流水线阻塞、通信延迟和计算资源浪费的问题，严重影响推理效率与吞吐。原生Ulysses模型在推理/预训练过程中仍存在诸多性能瓶颈，它的切分sequence的方式颗粒度较粗，导致存在较多的空泡现象，计算资源利用率不高。同时，其模块之间的同步依赖强，通信与计算无法有效重叠，容易引发阻塞。

## 解决方案

FPDT在原生Ulysses切分sequence逻辑的基础上，将切分后的sequence进一步拆解成多个chunks，结合计算流与通信流并行调度实现模块内并发，并通过CPU-NPU间load/offload操作，缓解性能和内存瓶颈，有效提升资源利用率。效果因模型、参数量不同而各有差异，可针对 `FPDT_chunk_number`、`FPDT_with_offload` 指标进行调优，均有收益。

## 使用方法

使用场景：视频分辨率/帧数设置得很大时，训练过程中CP切分下单卡显存不足（OOM），需要开启FPDT。当前仅wan2.1的1.3B T2V示例（`WanDiT`）提供配置。

使能方式：开启CP > 1，在启动脚本pretrain_model.json的 `predictor` 段中修改如下JSON配置键（前提：`context_parallel_algo` 需配置为 `ulysses_cp_algo` 或 `hybrid_cp_algo`，且attention类型为self-attention）：

```json
"predictor": {
  "FPDT": true,
  "FPDT_chunk_number": 4,
  "FPDT_with_offload": true
}
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `FPDT` | bool | `false` | 使能键 |
| `FPDT_chunk_number` | int | `None` | chunk数量，需确保per_gpu_seq_len可被其整除；开启FPDT时必填，缺省（`None`）会在FPDTFlashAttention初始化时报错 |
| `FPDT_with_offload` | bool | `false` | 是否开启offload特性 |

### 示例脚本

**示例脚本**：`examples/wan2.1/1.3b/t2v/pretrain.sh`（特性配置键见 `examples/wan2.1/1.3b/t2v/pretrain_model.json` 的 `predictor` 段；示例中 `FPDT` 默认 `false`，需置为 `true`）。

## 注意事项

### 已知约束

- FPDT的前提（CP > 1、`context_parallel_algo` 为 `ulysses_cp_algo`/`hybrid_cp_algo`、attention类型为self-attention）在代码中无强制校验；不满足时attention不启用FPDT，但embs/rope切分与FFN分块仍会生效，请注意避免此种半生效配置。
- per_gpu_seq_len可被 `FPDT_chunk_number` 整除的校验在attention前向运行时进行，配置期不检查。

## 鸣谢

1. GitHub项目tutorial：<https://www.deepspeed.ai/tutorials/ulysses-offload/>
2. GitHub代码：<https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/fpdt_layer.py>
