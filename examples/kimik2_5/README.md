# Kimi-K2.5 / K2.6 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#环境安装)
  - [环境准备](#1-环境准备)
  - [环境搭建](#2-环境搭建)
- [数据集准备及处理](#数据集准备及处理)
- [训练](#训练)
  - [准备工作](#1-准备工作)
  - [配置参数](#2-配置参数)
  - [启动训练](#3-启动训练)
- [环境变量声明](#环境变量声明)
- [注意事项](#注意事项)

## 版本说明

### 参考实现

**Kimi-K2.5**

```shell
url=https://huggingface.co/moonshotai/Kimi-K2.5/tree/main
commit_id=3367c8d
```

**Kimi-K2.6**

```shell
url=https://huggingface.co/moonshotai/Kimi-K2.6/tree/main
commit_id=7eb5002
```

### 变更记录

2026.02.13: 首次支持Kimi-K2.5模型

2026.06.05: 首次支持Kimi-K2.6模型

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/tree/master/docs/zh/pytorch/install_guide.md)，完成昇腾软件安装。

‼️ 部分特性依赖较新版本的CANN，请使用 8.5.0 以上版本:

- [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler)

<a id="jump1.2"></a>

### 2. 环境搭建

执行如下指令一键安装：

```bash
bash scripts/install.sh && bash pip install tiktoken==0.12.0
```

---

<a id="jump2"></a>

## 数据集准备及处理

- 使用**真实数据集**训练：参考[针对VL模型的数据构造 · 使用真实数据集](../../docs/zh/features/building_data_for_VLModel.md#real-data)（下载COCO2017 → 下载LLaVA-Instruct-150K标注 → 运行转换脚本生成`mllm_format_llava_instruct_data.json`）。
- 使用**虚构数据**做功能/性能测试：参考[针对VL模型的数据构造 · 使用虚构数据](../../docs/zh/features/building_data_for_VLModel.md#mock-data)。

## 训练

<a id="jump3.1"></a>

### 1. 准备工作

从Huggingface库下载下列文件并放置于本地`mindspeed_mm/fsdp/models/kimik2_5`路径下：

- Kimi-K2.5: [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5/tree/main)
- Kimi-K2.6: [Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6/tree/main)

> **说明**：Kimi-K2.6 与 Kimi-K2.5 共用相同的模型代码目录 `mindspeed_mm/fsdp/models/kimik2_5`，仅权重文件不同。

以 Kimi-K2.5 为例：

```shell
# HF_PATH配置为HuggingFace库下载文件的存放路径
HF_PATH="/download/Kimi-K2.5"
# MM_PATH配置为MindSpeed-MM根目录路径
MM_PATH="/home/workspace/MindSpeed-MM"

cd ${HF_PATH}
cp -f \
  chat_template.jinja \
  config.json \
  configuration_deepseek.py \
  configuration_kimi_k25.py \
  generation_config.json \
  kimi_k25_processor.py \
  kimi_k25_vision_processing.py \
  preprocessor_config.json \
  tiktoken.model \
  tokenizer_config.json \
  tool_declaration_ts.py \
  ${MM_PATH}/mindspeed_mm/fsdp/models/kimik2_5/
cd ${MM_PATH}
```

Kimi-K2.5/K2.6 模型需要配置多机训练，如需拉起多机训练，请修改启动脚本下的 `MASTER_ADDR`、`NNODES` 以及 `NODE_RANK` 变量：

``` shell
MASTER_ADDR: 主节点IP地址
NNODES: 总节点数量
NODE_RANK: 当前节点序号
```

配置脚本前需要完成前置准备工作，包括：**环境安装**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump3.2"></a>

### 2. 配置参数

以下配置项在 `kimik2_5_config.yaml` 中设置：

| 配置项 | 配置路径 | 参数说明 | 调整说明 |
|--------|----------|----------|----------|
| `ulysses_parallel_size` | `parallel` | ulysses-cp 并行度 | 值为1时不开启，根据实际情况调整 |
| `num_to_forward_prefetch` | `parallel->fsdp_plan` | 前向计算时预取后续层参数 | 减少通信等待开销 |
| `num_to_backward_prefetch` | `parallel->fsdp_plan` | 反向计算时预取后续层参数 | 减少通信等待开销 |
| `enable_preload` | `data->dataloader_param` | 数据预加载开关 | 开启后数据加载与计算重叠，减少训练等待时间 |
| `enable_activation_offload` | `features` | 激活值卸载到Host侧内存开关 | 开启后降低Device显存占用，`apply_modules`指定需要开启该特性的module |
| `enable_chunk_mbs` | `features` | 是否开启chunkmbs特性 | 需与`chunkmbs_plan`关联使用，开启后将MicroBatch维度切分为多个微块依次计算，可压缩激活显存峰值并提升训练吞吐 |
| `chunkmbs_plan` | `features` | chunkmbs切分策略配置 | 仅在`enable_chunk_mbs`启用时生效，包含`chunk_mbs`、`batch_dim`、`chunk_arg_indexs`、`chunk_kwarg_names`等子字段，详细说明请参考[chunkmbs文档](../../docs/zh/features/chunkmbs.md) |

<a id="jump3.3"></a>

### 3. 启动训练

(1) 修改 `kimik2_5_config.yaml` 中 `data->dataset_param->basic_parameters->dataset` 字段，配置实际的数据集路径；

(2) 启动训练：

```shell
bash examples/kimik2_5/finetune_kimik2_5.sh
```

<a id="jump4"></a>

## 环境变量声明

| 环境变量                      | 描述                                                                 | 取值说明                                                                                         |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                                    | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                              |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                    | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                                     |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                                  | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                          | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征                                            |
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用                                                               |

---

<a id="jump5"></a>

## 注意事项
