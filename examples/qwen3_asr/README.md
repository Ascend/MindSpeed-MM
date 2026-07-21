# Qwen3-ASR-1.7B 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#环境安装)
  - [环境准备](#1-环境准备)
  - [环境搭建](#2-环境搭建)
- [权重下载及转换](#权重下载及转换)
  - [权重下载](#1-权重下载)
  - [权重转换](#2-权重转换)
- [数据集准备及处理](#数据集准备及处理)
  - [数据集下载](#1-数据集下载)
  - [数据集处理](#2-数据集处理)
- [微调](#微调)
  - [准备工作](#1-准备工作)
  - [配置参数](#2-配置参数)
  - [启动微调](#3-启动微调)
- [环境变量声明](#环境变量声明)

## 版本说明

### 参考实现

```shell
1. Qwen3-ASR 官方仓库: https://github.com/QwenLM/Qwen3-ASR
   - 参考了音频编码器、文本解码器、交叉注意力等组件的实现
   - 使用了官方的数据处理和特征提取逻辑

2. MindSpeed-MM 示例
   - 参考了其他模型（如 Whisper、Qwen2-VL）的训练脚本结构
   - 遵循 MindSpeed-MM 的模型注册和数据加载规范
```

### 变更记录

2026.4.30: 首次支持Qwen3-ASR模型

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](../../docs/zh/pytorch/install_guide.md)，完成昇腾软件安装。

<a id="jump1.2"></a>

### 2. 环境搭建

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git

# 安装mindspeed及依赖
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
cp -r mindspeed ../MindSpeed-MM/

# 安装mindspeed mm及依赖
cd ../MindSpeed-MM
pip install -e .

# 安装音频处理依赖
pip install librosa==0.11.0 soundfile==0.13.1
```

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

### 1. 权重下载

从Hugging Face库下载对应的模型权重:

- 模型地址: [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)；

将下载的模型权重保存到本地的`ckpt/hf_path/Qwen3-ASR-1.7B`目录下。

<a id="jump2.2"></a>

### 2. 权重转换

```bash
mm-convert GenericDCPConverter hf_to_dcp \
  --hf_dir ./ckpt/hf_path/Qwen3-ASR-1.7B \
  --dcp_dir ./ckpt/mm_path/Qwen3-ASR-1.7B-dcp-tie-noprefix \
  --tie_weight_mapping '{"thinker.lm_head.weight":"thinker.model.embed_tokens.weight"}'

# 转换后的目录结构为：
# ———— Qwen3-ASR-1.7B-dcp-tie-noprefix
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

并在`qwen3_asr_1.7B_config.yaml`中将`init_model_with_meta_device`参数配置为`true`，同时将`load`参数配置为转换后的DCP权重路径（写到`release`文件夹的上一级目录）。

---
<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

### 1. 数据集下载

用户可以自行下载[Common Voice Scripted Speech 25.0 - Chinese (China)](https://mozilladatacollective.com/datasets/cmn3iaztg00e4mb070uvufz7q)，并解压到项目目录下的`./data/cv-corpus-25.0-2026-03-09/`文件夹中。

<a id="jump3.2"></a>

### 2. 数据集处理

Qwen3-ASR示例使用json/jsonl数据。每条样本至少需要包含音频路径和目标文本，音频路径可以是绝对路径，也可以是相对于`dataset_dir`的相对路径。

推荐的简洁格式如下：

```json
{
  "audio": "audio/demo_zh.wav",
  "prompt": "请将音频转写为中文。",
  "text": "欢迎使用通义千问语音识别模型。"
}
```

也可以使用`messages`字段组织system/user/assistant信息，数据加载器会从assistant消息中提取目标文本：

```json
{
  "audio": "audio/demo_en.wav",
  "messages": [
    {"role": "system", "content": "Transcribe the speech in the audio."},
    {"role": "user", "content": "<audio>"},
    {"role": "assistant", "content": "Welcome to Qwen speech recognition."}
  ]
}
```

## 微调

<a id="jump4.1"></a>

### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

### 2. 配置参数

根据实际情况修改`qwen3_asr_1.7B_config.yaml`中的数据集和权重路径，包括`model_name_or_path`、`dataset_dir`、`dataset`、`load`和`save`等字段。其中`load`需要配置为权重转换后生成的DCP目录，例如`./ckpt/mm_path/Qwen3-ASR-1.7B-dcp-tie-noprefix`。

### 3. 启动微调

在`qwen3_asr_1.7B_config.yaml`文件中配置好数据集和权重路径后，使用如下命令，即可实现Qwen3-ASR的微调：

```shell
bash examples/qwen3_asr/finetune_qwen3_asr_1.7B.sh
```

<a id="jump10"></a>

## 环境变量声明

| 环境变量                      | 描述                                                                 | 取值说明                                                                                         |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 是否开启日志打印                                                           | `0`: 关闭日志打屏<br>`1`: 开启日志打屏                                                                   |
| `ASCEND_GLOBAL_LOG_LEVEL`     | 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志                             | `0`: 对应DEBUG级别<br>`1`: 对应INFO级别<br>`2`: 对应WARNING级别<br>`3`: 对应ERROR级别<br>`4`: 对应NULL级别，不输出日志 |
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                                    | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                              |
| `COMBINED_ENABLE`             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景 | `0`: 关闭<br>`1`: 开启                                                                           |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                    | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                                     |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                                  | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                          | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征                                            |
| `HCCL_EXEC_TIMEOUT`           | 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步         | 需要配置为整数，取值范围`[68,17340]`，默认值为`1800`，单位`s`                                                    |
| `ACLNN_CACHE_LIMIT`           | 配置单算子执行API在Host侧缓存的算子信息条目个数                                  | 需要配置为整数，取值范围`[1, 10,000,000]`，默认值为`10000`                                                    |
| `TOKENIZERS_PARALLELISM`      | 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为    | `False`: 禁用并行分词<br>`True`: 开启并行分词                                                            |
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用                                                               |
| `NPU_ASD_ENABLE`   | 控制是否开启TorchNPU的特征值检测功能 | 设置`0`或未设置: 关闭特征值检测<br>`1`: 表示开启特征值检测，只打印异常日志，不告警<br>`2`:开启特征值检测，并告警<br>`3`:开启特征值检测，并告警，同时会在device侧info级别日志中记录过程数据 |
| `ASCEND_LAUNCH_BLOCKING`   | 控制算子执行时是否启动同步模式 | `0`: 采用异步方式执行<br>`1`: 强制算子采用同步模式运行                                                               |
| `NPUS_PER_NODE`               | 配置一个计算节点上使用的NPU数量                                                  | 整数值（如 `1`, `8` 等）                                                                            |

---
