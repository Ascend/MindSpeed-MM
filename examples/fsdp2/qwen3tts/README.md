# Qwen3_TTS 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#jump1)
  - [环境准备](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
  - [数据转换](#jump3.2)
  - [数据提取](#jump3.3)
- [微调](#jump4)
  - [准备工作](#jump4.1)
  - [启动微调](#jump4.2)
- [环境变量声明](#jump10)
- [注意事项](#jump11)

## 版本说明
#### 参考实现
```
url=https://github.com/QwenLM/Qwen3-TTS
commit_id=8a98526
```

#### 变更记录

2026.01.29: 首次支持Qwen3-TTS模型

---
<a id="jump1"></a>
## 环境安装

<a id="jump1.1"></a>
#### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)，完成昇腾软件安装。

<a id="jump1.2"></a>
#### 2. 环境搭建
```bash
git clone --branch fsdp2_dev https://gitcode.com/Ascend/MindSpeed-MM.git

# 安装mindspeed及依赖
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
cp -r mindspeed ../MindSpeed-MM/

# 安装mindspeed mm及依赖
cd ../MindSpeed-MM
pip install -e .

# 安装其它依赖
pip install -r examples/fsdp2/qwen3tts/requirements.txt
```

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)；

将下载的模型权重保存到本地目录下.

---
<a id="jump3"></a>
## 数据集准备及处理

<a id="jump3.1"></a>
#### 1. 数据集下载 (以KAN-TTS数据集为例)

用户需要自行下载[达摩院语音KAN-TTS开源数据集](https://modelscope.cn/datasets/modelscope/DAMO.NLS.KAN-TTS.OpenDataset/files)，并解压到项目目录下:

```
# 执行解压命令
unzip -o opentts_data.zip -d opentts_data
```
<a id="jump3.2"></a>
#### 2. 数据转换 (以KAN-TTS数据集为例)
运行数据格式转换脚本`process_data.py`将数据处理成相应格式：

```
# 执行数据格式转换脚本
python examples/fsdp2/qwen3tts/process_data.py \
  --opentts_data_path opentts_data \
  --output_jsonl_path train_raw.jsonl \
  --ref_audio_path ref_audio.wav \

# 其中：
# opentts_data_path: 数据集路径
# output_jsonl_path: 转换后数据文件的保存路径
# ref_audio_path：参考音频文件路径，若参数不存在或参考音频不存在，则从数据集中随机选择一条数据作为参考音频
```
通过上述数据转换脚本将原始数据文件处理为JSONL格式的train_raw.jsonl文件，每行为一个JSON对象，包含以下字段：
| 字段      | 内容 |
| ----------- | -------------------------------------------------- |
| audio     | 目标训练音频文件路径，支持wav格式，频率仅支持24kHz |
| text      | 目标训练音频对应的文本内容                       |
| ref_audio | 参考音频文件路径，支持wav格式                    |

转换后示例如下：

```bash
{"audio":"/opentts_data/000001.wav","text":"有一回来个参观团，是县文物局组织的。","ref_audio":"/opentts_data/ref.wav"}
{"audio":"/opentts_data/000002.wav","text":"基础设施是产业发展的前提。","ref_audio":"/opentts_data/ref.wav"}
```

#### 3. 分词器下载

从Hugging Face库或ModelScope库下载对应的Tokenizer文件:

- Hugging Face下载地址: [Qwen3-TTS-Tokenizer-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz)；
- ModelScope下载地址：[Qwen3-TTS-Tokenizer-12Hz](https://modelscope.cn/models/Qwen/Qwen3-TTS-Tokenizer-12Hz)；

下载后将Tokenizer文件保存到本地目录下。

#### 4. 数据提取
运行数据提取脚本`prepare_data.py`将 `train_raw.jsonl` 转换为包含音频编码`audio_codes`的训练集JSONL文件`train_with_codes.jsonl `

```
# 设置环境变量
export NON_MEGATRON=true
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 执行数据提取脚本
python examples/fsdp2/qwen3tts/prepare_data.py \
  --device npu:0 \
  --tokenizer_model_path /Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl \

# 其中：
# device：数据提取的设备
# tokenizer_model_path：tokenizer文件路径
# input_jsonl：包含训练音频文件路径、对应文本内容和参考音频文件路径的JSONL文件路径
# output_jsonl：提取后数据文件的保存路径
```
处理后示例如下：

```bash
{"audio":"/opentts_data/000001.wav","text": "有一回来个参观团，是县文物局组织的。","ref_audio": "/opentts_data/ref.wav","audio_codes": [[1995, ..., 901]]}
{"audio":"/opentts_data/000002.wav","text": "基础设施是产业发展的前提。","ref_audio": "/opentts_data/ref.wav","audio_codes": [[1995, ..., 901]]}
```
---

<a id="jump4"></a>
## 微调

<a id="jump4.1"></a>
#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>
#### 2. 启动微调

在 `qwen3tts_config.yaml` 文件中配置好数据集和权重路径后，使用如下命令，即可实现Qwen3-TTS的微调：

```shell
bash examples/fsdp2/qwen3tts/finetune_qwen3tts.sh
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
| `NPU_ASD_ENABLE`   | 控制是否开启Ascend Extension for PyTorch的特征值检测功能 | 设置`0`或未设置: 关闭特征值检测<br>`1`: 表示开启特征值检测，只打印异常日志，不告警<br>`2`:开启特征值检测，并告警<br>`3`:开启特征值检测，并告警，同时会在device侧info级别日志中记录过程数据 |
| `ASCEND_LAUNCH_BLOCKING`   | 控制算子执行时是否启动同步模式 | `0`: 采用异步方式执行<br>`1`: 强制算子采用同步模式运行                                                               |
| `NPUS_PER_NODE`               | 配置一个计算节点上使用的NPU数量                                                  | 整数值（如 `1`, `8` 等）                                                                            |

---
<a id="jump11"></a>
## 注意事项