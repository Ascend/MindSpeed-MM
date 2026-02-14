# CosyVoice3 使用指南

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
- [数据集准备及处理](#数据集准备及处理)
  - [数据集下载](#1-数据集下载)
  - [数据预处理](#2-数据预处理)
- [微调](#微调)
  - [准备工作](#1-准备工作)
  - [启动微调](#2-启动微调)
- [环境变量声明](#环境变量声明)
- [注意事项](#注意事项)

## 版本说明
### 参考实现
```
url=https://github.com/FunAudioLLM/CosyVoice
commit_id=1822c5c
```

### 变更记录

2026.02.14: 首次支持CosyVoice3模型训练

---
## 环境安装

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)，完成昇腾软件安装。

### 2. 环境搭建
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
pip install -r examples/fsdp2/cosyvoice3/requirements.txt
```

---

## 权重下载及转换

### 1. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)；

将下载的模型权重保存到本地目录下.

---
## 数据集准备及处理

### 1. 数据集下载
(以LibriTTS语料库中的train-clean-100数据集为例)

用户需要自行从[LibriTTS语料库](https://www.openslr.org/60/)下载[train-clean-100开源数据集](https://openslr.trmal.net/resources/60/train-clean-100.tar.gz)，并解压到项目目录下:

```
# 执行解压命令
tar -xzvf train-clean-100.tar.gz
```
### 2. 数据预处理
修改数据预处理脚本`examples\fsdp2\cosyvoice3\data_preprocess.sh`中的`data_dir`与`pretrained_model_dir`配置，运行数据预处理脚本：

```
# 修改路径，执行数据格式转换脚本，预处理后数据将默认保存在./data路径下
# data_dir: 修改为本地数据集路径，若本地路径为/data/LibriTTS，修改为/data
# pretrained_model_dir: 修改为本地下载的权重路径

bash examples\fsdp2\cosyvoice3\data_preprocess.sh
```

## 微调

### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载**、**数据集准备及处理**，详情可查看对应章节。

### 2. 启动微调

在 `examples\fsdp2\cosyvoice3\cosyvoice3_config.yaml` 文件中配置好数据集和权重路径:
1. 将所有`<local_path>`修改为本地权重所在目录
2. 将`DATASET_PATH`修改为数据预处理后保存的parquet路径，若使用train-clean-100数据集按照上述命令执行则无需修改

使用如下命令，启动CosyVoice3的微调任务：
```shell
bash examples\fsdp2\cosyvoice3\finetune_cosyvoice3.sh
```

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
## 注意事项

1. 若安装openai-whisper包时构建失败，可能为创建的临时构建环境下载了最新版本的setuptools包，导致构建失败，建议使用`--no-build-isolation`参数进行`pip install`命令。