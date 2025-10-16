# CogVideoX (MindSpore后端) 使用指南

<p align="left">
</p>

## 目录
- [CogVideoX (MindSpore后端) 使用指南](#cogvideox-mindspore后端-使用指南)
  - [支持任务列表](#支持任务列表)
  - [环境安装](#环境安装)
      - [仓库拉取及环境搭建](#仓库拉取及环境搭建)
      - [Decord搭建](#decord搭建)
  - [权重下载及转换](#权重下载及转换)
      - [VAE下载](#vae下载)
      - [transformer文件下载](#transformer文件下载)
      - [T5模型下载](#t5模型下载)
      - [权重转换](#权重转换)
  - [数据集准备及处理](#数据集准备及处理)
  - [预训练](#预训练)
      - [准备工作](#准备工作)
      - [配置参数](#配置参数)
      - [启动预训练](#启动预训练)
  - [预训练模型扩参示例(15B)](#预训练模型扩参示例15b)
      - [模型参数修改](#模型参数修改)
      - [启动脚本修改](#启动脚本修改)
  - [环境变量声明](#环境变量声明)
---

## 支持任务列表
支持以下模型任务类型

|      模型      | 任务类型 | 任务列表 | 是否支持 |
|:------------:|:----:|:----:|:-----:|
| CogVideoX-5B | t2v  |预训练  | ✔ |
| CogVideoX-5B | i2v  |预训练  | ✔ |


## 环境安装

MindSpeed-MM MindSpore后端的依赖配套如下表，安装步骤参考[基础安装指导](../../../docs/mindspore/install_guide.md)。

| 依赖软件         |                                                              |
| ---------------- | ------------------------------------------------------------ |
| 昇腾NPU驱动固件  | [在研版本](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha) |
| 昇腾 CANN        | [在研版本](https://www.hiascend.com/zh/developer/download/community/result?module=cann) |
| MindSpore        | [2.7.0](https://www.mindspore.cn/install/)         |
| Python           | >=3.9                                                        |                                          |


### 仓库拉取及环境搭建

针对MindSpeed MindSpore后端，昇腾社区提供了一键转换工具MindSpeed-Core-MS，旨在帮助用户自动拉取相关代码仓并对torch代码进行一键适配，进而使用户无需再额外手动开发适配即可在华为MindSpore+CANN环境下一键拉起模型训练。在进行一键转换前，用户需要拉取相关的代码仓以及进行环境搭建：

```
# 创建conda环境
conda create -n test python=3.10
conda activate test

# 使用环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# 安装MindSpeed-Core-MS转换工具
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.4.0

# 使用MindSpeed-Core-MS内部脚本自动拉取相关代码仓并一键适配、提供配置环境
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm

# 替换MindSpeed中的文件
cd MindSpeed-MM
cp examples/cogvideox/dot_product_attention.py ../MindSpeed/mindspeed/core/transformer/dot_product_attention.py
mkdir ckpt
mkdir data
mkdir logs
```

---
### Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

---

## 权重下载及转换


### VAE下载

+ [VAE下载链接](https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1)


### transformer文件下载
+ [CogVideoX1.0-5B-t2v](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX1.0-5B-i2v](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)
+ [CogVideoX1.5-5B-t2v](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main/transformer_t2v)
+ [CogVideoX1.5-5B-i2v](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main/transformer_i2v)


### T5模型下载
仅需下载tokenizer和text_encoder目录的内容：[下载链接](https://huggingface.co/THUDM/CogVideoX-5b/tree/main)

预训练权重结构如下：
   ```
   CogVideoX-5B
   ├── text_encoder
   │   ├── config.json
   │   ├── model-00001-of-00002.safetensors
   │   ├── model-00002-of-00002.safetensors
   │   └── model.safetensors.index.json   
   ├── tokenizer
   │   ├── added_tokens.json
   │   ├── special_tokens_map.json
   │   ├── spiece.model
   │   └── tokenizer_config.json   
   ├── transformer
   │   ├── 1000 (or 1)
   │   │   └── mp_rank_00_model_states.pt
   │   └── latest
   └── vae
       └── 3d-vae.pt
   ```

### 权重转换
权重转换source_path参数请配置transformer权重文件的路径：
```bash
python examples/cogvideox/cogvideox_sat_convert_to_mm_ckpt.py \
    --source_path <your source path> \
    --target_path <target path> \
    --task t2v \
    --tp_size 1 \
    --pp_size 10 11 11 10 \
    --num_layers 42 \
    --mode split
```
其中--tp_size 后为实际的tp切分策略， --task 的值为t2v或i2v，
当开启PP时，--pp_size 后参数值个数与PP的数值相等，并且参数之和与--num_layers 参数相等，举例：当PP=4, --num_layers 4, --pp_size 1 1 1 1; 当PP=4, --num_layers 42, --pp_size 10 11 11 10 

转换后的权重结构如下：

TP=1,PP=1时：
```
CogVideoX-5B-Converted
├── release
│   └──mp_rank_00
│      └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```
TP=2,PP=1, TP>2的情况依此类推：
```
CogVideoX-5B-Converted
├── release
│   ├──mp_rank_00
│   │    └──model_optim_rng.pt
│   └──mp_rank_01
│      └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```
TP=1,PP=4, PP>1及TP>1的情况依此类推：
```
CogVideoX-5B-Converted
├── release
│   ├──mp_rank_00_000
│   │   └──model_optim_rng.pt
│   ├──mp_rank_00_001
│   │   └──model_optim_rng.pt
│   ├──mp_rank_00_002
│   │   └──model_optim_rng.pt
│   └──mp_rank_00_003
│       └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```

---
<a id="jump4"></a>
## 数据集准备及处理

数据集格式应该如下：
```
.
├── data.jsonl
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```
每个 txt 与视频同名，为视频的标签。视频与标签应该一一对应。

data.jsonl文件内容如下示例：
```
{"file": "dataPath/1.mp4", "captions": "Content from 1.txt"}
{...}
...
```

---

## 预训练


### 准备工作
配置脚本前需要完成前置准备工作，包括：**[环境安装](#环境安装)**、**[权重下载及转换](#权重下载及转换)**、**[数据集准备及处理](#数据集准备及处理)**，详情可查看对应章节。


### 配置参数

CogvideoX训练阶段的启动文件为shell脚本，主要分为如下4个：`
|版本         | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 |  pretrain_cogvideox_i2v.sh |pretrain_cogvideox_t2v.sh  |
| 1.5 | pretrain_cogvideox_i2v_1.5.sh |pretrain_cogvideox_t2v_1.5.sh |

模型参数的配置文件如下：
|版本         | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 |  model_cogvideox_i2v.json |model_cogvideox_t2v.json  |
| 1.5 | model_cogvideox_i2v_1.5.json |model_cogvideox_t2v_1.5.json |

以及涉及训练数据集的`data.json`文件

默认的配置已经经过测试，用户可按照自身环境修改如下内容：

1. 权重配置

  需根据实际任务情况在启动脚本文件（如`pretrain_cogvideox_i2v.sh`）中的`LOAD_PATH="your_converted_dit_ckpt_dir"`变量中添加转换后的权重的实际路径，如`LOAD_PATH="./CogVideoX-5B-Converted"`,其中`./CogVideoX-5B-Converted`为转换后的权重的实际路径，其文件夹内容结构如权重转换一节所示。`LOAD_PATH`变量中填写的完整路径一定要正确，填写错误的话会导致权重无法加载但运行并不会提示报错。

根据需要填写`SAVE_PATH`变量中的路径，用以保存训练后的权重。

2. 数据集路径配置

  根据实际情况修改`data.json`中的数据集路径，分别为`"data_path":"/your_data_path/data.jsonl"`、`"data_folder":"/your_data_path/"`，替换`"/your_data_path/"`为实际的数据集路径。

3. VAE及T5模型路径配置

  根据实际情况修改模型参数配置文件（如`model_cogvideox_i2v.json`）以及`data.json`文件中VAE及T5模型文件的实际路径。其中，T5文件的路径字段为`"from_pretrained": "5b-cogvideo/tokenizer"`及`"from_pretrained": "5b-cogvideo"`，替换`5b-cogvideo`为实际的路径；VAE模型文件的路径字段为`"from_pretrained": "3d-vae.pt"`，替换`3d-vae.pt`为实际的路径。

  当需要卸载VAE跟T5时，将模型参数配置文件中的`"load_video_features": false`及`"load_text_features": false`字段中的值分别改为`true`。将`data.json`中的`"use_feature_data"`字段的值改为`true`。

4. 切分策略配置

* 当PP开启时，在启动脚本文件中添加`--optimization-level 2 --use-multiparameter-pipeline-model-parallel`参数，并且在模型参数配置文件中的将`pipeline_num_layers`参数的值由`null`改为实际切分情况，例如PP=4，num_layers=42时，`"pipeline_num_layers":[11, 10, 11, 10]`，具体值根据实际的PP切分策略确定。

* 当开启VAE CP时，修改模型参数配置文件中的`ae`字典内的关键字`cp_size`的值为所需要的值，不兼容Encoder-DP、未验证与分层Zero效果。

* 当开启SP时，在启动脚本文件中添加`--sequence-parallel`参数。

* 当开启Encoder-DP时，需要将[model_cogvideox_i2v_1.5.json](i2v_1.5/model_cogvideox_i2v_1.5.json) 或者[model_cogvideox_t2v_1.5.json](t2v_1.5/model_cogvideox_t2v_1.5.json)中的`enable_encoder_dp`选项改为`true`。注意:需要在开启CP/TP，并且`load_video_features`为`false`及`load_text_features`为`false`才能启用，不兼容PP场景、VAE-CP、分层Zero。

* 当开启分层Zero时，需要在[pretrain_cogvideox_t2v_1.5.sh](t2v_1.5/pretrain_cogvideox_t2v_1.5.sh)或者[pretrain_cogvideox_i2v_1.5.sh](i2v_1.5/pretrain_cogvideox_i2v_1.5.sh)里面添加下面的参数。
  注意：不兼容Encoder-DP特性、TP场景、PP场景，与VAE-CP效果未验证。
  ```shell
  --layerzero \
  --layerzero-config ./zero_config.yaml \
  ```
  参数里面的yaml文件如下面所示:
  ```yaml
  zero3_size: 8  
  transformer_layers:
    - mindspeed_mm.models.predictor.dits.sat_dit.VideoDiTBlock
  backward_prefetch: 'BACKWARD_PRE'
  param_dtype: "bf16"
  reduce_dtype: "fp32"
  forward_prefetch: True
  limit_all_gathers: True
  ignored_modules:
    - ae
    - text_encoder
  ```
  
  该特性和TP不能兼容，开启时TP必须设置为1，使用该特性训练时，保存的权重需要使用下面的转换脚本进行后处理才能用于推理：
  
      ```bash
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      # your_mindspeed_path和your_megatron_path分别替换为之前通过MindSpeed-Core-MS一键脚本拉取的mindspeed和megatron仓库的具体路径。这两个路径通常位于MindSpeed-Core-MS目录的相应子目录中。
      export PYTHONPATH=$PYTHONPATH:<your_mindspeed_path>
      export PYTHONPATH=$PYTHONPATH:<your_megatron_path>
      # input_folder为layerzero训练保存权重的路径，output_folder为输出的megatron格式权重的路径
      python <your_mindspeed_path>/mindspeed/core/distributed/layerzero/state/scripts/convert_to_megatron.py --input_folder ./save_ckpt/hunyuanvideo/iter_000xxxx/ --output_folder ./save_ckpt/hunyuanvideo_megatron_ckpt/iter_000xxxx/ --prefix predictor
      ```

模型参数配置文件中的`head_dim`字段原模型默认配置为64。此字段调整为128会更加亲和昇腾。

在sh启动脚本中可以修改运行卡数(NNODES为节点数，GPUS_PER_NODE为每个节点的卡数，相乘即为总运行卡数)：
```shell
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1  
NODE_RANK=0  
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

### 启动预训练

t2v 1.0版本任务启动预训练
```shell
bash examples/mindspore/cogvideox/t2v_1.0/pretrain_cogvideox_t2v.sh
```
t2v 1.5版本任务启动预训练
```shell
bash examples/mindspore/cogvideox/t2v_1.5/pretrain_cogvideox_t2v_1.5.sh
```
i2v 1.0版本任务启动预训练
```shell
bash examples/mindspore/cogvideox/i2v_1.0/pretrain_cogvideox_i2v.sh
```
i2v 1.5版本任务启动预训练
```shell
bash examples/mindspore/cogvideox/i2v_1.5/pretrain_cogvideox_i2v_1.5.sh
```
---


## 预训练模型扩参示例(15B)


### 模型参数修改
通过增加扩散模型层数等配置可以模拟15B参数量，如下所示，修改模型参数配置文件（`model_cogvideox_i2v.json`）中`"predictor"`下的`"num_layers"`、`"num_heads"`和`"head_dim"`的值

```
"predictor": {
    "num_layers": 64,
    "num_heads": 32,
    "head_dim": 128,
    ...
}
```


### 启动脚本修改

修改GPT_ARGS参数如下，根据实际分辨率、帧数调整启动脚本中的分布式配置（单机16卡CP4效果较佳）：

```shell
GPUS_PER_NODE=16
MASTER_ADDR={主节点IP}
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

TP=4
PP=1
CP=4
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

GPT_ARGS="
    --num-attention-heads 64 \
    --seq-length 32 \
    --max-position-embeddings 32 \
    --recompute-num-layers 64 \
    ...
"
```

## 环境变量声明
ASCEND_RT_VISIBLE_DEVICES： 指定NPU设备的索引值

ASCEND_SLOG_PRINT_TO_STDOUT： 是否开启日志打印， 0：关闭日志打屏，1：开启日志打屏

ASCEND_GLOBAL_LOG_LEVEL： 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。
0：对应DEBUG级别，1：对应INFO级别，2：对应WARNING级别，3：对应ERROR级别，4：对应NULL级别，不输出日志

HCCL_CONNECT_TIMEOUT:  用于限制不同设备之间socket建链过程的超时等待时间，需要配置为整数，取值范围[120,7200]，默认值为120，单位s

HCCL_EXEC_TIMEOUT： 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步

ASCEND_LAUNCH_BLOCKING： 控制算子执行时是否启动同步模式，0：采用异步方式执行，1：强制算子采用同步模式运行

MS_DEV_HOST_BLOCKING_RUN：控制动态图算子是否单线程下发。0：多线程下发，1：单线程下发

MS_DEV_LAUNCH_BLOCKING：控制算子是否同步下发。0：异步下发，1：采用单线程下发且流同步

ACLNN_CACHE_LIMIT： 配置单算子执行API在Host侧缓存的算子信息条目个数

TOKENIZERS_PARALLELISM： 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为

NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量