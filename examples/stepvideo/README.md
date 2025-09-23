# StepVideo 使用指南

<p align="left">
</p>

## 目录
- [StepVideo 使用指南](#stepvideo-使用指南)
  - [目录](#目录)
  - [- 环境变量声明](#--环境变量声明)
  - [支持任务列表](#支持任务列表)
  - [环境安装](#环境安装)
      - [仓库拉取](#仓库拉取)
      - [环境搭建](#环境搭建)
      - [Decord搭建](#decord搭建)
  - [权重下载及转换](#权重下载及转换)
      - [权重下载](#权重下载)
      - [权重转换](#权重转换)
  - [数据集准备及处理](#数据集准备及处理)
  - [预训练](#预训练)
      - [准备工作](#准备工作)
      - [特征提取](#特征提取)
      - [配置参数](#配置参数)
      - [启动预训练](#启动预训练)
  - [推理](#推理)
      - [准备工作](#准备工作-1)
      - [配置参数](#配置参数-1)
      - [启动推理](#启动推理)
  - [Dpo训练](#dpo训练)
      - [环境准备](#环境准备)
      - [生成视频样本](#生成视频样本)
      - [生成偏好数据集](#生成偏好数据集)
      - [训练参数配置](#训练参数配置)
      - [启动dpo训练](#启动dpo训练)
  - [环境变量声明](#环境变量声明)
---

## 支持任务列表
支持以下模型任务类型

|    模型    | 任务类型 | 任务列表 | 是否支持 |
|:---------:|:----:|:----:|:-----:|
| StepVideo | t2v  |预训练  | ✔ |
| StepVideo | t2v  |在线推理 | ✔ |
| StepVideo | i2v  |预训练  | ✔ |
| StepVideo | i2v  |在线推理 | ✔ |

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

#### 仓库拉取

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
```

#### 环境搭建

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

# apex for Ascend 参考 https://gitcode.com/Ascend/apex
# 建议从原仓编译安装 

# 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 安装加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
pip install -r requirements.txt 
pip install -e .
cd ..

# 安装其余依赖库
pip install -e .
```

#### Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

---

## 权重下载及转换

#### 权重下载
StepVideo t2v权重下载(需要下载 VAE、transformer、text_encoder、tokenizer)

[StepVideo-t2v下载链接](https://hf-mirror.com/stepfun-ai/stepvideo-t2v/tree/main)

StepVideo i2v权重下载(需要下载 VAE、transformer、text_encoder、tokenizer)

[StepVideo-i2v下载链接](https://hf-mirror.com/stepfun-ai/stepvideo-ti2v/tree/main)

预训练/推理权重结构如下：
```
   stepvideo-ti2v/t2v
   ├── hunyuan_clip
   │   ├── clip_text_encoder
   │   │   ├── config.json
   │   │   └── pytorch_model.bin
   │   ├── tokenizer
   │   │   ├── special_tokens_map.json
   │   │   ├── tokenizer_config.json
   │   │   ├── vocab.txt
   │   │   └── vocab_org.txt
   ├── step_llm
   │   ├── config.json
   │   ├── model-00001-of-00009.safetensors
   │   ├── model-00002-of-00009.safetensors
   │   ├── ...
   │   ├── model-00009-of-00009.safetensors
   │   ├── model.safetensors.index.json
   │   └── step1_chat_tokenizer.model 
   ├── transformer
   │   ├── config.json
   │   ├── diffusion_pytorch_model-00001-of-00006.safetensors
   │   ├── diffusion_pytorch_model-00002-of-00006.safetensors
   │   ├── ...
   │   ├── diffusion_pytorch_model-00006-of-00006.safetensors
   │   └── diffusion_pytorch_model.safetensors.index.json
   └── vae
       ├── vae.safetensors
       └── vae_v2.safetensors
```

#### 权重转换
权重转换source_path参数请配置transformer权重文件的路径：
```bash
mm-convert StepVideoConverter hf_to_mm \
  --cfg.source_path <your source path> \
  --cfg.target_path <your target path> \
  --cfg.target_parallel_config.tp_size <tp_size> \
  --cfg.target_parallel_config.pp_layers <pp_layers> \
```
其中`tp_size`表示TP切分数，`pp_layers`表示pp切分后每个stage的层数，如`48`表示不切分，`[24,24]`表示PP=2，每个PP stage 24层。

转换后的权重结构如下：

TP=1,PP=1时：
```
StepVideo-Converted
├── release
│   └──mp_rank_00
│      └──model_optim_rng.pt
└──latest_ckeckpointed_iterations.txt
```
TP=2,PP=1, TP>2的情况依此类推：
```
StepVideo-Converted
├── release
│   ├──mp_rank_00
│   │    └──model_optim_rng.pt
│   └──mp_rank_01
│      └──model_optim_rng.pt
└──latest_ckeckpointed_iterations.txt
```

---

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

#### 准备工作
配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

#### 特征提取

1.配置特征提取参数

检查模型权重路径、数据集路径、提取后的特征保存路径等配置是否完成

| t2v配置文件                                                     |       修改字段        | 修改说明                                            |
| ------------------------------------------------------------ | :-------------------: | :-------------------------------------------------- |
| examples/stepvideo/feature_extract/data.json              |      basic_parameters   | 数据集路径，`data_path`和`data_folder`分别配置data.jsonl的文件路径和目录 |
| examples/stepvideo/feature_extract/data.json              |      num_frames        | 最大的帧数，超过则随机选取其中的num_frames帧, 其中i2v配置102,t2v配置136 |
| examples/stepvideo/feature_extract/data.json              |      tokenizer_config  | tokenizer分词器选择，配置两种分词器的路径`"from_pretrained": "/model_path/step_llm/step1_chat_tokenizer.model"` 及`"from_pretrained": "/model_path/hunyuan_clip/tokenizer"` |
| examples/stepvideo/feature_extract/model_stepvideo.json   |      text_encoder    | 配置两种文本编译器路径`"from_pretrained": "./weights/step_llm/"`及`"from_pretrained": "./weights/hunyuan_clip/clip_text_encoder"` |
| examples/stepvideo/feature_extract/model_stepvideo.json   |      ae              | 配置VAE模型路径`"from_pretrained": "./weights/vae/vae_v2.safetensors"` |
| mindspeed_mm/tools/tools.json                             |      save_path       | 提取后的特征保存路径                                |

| i2v配置文件                                                     |       修改字段        | 修改说明                                            |
| ------------------------------------------------------------ | :-------------------: | :-------------------------------------------------- |
| examples/stepvideo/feature_extract/data_i2v.json              |      basic_parameters   | 数据集路径，`data_path`和`data_folder`分别配置data.jsonl的文件路径和目录 |
| examples/stepvideo/feature_extract/data_i2v.json              |      num_frames        | 最大的帧数，超过则随机选取其中的num_frames帧, 其中i2v配置102,t2v配置136 |
| examples/stepvideo/feature_extract/data_i2v.json              |      tokenizer_config  | tokenizer分词器选择，配置两种分词器的路径`"from_pretrained": "/model_path/step_llm/step1_chat_tokenizer.model"` 及`"from_pretrained": "/model_path/hunyuan_clip/tokenizer"` |
| examples/stepvideo/feature_extract/model_stepvideo_i2v.json   |      text_encoder    | 配置两种文本编译器路径`"from_pretrained": "./weights/step_llm/"`及`"from_pretrained": "./weights/hunyuan_clip/clip_text_encoder"` |
| examples/stepvideo/feature_extract/model_stepvideo_i2v.json   |      ae              | 配置VAE模型路径`"from_pretrained": "./weights/vae/vae_v2.safetensors"` |
| mindspeed_mm/tools/tools.json                                 |      save_path       | 提取后的特征保存路径                                |

2.启动特征提取

t2v执行命令
```bash
bash examples/stepvideo/feature_extract/feature_extraction.sh
```

i2v执行命令
```bash
bash examples/stepvideo/feature_extract/feature_extraction_i2v.sh
```

#### 配置参数
stepvideo训练阶段的启动文件为shell脚本，主要分为如下2个：

| I2V | T2V |
|:----:|:----:|
| pretrain_i2v.sh |pretrain_t2v.sh  |


模型参数的配置文件如下：

| I2V | T2V |
|:----:|:----:|
| pretrain_i2v_model.json |pretrain_t2v_model.json  |

以及涉及训练数据集的`data_static_resolution.json`文件

默认的配置已经经过测试，用户可按照自身环境修改如下内容：

| 配置文件                                                   |      修改字段       | 修改说明                                            |
| ---------------------------------------------------------- | :-----------------: | :-------------------------------------------------- |
| examples/stepvideo/{task_name}/data_static_resolution.json |  basic_parameters  | 数据集路径，`data_path`和`data_folder`分别配置提取后的特征的文件路径和目录 |
| examples/stepvideo/{task_name}/pretrain_*.sh        |    NPUS_PER_NODE    | 每个节点的卡数                                      |
| examples/stepvideo/{task_name}/pretrain_*.sh        |       NNODES        | 节点数量                                            |
| examples/stepvideo/{task_name}/pretrain_*.sh        |      LOAD_PATH      | 权重转换后的预训练权重路径                          |
| examples/stepvideo/{task_name}/pretrain_*.sh        |      SAVE_PATH      | 训练过程中保存的权重路径                            |
| examples/stepvideo/{task_name}/pretrain_*.sh        |         TP          | 训练时的TP size（建议根据训练时设定的分辨率调整）   |
| examples/stepvideo/{task_name}/pretrain_*.sh        |         CP          | 训练时的CP size（建议根据训练时设定的分辨率调整）   |


【并行化配置参数说明】：

当调整模型参数或者视频序列长度时，需要根据实际情况启用以下并行策略，并通过调试确定最优并行策略。

+ CP: 序列并行，当前支持Ulysess序列并行。

  - 使用场景：在视频序列（分辨率X帧数）较大时，可以开启来降低内存占用。
  
  - 使能方式：在启动脚本中设置 CP > 1，如：CP=2；
  
  - 限制条件：num_attention_heads 数量需要能够被TP*CP整除（在`exmaples/stepvideo/{task_name}/pretrain_xx_model.json`中配置，默认为48）
  
+ TP: 张量模型并行

  - 使用场景：模型参数规模较大时，单卡上无法承载完整的模型，通过开启TP可以降低静态内存和运行时内存。

  - 使能方式：在启动脚本中设置 TP > 1，如：TP=8

  - 限制条件：num_attention_heads 数量需要能够被TP*CP整除（在`exmaples/stepvideo/{task_name}/pretrain_xx_model.json`中配置，默认为48）


#### 启动预训练

t2v 启动预训练
```shell
bash examples/stepvideo/t2v/pretrain_t2v.sh
```

i2v 启动预训练
```shell
bash examples/stepvideo/i2v/pretrain_i2v.sh
```
---

## 推理

#### 准备工作

在开始之前，请确认环境准备、模型权重下载已完成

#### 配置参数

StepVideo推理启动文件为shell脚本，主要分为如下2个：
| I2V | T2V |
|:----:|:----:|
| inference_i2v.sh |inference_t2v.sh |

模型参数的配置文件如下：
| I2V | T2V |
|:----:|:----:|
| inference_i2v_model.json | inference_t2v_model.json  |


1. 权重配置

  需根据实际任务情况在启动脚本文件（如`inference_i2v.sh`）中的`LOAD_PATH="your_converted_dit_ckpt_dir"`变量中添加转换后的权重的实际路径(注意推理默认配置tp=4)，如`LOAD_PATH="./StepVideo-Converted"`,其中`./StepVideo-Converted`为转换后的权重的实际路径，其文件夹内容结构如权重转换一节所示。`LOAD_PATH`变量中填写的完整路径一定要正确，填写错误的话会导致权重无法加载但运行并不会提示报错。

2. VAE及T5模型路径配置

  根据实际情况修改模型参数配置文件，如`inference_i2v_model.json`文件中`text_encoder`字段配置两种文本编译器路径`"from_pretrained": "./weights/step_llm/"`及`"from_pretrained": "./weights/hunyuan_clip/clip_text_encoder"`，`ae`字段配置VAE模型路径`"from_pretrained": "./weights/vae/vae_v2.safetensors"`
  
  在`tokenizer`字段配置两种分词器路径`"from_pretrained": "/model_path/step_llm/step1_chat_tokenizer.model"`,及`"from_pretrained": "/model_path/hunyuan_clip/tokenizer"`

3. prompts配置

| t2v prompts配置文件                      |               修改字段               |                修改说明                 |
|----------------------------------------|:--------------------------------:|:-----------------------------------:|
| examples/stepvideo/t2v/samples_prompts.txt |               文件内容            |      自定义prompt      |


| i2v prompts配置文件                                   |               修改字段               |       修改说明       |
|--------------------------------------------|:--------------------------------:|:----------------:|
| examples/stepvideo/i2v/samples_i2v_images.txt  |               文件内容               |       图片路径       |
| examples/stepvideo/i2v/samples_i2v_prompts.txt |               文件内容               |    自定义prompt     |


如果使用训练后保存的权重进行推理，需要使用脚本进行转换，权重转换source_path参数请配置训练时的保存路径

```bash
mm-convert StepVideoConverter resplit \
  --cfg.source_path <your source path> \
  --cfg.target_path <your source path> \
```

#### 启动推理
t2v 启动推理脚本

```bash
bash examples/stepvideo/t2v/inference_t2v.sh
```

i2v 启动推理脚本

```bash
bash examples/stepvideo/i2v/inference_i2v.sh
```

---

## Dpo训练
目前仅以t2v穿刺dpo基础训练，更多功能待后续完善。

#### 环境准备

1. 参考docs/features/vbench-evaluate.md中的环境安装指导完成vbench及依赖三方件的安装
2. 将VBench的 [t2v json](https://github.com/Vchitect/VBench/blob/master/vbench/VBench_full_info.json) 下载到MM代码根路径"./vbench/VBench_full_info.json"

#### 生成视频样本

1. 修改推理配置文件：

| 参数配置文件                                                 |               修改字段               | 修改说明                          |
|------------------------------------------------------------|:--------------------------------:|:----------------------------------|
| examples/stepvideo/{task_name}/inference_*_model.json      |         from_pretrained          | 修改为下载的权重所对应路径（包括VAE、Text Encoder） |
| examples/stepvideo/{task_name}/inference_*_model.json      |  num_inference_videos_per_sample | 每个prompt生成的视频样本数量，建议至少大于2         |
| examples/stepvideo/{task_name}/inference_*_model.json        |  save_path | 生成视频的保存路径                         |
| examples/stepvideo/{task_name}/inference_*.sh              |   LOAD_PATH | 转换之后的transform部分权重路径              |

| t2v prompts配置文件                      |               修改字段               |                修改说明                 |
|----------------------------------------|:--------------------------------:|:-----------------------------------:|
| examples/stepvideo/t2v/samples_prompts.txt |               文件内容            |      自定义prompt      |


2. 启动推理流程生成视频样本：
```shell
bash examples/stepvideo/{task_name}/inference_{task_name}.sh
```

3. 删除视频样本保存路径下的video_grid.mp4，最终视频样本数量为：prompt条数 * $num_inference_videos_per_sample

#### 生成偏好数据集

执行如下命令，为生成的视频样本打分，并生成偏好数据文件
```bash
python examples/stepvideo/histgram_generator.py --prompt_file <prompt文件路径> --videos_path <视频样本路径> --num_inference_videos_per_sample <每个prompt生成的视频样本数量>
```

生成偏好数据集脚本的参数说明如下：

|参数| 含义 | 如何配置 |
|:------------|:----|:----|
| --prompt_file | prompt文件路径 | 与生成视频样本时，推理配置文件中的prompt字段值一致 |
| --videos_path | 视频样本路径 | 与生成视频样本时，推理配置文件中的save_path字段值一致 |
| --num_inference_videos_per_sample | 每个prompt生成的视频样本数量 | 与生成视频样本时，推理配置文件中的num_inference_videos_per_sample字段值一致 |

执行脚本后，会生成偏好数据集文件"data.jsonl"和评分概率直方图文件"video_score_histogram.json"，默认与视频样本目录平级

data.jsonl中包含成对的视频偏好数据和文本信息，具体示例如下：

```json
[
    {
        "file": "video_0.mp4",
        "file_rejected": "video_2.mp4",
        "captions": "prompt1",
        "score": 0.646468401,
        "score_rejected": 0.5799660087
    },
    {
        "file": "video_4.mp4",
        "file_rejected": "video_5.mp4",
        "captions": "prompt2",
        "score": 0.7914018631,
        "score_rejected": 0.69968328357
    },
    ......
]
```

#### 训练参数配置

在开始之前，请确认环境准备、模型权重准备、偏好数据准备已完成。

1. 权重配置

  需根据实际任务情况在启动脚本文件（如`posttrain_t2v_dpo.sh`）中的`LOAD_PATH="your_converted_dit_ckpt_dir"`变量中添加转换后的权重的实际路径，如`LOAD_PATH="./StepVideo-Converted"`,其中`./StepVideo-Converted`为转换后的权重的实际路径，其文件夹内容结构如权重转换一节所示。`LOAD_PATH`变量中填写的完整路径一定要正确，填写错误的话会导致权重无法加载但运行并不会提示报错。
根据需要填写`SAVE_PATH`变量中的路径，用以保存训练后的权重。

2. 偏好数据集路径配置

  根据实际情况修改`data_dpo.json`中的偏好数据集路径，分别为`"data_path":"/data_path/data.jsonl"`替换为实际的data.jsonl所在路径,`"data_folder":"/data_path/"`替换`"/data_path/"`为实际的视频样本所在路径。

3. VAE及text_encoder、tokenizer路径配置

  根据实际情况修改模型参数配置文件，如`posttrain_*_model.json`文件中`text_encoder`字段配置两种文本编译器路径`"from_pretrained": "./weights/step_llm/"`及`"from_pretrained": "./weights/hunyuan_clip/clip_text_encoder"`，`ae`字段配置VAE模型路径`"from_pretrained": "./weights/vae/vae_v2.safetensors"`
  `data_dpo.json`文件中`tokenizer_config`字段配置两种分词器路径`"from_pretrained": "/model_path/step_llm/step1_chat_tokenizer.model"` 及`"from_pretrained": "/model_path/hunyuan_clip/tokenizer"`

4. dpo参数配置

  根据实际情况修改`posttrain_t2v_model.json`中的直方图文件路径，即将`histgram_path`的值配置为执行生成偏好数据集脚本后，生成的"video_score_histogram.json"文件路径

#### 启动dpo训练

```bash
bash examples/stepvideo/{task_name}/posttrain_*_dpo.sh
```

---

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
