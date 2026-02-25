# HunyuanVideo1.5 使用指南

- [HunyuanVideo1.5 使用指南](#hunyuanvideo15-使用指南)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
    - [变更记录](#变更记录)
  - [环境安装](#环境安装)
    - [仓库拉取](#1-仓库拉取)
    - [环境搭建](#2-环境搭建)
    - [Decord搭建](#3-decord搭建)
  - [权重下载](#权重下载)
    - [权重下载](#权重下载)
  - [预训练](#预训练)
    - [数据预处理](#数据预处理)
    - [训练](#训练)
      - [准备工作](#准备工作)
      - [参数配置](#参数配置)
      - [启动训练](#启动训练)
  - [环境变量声明](#环境变量声明)

## 版本说明

#### 参考实现

【T2V 任务】

```
url=https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
commit_id=bf576ef1d5ddc643cf814b1dff4f4dcc9a7581c7
```

【推理】

```
url=https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
commit_id=bf576ef1d5ddc643cf814b1dff4f4dcc9a7581c7
```

#### 变更记录

2026.02.12: 首次支持HunyuanVideo1.5模型

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md)

### 1. 仓库拉取
拉取MindSpeed MM代码仓，并进入代码仓根目录：
```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

### 2. 环境搭建
对于X86架构机器，执行如下指令：
```bash
bash scripts/install.sh --arch x86 --msid d76dbddd4517d48a2fc1cd494de8b9a6cfdbfbab && pip install -r examples/hunyuanvideo_1.5/requirements.txt
```
对于ARM架构机器，执行如下指令：
```bash
bash scripts/install.sh --arch arm --msid d76dbddd4517d48a2fc1cd494de8b9a6cfdbfbab && pip install -r examples/hunyuanvideo_1.5/requirements.txt
```

### 3. Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

---



## 权重下载

1. 下载预训练的DiT和VAE权重
``` bash
mkdir HunyuanVideo1.5
hf download tencent/HunyuanVideo-1.5 --local-dir ./HunyuanVideo1.5
```
离线链接：
- [tencent/HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo)
2. 下载文本编码器

``` bash
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./HunyuanVideo1.5/text_encoder/llm
hf download google/byt5-small --local-dir ./HunyuanVideo1.5/text_encoder/byt5-small
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./HunyuanVideo1.5/text_encoder/Glyph-SDXL-v2
```
离线链接：
- [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)
- [google/byt5-small](https://huggingface.co/google/byt5-small/tree/main)
- [AI-ModelScope/Glyph-SDXL-v2](https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/files)

3. 下载视觉编码器
```bash
hf download black-forest-labs/FLUX.1-Redux-dev --local-dir ./ckpts/vision_encoder/siglip --token <your_hf_token>
```
离线链接：
- [black-forest-labs/FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/tree/main)

4. 最终文件结构如下：
```bash
HunyuanVideo1.5
├── text_encoder
│   ├── Glyph-SDXL-v2
│   │   ├── assets
│   │   │   ├── color_idx.json
│   │   │   ├── multilingual_10-lang_idx.json
│   │   │   └── ...
│   │   └── checkpoints
│   │       ├── byt5_model.pt
│   │       └── ...
│   └─  ...
└─  scheduler
└─  transformer
│   ├── 720p_t2v
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model.safetensors
└─  vae
└─  scheduler
└─  vision_encoder
└─  upsampler
```

---

## 预训练

### 数据预处理

将数据处理成如下格式

```bash
</dataset>
  ├──data.json
  ├──videos
  │  ├──video0001.mp4
  │  ├──video0002.mp4
```

其中，`videos/`下存放视频，data.json中包含该数据集中所有的视频-文本对信息，具体示例如下：

```json
[
    {
        "path": "videos/video0001.mp4",
        "cap": "Video discrimination1.",
        "num_frames": 81,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 832
        }
    },
    {
        "path": "videos/video0002.mp4",
        "cap": "Video discrimination2.",
        "num_frames": 81,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 832
        }
    },
    ......
]
```

在模型权重文件中新建`HunyuanVideo1.5/dataset/data.txt`文件，其中每一行表示一个数据集，第一个参数表示数据文件夹的路径，第二个参数表示`data.json`文件的路径，用`,`分隔

### 训练

#### 准备工作

在开始之前，请确认环境准备、模型权重下载、数据预处理已完成。

#### 参数配置

检查数据集路径、模型权重路径、并行参数配置等是否完成

| 配置文件                                                         |      修改字段       | 修改说明                                           |
|--------------------------------------------------------------| :---: |:-----------------------------------------------|
| examples/hunyuanvideo_1.5/{task}/data.json                   |  from_pretrained  | 修改为下载的tokenizer的权重所对应的路径                       |
| examples/hunyuanvideo_1.5/{task}/model_hunyuanvideo_15.json  |  from_pretrained  | 修改为下载的权重所对应路径（包括vae,  text_encoder）            |
| examples/hunyuanvideo_1.5/{task}/model_hunyuanvideo_15.json  |  byT5_ckpt_path  | 修改为下载的byt5_model.pt所对应路径                       |
| examples/hunyuanvideo_1.5/{task}/pretrain_hunyuanvideo_15.sh |    NPUS_PER_NODE  | 每个节点的卡数                                        |
| examples/hunyuanvideo_1.5/{task}/pretrain_hunyuanvideo_15.sh             |       NNODES      | 节点数量                                           |
| examples/hunyuanvideo_1.5/{task}/pretrain_hunyuanvideo_15.sh             |      LOAD_PATH    | 预训练DiT权重路径,下面一级目录包含config文件                    |
| examples/hunyuanvideo_1.5/{task}/pretrain_hunyuanvideo_15.sh             |      SAVE_PATH    | 训练过程中保存的权重路径                                   |

**注**： 当前LOAD_PATH路径无效时，MindSpeed会对模型随机初始化从头训练。为防止加载失败，请留意日志中的warning信息，或者自行确认路径合法。

【并行化配置参数说明】：
- fsdp2

  - 使用场景：在模型参数规模较大时，可以通过开启fsdp2降低静态内存，默认开启。
  
  - 使能方式：`examples/hunyuanvideo_1.5/{task}/pretrain_*.sh`的`GPT_ARGS`中加入`--use-torch-fsdp2`，`--fsdp2-config-path ${fsdp2_config}`，`--untie-embeddings-and-output-weights`以及`--ckpt-format torch_dcp`，其中fsdp2_config配置请参考：[FSDP2说明](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/fsdp2.md)
  <a id="jump1"></a>
#### 启动训练

【T2V 任务】

```bash
bash examples/hunyuanvideo_1.5/{task}/pretrain_*.sh
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

