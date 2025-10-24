# OpenSoraPlan1.5使用指南

- [OpenSoraPlan1.5使用指南](#opensoraplan15使用指南)
  - [环境安装](#环境安装)
    - [仓库拉取](#仓库拉取)
    - [环境搭建](#环境搭建)
    - [Decord搭建](#decord搭建)
  - [权重下载及转换](#权重下载及转换)
  - [预训练](#预训练)
    - [数据预处理](#数据预处理)
    - [训练](#训练)
      - [准备工作](#准备工作)
      - [参数配置](#参数配置)
      - [启动训练](#启动训练)
  - [推理](#推理)
    - [准备工作](#准备工作-1)
    - [参数配置](#参数配置-1)
    - [启动推理](#启动推理)
  - [环境变量声明](#环境变量声明)

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

### 仓库拉取

```shell
git clone --branch 2.2.0 https://gitcode.com/Ascend/MindSpeed-MM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ../MindSpeed-MM
```

### 环境搭建

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch及torch_npu包
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl


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

### Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

## 权重下载及转换

权重下载链接：

VAE和DiT： [opensoraplan1.5](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.5.0)

Text Encoder：[t5](https://huggingface.co/google/t5-v1_1-xl) 和 [CLIP](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)

需要对下载后的opensoraplan1.5模型 `vae`部分进行权重转换，运行权重转换脚本

```bash
mm-convert OpenSoraPlanConverter --version v1.5 vae_convert \
 --cfg.source_path <"./raw_ckpt/open-sora-plan/vae/checkpoint.ckpt">
 --cfg.target_path <"./ckpt/vae/vae.pt">
```

需要对下载后的opensoraplan1.5模型 `DiT`部分进行权重转换，运行权重转换脚本

```bash
mm-convert OpenSoraPlanConverter --version v1.5 source_to_mm \
 --cfg.source_path <"./raw_ckpt/open-sora-plan/model_ema.pt/">
 --cfg.target_path <"./ckpt/open-sora-plan/">
 --cfg.target_parallel_config.tp_size <tp_size>
```

权重转换脚本的参数说明如下：

| 参数                                 | 含义                     | 默认值           |
| ------------------------------------ | ------------------------ | ---------------- |
| --version                            | opensoraplan系列不同版本 | 需要设置为`v1.5` |
| --cfg.source_path                    | 原始权重路径             | /                |
| --cfg.target_path                    | 转换或切分后权重保存路径 | /                |
| --cfg.target_parallel_config.tp_size | dit部分切分时的tp size   | 1                |

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
        "num_frames": 57,
        "fps": 12,
        "resolution": {
            "height": 288,
            "width": 512
        }
    },
    {
        "path": "videos/video0002.mp4",
        "cap": "Video discrimination2.",
        "num_frames": 57,
        "fps": 12,
        "resolution": {
            "height": 288,
            "width": 512
        }
    },
    ......
]
```

修改 `examples/opensoraplan1.5/data.txt`文件，其中每一行表示个数据集，第一个参数表示数据文件夹的路径，第二个参数表示 `data.json`文件的路径，用 `,`分隔

### 训练

#### 准备工作

在开始之前，请确认环境准备、模型权重下载已完成

#### 参数配置

检查模型权重路径、并行参数配置等是否完成

| 配置文件                                     | 修改字段        | 修改说明                                          |
| -------------------------------------------- | --------------- | ------------------------------------------------- |
| examples/opensoraplan1.5/data.txt            | 文件内容        | 训练数据集路径                                    |
| examples/opensoraplan1.5/pretrain.sh         | NPUS_PER_NODE   | 每个节点的卡数                                    |
| examples/opensoraplan1.5/pretrain.sh         | NNODES          | 节点数量                                          |
| examples/opensoraplan1.5/pretrain.sh         | LOAD_PATH       | 权重转换后的预训练权重路径                        |
| examples/opensoraplan1.5/pretrain.sh         | SAVE_PATH       | 训练过程中保存的权重路径                          |
| examples/opensoraplan1.5/pretrain.sh         | TP              | 训练时的TP size（建议根据训练时设定的分辨率调整） |
| examples/opensoraplan1.5/pretrain_model.json | from_pretrained | vae和text encoder的权重路径                       |

【并行化配置参数说明】

当调整模型参数或者视频序列长度时，需要根据实际情况启用以下并行策略，并通过调试确定最优并行策略。

- TP: 张量模型并行
  - 使用场景：模型参数规模较大时，单卡上无法承载完整的模型，通过开启TP可以降低静态内存和运行时内存。
  - 使能方式：在启动脚本 `examples/opensoraplan1.5/pretrain.sh`中设置 TP > 1，如：TP=8
  - 限制条件：head 数量需要能够被TP*CP整除（在 `examples/opensoraplan1.5/pretrain_model.json`中配置，默认为24）
- TP-SP
  - 使用场景：在张量模型并行的基础上，进一步对 LayerNorm 和 Dropout 模块的序列维度进行切分，以降低动态内存。
  - 使能方式：在 GPT_ARGS 设置 --sequence-parallel
  - 使用建议：建议在开启TP时同步开启该设置，该配置默认开启

#### 启动训练

```bash
bash examples/opensoraplan1.5/pretrain.sh
```

## 推理

### 准备工作

在开始之前，请确认环境准备、模型权重下载已完成

### 参数配置

| 配置文件                                      | 修改字段        | 修改说明                               |
| --------------------------------------------- | --------------- | -------------------------------------- |
| examples/opensoraplan1.5/inference_model.json | from_pretrained | vae和text encoder的权重路径            |
| examples/opensoraplan1.5/inference_model.json | save_path       | 推理结果的保存路径                     |
| examples/opensoraplan1.5/inference_model.json | input_size      | 生成视频的分辨率，格式为 [t, h, w]     |
| examples/opensoraplan1.5/samples_prompts.txt  | 文件内容        | 可自定义自己的prompt，一行为一个prompt |
| examples/opensoraplan1.5/inference.sh         | LOAD_PATH       | 转换之后的dit部分权重路径              |

### 启动推理

```bash
bash examples/opensoraplan1.5/inference.sh
```

如果使用训练后的权重进行推理，且在训练时对dit部分进行了TP切分，推理时可以运行下面的脚本将该部分权重进行合并

```bash
mm-convert OpenSoraPlanConverter --version v1.5 resplit \
 --cfg.source_path <"./ckpt/open-sora-plan">
 --cfg.target_path <"./ckpt/open-sora-plan_merge/">
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

