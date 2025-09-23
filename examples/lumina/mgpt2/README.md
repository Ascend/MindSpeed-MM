# Lumina-mGPT2使用指南

- [Lumina-mGPT2使用指南](#Lumina-mGPT2使用指南)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
    - [变更记录](#变更记录)
  - [环境安装](#环境安装)
    - [仓库拉取](#仓库拉取)
    - [环境搭建](#环境搭建)
    - [Decord搭建](#decord搭建)
  - [权重下载及转换](#权重下载及转换)
    - [权重转换](#权重转换)
  - [预训练](#预训练)
    - [数据预处理](#数据预处理)
    - [特征提取](#特征提取)
      - [准备工作](#准备工作)
      - [参数配置](#参数配置)
      - [启动特征提取](#启动特征提取)
    - [训练](#训练)
      - [准备工作](#准备工作-1)
      - [参数配置](#参数配置-1)
      - [启动训练](#启动训练)
  - [环境变量声明](#环境变量声明)

## 版本说明

### 参考实现

T2I微调任务

```
url=https://github.com/Alpha-VLLM/Lumina-mGPT-2.0
commit_id=978feb32473b57b79ea6a709687d01107e630478
```

### 变更记录

2025.08.15：首次发布Lumina-mGPT2微调任务

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

### 仓库拉取

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
```

### 环境搭建

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
git checkout e92252f4f1b7cbd78868922e6fe5659f8b762bf8
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

---

## 权重下载及转换

|   模型   |   下载链接   |
| ---- | ---- |
|   Lumina-mGPT2 7B  |   <https://huggingface.co/Alpha-VLLM/Lumina-mGPT-2.0/tree/main>   |
|  MoVQGAN  |  <https://huggingface.co/ai-forever/MoVQGAN/resolve/main/movqgan_270M.ckpt>    |

### 权重转换

需要对下载后的Lumina-mGPT2模型权重进行权重转换，运行权重转换脚本：

```shell
mm-convert LuminaConverter hf_to_mm \
 --cfg.source_path <./Alpha-VLLM/Lumina-mGPT-2.0/> \
 --cfg.target_path <./Lumina/Lumina-mGPT-2.0-mm-convert/> \
```

权重转换脚本的参数说明如下：

|参数| 含义 | 默认值 |
|:------------|:----|:----|
| --cfg.source_path | 原始权重路径 | / |
| --cfg.target_path | 转换后的权重保存路径 | / |

---

## 预训练

### 数据预处理

将数据处理成如下格式

```bash
</data/hunyuanvideo/dataset>
  ├──data.json
  ├──images
  │  ├──image0001.jpg
  │  ├──image0002.png
```

其中，`images/`下存放图片，data.json中包含该数据集中所有的图片-文本对信息，具体示例如下：

```json
[
    {
        "file": "images/image0001.jpg",
        "prompt": "Image discrimination1.",
    },
    {
        "file": "images/image0002.jpg",
        "prompt": "Image discrimination2.",
    },
    ......
]
```

### 特征提取

#### 准备工作

在开始之前，请确认环境准备、模型权重和数据集预处理已经完成

#### 参数配置

检查模型权重路径、数据集路径、提取后的特征保存路径等配置是否完成

| 配置文件                                                     |       修改字段        | 修改说明                                            |
| ------------------------------------------------------------ | :-------------------: | :-------------------------------------------------- |
| examples/lumina/mgpt2/feature_extract/data.json              |         path          | 数据集`data.json`文件的路径        |
| examples/lumina/mgpt2/feature_extract/data.json              |     from_pretrained    | 修改为下载的Lumina mGPT2权重所对应路径     |
| examples/lumina/mgpt2/feature_extract/model.json              |    from_pretrained    | 修改为下载的MoVQGAN权重所对应路径 |
| examples/lumina/mgpt2/feature_extract/feature_extraction.sh  |     NPUS_PER_NODE     | 卡数                                                |
| mindspeed_mm/tools/tools.json                                |       save_path       | 提取后的特征保存路径                                |

#### 启动特征提取

```bash
bash examples/lumina/mgpt2/feature_extract/feature_extraction.sh
```

### 训练

#### 准备工作

在开始之前，请确认环境准备、模型权重下载、特征提取已完成。

#### 参数配置

检查模型权重路径、并行参数配置等是否完成

| 配置文件                                                   |      修改字段       | 修改说明                                            |
| ---------------------------------------------------------- | :-----------------: | :-------------------------------------------------- |
| examples/lumina/mgpt2/feature_data.json        | basic_parameters   | 数据集路径，`path`配置提取后的特征的文件路径 |
| examples/lumina/mgpt2/model.json  |    vocabulary_map_path    | 词表文件路径，配置为下载的Lumina mGPT2原始权重所对应路径 |
| examples/lumina/mgpt2/pretrain.sh |       NPUS_PER_NODE        | 每个节点的卡数  |
| examples/lumina/mgpt2/pretrain.sh |       NNODES        | 节点数量  |
| examples/lumina/mgpt2/pretrain.sh |      LOAD_PATH      | 权重转换后的预训练权重路径                          |
| examples/lumina/mgpt2/pretrain.sh |      SAVE_PATH      | 训练过程中保存的权重路径                            |

【并行化配置参数说明】：

当调整模型参数或者token序列长度时，需要根据实际情况启用以下并行策略，并通过调试确定最优并行策略。

- fsdp1

  - 使用场景：在模型参数规模较大时，单卡上无法承载完整的模型，可以通过开启fsdp1降低内存。
  
  - 使能方式：`examples/lumina/mgpt2/model.json`中添加fsdp1配置信息

  - 限制条件: 该特性目前不兼容模型切分，使能该特性时，TP、PP等须设置为1
  
> ⚠️**目前未适配CP与TPSP**

#### 启动训练

```bash
bash examples/lumina/mgpt2/pretrain.sh
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

