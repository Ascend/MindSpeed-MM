# Wan2.1 使用指南

- [Wan2.1 使用指南](#wan21-使用指南)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
    - [变更记录](#变更记录)
  - [任务支持列表](#任务支持列表)
  - [环境安装](#环境安装)
    - [仓库拉取](#仓库拉取)
    - [环境搭建](#环境搭建)
    - [Decord搭建](#decord搭建)
  - [权重下载及转换](#权重下载及转换)
    - [Diffusers权重下载](#diffusers权重下载)
    - [权重转换](#权重转换)
  - [预训练](#预训练)
    - [数据预处理](#数据预处理)
    - [训练](#训练)
      - [准备工作](#准备工作-1)
      - [参数配置](#参数配置-1)
      - [启动训练](#启动训练)
  - [lora 微调](#lora-微调)
    - [准备工作](#准备工作-2)
    - [参数配置](#参数配置-2)
    - [启动微调](#启动微调)
  - [环境变量声明](#环境变量声明)

## 版本说明

#### 参考实现

T2V I2V LoRA微调任务

```
url=https://github.com/modelscope/DiffSynth-Studio.git
commit_id=03ea278
```


## 任务支持列表

| 模型大小 | 任务类型 | 预训练 | lora微调 | 在线T2V推理 | 在线I2V推理 | 在线FLF2V推理 | 在线V2V推理 |
|------|:----:|:----|:-------|:-----|:-----|:-----|:-----|
| 1.3B | t2v  | ✔ | ✔ | ✔ |  |  | ✔ |
| 1.3B | i2v  | ✔ |  |  |  |  |  |

## 环境安装

MindSpeed-MM MindSpore后端的依赖配套如下表，安装步骤参考[基础安装指导](../../../docs/mindspore/install_guide.md)。

| 依赖软件         |                                                              |
| ---------------- | ------------------------------------------------------------ |
| 昇腾NPU驱动固件  | [在研版本](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha) |
| 昇腾 CANN        | [在研版本](https://www.hiascend.com/zh/developer/download/community/result?module=cann) |
| MindSpore        | [2.7.0](https://www.mindspore.cn/install/)         |
| Python           | >=3.9                                                        |                                          |

<a id="jump1.1"></a>
### 1. 仓库拉取及环境搭建

针对MindSpeed MindSpore后端，昇腾社区提供了一键拉起工具MindSpeed-Core-MS，旨在帮助用户自动拉取相关代码仓并对torch代码进行一键适配，进而使用户无需再额外手动开发适配即可在华为MindSpore+CANN环境下一键拉起模型训练。在进行一键拉起前，用户需要拉取相关的代码仓以及进行环境搭建：


```
# 创建conda环境
conda create -n test python=3.10
conda activate test

# 使用环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# 安装MindSpeed-Core-MS拉起工具
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git

# 使用MindSpeed-Core-MS内部脚本自动拉取相关代码仓并一键适配、提供配置环境
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm

mkdir ckpt
mkdir data
mkdir logs
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

### Diffusers权重下载

|   模型   |   Huggingface下载链接   |
| ---- | ---- |
|   T2V-1.3B   |   <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>   |

### 权重转换

需要对下载后的Wan2.1模型权重`transformer`部分进行权重转换，运行权重转换脚本：

```shell
mm-convert WanConverter hf_to_mm \
 --cfg.source_path <./weights/Wan-AI/Wan2.1-{T2V/I2V}-1.3B-Diffusers/transformer/> \
 --cfg.target_path <./weights/Wan-AI/Wan2.1-{T2V/I2V}-1.3B-Diffusers/transformer/>
 --cfg.target_parallel_config.pp_layers <pp_layers>
```

权重转换脚本的参数说明如下：

| 参数              | 含义                     | 默认值                                                       |
| :---------------- | :----------------------- | :----------------------------------------------------------- |
| --cfg.source_path | 原始权重路径             | /                                                            |
| --cfg.target_path | 转换或切分后权重保存路径 | /                                                            |
| --pp_layers   | PP/VPP层数               | 开启PP时, 使用PP和VPP需要指定各stage的层数并转换, 默认为`[]`，即不使用 |

如需转回Hugging Face格式，需运行权重转换脚本：

**注**： 如进行layer zero进行训练，则需首先进行其[训练权重后处理](#jump1)，再进行如下操作：

```shell
mm-convert WanConverter mm_to_hf \
 --cfg.source_path <path for your saved weight/> \
 --cfg.target_path <./converted_weights/Wan-AI/Wan2.1-{T2V/I2V}-1.3B-Diffusers/transformer/>
 --cfg.hf_path <weights/Wan-AI/Wan2.1-{T2V/I2V}-1.3B-Diffusers/transformer/>
```

权重转换脚本的参数说明如下：

|参数| 含义 | 默认值 |
|:------------|:----|:----|
| --cfg.source_path | MindSpeed MM保存的权重路径                                   | /      |
| --cfg.target_path | 转换后的Hugging Face权重路径                                 | /      |
| --cfg.hf_path     | 原始Hugging Face权重路径，需要从该目录下获取原始huggingface配置文件 |    /   |

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

修改`examples/mindsporewan2.1/feature_extract/data.txt`文件，其中每一行表示个数据集，第一个参数表示数据文件夹的路径，第二个参数表示`data.json`文件的路径，用`,`分隔


### 训练

#### 准备工作

在开始之前，请确认环境准备、模型权重下载、特征提取已完成。

#### 参数配置

检查模型权重路径、并行参数配置等是否完成

| 配置文件   |      修改字段       | 修改说明      |
| --- | :---: | :--- |
| examples/mindsporewan2.1/{model_size}/{task}/feature_data.json   |  basic_parameters   | 数据集路径，`data_path`和`data_folder`分别配置提取后的特征的文件路径和目录 |
| examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh |    NPUS_PER_NODE    | 每个节点的卡数                                      |
| examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh |       NNODES        | 节点数量                                            |
| examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh |      LOAD_PATH      | 权重转换后的预训练权重路径                          |
| examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh |      SAVE_PATH      | 训练过程中保存的权重路径                            |
| examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh |         CP          | 训练时的CP size（建议根据训练时设定的分辨率调整）   |

【并行化配置参数说明】：

当调整模型参数或者视频序列长度时，需要根据实际情况启用以下并行策略，并通过调试确定最优并行策略。

- CP: 序列并行。

  - 使用场景：在视频序列（分辨率X帧数）较大时，可以开启来降低内存占用。
  
  - 使能方式：在启动脚本中设置 CP > 1，如：CP=2；
  
  - 限制条件：head 数量需要能够被CP整除（在`examples/mindsporewan2.1/{model_size}/{task}/pretrain_model.json`中配置，参数为`num_heads`）

  - 默认使能方式为Ulysses序列并行。

  - DiT-RingAttention：DiT RingAttention序列并行请[参考文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/dit_ring_attention.md)

  - DiT-USP: DiT USP混合序列并行（Ulysses + RingAttention）请[参考文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/dit_usp.md)

  - FPDT(Fully Pipelined Distributed Transformer): Ulysses Offload 并行请[参考文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/fpdt.md)

- layer_zero

  - 使用场景：在模型参数规模较大时，单卡上无法承载完整的模型，可以通过开启layerzero降低静态内存。
  
  - 使能方式：`examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh`的`GPT_ARGS`中加入`--layerzero`和`--layerzero-config ${layerzero_config}`
  
  <a id="jump1"></a>
  - 训练权重后处理：使用该特性训练时，保存的权重需要使用下面的转换脚本进行后处理才能用于推理：

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    mm-convert WanConverter layerzero_to_mm \
     --cfg.source_path <./save_ckpt/wan2.1/> \
     --cfg.target_path <./save_ckpt/wan2.1_megatron_ckpt/>
    ```

- PP：流水线并行

  目前支持将predictor模型切分流水线。

  - 使用场景：模型参数较大时候，通过流水线方式切分并行，降低训练内存占用

  - 使能方式：
    - 修改在 pretrain_model.json 文件中的"pipeline_num_layers", 类型为list。该list的长度即为 pipeline rank的数量，每一个数值代表rank_i中的层数。例如，[7, 8, 8, 7]代表有4个pipeline stage， 每个容纳7/8个dit layers。注意list中 所有的数值的和应该和总num_layers字段相等。此外，pp_rank==0的stage中除了包含dit层数以外，还会容纳text_encoder和ae，因此可以酌情减少第0个stage的dit层数。注意保证PP模型参数配置和模型转换时的参数配置一致。
    - 此外使用pp时需要在运行脚本GPT_ARGS中打开以下几个参数
  
    ```shell
    PP = 4 # PP > 1 开启 
    GPT_ARGS="
    --optimization-level 2 \
    --use-multiparameter-pipeline-model-parallel \  #使用PP或者VPP功能必须要开启
    --variable-seq-lengths \  #按需开启，动态shape训练需要加此配置，静态shape不要加此配置
    “
    ```

- VP: 虚拟流水线并行

  目前支持将predictor模型切分虚拟流水线并行。

  - 使用场景：对流水线并行进行进一步切分，通过虚拟化流水线，降低空泡
  - 使能方式:
    - 如果想要使用虚拟流水线并行，将pretrain_model.json文件中的"pipeline_num_layers"一维数组改造为两维，其中第一维表示虚拟并行的数量，二维表示流水线并行的数量，例如[[3, 4, 4, 4], [3, 4, 4, 4]]其中第一维两个数组表示vp为2, 第二维的stage个数为4表示流水线数量pp为3或4。
    - 需要在pretrain.sh当中修改如下变量，需要注意的是，VP仅在PP大于1的情况下生效:

    ```shell
    PP=4
    VP=2
    
    GPT_ARGS="
      --pipeline-model-parallel-size ${PP} \
      --virtual-pipeline-model-parallel-size ${VP} \
      --optimization-level 2 \
      --use-multiparameter-pipeline-model-parallel \  #使用PP或者VPP功能必须要开启
      --variable-seq-lengths \  #按需开启，动态shape训练需要加此配置，静态shape不要加此配置
    ”
    ```


## lora 微调

### 准备工作

数据处理、特征提取、权重下载及转换同预训练章节

### 参数配置

参数配置同训练章节，除此之外，中涉及lora微调特有参数：

| 配置文件                                             |        修改字段         | 修改说明                         |
|--------------------------------------------------|:-------------------:|:-----------------------------|
| examples/mindsporewan2.1/{model_size}/{task}/finetune_lora.sh |       lora-r        | lora更新矩阵的维度                  |
| examples/mindsporewan2.1/{model_size}/{task}/finetune_lora.sh |     lora-alpha      | lora-alpha 调节分解后的矩阵对原矩阵的影响程度 |
| examples/mindsporewan2.1/{model_size}/{task}/finetune_lora.sh | lora-target-modules | 应用lora的模块列表                  |

### 启动微调

```bash
bash examples/mindsporewan2.1/{model_size}/{task}/finetune_lora.sh
```

微调完成后，可以使用权重转换工具，将训练好的lora权重与原始权重进行合并

```bash
mm-convert WanConverter merge_lora_to_base \
 --cfg.source_path <./converted_weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer/> \
 --cfg.target_path <./converted_weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer_merge/> \
 --cfg.lora_path <lora_save_path> \
 --lora_alpha 64 \
 --lora_rank 64
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

