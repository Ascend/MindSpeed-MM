# Whisper 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
  - [权重转换](#jump2.2)
- [数据集准备](#jump3)
  - [数据集与权重下载](#jump3.1)
- [预训练](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动预训练](#jump4.3)
- [模型推理与权重转换](#jump5)
- [环境变量声明](#jump6)

---

<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

<a id="jump1.1"></a>

#### 1. 仓库拉取

```shell
    git clone --branch 2.2.0 https://gitcode.com/Ascend/MindSpeed-MM.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_v0.12.1
    cp -r megatron ../MindSpeed-MM/
    cd ..
    cd MindSpeed-MM
```

<a id="jump1.2"></a>

#### 2. 环境搭建

```bash
    # python3.10
    conda create -n test python=3.10
    conda activate test

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch及torch_npu包
    pip install torch-2.7.1-cp310-cp310-manylinux2014_aarch64.whl 
    pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 安装加速库
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed
    # checkout commit from MindSpeed core_r0.12.1
    git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其余依赖库
    pip install librosa
    conda install -c conda-forge libsndfile
    pip install -e .
```

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

- [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3/tree/main)：Whisper large-v3模型；

获取权重结构如下：

   ```
   $whisper-large-v3
   ├── config.json
   ├── pytorch_model.bin
   ├── tokenizer.json
   └── ...
   ```

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，因此需要使用如下脚本代码对下载的预训练权重进行转换。

```python
import torch

pretrained_checkpoint = torch.load("your pretrained ckpt path", map_location="cpu")
new_checkpoint = {}
for key in pretrained_checkpoint.keys():
    model_key = key.replace("q_proj", "proj_q")
    model_key = model_key.replace("k_proj", "proj_k")
    model_key = model_key.replace("v_proj", "proj_v")
    model_key = model_key.replace("out_proj", "proj_out")
    new_checkpoint[model_key] = pretrained_checkpoint[key]

torch.save(new_checkpoint, "whisper.pth")
```

---

<a id="jump3"></a>

## 数据集准备

<a id="jump3.1"></a>

#### 1. 数据集下载

用户需自行获取mozilla-foundation/common_voice_11_0数据集，
获取数据结构如下：

   ```
   $common_voice_11_0
   ├── audio
   ├── ├── hi
   ├── ├── ├── train
   ├── ├── ├── ├── hi_train_0.tar
   ├── ├── ├── test
   ├── ├── ├── ...
   ├── ├── en
   ├── ├── ...
   ├── transcript
   ├── ├── hi
   ├── ├── ├── train.tsv
   ├── ├── ├── test.tsv
   ├── ├── ├── ...
   ├── ├── en
   ├── ├── ...
   ├── common_voice_11_0.py
   ├── count_n_shard.py
   └── ...
   ```

---

<a id="jump4"></a>

## 预训练

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备**等，详情可查看对应章节

<a id="jump4.2"></a>

#### 2. 配置参数

需根据实际情况修改`model.json`和`data.json`中的权重和数据集路径，包括`ckpt_path`、`dataset_name_or_path`、`processor_name_or_path`等字段

【单机运行】

```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=localhost
    MASTER_PORT=29501
    NNODES=1  
    NODE_RANK=0  
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

【多机运行】

```shell
    # 根据分布式集群实际情况配置分布式参数
    GPUS_PER_NODE=8  #每个节点的卡数
    MASTER_ADDR="your master node IP"  #都需要修改为主节点的IP地址（不能为localhost）
    MASTER_PORT=29501
    NNODES=2  #集群里的节点数，以实际情况填写,
    NODE_RANK="current node id"  #当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动预训练

```shell
    bash examples/whisper/pretrain_whisper.sh
```

**注意**：

- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，需要在每个节点准备训练数据

<a id="jump5"></a>

#### 4. 模型推理与权重转换

当前MindSpeed-MM未提供whisper模型的推理代码，需要将训练后模型转回huggingface格式自行推理
转回脚本代码示例如下：

```python
import torch
import mindspeed.megatron_adaptor

pretrained_checkpoint = torch.load("your trained ckpt path/model_optim_rng.pt", map_location="cpu")
pretrained_checkpoint = pretrained_checkpoint['model']

new_checkpoint = {}
for key in pretrained_checkpoint.keys():
    if key == "proj_out.weight":
        model_key = key
    else:
        model_key = key.replace("proj_q", "q_proj")
        model_key = model_key.replace("proj_k", "k_proj")
        model_key = model_key.replace("proj_v", "v_proj")
        model_key = model_key.replace("proj_out", "out_proj")
    new_checkpoint[model_key] = pretrained_checkpoint[key]

torch.save(new_checkpoint, "whisper_hf.bin")
```

**注意**：

- 该转回脚本需在MindSpeed-MM根目录下执行
<a id="jump6"></a>

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

