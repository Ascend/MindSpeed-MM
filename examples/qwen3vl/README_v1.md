# Qwen3VL 使用指南

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
- [微调](#微调)
  - [准备工作](#1-准备工作)
  - [配置参数](#2-配置参数)
  - [启动微调](#3-启动微调)
- [环境变量声明](#环境变量声明)

## 版本说明

### 参考实现

```shell
url=https://github.com/huggingface/transformers.git
commit_id=c0dbe09
```

### 变更记录

2026.03.16：首次基于FSDP2后端支持Qwen3-VL模型。

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/install_guide.md)，完成昇腾软件安装。
> Python版本推荐3.10，torch和TorchNPU版本推荐2.7.1版本

‼️MoE部分的加速特性依赖较新版本的TorchNPU和CANN，推荐使用以下版本

- [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler)
- [TorchNPU](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_description.md)

<a id="jump1.2"></a>

### 2. 环境搭建

```bash
# 拉取MindSpeed MM代码仓，并进入代码仓根目录：
git clone https://gitcode.com/Ascend/MindSpeed-MM.git

# 安装mindspeed及依赖
git clone -b 26.1.0_core_r0.12.1 https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
cp -r mindspeed ../MindSpeed-MM/

# 安装mindspeed mm及依赖
cd ../MindSpeed-MM
pip install -e .
```

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

### 1. 权重下载

从Hugging Face库下载对应的模型权重:

> [!NOTE]
>
> 如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

- 模型地址: [Qwen3-VL-*B](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe)；

 建议将下载的模型权重保存到上述搭建好的工程目录（MindSpeed-MM）下，如`ckpt/Qwen3-VL-*B-Instruct`。(*表示对应的尺寸)

如果使用随机初始化参数，保持`load`参数注释即可。

如果使用fsdp2的meta init初始化模型，需要先完成以下权重转换

```bash
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 执行权重转换
mm-convert GenericDCPConverter hf_to_dcp \
--hf_dir ckpt/Qwen3-VL-30B-Instruct \
--dcp_dir ckpt/Qwen3-VL-30B-Instruct-dcp

# 转换后的目录结构为：
# ———— Qwen3-VL-30B-Instruct-dcp
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

并在当前工程目录下的`examples/qwen3vl/qwen3vl_30B_config_v1.yaml`中将`init_model_with_meta_device`参数配置为`true`（当前默认值），并取消`load`参数注释，将`load`配置成转换后的dcp权重路径（写到`release`文件夹的上一级目录）。

---
<a id="jump3"></a>

## 数据集准备及处理

- 使用**真实数据集**训练：参考[针对VL模型的数据构造 · 使用真实数据集](../../docs/zh/features/building_data_for_VLModel.md#real-data)（下载COCO2017 → 下载LLaVA-Instruct-150K标注 → 运行转换脚本生成`mllm_format_llava_instruct_data.json`）。
- 使用**虚构数据**做功能/性能测试：参考[针对VL模型的数据构造 · 使用虚构数据](../../docs/zh/features/building_data_for_VLModel.md#mock-data)。

## 微调

<a id="jump4.1"></a>

### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

### 2. 配置参数

【权重与数据目录配置】（必选）

根据实际情况修改当前工程目录下的`examples/qwen3vl/qwen3vl_30B_config_v1.yaml`中的权重和数据集路径，包括`data`参数下的`model_name_or_path`、`dataset_dir`、`dataset`等字段，建议均配置为绝对路径。

权重路径填写示例：如果下载的模型权重路径为/home/data/Qwen3-VL-30B-A3B-Instruct，此时配置如下：
`model_name_or_path`配置为&HF_MODEL_LOAD_PATH /home/data/Qwen3-VL-30B-A3B-Instruct
其中&HF_MODEL_LOAD_PATH 是框架约定的固定锚点标记，请勿修改名称

数据集路径填写示例：如果数据及其对应的json都在/home/user/data/目录下，其中json目录为/home/user/data/mllm_format_llava_instruct_data.json，此时配置如下：
`dataset_dir`配置为/home/user/data/
`dataset`配置为/home/user/data/mllm_format_llava_instruct_data.json

**注意：配置`data`参数下的`cache_dir`目录时，如果是多机运行，不要配置成同一个挂载目录，避免写入同一个文件导致冲突**。

【模块冻结配置】（可选）

当前支持自定义冻结模块，在`examples/qwen3vl/qwen3vl_30B_config_v1.yaml`中model->freeze字段中配置需要冻结的模块即可实现相应模块冻结。

【模型保存加载及日志信息配置】（可选）

根据实际情况配置`examples/qwen3vl/qwen3vl_30B_config_v1.yaml`的`training`参数，包括保存路径以及保存间隔`save`、`save_interval`。
注意：`save`参数当前被注释，不会保存模型，如果需要保存模型请配置保存路径。

【单机运行配置】（根据实际情况选择单机/多机）
以Qwen3-VL-30B模型为例：
配置`examples/qwen3vl/finetune_qwen3vl_30B_v1.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
```

【多机运行配置】（根据实际情况选择单机/多机） 
如需拉起多机训练，修改启动脚本下 MASTER_ADDR、NODE_ADDR、NODES以及NODE_RANK变量

``` shell
MASTER_ADDR: 主节点IP地址
NODE_ADDR: 本机IP地址
NODE_RANK: 第几个节点
NNODES: 一共几个节点
```

---

<a id="jump4.3"></a>

### 3. 启动微调

loss计算方式差异会对训练效果造成不同的影响，在启动训练任务之前，请查看关于loss计算的文档，根据`FSDP2后端`选择合适的loss计算方式[vlm_model_loss_calculate_type.md](../../docs/zh/features/vlm_model_loss_calculate_type.md)，在`examples/qwen3vl/qwen3vl_30B_config_v1.yaml`配置`loss_type`参数，可以设置成default（默认）、per_sample_loss、per_token_loss这3个值。

在代码仓根目录（MindSpeed-MM）下执行以下命令启动微调任务：

```shell
bash examples/qwen3vl/finetune_qwen3vl_30B_v1.sh
```

<a id="jump10"></a>

## 环境变量声明

| 环境变量                      | 描述                                                                 | 取值说明                                                                                         |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                                    | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                              |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                    | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                                     |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                                  | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                          | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征                                            |
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用                                                               |
| `NPUS_PER_NODE`               | 配置一个计算节点上使用的NPU数量                                                  | 整数值（如 `1`, `8` 等）                                                                            |

---
