# Glm4.5v 使用指南

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
  - [混合数据集处理](#jump3.2)
- [微调](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动微调](#jump4.3)
  - [启动推理](#jump4.4)
- [环境变量声明](#jump10)
- [注意事项](#jump11)

## 版本说明
#### 参考实现
```
url=https://github.com/huggingface/transformers.git
commit_id=8cb5963
```

#### 变更记录

2025.11.29: 首次支持Glm4.5v模型

---
<a id="jump1"></a>
## 环境安装

<a id="jump1.1"></a>
#### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md)，完成昇腾软件安装。
> Python版本推荐3.10，torch和torch_npu版本推荐2.7.1版本

‼️MoE部分的加速特性依赖较新版本的torch_npu和CANN，推荐使用以下版本
- [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)
- [torch_npu](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/insg_0004.html)

<a id="jump1.2"></a>
#### 2. 环境搭建
```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
mkdir logs data ckpt

# 安装加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout d76dbddd4517d48a2fc1cd494de8b9a6cfdbfbab

# 安装mindspeed及依赖
pip install -e .
cd ..
# 安装mindspeed mm及依赖
pip install -e .

# 安装新版transformers（支持glm4.5v模型）
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 8cb5963
pip install -e .
```

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Hugging Face库下载对应的模型权重:

- 模型地址: [GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V)；

 将下载的模型权重保存到本地的`ckpt/hf_path/GLM-4.5V`目录下。(*表示对应的尺寸)

为了使网络能够流畅运行，MindSpeed MM修改了moe中专家的结构名称，需对原始预训练权重进行转换：
```bash
mm-convert ExpertMergeDcpConverter hf_to_dcp --hf_dir "ckpt/hf_path/GLM-4.5V" --save_dir "ckpt/mm_path/GLM-4.5V"
```
由于glm4.5v参数量较大, 必须使用meta init初始化加载权重；
需要在examples/glm4.5v/finetune_glm4_5v106B.sh的`GPT_ARGS`中加入`--init-model-with-meta-device`参数. 默认脚本中已添加。
此外，使用meta init初始化加载权重时，需要将examples/glm4.5v/finetune_glm4_5v106B.sh中LOAD_PATH设置为上述权重转换完后保存的路径"ckpt/mm_path/GLM-4.5V"。

---
<a id="jump3"></a>
## 数据集准备及处理

<a id="jump3.1"></a>
#### 1. 数据集下载（以COCO2017数据集为例）

(1)用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中。

(2)获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下。

(3)运行数据转换脚本python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py，转换后参考数据目录结构如下：

   ```
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

---
当前支持读取多个以`,`（注意不要加空格）分隔的数据集，配置方式为`data_106B.json`中
dataset_param->basic_parameters->dataset
从"./data/mllm_format_llava_instruct_data.json"修改为"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"

同时注意`data_106B.json`中`dataset_param->basic_parameters->max_samples`的配置，会限制数据只读`max_samples`条，这样可以快速验证功能。如果正式训练时，可以把该参数去掉则读取全部的数据。

<a id="jump3.2"></a>
#### 2.纯文本或有图无图混合训练数据(以LLaVA-Instruct-150K为例)

现在本框架已经支持纯文本/混合数据（有图像和无图像数据混合训练）。

在数据构造时，对于包含图片的数据，需要保留`image`这个键值。

```python
{
  "id": your_id,
  "image": your_image_path,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

在数据构造时，对于纯文本数据，可以去除`image`这个键值。

```python
{
  "id": your_id,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

<a id="jump4"></a>
## 微调

<a id="jump4.1"></a>
#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>
#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data_106B.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

示例：如果数据及其对应的json都在/home/user/data/目录下，其中json目录为/home/user/data/video_data_path.json，此时配置如下：
`dataset_dir`配置为/home/user/data/;
`dataset`配置为./data/video_data_path.json
注意此时`dataset`需要配置为相对路径

修改示例如下，注意`model_name_or_path`的权重路径为转换前的权重路径,即原始hf权重路径。

**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/GLM-4.5V",
            ...
        },
        "basic_parameters": {
            ...
            "dataset_dir": "./data",
            "dataset": "./data/mllm_format_llava_instruct_data.json",
            "cache_dir": "./data/cache_dir",
            ...
        },
        ...
    },
    ...
}
```

如果需要加载大批量数据，可使用流式加载，修改`data_106B.json`中的`sampler_type`字段，增加`streaming`字段。（注意：使用流式加载后当前仅支持`num_workers=0`，单进程处理数据，会有性能波动，并且不支持断点续训功能。）


```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            ...
            "streaming": true
            ...
        },
        ...
    },
    "dataloader_param": {
        ...
        "sampler_type": "stateful_distributed_sampler",
        ...
    }
}

```
【模块冻结配置】

当前支持vison encoder、vision projector、text decoder及lm head模块的冻结，其中，vison encoder、vision projector默认训练时为冻结状态，

通过配置`model.json`文件中各个模块的`freeze`字段，来修改各个模块的冻结与否。

【模型保存加载及日志信息配置】

根据实际情况配置`examples/glm4.5v/finetune_glm4_5v_106B.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 断点续训权重加载路径
LOAD_PATH="./ckpt/save_dir/GLM-4.5V"
# 保存路径
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # 不加载优化器状态，若需加载请移除
    --no-load-rng \  # 不加载随机数状态，若需加载请移除
    --no-save-optim \  # 不保存优化器状态，若需保存请移除
    --no-save-rng \  # 不保存随机数状态，若需保存请移除
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # 日志间隔
    --save-interval 5000 \  # 保存间隔
    --save $SAVE_PATH \ # 保存路径
"
```

根据实际情况配置`examples/glm4.5v/model_106B.json`中的`init_from_hf_path`参数，该参数表示初始权重的加载路径。
根据实际情况配置`examples/glm4.5v/model_106B.json`中的`image_encoder.vision_encoder.freeze`、`image_encoder.vision_projector.freeze`、`text_decoder.freeze`参数，该参数分别代表是否冻结vision model模块、projector模块、及language model模块。
注：当前`examples/glm4.5v/model_106B.json`中的各网络层数均为未过校验的无效配置，如需减层请修改原始hf路径下相关配置文件。
为了在NPU上更快得运行模型训练，moe模块可使能NPU亲和的融合算子，在`examples/glm4.5v/model_106B.json`中配置"use_npu_fused_moe"为true即可，此处默认配置为true。

【单机运行配置】
注意：单机只能跑减层，可作为模型调试用；完整模型运行请使用多机配置。
配置`examples/glm4.5v/model_106B.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/cann/set_env.sh
NPUS_PER_NODE=16  # 单机减层可跑
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

【多机运行配置】

配置`examples/glm4.5v/model_106B.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/cann/set_env.sh
# 根据分布式集群实际情况配置分布式参数
NPUS_PER_NODE=16  # 每个节点的卡数，以实际情况填写
MASTER_ADDR="your master node IP"  # 都需要修改为主节点的IP地址（不能为localhost）
MASTER_PORT=6000
NNODES=8  # 集群里的节点数，以实际情况填写
NODE_RANK="current node id"  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>
#### 3. 启动微调

启动微调训练任务需要确认loss计算方式。  
loss计算方式差异会对训练效果造成不同的影响，在启动训练任务之前，请查看关于loss计算的文档，选择合适的loss计算方式[vlm_model_loss_calculate_type.md](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/features/vlm_model_loss_calculate_type.md)

```shell
bash examples/glm4.5v/model_106B.sh
```

<a id="jump4.4"></a>

#### 4. hf权重转换
训练完成之后，将保存在`save_dir`目录下的权重转换成huggingface格式
```shell
mm-convert ExpertMergeDcpConverter dcp_to_hf --hf_dir "ckpt/hf_path/GLM-4.5V" --dcp_dir "save_dir/iter_000xx" --save_dir "ckpt/dcp_to_hf/GLM-4.5V"
```
其中，`--hf_dir`表示原始huggingface权重的路径，`--dcp_dir`表示微调后的权重保存路径，`iter_000xx`表示保存的第xx步的权重，`--save_dir`表示转换后的权重保存路径。

完成权重转换之后，即可使用相关库进行推理。

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
