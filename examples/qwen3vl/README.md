# Qwen3_VL 使用指南

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
commit_id=c0dbe09
```

#### 变更记录

2025.09.28: 首次支持Qwen3-VL模型

---
<a id="jump1"></a>
## 环境安装

<a id="jump1.1"></a>
#### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)，完成昇腾软件安装。
> Python版本推荐3.10，torch和torch_npu版本推荐2.7.1版本

‼️MoE部分的加速特性依赖较新版本的torch_npu和CANN，推荐使用以下版本
- [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)
- [torch_npu](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html)

<a id="jump1.2"></a>
#### 2. 环境搭建
拉取MindSpeed MM代码仓，并进入代码仓根目录：
```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```
对于X86架构机器，执行如下指令：
```bash
bash scripts/install.sh --arch x86 --msid 93c45456c7044bacddebc5072316c01006c938f9&& pip install -r examples/qwen3vl/requirements.txt
```
对于ARM架构机器，执行如下指令：
```bash
bash scripts/install.sh --arch arm --msid 93c45456c7044bacddebc5072316c01006c938f9&& pip install -r examples/qwen3vl/requirements.txt
```

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Qwen3-VL-*B](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe)；

 将下载的模型权重保存到本地的`ckpt/hf_path/Qwen3-VL-*B-Instruct`目录下。(*表示对应的尺寸)

如果使用fsdp2的meta init初始化模型，需要先完成以下权重转换
```bash
mm-convert Qwen3VLConverter hf_to_dcp \
  --hf_dir Qwen3-VL-xxB \
  --dcp_dir Qwen3-VL-xxB-dcp \
```
并在examples/qwen3vl/finetune_qwen3vl.sh的`GPT_ARGS`中加入`--init-model-with-meta-device`参数。

注意，针对Qwen3VL-30B和Qwen3VL-235B模型，必须使用meta init初始化加载权重。

---
<a id="jump3"></a>
## 数据集准备及处理

<a id="jump3.1"></a>
#### 1. 数据集下载(以coco2017数据集为例)

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
当前支持读取多个以`,`（注意不要加空格）分隔的数据集，配置方式为`data_xxB.json`中
dataset_param->basic_parameters->dataset
从"./data/mllm_format_llava_instruct_data.json"修改为"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"

同时注意`data_xxB.json`中`dataset_param->basic_parameters->max_samples`的配置，会限制数据只读`max_samples`条，这样可以快速验证功能。如果正式训练时，可以把该参数去掉则读取全部的数据。

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

根据实际情况修改`data_xxB.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

示例：如果数据及其对应的json都在/home/user/data/目录下，其中json目录为/home/user/data/video_data_path.json，此时配置如下：
`dataset_dir`配置为/home/user/data/;
`dataset`配置为./data/video_data_path.json
注意此时`dataset`需要配置为相对路径

以Qwen3VL-xxB为例，`data_xxB.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径,即原始hf权重路径。

**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen3-VL-xxB-Instruct",
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

如果需要加载大批量数据，可使用流式加载，修改`data_xxB.json`中的`sampler_type`字段，增加`streaming`字段。（注意：使用流式加载后当前仅支持`num_workers=0`，单进程处理数据，会有性能波动，并且不支持断点续训功能。）


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

【MoE 加速配置】

开启MoE融合可以提升模型训练性能，开启方式为将`model.json`文件中修改`use_npu_fused_moe`字段为`true`

注意：FusedMoE特性依赖较新版本，参考【环境安装】章节进行环境安装。

【序列并行配置】

当前已支持Ulysses序列并行，当使用长序列训练时，需要开启CP特性，开启方式为在训练bash脚本设置CP > 1，例如
```bash
CP=4
GPT_ARGS="
    ...
    --context-parallel-size ${CP} \
    ...
"
```

【Attention配置】

当前支持vision和text模块选择不同的Attntion实现方式，具体为在`model.json`文件中修改`attn_implementation`字段，当前支持`eager`、`flash_attention_2`、`sdpa`三种配置。

注意：当sdpa和FusedMoE一起使用时，可能会存在显存未及时释放导致OOM的问题，可以开启`flash_attention_2`缓解多流复用带来显存未及时释放问题，降低部分显存使用。

【activation_offload配置】
使用activation_offload可以将重计算过程中产生的checkpoint点的激活值移动到host，反向异步从host传输到device，降低device激活显存占用，配置方式为在`model.json`中将`activation_offload`字段设置为True。

【chunkloss 配置】
参考[chunk loss文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/chunkloss.md)

【模型保存加载及日志信息配置】

根据实际情况配置`examples/qwen3vl/finetune_qwen3vl_xxB.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 断点续训权重加载路径
LOAD_PATH="./ckpt/save_dir/Qwen3-VL-xxB-Instruct"
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

根据实际情况配置`examples/qwen3vl/model_xxB.json`中的`init_from_hf_path`参数，该参数表示初始权重的加载路径。
根据实际情况配置`examples/qwen3vl/model_xxB.json`中的`image_encoder.vision_encoder.freeze`、`image_encoder.vision_projector.freeze`、`text_decoder.freeze`参数，该参数分别代表是否冻结vision model模块、projector模块、及language model模块。
注：当前`examples/qwen3vl/model_xxB.json`中的各网络层数均为未过校验的无效配置，如需减层请修改原始hf路径下相关配置文件。

【单机运行配置】

配置`examples/qwen3vl/finetune_qwen3vl_xxB.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>
#### 3. 启动微调

以Qwen3VL-xxB为例，启动微调训练任务。  
loss计算方式差异会对训练效果造成不同的影响，在启动训练任务之前，请查看关于loss计算的文档，选择合适的loss计算方式[vlm_model_loss_calculate_type.md](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/vlm_model_loss_calculate_type.md)

```shell
bash examples/qwen3vl/finetune_qwen3vl_xxB.sh
```
**优化特性：**

- ChunkLoss：可以参考文档[ChunkLoss](docs/features/chunkloss.md)开启该特性优化长序列时的显存占用。

---

<a id="jump4.4"></a>

#### 4. 启动推理
训练完成之后，以Qwen3VL-xxB为例，将保存在`save_dir`目录下的权重转换成huggingface格式
```shell
mm-convert Qwen3VLConverter dcp_to_hf \
  --load_dir save_dir/iter_000xx/ \
  --save_dir save_dir/iter_000xx_hf/ \
  --model_assets_dir ./ckpt/Qwen3-VL-xxB-Instruct \
```
其中，`iter_000xx`表示保存的第xx步的权重，`--save_dir`表示转换后的权重保存路径，`--model_assets_dir`原始huggingface权重的路径。

完成权重转换之后，即可使用transformers库进行推理。

【多机运行配置】

如需拉起多机训练，修改启动脚本下 MASTER_ADDR、NODE_ADDR、NODES以及NODE_RANK变量
``` shell
MASTER_ADDR: 主节点IP地址
NODE_ADDR: 本机IP地址
NODE_RANK: 第几个节点
NODES: 一共几个节点
```

---



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
