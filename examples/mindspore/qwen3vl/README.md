# Qwen3_VL 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#jump1)
  - [仓库拉取及环境搭建](#jump1.1)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
  - [权重转换hf2mm](#jump2.2)
  - [权重转换mm2hf](#jump2.3)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
  - [混合数据集处理](#jump3.2) 
- [微调](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动微调](#jump4.3)
- [环境变量声明](#jump5)

---
<a id="jump1"></a>
## 环境安装

MindSpeed-MM MindSpore后端的依赖配套如下表，安装步骤参考[基础安装指导](../../../docs/mindspore/install_guide.md)。

| 依赖软件         |                                                              |
| ---------------- | ------------------------------------------------------------ |
| 昇腾NPU驱动固件  | 在研版本 |
| 昇腾 CANN        | 在研版本 |
| MindSpore        | [2.7.2](https://www.mindspore.cn/install/)         |
| Python           | >=3.9                                                        |                                          |
|transformers     |      [v4.57.0](https://github.com/huggingface/transformers/tree/v4.57.0)    |    |

<a id="jump1.1"></a>
### 1. 仓库拉取及环境搭建

针对MindSpeed MindSpore后端，昇腾社区提供了模型一键拉起部署MindSpeed-Core-MS，旨在帮助用户自动拉取相关代码仓并对torch代码进行一键适配，进而使用户无需再额外手动开发适配即可在华为MindSpore+CANN环境下一键拉起模型训练。在进行一键拉起前，用户需要拉取相关的代码仓以及进行环境搭建：


```shell
# 创建conda环境
conda create -n test python=3.10
conda activate test

# 使用环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# 安装MindSpeed-Core-MS一键拉起部署
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.5.0

# 使用MindSpeed-Core-MS内部脚本自动拉取相关代码仓并一键适配
cd MindSpeed-Core-MS
pip install -r requirements.txt 
source auto_convert.sh mm

# 安装新版transformers（支持qwen3vl模型）
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout c0dbe09
pip install -e .

mkdir ckpt
mkdir data
mkdir logs
```
---
<a id="jump2"></a>
## 权重下载及转换

<a id="jump2.1"></a>
### 1. 权重下载

从Hugging Face库下载对应的模型权重:

- 模型地址: [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/main)；

- 模型地址: [Qwen3-VL-30B](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main)；


 将下载的模型权重保存到本地的`ckpt/hf_path/Qwen3-VL-*B-Instruct`目录下(*表示对应的尺寸)。

<a id="jump2.2"></a>
#### 2. 权重转换(hf2mm)

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的互相转换以及PP（Pipeline Parallel）权重的重切分。参考[权重转换工具](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)了解该工具的具体使用。**注意当前在MindSpore后端下，转换出的权重无法用于Torch后端的训练**。

> 注：基于mindspore后端执行权重转换时,`mm-convert`执行的脚本为[convert_cli.py](../checkpoint/convert_cli.py)

```bash
  
# 8b
mm-convert  Qwen3VLMegatronConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen3-VL-8B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen3-VL-8B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[6,10,10,10]] \
  --cfg.parallel_config.vit_pp_layers [[27,0,0,0]] \
  --cfg.parallel_config.tp_size 1

# 30b
mm-convert  Qwen3VLMegatronConverter hf_to_mm \
--cfg.mm_dir "ckpt/mm_path/Qwen3-VL-30B" \
--cfg.hf_config.hf_dir "ckpt/hf_path/Qwen3-VL-30B-Instruct" \
--cfg.parallel_config.llm_pp_layers [[6,14,14,14]] \
--cfg.parallel_config.vit_pp_layers [[27,0,0,0]] \
--cfg.parallel_config.tp_size 1 \
--cfg.parallel_config.ep_size 1

# 其中：
# mm_dir: 转换后保存目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
# ep_size: 专家并行数量，注意要和微调启动脚本中的配置一致
```

如果需要用转换后模型训练的话，同步修改`examples/mindspore/qwen3vl/finetune_qwen3vl_*B.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重 `ckpt/hf_path/Qwen3-VL-xxB-Instruct`进行区分。

```shell
LOAD_PATH="ckpt/mm_path/Qwen3-VL-xxB"
```

<a id="jump2.3"></a>
#### 3. 权重转换(mm2hf)

MindSpeed MM修改了部分原始网络的结构名称，在微调后，如果需要将权重转回huggingface格式，可使用`mm-convert`权重转换工具对微调后的权重进行转换，将权重名称修改为与原始网络一致。

```bash
# 8b
mm-convert  Qwen3VLMegatronConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen3-VL-8B" \
  --cfg.mm_dir "ckpt/mm_path/Qwen3-VL-8B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen3-VL-8B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [6,10,10,10] \
  --cfg.parallel_config.vit_pp_layers [27,0,0,0] \
  --cfg.parallel_config.tp_size 1

# 30b
mm-convert  Qwen3VLMegatronConverter mm_to_hf \
--cfg.save_hf_dir "ckpt/mm_to_hf/Qwen3-VL-30B" \
--cfg.mm_dir "ckpt/mm_path/Qwen3-VL-30B" \
--cfg.hf_config.hf_dir "ckpt/hf_path/Qwen3-VL-30B-Instruct" \
--cfg.parallel_config.llm_pp_layers [6,14,14,14] \
--cfg.parallel_config.vit_pp_layers [27,0,0,0] \
--cfg.parallel_config.tp_size 1 \
--cfg.parallel_config.ep_size 1
# 其中：
# save_hf_dir: mm微调后转换回hf模型格式的目录
# mm_dir: 微调后保存的权重目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
# ep_size: 专家并行数量，注意要和微调启动脚本中的配置一致
```

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

【模型保存加载及日志信息配置】

根据实际情况配置`examples/mindspore/qwen3vl/finetune_qwen3vl_xxB.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

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

根据实际情况配置`examples/mindspore/qwen3vl/model_xxB.json`中的`init_from_hf_path`参数，该参数表示初始权重的加载路径。
根据实际情况配置`examples/mindspore/qwen3vl/model_xxB.json`中的`image_encoder.vision_encoder.freeze`、`image_encoder.vision_projector.freeze`、`text_decoder.freeze`参数，该参数分别代表是否冻结vision model模块、projector模块、及language model模块。
注：当前`examples/mindspore/qwen3vl/model_xxB.json`中点各网络层数均为未过校验的无效配置，如需减层请修改原始hf路径下相关配置文件。

【单机运行配置】

配置`examples/mindspore/qwen3vl/finetune_qwen3vl_xxB.sh`参数如下

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
bash examples/mindspore/qwen3vl/finetune_qwen3vl_xxB.sh
```
---

<a id="jump5"></a>
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
