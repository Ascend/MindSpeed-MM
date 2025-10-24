# 类Qwen3_VL 使用指南

**注意**

本目录下模型为类Qwen3VL模型，在Qwen2.5VL的基础结构上将LLM模块替换成了Qwen3-MoE，正式的Qwen3VL待官方发布后支持。

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
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
- [推理](#jump5)
  - [准备工作](#jump5.1)
  - [启动推理](#jump5.2)
- [视频理解](#jump6)
  - [加载数据集](#jump6.1)
  - [配置参数](#jump6.2)
  - [启动微调](#jump6.3)
- [评测](#jump7)
  - [数据集准备](#jump7.1)
  - [配置参数](#jump7.2)
  - [启动评测](#jump7.3)
- [环境变量声明](#jump8)
- [注意事项](#jump9)

## 版本说明
#### 参考实现
```
url=https://github.com/hiyouga/LLaMA-Factory.git
commit_id=e8c1979
# transformers版本
url=https://github.com/huggingface/transformers.git
commit_id=fa56dcc
```

#### 变更记录

2025.06.05: 参考Qwen3作为语言模块，Qwen2.5ViT或InternViT作为视觉模块，基于Qwen2.5-VL实现

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
mkdir logs
mkdir data
mkdir ckpt
```

<a id="jump1.2"></a>
#### 2. 环境搭建

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch及torch_npu包
# 下载路径参考 https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

# 安装加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
pip install -r requirements.txt
pip3 install -e .
cd ..
# 安装其余依赖库
pip install -e .

```

---
<a id="jump2"></a>
## 权重下载及转换

<a id="jump2.1"></a>
#### 1. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main)；
- 模型地址: [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)；
- 模型地址: [Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/tree/main)；
- 模型地址: [Qwen2.5-VL-72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/tree/main)；
- 模型地址: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)；

 将下载的模型权重分别保存到本地的`ckpt/hf_path/Qwen2-VL-*B-Instruct`和`ckpt/hf_path/Qwen3-30B-A3B`目录下。(*表示对应的尺寸)

<a id="jump2.2"></a>
#### 2. 权重转换(hf2mm)

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。详细用法参考[权重转换工具](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)，该工具目前支持LLM部分权重单独从纯LLM的hf权重中加载，添加`--config.llm_hf_config.hf_dir`。

```bash

# 7b + qwen3_30b_a3b
mm-convert  Qwen3_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen3-VL-30B-A3B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.llm_hf_config.hf_dir "ckpt/hf_path/Qwen3-30B-A3B" \
  --cfg.parallel_config.ep_size 8 \
  --cfg.parallel_config.tp_size 1 \
  --cfg.parallel_config.llm_pp_layers [[48]] \
  --cfg.parallel_config.vit_pp_layers [[32]] 
  

# 其中：
# mm_dir: 转换后保存目录
# hf_config.hf_dir: VLM的huggingface权重目录
# llm_hf_config.hf_dir： LLM的huggingface权重目录，如果添加了该参数，模型中LLM的权重将会从该目录下加载并替换VLM中的LLM权重。
# ep_size： ep并行数量，注意要和微调启动脚本中的配置一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
```

同步修改`examples/qwen3vl_dev/finetune_qwen3_vl_30b_a3b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重 `ckpt/hf_path/Qwen2.5-VL-7B-Instruct`进行区分。

**注意**
如果模型中的LLM权重是单独加载的，那么模型的projector权重将随机初始化，需要在`megatron/training/checkpointing.py`的`load_checkpoint`函数中手动将strict设置为False。

```shell
LOAD_PATH="ckpt/mm_path/Qwen3-VL-30B-A3B"
```

<a id="jump3"></a>
## 数据集准备及处理

<a id="jump3.1"></a>
#### 1. 数据集下载(以coco2017数据集为例)

(1)用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中

(2)获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下;

(3)运行数据转换脚本python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py;

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
当前支持读取多个以`,`（注意不要加空格）分隔的数据集，配置方式为`data.json`中
dataset_param->basic_parameters->dataset
从"./data/mllm_format_llava_instruct_data.json"修改为"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"

同时注意`data.json`中`dataset_param->basic_parameters->max_samples`的配置，会限制数据只读`max_samples`条，这样可以快速验证功能。如果正式训练时，可以把该参数去掉则读取全部的数据。

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

**注意**: 如果运行internvit+qwen3模型，数据准备参考InternVL章节的[数据集准备](https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/internvl2#jump3)。

<a id="jump4"></a>
## 微调

<a id="jump4.1"></a>
#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>
#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

以Qwen3VL为例，`data.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径。

**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2.5-VL-7B-Instruct",
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
**注意**：若运行internvit+qwen3模型，请参考`data_internvit.json`配置相应参数。

【模型保存加载及日志信息配置】

根据实际情况配置`examples/qwen3vl_dev/finetune_qwen3_vl_30b_a3b.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 加载路径
LOAD_PATH="ckpt/mm_path/Qwen3-VL-30B-A3B"
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
    ...
    --log-tps \  # 增加此参数可使能在训练中打印每步语言模块的平均序列长度，并在训练结束后计算每秒吞吐tokens量。
"
```

若需要加载指定迭代次数的权重、优化器等状态，需将加载路径`LOAD_PATH`设置为保存文件夹路径`LOAD_PATH="save_dir"`，并修改`latest_checkpointed_iteration.txt`文件内容为指定迭代次数
(此功能coming soon)

```
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

**注意**：由于internvit+qwen3模型暂不支持权重转换，因此`examples/qwen3vl_dev/finetune_internvit_qwen3_vl.sh`中不设置`--LOAD_PATH`，模型将加载
随机初始化参数。

【单机运行配置】

配置`examples/qwen3vl_dev/finetune_qwen3_vl_30b_a3b.sh`参数如下

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
多机运行用户请根据使用机器的情况，相应修改`NNODES`,`NPUS_PER_NODE` 配置.

**注意**

- 当开启PP时，`model.json`中配置的`vision_encoder`和`text_decoder`的`pipeline_num_layer`参数控制了各自的PP切分策略。对于流水线并行，要先处理`vision_encoder`再处理`text_decoder`。
比如7b默认的值`[32,0,0,0]`、`[1,10,10,7]`，其含义为PP域内第一张卡先放32层`vision_encoder`再放1层`text_decoder`、第二张卡放`text_decoder`接着的10层、第三张卡放`text_decoder`接着的10层、第四张卡放`text_decoder`接着的7层，`vision_encoder`没有放完时不能先放`text_decoder`（比如`[30,2,0,0]`、`[1,10,10,7]`的配置是错的）

- 如果某张卡上的参数全部冻结时会导致没有梯度（比如`vision_encoder`冻结时PP配置`[30,2,0,0]`、`[0,11,10,7]`），需要在`finetune_qwen2_5_vl_7b.sh`中`GPT_ARGS`参数中增加`--enable-dummy-optimizer`，参考[dummy_optimizer特性文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/dummy_optimizer.md)。

<a id="jump4.3"></a>
#### 3. 启动微调

以Qwen3-30B-A3B为例，启动微调训练任务。

```shell
bash examples/qwen3vl_dev/finetune_qwen3_vl_30b_a3b.sh
```

---
<a id="jump5"></a>
## 推理

Coming soon...

<a id="jump7"></a>
## 评测

Coming soon...

<a id="jump8"></a>
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
<a id="jump9"></a>
## 注意事项

1. 在 `finetune_xx.sh`里，与模型结构相关的参数并不生效，以`examples/qwen2.5vl/model_xb.json`里同名参数配置为准，非模型结构的训练相关参数在 `finetune_xx.sh`修改。
2. 在使用单卡进行3B模型训练时，如果出现Out Of Memory，可以使用多卡并开启分布式优化器进行训练。
3. `model.json`设置use_remove_padding为true时，在`examples/qwen2vl/dot_product_attention.py`中，attention_mask形状当前固定为[2048, 2048]，如需更改请参考[昇腾官网FlashAttentionScore](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0027.html)的替换指南