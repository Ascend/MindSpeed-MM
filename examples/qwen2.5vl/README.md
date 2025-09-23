# Qwen2_5_VL 使用指南

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
  - [权重转换hf2mm](#jump2.2)
  - [权重转换mm2hf](#jump2.3)
  - [权重重切分](#jump2.4)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
  - [混合数据集处理](#jump3.2)
- [微调](#jump4)
  - [长序列支持](#长序列支持)
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
commit_id=52f2565
# transformers版本
url=https://github.com/huggingface/transformers.git
commit_id=fa56dcc
```

#### 变更记录

2025.03.26: 首次支持Qwen2.5-VL模型
2025.05.29：同步开源仓数据处理修改

---
<a id="jump1"></a>
## 环境安装

<a id="jump1.1"></a>
#### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)，完成昇腾软件安装。

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
# checkout commit from MindSpeed core_r0.12.1
git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
# 安装mindspeed及依赖
pip install -e .
cd ..
# 安装mindspeed mm及依赖
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


 将下载的模型权重保存到本地的`ckpt/hf_path/Qwen2.5-VL-7B-Instruct`目录下。

<a id="jump2.2"></a>
#### 2. 权重转换(hf2mm)

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的互相转换以及PP（Pipeline Parallel）权重的重切分。参考[权重转换工具](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)

```bash
# 3b
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[36]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 1
  
# 7b
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[12,16]] \
  --cfg.parallel_config.vit_pp_layers [[32,0]] \
  --cfg.parallel_config.tp_size 1

# 32b
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-32B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-32B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[1,9,9,9,9,9,9,9]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0,0,0,0,0]] \
  --cfg.parallel_config.tp_size 2

# 72b
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-72B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-72B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[6,11,11,11,11,11,11,8]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0,0,0,0,0]] \
  --cfg.parallel_config.tp_size 2

# 其中：
# mm_dir: 转换后保存目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
```

<a id="jump2.3"></a>
#### 3. 权重转换(mm2hf)

MindSpeed-MM修改了部分原始网络的结构名称，在微调后，如果需要将权重转回huggingface格式，可使用`mm-convert`权重转换工具对微调后的权重进行转换，将权重名称修改为与原始网络一致。

```bash
mm-convert  Qwen2_5_VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-7B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 1
# 其中：
# save_hf_dir: mm微调后转换回hf模型格式的目录
# mm_dir: 微调后保存的权重目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
```
如果需要用转换后模型训练的话，同步修改`examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重 `ckpt/hf_path/Qwen2.5-VL-7B-Instruct`进行区分。

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
```

<a id="jump2.4"></a>
#### 4. 训练后重新切分权重

权重下载及转换部分会把权重进行pp切分和tp切分，在微调后，如果需要对权重重新进行切分，可使用`mm-convert`权重转换工具对微调后的权重进行切分。
注意：当前还不支持VPP切分。

```bash
mm-convert  Qwen2_5_VLConverter resplit \
  --cfg.source_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.target_dir "ckpt/mm_resplit_pp/Qwen2.5-VL-7B-Instruct" \
  --cfg.source_parallel_config.llm_pp_layers [12,16] \
  --cfg.source_parallel_config.vit_pp_layers [32,0] \
  --cfg.source_parallel_config.tp_size 1 \
  --cfg.target_parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.target_parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.target_parallel_config.tp_size 2
# 其中
# source_dir: 微调后保存的权重目录
# target_dir: 希望重新pp切分后保存的目录
# source_parallel_config.llm_pp_layers: 微调时llm的pp配置
# source_parallel_config.vit_pp_layers: 微调时vit的pp配置
# source_parallel_config.tp_size: 微调时tp并行配置
# target_parallel_config.llm_pp_layers: 期望的重切分llm模块切分层数
# target_parallel_config.vit_pp_layers: 期望的重切分vit模块切分层数
# target_parallel_config.tp_size: 期望的tp并行配置（tp_size不能超过原仓config.json中的num_key_value_heads）
```

---
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

<a id="jump4"></a>
## 微调

#### 长序列支持

在多模态理解任务中，当训练数据存在长视频或高分辨率多图时，训练任务可能会因为序列长度过长导致显存占用过多、默认切分配置不适用，此处提供长序列场景的训练支持，下方提供长序列需修改的配置（以下方首条训练配置为例）：

将`finetune_qwen2_5_vl_72b.sh`中的`--swap-attention \`去除、`TP=2`改为`TP=8`、`PP=8`改为`PP=4`、`CP=1`改为`CP=4`、`GRAD_ACC_STEP=96`改为`GRAD_ACC_STEP=1`、`--seq-length 1024`改为`--seq-length 131072`、`--context-parallel-algo ulysses_cp_algo`改为`--context-parallel-algo megatron_cp_algo`；

将`data_72b.json`中的`"video_max_pixels": 16384`改为`"video_max_pixels": 262144`、`"video_fps": 2.0`改为`"video_fps": 60.0`、`"video_maxlen": 64`改为`"video_maxlen": 768`、`"images": "images"`改为`"images": null`、`"videos": null`改为`"videos": "videos"`；

将`model_72b.json`中的`"pipeline_num_layers": [32, 0, 0, 0, 0, 0, 0, 0]`改为`"pipeline_num_layers": [32, 0, 0, 0]`、`"pipeline_num_layers": [6, 11, 11, 11, 11, 11, 11, 8]`改为`"pipeline_num_layers": [6, 25, 25, 24]`、`"max_position_embeddings": 128000`改为`max_position_embeddings": 131072`并在下方加入`"recompute_granularity": "full",`、`"recompute_method": "uniform",`、`"recompute_num_layers": 1,`。

| **训练数据配置** | **模型规模** | **集群规模** | **模型及切分配置** | **性能数据** |
| --------------- | ----------- | ----------- | ----------- | ------------ |
| "video_max_pixels":262144,<br>"video_fps":60.0,<br>"video_maxlen":768,<br>"seq-length":131072 | 72B | 8*8(A3) | TP8<br>PP4(vit pp_layers:[32,0,0,0], llm pp_layers:[6,25,25,24])<br>CP4(context-parallel-algo:megatron_cp_algo)<br>text_decoder full recompute:<br> &ensp; "recompute_granularity": "full",<br> &ensp; "recompute_method": "uniform",<br> &ensp; "recompute_num_layers": 1 |/|

---

<a id="jump4.1"></a>
#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>
#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

以Qwen2.5VL-7B为例，`data.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径。

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

【模型保存加载及日志信息配置】

根据实际情况配置`examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 加载路径
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
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

【单机运行配置】

配置`examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`参数如下

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
注意，当开启PP时，`model.json`中配置的`vision_encoder`和`text_decoder`的`pipeline_num_layer`参数控制了各自的PP切分策略。对于流水线并行，要先处理`vision_encoder`再处理`text_decoder`。
比如7b默认的值`[32,0,0,0]`、`[1,10,10,7]`，其含义为PP域内第一张卡先放32层`vision_encoder`再放1层`text_decoder`、第二张卡放`text_decoder`接着的10层、第三张卡放`text_decoder`接着的10层、第四张卡放`text_decoder`接着的7层，`vision_encoder`没有放完时不能先放`text_decoder`（比如`[30,2,0,0]`、`[1,10,10,7]`的配置是错的）

同时注意，如果某张卡上的参数全部冻结时会导致没有梯度（比如`vision_encoder`冻结时PP配置`[30,2,0,0]`、`[0,11,10,7]`），需要在`finetune_qwen2_5_vl_7b.sh`中`GPT_ARGS`参数中增加`--enable-dummy-optimizer`，参考[dummy_optimizer特性文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/dummy_optimizer.md)。

【vit模块重计算配置（可选）】

当放开vit训练时（默认配置中冻结vit，若要放开请将model.json文件中`vision_encoder`部分配置为`"vision_encoder": {"freeze": false}`。），可以启用重计算以降低显存（注意，此举会对性能产生影响）

若要开启vit重计算，需在model.json中的vision_encoder部分添加重计算相关参数。
通过`recompute_granularity`参数可以配置重计算模块为`full`或`selective`。

1. full模式

TransformerLayer中的所有组件（layernorm、attention、mlp）都进行重计算，此时可以配置重计算的层数。
- `recompute_method`: 控制重计算层数计算的方法，可选值为`uniform`（均匀重计算）或`block`（按块重计算）。
- `recompute_num_layers`: 控制重计算的层数，指定需要重计算的层数量。

示例配置如下：

```json
{
  "model_id": "qwen2_5vl",
  "img_context_token_id": 151655,
  "vision_start_token_id": 151652,
  "image_encoder": {
    "vision_encoder": {
      "recompute_granularity": "full",
      "recompute_method": "uniform",
      "recompute_num_layers": 1
    }
  }
}
```
2. selective模式

仅对TransformerLayer中attention的core_attention组件进行重计算。

示例配置如下：

```json
{
  "model_id": "qwen2_5vl",
  "img_context_token_id": 151655,
  "vision_start_token_id": 151652,
  "image_encoder": {
    "vision_encoder": {
      "recompute_granularity": "selective"
    }
  }
}
```


<a id="jump4.3"></a>
#### 3. 启动微调

以Qwen2.5VL-7B为例，启动微调训练任务。

```shell
bash examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
```

---
<a id="jump5"></a>
## 推理

<a id="jump5.1"></a>
#### 1、配置参数

根据实际情况修改examples/qwen2.5vl/inference_qwen2_5_vl_7b.json和examples/qwen2.5vl/inference_qwen2_5_vl_7b.sh中的路径配置，包括tokenizer的加载路径from_pretrained。需注意

（1）tokenizer/from_pretrained配置的路径为从huggingface下载的原始Qwen2.5-VL-7B-Instruct路径。

（2）shell文件中的LOAD_PATH的路径为经过权重转换后的模型路径(可PP切分)。

<a id="jump5.2"></a>
#### 2、启动推理

```shell
bash examples/qwen2.5vl/inference_qwen2_5_vl_7b.sh
```

---
<a id="jump6"></a>
## Qwen2.5vl支持视频理解

<a id="jump6.1"></a>
### 1、加载视频数据集

数据集中的视频数据集取自llamafactory，https://github.com/hiyouga/LLaMA-Factory/tree/main/data

视频取自mllm_demo_data，使用时需要将该数据放到自己的data文件夹中去，同时将llamafactory上的mllm_video_demo.json也放到自己的data文件中

之后根据实际情况修改 `data.json` 中的数据集路径，包括 `model_name_or_path` 、 `dataset_dir` 、 `dataset` 字段，并修改"attr"中  `images` 、 `videos` 字段，修改结果参考下图。

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./Qwen2.5-VL-7B-Instruct",
            ...
        },
        "basic_parameters": {
            ...
            "dataset_dir": "./data",
            "dataset": "./data/mllm_video_demo.json",
            "cache_dir": "./data/cache_dir",
            ...
        },
        ...
        "attr": {
            "system": null,
            "images": null,
            "videos": "videos",
            ...
        },
    },
    ...
}
```

<a id="jump6.2"></a>
### 2、修改模型配置

在model_xxx.json中，修改`img_context_token_id`为下图所示：
```
"img_context_token_id": 151656
```
注意， `image_token_id` 和 `img_context_token_id`两个参数作用不一样。前者是固定的，是标识图片的 token ID，在qwen2_5_vl_get_rope_index中用于计算图文输入情况下序列中的图片数量。后者是标识视觉内容的 token ID，用于在forward中标记视觉token的位置，所以需要根据输入做相应修改。

<a id="jump6.3"></a>
### 3、启动微调
以Qwen2.5VL-7B为例，启动微调训练任务。

```shell
bash examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
```

---
<a id="jump7"></a>
## 评测

<a id="jump7.1"></a>
### 数据集准备

当前模型支持AI2D(test)、ChartQA(test)、Docvqa(val)、MMMU(val)四种数据集的评测。
数据集参考下载链接：

- [MMMU_DEV_VAL](https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv)
- [DocVQA_VAL](https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv)
- [AI2D_TEST](https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv)
- [ChartQA_TEST](https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv)

<a id="jump7.2"></a>
### 参数配置

如果要进行评测需要将要评测的数据集名称和路径传到examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.json
需要更改的字段有

- `tokenizer`中的`from_pretrained`为huggingface的Qwen2.5-VL的权重，参考readme上面链接自行下载传入
- `dataset_path`为上述评测数据集的本地路径
- `evaluation_dataset`为评测数据集的名称可选的名称有(`ai2d_test`、`mmmu_dev_val`、`docvqa_val`、`chartqa_test`)， **注意**：需要与上面的数据集路径相对应。
- `result_output_path`为评测结果的输出路径，**注意**：每次评测前需要将之前保存在该路径下评测文件删除。

```json
    "tokenizer": {
        "from_pretrained": "./Qwen2.5-VL-7B-Instruct",

    },
    "dataset_path": "./AI2D_TEST.tsv",
    "evaluation_dataset":"ai2d_test",
    "evaluation_model":"qwen2_vl_7b",
    "result_output_path":"./evaluation_outputs/"

```

examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.json改完后，需要将json文件的路径传入到examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh MM_MODEL字段中。

以及需要将上面提到的权重转换后模型传入examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh中的LOAD_PATH字段中。

```shell
MM_MODEL=examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.json
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"

```
评测支持多卡DP评测需要更改的配置,为NPU卡数量

```shell
NPUS_PER_NODE=8
```

<a id="jump7.3"></a>
### 启动评测
评测额外依赖一些python包，使用下面命令进行安装

```shell
pip install -e ".[evaluate]"
```

<a id="jump7.4"></a>
启动shell开始评测
```shell
# 在MindSpeed-MM目录下执行
bash examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh
```

评测结果会输出到`result_output_path`路径中，会输出结果文件：

- *.xlsx文件，这个文件会输出每道题的预测结果和答案等详细信息。
- *.csv文件，这个文件会输出统计准确率等数据。

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