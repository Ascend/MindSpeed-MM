# Qwen2.5VL（MindSpore后端）使用指南

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

---
<a id="jump1"></a>
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

针对MindSpeed MindSpore后端，昇腾社区提供了一键转换工具MindSpeed-Core-MS，旨在帮助用户自动拉取相关代码仓并对torch代码进行一键适配，进而使用户无需再额外手动开发适配即可在华为MindSpore+CANN环境下一键拉起模型训练。在进行一键转换前，用户需要拉取相关的代码仓以及进行环境搭建：


```
# 创建conda环境
conda create -n test python=3.10
conda activate test

# 使用环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# 安装MindSpeed-Core-MS转换工具
git clone https://gitee.com/ascend/MindSpeed-Core-MS.git -b r0.3.0

# 使用MindSpeed-Core-MS内部脚本自动拉取相关代码仓并一键适配、提供配置环境
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert_mm.sh

# 替换MindSpeed中的文件
cd MindSpeed-MM
cp examples/mindspore/qwen2vl/dot_product_attention.py ../MindSpeed/mindspeed/core/transformer/dot_product_attention.py

# 安装对应版本的tokenizers
pip install tokenizers==0.21
mkdir ckpt
mkdir data
mkdir logs
```

---
<a id="jump2"></a>
## 权重下载及转换

<a id="jump2.1"></a>
### 1. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main)；
- 模型地址: [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)；
- 模型地址: [Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/tree/main)；
- 模型地址: [Qwen2.5-VL-72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/tree/main)；


 将下载的模型权重保存到本地的`ckpt/hf_path/Qwen2.5-VL-*B-Instruct`目录下(*表示对应的尺寸)。

<a id="jump2.2"></a>
### 2. 权重转换(hf2mm)

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的互相转换以及PP（Pipeline Parallel）权重的重切分。参考[权重转换工具](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)了解该工具的具体使用。**注意当前在MindSpore后端下，转换出的权重无法用于Torch后端的训练**。

MindSpore后端默认在Device侧进行权重转换，在模型规模较大时存在OOM风险，因此建议用户手动修改`MindSpeed-MM/checkpoint/convert_cli.py`，加入如下代码将其设置为CPU侧权重转换：

```python
import mindspore as ms
ms.set_context(device_target="CPU", pynative_synchronize=True)
import torch
torch.configs.set_pyboost(False)
```


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
  --cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
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

如果需要使用转换的模型进行训练，同步修改`examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重目录，注意与原始权重 `ckpt/hf_path/Qwen2.5-VL-7B-Instruct`进行区分。

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
```

<a id="jump2.3"></a>
### 3. 权重转换(mm2hf)

MindSpeed-MM修改了部分原始网络的结构名称，在微调后，如果需要将权重转回huggingface格式，可使用`mm-convert`权重转换工具对微调后的权重进行转换，将权重名称修改成和原始网络一致。

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

<a id="jump3"></a>
## 数据集准备及处理

<a id="jump3.1"></a>
### 1. 数据集下载(以coco2017数据集为例)

(1)用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中(以当前目录MindSpeed-MM/为例)；

(2)获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下;

(3)运行数据转换脚本python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py，在./data路径下将生成文件mllm_format_llava_instruct_data.json(如果该文件已存在，请先移除或重命名备份);

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
当前支持读取多个以`,`（注意不要加空格）分隔的数据集，配置方式为`data_*b.json`中(*表示对应的模型尺寸)
dataset_param->basic_parameters->dataset
从"./data/mllm_format_llava_instruct_data.json"修改为"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json".

同时注意`data_*b.json`中`dataset_param->basic_parameters->max_samples`参数(*表示对应的模型尺寸)，该参数作用是限制数据只读`max_samples`条，以方便快速验证功能。如果正式训练时，可以把该参数去掉则读取全部数据。

<a id="jump3.2"></a>
### 2. 纯文本或有图无图混合训练数据(以LLaVA-Instruct-150K为例)

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
### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>
### 2. 配置参数

【数据目录配置】

根据实际情况修改`data_*b.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

以Qwen2.5VL-7B为例，`data_7b.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径。

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
}
```

如果需要加载大批量数据，可使用流式加载，修改`data_7b.json`中的`sampler_type`字段，增加`streaming`字段。（注意：使用流式加载后当前仅支持`num_worker=0`，单进程处理数据，会有性能波动，并且不支持断点续训功能。）

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

如果需要计算validation loss，需要在shell脚本中修改`eval-interval`参数和`eval-iters`参数；需要在`data_7b.json`中的`basic_parameters`内增加字段：    
对于非流式数据有两种方式：①根据实际情况增加`val_dataset`验证集路径，②增加`val_rate`字段对训练集进行切分；    
对于流式数据，仅支持增加`val_dataset`字段进行计算。

```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            ...
            "val_dataset": "./data/val_dataset.json",
			"val_max_samples": null,
			"val_rate": 0.1,
            ...
        },
        ...
    },
   ...
    }
}
```

【模型保存加载及日志信息配置】

根据实际情况配置`examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

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

配置`examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`参数如下

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
比如7b默认的值`[32,0,0,0]`、`[1,10,10,7]`，其含义为PP域内第一张卡先放32层`vision_encoder`再放1层`text_decoder`、第二张卡放`text_decoder`接着的10层、第三张卡放`text_decoder`接着的10层、第四张卡放`text_decoder`接着的7层，`vision_encoder`没有放完时不能先放`text_decoder`（比如`[30,2,0,0]`、`[1,10,10,7]`的配置是错的）。

同时注意，如果某张卡上的参数全部冻结时会导致没有梯度（比如`vision_encoder`冻结时PP配置`[30,2,0,0]`、`[0,11,10,7]`），需要在`finetune_qwen2_5_vl_7b.sh`中`GPT_ARGS`参数中增加`--enable-dummy-optimizer`，参考[dummy_optimizer特性文档](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/dummy_optimizer.md)。


<a id="jump4.3"></a>
### 3. 启动微调

以Qwen2.5VL-7B为例，启动微调训练任务。

```shell
bash examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
```

---
<a id="jump5"></a>
## 推理

<a id="jump5.1"></a>
### 1. 准备工作（以微调环境为基础，包括环境安装、权重下载及转换-目前支持PP切分的推理）

追加安装：

```shell
pip install qwen_vl_utils
```

注：如果使用huggingface下载的原始权重，需要权重转换，权重转换步骤中，根据具体需求设置PP切分的参数。

注：如果使用的MindSpeed-MM中保存的权重则无需进行转换，可直接加载（需要保证与训练的切分一致）。

<a id="jump5.2"></a>
### 2. 配置参数

根据实际情况修改examples/mindspore/qwen2.5vl/inference_qwen2_5_vl_7b.json和examples/mindspore/qwen2.5vl/inference_qwen2_5_vl_7b.sh中的路径配置，包括tokenizer的加载路径from_pretrained、以及图片处理器的路径image_processer_path。需注意

（1）tokenizer/from_pretrained配置的路径为从huggingface下载的原始Qwen2.5-VL-7B-Instruct路径。

（2）shell文件中的LOAD_PATH的路径为经过权重转换后的模型路径（可PP切分）。

<a id="jump5.3"></a>
### 3. 启动推理

```shell
bash examples/mindspore/qwen2.5vl/inference_qwen2_5_vl_7b.sh
```

注：单卡推理需打开FA，否则可能会显存不足报错，开关--use-flash-attn 默认已开，确保FA步骤完成即可。如果使用多卡推理则需要调整相应的PP参数和NPU使用数量的NPUS_PER_NODE参数。以PP4为例，shell修改参数如下：

```shell
NPUS_PER_NODE=4 # 可用几张卡 要大于 PP*TP*CP
PP=4 #PP并行参数
```

---
<a id="jump6"></a>
## Qwen2.5VL支持视频理解

<a id="jump6.1"></a>
### 1. 加载视频数据集

数据集中的视频数据集取自 [llamafactory](https://github.com/hiyouga/LLaMA-Factory/tree/main/data)，视频取自mllm_demo_data，使用时需要将该数据放到自己的data文件夹中去，同时将llamafactory上的mllm_video_demo.json也放到自己的data文件中。

之后根据实际情况修改 `data_*b.json` 中的数据集路径，包括 `model_name_or_path` 、 `dataset_dir` 、 `dataset` 字段，并修改"attr"中  `images` 、 `videos` 字段，修改结果参考如下：

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
### 2. 修改模型配置

在model_xxx.json中，修改`img_context_token_id`如下所示：
```
"img_context_token_id": 151656
```
注意， `image_token_id` 和 `img_context_token_id`两个参数作用不一样。前者是固定的，是标识图片的 token ID，在qwen2_5_vl_get_rope_index中用于计算图文输入情况下序列中的图片数量。后者是标识视觉内容的 token ID，用于在forward中标记视觉token的位置，所以需要根据输入做相应修改。

<a id="jump6.3"></a>
### 3. 启动微调
以Qwen2.5VL-7B为例，启动微调训练任务：

```shell
bash examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
```

<a id="jump7"></a>
## 评测

<a id="jump7.1"></a>
### 1. 数据集准备

当前模型支持AI2D(test)、ChartQA(test)、Docvqa(val)、MMMU(val)四种数据集的评测，
数据集参考下载链接：

- [MMMU_DEV_VAL](https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv)
- [DocVQA_VAL](https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv)
- [AI2D_TEST](https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv)
- [ChartQA_TEST](https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv)

<a id="jump7.2"></a>
### 2. 参数配置

如果要进行评测，需要将要评测的数据集名称和路径修改到examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.json，
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

examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.json改完后，需要将该文件的路径配置到examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh的MM_MODEL字段中。

以及需要将上面提到的权重转换后模型权重路径配置到examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh中的LOAD_PATH字段中。

```shell
MM_MODEL=examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.json
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
```

当前评测也支持多卡数据并行模式评测，需要更改NPU卡数量配置

```shell
NPUS_PER_NODE=1
```

<a id="jump7.3"></a>
### 3. 启动评测
评测额外依赖一些python包，使用下面命令进行安装

```shell
pip install -e ".[evaluate]"
```

启动shell开始评测
```shell
bash examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh
```

评测结果会输出到`result_output_path`路径中，输出的结果文件有：

- *.xlsx文件，这个文件会输出每道题的预测结果和答案等详细信息。
- *.csv文件，这个文件会输出统计准确率等数据。

---

<a id="jump8"></a>
## 环境变量声明
ASCEND_RT_VISIBLE_DEVICES： 指定NPU设备的索引值

ASCEND_SLOG_PRINT_TO_STDOUT： 是否开启日志打印， 0：关闭日志打屏，1：开启日志打屏

ASCEND_GLOBAL_LOG_LEVEL： 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。
0：对应DEBUG级别，1：对应INFO级别，2：对应WARNING级别，3：对应ERROR级别，4：对应NULL级别，不输出日志

HCCL_CONNECT_TIMEOUT:  用于限制不同设备之间socket建链过程的超时等待时间，需要配置为整数，取值范围[120,7200]，默认值为120，单位s

HCCL_EXEC_TIMEOUT： 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步

ASCEND_LAUNCH_BLOCKING： 控制算子执行时是否启动同步模式，0：采用异步方式执行，1：强制算子采用同步模式运行

MS_DEV_HOST_BLOCKING_RUN：控制动态图算子是否单线程下发。0：多线程下发，1：单线程下发

MS_DEV_LAUNCH_BLOCKING：控制算子是否同步下发。0：异步下发，1：采用单线程下发且流同步

ACLNN_CACHE_LIMIT： 配置单算子执行API在Host侧缓存的算子信息条目个数

TOKENIZERS_PARALLELISM： 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为

NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量

---
<a id="jump9"></a>
## 注意事项

1. 在 `finetune_xx.sh`里，与模型结构相关的参数并不生效，以`examples/mindspore/qwen2.5vl/model_xb.json`里同名参数配置为准，非模型结构的训练相关参数在 `finetune_xx.sh`修改。
2. 在使用单卡进行3B模型训练时，如果出现Out Of Memory，可以使用多卡并开启分布式优化器进行训练。
3. `model.json`设置use_remove_padding为true时，在`examples/mindspore/qwen2vl/dot_product_attention.py`中，attention_mask形状当前固定为[2048, 2048]，如需更改请参考[昇腾官网FlashAttentionScore](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0027.html)的替换指南。