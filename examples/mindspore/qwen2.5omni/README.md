# Qwen2_5_Omni（MindSpore后端）使用指南

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
  - [视频音频数据集](#jump3.3)  
- [微调](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动微调](#jump4.3)
- [环境变量声明](#jump5)
- [注意事项](#jump6)

---
<a id="jump1"></a>
## 环境安装

MindSpeed-MM MindSpore后端的依赖配套如下表，安装步骤参考[基础安装指导](../../../docs/mindspore/install_guide.md)。

| 依赖软件         |                                                              |
| ---------------- | ------------------------------------------------------------ |
| 昇腾NPU驱动固件  | [在研版本](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.3.RC1&driver=Ascend+HDK+25.3.RC1) |
| 昇腾 CANN        | [在研版本](https://www.hiascend.com/zh/developer/download/community/result?module=cann) |
| MindSpore        | [2.7.2](https://www.mindspore.cn/install/)         |
| Python           | >=3.9                                                        |                                          |
|transformers     |      [v4.53.0](https://github.com/huggingface/transformers/tree/v4.53.0)    |    |
|mindspore_op_plugin | [master](https://gitee.com/mindspore/mindspore_op_plugin) |

<a id="jump1.1"></a>
### 1. 仓库拉取

针对MindSpeed MindSpore后端，昇腾社区提供了模型一键拉起部署MindSpeed-Core-MS，旨在帮助用户自动拉取相关代码仓并对torch代码进行一键适配，进而使用户无需再额外手动开发适配即可在华为MindSpore+CANN环境下一键拉起模型训练。在进行一键拉起前，用户需要拉取相关的代码仓：


```
# 使用环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# 安装MindSpeed-Core-MS一键拉起部署
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master

# 使用MindSpeed-Core-MS内部脚本自动拉取相关代码仓并一键适配
cd MindSpeed-Core-MS
source auto_convert.sh mm

# 拉取并安装mindspore_op_plugin
git clone https://gitee.com/mindspore/mindspore_op_plugin.git
cd mindspore_op_plugin
bash build.sh
source env.source
cd ..

mkdir ckpt
mkdir data
mkdir logs
```
--
<a id="jump2"></a>
## 权重下载及转换

<a id="jump2.1"></a>
### 1. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/tree/main)；


 将下载的模型权重保存到本地的`ckpt/hf_path/Qwen2.5-Omni-7B`目录下。

<a id="jump2.2"></a>
#### 2. 权重转换(hf2mm)

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的互相转换以及PP（Pipeline Parallel）权重的重切分。参考[权重转换工具](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)了解该工具的具体使用。**注意当前在MindSpore后端下，转换出的权重无法用于Torch后端的训练**。

> 注：权重准换依赖[mindspore_op_plugin](https://gitee.com/mindspore/mindspore_op_plugin)，请确保已安装该软件，使用教程请参考[op_plugin CPU 算子开发指南](https://gitee.com/mindspore/mindspore_op_plugin/wikis/op_plugin%20CPU%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97)

```bash
  
# 7b
mm-convert  Qwen2_5_OmniConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-Omni-7B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-Omni-7B" \
  --cfg.parallel_config.llm_pp_layers [[11,17]] \
  --cfg.parallel_config.vit_pp_layers [[32,0]] \
  --cfg.parallel_config.audio_pp_layers [[32,0]] \
  --cfg.parallel_config.tp_size 1

# 其中：
# mm_dir: 转换后保存目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# audio_pp_layers: audio在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
```

如果需要用转换后模型训练的话，同步修改`examples/mindspore/qwen2.5omni/finetune_qwen2_5_omni_7b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重 `ckpt/hf_path/Qwen2.5-Omni-7B`进行区分。

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-Omni-7B"
```

<a id="jump2.3"></a>
#### 3. 权重转换(mm2hf)

MindSpeed MM修改了部分原始网络的结构名称，在微调后，如果需要将权重转回huggingface格式，可使用`mm-convert`权重转换工具对微调后的权重进行转换，将权重名称修改为与原始网络一致。

```bash
mm-convert  Qwen2_5_OmniConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-Omni-7B" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-Omni-7B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-Omni-7B" \
  --cfg.parallel_config.llm_pp_layers [11,17] \
  --cfg.parallel_config.vit_pp_layers [32,0] \
  --cfg.parallel_config.audio_pp_layers [32,0] \
  --cfg.parallel_config.tp_size 1
# 其中：
# save_hf_dir: mm微调后转换回hf模型格式的目录
# mm_dir: 微调后保存的权重目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# audio_pp_layers: audio在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
```



<a id="jump3"></a>
## 数据集准备及处理

<a id="jump3.1"></a>
#### 1. 数据集下载(以coco2017数据集为例)

(1)用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中

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

<a id="jump3.3"></a>
#### 3.视频音频数据集

##### 1）加载视频数据集

数据集中的视频数据集取自llamafactory，https://github.com/hiyouga/LLaMA-Factory/tree/main/data

视频取自mllm_video_demo，使用时需要将该数据放到自己的data文件夹中去，同时将llamafactory上的mllm_video_audio_demo.json也放到自己的data文件中

之后根据实际情况修改 `data.json` 中的数据集路径，包括 `model_name_or_path` 、 `dataset_dir` 、 `dataset` 字段，并修改"attr"中  `images` 、 `videos` 字段，修改结果参考下图。

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./Qwen2.5-Omni-7B",
            ...
        },
        "basic_parameters": {
            ...
            "dataset_dir": "./data",
            "dataset": "./data/mllm_video_audio_demo.json",
            "cache_dir": "./data/cache_dir",
            ...
        },
        ...
        "attr": {
            "system": null,
            "images": null,
            "videos": "videos",
            "audios": "audios",
            ...
        },
    },
    ...
}
```

##### 2）修改模型配置

在model.json中，修改`img_context_token_id`为下图所示：
```
"img_context_token_id": 151656
```
注意， `image_token_id` 和 `img_context_token_id`两个参数作用不一样。前者是固定的，是标识图片的 token ID，在qwen2_5_omni_get_rope_index中用于计算图文输入情况下序列中的图片数量。后者是标识视觉内容的 token ID，用于在forward中标记视觉token的位置，所以需要根据输入做相应修改。


<a id="jump4"></a>
## 微调

<a id="jump4.1"></a>
#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>
#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

以Qwen2.5Omni-7B为例，`data.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径。

**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2.5-Omni-7B",
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

根据实际情况配置`examples/mindspore/qwen2.5omni/finetune_qwen2_5_omni_7b.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 加载路径
LOAD_PATH="ckpt/mm_path/Qwen2.5-Omni-7B"
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

```
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

【单机运行配置】

配置`examples/mindspore/qwen2.5omni/finetune_qwen2_5_omni_7b.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
export LOCAL_WORLD_SIZE=8
export MS_NODE_TIMEOUT=1800
```
注意，当开启PP时，`model.json`中配置的`vision_encoder`和`text_decoder`的`pipeline_num_layer`参数控制了各自的PP切分策略。对于流水线并行，要先处理`vision_encoder`再处理`text_decoder`。
比如7b默认的值`[32,0,0,0]`、`[1,10,10,7]`，其含义为PP域内第一张卡先放32层`vision_encoder`再放1层`text_decoder`、第二张卡放`text_decoder`接着的10层、第三张卡放`text_decoder`接着的10层、第四张卡放`text_decoder`接着的7层，`vision_encoder`没有放完时不能先放`text_decoder`（比如`[30,2,0,0]`、`[1,10,10,7]`的配置是错的）

同时注意，如果某张卡上的参数全部冻结时会导致没有梯度（比如`vision_encoder`冻结时PP配置`[30,2,0,0]`、`[0,11,10,7]`），需要在`finetune_qwen2_5_omni_7b.sh`中`GPT_ARGS`参数中增加`--enable-dummy-optimizer`，参考[dummy_optimizer特性文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/dummy_optimizer.md)。

<a id="jump4.3"></a>
#### 3. 启动微调

以Qwen2.5Omni-7B为例，启动微调训练任务。  
loss计算方式差异会对训练效果造成不同的影响，在启动训练任务之前，请查看关于loss计算的文档，选择合适的loss计算方式[vlm_model_loss_calculate_type.md](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/vlm_model_loss_calculate_type.md)
```shell
bash examples/mindspore/qwen2.5omni/finetune_qwen2_5_omni_7b.sh
```

---
<a id="jump5"></a>
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
<a id="jump6"></a>
## 注意事项

1. 在 `finetune_xx.sh`里，与模型结构相关的参数并不生效，以`examples/mindspore/qwen2.5omni/model_xb.json`里同名参数配置为准，非模型结构的训练相关参数在 `finetune_xx.sh`修改。