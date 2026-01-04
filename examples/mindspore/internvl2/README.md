# InternVL2 (MindSpore后端) 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#环境安装)
  - [仓库拉取及环境搭建](#仓库拉取及环境搭建)
- [权重下载及转换](#权重下载及转换)
  - [权重下载](#权重下载)
- [数据集准备及处理](#数据集准备及处理)
  - [数据集下载](#数据集下载)
- [微调](#微调)
  - [准备工作](#准备工作)
  - [配置参数](#配置参数)
  - [启动预训练](#启动预训练)
- [环境变量声明](#环境变量声明)

---

## 环境安装

【MindSpeed-MM MindSpore后端的依赖配套如下表，安装步骤参考[基础安装指导](../../../docs/mindspore/install_guide.md)。

| 依赖软件         |                                                              |
| ---------------- | ------------------------------------------------------------ |
| 昇腾NPU驱动固件  | [在研版本](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha) |
| 昇腾 CANN        | [在研版本](https://www.hiascend.com/zh/developer/download/community/result?module=cann) |
| MindSpore        | [2.7.1](https://www.mindspore.cn/install/)         |
| Python           | >=3.9                                                        |                                          |


### 仓库拉取及环境搭建


针对MindSpeed MindSpore后端，昇腾社区提供了一键转换工具MindSpeed-Core-MS，旨在帮助用户自动拉取相关代码仓并对torch代码进行一键适配，进而使用户无需再额外手动开发适配即可在华为MindSpore+CANN环境下一键拉起模型训练。在进行一键转换前，用户需要拉取相关的代码仓以及进行环境搭建：


```
# 创建conda环境
conda create -n test python=3.10
conda activate test

# 使用环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# 安装MindSpeed-Core-MS转换工具
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.4.0

# 使用MindSpeed-Core-MS内部脚本自动拉取相关代码仓并一键适配、提供配置环境
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm

# 替换MindSpeed中的文件
cd MindSpeed-MM
cp examples/internvl2/dot_product_attention.py MindSpeed/mindspeed/core/transformer/dot_product_attention.py
mkdir ckpt
mkdir data
mkdir logs
```

## 权重下载及转换


### 权重下载

从Hugging Face等网站下载开源模型权重

- [InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B/tree/main)；
- [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main)；
- [InternVL2-26B](https://huggingface.co/OpenGVLab/InternVL2-26B/tree/main)；
- [InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B/tree/main)；

将模型权重保存在`raw_ckpt`目录下，例如`raw_ckpt/InternVL2-8B`。

### 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的转换以及PP（Pipeline Parallel）和VPP（Virtual Pipeline Parallel）的权重切分(详细VPP配置参考[vpp特性说明](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/virtual_pipeline_parallel.md))。

`mm-convert`工具详细用法参考[权重转换工具](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)
**注意当前在MindSpore后端下，转换出的权重无法用于Torch后端的训练**。

MindSpore后端默认在Device侧进行权重转换，在模型规模较大时存在OOM风险，因此建议用户手动修改`MindSpeed-MM/checkpoint/convert_cli.py`，加入如下代码将其设置为CPU侧权重转换：

```python
import mindspore as ms
ms.set_context(device_target="CPU", pynative_synchronize=True)
import torch
torch.configs.set_pyboost(False)
```

以InternVL2-8B为例，使用命令如下


```bash
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 2B
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-2B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-2B" \
  --cfg.parallel_config.llm_pp_layers [[24]] \
  --cfg.parallel_config.vit_pp_layers [[24]] \
  --cfg.trust_remote_code True

# 8B
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-8B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-8B" \
  --cfg.parallel_config.llm_pp_layers [[6,9,9,8]] \
  --cfg.parallel_config.vit_pp_layers [[24,0,0,0]] \
  --cfg.trust_remote_code True

# 8B VPP
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-8B-vpp" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-8B" \
  --cfg.parallel_config.llm_pp_layers [[0,0,0,1],[4,4,4,4],[4,4,4,3]] \
  --cfg.parallel_config.vit_pp_layers [[6,7,7,4],[0,0,0,0],[0,0,0,0]] \
  --cfg.trust_remote_code True

# 76B
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-Llama3-76B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-Llama3-76B" \
  --cfg.parallel_config.llm_pp_layers [[0,0,0,1,5,6,7,7,7,7,7,7,7,7,6,6]] \
  --cfg.parallel_config.vit_pp_layers [[11,12,12,10,0,0,0,0,0,0,0,0,0,0,0,0]] \
  --cfg.trust_remote_code True

```
- 其中：
- mm_dir: 转换后保存目录
- hf_dir: huggingface权重目录
- llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
- vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
- trust_remote_code: 为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性

---


## 数据集准备及处理


### 数据集下载

【图片数据】

用户需自行获取并解压[InternVL-Finetune](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data)数据集到`dataset/playground`目录下，以数据集ai2d为例，解压后的数据结构如下：

   ```
   $playground
   ├── data
       ├── ai2d
           ├── abc_images
           ├── images
   ├── opensource
       ├── ai2d_train_12k.jsonl
   ```

【视频数据】

若要使用视频进行训练，用户可参考[视频数据集构造](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data)自行构造视频数据集。

---


## 微调


### 准备工作

配置脚本前需要完成前置准备工作，包括：**[环境安装](#环境安装)**、**[权重下载及转换](#权重下载及转换)**、**[数据集准备及处理](#数据集准备及处理)**，详情可查看对应章节。


### 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`from_pretrained`、`data_path`、`data_folder`等字段。

以InternVL2-8B为例，`data_8B.json`进行以下修改，注意`tokenizer_config`的权重路径为转换前的权重路径。

```json
{
  "dataset_param": {
      ...
      "basic_parameters": {
          "data_path": "dataset/playground/opensource/ai2d_train_12k.jsonl",
          "data_folder": "dataset/playground/data/ai2d"
      },
      ...
      "tokenizer_config": {
          ...
          "from_pretrained": "raw_ckpt/InternVL2-8B",
          ...
      },
      ...
  },
  ...
}
```

如果需要加载大批量数据，可使用流式加载，修改`data.json`中的`sampler_type`字段，增加`streaming`字段。（注意：使用流式加载后当前仅支持`num_worker=0`，单进程处理数据，会有性能波动，并且不支持断点续训功能。）

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

如果需要计算validation loss，还需要设置以下参数：
- 在shell脚本中设置`eval-interval`和`eval-iters`参数；
- 在`data.json`中的`basic_parameters`内增加字段：
  - 对于非流式数据有两种方式：①根据实际情况增加`val_dataset`验证集路径，②增加`val_rate`字段对训练集进行切分；    
  - 对于流式数据，仅支持增加`val_dataset`字段进行计算。

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
```

【模型保存加载】

根据实际情况配置`examples/mindspore/internvl2/finetune_internvl2_xx.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）, 以InternVL2-8B为例：

```shell
...
# 加载路径
LOAD_PATH="ckpt/InternVL2-8B"
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
    ···
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


【日志信息配置】

配置`examples/mindspore/internvl2/finetune_internvl2_xx.sh`参数如下

```shell
OUTPUT_ARGS="
    ···
    --log-interval 1 \  # 日志间隔
    ...
"
```


【单机运行配置】

配置`examples/mindspore/internvl2/finetune_internvl2_xx.sh`参数如下

```shell
  # 根据实际情况修改 ascend-toolkit 路径
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  NPUS_PER_NODE=8
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```


### 启动微调

以InternVL2-8B为例，启动微调训练任务。

```shell
bash examples/mindspore/internvl2/finetune_internvl2_8B.sh
```


---

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