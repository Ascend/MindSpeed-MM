# DeepSeekVL2 使用指南

<p align="left">
</p>

## 目录

- [DeepSeekVL2 使用指南](#deepseekvl2-使用指南)
  - [目录](#目录)
  - [环境安装](#环境安装)
      - [1. 仓库拉取](#1-仓库拉取)
  - [权重下载及转换](#权重下载及转换)
      - [1. 权重下载](#1-权重下载)
      - [2. 权重转换](#2-权重转换)
  - [数据集准备及处理](#数据集准备及处理)
      - [1. 数据集下载](#1-数据集下载)
  - [微调](#微调)
      - [1. 准备工作](#1-准备工作)
      - [2. 配置参数](#2-配置参数)
      - [3. 启动微调](#3-启动微调)
  - [性能数据](#性能数据)
  - [环境变量声明](#环境变量声明)

---
<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

<a id="jump1.1"></a>

#### 1. 仓库拉取

```shell
# 安装MindSpeed-Core-MS一键拉起部署
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.5.0

# 使用MindSpeed-Core-MS内部脚本自动拉取相关代码仓并一键适配
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm

mkdir logs
mkdir dataset
mkdir ckpt
```

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Hugging Face等网站下载开源模型权重

- [DeepSeekVL2](https://huggingface.co/deepseek-ai/deepseek-vl2)；

将模型权重保存在`raw_ckpt`目录下，例如`raw_ckpt/DeepSeekVL2`。

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的转换以及TP（Tensor Parallel）和EP（Expert Parallel）的权重切分。

`mm-convert`工具详细用法参考[权重转换工具](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)

**注意**

1. DeepSeekVL权重转换依赖deepseekvl2包，安装过程参考[链接](https://github.com/deepseek-ai/DeepSeek-VL2)。deepseekvl2包与特定版本的transformers兼容，建议安装transformers 4.45.0或transformers 4.38.2版本以确保兼容性。
2. 转换前需要在hf格式权重目录下，修改config.json的`"_attn_implementation"`字段改为`"eager"`。

转换命令如下

```bash
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

mm-convert  DeepSeekVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/DeepSeekVl2" \
  --cfg.hf_config.hf_dir "raw_ckpt/DeepSeekVL2" \
  --cfg.parallel_config.llm_pp_layers [[13,17]] \
  --cfg.parallel_config.vit_pp_layers [[27,0]] \
  --cfg.parallel_config.ep_size 8 \
  --cfg.parallel_config.tp_size 1 \
  --cfg.trust_remote_code True

# 其中：
# mm_dir: 转换后保存目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# ep_size: ep并行数量，注意要和微调启动脚本中的配置一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
# trust_remote_code: 为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性
```

---

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

*注意：DeepSeekVL2原仓未开源训练数据集，这里以InternVL开源的数据集作为示例，用户可自行构造训练数据集。*

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

修改convert_ai2d_to_dsvl.py文件的input_file和output_file，例如：

```python
input_file = "dataset/playground/opensource/ai2d_train_12k.jsonl"    # 替换为实际输入路径
output_file = "dataset/playground/opensource/ai2d_train_12k_dsvl.jsonl"  # 替换为实际输出路径
```

运行数据格式转换脚本

```shell
python convert_ai2d_to_dsvl.py
```

---

<a id="jump4"></a>

## 微调

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`from_pretrained`、`data_path`、`data_folder`等字段。
注意`processor_path`为转换前的权重路径。

```json
{
  "dataset_param": {
      ...
      "basic_parameters": {
          "data_path": "dataset/playground/opensource/ai2d_train_12k_dsvl.jsonl",
          "data_folder": "dataset/playground/data/ai2d"
      },
      ...
      "processor_path": "deepseek-ai/deepseek-vl2",
      ...
  },
  ...
}
```

【模型保存加载及日志信息配置】

根据实际情况配置`examples/deepseekvl2/finetune_deepseekvl2.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）：

```shell
...
# 加载路径
LOAD_PATH="ckpt/DeepSeekVL2"
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

配置`examples/experimental/deepseekvl2/finetune_deepseekvl.sh`参数如下

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

<a id="jump4.3"></a>

#### 3. 启动微调

启动微调训练任务。

```shell
bash examples/experimental/deepseekvl2/finetune_deepseekvl2.sh
```

---
## 性能数据

| 模型                  | 机器型号           | 集群 | 任务 | 端到端 SPS |
|----------------------|----------|-----|----------|---------|
| DeepSeekVL2           | Atlas 800T A2 | 4*8  | 微调    | 4.924     |

注：此处 SPS 代表 Samples per Second。

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
