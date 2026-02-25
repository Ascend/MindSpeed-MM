# InternVL2.5（MindSpore后端）使用指南

<p align="left">
</p>

## 目录

- [环境安装](#jump1)
  - [仓库拉取与环境搭建](#jump1.1)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
- [预训练](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动预训练](#jump4.3)
- [环境变量声明](#jump5)
- [注意事项](#jump6)

<a id="jump1"></a>

## 环境安装

MindSpeed-MM MindSpore后端的依赖配套如下表，安装步骤参考[基础安装指导](../../../docs/zh/mindspore/install_guide.md)。

| 依赖软件         |                                                                                                                                 |
| ---------------- |---------------------------------------------------------------------------------------------------------------------------------|
| 昇腾NPU驱动固件  | [在研版本](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.3.RC1&driver=Ascend+HDK+25.3.RC1) |
| 昇腾 CANN        | [8.3.rc1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1)                                |
| MindSpore        | [2.7.2](https://www.mindspore.cn/install/)                                                                                      |
| Python           | >=3.9                                                                                                                           |                                          |
|transformers     | [v4.53.0](https://github.com/huggingface/transformers/tree/v4.53.0)                                                             |    |

<a id="jump1.1"></a>

#### 1. 仓库拉取与环境搭建
模型一键拉起部署MindSpeed-Core-MS，自动拉取相关代码仓并对torch代码进行一键适配。一键拉起前，用户需拉取相关的代码仓以及进行环境搭建：

```shell
# 创建conda环境
conda create -n test python=3.10
conda activate test

# 使用环境变量
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# 安装MindSpeed-Core-MS一键拉起部署
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master

# 使用MindSpeed-Core-MS内部脚本自动拉取相关代码仓并一键适配
cd MindSpeed-Core-MS
pip install -r requirements.txt 
source auto_convert.sh mm

mkdir ckpt
mkdir data
mkdir logs
```
## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Hugging Face等网站下载开源模型权重

- [
InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B)；

- [
InternVL2_5-78B](https://huggingface.co/OpenGVLab/InternVL2_5-78B)；

将模型权重保存在`raw_ckpt`目录下，例如`raw_ckpt/InternVL2_5-78B`。

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的转换以及PP（Pipeline Parallel）的权重切分。

`mm-convert`工具详细用法参考[权重转换工具](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/features/mm_convert.md)。

```bash
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/cann/set_env.sh

# 4B
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2_5-4B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2_5-4B" \
  --cfg.parallel_config.llm_pp_layers [[36]] \
  --cfg.parallel_config.vit_pp_layers [[24]] \
  --cfg.trust_remote_code True

# 78B
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2_5-78B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2_5-78B" \
  --cfg.parallel_config.llm_pp_layers [[0,3,6,6,6,6,6,6,6,6,6,6,5,5,5,2]] \
  --cfg.parallel_config.vit_pp_layers [[45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] \
  --cfg.trust_remote_code True

# 其中：
# mm_dir: 转换后保存目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# trust_remote_code: 为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性
```

同步修改`examples/mindspore/internvl2.5/finetune_internvl2.5_*b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重`raw_ckpt/InternVL2_5-*B`进行区分。

以`InternVL2_5-78B`为例

```shell
LOAD_PATH="pretrained/InternVL2_5-78B"
```

---

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

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

使用视频进行训练，可参考[视频数据集构造](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data)自行构造视频数据集。

同时依赖Decord库读取视频，Decord安装方法如下：

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

<a id="jump4"></a>

## 微调

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`from_pretrained`、`data_path`、`data_folder`等字段。

以InternVL2_5-78B为例，`data_78B.json`进行以下修改，注意`tokenizer_config`的权重路径为转换前的权重路径。

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
          "from_pretrained": "raw_ckpt/InternVL2_5-78B",
          ...
      },
      ...
  },
  ...
}
```

【模型保存加载及日志信息配置】

根据实际情况配置`examples/internvl2.5/finetune_internvl2.5_xx.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）, 以InternVL2.5-78B为例：

```shell
...
# 加载路径
LOAD_PATH="ckpt/InternVL2_5-78B"
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

配置`examples/internvl2.5/finetune_internvl2.5_xx.sh`参数如下

```shell
  # 根据实际情况修改 ascend-toolkit 路径
  source /usr/local/Ascend/cann/set_env.sh
  NPUS_PER_NODE=8
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动微调

以InternVL2_5-78B为例，启动微调训练任务。

```shell
bash examples/mindspore/internvl2.5/finetune_internvl2.5_78B.sh
```

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


<a id="jump6"></a>

## 注意事项

1. 在使用流水线并行策略进行多机训练可能会出现卡住现象，可参考[此处](https://gitcode.com/Ascend/MindSpeed/pulls/1627/files)修改。
