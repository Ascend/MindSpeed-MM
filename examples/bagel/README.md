# Bagel 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
- [环境安装](#jump1)
  - [环境搭建](#jump1.1)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
  - [权重转换](#jump2.2)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
- [微调](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动微调](#jump4.3)
- [环境变量声明](#jump10)

## 版本说明

#### 参考实现
【训练】
```
url=https://github.com/bytedance-seed/BAGEL
commit_id = 57c390
```

---
<a id="jump1"></a>
## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md)，完成昇腾软件安装。
> Python版本推荐3.10，torch和torch_npu版本推荐2.7.1版本

<a id="jump1.1"></a>
#### 1. 环境搭建
拉取MindSpeed MM代码仓，并进入代码仓根目录：
```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```
对于X86架构机器，执行如下指令：
```bash
bash scripts/install.sh --arch x86 --msid 93c45456c7044bacddebc5072316c01006c938f9
```
对于ARM架构机器，执行如下指令：
```bash
bash scripts/install.sh --arch arm --msid 93c45456c7044bacddebc5072316c01006c938f9

---
<a id="jump2"></a>
## 权重下载及转换

<a id="jump2.1"></a>
#### 1. 权重下载

从Huggingface库下载对应的模型权重:
- 模型地址: [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT/tree/main)；

 将下载的模型权重保存到本地的`ckpt/hf_path/BAGEL-7B-MoT`目录下。

 <a id="jump2.2"></a>
#### 2. 权重转换(hf2mm)

Bagel模型需要对下载后的权重进行权重转换，运行权重转换脚本：
```shell
# Bagel
mm-convert BagelConverter hf_to_mm \
 --cfg.source_path <./ckpt/hf_path/BAGEL-7B-MoT/> \
 --cfg.target_path <./ckpt/mm_path/BAGEL-7B-MoT/> \
```

权重转换脚本的参数说明如下：

| 参数              | 含义                     | 默认值                                                       |
| :---------------- | :----------------------- | :----------------------------------------------------------- |
| --cfg.source_path | 原始权重路径             | /                                                            |
| --cfg.target_path | 转换或切分后权重保存路径 | /                                                            |

---
<a id="jump3"></a>
## 数据集准备及处理

<a id="jump3.1"></a>
#### 1. 数据集下载

```bash
https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip
```

将数据处理成如下格式

```bash
</dataset>
bagel_example
├── t2i/                           # text-to-image (parquet)
└── vlm/
    ├── images/                    # JPEG / PNG frames
    └── llava_ov_si.jsonl          # vision‑language SFT conversations
```

若需要自行添加数据集，请将数据处理成与上述数据统一格式

<a id="jump4"></a>
## 微调

<a id="jump4.1"></a>
#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。


<a id="jump4.2"></a>
#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径。其中，num_files表示t2i数据文件数，必须为卡数的整数倍，num_total_samples为数据总数

```json

    "t2i": {
      "data_dir": "data/t2i",
      "num_files": 8, 
      "num_total_samples": 800
    },
    "llava_ov":{
      "data_dir": "data/vlm/images",
      "jsonl_path": "data/vlm/llava_ov_si.jsonl",
      "num_total_samples": 2000
    },
    ......
```

【权重路径配置】

| 配置文件                                                |   修改字段  | 修改说明                                |
|-----------------------------------------------------| :---: |:------------------------------------|
| examples/bagel/data.json             |      model_path       | 修改为下载的tokenizer的权重所对应的路径           |
| examples/bagel/model.json             | from_pretrained | 修改为权重转换后的权重路径              |


<a id="jump4.3"></a>
#### 3. 启动微调

在开始之前，请确认环境准备、模型权重下载与转换已完成。

【并行化配置参数说明】：

- fsdp2

  - 使用场景：在模型参数规模较大时，可以通过开启fsdp2降低静态内存。
  
  - 使能方式：`examples/bagel/finetune_bagel.sh`的`GPT_ARGS`中加入`--use-torch-fsdp2`，`--fsdp2-config-path ${fsdp2_config}`，`--untie-embeddings-and-output-weights`以及`--ckpt-format torch_dcp`，其中fsdp2_config配置请参考：[FSDP2说明](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/fsdp2.md)

启动训练

```bash
bash examples/bagel/finetune_bagel.sh
```

## 环境变量声明

| 环境变量                          | 描述                                                       | 取值说明                                                                                                              |
|-------------------------------|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 是否开启日志打印                                                 | `0`: 关闭日志打屏<br>`1`: 开启日志打屏                                                                                        |
| `ASCEND_GLOBAL_LOG_LEVEL`     | 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志                             | `0`: 对应DEBUG级别<br>`1`: 对应INFO级别<br>`2`: 对应WARNING级别<br>`3`: 对应ERROR级别<br>`4`: 对应NULL级别，不输出日志                      |
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                              | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                                                   |
| `COMBINED_ENABLE`             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景       | `0`: 关闭<br>`1`: 开启                                                                                                |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                                                          |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                              | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                                          |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征                                                                 |
| `HCCL_EXEC_TIMEOUT`           | 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步                 | 需要配置为整数，取值范围`[68,17340]`，默认值为`1800`，单位`s`                                                                         |
| `ACLNN_CACHE_LIMIT`           | 配置单算子执行API在Host侧缓存的算子信息条目个数                              | 需要配置为整数，取值范围`[1, 10,000,000]`，默认值为`10000`                                                                         |
| `TOKENIZERS_PARALLELISM`      | 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为 | `False`: 禁用并行分词<br>`True`: 开启并行分词                                                                                 |
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启                                             | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用                                                                                    |
| `NPU_ASD_ENABLE`              | 控制是否开启Ascend Extension for PyTorch的特征值检测功能               | 设置`0`或未设置: 关闭特征值检测<br>`1`: 表示开启特征值检测，只打印异常日志，不告警<br>`2`:开启特征值检测，并告警<br>`3`:开启特征值检测，并告警，同时会在device侧info级别日志中记录过程数据 |
| `ASCEND_LAUNCH_BLOCKING`      | 控制算子执行时是否启动同步模式                                          | `0`: 采用异步方式执行<br>`1`: 强制算子采用同步模式运行                                                                                |
| `NPUS_PER_NODE`               | 配置一个计算节点上使用的NPU数量                                        | 整数值（如 `1`, `8` 等）