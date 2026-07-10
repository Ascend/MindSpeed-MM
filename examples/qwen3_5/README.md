# Qwen3_5 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#环境安装)
  - [环境准备](#1-环境准备)
  - [环境搭建](#2-环境搭建)
  - [安装配套版本的TriTon-Ascend](#3-安装配套版本的triton-ascend)
  - [安装fla-npu以适配AscendC](#4-安装fla-npu以适配ascendc)
- [权重下载及转换](#权重下载及转换)
  - [权重下载](#1-权重下载)
- [数据集准备及处理](#数据集准备及处理)
- [微调](#微调)
  - [准备工作](#1-准备工作)
  - [配置参数](#2-配置参数)
  - [启动微调](#3-启动微调)
- [环境变量声明](#环境变量声明)

## 版本说明

### 参考实现

```shell
url=https://github.com/huggingface/transformers.git
commit_id=fc91372
```

### 变更记录

2026.02.10: 首次支持Qwen3_5模型

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/install_guide.md)，完成昇腾软件安装。
> Python版本推荐3.10，torch和torch_npu版本推荐2.7.1版本，CANN推荐使用8.5.2版本；

‼️MoE部分的加速特性依赖较新版本的torch_npu和CANN，推荐使用以下版本

- [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler)
- [torch_npu](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_description.md)

<a id="jump1.2"></a>

### 2. 环境搭建

拉取MindSpeed MM代码仓，并进入代码仓根目录：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

执行如下指令一键安装：

```bash
bash scripts/install.sh --msid eb10b92 && bash examples/qwen3_5/install_extensions.sh
```

### 3. 安装配套版本的Triton-Ascend

安装配套版本的Triton-Ascend，请参考《Triton-Ascend》中的"[通过pip安装Triton-Ascend](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#piptriton-ascend)"章节，获取配套版本的Triton-Ascend安装指令。

可参考如下安装命令：

```shell
# 注意：triton-ascend 3.2.0 及以下 Triton-Ascend 和 Triton 不能同时存在。需要先卸载社区 Triton，再安装 Triton-Ascend。
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

### 4. 安装fla-npu以适配AscendC

拉取flash-linear-attention-npu代码仓，并进入代码仓根目录，切到对应commitID

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu
cd flash-linear-attention-npu
# 由于flash-linear-attention-npu为开源仓实现，当前使用充分验证的历史版本，后续充分验证适配后切换v26.6.0分支
git checkout 60a791f
# 适配最新相关组件修改，cherry-pick需要提前配置好git的name及email
# git config user.email "you@example.com"
# git config user.name "Your Name"
git cherry-pick 50cba07
```

安装步骤：可参考fla-npu仓README：[flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu/blob/release/v26.1.0/README.md)

检验fla_npu是否安装成功

```bash
pip list | grep fla_npu
```

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

### 1. 权重下载

从Huggingface库下载对应的模型权重:

> [!NOTE]
>
> 如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

- 模型地址: [Qwen3.5-*B](https://huggingface.co/collections/Qwen/qwen35)；

 将下载的模型权重保存到本地的`ckpt/hf_path/xxxxxxx`目录下。(*表示对应的尺寸)

如果使用fsdp2的meta init初始化模型或MoE模型需要支持mtp，都需要先根据模型配置完成以下权重转换：

(1) 模型配置文件config.json中的`tie_word_embeddings`字段为`true`时（例如0.8B，2B，4B模型），使用以下转换脚本：

```bash
mm-convert Qwen35Converter hf_to_dcp \
--hf_dir ckpt/hf_path/xxxxxxx \
--dcp_dir ckpt/dcp_path/xxxxxxx \
--tie_weight_mapping '{"lm_head.weight":"model.language_model.embed_tokens.weight"}' \
--num_workers 0

# 其中：
# hf_dir: huggingface权重目录
# dcp_dir: 转换后DCP格式的权重保存目录
# tie_weight_mapping: 权重绑定映射关系
# num_workers: 并行工作线程数，0表示串行执行，若存储IO性能允许，可适当调大并发数以提升转换效率，推荐设置为4

# 转换后的目录结构为：
# ———— xxxxxxx
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

(2) 其它场景:

```bash
mm-convert Qwen35Converter hf_to_dcp \
--hf_dir ckpt/hf_path/xxxxxxx \
--dcp_dir ckpt/dcp_path/xxxxxxx \
--num_workers 0

# 其中：
# hf_dir: huggingface权重目录
# dcp_dir: 转换后DCP格式的权重保存目录
# num_workers: 并行工作线程数，0表示串行执行，若存储IO性能允许，可适当调大并发数以提升转换效率，推荐设置为4

# 转换后的目录结构为：
# ———— xxxxxxx
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

并在`xxx_config.yaml`中将`init_model_with_meta_device`参数配置为`True`，同时将`load`参数修改为转换后的dcp权重路径（写到`release`文件夹的上一级目录）。
注意：如果MoE模型不支持mtp，可在执行`mm-convert`权重转换前将`ckpt/hf_path/xxxxxxx/config.json`中的`mtp_num_hidden_layers`设置为0，以跳过mtp专家权重合并，缩短转换时间，如397B模型可以缩短约5分钟。

MindSpeed MM保存权重的格式也为dcp格式，可使用如下命令将dcp权重转换回HF权重：

```bash
# 待转换的dcp权重目录结构样例为：
# ———— xxxxxxx
#   |—— release
#   |—— latest_checkpointed_iteration.txt

mm-convert Qwen35Converter dcp_to_hf \
--save_hf_dir ckpt/save_hf_path/Qwen3.5-xxB-hf-save \
--dcp_dir ./save_path/iter_000xx \
--origin_hf_dir ckpt/hf_path/Qwen3.5-xxB \
--to_bf16 false
```

其中，`--save_hf_dir`表示转换后的权重保存路径，`--dcp_dir`表示保存的权重路径，`iter_000xx`表示保存的第xx步的权重，`--origin_hf_dir`表示原始huggingface权重的路径，`--to_bf16`表示权重数据类型是否从fp32转换成bf16。
注意：如果模型没有开启mtp（即，在`xxx_config.yaml`中model下的`mtp_num_layers`字段配置为0或没有配置），默认转换后的权重中不会包含mtp层的权重，可以通过设置`--keep_origin_mtp_weights true`来保留mtp层的权重。

---
<a id="jump3"></a>

## 数据集准备及处理

- 使用**真实数据集**训练：参考[针对VL模型的数据构造 · 使用真实数据集](../../docs/zh/features/building_data_for_VLModel.md#real-data)（下载COCO2017 → 下载LLaVA-Instruct-150K标注 → 运行转换脚本生成`mllm_format_llava_instruct_data.json`）。
- 使用**虚构数据**做功能/性能测试：参考[针对VL模型的数据构造 · 使用虚构数据](../../docs/zh/features/building_data_for_VLModel.md#mock-data)。

## 微调

<a id="jump4.1"></a>

### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

### 2. 配置参数

【数据目录配置】

根据实际情况修改`xxx_config.yaml`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

示例：如果数据及其对应的json都在/home/user/data/目录下，其中json目录为/home/user/data/video_data_path.json，此时配置如下：
`dataset_dir`配置为/home/user/data/;
`dataset`配置为./data/video_data_path.json
注意此时`dataset`需要配置为相对路径
**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

【模块冻结配置】

当前支持自定义冻结模块，在`xxx_config.yaml`中model->freeze字段中配置需要冻结的模块即可实现相应模块冻结。

【模型保存加载及日志信息配置】

根据实际情况配置`xxx_config.yaml`的`training`参数，包括保存路径以及保存间隔`save`、`save_interval`
根据实际情况配置`xxx_config.yaml`中的`init_from_hf_path`参数，该参数表示初始权重的加载路径。

【ulysses-cp并行配置】

根据实际情况配置`xxx_config.yaml`中的`ulysses_parallel_size`以调整ulysses-cp的并行度。（`ulysses_parallel_size`为1时不开启ulysses-cp）

**注意在开启ulysses-cp时，请将`xxx_config.yaml`中的`attn_implementation`配置为`flash_attention_2`**

【EP并行配置】

根据实际的需求配置`xxx_config.yaml`中的`expert_parallel_size`（注意仅对MoE模型生效）

根据`expert_parallel_size`可以自行选择更合适的`ep_plan.dispatcher`，推荐`expert_parallel_size`小于`topk`时，`dispatcher`选择`allgather`，`expert_parallel_size`大于`topk`时选择`alltoall`。

【MoE aux loss配置】

针对MoE模型，如果训练的过程中需要在交叉熵损失的基础上增加`router_aux_loss`使得训练过程中的专家负载分配区域平衡的话，可以配置`xxx_config.yaml`中的`features.loss_cfg.router_aux_loss_coef`字段，该字段表示负载均衡损失的系数。

【mtp配置】
当前模型支持配置mtp模块，在`xxx_config.yaml`中model下的`mtp_num_layers`字段配置为1，默认为0；`mtp_loss_scaling_factor`字段也支持配置，默认为0.1
注意：qwen3.5的mtp layer目前只支持配置1层。

【性能优化配置】

- 重计算
  - 在`features.recompute`配置，`true`表示开启，`false`表示关闭，默认开启。
  - 开启后可以节省显存占用
- [chunkloss](../../docs/zh/features/chunkloss.md)
  - 在`features.enable_chunk_loss`配置，`true`表示开启，`false`表示关闭
  - `features.chunkloss_plan.chunk_size`表示计算loss的时候在seq维度切分成大小为`chunk_size`的小块进行计算。
  - 开启后可以大幅降低loss计算时的显存尖刺，节省整体显存占用
- [async activation offload](../../docs/zh/features/async_activation_offload.md)
  - 在`features.enable_activation_offload`配置，`true`表示开启，`false`表示关闭
  - 开启后可以异步将重计算入口的激活值offload至host侧，在开启了重计算的场景下可以进一步节省显存。
- [chunkmbs](../../docs/zh/features/chunkmbs.md)
  - 在`features.enable_chunk_mbs`配置，`true`表示开启，`false`表示关闭
  - `features.chunkmbs_plan.chunk_mbs`表示切分以后单次计算的`micro_batch_size`
  - 开启该特性时需要同时使能重计算和async activation offload特性，可以增加FSDP2单次unshard对应的计算密度，提高整网吞吐。
- 选择性重计算
  - 在开启重计算的场景下，可以跳过linear attention层的gdn重计算，或者full attention层的flash attention重计算，并异步offload中间保存的tensor，在显存占用不变的条件下，减少计算量，提升训练吞吐
  - 在`model.skip_gdn_recompute`配置是否跳过linear attention层gdn的重计算，`true`表示跳过，`false`表示不跳过
  - 在`model.skip_flash_attn_recompute`配置是否跳过full attention层的flash attention的重计算，`true`表示跳过，`false`表示不跳过
  - 开启该特性时需要同时使能重计算和async activation offload特性

【单机运行配置】
以qwen3_5模型为例：
配置`examples/qwen3_5/finetune_qwen3_5.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
```

【多机运行配置】
如需拉起多机训练，修改启动脚本下 MASTER_ADDR、NODE_ADDR、NNODES以及NODE_RANK变量

``` shell
MASTER_ADDR: 主节点IP地址
NODE_ADDR: 本机IP地址
NODE_RANK: 第几个节点
NNODES: 一共几个节点
```

---

<a id="jump4.3"></a>

### 3. 启动微调

loss计算方式差异会对训练效果造成不同的影响，在启动训练任务之前，请查看关于loss计算的文档，选择合适的loss计算方式[vlm_model_loss_calculate_type.md](../../docs/zh/features/vlm_model_loss_calculate_type.md)
可在`xxx_config.yaml`的`model`参数中配置上述文档中的`loss_type`。

```shell
bash examples/qwen3_5/finetune_qwen3_5_xxB.sh
```

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
