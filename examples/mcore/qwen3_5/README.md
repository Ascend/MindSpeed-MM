# Qwen3_5 More后端使用指南

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
- [权重下载](#权重下载)
- [数据集准备及处理](#数据集准备及处理)
- [微调](#微调)
  - [准备工作](#1-准备工作)
  - [配置参数](#2-配置参数)
  - [启动微调](#3-启动微调)
- [环境变量声明](#环境变量声明)
- [注意事项](#注意事项)

## 版本说明

### 参考实现

```shell
url=https://github.com/huggingface/transformers.git
commit_id=fc91372
```

### 变更记录

2026.07.15: 首次支持Qwen3_5 35B Mcore后端

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/install_guide.md)，完成昇腾软件安装。
> Python版本推荐3.10，torch和torch_npu版本推荐2.7.1版本，CANN推荐使用8.5.2版本；

<a id="jump1.2"></a>

### 2. 环境搭建

拉取MindSpeed MM代码仓，并进入代码仓根目录：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

执行如下指令一键安装：

```bash
bash scripts/install.sh --msid eb10b92 --megatron && bash examples/qwen3_5/install_extensions.sh
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
git checkout c2e3d83f
```

安装步骤：可参考fla-npu仓README：[flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu/blob/release/v26.1.0/README.md)

推荐使用以下安装命令

```shell
# source 实际的cann路径
source /usr/local/Ascend/cann/set_env.sh

# 编译算子 run 包，--soc 需指定为当前机器芯片类型 {ascend910b/ascend910_93/ascend950}
bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu
bash build_out/fla-npu_*.run
cd torch_custom/fla_npu/
bash build.sh
```

检验fla_npu是否安装成功

```bash
pip list | grep fla_npu
```

---

<a id="jump2"></a>

## 权重下载

<a id="jump2.1"></a>

从Huggingface库下载对应的模型权重:

> [!NOTE]
>
> 如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

- 模型地址: [Qwen3.5-*B](https://huggingface.co/collections/Qwen/qwen35)；

 将下载的模型权重保存到本地的`ckpt/hf_path/xxxxxxx`目录下。(*表示对应的尺寸)

---
<a id="jump3"></a>

## 数据集准备及处理

- 使用**真实数据集**训练：参考[针对VL模型的数据构造 · 使用真实数据集](../../../docs/zh/features/building_data_for_VLModel.md#real-data)（下载COCO2017 → 下载LLaVA-Instruct-150K标注 → 运行转换脚本生成`mllm_format_llava_instruct_data.json`）。
- 使用**虚构数据**做功能/性能测试：参考[针对VL模型的数据构造 · 使用虚构数据](../../../docs/zh/features/building_data_for_VLModel.md#mock-data)。

## 微调

<a id="jump4.1"></a>

### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

### 2. 配置参数

【数据目录配置】

根据实际情况修改`xxx_config.yaml`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`、`cache_dir`等字段。

示例：如果数据及其对应的json都在/home/user/data/目录下，其中json目录为/home/user/data/video_data_path.json，此时配置如下：
`dataset_dir`配置为/home/user/data/;
`dataset`配置为./data/video_data_path.json
注意此时`dataset`需要配置为相对路径
**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

【模块冻结配置】

当前支持自定义冻结模块，在`xxx_config.yaml`中`model`字段中配置需要冻结的模块即可实现相应模块冻结。

```yaml
### 模型相关配置
model:
  freeze_vision_model: true
  freeze_vision_projection: true
  freeze_language_model: false
```

【模型保存加载及日志信息配置】

根据实际情况配置`xxx_config.yaml`的`gpt_args`参数

```yaml
gpt_args:
  load: ckpt/Qwen3.5-35B-A3B  # 可以直接设置为下载的huggingface权重路径，依赖bridge_patch自动权重转换
  no_load_optim: true  # 不加载优化器状态，若需加载请移除
  no_load_rng: true  # 不加载随机数状态，若需加载请移除
  no_save_optim: true  # 不保存优化器状态，若需保存请移除
  no_save_rng: true  # 不保存随机数状态，若需保存请移除
  save_interval: 10000
  eval_interval: 10000
  eval_iters: 5000
  save: save_dir
```

【MoE配置】

根据实际需求配置`xxx_config.yaml`中moe相关配置，主要涉及`gpt_args`和`model`两个部分的参数设置

```yaml
gpt_args:
  expert_model_parallel_size: &EP 8
  moe_grouped_gemm: &MOE_GROUPED_GEMM true
  moe_permute_fusion: &MOE_PERMUTE_FUSION true
  moe_token_dispatcher_type: &MOE_TOKEN_DISPATCHER_TYPE alltoall
......

model:
  text_decoder:
    num_moe_experts: 256
    moe_router_topk: 8
    moe_ffn_hidden_size: 512
    moe_shared_expert_intermediate_size: 512
    moe_shared_expert_gate: true
    moe_router_load_balancing_type: none
    moe_aux_loss_coeff: 1e-3
    moe_grouped_gemm: *MOE_GROUPED_GEMM
    moe_router_pre_softmax: true
    moe_token_dispatcher_type: *MOE_TOKEN_DISPATCHER_TYPE
    moe_permute_fusion: *MOE_PERMUTE_FUSION
    moe_router_force_load_balancing: false
```

【MoE配置】显存优化配置

当训练序列长度比较大时，可以开启chunkloss降低峰值激活显存，可以设置`chunk_loss`为`True`，`chunk_size`表示计算loss的时候在seq维度切分成大小为`chunk_size`的小块进行计算

【重计算】

默认为全层重计算，当显存足够时，可以减小`recompute_num_layers`提升性能。

```yaml
recompute_num_layers: 40
recompute_granularity: full
recompute_method: block
```

【GDN算子】

- causal_conv1d_implementation支持`torch`、`triton`、`triton_with_transpose`、`ascendc`四种实现，`triton_with_transpose`通过将tranpose融合进去进一步提升性能；

- gdn_implementation支持`torch`、`triton`、`ascendc`三种实现；

- 目前仅支持的组合有如下:

  - | causal_conv1d_implementation | gdn_implementation |
    | ---------------------------- | ------------------ |
    | torch                        | torch、triton      |
    | triton                       | torch、triton      |
    | triton_with_transpose        | ascendc            |
    | ascendc                      | ascendc            |

```yaml
causal_conv1d_implementation: ascendc
gdn_implementation: ascendc
```

【单机运行配置】
以35B模型为例：
配置`examples/mcore/qwen3_5/finetune_qwen3_5_35B.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPUS_PER_NODE=16
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

```shell
bash examples/mcore/qwen3_5/finetune_qwen3_5_xxB.sh
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
<a id="jump11"></a>

## 注意事项

1. 在加载 processor 过程中，会因 `mistral_common` 三方库版本的兼容性问题导致无法找到 processor，进而训练报错退出，可通过以下方式解决：
   - 卸载`mistral_common` 三方库：pip uninstall -y mistral_common
   - 升级`mistral_common` 三方库至最新版本：pip install --upgrade mistral_common
