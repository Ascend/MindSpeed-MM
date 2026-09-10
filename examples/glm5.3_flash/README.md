# GLM-5.3-Flash 使用指南

<p align="left">
</p>

## 目录

- [GLM-5.3-Flash 使用指南](#glm-53-flash-使用指南)
  - [目录](#目录)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
    - [变更记录](#变更记录)
  - [环境安装](#环境安装)
    - [1. 环境准备](#1-环境准备)
    - [2. 环境搭建](#2-环境搭建)
    - [3. 安装配套版本的Triton Ascend](#3-安装配套版本的triton-ascend)
    - [4. 安装 fla-npu](#4-安装-fla-npu)
  - [数据集准备及处理](#数据集准备及处理)
  - [训练](#训练)
    - [1. 准备工作](#1-准备工作)
    - [2. 配置参数](#2-配置参数)
    - [3. 受限场景支持](#3-受限场景支持)
    - [4. 启动训练](#4-启动训练)
  - [环境变量声明](#环境变量声明)
  - [注意事项](#注意事项)

## 版本说明

> 当前版本仅验证减层受限场景训练，使用前请先阅读[受限场景支持](#3-受限场景支持)章节。
> 更多能力正在支持，敬请期待！

当前目录提供GLM-5.3-Flash在MindSpeed-MM FSDP2训练流程中的示例配置。模型实现位于`mindspeed_mm/fsdp/models/glm5_next`，训练配置和启动脚本位于`examples/glm5.3_flash`。

### 参考实现

```shell
url=https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/tree/main
```

### 变更记录

2026.08.27: 首次支持GLM-5.3-Flash模型

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](../../docs/zh/pytorch/install_guide.md)，完成昇腾软件安装。

推荐环境版本如下：

| 组件 | 推荐版本 |
|------|----------|
| Python | 3.12 |
| torch | 2.10.0|
| TorchNPU |2.10.0.post4 及以上 |
| CANN | 9.1.0 及以上 |

<a id="jump1.2"></a>

### 2. 环境搭建

拉取 MindSpeed MM 代码仓，并进入代码仓根目录：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
bash scripts/install.sh --msbranch master
# 调整部分依赖库：
pip uninstall -y torchaudio
cd ..
```

执行如下指令安装基础依赖：

```bash
# 安装transformer库
pip install transformers==5.16.1
pip install tiktoken==0.12.0
```

### 3. 安装配套版本的Triton Ascend

GLM-5.3-Flash的KDA（Kimi Delta Attention）等线性注意力融合算子基于Triton实现，在昇腾环境下需要安装配套版本的Triton-Ascend
请参考《Triton-Ascend》中的"[通过pip安装Triton-Ascend](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#piptriton-ascend)"章节，获取配套版本的Triton-Ascend安装指令。

```shell
# 供参考, 命令以文档为准
pip install triton-ascend==3.2.2 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

KDA 算子实现依赖`triton-ascend-kernels`算子库（`modeling_glm5_next.py` 中的 `chunk_kda` 来自该包），安装步骤如下：

atlas A2&A3训练产品安装步骤

```shell
# 拉取 triton-ascend-kernels 代码仓
git clone https://gitcode.com/fengrui886/triton-ascend-kernels
cd triton-ascend-kernels
git checkout kda_a3

# 安装
pip install -e . --no-build-isolation --no-deps
cd ..
```

950系列产品安装步骤

```shell
# 拉取 triton-ascend-kernels 代码仓
git clone https://gitcode.com/shenzhaofeng/triton-ascend-kernels.git
cd triton-ascend-kernels

# 拉取配套分支
git checkout kda_a5

# 安装
pip install -e . --no-build-isolation --no-deps
cd ..
```

### 4. 安装 fla-npu

GLM-5.3-Flash适配包含 AscendC KDA wrapper，需安装 fla-npu 以保证相关模块可正常导入。

拉取 flash-linear-attention-npu 代码仓，并进入代码仓根目录，切到对应 commit：

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu
cd flash-linear-attention-npu
git checkout 2062460754c8
```

安装步骤可参考fla-npu仓README：[flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu/blob/main/README.md), 参考"源码一键编译并生成 wheel"完整fla编译。

```bash
# fla需安装gawk
python -m pip install -r requirements.txt
# 检查编译环境满足要求, 如果不满足需解决
python scripts/check_npu_env.py
# FLA_NPU_SOC对应A2/A3/A5分别填ascend910b/ascend910_93/ascend950
FLA_NPU_SOC=ascend910_93 python scripts/build_wheel.py
# 编译成功后安装
./build_out/*.run
pip install --force-reinstall --no-cache-dir --no-deps ./dist/*.whl
cd ..
```

安装后检验fla-npu是否安装成功：

```bash
pip list | grep fla
```

---
<a id="jump2"></a>

## 数据集准备及处理

- 使用**真实数据集**训练：参考[针对VL模型的数据构造 · 使用真实数据集](../../docs/zh/features/building_data_for_VLModel.md#real-data)（下载COCO2017 → 下载LLaVA-Instruct-150K标注 → 运行转换脚本生成`mllm_format_llava_instruct_data.json`）。
- 使用**虚构数据**做功能/性能测试：参考[针对VL模型的数据构造 · 使用虚构数据](../../docs/zh/features/building_data_for_VLModel.md#mock-data)。

## 训练

<a id="jump3.1"></a>

### 1. 准备工作

从 HuggingFace 下载 GLM-5.3-Flash 模型文件到本地目录，并将配置文件examples/glm5.3_flash/glm5.3_next_config.yaml中的 `HF_MODEL_LOAD_PATH` 指向该目录：

```yaml
data:
  dataset_param:
    preprocess_parameters:
      model_name_or_path: &HF_MODEL_LOAD_PATH /path/to/GLM-5.3-Flash

model:
  model_name_or_path: *HF_MODEL_LOAD_PATH
```

**说明**：代码仓中已包含适配 MindSpeed-MM FSDP2 的模型实现文件：

```text
mindspeed_mm/fsdp/models/glm5_next/configuration_glm5_next.py
mindspeed_mm/fsdp/models/glm5_next/modeling_glm5_next.py
mindspeed_mm/fsdp/models/glm5_next/video_processing_glm5_next.py
```

请勿直接使用模型仓库中的同名 Python 文件覆盖上述文件；如需同步上游模型代码，应先保留本仓中的 FSDP2、KDA、DSA、EP、chunk loss 等适配逻辑。

如需加载已有权重，或断点续训, 请修改 `training.load`：

```yaml
training:
  load: /path/to/GLM-5.3-Flash
```

如果不加载权重（随机初始化跑通功能），需要把`load`整行注释掉，并将`load_rank0_and_broadcast`置为`false`。

<a id="jump3.2"></a>

### 2. 配置参数

以下配置项在 `examples/glm5.3_flash/glm53_flash_config.yaml` 中设置：

| 配置项 | 配置路径 | 参数说明 | 调整说明 |
|--------|----------|----------|----------|
| `expert_parallel_size` | `parallel` | EP专家并行度 | 值为1时不开启，仅对MoE模型生效 |
| `kda_implementation` | `model` | KDA实现选择 | `ascendc` / `triton` / `eager` |
| `causal_conv1d_implementation` | `model` | KDA短卷积实现选择 | `ascendc` / `triton` / `eager` |
| `dsa_implementation` | `model` | DSA实现选择 | `dense` / `sfa` |
| `indexer_implementation` | `model` | DSAindexer实现选择 | `eager` / `triton` |
| `recompute` | `features` | 重计算开关 | 开启后节省显存占用 |
| `enable_activation_offload` | `features` | 激活值异步卸载到Host侧内存开关 | 开启后降低Device显存占用，`apply_modules`指定需要开启该特性的module |
| `enable_chunk_loss` | `features` | chunk loss 开关 | 默认开启，需与 `chunkloss_plan` 配套 |

【数据目录配置】

根据实际情况修改 `glm53_flash_config.yaml` 中的数据路径：

```yaml
data:
  dataset_param:
    basic_parameters:
      dataset_dir: /home/user/data/
      dataset: mllm_format_llava_instruct_data.json
```

【模型保存加载配置】

根据实际情况配置 `training` 参数，包括保存路径、保存间隔和断点续训路径：

```yaml
training:
  load: /path/to/load_ckpt
  save: /path/to/save_ckpt
  save_interval: 1000
  save_format: dcp
```

【EP并行配置】

根据实际的需求配置`glm53_flash_config.yaml`中的`expert_parallel_size`（值为1时不开启EP）。

【性能优化配置】

- 重计算
  - 在`features.recompute`配置，`true`表示开启，`false`表示关闭。
  - 开启后可以节省显存占用
- [chunkloss](../../docs/zh/features/chunkloss.md)
  - 在`features.enable_chunk_loss`配置，`true`表示开启，`false`表示关闭
  - `features.chunkloss_plan.chunk_size`表示计算loss的时候在seq维度切分成大小为`chunk_size`的小块进行计算。
  - 开启后可以大幅降低loss计算时的显存尖刺，节省整体显存占用
- [async activation offload](../../docs/zh/features/async_activation_offload.md)
  - 在`features.enable_activation_offload`配置，`true`表示开启，`false`表示关闭
  - 开启后可以异步将重计算入口的激活值offload至host侧，在开启了重计算的场景下可以进一步节省显存。

【单机运行配置】

启动脚本默认按单机16卡配置：

```shell
NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
```

如需改为其他卡数，需要同步调整 `NPUS_PER_NODE`、`parallel.expert_parallel_size` 以及相关并行配置。

<a id="jump3.3"></a>

### 3. 受限场景支持

当前版本已验证的场景如下。更多场景正在支持中，敬请期待！

- **减层训练**：基于减层、减专家模型配置进行训练验证；
  - 调整层数：修改模型配置路径下 `config.json` 中的 `num_hidden_layers` 字段, 减层时需同步调整`mlp_layer_types`、`layer_types`、`indexer_types`、`linear_attn_config.kda_layers`、`linear_attn_config.full_attn_layers`等字段，建议至少保留4层以上,包含KDA+DSA结构。
  - 调整专家个数：修改 `config.json` 中的 `num_experts` 字段，注意需与 `glm5.3_flash_config.yaml` 中的 `expert_parallel_size` 配套调整（专家个数需能被EP并行度整除）；
  - 参考配置：当前可配置 `num_hidden_layers=4`、`num_experts=288`；
- **序列长度**：mbs=1时支持16k序列长度以下；
- **ep**: 不减专家数时,需开启专家并行,避免OOM。

<a id="jump3.4"></a>

### 4. 启动训练

完成环境、模型路径和数据路径配置后，执行：

```shell
cd MindSpeed-MM
bash examples/glm5.3_flash/finetune_glm53_flash.sh
```

训练日志默认写入 `logs/train_${logfile}.log`，脚本结束后会统计平均 step time 和 samples per second。

---
<a id="jump4"></a>

## 环境变量声明

| 环境变量 | 描述 | 取值说明 |
|----------|------|----------|
| `ASCEND_LAUNCH_BLOCKING` | NPU 算子同步执行调试开关 | 默认 `0` |
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                                    | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                              |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                    | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                                     |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                                  | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                          | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征                                            |
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用                                                               |
| `TRITON_ALWAYS_COMPILE`       | 控制Triton算子是否总是重新编译 | `0`: 命中编译缓存时不重复编译<br>`1`: 每次运行强制重新编译（一般用于算子调试） |

---

<a id="jump5"></a>

## 注意事项

1. 启动脚本中的CANN路径为示例环境路径，使用前需改为当前机器实际路径。
2. 默认配置使用16卡单机训练，运行前请检查`npu-smi info`，避免占用他人任务。
3. 默认`dsa_implementation: sfa`依赖`torch_npu.npu_sparse_flash_attention`；如果当前 TorchNPU 不支持该接口，可先切换为 `dense` 做功能验证。
