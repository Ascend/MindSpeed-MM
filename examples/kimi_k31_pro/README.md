# Kimi-K3.1-Pro 使用指南

<p align="left">
</p>

## 目录

- [Kimi-K3.1-Pro 使用指南](#kimi-k31-pro-使用指南)
  - [目录](#目录)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
    - [变更记录](#变更记录)
  - [环境安装](#环境安装)
    - [1. 环境准备](#1-环境准备)
    - [2. 环境搭建](#2-环境搭建)
    - [3. 安装配套版本的Triton-Ascend](#3-安装配套版本的triton-ascend)
    - [4. 安装fla-npu以适配AscendC](#4-安装fla-npu以适配ascendc)
    - [5. 安装ops-transformer以适配AscendC版本DSA](#5-安装ops-transformer以适配ascendc版本dsa)
    - [6. 安装MindSpeed-Ops以支持Triton版本causal conv1d](#6-安装mindspeed-ops以支持triton版本causal-conv1d)
  - [权重下载及转换](#权重下载及转换)
    - [1. 权重下载](#1-权重下载)
    - [2. 反量化为 BF16 格式](#2-反量化为-bf16-格式)
    - [3. 转换为 DCP 格式](#3-转换为-dcp-格式)
  - [数据集准备及处理](#数据集准备及处理)
  - [训练](#训练)
    - [1. 准备工作](#1-准备工作)
    - [2. 配置参数](#2-配置参数)
    - [3. 受限场景支持](#3-受限场景支持)
    - [4. 启动训练](#4-启动训练)
    - [5. 分布式 LoRA 微调](#5-分布式-lora-微调)
  - [环境变量声明](#环境变量声明)
  - [注意事项](#注意事项)

## 版本说明

> 当前版本已完成减层、减专家场景的训练验证和完整参数的LoRA微调训练验证，使用前请先阅读[受限场景支持](#3-受限场景支持)章节。
> 更多能力正在支持，敬请期待！

Kimi-K3.1-Pro 复用 Kimi-K3 的视觉模块、KDA、MoE 和 FSDP2 训练链路，主要结构差异是 full-attention 层由 MLA 改为带 Lightning Indexer 的 DSA（Gated Sparse MLA）。模型插件位于 `mindspeed_mm/fsdp/models/kimi_k31_pro`，注册名为 `kimi_k31_pro`。

### 参考实现

```shell
url=https://huggingface.co/moonshotai/Kimi-K3.1-Pro/tree/main
```

### 变更记录

2026.11.04: 首次支持Kimi-K3.1-Pro模型

---

<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/tree/master/docs/zh/guides/installation/install_guide.md)，完成昇腾软件安装。

<a id="jump1.2"></a>

### 2. 环境搭建

拉取MindSpeed MM代码仓，并进入代码仓根目录：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

执行如下指令安装基础依赖：

```bash
bash scripts/install.sh --msbranch master
python -m pip install --upgrade --no-deps torch-npu==2.10.0.post6
python -m pip install tiktoken==0.12.0 "transformers>=4.57.6"
```

模型配置由 Transformers 4.57.6 生成，建议使用该版本或兼容的更新版本。推荐使用 CANN 9.2.0 或更高版本。

### 3. 安装配套版本的Triton-Ascend

Kimi-K3.1-Pro 的 KDA（Kimi Delta Attention）线性注意力算子在 `kda_implementation: triton` 时基于 Triton 实现，需要安装配套版本的 Triton-Ascend。请参考《Triton-Ascend》中的[通过pip安装Triton-Ascend](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#piptriton-ascend)章节，获取配套版本的安装指令。

KDA 算子依赖 `triton-ascend-kernels` 算子库，安装步骤如下：

```shell
git clone https://gitcode.com/Ascend/triton-ascend-kernels.git
cd triton-ascend-kernels
python -m pip install -e . --no-build-isolation --no-deps
```

KDA 短卷积在 `causal_conv1d_implementation: triton` 时还需安装 MindSpeed-Ops，详见[安装MindSpeed-Ops以支持Triton版本causal conv1d](#6-安装mindspeed-ops以支持triton版本causal-conv1d)。

### 4. 安装fla-npu以适配AscendC

Kimi-K3.1-Pro 的 KDA 短卷积算子（`causal_conv1d_implementation: ascendc`）以及 KDA 算子（`kda_implementation: ascendc`）基于 fla-npu 的 AscendC 融合算子实现，需要安装 fla-npu。构建 wheel 时须按目标芯片选择 SoC：A2 使用 `ascend910b`，A3 使用 `ascend910_93`，昇腾950系列使用 `ascend950`。

拉取flash-linear-attention-npu代码仓，并进入代码仓根目录：

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu.git
cd flash-linear-attention-npu

# 昇腾950系列使用 commit id：90176b44321e，A3&A2 使用 commit id：e22e0bf
export FLA_NPU_COMMIT=e22e0bf  # A3/A2；昇腾950系列改为 90176b44321e
git checkout ${FLA_NPU_COMMIT}
git rev-parse HEAD
```

请确保构建环境已安装 `gawk` 和 `cmake>=3.16`，否则环境检查或编译可能失败：

```shell
# Ubuntu / Debian
apt-get update
apt-get install -y gawk
# openEuler / CentOS / RHEL
yum install -y gawk

python -m pip install "cmake>=3.16"
cmake --version
```

推荐使用以下安装命令：

```shell
# source 实际的CANN路径
source /usr/local/Ascend/cann/set_env.sh

python -m pip install -r requirements.txt
python scripts/check_npu_env.py

# 按训练设备选择目标SoC，以下命令以昇腾950系列为例
FLA_NPU_SOC=ascend950 python scripts/build_wheel.py
```

`build_wheel.py` 会输出本次构建 wheel 的准确路径和安装命令，请按输出安装，避免使用通配符选中 `dist/` 中的旧 wheel。

检验fla_npu是否安装成功：

```bash
python -c "import fla_npu; from fla_npu.ops import ascendc; names = ('chunk_fwd_o', 'chunk_kda_fwd', 'chunk_kda_bwd'); missing = [name for name in names if not hasattr(ascendc, name)]; assert not missing, f'missing fla-npu APIs: {missing}'; print('fla-npu API check passed.')"
```

### 5. 安装ops-transformer以适配AscendC版本DSA

Kimi-K3.1-Pro 的 DSA 由 Lightning Indexer 和稀疏 MLA 组成：Indexer 为每个 query 选择 Top-K key，稀疏注意力基于共享 latent K/V 完成吸收式 MLA 计算。

`dsa_implementation: ascendc` 时使用 ops-transformer 的以下算子：

- `lightning_indexer`：计算Top-K索引；
- `sparse_flash_attention`：执行稀疏注意力前向；
- `sparse_flash_attention_grad`：执行稀疏注意力反向。

`eager` 为 Torch 小算子实现，不依赖上述自定义算子，可用于功能对齐验证。

安装步骤如下：

```shell
git clone https://gitcode.com/cann/ops-transformer.git
cd ops-transformer

# 根据芯片类型进行编译
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --soc=${soc_version} \
  --ops=lightning_indexer,sparse_flash_attention,sparse_flash_attention_grad -j16
bash build/cann-ops-transformer-custom_linux-aarch64.run

# 编译安装torch依赖
bash build.sh --torch_extension
python -m pip install torch_extension/dist/*.whl --force-reinstall --no-deps

# 使用安装过程中提示的实际动态库路径
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

产品名对应的`${soc_version}`取值如下，请按实际场景传参：

- Atlas A2 训练系列产品/Atlas A2 推理系列产品：取值为ascend910b
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：取值为ascend910_93
- 950系列产品：取值为ascend950

实际安装包名称和动态库路径以构建输出为准。

### 6. 安装MindSpeed-Ops以支持Triton版本causal conv1d

Kimi-K3.1-Pro 的 KDA 短卷积算子在 `causal_conv1d_implementation: triton` 时使用 MindSpeed-Ops 仓提供的 Triton 版本 causal conv1d 融合算子实现，需安装 MindSpeed-Ops：

```shell
git clone https://gitcode.com/Ascend/MindSpeed-Ops.git
cd MindSpeed-Ops
python -m pip install -e . --no-build-isolation --no-deps
```

---

<a id="jump25"></a>

## 权重下载及转换

<a id="jump25.1"></a>

### 1. 权重下载

从 Hugging Face 下载 Kimi K3.1 Pro 开源模型权重：https://huggingface.co/moonshotai/Kimi-K3.1-Pro/tree/main

> [!NOTE]
>
> 如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

<a id="jump25.2"></a>

### 2. 反量化为 BF16 格式

Kimi-K3.1-Pro 开源权重为 MXFP4 量化格式，训练前需反量化为 BF16：

```shell
MXFP4_MODEL_PATH=/mnt/weight/moonshotai/Kimi-K3.1-Pro
BF16_MODEL_PATH=/mnt/weight/moonshotai/Kimi-K3.1-Pro-bf16

python examples/kimi_k31_pro/convert_kimi_k31_mxfp4_to_bf16_mp.py \
    --source ${MXFP4_MODEL_PATH} \
    --output ${BF16_MODEL_PATH} \
    --workers 8 \
    --overwrite
```

<a id="jump25.3"></a>

### 3. 转换为 DCP 格式

完成 BF16 权重转换后，进一步将反量化后的 Safetensors 权重转换为适配 FSDP2 分布式训练的 DCP（Distributed Checkpoint）格式，完成模型权重加载格式适配，为多机多卡训练提供支持：

```shell
python checkpoint/vlm_model/converters/kimi_k31.py \
    --src-dir /mnt/weight/moonshotai/Kimi-K3.1-Pro-bf16 \
    --dst-dir /mnt/weight/moonshotai/Kimi-K3.1-Pro-bf16-fsdp128-ep64 \
    --fsdp-shards 128 \
    --ep-size 64 \
    --workers 8 \
    --read-threads 4
```

转换完成后，在训练配置中将`training.load`配置为转换后的 DCP 权重目录，`training.load_format`配置为`dcp`。

---

<a id="jump2"></a>

## 数据集准备及处理

数据链路使用本地 JSON 的 `huggingface` 数据集插件、`kimi_k3` template/mm_plugin 和 `qwen3vl` collator。

- 使用**真实数据集**训练：参考[针对VL模型的数据构造 · 使用真实数据集](../../docs/zh/features/data/building_data_for_VLModel.md#real-data)（下载COCO2017 → 下载LLaVA-Instruct-150K标注 → 运行转换脚本生成`mllm_format_llava_instruct_data.json`）。
- 使用**虚构数据**做功能/性能测试：参考[针对VL模型的数据构造 · 使用虚构数据](../../docs/zh/features/data/building_data_for_VLModel.md#mock-data)。

## 训练

<a id="jump3.1"></a>

### 1. 准备工作

从Huggingface库下载模型文件，并将下列文件放置于本地`mindspeed_mm/fsdp/models/kimi_k31_pro`路径下：

> [!NOTE]
>
> 如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

```shell
# HF_PATH配置为HuggingFace库下载文件的存放路径
HF_PATH="/download/Kimi-K3.1-Pro"
# MM_PATH配置为MindSpeed-MM根目录路径
MM_PATH="/home/workspace/MindSpeed-MM"

cd ${HF_PATH}
cp -f \
  config.json \
  configuration_kimi_k3.py \
  encoding_k3.py \
  generation_config.json \
  kimi_k3_processor.py \
  kimi_k3_vision_processing.py \
  media_utils.py \
  preprocessor_config.json \
  tiktoken.model \
  tokenizer_config.json \
  ${MM_PATH}/mindspeed_mm/fsdp/models/kimi_k31_pro/
cd ${MM_PATH}
```

> **说明**：代码仓中已包含适配 MindSpeed-MM FSDP2 的 DSA 实现 `modeling_kimi_linear.py`、插件入口 `__init__.py` 和 tokenizer 文件 `tokenization_kimi.py`，其余模型主体、MoE 和训练适配逻辑复用 `mindspeed_mm/fsdp/models/kimi_k3`，请勿使用其他模型仓库中的同名文件覆盖。

加载原始 HF 权重时，将 `training.load` 配置为权重目录；使用 DCP 时，先参考仓内检查点文档转换，再设置 `training.init_model_with_meta_device: true` 和对应的 `training.load` 路径。

如需多机训练，请修改启动脚本下的 `MASTER_ADDR`、`NNODES` 以及 `NODE_RANK` 变量：

```shell
MASTER_ADDR: 主节点IP地址
NNODES: 总节点数量
NODE_RANK: 当前节点序号
```

配置脚本前需要完成前置准备工作，包括：**环境安装**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump3.2"></a>

### 2. 配置参数

以下配置项在 `kimik31_pro_config.yaml` 中设置：

| 配置项 | 配置路径 | 参数说明 | 调整说明 |
|--------|----------|----------|----------|
| `model_id` | `model` | 模型注册名 | 固定为 `kimi_k31_pro` |
| `model_name_or_path` | `model` | 模型配置、tokenizer和processor路径 | 默认为插件目录 |
| `ulysses_parallel_size` | `parallel` | ulysses-cp并行度 | DSA当前仅支持值为1 |
| `expert_parallel_size` | `parallel` | EP专家并行度 | 值为1时不开启，需能整除专家数 |
| `ep_plan` | `parallel` | EP调度策略配置 | `dispatcher`可选`alltoall`、`mc2`、`allgather`，默认配置使用`alltoall` |
| `num_to_forward_prefetch` | `parallel->fsdp_plan` | 前向计算时预取后续层参数 | 减少通信等待开销 |
| `num_to_backward_prefetch` | `parallel->fsdp_plan` | 反向计算时预取后续层参数 | 减少通信等待开销 |
| `enable_preload` | `data->dataloader_param` | 数据预加载开关 | 开启后数据加载与计算重叠 |
| `use_grouped_expert_matmul` | `model` | MoE专家分组矩阵乘融合算子开关 | 开启后使用NPU融合算子加速MoE专家计算 |
| `kda_implementation` | `model` | KDA算子实现选择 | `ascendc`：fla-npu融合算子（默认）<br>`triton`：triton-ascend-kernels融合算子<br>`eager`：Torch小算子实现 |
| `causal_conv1d_implementation` | `model` | KDA短卷积算子实现选择 | `ascendc`：fla-npu AscendC融合算子（默认）<br>`triton`：MindSpeed-Ops Triton融合算子 |
| `dsa_implementation` | `model` | DSA算子实现选择 | `ascendc`：ops-transformer LI+SFA/SFAG融合算子（默认）<br>`eager`：Torch小算子实现，可用于功能对齐 |
| `situ_glu_implementation` | `model` | SituGLU算子实现选择 | 默认`ascendc`，最新CANN已集成 |
| `attn_res_implementation` | `model` | attn_res算子实现选择 | 默认`ascendc`，最新CANN已集成 |
| `skip_flash_attn_recompute` | `model` | 跳过attention层flash attention重计算 | 选择性重计算，需同时使能重计算和`enable_activation_offload` |
| `skip_kda_recompute` | `model` | 跳过linear attention层KDA重计算 | 选择性重计算，需同时使能重计算和`enable_activation_offload` |
| `recompute` | `features` | 重计算开关 | 开启后可以节省显存占用 |
| `enable_activation_offload` | `features` | 激活值异步卸载到Host侧内存开关 | 开启后降低Device显存占用 |
| `enable_chunk_loss` | `features` | chunkloss特性开关 | 需与`chunkloss_plan`关联使用 |
| `enable_chunk_mbs` | `features` | chunkmbs特性开关 | 需与`chunkmbs_plan`关联使用 |

【数据目录配置】

根据实际情况修改`kimik31_pro_config.yaml`中的`dataset_dir`、`dataset`等字段。

示例：若图片解压到`/home/user/data/coco/COCO2017/train2017/`，标注文件保存为`/home/user/data/coco/mllm_format_llava_instruct_data.json`，则：

- `dataset_dir`配置为`/home/user/data/coco/COCO2017`；
- `dataset`配置为`/home/user/data/coco/mllm_format_llava_instruct_data.json`。

> **注意**：运行时会将`dataset_dir`与标注 JSON 中的图片相对路径拼接，因此`dataset_dir`必须配置为图片实际所在目录的父目录，否则会报`FileNotFoundError`。

【模块冻结配置】

在`kimik31_pro_config.yaml`中`model->freeze`字段配置需要冻结的模块。

【模型保存加载配置】

根据实际情况配置`training`参数，包括保存路径、保存间隔`save`、`save_interval`；断点续训场景将`load`配置为checkpoint路径。`load_format`/`save_format`支持`hf`和`dcp`两种格式，配置为`auto`时自动识别。

【EP并行配置】

根据实际需求配置`expert_parallel_size`（值为1时不开启EP），并确保专家个数能被EP并行度整除。

【性能优化配置】

- **重计算**：在`features.recompute`配置，开启后可以节省显存占用。
- **chunkloss**：在`features.enable_chunk_loss`配置，可降低loss计算时的显存尖刺，详见[chunkloss文档](../../docs/zh/features/optimization/chunkloss.md)。
- **async activation offload**：在`features.enable_activation_offload`配置，可在重计算场景下进一步节省显存，详见[async activation offload](../../docs/zh/features/memory/async_activation_offload.md)。
- **chunkmbs**：在`features.enable_chunk_mbs`配置，可将MicroBatch切分为多个微块依次计算，详见[chunkmbs文档](../../docs/zh/features/optimization/chunkmbs.md)。
- **选择性重计算**：可通过Swap Core和Op Replay实现。Op Replay在前向时将`recompute_plan.op_replay_scopes`指定范围内的白名单算子输出换出，重计算时直接取回并跳过重算；Swap Core提供HBM与DDR之间的异步换入换出及显存预算管理。使用时需开启`features.recompute`，设置`recompute_plan.use_reentrant: false`，并通过`apply_modules`和`cache_ops`配置作用模块与算子白名单。详见[Swap Core](../../docs/zh/features/memory/swap_core.md)和[Op Replay](../../docs/zh/features/optimization/op_replay.md)文档。

【单机运行配置（昇腾950系列产品）】

在昇腾950系列产品上配置`examples/kimi_k31_pro/finetune_kimik31_pro.sh`参数如下：

```shell
# 根据实际情况修改ascend-toolkit路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPROC_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPROC_PER_NODE*$NNODES))
```

【多机运行配置】

如需拉起多机训练，修改启动脚本下的`MASTER_ADDR`、`NNODES`以及`NODE_RANK`变量。

<a id="jump3.3"></a>

### 3. 受限场景支持

当前版本已验证的场景如下。更多场景正在支持中，敬请期待！

- **减层减专家训练**：当前版本仅进行了至多8机验证，验证规格减少模型尺寸为28专家，尚未进行全层训练验证。
- **DSA长序列路径**：测试序列长度需大于`index_topk`，才能覆盖实际的稀疏Top-K路径。K3.1默认`index_topk=2048`。
- Top-K选择是离散操作，当前Indexer参数不从语言模型损失接收梯度，将会尽快支持。
- **完整参数LoRA微调**：当前版本进行了128卡Atlas 950 SuperPoD验证.

<a id="jump3.4"></a>

### 4. 启动训练

(1) 修改`kimik31_pro_config.yaml`中`data->dataset_param->basic_parameters->dataset`字段，配置实际的数据集路径；

(2) 启动训练：

```shell
bash examples/kimi_k31_pro/finetune_kimik31_pro.sh
```

<a id="jump3.5"></a>

### 5. 分布式 LoRA 微调

基于128卡Atlas 950 SuperPoD开展Kimi-K3.1-Pro LoRA微调，采用FSDP2+EP混合并行策略，实现超大规模模型的分布式部署与高效训练。

(1) 在`kimik31_pro_lora_128p.yaml`配置文件中完成模型权重、训练数据集、LoRA参数及分布式并行策略等配置（LoRA参数见`training.lora`字段，分布式并行策略见`parallel`字段）；

(2) 启动分布式LoRA微调：

```shell
bash examples/kimi_k31_pro/kimik31_pro_lora_128p.sh
```

<a id="jump4"></a>

## 环境变量声明

| 环境变量 | 描述 | 取值说明 |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `TASK_QUEUE_ENABLE` | 用于控制开启task_queue算子下发队列优化的等级 | `0`：关闭<br>`1`：开启Level 1优化<br>`2`：开启Level 2优化 |
| `CPU_AFFINITY_CONF` | 控制CPU端算子任务的处理器亲和性，即设定任务绑核 | 设置`0`或未设置：不启用绑核<br>`1`：开启粗粒度绑核<br>`2`：开启细粒度绑核 |
| `HCCL_CONNECT_TIMEOUT` | 用于限制不同设备之间socket建链过程的超时等待时间 | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s` |
| `PYTORCH_NPU_ALLOC_CONF` | 控制缓存分配器行为 | `expandable_segments:<value>`：使能内存池扩展段功能 |
| `MULTI_STREAM_MEMORY_REUSE` | 配置多流内存复用是否开启 | `0`：关闭多流内存复用<br>`1`：开启多流内存复用 |
| `TRITON_ALWAYS_COMPILE` | 控制Triton算子是否总是重新编译 | `0`：命中编译缓存时不重复编译<br>`1`：每次运行强制重新编译 |
| `NUM_HIDDEN_LAYERS` | 功能验证时覆盖语言模型层数 | 不设置时使用模型配置；仅在减层调试时设置 |
| `NUM_EXPERTS` | 功能验证时覆盖专家个数 | 不设置时使用模型配置；需与EP并行度配套 |
| `NUM_EXPERTS_PER_TOKEN` | 功能验证时覆盖每个token选择的专家数 | 不设置时使用模型配置 |

---

<a id="jump5"></a>

## 注意事项

- 当前适配面向训练，DSA不支持KV-cache推理和Ulysses context parallel，`ulysses_parallel_size`需保持为1。
