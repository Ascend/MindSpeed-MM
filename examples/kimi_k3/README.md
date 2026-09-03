# Kimi-K3 使用指南

<p align="left">
</p>

## 目录

- [Kimi-K3 使用指南](#kimi-k3-使用指南)
  - [目录](#目录)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
    - [变更记录](#变更记录)
  - [环境安装](#环境安装)
    - [1. 环境准备](#1-环境准备)
    - [2. 环境搭建](#2-环境搭建)
    - [3. 安装配套版本的Triton-Ascend](#3-安装配套版本的triton-ascend)
    - [4. 安装fla-npu以适配AscendC](#4-安装fla-npu以适配ascendc)
    - [5. 安装ops-nn以适配AscendC版本SituGLU](#5-安装ops-nn以适配ascendc版本situglu)
    - [6. 安装ops-transformer以适配AscendC版本attn\_res](#6-安装ops-transformer以适配ascendc版本attn_res)
    - [7. 安装MindSpeed-Ops以支持Triton版本causal conv1d](#7-安装mindspeed-ops以支持triton版本causal-conv1d)
  - [数据集准备及处理](#数据集准备及处理)
  - [训练](#训练)
    - [1. 准备工作](#1-准备工作)
    - [2. 配置参数](#2-配置参数)
    - [3. 受限场景支持](#3-受限场景支持)
    - [4. 启动训练](#4-启动训练)
  - [环境变量声明](#环境变量声明)
  - [注意事项](#注意事项)

## 版本说明

> 当前版本仅支持减层受限场景训练，使用前请先阅读[受限场景支持](#3-受限场景支持)章节。
> 更多能力正在支持，敬请期待！

### 参考实现

```shell
url=https://huggingface.co/moonshotai/Kimi-K3/tree/main
```

### 变更记录

2026.07.26: 首次支持Kimi-K3模型

2026.08.27: 新增SituGLU、attn_res AscendC融合算子支持及配套安装说明

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/tree/master/docs/zh/pytorch/install_guide.md)，完成昇腾软件安装。

<a id="jump1.2"></a>

### 2. 环境搭建

拉取MindSpeed MM代码仓，并进入代码仓根目录：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

执行如下指令一键安装：

```bash
bash scripts/install.sh --msbranch master && pip install tiktoken==0.12.0  transformers==4.56.2
```

### 3. 安装配套版本的Triton-Ascend

Kimi-K3 的 KDA（Kimi Delta Attention）等线性注意力融合算子基于 Triton 实现，在昇腾环境下需要安装配套版本的 Triton-Ascend，请参考《Triton-Ascend》中的"[通过pip安装Triton-Ascend](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#piptriton-ascend)"章节，获取配套版本的Triton-Ascend安装指令。

KDA 算子实现依赖 `triton-ascend-kernels` 算子库（`modeling_kimi_linear.py` 中的 `chunk_kda` 来自该包）安装步骤如下：

atlas A2&A3训练产品安装步骤

```shell
# 拉取 triton-ascend-kernels 代码仓
git clone https://gitcode.com/fengrui886/triton-ascend-kernels
cd triton-ascend-kernels
git checkout kda_a3

# 安装
pip install -e . --no-build-isolation --no-deps
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
```

### 4. 安装fla-npu以适配AscendC

Kimi-K3 的 KDA 短卷积算子（`causal_conv1d_implementation: ascendc`）以及KDA算子（`kimi_delta_attention: ascendc`）基于 fla-npu 的 AscendC 融合算子实现，需要安装 fla-npu。注意：`kimi_delta_attention: ascendc` 算子当前仅支持ascend910_93, 暂不支持950系列（支持中）。

拉取flash-linear-attention-npu代码仓，并进入代码仓根目录，切到对应commitID

```bash
git clone https://github.com/Ensley0304/flash-linear-attention-npu.git
cd flash-linear-attention-npu
git checkout 0af57d0f7bd
```

安装步骤：可参考fla-npu仓README：[flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu/blob/release/v26.1.0/README.md)

推荐使用以下安装命令

```shell
# source 实际的cann路径
source /usr/local/Ascend/cann/set_env.sh

# 编译算子 run 包，--soc 需指定为当前机器芯片类型 {ascend910b/ascend910_93/ascend950}
bash build.sh --soc=ascend910_93 --pkg --vendor_name=fla_npu
bash build_out/fla-npu-*.run
cd torch_custom/fla_npu/
bash build.sh

# 导入环境变量
FLA_NPU_PATH=$(python3 -c "import fla_npu, os; print(os.path.dirname(fla_npu.__file__))")
export ASCEND_CUSTOM_OPP_PATH="${FLA_NPU_PATH}/opp/vendors/fla_npu_transformer:${FLA_NPU_PATH}/opp/vendors/fla_npu_transformer/op_api/lib:${ASCEND_CUSTOM_OPP_PATH}"

```

检验fla_npu是否安装成功

```bash
pip list | grep fla_npu
```

### 5. 安装ops-nn以适配AscendC版本SituGLU

Kimi-K3 的 MLP 激活采用 SituGLU 结构：对 gate 分支施加 SiTU 激活 `beta * tanh(gate / beta) * sigmoid(gate)`，再与 up 分支逐元素相乘；当模型配置 `linear_beta` 时，up 分支会先经过 `linear_beta * tanh(up / linear_beta)` 变换。SiTU 可视为带饱和门控的 SiLU（Swish）推广形式：`beta` 为输出的饱和界，当 `beta` 较大时 SiTU 近似退化为 SiLU。

`situ_glu_implementation: ascendc` 时，上述"激活 + 门控乘"计算由 ops-nn 仓的 AscendC 融合算子一次完成（前向 `situ_glu`、反向 `situ_glu_grad`），算子介绍见：[ops-nn situ_glu](https://gitcode.com/cann/ops-nn/tree/master/activation/situ_glu)；`triton` 使用仓内 Triton 融合算子（`mindspeed_mm/fsdp/ops/glu/situ_triton.py`，依赖 Triton-Ascend）；`eager` 为 torch 小算子实现，可用于功能对齐验证。

使用 AscendC 版本 SituGLU 需安装 ops-nn，安装步骤如下：

```shell
# 拉取代码
git clone https://gitcode.com/cann/ops-nn.git && cd ops-nn
git checkout b7af75e68cc07
# 根据芯片类型进行编译
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --soc=${soc_version} --ops=situ_glu,situ_glu_grad -j16
# 安装算子包
./build_out/*.run
cd ops-nn
bash build.sh --torch_extension
pip install build_out/*.whl
# 使用需导入安装过程中提示的环境变量
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/custom_nn/op_api/lib/:${LD_LIBRARY_PATH}
```

产品名对应的`${soc_version}`取值如下，请按实际场景传参：

- Atlas A2 训练系列产品/Atlas A2 推理系列产品：取值为ascend910b
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：取值为ascend910_93
- 950系列产品：取值为ascend950

### 6. 安装ops-transformer以适配AscendC版本attn_res

Kimi-K3 采用 attn_res（attention residual）机制替代传统 Transformer 的固定残差连接：模型每 `attn_res_block_size` 层将当前 hidden states 追加到 `block_residual` 残差历史中；在每层的注意力与 MLP 计算前（以及模型输出位置），将残差历史与当前前缀和拼接，经 RMSNorm 归一化后，用可学习的投影权重对每个历史块打分，再经 softmax 加权融合得到该位置的残差输入，实现跨层残差路径的软选择。

`attn_res_implementation: ascendc` 时，上述"归一化 + 打分 + 加权融合"计算使用 ops-transformer 仓的 AscendC 融合算子完成，需安装 ops-transformer；`eager` 为 torch 小算子实现，可用于功能对齐验证。

> [!NOTE]
>
> attn_res 算子代码位于 ops-transformer 仓的 MR9720 分支，安装时需先按如下步骤检出该MR分支。

安装步骤如下：

```shell
# 拉取代码
git clone https://gitcode.com/cann/ops-transformer.git && cd ops-transformer
# 检出attn_res算子所在的MR分支
git fetch https://gitcode.com/cann/ops-transformer.git +refs/merge-requests/9720/head:pr_9720
git checkout pr_9720
# 根据芯片类型进行编译
source /usr/local/Ascend/cann/set_env.sh
bash build_install_attn_res.sh --soc=${soc_version}
# 安装算子包
conda activate ${conda_env_name} # 激活对应conda环境
pip install torch_extension/dist/*.whl --force-reinstall --no-deps
# 导入安装过程中提示的环境变量 
source xxx/ops-transformer/install/vendors/custom_transformer/bin/set_env.bash
```

产品名对应的`${soc_version}`取值如下，请按实际场景传参：

- Atlas A2 训练系列产品/Atlas A2 推理系列产品：取值为ascend910b
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：取值为ascend910_93
- 950系列产品：取值为ascend950

### 7. 安装MindSpeed-Ops以支持Triton版本causal conv1d

Kimi-K3 的 KDA 短卷积算子在 `causal_conv1d_implementation: triton` 时使用 MindSpeed-Ops 仓提供的 Triton 版本 causal conv1d 融合算子实现，需安装 MindSpeed-Ops。

安装步骤如下：

```shell
# 拉取代码
git clone https://gitcode.com/Ascend/MindSpeed-Ops.git && cd MindSpeed-Ops
# 安装
pip install -e . --no-build-isolation --no-deps
```

---

<a id="jump2"></a>

## 数据集准备及处理

- 使用**真实数据集**训练：参考[针对VL模型的数据构造 · 使用真实数据集](../../docs/zh/features/building_data_for_VLModel.md#real-data)（下载COCO2017 → 下载LLaVA-Instruct-150K标注 → 运行转换脚本生成`mllm_format_llava_instruct_data.json`）。
- 使用**虚构数据**做功能/性能测试：参考[针对VL模型的数据构造 · 使用虚构数据](../../docs/zh/features/building_data_for_VLModel.md#mock-data)。

## 训练

<a id="jump3.1"></a>

### 1. 准备工作

从Huggingface库下载模型文件，并将下列文件放置于本地`mindspeed_mm/fsdp/models/kimi_k3`路径下：

> [!NOTE]
>
> 如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

```shell
# HF_PATH配置为HuggingFace库下载文件的存放路径
HF_PATH="/download/Kimi-K3"
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
  ${MM_PATH}/mindspeed_mm/fsdp/models/kimi_k3/
cd ${MM_PATH}
```

> **说明**：代码仓中已包含适配 MindSpeed-MM FSDP2 的模型实现文件（`modeling_kimi_k3.py`、`modeling_kimi_linear.py`、`tokenization_kimi.py`、`kimi_moe_patch.py`），请勿使用模型仓库中的同名文件覆盖。

Kimi-K3 模型需要配置多机训练，如需拉起多机训练，请修改启动脚本下的 `MASTER_ADDR`、`NNODES` 以及 `NODE_RANK` 变量：

``` shell
MASTER_ADDR: 主节点IP地址
NNODES: 总节点数量
NODE_RANK: 当前节点序号
```

配置脚本前需要完成前置准备工作，包括：**环境安装**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump3.2"></a>

### 2. 配置参数

以下配置项在 `kimik3_config.yaml` 中设置：

| 配置项 | 配置路径 | 参数说明 | 调整说明 |
|--------|----------|----------|----------|
| `ulysses_parallel_size` | `parallel` | ulysses-cp 并行度 | 值为1时不开启，根据实际情况调整；|
| `expert_parallel_size` | `parallel` | EP专家并行度 | 值为1时不开启，仅对MoE模型生效 |
| `ep_plan` | `parallel` | EP调度策略配置 | 包含`dispatcher`、`use_npu_fused_ops`等子字段，`dispatcher`可选`alltoall` |
| `num_to_forward_prefetch` | `parallel->fsdp_plan` | 前向计算时预取后续层参数 | 减少通信等待开销 |
| `num_to_backward_prefetch` | `parallel->fsdp_plan` | 反向计算时预取后续层参数 | 减少通信等待开销 |
| `enable_preload` | `data->dataloader_param` | 数据预加载开关 | 开启后数据加载与计算重叠，减少训练等待时间 |
| `use_grouped_expert_matmul` | `model` | MoE专家分组矩阵乘融合算子开关 | 开启后使用NPU融合算子加速MoE专家计算 |
| `kda_implementation` | `model` | KDA算子实现选择 | `triton`: triton-ascend-kernels融合算子（默认），需安装Triton-Ascend及配套算子库<br>`ascendc`: AscendC融合算子（仅NPU）<br>`eager`: 仓内小算子实现，可用于功能对齐验证 |
| `causal_conv1d_implementation` | `model` | KDA短卷积算子实现选择 | `triton`: triton实现（默认）<br>`ascendc`: fla_npu AscendC融合算子（仅NPU），需安装fla-npu，参考[安装fla-npu以适配AscendC](#4-安装fla-npu以适配ascendc) |
| `situ_glu_implementation` | `model` | SituGLU算子实现选择 | `ascendc`: ops-nn AscendC融合算子（默认，仅NPU），需安装ops-nn，参考[安装ops-nn以适配AscendC版本SituGLU](#5-安装ops-nn以适配ascendc版本situglu)<br>`triton`: 仓内Triton融合算子（依赖Triton-Ascend）<br>`eager`: torch小算子实现，可用于功能对齐验证 |
| `attn_res_implementation` | `model` | attn_res算子实现选择 | `ascendc`: ops-transformer AscendC融合算子（默认，仅NPU），需安装ops-transformer，参考[安装ops-transformer以适配AscendC版本attn_res](#6-安装ops-transformer以适配ascendc版本attn_res)<br>`eager`: torch小算子实现，可用于功能对齐验证 |
| `skip_flash_attn_recompute` | `model` | 跳过full attention层flash attention重计算 | 选择性重计算，需同时使能重计算和`enable_activation_offload` |
| `skip_kda_recompute` | `model` | 跳过linear attention层KDA重计算 | 选择性重计算，需同时使能重计算和`enable_activation_offload` |
| `recompute` | `features` | 重计算开关 | 开启后可以节省显存占用 |
| `enable_activation_offload` | `features` | 激活值异步卸载到Host侧内存开关 | 开启后降低Device显存占用，`apply_modules`指定需要开启该特性的module |
| `enable_chunk_loss` | `features` | chunkloss特性开关 | 需与`chunkloss_plan`关联使用，开启后大幅降低loss计算时的显存尖刺，详细说明请参考[chunkloss文档](../../docs/zh/features/chunkloss.md) |
| `enable_chunk_mbs` | `features` | 是否开启chunkmbs特性 | 需与`chunkmbs_plan`关联使用，开启后将MicroBatch维度切分为多个微块依次计算，可压缩激活显存峰值并提升训练吞吐，详细说明请参考[chunkmbs文档](../../docs/zh/features/chunkmbs.md) |

【数据目录配置】

根据实际情况修改`kimik3_config.yaml`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

示例：如果数据及其对应的json都在/home/user/data/目录下，其中json目录为/home/user/data/mllm_format_llava_instruct_data.json，此时配置如下：
`dataset_dir`配置为/home/user/data/;
`dataset`配置为./data/mllm_format_llava_instruct_data.json
注意此时`dataset`需要配置为相对路径

【模块冻结配置】

当前支持自定义冻结模块，在`kimik3_config.yaml`中model->freeze字段中配置需要冻结的模块即可实现相应模块冻结。

【模型保存加载配置】

根据实际情况配置`kimik3_config.yaml`的`training`参数，包括保存路径以及保存间隔`save`、`save_interval`，断点续训场景配置`load`为checkpoint路径；`load_format`/`save_format`支持`hf`和`dcp`两种格式，配置为`auto`时自动识别。

【EP并行配置】

根据实际的需求配置`kimik3_config.yaml`中的`expert_parallel_size`（值为1时不开启EP）。

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
- [chunkmbs](../../docs/zh/features/chunkmbs.md)
  - 在`features.enable_chunk_mbs`配置，`true`表示开启，`false`表示关闭
  - `features.chunkmbs_plan.chunk_mbs`表示切分以后单次计算的`micro_batch_size`
  - 开启该特性时需要同时使能重计算和async activation offload特性，可以增加FSDP2单次unshard对应的计算密度，提高整网吞吐。
- 选择性重计算
  - 在开启重计算的场景下，可以跳过linear attention层的KDA重计算，或者full attention层的flash attention重计算，并异步offload中间保存的tensor，在显存占用不变的条件下，减少计算量，提升训练吞吐
  - 在`model.skip_kda_recompute`配置是否跳过linear attention层KDA的重计算，`true`表示跳过，`false`表示不跳过
  - 在`model.skip_flash_attn_recompute`配置是否跳过full attention层的flash attention的重计算，`true`表示跳过，`false`表示不跳过
  - 开启该特性时需要同时使能重计算和async activation offload特性
- MoE融合算子
  - 在`model.use_grouped_expert_matmul`配置，`true`表示开启，`false`表示关闭
  - 开启后MoE路由专家权重以3-D tensor组织，使用NPU grouped GEMM/permute/unpermute融合算子加速专家计算

【单机运行配置】

配置`examples/kimi_k3/finetune_kimik3.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPROC_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6087
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPROC_PER_NODE*$NNODES))
```

【多机运行配置】

如需拉起多机训练，修改启动脚本下 MASTER_ADDR、NNODES以及NODE_RANK变量

``` shell
MASTER_ADDR: 主节点IP地址
NODE_RANK: 第几个节点
NNODES: 一共几个节点
```

<a id="jump3.3"></a>

### 3. 受限场景支持

当前版本已验证的场景如下。更多场景正在支持中，敬请期待！

- **减层训练**：基于减层、减专家模型配置进行训练验证；
  - 调整层数：修改模型配置路径 `mindspeed_mm/fsdp/models/kimi_k3` 下 `config.json` 中的 `num_hidden_layers` 字段；
  - 调整专家个数：修改 `config.json` 中的 `num_experts` 字段，注意需与 `kimik3_config.yaml` 中的 `expert_parallel_size` 配套调整（专家个数需能被EP并行度整除）；
  - 参考配置：当前 A3 单节点可配置 `num_hidden_layers=16`、`num_experts=32`；
- **序列长度**：mbs=1时支持6k序列长度以下；
- **权重加载**：当前采用随机初始化权重（加载预训练权重能力后续支持）；
- **CP 长序列训练**：支持 ulysses-cp 长序列训练，配置 `kimik3_config.yaml` 中 `parallel->ulysses_parallel_size` 调整并行度（值为1时不开启）。

<a id="jump3.4"></a>

### 4. 启动训练

(1) 修改 `kimik3_config.yaml` 中 `data->dataset_param->basic_parameters->dataset` 字段，配置实际的数据集路径；

(2) 启动训练（当前仅支持减层减专家场景）：

```shell
bash examples/kimi_k3/finetune_kimik3.sh
```

<a id="jump4"></a>

## 环境变量声明

| 环境变量                      | 描述                                                                 | 取值说明                                                                                         |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                                    | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                              |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                    | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                                     |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                                  | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                          | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征                                            |
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用                                                               |
| `TRITON_ALWAYS_COMPILE`       | 控制Triton算子是否总是重新编译 | `0`: 命中编译缓存时不重复编译<br>`1`: 每次运行强制重新编译（一般用于算子调试） |

---

<a id="jump5"></a>

## 注意事项
