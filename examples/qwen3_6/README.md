# Qwen3_6 使用指南

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
  - [权重加载](#2-权重加载)
  - [权重保存](#3-权重保存)
- [数据集准备及处理](#数据集准备及处理)
  - [Agentical Trace（OpenAI 格式）数据集](#agentical-traceopenai-格式数据集)
- [微调](#微调)
  - [准备工作](#1-准备工作)
  - [配置参数](#2-配置参数)
  - [启动微调](#3-启动微调)
  - [LoRA 微调](#4-lora-微调)
- [环境变量声明](#环境变量声明)
- [注意事项](#注意事项)

## 版本说明

### 参考实现

```shell
url=https://github.com/huggingface/transformers.git
commit_id=7d9754a
```

### 变更记录

2026.04.17: 首次支持Qwen3.6-35B-A3B模型

环境变更记录：

Latest:  2026.9.16:  为适配 CANN9.1.0 和 torch 2.10.0，将 Triton-Ascend 参考版本改为 3.2.2，fla-npu参考版本保持v26.6.0

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](../../docs/zh/pytorch/install_guide.md)，完成 CANN 相关配置（驱动、固件及 Toolkit 工具包）。

<a id="jump1.2"></a>

### 2. 环境搭建

拉取MindSpeed MM代码仓，并进入代码仓根目录(若需要使用低版本的python，可参考[安装指南](../../docs/zh/pytorch/install_guide.md)的拉取方法)：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

若存在自由指定依赖等手动安装需求，可以参考[安装指南](../../docs/zh/pytorch/install_guide.md)中的手动安装流程（注：Qwen3.6不需要安装该流程中的Megatron-LM库）

执行如下指令一键安装：

```bash
bash scripts/install.sh --msbranch master && pip install transformers==5.2.0
```

这里会自动安装 torch 以及 torch_npu 库，版本可以通过参数自行选择（默认 2.10.0），示例如下：

```bash
bash scripts/install.sh --msbranch master --torchversion 2.10.0 && pip install transformers==5.2.0
```

请参考[版本配套说明](../../docs/zh/release_notes_mm.md)和[Triton-Ascend3.2.2适配版本](https://github.com/triton-lang/triton-ascend/releases)选择适配的 pytorch/torch_npu 版本

> [!NOTE]
> 当前一键安装脚本torch_npu版本与torch为严格对应（无post后缀），建议按照链接中的适配版本自行安装带后缀的torch版本，否则可能无法通过后续环境检测脚本

### 3. 安装配套版本的Triton-Ascend

安装配套版本的Triton-Ascend，请参考《Triton-Ascend》中的"[通过pip安装Triton-Ascend](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#piptriton-ascend)"章节，获取配套版本的Triton-Ascend安装指令。

可参考如下安装命令：

```shell
# 注意：triton-ascend 3.2.0 及以下 Triton-Ascend 和 Triton 不能同时存在。需要先卸载社区 Triton，再安装 Triton-Ascend。
pip install triton-ascend==3.2.2 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

> [!NOTE]
> 当前triton-ascend==3.2.2可能会遇到一些兼容问题，若安装triton-ascend==3.2.2问题无法解决，可以尝试退回triton-ascend==3.2.1，经测试3.2.1在python3.12环境下可以跑通完整流程

### 4. 安装fla-npu以适配AscendC

拉取flash-linear-attention-npu代码仓，并进入代码仓根目录

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu -b v26.6.0
cd flash-linear-attention-npu
```

安装步骤：可参考fla-npu仓README：[flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu/blob/v26.6.0/README.md)

> **说明：** 请确保操作系统已安装 `gawk`和`cmake`，否则后续安装会失败。可参考以下命令安装：

```shell
# Ubuntu / Debian
apt-get update
apt-get install gawk cmake
# openEuler / CentOS / RHEL
yum update
yum install gawk cmake
```

推荐使用以下安装命令

```shell
# source 实际的cann路径
source /usr/local/Ascend/cann/set_env.sh

python -m pip install -r requirements.txt
python scripts/check_npu_env.py --build-only

# --soc 需指定为当前机器芯片类型 {ascend910b/ascend910_93/ascend950}
FLA_NPU_SOC=ascend910_93 python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
```

检验fla_npu是否安装成功

```bash
# 显示True
python -c "import fla_npu; import torch; print(hasattr(torch.ops.npu, 'npu_chunk_fwd_o'))"

# 显示Packaged wheel API check passed.
python scripts/check_packaged_wheel_api.py

# 显示flash-linear-attention-npu 26.6.0
pip list | grep fla
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

- 模型地址: [Qwen3.6-*B](https://huggingface.co/collections/Qwen/qwen36)；

 将下载的模型权重保存到本地的`ckpt/hf_path/xxxxxxx`目录下。(*表示对应的尺寸)

<a id="jump2.2"></a>

### 2. 权重加载

当前支持huggingface权重或dcp权重加载，在`xxx_config.yaml`中`training->load_format`字段中配置加载权重的类型，支持`hf`, `dcp`和`auto`，设置为`auto`时会根据权重文件格式自行判断权重类型。

如果需要加载dcp权重，请先根据模型配置完成以下hf权重到dcp权重的转换：

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
注意：如果MoE模型不支持mtp，可在执行`mm-convert`权重转换前将`ckpt/hf_path/xxxxxxx/config.json`中的`mtp_num_hidden_layers`设置为0，以跳过mtp专家权重合并，缩短转换时间。

<a id="jump2.3"></a>

### 3. 权重保存

MindSpeed MM保存权重类型支持huggingface格式和dcp格式，在`xxx_config.yaml`中`training->save_format`字段中配置保存权重的类型，支持`hf`, `dcp`和`auto`:

（1）`save_format`配置为`auto`时,保存权重类型与加载权重类型保持一致;

（2）`save_format`配置为`hf`时,会将保存权重文件转换为safetensors格式;

**注意：hf保存格式仅支持保存权重，不支持保存优化器状态和随机数状态，若需要进行断点续训，请保存为dcp格式**

（3）`save_format`配置为`dcp`时,可使用如下命令将dcp权重转换回hf权重：

```bash
# 待转换的dcp权重目录结构样例为：
# ———— xxxxxxx
#   |—— release
#   |—— latest_checkpointed_iteration.txt

mm-convert Qwen35Converter dcp_to_hf \
--save_hf_dir ckpt/save_hf_path/Qwen3.6-xxB-hf-save \
--dcp_dir ./save_path/iter_000xx \
--origin_hf_dir ckpt/hf_path/Qwen3.6-xxB \
--to_bf16 false \
--num_workers 0

# 其中：
# save_hf_dir: 转换后Huggingface格式的权重保存目录
# dcp_dir: 保存的DCP格式权重目录，`iter_000xx`表示保存的第xx步的权重
# origin_hf_dir：原始Huggingface格式权重目录
# to_bf16：是否将权重数据类型从fp32转换成bf16
# num_workers: 并行工作线程数，0表示串行执行，若存储IO性能允许，可适当调大并发数以提升转换效率，推荐设置为4
```

注意：如果模型没有开启mtp（即，在`xxx_config.yaml`中model下的`mtp_num_layers`字段配置为0或没有配置），默认转换后的权重中不会包含mtp层的权重，可以通过设置`--keep_origin_mtp_weights true`来保留mtp层的权重。

---
<a id="jump3"></a>

## 数据集准备及处理

- 使用**真实数据集**训练：参考[针对VL模型的数据构造 · 使用真实数据集](../../docs/zh/features/building_data_for_VLModel.md#real-data)（下载COCO2017 → 下载LLaVA-Instruct-150K标注 → 运行转换脚本生成`mllm_format_llava_instruct_data.json`）。
- 使用**虚构数据**做功能/性能测试：参考[针对VL模型的数据构造 · 使用虚构数据](../../docs/zh/features/building_data_for_VLModel.md#mock-data)。

### Agentical Trace（OpenAI 格式）数据集

除常规的 `alpaca` / `sharegpt` 格式外，Qwen3.6 支持直接使用 **OpenAI ChatCompletion 风格的智能体轨迹（agentical trace）数据集**进行 SFT。该格式可无损表达多轮工具调用轨迹，包含函数调用 `tool_calls`、思考过程 `reasoning_content`、工具返回 `tool` 以及本轮可用的函数 schema `tools`。

#### 1. 数据格式

每条样本是一个 JSON 对象，核心字段为 `messages`（必选）与 `tools`（可选）。一个完整示例如下：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful coding agent."
    },
    {
      "role": "user",
      "content": "查看当前目录有哪些文件"
    },
    {
      "role": "assistant",
      "reasoning_content": "用户想列出目录内容，调用 run_shell 工具执行 ls。",
      "content": "我来帮你查看。",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "run_shell",
            "arguments": {"command": "ls -la"}
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "total 0\ndrwxr-xr-x  2 user user 4096 ...\n-rw-r--r--  1 user user   12 README.md"
    },
    {
      "role": "assistant",
      "content": "当前目录下有 1 个文件：README.md。"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "run_shell",
        "description": "Run a shell command and return its stdout/stderr.",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {"type": "string", "description": "The shell command to run"}
          },
          "required": ["command"]
        }
      }
    }
  ]
}
```

字段说明：

| 字段 | 位置 | 必选 | 说明 |
|---|---|---|---|
| `messages` | 顶层 | 是 | 对话消息列表，按时间顺序排列 |
| `tools` | 顶层 | 否 | 本轮可用的函数 schema 列表（OpenAI function 格式）。无工具调用的数据可省略 |
| `role` | 每条消息 | 是 | 取值 `system` / `user` / `assistant` / `tool` |
| `content` | 每条消息 | 是 | 文本内容。`tool` 角色表示工具/环境的返回结果 |
| `tool_calls` | `assistant` 消息 | 否 | 函数调用列表，每项含 `function.name` 与 `function.arguments`；`arguments` 可为 dict 或 JSON 字符串 |
| `reasoning_content` | `assistant` 消息 | 否 | 模型思考过程，转换时会合并为 `<think>...</think>` 块拼接在回答前 |

数据组织约束（不满足的样本会被自动跳过并打印精简告警）：

- 轨迹须以 `assistant` 轮**结束**；
- 连续的多个 `tool` 返回会被**自动合并**为一个工具响应轮；
- 轨迹中途穿插的 `user` 消息（如系统提醒、追加指令）会被**自动折叠**进相邻的输入流，不会因打破角色交替而丢弃整条样本；
- 工具调用按 **Qwen3.6 官方 XML 格式**序列化，因此须搭配 `qwen3_6` 或 `qwen3_6_nothink` 模板，token 序列才能与推理端一致。

数据可保存为 `json` / `jsonl` 文件（与现有数据集加载方式一致）。

#### 2. 配置使用

在 `xxx_config.yaml` 的 `dataset_param` 下，将数据格式声明为 `openai`、指定 `tools` 列，并选用 Qwen3.6 模板：

```yaml
dataset:
  dataset_param:
    dataset_type: huggingface
    attr:
      images: null                 # 纯文本 agent trace 置 null
      messages: messages           # 对话字段名
      tools: tools                 # 工具 schema 字段名（无工具时可省略）
      role_tag: role               # 角色字段名
      content_tag: content         # 内容字段名
      user_tag: user               # 用户角色标识
      assistant_tag: assistant     # 助手角色标识
      system_tag: system           # 系统角色标识
      observation_tag: tool        # 工具返回对应的 role（OpenAI 格式为 "tool"）
      formatting: openai           # 关键：选用 OpenAI 转换器
    basic_parameters:
      template: qwen3_6            # 或 qwen3_6_nothink（不带 thinking 的变体）
      enable_thinking: true        # 训练含 reasoning_content 的思考数据时置 true
      cutoff_len: 32768            # agent 轨迹通常较长，按需调大
      dataset_dir: ./data
      dataset: ./data/your_agent_trace.json
      cache_dir: ./cache_dir/
```

模板选择：

- `qwen3_6`：带 thinking，渲染 `reasoning_content` 为 `<think>` 块，适合含思考过程的轨迹；
- `qwen3_6_nothink`：不带 thinking，助手回复为直接输出，适合纯文本终端类轨迹（如命令行 trace）。

> 说明：完整可运行的示例配置可参考 `examples/qwen3_6/` 目录。

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
注意：qwen3.6的mtp layer目前只支持配置1层。

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
- `gdn_implementation`和`causal_conv1d_implementation`
  - gdn_implementation和causal_conv1d_implementation分别支持`eager`，`triton`和`ascendc`配置，使用`ascendc`性能最佳，需要安装fla_npu库
  - 当gdn_implementation配置为`ascendc`时，causal_conv1d_implementation只支持和`triton`和`ascendc`，防止算子之间的布局不匹配

【单机运行配置】
以qwen3_6模型为例：
配置`examples/qwen3_6/finetune_qwen3_6_35B.sh`参数如下

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

loss计算方式差异会对训练效果造成不同的影响，在启动训练任务之前，请查看关于loss计算的文档，选择合适的loss计算方式[vlm_model_loss_calculate_type.md](../../docs/zh/features/vlm_model_loss_calculate_type.md)
可在`xxx_config.yaml`的`model`参数中配置上述文档中的`loss_type`。

```shell
bash examples/qwen3_6/finetune_qwen3_6_35B.sh
```

<a id="jump4.4"></a>

### 4. LoRA 微调

将 `training.lora.enable` 设为 `true`，并按需配置其余参数，使用与全量微调相同的启动脚本进行LoRA微调。

更详细的 LoRA 配置与参数说明见 [LoRA 微调特性文档](../../docs/zh/features/lora_finetune_fsdp2.md)。

【并行策略调整建议】

开启 LoRA 后，基础模型参数全部冻结，仅 LoRA 适配器参数参与训练，可训练参数量相比全量微调大幅下降，梯度与优化器状态的显存占用也随之显著降低。因此，建议结合可训练参数的实际体量重新评估 `parallel` 模块中的并行切分策略，而不必沿用全量微调的配置。例如，yaml配置文件中的 `fully_shard_parallel_size` 不必保持为 `auto`（即在全部设备上切分参数），可结合显存情况显式指定较小的切分度，以减少参数切分引入的通信开销；对于 MoE 类模型，LoRA 微调时可以根据实际情况将 `EP` 开至很低甚至无需开启。

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
| `NPU_ASD_ENABLE`   | 控制是否开启TorchNPU的特征值检测功能 | 设置`0`或未设置: 关闭特征值检测<br>`1`: 表示开启特征值检测，只打印异常日志，不告警<br>`2`:开启特征值检测，并告警<br>`3`:开启特征值检测，并告警，同时会在device侧info级别日志中记录过程数据 |
| `ASCEND_LAUNCH_BLOCKING`   | 控制算子执行时是否启动同步模式 | `0`: 采用异步方式执行<br>`1`: 强制算子采用同步模式运行                                                               |
| `NPUS_PER_NODE`               | 配置一个计算节点上使用的NPU数量                                                  | 整数值（如 `1`, `8` 等）                                                                            |

---
<a id="jump11"></a>

## 注意事项

1. 在加载 processor 过程中，会因 `mistral_common` 三方库版本的兼容性问题导致无法找到 processor，进而训练报错退出，可通过以下方式解决：
   - 卸载`mistral_common` 三方库：pip uninstall -y mistral_common
   - 升级`mistral_common` 三方库至最新版本：pip install --upgrade mistral_common
