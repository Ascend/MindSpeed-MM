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
- [数据集准备及处理](#数据集准备及处理)
  - [Agentical Trace（OpenAI 格式）数据集](#agentical-traceopenai-格式数据集)
- [微调](#微调)
  - [准备工作](#1-准备工作)
  - [配置参数](#2-配置参数)
  - [启动微调](#3-启动微调)
- [环境变量声明](#环境变量声明)

## 版本说明

### 参考实现

```shell
url=https://github.com/huggingface/transformers.git
commit_id=7d9754a
```

### 变更记录

2026.04.17: 首次支持Qwen3.6-35B-A3B模型

---

## 环境安装

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](../../docs/zh/pytorch/install_guide.md)，完成昇腾软件安装。
> Python版本推荐3.10，torch和TorchNPU版本推荐2.7.1版本，CANN推荐使用8.5.2版本；

### 2. 环境搭建

拉取MindSpeed MM代码仓，并进入代码仓根目录：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

执行如下指令安装：

```bash
bash scripts/install.sh --msbranch 26.1.0_core_r0.12.1
pip install transformers==5.2.0 triton-ascend==3.2.0 accelerate==1.2.0
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

## 权重下载及转换

### 1. 权重下载

从Huggingface库下载对应的模型权重:

> [!NOTE]
>
> 如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

- 模型地址: [Qwen3.6-*B](https://huggingface.co/collections/Qwen/qwen36)；

 将下载的模型权重保存到本地的`ckpt/hf_path/xxxxxxx`目录下。(*表示对应的尺寸)

如果使用fsdp2的meta init初始化模型，需要先完成以下权重转换：

```bash
mm-convert Qwen35Converter hf_to_dcp \
--hf_dir ckpt/hf_path/xxxxxxx \
--dcp_dir ckpt/dcp_path/xxxxxxx

# 转换后的目录结构为：
# ———— xxxxxxx
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

并在`xxx_config.yaml`中将`init_model_with_meta_device`参数配置为`True`，同时将`load`参数修改为转换后的dcp权重路径（写到`release`文件夹的上一级目录）。

MindSpeed MM保存权重的格式也为dcp格式。可使用如下命令将dcp权重转换回HF权重

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

其中，`--save_hf_dir`表示转换后的权重保存路径，`--dcp_dir`表示保存的权重路径，`--origin_hf_dir`表示原始huggingface权重的路径，`--to_bf16`表示权重数据类型是否从fp32转换成bf16。

---

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

> 说明：完整可运行的示例配置可参考 `examples/qwen3_6/agentical_ascendc_sft/` 目录。

## 微调

### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

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

【单机运行配置】
配置`examples\qwen3_6\finetune_qwen3_6_35B_A3B.sh`参数如下

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

### 3. 启动微调

loss计算方式差异会对训练效果造成不同的影响，在启动训练任务之前，请查看关于loss计算的文档，选择合适的loss计算方式[vlm_model_loss_calculate_type.md](../../docs/zh/features/vlm_model_loss_calculate_type.md)
可在`xxx_config.yaml`的`model`参数中配置上述文档中的`loss_type`。

```shell
bash examples/qwen3_6/finetune_qwen3_6_35B_A3B.sh
```

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
