# 预训练

## 纯文本预训练

### 适用后端

纯文本预训练适用于纯 FSDP2 后端和含 Megatron 后端。两种后端选择预训练数据处理流程的配置项不同，请按照下文对应后端的示例进行配置。

### 使用场景

预训练（Pretraining）是语言模型发展的核心步骤，目标是让模型通过大规模无标签语料学习语言规律与世界知识。预训练过程更关注语言建模本身，而非具体任务执行。以GPT类模型为例，它是一种典型的自回归语言模型，其核心思想是基于历史上下文预测下一个标记。预训练的过程就是通过反复优化这种预测能力，模型逐渐学会如何理解语境、保持句子连贯性，并掌握更高层次的语言结构，为多种下游任务提供通用的语言表示能力。  
预训练数据通常为纯文本格式，无任务导向，例如：

```json
{"text": "今天是个好天气，我们一起去爬山。"}
{"text": "深度学习正在改变世界。"}
{"text": "AI的出现推动了人类社会的发展。"}
```

### 使用方法

#### 纯 FSDP2 后端

在 `xx_config.yaml` 文件中配置预训练相关参数：

```yaml
### 数据相关配置
data:
  dataset_param:
    ...
    attr:
      formatting: alpaca
      prompt: text
    basic_parameters:
      stage: pretrain
      template: default
  dataloader_param:
    collate_param:
      collator_id: llm_pretrain
  ...
```

> [!NOTE]
>
> FSDP2 后端通过 `basic_parameters.stage: pretrain` 选择预训练数据处理流程。切换到预训练时，请将 `attr` 下原有的配置（如 SFT 场景的列映射配置）注释或移除，按上述示例仅保留 `formatting` 和 `prompt` 配置，避免残留配置导致数据对齐失败。

#### 含 Megatron 后端

在 `data.json` 文件中配置预训练相关参数：

```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            "template": "default",
        },
        "attr": {
            "formatting": "alpaca",
            "pretrain": true,
            "system": null,
            "images": null,
            "videos": null,
            "audios": null,
            "prompt": "text",
            "query": null,
            "response": null,
            "history": null
        }
    },
    "dataloader_param": {
        ...
        "collate_param": {
            "model_name": "llm_pretrain"
        },
        ...
    }
}
```

#### 参数说明

FSDP2 后端需要按照上述示例配置 `basic_parameters.stage`、`attr` 和 `collate_param`；含 Megatron 后端需要按照上述示例配置 `attr` 和 `collate_param`。其他参数请根据实际训练任务修改。

- **`basic_parameters.packing`**
  - 描述：将多个短文本样本拼接成符合模型最大长度 `cutoff_len` 的长序列。对于纯文本大规模预训练，该参数默认为 `true`，以充分利用显存并提升训练效率。
  - 取值：
    - `true`：开启样本拼接，默认值，可以不配置该参数。
    - `false`：关闭样本拼接。

- **`cutoff_len`**
  - 描述：训练序列的最大长度。
  - 配置位置：
    - 纯 FSDP2 后端：对应 `xx_config.yaml` 中的 `cutoff_len`。
    - 含 Megatron 后端：对应 `finetune_xx.sh` 中的 `SEQ_LEN`。

### 注意事项

以下说明适用于纯文本预训练（`attr` 中未配置 `images`、`videos`、`audios` 列）。packing 开启（默认）的场景下，数据预处理会在每个批次内拼接文本，并按 `cutoff_len` 切分为定长序列。需要保证每个预处理批次中有效文本的 token 总数不小于 `cutoff_len`，否则该批次无法生成训练样本。可通过增大 `preprocessing_batch_size`、增加样本长度或减小 `cutoff_len` 解决。

## 多模态预训练 Packing（纯 FSDP2）

纯 FSDP2 数据层支持在 `stage: pretrain` 下先独立展开每条多模态文档的图像、视频或音频 token，再按完整文档进行 packing。不同文档不会被切开；`cu_seqlens`、重置后的 position ids 和文档首 token 的 ignore label 分别用于隔离 attention/线性注意力状态、MRoPE 位置以及跨文档 next-token loss。

当前该能力仅支持配置为 `dataset_type: huggingface` 的模型。数据层按 `attr` 中是否配置了 `images`、`videos` 或 `audios` 列选择处理流程：配置了任一媒体列时使用本节的多模态流程；都未配置时使用上文的纯文本流程，行为与之前保持一致。预训练数据可以使用 assistant-only 文档格式，媒体占位符的出现顺序必须与对应媒体数组一致。例如：

```json
{
  "messages": [{"role": "assistant", "content": "<image> is a red square."}],
  "images": ["/absolute/path/red_square.png"]
}
```

以 Qwen3.5 FSDP2 为例，需要配置：

```yaml
data:
  dataset_param:
    dataset_type: huggingface
    attr:
      images: images  # 必须配置媒体列，才会使用多模态预训练流程
      messages: messages
      role_tag: role
      content_tag: content
      user_tag: user
      assistant_tag: assistant
    basic_parameters:
      stage: pretrain
      packing: true
      preprocess_on_fly: false
      cutoff_len: 4096
  dataloader_param:
    collate_param:
      collator_id: qwen3vl

training:
  micro_batch_size: 1
```

使用时请注意：

1. 多模态 packing 依赖离线 batched preprocess，不支持同时开启 `preprocess_on_fly`。
2. 超过 `cutoff_len` 的完整多模态文档会被丢弃，不会在媒体 token 或媒体特征中间切分。
3. 当前 packed `cu_seqlens` 链路要求使用右侧 padding，并要求 `micro_batch_size` 为 1；一条 packed row 内仍可包含多条原始文档。
4. 模型侧必须支持消费 `cu_seqlens`，以隔离不同文档的 attention 及其他序列状态。Qwen3.5 推荐沿用示例配置中的 `gdn_implementation: ascendc` 和 `causal_conv1d_implementation: triton`。
5. 展开后的图像、视频、音频占位 token（如 `<|image_pad|>`）不参与 loss，仅作为上下文；`<|vision_start|>`、`<|vision_end|>` 边界 token 以及后续文本仍参与 loss。该规则对基于 `Qwen2VLPlugin` 的模板生效（Qwen2-VL、Qwen3-VL、GLM-4.1V、Qwen Omni 系列）。
6. 多模态预训练数据的原始文本中不能直接写入媒体 token（如 `<|vision_start|>`、`<|image_pad|>`），请使用 `<image>`、`<video>` 等占位符。含有媒体 token 的文档会被丢弃并打印 warning。该检查仅作用于多模态预训练，不影响 SFT。
