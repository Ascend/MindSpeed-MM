---
name: mindspeed-mm-fsdp2-model-only-vlm-migration
description: Use when 需要将仅开源模型权重和代码、没有训练数据或训练 pipeline 的视觉语言理解模型迁移到 MindSpeed-MM FSDP2 插件式后端；适用于 FSDP2 模型、数据、YAML、template、mm_plugin、collator 和迁移分支决策。
---

# MindSpeed-MM FSDP2 仅模型开源 VLM 迁移

## 概览

使用本技能规划和实施视觉语言理解模型向 MindSpeed-MM FSDP2 的迁移。目标模型的上游发布通常只提供权重、推理代码、processor/tokenizer 资产和 Hugging Face 风格 custom code，例如 `stepfun-ai/Step3-VL-10B` 这类模型卡；迁移时需要从模型 I/O 契约反推训练接入方式，而不是复用现成训练 pipeline。

始终使用插件式 FSDP2 后端。不要把本流程与 Megatron 桥接式 FSDP2 路线混用。核心链路是：

```text
training.plugin
  -> ModelHub.build
  -> ParallelApplier(TP/EP/FSDP2)
  -> build_mm_dataset
  -> build_mm_dataloader
  -> TrainEngine.train_step
  -> model(**batch_data, use_cache=False).loss
```

当后端约定或 YAML 字段不清楚时，先阅读 `docs/zh/features/fsdp2_developer_migration_guide.md`。

## 必备输入

开始改代码前，先收集以下事实：

| 输入 | 需要识别的内容 |
| --- | --- |
| 上游模型资产 | 本地模型目录、config、tokenizer、processor、自定义 modeling 文件、图像/视频处理文件。 |
| 模型构建路径 | `AutoConfig.from_pretrained(..., trust_remote_code=True)` 是否可用，以及是否需要用 `model_id` 强制选择注册类。 |
| Forward 契约 | 精确的 `forward` kwargs、必需模态 tensor、可选 kwargs，以及传入 labels 后是否产生 `output.loss`。 |
| Processor 契约 | 上游推理如何构造 `input_ids`、`pixel_values`、grid、媒体 token 占位符、position ids 和 chat template 文本。 |
| 权重策略 | 直接 HF 加载、rank0 广播，或配合 `training.init_model_with_meta_device: true` 使用 DCP 加载。 |
| 数据假设 | 本地 JSON 样本是否能映射为 ShareGPT/Alpaca 类记录，或必须新增自定义 dataset。 |
| 并行目标 | 从 `named_modules()` 得到的 FSDP、可选 EP、重计算、prefetch 和 freeze 模块名。 |

对于仅模型开源的发布，模型卡和推理 demo 是 processor 与 forward 输入的事实来源，但不要假设它们天然适合训练。

## 阶段 0：界定迁移边界

如果同时满足以下条件，将任务归类为“仅模型开源迁移”：

- 上游发布提供模型代码/权重和推理示例。
- 没有可直接迁移的上游训练 dataloader、训练脚本或 loss pipeline。
- 开发者需要创建或适配 MindSpeed-MM 的模型注册、数据接线和 YAML 配置。
- 目标是 FSDP2 训练兼容，而不是只完成推理加载。

记录源模型的最小训练契约：

- 模型类名和 config 类名。
- 必需本地文件：`config.json`、tokenizer 文件、processor 文件、自定义 `modeling_*.py`、processing 工具，以及图像/音频/视频辅助代码。
- 图像、视频或音频占位 token，以及这些 token 的展开方式。
- `forward` key，尤其是 `input_ids`、`attention_mask`、`labels`、`pixel_values`、`image_grid_thw`、`pixel_values_videos`、`video_grid_thw`、`position_ids`、`rope_deltas`、`token_type_ids` 或自定义 mask。
- 模型在传入 `labels` 时是否内部计算 loss。

如果模型卡依赖 remote code，优先把必需源码复制或适配到 `mindspeed_mm/fsdp/models/<model_name>/`，不要依赖训练运行时联网拉取。

## 阶段 1：确认 FSDP2 插件式路线

使用插件式后端：

- 训练入口：`mindspeed_mm/fsdp/train/trainer.py`。
- 模型注册：`mindspeed_mm/fsdp/utils/register.py::model_register`。
- 数据注册：`mindspeed_mm/fsdp/utils/register.py::data_register`。
- 模型构建：`mindspeed_mm/fsdp/models/modelhub.py::ModelHub.build`。
- 数据构建：`mindspeed_mm/fsdp/data/__init__.py::build_mm_dataset` 和 `build_mm_dataloader`。
- FSDP plan 应用：`mindspeed_mm/fsdp/distributed/torch_parallelize.py::ParallelApplier`。
- 训练调用：`mindspeed_mm/fsdp/train/train_engine.py::TrainEngine.train_step`。

YAML 必须通过 `training.plugin` 导入所有插件包。路径可使用 slash 形式，例如：

```yaml
training:
  plugin:
    - mindspeed_mm/fsdp/models/<model_name>
    - mindspeed_mm/fsdp/data/datasets/huggingface
```

插件导入器会递归遍历包内模块，因此注册代码可以放在嵌套 Python 文件中，只要对应包可导入即可。

## 阶段 2：开发模型插件

将模型适配代码放到：

```text
mindspeed_mm/fsdp/models/<model_name>/
```

注册可训练模型类：

```python
from mindspeed_mm.fsdp.utils.register import model_register


@model_register.register("<model_id>")
class XxxForConditionalGeneration(...):
    ...
```

使用以下决策规则：

| 场景 | 推荐实现 |
| --- | --- |
| `AutoConfig.from_pretrained(model_name_or_path)` 可成功执行，且上游类接近 Transformers 风格 | 用 `model_id` 注册模型类，保持 `from_pretrained` 兼容，让 `ModelHub._build_transformers_model` 构建模型。 |
| 上游构建需要非标准参数或额外源码 | 用注册类包裹或适配上游模型，并实现 `_from_config` / `from_pretrained`。 |
| 大模型需要 meta init | 实现 `_from_config`，确保完整模块树可在 `init_empty_weights()` 下创建，再通过 `training.load` 加载 DCP。 |

模型要求：

- YAML 中的 `model.model_id` 必须与 `@model_register.register("<model_id>")` 一致。
- `model.model_name_or_path` 必须指向本地模型/config 目录，保证训练可复现。
- `forward(**batch, use_cache=False)` 必须通过 `**kwargs` 容忍 dataloader 产生的额外 key。
- 当 `features.loss_cfg.loss_type: raw` 时，forward 输出必须暴露 `.loss`；使用非 raw loss 时，模型必须能接收框架注入的 `loss_function`。
- 如果模型使用 mRoPE 或模态相关位置逻辑，应暴露与 collator 兼容的 `get_rope_index`，或在 `forward` 内部计算等价字段。
- 如果模型代码依赖 NPU 特定优化算子，将 monkey patch 隔离到类似 `npu_patch.py` 的小模块中，并由模型插件导入。
- 对 MoE 模型，若训练需要辅助损失，应显式处理 router aux loss 配置和 `output_router_logits`。

权重加载选择：

| 情况 | YAML 与代码选择 |
| --- | --- |
| 构建期间模型可放入 CPU 内存 | `training.init_model_with_meta_device: false`；使用 `from_pretrained(..., device_map="cpu", dtype=torch.float32)` 或等价方式。 |
| 模型需要降低初始化峰值内存 | 先将 HF 权重转换为 DCP；设置 `training.init_model_with_meta_device: true` 和 `training.load: <dcp_root>`。 |
| 需要先从 HF 初始化、后续保存 FSDP2 检查点 | 先直接 HF 加载，完成首次可训练运行后再使用 FSDP2 保存路径。 |

## 阶段 3：选择数据策略

先判断能否复用现有 `huggingface` 数据插件：

```text
mindspeed_mm/fsdp/data/datasets/huggingface/qwen2vl_dataset.py
```

重要易混点：本仓的 `dataset_type: huggingface` 并不是训练时直接读取远端 Hugging Face Hub 数据集。它注册的是 `@data_register.register("huggingface")`，通过 `datasets.load_dataset(path="json", data_files=...)` 加载本地 JSON，然后走 LLaMA-Factory 风格的格式转换和预处理链路：

```text
DatasetAttr
  -> align_dataset(...)
  -> _prompt/_response/_system/_images/_videos/_audios/_tools
  -> SupervisedDatasetProcessor 或 PackedSupervisedDatasetProcessor
  -> input_ids/attention_mask/labels/images/videos/audios
```

### 分支 A：完全复用 `huggingface`

同时满足以下条件时选择该分支：

- 原始样本可通过 `data.dataset_param.attr` 映射到 ShareGPT 或 Alpaca 风格字段。
- 现有 `DataArguments` 已覆盖所需预处理参数。
- 现有 template 兼容，或 tokenizer 自带 `chat_template` 可被解析，且不需要新的多模态插件。
- 现有 `MultiModalDataCollatorForSeq2Seq` 输出 key 与模型 `forward` 匹配。
- `data.dataloader_param.collate_param.model_name` 可使用现有 collator，VLM SFT 通常为 `qwen3vl`。

预期只改 YAML：

```yaml
data:
  dataset_param:
    dataset_type: huggingface
    attr:
      images: images
      messages: messages
      role_tag: role
      content_tag: content
      user_tag: user
      assistant_tag: assistant
    preprocess_parameters:
      model_name_or_path: <local_processor_path>
      trust_remote_code: true
    basic_parameters:
      template: <existing_template_or_null>
      dataset: <local_json_path>
  dataloader_param:
    collate_param:
      model_name: qwen3vl
      ignore_pad_token_for_loss: true
```

### 分支 B：部分复用 `huggingface`

当本地 JSON 映射和 dataset 预处理链路可复用，但模型的 chat 语法、媒体占位符、模态 token 展开或 batch key 不同时，选择该分支。

保留 `dataset_type: huggingface`，只补缺失的兼容层：

- 当文本对话格式不同，在 `mindspeed_mm/fsdp/data/data_utils/func_utils/template.py` 新增 template。
- 当图像/视频/音频占位符或模态 token 展开方式不同，在 `mindspeed_mm/fsdp/data/data_utils/func_utils/mm_plugin.py` 新增 `mm_plugin`。
- 当 processor 输出 key 与现有 `qwen3vl` 或 `qwen3omni` 行为不一致，在 `mindspeed_mm/fsdp/data/dataloader/data_collator.py` 新增或扩展 collator。
- 保持 `attr` 映射显式，让原始样本稳定转换为 `_prompt`、`_response`、`_system`、`_images`、`_videos`、`_audios` 和可选 `_tools`。

分支 B 是大多数仅模型开源 VLM 迁移的默认路线。

### 分支 C：不复用 `huggingface`

出现以下任一情况时选择该分支：

- 原始样本不是对话 JSON，且不能被 `DatasetAttr` 清晰映射。
- 模型需要嵌套或非标准 batch 结构，而 `move_to_device` 无法安全处理为顶层 tensor、tensor list、基础值或 `None`。
- Processor 调用必须发生在 dataset 内部，且与 `SupervisedDatasetProcessor` 冲突。
- 模型需要自定义采样、packing、模态 batching 或预计算特征加载。

在以下路径创建新的数据插件：

```text
mindspeed_mm/fsdp/data/datasets/<dataset_or_model_name>/
```

注册 factory 或 dataset class：

```python
from mindspeed_mm.fsdp.utils.register import data_register


@data_register.register("<dataset_type>")
def build_xxx_dataset(basic_param, preprocess_param, dataset_param=None, **kwargs):
    return XxxDataset(...)
```

如果 batch 格式特殊，优先在 dataset 中实现本地 `collate_fn`，这样 `build_mm_dataloader` 会先使用它，而不是查找 `DATA_COLLATOR`。

## 阶段 4：对齐 template、`mm_plugin`、collator 与 batch key

默认 VLM SFT 路径为：

```text
load_tokenizer(ProcessorArguments)
  -> get_template_and_fix_tokenizer(...)
  -> template.mm_plugin.process_messages(...)
  -> template.mm_plugin.process_token_ids(...)
  -> SupervisedDatasetProcessor.preprocess_dataset(...)
  -> MultiModalDataCollatorForSeq2Seq(...)
```

使用以下兼容性清单：

| 层级 | 必须匹配的内容 |
| --- | --- |
| `attr` | 原始样本中的 messages、role、text、images、videos、audios、system 和 tools 字段名。 |
| `template` | Chat 角色包装、assistant 后缀、EOS 替换、thinking 标签、tool prompt 和默认 system prompt。 |
| `mm_plugin` | 占位 token 名称、图像/视频/音频展开、processor 媒体加载方式和返回的多模态 key。 |
| `collator` | Padding、FSDP 空媒体 fake 输入兜底、`get_rope_index`、mRoPE `position_ids`、`rope_deltas` 和最终 tensor 名称。 |
| 模型 `forward` | 接收所有产出的 batch key，并忽略无害额外 key。 |

典型 VLM batch key 包括：

```text
input_ids
attention_mask
labels
pixel_values
image_grid_thw
pixel_values_videos
video_grid_thw
second_per_grid_ts
position_ids
rope_deltas
token_type_ids
cross_attention_mask
```

出现以下任一不兼容时，不要强行复用现有 template 或 plugin：

- 文本 chat 分隔符或 assistant mask 行为不同。
- 图像/视频占位符不同。
- 视觉 token 展开依赖不同 processor API。
- mRoPE 或 position id 行为不同。
- 模型特定 tensor 名称不同。

## 阶段 5：编写 YAML、启动脚本与 FSDP2 plan

在以下路径创建模型示例文件：

```text
examples/<model_name>/
```

仅模型开源 VLM 迁移可优先参考 Qwen3VL 和 KimiK2.5 配置：

- `examples/qwen3vl/qwen3vl_30B_config_v1.yaml`
- `examples/kimik2_5/kimik2_5_config.yaml`

YAML 最小一致性规则：

| YAML 字段 | 必须匹配的对象 |
| --- | --- |
| `model.model_id` | `@model_register.register("<model_id>")`。 |
| `model.model_name_or_path` | `ModelHub.build` 使用的本地模型/config 路径。 |
| `data.dataset_param.dataset_type` | `@data_register.register("<dataset_type>")`，分支 A/B 通常为 `huggingface`。 |
| `data.dataset_param.preprocess_parameters.model_name_or_path` | 本地 tokenizer/processor 路径。 |
| `data.dataset_param.basic_parameters.template` | 已有或新增注册的 template 名；若解析 tokenizer `chat_template` 则可为空。 |
| `data.dataloader_param.collate_param.model_name` | 已有或新增注册的 `DATA_COLLATOR` key。 |
| `parallel.fsdp_plan.apply_modules` | 来自 `model.named_modules()` 的真实模块名。 |
| `training.plugin` | 导入所有模型和数据插件包。 |

在模型可构建后，从模块名推导 `fsdp_plan`：

1. 打印或检查 `model.named_modules()`。
2. 优先识别大而重复的模块：语言层、视觉 blocks、projector、merger、embeddings 和 `lm_head`。
3. 若参考模型采用这种模式，按从较叶子的 block 到较宽包装模块的顺序配置 `apply_modules`。
4. 使用 EP 或 prefetch 时，如果 hook 应挂到稳定的层级模块，添加 `hook_modules`。
5. 仅对必须避免分片的模块添加 `ignored_modules`。
6. 只有模块顺序明确后才添加 `num_to_forward_prefetch` / `num_to_backward_prefetch`；对于复杂跨塔顺序，实现模型级 `set_modules_to_prefetch`。

当前插件式 FSDP2 路线保持 `parallel.tensor_parallel_size: 1`。

## 阶段 6：开发顺序

新增仅模型开源 VLM 时，按以下顺序推进：

1. 阅读模型卡、config、processor、tokenizer、custom code 和推理 demo。
2. 记录 forward key，并确认 `labels` 是否产生 `.loss`。
3. 选择直接 HF 加载、wrapper 加载或 DCP/meta-init 加载。
4. 新增模型插件并注册 `model_id`。
5. 在写 dataset 代码前，先决定数据分支 A、B 或 C。
6. 使用分支 A/B 时，配置 `attr`、template、processor 参数和 collator 名称。
7. 使用分支 B 时，以最小兼容改动补充缺失的 template、`mm_plugin` 或 collator。
8. 使用分支 C 时，新增 `dataset_type`，必要时提供 dataset-local `collate_fn`。
9. 在 `examples/<model_name>/` 下创建顶层 FSDP2 YAML 和启动脚本。
10. 根据真实 `named_modules()` 输出填写 `fsdp_plan.apply_modules`。
11. 在基础模型/数据契约清晰前，先关闭 EP、activation offload、chunk loss、prefetch 和 LoRA 等性能特性。

## 常见错误

| 错误 | 修正方式 |
| --- | --- |
| 把 `dataset_type: huggingface` 理解为远端 Hub 加载 | 将其理解为本地 JSON 加 LLaMA-Factory 风格预处理插件。 |
| 不检查占位符就复制 Qwen template 名称 | 先比较 chat 分隔符、媒体 token 和 processor 展开方式。 |
| 假设推理输入等于训练输入 | 单独确认 labels、loss、padding、mRoPE 和 batch key 名称。 |
| 凭猜测填写 `fsdp_plan.apply_modules` | 从适配后模型的 `named_modules()` 推导路径。 |
| 过早新增 dataset | 先证明分支 A/B 无法满足模型契约。 |
| 首次 bring-up 就开启 EP、prefetch、offload 或 chunk loss | 先完成基础模型/数据兼容，再逐步加入性能特性。 |
| 返回嵌套 batch 结构 | 除非模型入口会在 device movement 前处理嵌套，否则将 batch 输出扁平化。 |

## 输出要求

使用本技能时，产出迁移指导应包含：

- 选择的数据分支及原因。
- 模型插件路径和 `model_id`。
- 数据插件路径和 `dataset_type`。
- 必需的 template、`mm_plugin` 与 collator 决策。
- 预期 batch key 和模型 `forward` 兼容性说明。
- 必须修改的 YAML 字段。
- `apply_modules`、`hook_modules`、`ignored_modules` 和 prefetch 的 FSDP2 模块路径策略。

除非用户明确要求，不要新增独立的精度评估、验收测试或训练验证章节。
