---
name: mindspeed-mm-fsdp2-model-only-migration
description: Use when migrating a model-only Hugging Face/custom-code VLM or multimodal understanding model (open weights/inference code but no training pipeline) into MindSpeed-MM plugin-style FSDP2. Triggers： "model-only migration", "migrate model to MindSpeed-MM FSDP2", "integrate <model_name> without training pipeline", "auto migration".
---

# MindSpeed-MM FSDP2 仅模型开源模型迁移

## 概览

使用本技能规划和实施“仅模型开源”的模型迁移。典型上游只提供权重、推理 demo、processor/tokenizer 资产和 Hugging Face custom code，缺少训练数据链路；迁移时需要从模型 I/O、processor 行为和仓内 FSDP2 训练链路反推出可训练接入方式。

始终使用 MindSpeed-MM 插件式 FSDP2 后端。不要混用 Megatron 桥接式 FSDP2 路线。核心链路是：

```text
training.plugin
  -> ModelHub.build
  -> ParallelApplier(TP/EP/FSDP2)
  -> build_mm_dataset
  -> build_mm_dataloader
  -> TrainEngine.train_step
  -> model(**batch_data, use_cache=False).loss
```

迁移前先阅读 `docs/zh/features/fsdp2_developer_migration_guide.md`，理解插件式 FSDP2 的路线、入口、注册、YAML、权重加载和启动方式。

## 开始前：创建 Todo

开始执行前，先确认可用运行环境、模型路径、数据样本路径和启动方式；无法确认时先问用户。

然后用可用的任务跟踪工具记录阶段状态，例如 TodoWrite、`update_plan` 或等价清单：

```text
阶段 1：分析 HF 模型资产与仓内相似模型          -> in_progress
阶段 2：模型接入                              -> pending
阶段 3：选择数据分支 A/B/C                    -> pending
阶段 4：对齐 template / mm_plugin / collator -> pending
阶段 5：编写 YAML、启动脚本与 FSDP2 plan      -> pending
阶段 6：完成最小端到端 E2E 测试               -> pending
```

## 阶段 1：分析模型资产与相似案例

先确认这真的是“仅模型开源迁移”：

- 上游发布提供模型权重、模型代码和推理示例。
- 没有可直接迁移的上游训练 dataloader、训练脚本或 loss pipeline。
- 目标是接入 FSDP2 训练链路，而不是只完成推理加载。
- 开发者需要自行建立模型注册、数据接线、YAML 配置和端到端 E2E 测试路径。

检查上游或本地模型目录中的关键资产：

| 类别 | 要确认的事实 |
| --- | --- |
| 模型文件 | 识别 HF 模型资产，阅读 `config.json`、`modeling_*.py`、processor/tokenizer 文件、`chat_template.jinja` 等关键文件。 |
| Processor/Tokenizer | tokenizer、processor、image/video/audio processor、`chat_template.jinja` 或 tokenizer 内置 chat template。 |
| Forward 契约 | 训练时 `forward` 接收哪些 key，传入 `labels` 后是否返回 `.loss`，是否需要 `position_ids`、`rope_deltas`、自定义 mask。 |
| Processor 契约 | 上游如何构造文本、媒体占位 token、`pixel_values`、grid、position metadata、视频/音频输入。 |
| 数据集 | 数据集 JSON 为 ShareGPT/Alpaca 类格式，还是其他类型。 |

然后找仓内最近似模型，优先比较：

- 同模态：图文 VLM、视频理解、音频理解、全模态。
- 同 processor 风格：Qwen 系、自定义 processor、Step 系。
- 同构建方式：HF `PreTrainedModel`、自定义 `PreTrainedModel`、最小适配 patch、自定义 `from_pretrained`。
- 同 batch key：图像/视频/音频输入、媒体占位符、processor 输出结构，例如`image_grid_thw`、`pixel_values_videos`等。

如果以下信息无法从仓库、模型目录或示例中确认，再与用户交互：

- 本地真实模型路径
- 本地数据格式和样本结构
- 是否必须复用现有某条训练数据链路
- 是否允许引入新 dataset plugin 或只接受 A/B 分支

## 阶段 2：模型接入

模型适配通常放在：

```text
mindspeed_mm/fsdp/models/{model_name}/
```

通过 `training.plugin` 导入插件包，并用 `model_register` 注册训练模型类；注册名必须与 YAML 中的 `model.model_id` 一致。

`ModelHub.build` 会先尝试 `AutoConfig.from_pretrained(model.model_name_or_path)`；成功时走 Transformers 构建路径，失败时走 custom `BaseModel` 路径。Transformers 路径下 `_from_config` 接收 HF config，custom 路径下 `_from_config` / `from_pretrained` 接收完整 `ModelArguments`。

模型接入决策：

| 场景 | 推荐路径 |
| --- | --- |
| HF config 可加载，模型类接近 Transformers 风格 | 注册模型类并保持 `from_pretrained` 兼容，让 `ModelHub._build_transformers_model` 构建。 |
| 上游构建依赖非标准参数、额外初始化或本地 custom code | 优先在原始模型类上最小 patch；必要时再写适配层。 |
| 大模型需要降低初始化峰值内存 | 实现完整 `_from_config`，配合 DCP 和 `training.init_model_with_meta_device: true`。 |

模型插件必须满足：

- `model.model_name_or_path` 指向本地模型/config 目录，不依赖训练时联网下载。
- `forward(**batch, use_cache=False)` 能接收 dataloader 产出的 key，并通过 `**kwargs` 容忍无害额外字段。
- 当 `model.loss_cfg.loss_type: raw` 时输出对象暴露 `.loss`；使用框架 loss 时能接收注入的 `loss_function`。
- 若需要 mRoPE 或模态位置编码，提供与 collator 兼容的 `get_rope_index`，或在 `forward` 内部处理等价字段。
- NPU patch、fused op、MoE aux loss、chunk loss 等特性不属于最小 E2E 目标；除非模型跑通必需或用户明确要求，否则后置到独立性能/长训任务。

权重加载建议：

| 情况 | 配置策略 |
| --- | --- |
| CPU 内存足够直接构建 | `training.init_model_with_meta_device: false`，在 `from_pretrained` 中从 HF/本地权重加载。 |
| 初始化峰值内存过高 | 先将权重转为 DCP，设置 `training.init_model_with_meta_device: true` 与 `training.load: <dcp_root>`。 |
| 先 E2E 测试后再训练续跑 | 初期可直接 HF 加载；确认端到端训练链路后再切换 DCP 保存/加载策略。 |

## 阶段 3：选择数据分支

本阶段根据真实样本结构、processor 输出和模型结构，决定走 A/B/C 分支。

先判断能否复用现有 `huggingface` 数据插件：

```text
mindspeed_mm/fsdp/data/datasets/huggingface/qwen2vl_dataset.py
```

重要易混点：本仓 `dataset_type: huggingface` 不是训练时直接读取远端 Hugging Face Hub 数据集。它注册的是 `@data_register.register("huggingface")`，通过 `datasets.load_dataset(path="json", data_files=...)` 加载本地 JSON，再执行 LLaMA-Factory 风格的格式转换和预处理：

```text
DatasetAttr
  -> align_dataset(...)
  -> _prompt/_response/_system/_images/_videos/_audios/_tools
  -> SupervisedDatasetProcessor 或 PackedSupervisedDatasetProcessor
  -> input_ids/attention_mask/labels/images/videos/audios
```

### 分支 A：完全复用 `huggingface`

选择条件：本地 JSON 可映射到 ShareGPT/Alpaca 风格字段，现有 `DataArguments` / `ProcessorArguments`、template、`mm_plugin` 和 collator 均与模型 `forward` 匹配。

动作：只改 YAML，显式配置 `attr`、tokenizer/processor、template 和 `collate_param.model_name`。

### 分支 B：部分复用 `huggingface`

选择条件：本地 JSON 映射和预处理链路可复用，但 chat 语法、媒体占位符、模态 token 展开或 batch key 不一致。

动作：保持 `dataset_type: huggingface`，只补缺失兼容层：

- 文本对话格式不同：在 `mindspeed_mm/fsdp/data/data_utils/func_utils/template.py` 新增 template。
- 图像/视频/音频占位符或模态 token 展开不同：在 `mindspeed_mm/fsdp/data/data_utils/func_utils/mm_plugin.py` 新增 `mm_plugin`。
- Processor 输出 key 或位置编码 metadata 不同：在 `mindspeed_mm/fsdp/data/dataloader/data_collator.py` 新增或扩展 collator。
- 原始字段名不同：优先通过 `attr` 映射到 `_prompt`、`_response`、`_system`、`_images`、`_videos`、`_audios`、`_tools`。

分支 B 是大多数仅模型开源 VLM 迁移的默认路线。

### 分支 C：不复用 `huggingface`

选择条件：现有 `huggingface` 链路无法把样本经 `DatasetAttr`、`SupervisedDatasetProcessor` 和 collator 转成模型需要的扁平 batch，例如原始样本不是可清晰映射的对话 JSON，processor 必须在 dataset 内逐样本调用且与预处理链路冲突，或模型需要自定义采样、packing、模态 batching、预计算特征加载。

动作：

- 新增数据插件并注册新的 `dataset_type`；在以下路径创建新的数据插件：`mindspeed_mm/fsdp/data/datasets/<dataset_or_model_name>`
- 如果 batch 格式特殊，优先在 dataset 中实现 `collate_fn`，再考虑新增通用 `DATA_COLLATOR`。

## 阶段 4：对齐 template、mm_plugin、collator 与 batch key

默认 VLM SFT 数据路径是：

```text
load_tokenizer(ProcessorArguments)
  -> get_template_and_fix_tokenizer(...)
  -> template.mm_plugin.process_messages(...)
  -> template.mm_plugin.process_token_ids(...)
  -> SupervisedDatasetProcessor.preprocess_dataset(...)
  -> MultiModalDataCollatorForSeq2Seq(...)
```

判断能否复用现有 template / `mm_plugin` / collator 时，必须形成四层证据链：

| 证据层 | 检查内容 |
| --- | --- |
| `chat_template` | user/assistant/system 包裹格式、assistant 结束符、mask 行为、模态占位符插入位置。 |
| processor | 媒体占位 token 展开规则，图像/视频转成哪些 tensor 和 metadata，谁负责视觉 token 展开。 |
| config | special token 是否真正被 processor/forward 使用，图像/视频 token id 与 position 相关配置是否匹配。 |
| `forward` | 最终接收哪些 batch key，是否需要 `position_ids`、`rope_deltas`、`token_type_ids`、`cross_attention_mask`，是否返回 `.loss`。 |

职责边界：

| 层级 | 职责 |
| --- | --- |
| dataset | 将原始样本映射到 `_prompt`、`_response`、`_images`、`_videos`、`_audios` 等中间格式。 |
| template | 处理文本 chat 壳、角色包装、assistant 收尾、thinking/tool prompt 等文本格式。 |
| `mm_plugin` | 校验媒体占位符，执行多模态替换/展开策略，对接 processor 的媒体输入。 |
| collator | 处理 padding、视觉 tensor、position metadata、fake 输入兜底和最终 batch key。 |

关键规则：

- 如果媒体占位符替换影响 `input_ids/labels`，通常应在 dataset 预处理前半段通过 template / `mm_plugin` 解决，不要期待 collator 事后修正。
- 如果问题只体现在 tensor 名称、position metadata 或媒体 tensor 结构，优先改 collator。
- 只有当原始样本结构无法映射进现有 `huggingface` 链路时，才新增 dataset。

不要只看 `special_tokens_map.json`、token 名称、template 名称或推理 demo 能跑通，就判断多模态训练协议兼容。

## 阶段 5：编写 YAML 配置文件、启动脚本

这一阶段按“基线 example -> 模型入口 -> 数据入口 -> 训练基础项 -> FSDP2 plan -> 配置复查”的顺序推进。第一版配置服务端到端 E2E 测试，不追求性能项完整。

### 1. 选择基线 example

选择基线 example：优先复用同模态、同结构、同数据链路的 `examples/{model_name}/`，产出 YAML、启动脚本、日志与保存目录。

### 2. 配置文件

- 填写模型/数据入口：明确 model_id、model_name_or_path、training.plugin；按阶段 3 的 A/B/C 分支配置数据路径、attr、tokenizer/processor、template、mm_plugin 和 collator。
- 填写训练基础项：沿用基线的 optimizer、lr、batch、梯度累积、迭代步数、日志、load/save/checkpoint。
- 填写 FSDP2 plan：先构建模型，再用 `named_modules()` 确认模块名；首版只填最小训练链路必需的 `apply_modules`，TP 默认 1，不为 E2E 引入 EP、activation offload、chunk loss、prefetch、LoRA 等特性。

### 3. 启动脚本文件

创建启动脚本，并对照基线 example 启动脚本的结构，与其保持一致。

## 阶段 6：最小端到端 E2E 测试

除非用户明确只要求方案设计，否则迁移实现后必须走真实训练入口做最小端到端 E2E 测试。可以使用小样本、低 batch、少步数，但测试应从启动脚本/YAML 进入完整训练链路。

E2E 测试至少确认：

1. 启动脚本能加载 YAML，并通过 `training.plugin` 导入模型和数据插件。
2. 数据样本能经过 dataset、template / `mm_plugin`、processor 和 collator 进入 dataloader。
3. 模型能由 `ModelHub.build` 构建，batch key 与 `forward` 契约匹配。
4. 至少完成一次 `forward -> loss -> backward -> optimizer step`。

如果 E2E 失败，继续基于日志定位并修复自己负责的插件、数据兼容层、YAML 或启动脚本，然后重跑。不要在 E2E 未跑通时宣称迁移完成，也不要把可在当前代码范围内修复的问题交给用户。只有缺少可用环境、真实模型/数据路径、硬件权限，或问题被证明超出本迁移边界时，才说明阻塞原因并请求用户提供条件。

## 按需参考

在需要判断文件职责、选择接入位置或收尾核对改动范围时，阅读 `references/fsdp2_code_boundaries.md`；先看核心链路，再按问题读取文件树或改动分级。

## 常见反模式与陷阱

| 问题 | 修正方式 |
| --- | --- |
| 未分析 HF 模型和相似案例就写代码 | 先确认模型资产、推理 demo 和仓内相近配置；能从现有材料确认的事实不要先问用户。 |
| 环境不确定就开始承诺 E2E | 先确认可用运行环境、模型路径、数据样本路径和启动方式；无法确认时先问用户。 |
| 为接入新模型修改主训练框架 | 对照 FSDP2 框架地图中的改动分级；不要改 `trainer.py`、`train_engine.py`、`ModelHub`、通用 dataloader 构建器等框架核心代码。 |
| 用外层代理替代上游模型主体 | 优先保留上游模型类并做最小 patch；不要用 wrapper 代理属性和 `forward` 来重建继承关系、模块树、generation、embedding、state dict 或 checkpoint conversion 语义。只有上游模型无法本地导入、注册/加载或必须屏蔽不可控副作用时，才写适配层。 |
| 大幅重写上游模型代码 | 优先复用 HF 模型目录或开源仓代码，只改本地 import、注册、`.loss`、必要 FSDP2 兼容点；只有无法导入、无法注册/加载或必须隔离副作用时才复制并扩大修改。 |
| 把 `dataset_type: huggingface` 理解为远端 Hub 加载 | 将其理解为本地 JSON 加 LLaMA-Factory 风格预处理插件。 |
| 只看 token 名称就复用 Qwen template | 对比 chat 分隔符、媒体占位符、processor 展开和 `forward` key。 |
| 把推理输入协议当成训练输入协议 | 单独确认 labels、loss、padding、mask、position ids 和 metadata。 |
| 过早新增 dataset | 先证明分支 A/B 无法满足模型契约。 |
| YAML 字段和注册对象不一致 | 核对 `model_id`、`dataset_type`、`collate_param.model_name`、`training.plugin` 与对应 register / `DATA_COLLATOR`。 |
| 凭猜测填写 `fsdp_plan.apply_modules` | 从适配后模型的 `named_modules()` 推导路径，并先识别语言层、视觉 blocks、projector、merger、embedding、`lm_head` 等重复或大模块。 |
| E2E 失败后提前收尾 | 继续读日志、修复迁移范围内的问题并重跑；只有缺环境、缺模型/数据或超出迁移边界时才向用户说明阻塞。 |
| loss 为 NaN 或恒为 0 却判定 E2E 通过 | E2E 必须确认 loss 为有限且非异常的训练信号；NaN、Inf、恒 0 或明显不随训练变化，都说明 labels、mask、loss 路径或 batch 对齐仍需排查。 |
| 通过弱化配置绕过问题后宣称 E2E 通过 | 临时把 `train_iters` 设为 1、把 FSDP/EP 设为 1、冻结报错层或跳过问题分支只能用于定位；最终 E2E 必须覆盖目标训练配置中的关键并行策略和待迁移模块，不能用降级配置替代修复。 |
| 把 E2E 测试扩展成完整验收或性能集成 | E2E 只证明配置入口到 `optimizer step` 的最小训练链路跑通；精度评估、长训验收、性能调优以及 EP、activation offload、chunk loss、prefetch、LoRA 等高级特性另开任务。 |
| 返回复杂嵌套 batch | 除非模型入口会在 device movement 前处理，否则将 batch 输出扁平化。 |

## 输出要求

使用本技能时，迁移指导或实现总结应包含：

- 选择的数据分支 A/B/C 及原因。
- 模型插件路径和 `model_id`。
- 数据插件路径和 `dataset_type`。
- 必需的 template、`mm_plugin` 与 collator 决策。
- 兼容性证据链摘要：`chat_template`、processor、config、`forward` 分别如何判断。
- 预期 batch key 和模型 `forward` 兼容性说明。
- 必须修改的 YAML 字段。
- 实际改动范围是否符合 FSDP2 框架地图的改动分级；若触碰“不建议改”范围，说明必要性；不得触碰“不能改”范围。
- 最小端到端 E2E 测试结果或无法验证的原因。
