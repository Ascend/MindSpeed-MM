# MindSpeed MM FSDP2 模型迁移指南（以 Qwen3-VL 为例）

## 概述

本文以 **Qwen3-VL-30B-A3B（MoE 多模态理解模型）** 为例，介绍将一个新模型迁移并接入 MindSpeed MM FSDP2 后端的完整过程，并在每个关键决策节点说明选择依据，供迁移其他模型时参照。

本文面向需要将新模型接入 FSDP2 后端的研究人员、工程师与开发者，要求读者：

- 已按 [安装指导](../pytorch/install_guide.md) 完成昇腾环境与 MindSpeed MM 的安装；
- 具备 PyTorch 训练与模型开发调试的基础知识；
- 了解模型迁移、分布式训练及精度对齐的基本概念；
- 待迁移的模型已能在源平台（如 GPU）正常训练，并保留了 loss 基线，作为迁移的起点与后续精度对齐的参照。

若仅需使用现成样例跑通 Qwen3-VL 微调、而非接入新模型，请参考 [Qwen3VL README](../../../examples/qwen3vl/README_v1.md)。

## 1. 源模型与迁移目标

### 1.1 源模型结构

Qwen3-VL 的参考实现在 Hugging Face Transformers 仓库的 [src/transformers/models/qwen3_vl_moe](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_vl_moe) 目录，顶层类为 `Qwen3VLMoeForConditionalGeneration`，主要由视觉编码器、语言模型、输出头三部分组成，结构如下：

```text
Qwen3VLMoeForConditionalGeneration
├── model.visual                      # 视觉编码器（ViT）
│   ├── blocks.0 ~ blocks.N           # 视觉 Transformer 层
│   ├── merger                        # 视觉特征合并模块
│   └── deepstack_merger_list.{*}     # DeepStack 多层特征融合
├── model.language_model              # 语言模型（MoE 结构）
│   ├── embed_tokens
│   └── layers.0 ~ layers.M           # 每层的 mlp.experts 是稀疏专家
└── lm_head                           # 输出头
```

建议先梳理该模块树：后续的 FSDP 分片计划、冻结配置、重计算配置，填写的都是这些模块路径。样例 [`examples/qwen3vl/qwen3vl_30B_config_v1.yaml`](../../../examples/qwen3vl/qwen3vl_30B_config_v1.yaml) 的 `parallel.fsdp_plan.apply_modules` 即按这些路径配置，可对照参考。

### 1.2 迁移目标：插件式 FSDP2 后端

MindSpeed MM 新模型基于**插件式 FSDP2 后端**接入：训练入口为 `mindspeed_mm/fsdp/train/trainer.py`，通过一份顶层 YAML 驱动训练。本文即以该后端为准，使用启动脚本 `finetune_qwen3vl_30B_v1.sh` 与配置 `qwen3vl_30B_config_v1.yaml`。

明确目标后，迁移工作分为四部分，最终在仓库中产出以下文件：

| 工作项 | 落点 | Qwen3VL 的做法 |
|---|---|---|
| 模型接入 | `mindspeed_mm/fsdp/models/qwen3vl/` | 拷贝 HF modeling 进仓改造（见第 2 节） |
| 数据接入 | 无新增文件 | 复用通用 `huggingface` 数据集 + 内置 collator（见第 3 节） |
| 训练配置 | `qwen3vl_30B_config_v1.yaml` | 一份顶层 YAML（见第 4 节） |
| 启动脚本 | `examples/qwen3vl/finetune_qwen3vl_30B_v1.sh` | 通过 torchrun 启动训练器（见第 5 节） |

## 2. 模型接入

### 2.1 插件式 FSDP2 核心文件概览

模型接入的本质，是让模型类被框架的训练流程识别、构建并施加 FSDP2 策略。下表列出接入过程中涉及的核心文件，了解其职责有助于理解后续步骤（均位于 `mindspeed_mm/fsdp/` 下）：

| 文件 | 职责 | 接入时如何用到 |
|---|---|---|
| `train/trainer.py` | 训练总入口 | torchrun 启动它，按顺序加载 YAML、导入插件、构建模型与数据、进入训练循环 |
| `utils/register.py` | 注册器（`model_register` / `data_register`）与插件导入 | 模型/数据类用 `@model_register.register("<id>")` 注册；`training.plugin` 列出的目录被递归导入以触发注册 |
| `models/modelhub.py` | 模型构建中枢（`ModelHub.build`） | 先尝试 `AutoConfig.from_pretrained` 走 Transformers 风格构建，失败则走自定义模型构建 |
| `params/*_args.py`（如 `model_args.py`、`training_args.py`） | YAML 各段对应的参数定义（pydantic dataclass） | 写 YAML 字段前，以这里的定义为准（字段名、默认值、校验） |
| `distributed/torch_parallelize.py` | 并行/分片策略（`ParallelApplier`） | 按 `parallel.fsdp_plan`/`ep_plan` 对模型施加 `fully_shard` 分片，以及 TP、EP、prefetch 等 |
| `features/apply_features.py` | 训练特性应用（`FeaturesApplier`） | 按 `features` 段施加重计算、ChunkLoss、激活值卸载等 |
| `train/train_engine.py` | 单步训练逻辑（`TrainEngine`） | 以 `model(**batch, use_cache=False)` 调用模型并读取 `output.loss`，模型 forward 需匹配此调用方式 |
| `checkpoint/dcp_checkpointer.py` | 检查点读写（`DistributedCheckpointer`） | 训练检查点的保存与加载走 DCP 格式（meta init 加载 DCP 权重亦经此） |

后续 2.2~2.4 节的接入动作，都是围绕这几个文件展开。

### 2.2 接入方式的选择

模型接入有三种方式：自定义模型接入、Transformers 模型接入、第三方模型适配封装。**Qwen3VL 属于第二种"Transformers 模型接入"**：模型类保持继承 HF `PreTrainedModel`，沿用 HF 体系的构建与权重加载方式。

需要说明的是，同为 Transformers 模型接入，工程上有两种形态：上游模型无需修改内部逻辑时，直接 import 上游类、按需注册即可；而 Qwen3VL 需要在模型 **forward 内部**做改造，import 方式无法满足，因此将 `modeling_qwen3_vl_moe.py` 整体拷贝进仓再改造。需要改造的内容例如：

- 序列并行（Ulysses/Ring CP）：在视觉/文本前向中插入序列的切分与聚合通信；
- MoE 负载均衡辅助损失（aux loss）：在前向中计算辅助损失并累加进总 loss。

第三种"第三方模型适配封装"（外层 wrapper 负责权重加载与输入输出字段适配）适用于源模型结构不便修改的场景，同样覆盖不了上述 forward 内部改造，因此 Qwen3VL 未采用。

为新模型选择接入方式时，可按下表判断：

| 模型情况 | 接入方式 |
|---|---|
| 属 HF 体系，且无需改动内部逻辑 | Transformers 接入：直接 import 上游类、按需注册 |
| 属 HF 体系，但需深入 forward 改造（Qwen3VL 即此类） | Transformers 接入：拷贝 modeling 进仓改造，并尽量保持与上游文件的 diff 最小，便于后续跟随上游升级 |
| 非 HF 体系，或结构不便修改 | 自定义模型接入，或第三方适配封装 |

### 2.3 拷贝、注册与框架识别

模型文件放在约定目录下，Qwen3VL 只有两个文件：

```text
mindspeed_mm/fsdp/models/qwen3vl/
├── modeling_qwen3_vl_moe.py   # 从 transformers 拷贝并改造的模型实现
└── npu_patch.py               # NPU 融合算子替换
```

接入框架只需在顶层类上加一行注册装饰器（`modeling_qwen3_vl_moe.py`）：

```python
from mindspeed_mm.fsdp.utils.register import model_register

@model_register.register("qwen3_vl_moe")
class Qwen3VLMoeForConditionalGeneration(Qwen3VLMoePreTrainedModel, GenerationMixin):
    ...
```

注册名 `"qwen3_vl_moe"` 即后续 YAML 中 `model.model_id` 需要填写的值；框架会导入 `training.plugin` 列出的插件目录，使注册生效。

另一个要点：改造后的类**仍然继承 HF 的 `PreTrainedModel`**，因此框架会自动按 Transformers 方式构建模型并加载权重，无需自行实现加载逻辑（仅非 HF 体系的自定义模型才需实现 `_from_config`/`from_pretrained`）。

### 2.4 模型改造：必做项与可选增强

仓内版本的改动分为必做项和可选增强两类。建议先完成必做项、把模型以最朴素的形态跑通，可选增强按场景再叠加，不必一开始就全部完成。

必做项是让模型能被训练引擎正常调用：`forward` 接收 dataloader 产出的 batch 字段、并返回带 `.loss` 的输出对象（训练引擎以 `model(**batch_data, use_cache=False)` 调用并读取 `output.loss`）。继承自 HF `PreTrainedModel` 的模型还需设置 `accepts_loss_kwargs = False`（Qwen3VL 已自带，迁移其他 HF 模型时确认保留即可）。

完成必做项后，配合第 3、4 节的数据与 YAML 配置，模型即可正常训练。除此之外的改造都是可选的，仅在命中对应场景时才需要：需要**长序列训练**时，模型 forward 要适配序列并行（CP）的切分/聚合通信，否则保持 `parallel.ulysses_parallel_size: 1` 即可；追求**极致性能**时，可将热点算子替换为 `mindspeed_mm/fsdp/ops/` 下的融合算子；**MoE 模型需要专家负载均衡**时，模型要支持辅助损失。若你的模型不涉及这些场景，可跳过；Qwen3VL 同时涉及上述几类，其 `modeling_qwen3_vl_moe.py` 与 `npu_patch.py` 可作参考实现。

## 3. 数据接入

### 3.1 数据处理的复用与适配

原始数据（对话 json、图像/视频）按实际任务准备；迁移时需要判断**数据处理链路能否复用、还是要适配**。该链路由三部分组成，是否复用各看条件：

- **数据集构建**（`dataset_type`，负责加载原始数据并做字段映射、对话模板、tokenize 等预处理）：数据为标准对话格式时，复用注册名为 `huggingface` 的通用多模态数据集（`mindspeed_mm/fsdp/data/datasets/huggingface/`）；格式非标准（语音特征、视频 latent、自定义打包）时，用 `@data_register.register` 注册新的数据集构建逻辑；
- **多模态打包 Plugin**（`mindspeed_mm/fsdp/data/data_utils/func_utils/mm_plugin.py` 的 `PLUGINS` 表，处理特殊 token、视觉/视频 token 占位符展开，按模型不同）：模型已在 `PLUGINS` 中时直接复用，否则需新增或覆写对应 Plugin；
- **collator**（组 batch、产出训练字段）：有匹配实现时复用，否则在数据集中实现 `collate_fn` 或在 `data_collator.py` 注册新 collator。

Qwen3VL 三部分的实际选择：**数据集构建**复用通用 `huggingface`；**多模态打包**用 `Qwen3VLPlugin`（继承 `Qwen2VLPlugin`，按 Qwen3-VL 的视觉 token 占位、视频时间戳等覆写，把 `<image>`/`<video>` 占位符展开为带 vision 特殊 token 的序列并产出 `pixel_values`/`image_grid_thw`）；**collator** 复用 `DataCollatorForQwen2vl`，在 `data_collator.py` 注册为 `qwen3vl`。

### 3.2 数据接入的配置方式

数据集、Plugin、collator 就位后，数据接入主要体现在 YAML 的 `data` 段中（其中 `template` 字段会选定对应的 Plugin）。Qwen3VL 的关键配置（节选，其中路径均为示例、需按实际情况修改，完整内容见 `qwen3vl_30B_config_v1.yaml`）：

```yaml
data:
  dataset_param:
    dataset_type: huggingface          # 对应数据集注册名
    attr:                              # 键=框架概念，值=你的数据 json 中对应的名字（见下文说明）
      images: images
      messages: messages
      role_tag: role
      content_tag: content
      user_tag: user
      assistant_tag: assistant
    preprocess_parameters:
      model_name_or_path: /home/data/Qwen3-VL-30B-A3B-Instruct   # 原始 HF 目录，用于加载 tokenizer/processor
      image_max_pixels: 262144         # 图像分辨率上限，影响视觉 token 数与显存
    basic_parameters:
      template: qwen3_vl_nothink       # 对话模板，决定 prompt 拼接格式，必须与模型匹配
      cutoff_len: 1024                 # 截断长度
      dataset_dir: /home/usr/data/
      dataset: /home/usr/data/mllm_format_llava_instruct_data.json
      cache_dir: ./cache_dir/          # 预处理缓存；多机不要共享同一挂载目录
  dataloader_param:
    sampler_type: BaseRandomBatchSampler
    collate_param:
      collator_id: qwen3vl              # 对应内置 collator 名
```

关于 `attr` 映射的方向：**键是框架固定的概念名，值填你的数据中对应的内容**：`images`/`messages` 填数据 json 的列名，`role_tag`/`content_tag` 填每条消息内的键名，`user_tag`/`assistant_tag` 填角色字段的取值。本例数据中每条消息形如 `{"role": "user", "content": "..."}`，故配置 `role_tag: role`、`user_tag: user`；若你的数据是经典 sharegpt 格式（消息形如 `{"from": "human", "value": "..."}`），则应配置 `role_tag: from`、`content_tag: value`、`user_tag: human`。

两个易错点：

- `model_name_or_path` 在这里**只用来加载 tokenizer 和 processor**，不加载训练权重（权重由 `training.load` 指定，见 4.4 节）；
- `template` 填错会导致 prompt 拼接与模型预训练格式不一致，loss 正常下降但效果差，迁移新模型时务必确认模板。

## 4. 训练 YAML 配置

插件式 FSDP2 的全部训练行为由一份 YAML 驱动。下面按配置段说明 Qwen3VL 的选择，完整字段以样例 `qwen3vl_30B_config_v1.yaml` 为准。

### 4.1 parallel 段：分片计划的制定

```yaml
parallel:
  tensor_parallel_size: 1              # 插件式当前必须为 1
  fully_shard_parallel_size: auto      # 按总卡数自动设定 FSDP 分片组
  fsdp_plan:
    apply_modules:
      - model.visual.blocks.{*}        # 视觉每层单独分片
      - model.visual.merger
      - model.visual.deepstack_merger_list.{*}
      - model.visual
      - model.language_model.embed_tokens
      - model.language_model.layers.{*}   # LLM 每层单独分片
      - model.language_model
      - lm_head
    param_dtype: bf16
    reduce_dtype: fp32                 # 梯度规约用 fp32，保精度
  expert_parallel_size: 1
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts   # 预留：EP 开启时按专家切分
```

`apply_modules` 的取值来自第 1.1 节的模块树，`{*}` 通配层号；框架会对其中列出的每个模块、以及最外层 model 依次执行 `fully_shard` 分片（`apply_modules` 为空时只分片最外层 model）。常规微调直接沿用样例的这份配置即可；如需自定义，模块路径须取自模型 `named_modules()`，且开启 prefetch 时不要随意调整已验证配置中的模块顺序。

### 4.2 model 段：冻结决策

```yaml
model:
  model_id: qwen3_vl_moe               # 与 @model_register.register 的注册名一致
  model_name_or_path: /home/data/Qwen3-VL-30B-A3B-Instruct
  attn_implementation: flash_attention_2
  freeze:
    - model.visual                     # 微调场景冻结视觉编码器
```

`freeze` 列出的模块会被设为 `requires_grad=False`，不参与训练，也不再为其保存梯度与优化器状态（节省相应显存）。样例冻结了视觉编码器 `model.visual`；若任务需要训练视觉编码器，删除该行即可。

### 4.3 features 段：显存与 loss 策略

```yaml
features:
  loss_cfg:
    loss_type: default
    router_aux_loss_coef: 0.0          # MoE 辅助损失系数，仅 MoE 模型需要，默认 0 表示不启用
  recompute: true
  recompute_plan:
    apply_modules:                     # 视觉与 LLM 两部分均启用重计算
      - model.visual.blocks.{*}
      - model.language_model.layers.{*}
```

样例默认启用重计算（`recompute: true`，以计算换显存）。注意这些字段**必须放在顶层 `features:` 段**，放在 `model:` 段不会报错但完全不生效。

`recompute_plan` 中的 `model.visual.blocks`：视觉编码器被冻结后（见 4.2），反向不经过它，重计算对其不省显存；此处保留是为了在放开视觉编码器训练时无需改动该配置，确定不训练也可删去。

显存仍紧张时，可在 `features` 段进一步开启 ChunkLoss（`enable_chunk_loss`）、异步激活值卸载（`enable_activation_offload`）等特性，样例 YAML 已预留对应配置块。

### 4.4 training 段：权重加载方式

```yaml
training:
  micro_batch_size: 1
  gradient_accumulation_steps: 1       # 留空会关闭梯度累积，需累积时显式填数值
  lr: 1.0e-5
  lr_decay_style: cosine
  train_iters: 10000
  init_model_with_meta_device: true    # 参数量大的模型需开启
  # load: <DCP 权重目录>
  # save: <保存目录>
  plugin:
    - mindspeed_mm/fsdp/models/qwen3vl          # 启动时导入 -> 触发模型注册
    - mindspeed_mm/fsdp/data/datasets/huggingface   # 触发数据集注册
```

权重加载方式是大模型迁移中的一个关键决策：**为什么用 meta init + DCP 权重？** 大模型常规加载的初始化峰值内存很高，`init_model_with_meta_device` 用于降低它——先在 meta 设备上仅构建模型结构、不分配实际内存，再由 DCP（分布式检查点格式）按分片直接加载到各卡，每张卡只读自己那一份（以 Qwen3-VL-30B 为例，bf16 权重约 60GB，借此可避免多卡重复占用主机内存）。

meta init 需配合 DCP 格式权重使用，因此训练前要先把 HF 权重转换为 DCP（DCP 为分片检查点格式，转换一次可重复使用）。转换命令（在仓库根目录执行，假设 HF 权重已下载到 `ckpt/Qwen3-VL-30B-A3B-Instruct`）：

```bash
mm-convert GenericDCPConverter hf_to_dcp \
  --hf_dir ckpt/Qwen3-VL-30B-A3B-Instruct \
  --dcp_dir ckpt/Qwen3-VL-30B-A3B-Instruct-dcp
```

然后把 YAML 里 `training.load` 取消注释，填转换得到的 DCP 目录 `ckpt/Qwen3-VL-30B-A3B-Instruct-dcp`。转换工具的更多用法见 [权重转换](../pytorch/weight_conversion.md)。

`plugin` 列表则把第 2、3 节的成果接进框架：启动时按顺序导入这两个目录，模型与数据集完成注册，`model_id`/`dataset_type` 才找得到对应实现。

## 5. 启动脚本与运行

启动脚本 `examples/qwen3vl/finetune_qwen3vl_30B_v1.sh` 的骨架：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 按实际安装路径修改
export NON_MEGATRON=true            # 关键：选择插件式 FSDP2 初始化路径，必须设置
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1

NPUS_PER_NODE=16                    # 单机卡数，按实际修改
MASTER_ADDR=localhost               # 多机时改为主节点 IP
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3vl/qwen3vl_30B_config_v1.yaml
```

注意 torchrun 的入口是统一训练器 `mindspeed_mm/fsdp/train/trainer.py`，唯一参数就是那份 YAML，这意味着迁移新模型时启动脚本几乎可以原样复制，只改 YAML 路径和卡数。各环境变量的含义见脚本内注释。

卡数按实际硬件调整 `NPUS_PER_NODE` 即可，YAML 中 `fully_shard_parallel_size: auto` 会按总卡数自动设定分片组；卡数较少时可同步调小 `micro_batch_size`、`cutoff_len` 或开启更多显存优化（见 4.3 节）以适配显存。

确认数据与权重就绪后，在仓库根目录启动：

```bash
bash examples/qwen3vl/finetune_qwen3vl_30B_v1.sh
```

日志输出到 `logs/` 目录。权重下载、COCO 数据集准备等通用操作步骤本文不重复，按 [Qwen3VL README](../../../examples/qwen3vl/README_v1.md) 执行即可。

**如何确认跑通**：启动后训练日志会按 `log_interval` 周期打印每个 iteration 的关键指标，形如：

```text
iteration 1/10000 | consumed samples: 8 | elapsed time per iteration (ms): 6603.7 | learning rate: 0.000000E+00 | global batch size: 8 | loss: 1.016570E+01 | grad norm: 50.001 |
iteration 2/10000 | consumed samples: 16 | elapsed time per iteration (ms): 2231.6 | learning rate: 1.000000E-08 | global batch size: 8 | loss: 1.009848E+01 | grad norm: 49.063 |
```

只要日志能持续按 iteration 打印、`loss` 在合理范围且随训练总体下降、`grad norm` 未出现 NaN/Inf，即说明已正常跑通（首个 iteration 通常较慢，因包含编译与初始化开销，属正常现象）。若启动报错或卡住，可查阅 [FAQ](../FAQ.md)。

## 6. 训练跑通之后

- **精度对齐**：迁移的模型跑通后，建议与源仓（GPU/参考框架）对齐精度。具体做法是开启确定性计算（`training.use_deter_comp: true`）、固定随机种子、关闭数据 shuffle，消除随机性后对比两边的 loss 曲线是否一致；
- **性能调优**：采集 Profiling、定位瓶颈、按需开启序列并行/预取/ChunkLoss 等，见 [性能调优](../pytorch/performance_tuning.md)；
- **低成本微调**：显存预算有限时改用 [LoRA 微调（FSDP2）](./lora_finetune_fsdp2.md)；
- **导出权重**：训练产物为 DCP 格式，用 `mm-convert GenericDCPConverter dcp_to_hf` 转回 HF 格式，见 [权重转换](../pytorch/weight_conversion.md)。

本文以 Qwen3-VL 为例走完了完整迁移流程；更完整的接口说明与各配置段字段定义，可查阅 [FSDP2 迁移指南](./fsdp2_developer_migration_guide.md)。
