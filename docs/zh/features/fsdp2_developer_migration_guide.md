# MindSpeed MM FSDP2 迁移指南

本文面向需要将新模型、新数据集或第三方训练流程接入 MindSpeed MM FSDP2 后端的开发者。文档以仓内 `mindspeed_mm/fsdp` 新版插件式训练后端为主线，说明模型接入、数据接入、YAML 配置、启动脚本、权重加载与运行的开发约定。

MindSpeed MM FSDP2 基于 PyTorch FSDP2 构建。MindSpeed MM 在此基础上补充了面向昇腾平台的并行状态管理、模型注册、数据注册、DCP 检查点、重计算、LoRA、专家并行和多模态数据处理能力。

迁移工作通常分为以下部分：模型接入、数据接入、配置与启动脚本、运行。

## 1. 先识别 FSDP2 路线

MindSpeed MM 仓内同时存在两套容易混淆的 FSDP2 使用方式。它们的训练入口、配置文件和模型/数据接入方式不同，迁移前必须先确定目标路线。本文档聚焦新版插件式 FSDP2；新增模型和新增数据集推荐优先接入这一路线。Megatron 桥接式 FSDP2 属于过渡形态，主要用于存量入口兼容，后续不再作为新特性迭代方向；旧路线请继续参考 `docs/zh/features/fsdp2.md` 和对应 `examples/*/fsdp2_config.yaml`。

| 项目 | 新版插件式 FSDP2（本文主线） | Megatron 桥接式 FSDP2（旧路线） |
|---|---|---|
| 常见入口 | `mindspeed_mm/fsdp/train/trainer.py`，或 `mindspeed_mm/fsdp/tasks/*` 下的任务专用入口 | `pretrain_transformers.py`、`pretrain_vlm.py`、`pretrain_omni.py` 等 |
| 配置形态 | 一个顶层 YAML，包含 `parallel`、`data`、`model`、`training`、`tools` 等顶层配置块 | 主训练配置 + 额外 `fsdp2_config.yaml` |
| 启用方式 | 启动脚本直接传入顶层 YAML；通过 `training.plugin` 导入模型、数据等插件 | 命令行传入 `--use-torch-fsdp2`、`--fsdp2-config-path` |
| 模型接入 | `model_register` + `ModelHub.build` | `model_provider`、`forward_step`、`loss_func` |
| 数据接入 | `data_register` + `build_mm_dataset/build_mm_dataloader` | `train_valid_test_datasets_provider`、`get_batch` |
| 分片规模字段 | `parallel.fully_shard_parallel_size` | `sharding_size` |
| 子模块分片字段 | `parallel.fsdp_plan.apply_modules` | `sub_modules_to_wrap` |

上表只用于路线识别，避免把两套路由的入口和配置字段混用。

新版插件式链路可以概括为：

```text
torchrun
  -> mindspeed_mm/fsdp/train/trainer.py
  -> ConfigManager 加载顶层 YAML
  -> training.plugin 递归导入插件并触发注册
  -> ModelHub.build 构建模型
  -> LoRA（可选）注入
  -> TP/EP/Recompute/FSDP2 策略应用
  -> build_mm_dataset / build_mm_dataloader 构建数据
  -> TrainEngine: model(**batch_data, use_cache=False).loss
```

## 2. 迁移前准备

开始开发前，建议先准备以下信息。

| 准备项 | 需要确认什么 |
|---|---|
| 源仓入口 | 模型定义、权重加载、数据集构建、训练脚本分别在哪里 |
| 运行资产 | 模型权重路径、tokenizer/processor 路径、真实训练样本、图像/音频/视频/特征根目录 |
| 模型 I/O | `forward` 需要哪些训练 batch 字段，是否直接返回 `.loss`|
| 权重格式 | 直接从 Hugging Face/第三方权重加载，还是需要先转换为 DCP 权重以配合 meta 初始化 |
| 并行需求 | 是否需要 CP、EP、重计算、prefetch、activation offload、LoRA 等能力 |
| 参考案例 | 找一个同模态或同构建方式的 `examples/<case_name>` 作为基线 |

推荐优先从仓内已经接入的案例入口开始。下表仅作为入口索引，便于快速找到相近模型；具体字段复用和路线识别以实际脚本、YAML 和模型代码为准。

| 类型 | 案例 | 入口与配置 |
|---|---|---|
| 标准训练入口 `trainer.py` / VLM | Qwen3.5 Dense | `examples/qwen3_5/finetune_qwen3_5_{4B,9B,27B}.sh`，`qwen3_5_{4B,9B,27B}_config.yaml` |
| 标准训练入口 `trainer.py` / MoE VLM | Qwen3.5-MoE、Qwen3.6 | `examples/qwen3_5/qwen3_5_{35B,122B,397B}_config.yaml`，`examples/qwen3_6/qwen3_6_35B_A3B_config.yaml` |
| 标准训练入口 `trainer.py` / VLM | Qwen3VL v1 | `examples/qwen3vl/finetune_qwen3vl_30B_v1.sh`，`qwen3vl_30B_config_v1.yaml` |
| 标准训练入口 `trainer.py` / 全模态模型 | Qwen3Omni v1 | `examples/qwen3omni/finetune_qwen3omni_v1.sh`，`qwen3omni_config_v1.yaml` |
| 标准训练入口 `trainer.py` / 自定义 VLM | KimiK2.5 | `examples/kimik2_5/finetune_kimik2_5.sh`，`kimik2_5_config.yaml` |
| 标准训练入口 `trainer.py` / 视频/音频生成模型 | LTX2 | `examples/ltx2/finetune_ltx2_t2v.sh`，`finetune_ltx2_t2av.sh`，`ltx2_config_t2v.yaml`，`ltx2_config_t2av.yaml` |
| 标准训练入口 `trainer.py` / 语音合成 | Qwen3TTS | `examples/qwen3tts/finetune_qwen3tts.sh`，`qwen3tts_config.yaml` |
| 任务专用入口 / 语音识别 | FunASR | `examples/funasr/finetune_funasr.sh`，`mindspeed_mm/fsdp/tasks/funasr/trainer.py`，`funasr_config.yaml` |
| 任务专用入口 / 语音生成 | CosyVoice3 | `examples/cosyvoice3/finetune_cosyvoice3.sh`，`mindspeed_mm/fsdp/tasks/cosyvoice3/train.py`，`cosyvoice3_config.yaml` |

## 3. 模型接入

模型适配通常放在：

```text
mindspeed_mm/fsdp/models/<model_name>/
```

目标是让 `training.plugin` 能导入插件，让 `model.model_id` 能找到模型类，并让模型可以被 `ModelHub.build` 构建和后续 FSDP2 策略处理。

### 3.1 三种常见接入方式

| 方式 | 适用场景 | 开发重点 |
|---|---|---|
| 自定义模型接入 | 目标仓内维护模型主体，或源模型可按 MM 方式重构 | 继承 `BaseModel`，实现 `_from_config` 和 `from_pretrained` |
| Transformers 模型接入 | 模型基于 Hugging Face `PreTrainedModel` | 保持 HF `from_pretrained` 签名兼容，必要时注册模型类 |
| 第三方模型适配封装 | 源模型结构不适合大改 | 外层适配 wrapper 类负责加载、字段适配和 `.loss` 输出 |

`ModelHub.build` 会先尝试 `AutoConfig.from_pretrained(model.model_name_or_path)`。如果成功，通常走 Transformers 风格构建；如果失败，则走自定义模型构建，并调用 `from_pretrained(ModelArguments)`。

### 3.2 自定义模型最小接口

```python
import torch

from mindspeed_mm.fsdp.models.base_model import BaseModel
from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.utils.register import model_register


@model_register.register("<model_id>")
class XxxForTraining(torch.nn.Module, BaseModel):
    @classmethod
    def _from_config(cls, config: ModelArguments):
        # 只根据配置构建模型结构，如果开启 meta device 初始化，则必须实现。
        ...

    @classmethod
    def from_pretrained(cls, config: ModelArguments):
        # 从 config.model_name_or_path、config.checkpoint_path 或自定义字段加载权重。
        ...

    def forward(self, **batch):
        # TrainEngine 会调用 model(**batch_data, use_cache=False)，并读取 output.loss。
        ...
```

接口要点：

- 自定义模型链路向 `from_pretrained` 传入完整 `ModelArguments`，而不是单独的路径字符串。
- `_from_config` 必须构建完整模块结构，供 `training.init_model_with_meta_device: true` 使用。
- `forward` 需要能接收 dataloader 产出的 batch 字段，并兼容 `**kwargs`。
- 当 `model.loss_cfg.loss_type: raw` 时，模型输出必须包含 `.loss`；如果源模型返回 tuple 或 dict，建议封装成带 `.loss` 的对象。
- MoE 辅助损失需要模型原生支持。若从 Transformers 代码复制 MoE 模型，需确认原模型已支持 aux loss 计算，并参考 `mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py` 中 `Qwen3_5MoeForConditionalGeneration.overwrite_transformer_config` 覆盖 transformer config；同时确认需要捕获的 router logits 已配置在 `_can_record_outputs` 中，并且相关模块已正确使用 Transformers 的 `capture_outputs`。
- 特殊 token、embedding resize、`config.use_cache=False` 等逻辑只在源模型训练确实需要时添加，避免在迁移层引入不可追踪的行为差异。

## 4. 数据接入

数据适配通常放在：

```text
mindspeed_mm/fsdp/data/datasets/<dataset_or_model_name>/
```

目标是让配置里的 `data.dataset_param.dataset_type` 能找到数据构建逻辑。注册对象可以是 dataset class，也可以是 factory function。

### 4.1 数据集最小接口

框架会以 `basic_param/preprocess_param/dataset_param` 三个参数调用注册对象，因此源仓 dataset 构造函数通常需要做一层适配。

```python
from mindspeed_mm.fsdp.utils.register import data_register


@data_register.register("<dataset_type>")
class XxxDataset:
    def __init__(self, basic_param, preprocess_param, dataset_param=None, **kwargs):
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...

    def collate_fn(self, features):
        ...
```

或者：

```python
@data_register.register("<dataset_type>")
def build_xxx_dataset(basic_param, preprocess_param, dataset_param=None, **kwargs):
    return XxxDataset(...)
```

### 4.2 Dataloader 与 collate 约定

collate 选择规则如下：

1. 如果 dataset 对象实现了可调用的 `collate_fn`，优先使用 dataset 自己的 `collate_fn`。
2. 否则根据 `dataloader_param.collate_param.model_name` 从 `DATA_COLLATOR` 查找内置 collator，例如 `qwen3vl`、`qwen3omni`、`llm_pretrain`。
3. 对自定义数据集，若 batch 格式特殊，优先在 dataset 中实现 `collate_fn`，避免污染通用 collator。

### 4.3 训练 batch 字段

`TrainEngine` 会先把 batch 移到当前设备，再执行。
因此 batch key 必须能对上模型 `forward` 的入参。例如模型 `forward(input_ids, labels, pixel_values, **kwargs)`，数据侧就需要在 dataset 或 `collate_fn` 中产出同名字段。

迁移时经常需要适配 batch 字段，原因是源仓的数据样本格式、字段命名和 MindSpeed MM 的训练调用方式不一定一致。常见适配包括：

- 把源数据字段改成模型 `forward` 接收的名字。
- 在多模态任务中，把文本、图像、音频、视频或预计算特征组织成模型可直接读取的结构。
- 保持 batch 尽量为扁平 dict；当前 `move_to_device` 只处理顶层 tensor、tensor list、基础类型和 `None`，复杂嵌套结构需要自行在 collate 或模型入口处理。
- 浮点 tensor 会按 `parallel.fsdp_plan.param_dtype` 移动到设备并转换精度；整型 tensor 会保持整型。

常见字段示例：语言模型多见 `input_ids/labels/attention_mask`，图文模型多见 `pixel_values/image_grid_thw/image_flags`，语音模型多见 `speech_feat/speech_token/text_token`，视频生成模型多见 `video_latent/prompt_embeds/timesteps` 等。具体字段以当前模型 `forward` 为准。

## 5. YAML 配置

插件式 FSDP2 使用一份顶层 YAML，通常放在：

```text
examples/<model_name>/<model_name>_config.yaml
```

下面是一个基础骨架。实际模型可参考相近案例 YAML 删减或扩展字段。

```yaml
parallel:
  tensor_parallel_size: 1
  fully_shard_parallel_size: auto
  fsdp_plan:
    apply_modules:
      - model.language_model.layers.{*}
    # 开启 EP 或 FSDP prefetch 时建议配置；普通非 MoE 模型可按需删除。
    hook_modules:
      - model.language_model.layers.{*}
    param_dtype: bf16
    reduce_dtype: fp32
  ring_attention_size: 1
  ulysses_parallel_size: 1
  expert_parallel_size: 1
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts

data:
  dataset_param:
    dataset_type: <dataset_type>
    preprocess_parameters:
      model_name_or_path: <tokenizer_or_processor_path>
    basic_parameters:
      dataset_dir: <data_root>
      dataset: <train_data_path>
  dataloader_param:
    pin_memory: true
    shuffle: true
    dataloader_mode: sampler
    drop_last: true
    sampler_type: BaseRandomBatchSampler
    num_workers: 4
    collate_param:
      model_name: <collate_name>

model:
  model_id: <model_id>
  model_name_or_path: <model_path>
  trust_remote_code: true
  freeze: []
  loss_cfg:
    loss_type: raw
  recompute: false
  recompute_plan:
    apply_modules:
      - model.language_model.layers.{*}

training:
  micro_batch_size: 1
  gradient_accumulation_steps: 1
  seed: 42
  lr: 1.0e-5
  lr_decay_style: cosine
  lr_warmup_ratio: 0.1
  weight_decay: 0.0
  train_iters: 10
  clip_grad: 1.0
  init_model_with_meta_device: false
  optimizer: adamw
  adam_fused: true
  save_interval: 1000
  load: null
  save: null
  use_deter_comp: false
  plugin:
    - mindspeed_mm/fsdp/models/<model_name>
    - mindspeed_mm/fsdp/data/datasets/<dataset_or_model_name>

tools:
  profile:
    enable: false
  memory_profile:
    enable: false
```

关键一致性关系：

- `model.model_id` 必须与 `@model_register.register("<model_id>")` 一致。
- `data.dataset_param.dataset_type` 必须与 `@data_register.register("<dataset_type>")` 一致。
- `training.plugin` 必须包含模型插件和数据插件路径；插件路径可使用 `/`，导入时会转换为 Python 包路径。
- `parallel.fsdp_plan.apply_modules` 使用模型 `named_modules()` 中的模块路径模式；开启 prefetch 时不要随意调整已验证配置中的模块顺序。

重要字段按 YAML 分块理解：

并行策略字段：

| 字段 | 说明 |
|---|---|
| `parallel.fully_shard_parallel_size` | FSDP 分片组规模。`auto` 会按 `world_size // tensor_parallel_size` 推导。 |
| `parallel.tensor_parallel_size` | 当前插件式 FSDP2 代码中仍要求为 `1`；设置为非 `1` 会触发校验错误。 |
| `parallel.fsdp_plan.apply_modules` | 指定先被 `fully_shard` 包装的子模块；框架随后还会对最外层 model 调用 `fully_shard`。为空时只包装最外层 model。 |
| `parallel.fsdp_plan.hook_modules` | 指定 FSDP hook manager 挂载模块。开启 EP 时应配置到稳定的上层层级，例如 `model.language_model.layers.{*}`，否则专家层通信和预取容易带来显存压力。 |
| `parallel.fsdp_plan.cpu_offload` | 将 FSDP 参数等状态 offload 到 CPU；开启后初始化和通信后端也会走 CPU 相关路径，需结合内存与性能验证。 |
| `parallel.expert_parallel_size` / `ep_plan.apply_modules` | MoE 专家并行配置。仅在模型专家模块已适配 EP 时开启。 |

数据字段：

| 字段 | 说明 |
|---|---|
| `data.dataset_param.dataset_type` | 数据注册名，必须与 `@data_register.register("<dataset_type>")` 一致。 |
| `data.dataset_param.preprocess_parameters` | tokenizer、processor、采样和截断等预处理参数，具体字段由数据集实现读取。 |
| `data.dataset_param.basic_parameters` | 数据根目录、数据文件、模板、cache 等基础数据参数，具体字段由数据集实现读取。 |
| `data.dataloader_param.collate_param.model_name` | 内置 collator 名称；如果 dataset 自带 `collate_fn`，该字段不会被优先使用。 |

模型字段：

| 字段 | 说明 |
|---|---|
| `model.model_id` | 模型注册名，必须与 `@model_register.register("<model_id>")` 一致。|
| `model.model_name_or_path` | HF/第三方权重、config 或本地模型目录路径。 |
| `model.loss_cfg.loss_type` | 默认为 `raw`，表示直接使用模型输出的 `.loss`；|
| `model.freeze` | 按模块路径模式冻结参数。 |
| `model.recompute` / `recompute_plan.apply_modules` | 重计算配置，以计算换显存；模块路径同样来自 `named_modules()`。 |

训练与检查点字段：

| 字段 | 说明 |
|---|---|
| `training.micro_batch_size` / `gradient_accumulation_steps` | 单卡 micro batch 与梯度累积步数；`gradient_accumulation_steps` 为空时框架会关闭梯度累积。 |
| `training.init_model_with_meta_device` | 是否先在 meta device 上构建模型结构，用于大模型降低初始化峰值内存。 |
| `training.load` / `save` | 权重/检查点加载与保存路径；为空时对应动作不执行。`load` 可指向 DCP 目录或 HF safetensors 目录。 |
| `training.plugin` | 需要导入的模型、数据等插件路径。 |

工具字段：

| 字段 | 说明 |
|---|---|
| `tools.profile` | 性能 profiling 配置。 |
| `tools.memory_profile` | 显存快照配置。 |

当 `fully_shard_parallel_size: 1` 且不使用 meta 初始化时，框架会退化为 DDP 包装，便于小规模调试。

## 6. 权重加载与检查点

插件式 FSDP2 默认使用 `DistributedCheckpointer`，底层基于 `torch.distributed.checkpoint` 保存和加载 DCP 格式状态。

### 6.1 常见加载方式

| 场景 | 建议方式 |
|---|---|
| 模型可直接从 HF/第三方权重加载，且单机内存足够 | `training.init_model_with_meta_device: false`，在模型 `from_pretrained` 中加载原始权重 |
| 在线加载 HF 权重 | 设置 `training.init_model_with_meta_device: true`, 在`training.load: <hf_dir>`配置原始权重路径 |
| 大模型先离线转 DCP 再加载 | 先用 `hf_to_dcp` 转换；设置 `training.init_model_with_meta_device: true` 和 `training.load: <dcp_dir>` |
| 从框架保存的训练检查点续训 | 设置 `training.load`，按需配置 `no_load_optim`、`no_load_rng`、`load_strict` |

meta 初始化的关键语义：

- `init_model_with_meta_device: true` 时，模型先在 meta device 上构建结构，再根据 `training.load` 初始化参数。
- 如果 `training.load` 不为空，框架会将参数搬到目标设备或 CPU offload 设备，再由 DCP 加载状态。

保存相关字段：

- `training.save`：检查点保存根目录；为空时不保存。
- `training.save_interval`：按迭代步间隔保存。
- `training.no_save_optim` / `training.no_save_rng`：控制是否保存优化器和随机数状态。
- `training.no_load_optim` / `training.no_load_rng`：控制是否恢复优化器和随机数状态。
- `training.load_strict`：传给 DCP load planner；调试迁移时可先放宽，正式训练应尽量保持严格匹配。

## 7. 启动脚本

启动脚本通常放在：

```text
examples/<model_name>/finetune_<model_name>.sh
```

已有案例建议直接从对应脚本启动；新增模型可参考下面的骨架：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export NON_MEGATRON=true
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    examples/<model_name>/<model_name>_config.yaml
```

常见环境变量含义：

| 变量 | 作用 |
|---|---|
| `NON_MEGATRON=true` | 选择非 Megatron 初始化路径，适配插件式 FSDP2 入口。 |
| `HCCL_CONNECT_TIMEOUT` | 设置 HCCL 设备间建链超时时间，单位为秒；多卡或多机场景通常会调大。 |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | 配置 torch-npu 缓存分配器，启用可扩展内存段，缓解大模型训练中的内存碎片问题。 |
| `MULTI_STREAM_MEMORY_REUSE` | 控制多流内存复用策略；建议沿用相近模型已验证的取值。 |
| `TASK_QUEUE_ENABLE` | 控制 task queue 算子下发队列优化等级，常见取值为 `0/1/2`。 |
| `CPU_AFFINITY_CONF` | 控制 CPU 侧任务绑核策略，减少任务调度和 NUMA 访问带来的波动。 |

分布式启动变量含义：

| 变量 | 作用 |
|---|---|
| `NPUS_PER_NODE` | 当前节点参与训练的 NPU 数，也对应 `torchrun --nproc_per_node`。 |
| `NNODES` | 参与训练的节点总数。 |
| `NODE_RANK` | 当前节点编号，单机通常为 `0`，多机从 `0` 到 `NNODES-1` 排列。 |
| `MASTER_ADDR` | 主节点地址，多机时通常填写 `NODE_RANK=0` 的节点 IP。 |
| `MASTER_PORT` | 主节点通信端口，选择当前机器上未被占用的端口。 |
| `WORLD_SIZE` | 全局进程数，一般为 `NPUS_PER_NODE * NNODES`。 |

多机训练时，需要在每个节点上设置相同的 `MASTER_ADDR`、`MASTER_PORT`、`NNODES`，并为每个节点设置不同的 `NODE_RANK`。

## 8. 运行

完成上述模型、数据、 YAML 配置和启动脚本开发后，即可使用 `bash examples/<model_name>/finetune_<model_name>.sh` 启动训练。
