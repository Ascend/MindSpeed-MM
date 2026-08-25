# FSDP2 后端 LoRA 微调

LoRA（Low-Rank Adaptation）是一种高效的模型微调方法，通过在权重上添加低秩矩阵，使得微调过程更为轻量，节省计算资源和存储空间。

MindSpeed MM 在 FSDP2 后端原生支持 LoRA 微调，无需依赖 Megatron 并行框架，使用更为简洁的 YAML 配置方式即可完成 LoRA 微调任务。

## 原理简介

LoRA 的核心思想是将模型的参数更新分解为低秩的形式。具体步骤如下：

- **分解权重更新**：在传统的微调方法中，直接对模型的权重进行更新。而 LoRA 通过在每一层的权重矩阵中引入两个低秩矩阵 $A$ 和 $B$ 进行替代。即：

$$
W' = W + A \cdot B
$$

其中，$W'$ 是更新后的权重，$W$ 是原始权重，$A$ 和 $B$ 是需要学习的低秩矩阵。

- **降低参数量**：由于 $A$ 和 $B$ 的秩较低，所需的参数量显著减少，节省了存储和计算成本。

## 使能 LoRA 微调

在 FSDP2 后端中，LoRA 微调通过 YAML 配置文件中的 `training.lora` 字段进行配置，无需在启动脚本中添加额外的命令行参数。

### 配置示例

在模型的 YAML 配置文件（如 `examples/qwen3_5/qwen3_5_35B_config.yaml`）的 `training` 字段下添加 `lora` 配置：

```yaml
training:
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  lr: 1.0e-4
  train_iters: 100
  save_interval: 20
  save: ./save_path
  # ... 其他训练参数

  lora:
    enable: true
    rank: 8
    alpha: 16
    # all-linear: 自动展开模型中所有 nn.Linear 层（推荐，无需手写模块名）
    # 配合 model.freeze 排除组件（见下文 target_modules 配置说明）
    target_modules: all-linear
    dropout: 0.0
    init_lora_weights: true
    pretrained_lora_path: null
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明                                                                                                                                                               |
| :--- | :--- | :--- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `enable` | bool | `false` | 是否开启 LoRA 微调                                                                                                                                                     |
| `rank` | int | `8` | LoRA 低秩矩阵的维度。较低的 rank 值会使用更少的参数更新，减少计算量和内存消耗                                                                                                                     |
| `alpha` | int | `16` | 控制 LoRA 权重对原始权重的影响比例，数值越高影响越大。一般保持 `α/r` 为 2                                                                                                                     |
| `target_modules` | str \| List[str] | `["q_proj", "k_proj", "v_proj"]` | 需要添加 LoRA 的模块名称，或者通配符模式，或特殊关键字 `all-linear`                                                                                                          |
| `dropout` | float | `0.0` | LoRA 层的 dropout 比例，取值范围 `[0, 1)`                                                                                                                                 |
| `init_lora_weights` | bool \| str | `True` | 权重初始化方式。`True`；`False`；或选择以下字符串值：`"gaussian"`, `"eva"`, `"olora"`, `"pissa"`, `"pissa_niter_[number of iters]"`, `"corda"`, `"loftq"`, `"orthogonal"` |
| `pretrained_lora_path` | str | `null` | 预训练 LoRA 权重路径（可选），支持 `.safetensors` 和 `.pt/.bin` 格式                                                                                                              |
| `disable_peft_moe_conversion` | bool | `true` | 屏蔽 PEFT 对 MoE 模型 `gate_proj`/`up_proj`/`down_proj` 的 `target_modules→target_parameters` 自动重定向，使 LoRA 打在 `shared_expert` 的 `nn.Linear` 而非路由专家的 `nn.Parameter` 上。仅 MoE 模型相关 |

### target_modules 配置说明

`target_modules` 支持三种匹配模式：

#### 1. `all-linear` 关键字（推荐）

将 `target_modules` 设为 `all-linear`（字符串或列表元素均可），框架会自动扫描模型中所有 `nn.Linear` 层并注入 LoRA，无需手写模块名。

```yaml
lora:
  target_modules: all-linear
```

**多模态组件排除**：`all-linear` 默认匹配所有 `nn.Linear`，包括 ViT、aligner。若只想对语言模型注入 LoRA，可配合 `model.freeze` 进行组件冻结，被 `model.freeze` 覆盖的模块不会注入 LoRA：

```yaml
model:
  freeze:
    - model.visual        # 排除整个视觉塔（含 merger/aligner），LoRA 只打在 language_model
    # - model.visual.blocks   # 仅排除 vit blocks，保留 aligner(merger) 的 LoRA
lora:
  target_modules: all-linear
```

> MoE 模型：`all-linear` 会命中 `shared_expert` 的 `nn.Linear`（`gate_proj`/`up_proj`/`down_proj`），配合 `disable_peft_moe_conversion=true` 可防止 PEFT 把这些层重定向到路由专家层。

#### 2. 精确匹配

直接指定模块名称，如 `"q_proj"` 会匹配所有以 `q_proj` 结尾的模块：

```yaml
target_modules:
  - "q_proj"
  - "k_proj"
```

#### 3. 通配符匹配

使用 `{*}` 作为通配符，如 `"model.language_model.layers.{*}.self_attn.q_proj"` 会匹配 `layers.0`, `layers.1` 等所有层，适合需要精确控制注入范围的场景：

**仅对 Attention 模块进行 LoRA 微调**：

```yaml
target_modules:
  - "model.language_model.layers.{*}.self_attn.q_proj"
  - "model.language_model.layers.{*}.self_attn.k_proj"
  - "model.language_model.layers.{*}.self_attn.v_proj"
  - "model.language_model.layers.{*}.self_attn.o_proj"
```

**仅对 MLP 模块进行 LoRA 微调**：

```yaml
target_modules:
  - "model.language_model.layers.{*}.mlp.shared_expert.gate_proj"
  - "model.language_model.layers.{*}.mlp.shared_expert.up_proj"
  - "model.language_model.layers.{*}.mlp.shared_expert.down_proj"
```

## 加载预训练 LoRA 权重

若需加载预训练 LoRA 权重进行续训，需配置 `pretrained_lora_path` 参数：

```yaml
training:
  lora:
    enable: true
    pretrained_lora_path: ./save_path/iter_xxx  # 替换为 LoRA 权重保存路径
```

## 权重保存

### 仅保存 LoRA 权重

训练过程中仅保存 LoRA 适配器权重，保存格式为 safetensors，保存的文件结构：

```bash
save_path/
├── lora_adapter.safetensors
└── ...
```

## 启动训练

配置完成后，使用与全量微调相同的启动脚本即可：

```shell
bash examples/qwen3_5/finetune_qwen3_5_xxB.sh
```

训练启动后，会自动打印 LoRA 配置摘要，包括匹配的模块数量、可训练参数量等信息。

## 合并lora权重到HuggingFace权重

```bash
cd checkpoint/common
python merge_lora_safetensors_to_base.py \
    --base_hf_dir ./Qwen3.5-27B \
    --lora_safetensors ./save_path/lora_adapter_iteration_10.safetensors \
    --save_merged_hf_dir ./merged_qwen3_5_27B_lora
```

## lora断点续训

断点续训时，yaml配置文件中`load`路径需要指向上次训练保存的 checkpoint 路径。断点续训前一次的训练必须配置`no_save_optim`、`no_save_rng`为false，断点续训时`no_load_optim`、`no_load_rng`设置为false，才能恢复优化器状态。断点续训完成后可使用权重转换脚本，合并lora权重到HuggingFace权重。

## lora微调支持模型

qwen3vl，qwen3.5，qwen3.6，qwen3.8，qwen3omni

## 注意事项

- **依赖安装**：FSDP2 LoRA 微调依赖 `peft` 库，请确保已安装：`pip install peft`
- **冻结模块**：开启 LoRA 微调后，基础模型参数会被自动冻结，仅 LoRA 适配器参数参与训练
- **精度处理**：LoRA 参数会自动转换为 `float32` 精度进行训练，以保证训练稳定性
- **权重验证**：训练启动时会自动验证 LoRA 权重是否包含 NaN 或 Inf 值
- **分布式训练**：在 FSDP2 分布式训练环境下，LoRA 权重保存会自动处理 DTensor 分片，无需额外配置
- **与 Megatron 后端的区别**：FSDP2 后端使用 YAML 配置方式，而非命令行参数（如 `--lora-r`、`--lora-alpha` 等）
- **MoE路由专家参数微调**：MoE 模型的路由专家参数使用`nn.Parameter`类型，目前暂不支持LoRA微调，该功能正在开发中。

## 参考文献

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
