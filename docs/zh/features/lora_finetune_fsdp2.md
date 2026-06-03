# FSDP2 后端 LoRA 微调【实验特性】

LoRA（Low-Rank Adaptation）是一种高效的模型微调方法，通过在权重上添加低秩矩阵，使得微调过程更为轻量，节省计算资源和存储空间。

> **状态**：【实验特性】
> MindSpeed MM 在 FSDP2 后端原生支持 LoRA 微调，无需依赖 Megatron 并行框架，使用更为简洁的 YAML 配置方式即可完成 LoRA 微调任务。

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

在模型的 YAML 配置文件（如 `examples/qwen3_5/qwen3_5_4B_config.yaml`）的 `training` 字段下添加 `lora` 配置：

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
    target_modules:
      - "model.language_model.layers.{*}.self_attn.q_proj"
      - "model.language_model.layers.{*}.self_attn.k_proj"
      - "model.language_model.layers.{*}.self_attn.v_proj"
      - "model.language_model.layers.{*}.self_attn.o_proj"
      - "model.language_model.layers.{*}.mlp.gate_proj"
      - "model.language_model.layers.{*}.mlp.up_proj"
      - "model.language_model.layers.{*}.mlp.down_proj"
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
| `target_modules` | List[str] | `["q_proj", "k_proj", "v_proj"]` | 需要添加 LoRA 的模块名称或通配符模式                                                                                                                                            |
| `dropout` | float | `0.0` | LoRA 层的 dropout 比例，取值范围 `[0, 1)`                                                                                                                                 |
| `init_lora_weights` | bool \| str | `True` | 权重初始化方式。`True`；`False`；或选择以下字符串值：`"gaussian"`, `"eva"`, `"olora"`, `"pissa"`, `"pissa_niter_[number of iters]"`, `"corda"`, `"loftq"`, `"orthogonal"` |
| `pretrained_lora_path` | str | `null` | 预训练 LoRA 权重路径（可选），支持 `.safetensors` 和 `.pt/.bin` 格式                                                                                                              |

### target_modules 配置说明

`target_modules` 支持两种匹配模式：

- **精确匹配**：直接指定模块名称，如 `"q_proj"` 会匹配所有以 `q_proj` 结尾的模块
- **通配符匹配**：使用 `{*}` 作为通配符，如 `"model.language_model.layers.{*}.self_attn.q_proj"` 会匹配 `layers.0`, `layers.1` 等所有层

以 Qwen3.5 模型为例，常见的 `target_modules` 配置：

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
  - "model.language_model.layers.{*}.mlp.gate_proj"
  - "model.language_model.layers.{*}.mlp.up_proj"
  - "model.language_model.layers.{*}.mlp.down_proj"
```

**同时对 Attention 和 MLP 模块进行 LoRA 微调**：

```yaml
target_modules:
  - "model.language_model.layers.{*}.self_attn.q_proj"
  - "model.language_model.layers.{*}.self_attn.k_proj"
  - "model.language_model.layers.{*}.self_attn.v_proj"
  - "model.language_model.layers.{*}.self_attn.o_proj"
  - "model.language_model.layers.{*}.mlp.gate_proj"
  - "model.language_model.layers.{*}.mlp.up_proj"
  - "model.language_model.layers.{*}.mlp.down_proj"
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

## 合并lora权重到HuggingFase权重

```bash
cd checkpoint/common
python merge_lora_safetensors_to_base.py \
    --base_hf_dir ./Qwen3.5-27B \
    --lora_safetensors ./save_path/lora_adapter_iteration_10.safetensors \
    --save_merged_hf_dir ./merged_qwen3_5_27B_lora
```

## lora断点续训

断点续训时，yaml配置文件中`load`路径需要指向上次训练保存的 checkpoint 路径。断点续训前一次的训练必须配置`no_save_optim`、`no_save_rng`为false，断点续训时`no_load_optim`、`no_load_rng`设置为false，才能恢复优化器状态。断点续训完成后可使用权重转换脚本，合并lora权重到HuggingFace权重。

## 注意事项

- **依赖安装**：FSDP2 LoRA 微调依赖 `peft` 库，请确保已安装：`pip install peft`
- **冻结模块**：开启 LoRA 微调后，基础模型参数会被自动冻结，仅 LoRA 适配器参数参与训练
- **精度处理**：LoRA 参数会自动转换为 `float32` 精度进行训练，以保证训练稳定性
- **权重验证**：训练启动时会自动验证 LoRA 权重是否包含 NaN 或 Inf 值
- **分布式训练**：在 FSDP2 分布式训练环境下，LoRA 权重保存会自动处理 DTensor 分片，无需额外配置
- **与 Megatron 后端的区别**：FSDP2 后端使用 YAML 配置方式，而非命令行参数（如 `--lora-r`、`--lora-alpha` 等）

## 参考文献

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
