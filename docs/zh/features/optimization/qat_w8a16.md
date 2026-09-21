# MindSpeed MM FSDP2 后端 QAT W8A16 量化训练指南

## 介绍

本指南介绍如何在 MindSpeed-MM 框架下，基于 FSDP2 后端使用 FSDPTurbo 的 QAT W8A16 量化能力进行训练。

W8A16 将权重量化为 FP8 e4m3（per-32-block MX 共享指数格式），激活值保持高精度，反向传播使用直通估计器（STE）。与 MXFP8 不同，W8A16 **不触碰** FSDP 低精度 all-gather 通信路径（不引入 `PreQuantWeight`/`PostQuantWeight` 张量子类），因此可以与 MindSpeed-MM 的 FSDP2 分片透明组合，无需额外通信配置。

### 工作原理

W8A16 的 converter 由 FSDPTurbo 自动注册。当 MindSpeed-MM 的 `ParallelApplier.apply_quantization_modules` 导入 `fsdp_turbo.quantization.converter.model_converter.build_model_converter` 时，FSDPTurbo 的 `quantization` 包初始化会自动执行 `w8a16_converter.py`，将以下 converter 注册到全局 registry：

- `quantize.linear.w8a16`：将模型中的 `nn.Linear` 替换为 `W8A16Linear`，权重经 `w8a16_fake_quant` 做 FP8 e4m3 per-32-block 量化，激活值保持高精度，反向使用 STE

用户只需将 `quant_recipe` 设为 `w8a16` 并在 `converters` 中指定 converter 名称即可启用，无需额外插件或桥接模块。

## 前置依赖

W8A16 量化依赖 FSDPTurbo 包，使用前需安装：

```bash
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
pip install -e .
```

> ⚠️ 仅当配置中实际选择了 W8A16 converter 时才会触发量化转换。

## 使用方法

### 1. 参数说明

W8A16 量化复用 `training.quantization_plan` 配置，相关参数如下：

| 参数 | 类型  | 说明 |
|------|------|------|
| `quant_recipe` | str |  W8A16 场景下设为 `w8a16`，作为量化的使能标识 |
| `quant_apply_modules` | List[str] | 应用量化的层或模块，支持通配符 |
| `quant_ignored_modules` | List[str] | 不应用量化的子模块列表 |
| `converters` | List[str]  | 使用的量化转换器，W8A16 场景配置 `quantize.linear.w8a16` |

> 💡 `enable_fsdp_low_precision_all_gather` 与 `fsdp_low_precision_all_gather_mode` 是 MXFP8 专用参数，W8A16 不使用，无需配置。

### 2. 核心参数说明

#### quant_recipe

设为 `w8a16` 以使能量化流程。`quant_recipe` 非空时，MindSpeed-MM 的 `apply_quantization_modules` 会触发 `build_model_converter` 构建并执行 converter。

> 💡 `quant_recipe` 的值本身不被解析为具体量化格式，实际量化行为由 `converters` 列表中的 converter 名称决定。

#### quant_apply_modules

指定需要应用 W8A16 量化的层或模块，支持通配符。一般匹配 decoder layer 前缀（含 `layers.\d+`），层内所有 `nn.Linear` 均被量化。

```yaml
quant_apply_modules:
  - "model.language_model.layers.{*}"
```

> 💡 `lm_head` / `embed_tokens` 等不在 `layers.\d+` 下的模块会自动跳过。

#### quant_ignored_modules

指定不应用量化的子模块列表，支持通配符。如需排除特定子模块可在此列出，默认为空。

```yaml
quant_ignored_modules: []
```

#### converters

指定使用的 W8A16 量化转换器，目前支持以下类型：

- `quantize.linear.w8a16`：适用于普通线性层（如 FFN、Attention），将 `nn.Linear` 替换为 `W8A16Linear`

### 3. 启用步骤

在训练配置的 `training` 字段下添加 `quantization_plan`，将 `quant_recipe` 设为 `w8a16` 并指定 `converters` 即可启用 W8A16 量化：

```yaml
training:
  quantization_plan:
    quant_recipe: w8a16              # 使能量化流程
    quant_apply_modules:
      - "model.language_model.layers.{*}"
    quant_ignored_modules: []
    converters:
      - "quantize.linear.w8a16"
```

### 4. 完整示例

以下为 Qwen3.5-27B 模型启用 W8A16 线性层量化的完整配置示例：

```yaml
training:
  micro_batch_size: 2
  gradient_accumulation_steps: 1
  # ... 其他训练参数 ...
  use_deter_comp: false
  # QAT W8A16 量化配置：权重 fake-quant 到 FP8 e4m3 (per-32-block MX 共享指数)，
  # 激活值保持高精度，反向用直通估计器(STE)。
  quantization_plan:
    quant_recipe: w8a16              # 使能量化流程
    quant_apply_modules:
      - "model.language_model.layers.{*}"
    quant_ignored_modules: []
    converters:
      - "quantize.linear.w8a16"
  plugin:
    - mindspeed_mm/fsdp/models/qwen3_5
    - mindspeed_mm/fsdp/data/datasets/huggingface
```

## 注意事项

- ⚠️ 使用前请确保已安装 FSDPTurbo（`pip install -e FSDPTurbo`），否则在 converter 实例化时会报 `ModuleNotFoundError`。
- ⚠️ `quant_recipe` 设为 `w8a16`，`converters` 中必须配置 `quantize.linear.w8a16`，两者缺一不可。
- ⚠️ W8A16 为 fake-quant 训练路径，不修改 FSDP 低精度 all-gather 通信路径，因此 `enable_fsdp_low_precision_all_gather` / `fsdp_low_precision_all_gather_mode` 对其无效，无需配置。
- ⚠️ 低精度训练相关功能仅支持在 950 系列机器上运行，910B/C 等机器不支持。
- ⚠️ 低精度训练过程中可能引起精度损失，造成模型性能下降，非框架本身问题，建议谨慎使用。
