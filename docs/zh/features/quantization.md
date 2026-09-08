# MindSpeed MM FSDP2后端低精度训练指南

## 适用后端

FSDP2

## 介绍

本指南旨在帮助用户在 MindSpeed-MM 框架下，基于 FSDP2 后端实现低精度训练（如 mxfp8 等），
提升训练效率与显存利用率。通过配置量化配置（QuantizeConfig）与低精度all-gather模式，可在保持模型精度的前提下，显著降低通信开销与内存占用，适用于大模型训练场景。

## 前置依赖
 
使用前需安装FSDPTurbo：
 
```bash
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
pip install -e .
```

## 使用方法

### 1. 参数概览

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `quant_recipe` | str | null | 使用的量化配方名，同时也是使能量化的标识，支持 `mxfp8` |
| `quant_format` | str | null | 量化数据类型格式，支持 `e4m3`、`hybrid` |
| `block_size` | int | 32 | 每次量化的元素个数 |
| `quant_apply_modules` | List[str] | [] | 应用量化的层或模块列表，如 `["model.language_model.layers.{*}"]` |
| `quant_ignored_modules` | List[str] | [] | 不应用量化的子模块列表，如 `["*lm_head", "*gate"]` |
| `converters` | List[str] | [] | 量化转换器列表，支持 `quantize.linear.mx`、`quantize.moe.mx` |
| `enable_fsdp_low_precision_all_gather` | bool | `true` | 是否启用低精度通信 |
| `fsdp_low_precision_all_gather_mode` | str | `on-demand` | FSDP低精度all-gather模式, 支持 `on-demand`、 `all` |

### 2. 核心参数说明

#### ✅quant_recipe

quant_recipe的格式为：

```python
<scaling_strategy>_<scaling_granularity>[-blocksize0-blocksize1-blocksize2]_<inputs_dtype>_<weight_dtype>_<grads_dtype>
```

| 字段 | 说明 |
|------|------|
| `scaling_strategy` | 缩放策略，如 `dynamic`、`delayed` |
| `scaling_granularity` | 缩放粒度，如 `mx`（仅支持）、`per_tensor`、`per_channel` |
| `blocksize0-blocksize1-blocksize2` | 可选，块大小（仅用于块量化） |
| `inputs_dtype` / `weight_dtype` / `grads_dtype` | 输入、权重、梯度的数据类型，如 `E4M3`、`E5M2` |

#### 预定义配方示例

- `mxfp8`: `dynamic_mx-1-1-32_E4M3_E4M3_E4M3`
→ 支持 MX 量化策略，适用于大多数场景。

> ⚠️ 当前仅支持 `mxfp8` 缩放策略，后续将支持更多策略与配方。

#### ✅quant_format

量化数据类型格式，用于指定量化后的数值格式，例如 `e4m3`，不区分大小写。

#### ✅block_size

每次量化的元素个数，默认为 32。控制量化时分组粒度，较小的块大小可提供更精细的量化，可能带来更好的精度保持。

> ⚠️ 当前仅支持 `32` 量化块大小。

#### ✅quant_apply_modules

指定需要应用量化的层或模块，支持通配符。

**示例：**

```python
["model.layers.{*}"]                # 应用于所有 Transformer 层
["model.layers.0.self_attn"]        # 应用于第 0 层的自注意力模块
```

#### ✅quant_ignored_modules

指定不应用量化的子模块列表，支持通配符。

```python
["*lm_head"]       # 不应用量化到 lm_head 模块
["*gate"]          # 不应用量化到 MLP 中的 gate 部分
```

#### ✅converters

指定使用的量化转换器列表，目前支持以下类型：

- `quantize.linear.mx`：适用于普通线性层（如 FFN、Attention）的 MX 策略线性量化。
- `quantize.moe.mx`：专用于 MoE 模型专家模块中 GMM 的 MX 量化。

> 💡 在 MoE 模型中可以同时使用 `quantize.linear.mx` 和 `quantize.moe.mx`。

#### ✅enable_fsdp_low_precision_all_gather

是否启用 FSDP 的低精度 all-gather 模式。启用后，在前向/反向传播中，FSDP 会以低精度权重（如 mxfp8）进行参数的 all-gather 操作，显著降低通信开销和内存占用。

在开启低精度训练的同时，可以进一步启用该模式以最大化效率提升。

#### ✅fsdp_low_precision_all_gather_mode

指定低精度 all-gather 的通信模式：

| 模式 | 说明 |
|------|------|
| `on-demand` | 仅在前向或反向传播时，通信当前所需的权重 |
| `all` | 前向和反向均通信全部权重 |

> ⚠️ `all` 模式下 AG 通信全部权重会造成通信量翻倍，通信时间相较于 bf16 无明显变化；同时因需要通信缩放因子等额外参数，显存会有略微增长。

### 3. 示例脚本

以下是一个示例启动脚本，展示了如何配置量化参数与低精度通信：

```yaml
training:
  quantization_plan:
    quant_recipe: mxfp8
    quant_format: e4m3
    block_size: 32
    quant_apply_modules: ["model.layers.{*}"]
    quant_ignored_modules: ["*lm_head", "*gate"]
    converters: ["quantize.linear.mx", "quantize.moe.mx"]
    enable_fsdp_low_precision_all_gather: true
    fsdp_low_precision_all_gather_mode: "on-demand"
```

只需要在原有的训练配置文件基础上，在 `training` 字段下添加 `quantization_plan` 中的量化相关参数，即可启用低精度训练与通信。参考示例配置文件：`examples/qwen3vl/qwen3vl_30B_config_v1_A5.yaml`。

## 注意事项

- ⚠️目前低精度训练相关功能仅支持在950机器上运行，910B&C等机器不支持。
- ⚠️低精度训练过程中可能引起精度损失，造成模型性能下降，非框架本身问题，建议谨慎使用。
