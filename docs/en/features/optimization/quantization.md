# MindSpeed MM FSDP2 Backend Low-Precision Training Guide

## Applicable Backends

FSDP2

## Introduction

This guide helps users implement low-precision training (such as MXFP8) based on the FSDP2 backend within the MindSpeed MM framework,
improving training efficiency and memory utilization. By configuring the quantization configuration (`QuantizeConfig`) and the low-precision all-gather mode, communication overhead and memory usage can be significantly reduced while maintaining model precision, making it suitable for large-model training scenarios.

## Prerequisites
 
FSDPTurbo must be installed before use:
 
```bash
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
pip install -e .
```

## How to Use

### 1. Parameter Overview

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `quant_recipe` | string | null | The name of the quantization recipe to use, which also serves as the identifier for enabling quantization. `mxfp8` is supported |
| `quant_format` | string | null | The data type format of quantization. `e4m3` and `hybrid` are supported |
| `block_size` | int | 32 | The number of elements quantized at a time |
| `quant_apply_modules` | List[str] | [] | The list of layers or modules to which quantization is applied, such as `["model.language_model.layers.{*}"]` |
| `quant_ignored_modules` | List[str] | [] | The list of submodules to which quantization is not applied, such as `["*lm_head", "*gate"]` |
| `converters` | List[str] | [] | The list of quantization converters. `quantize.linear.mx` and `quantize.moe.mx` are supported |
| `enable_fsdp_low_precision_all_gather` | bool | `true` | Whether to enable low-precision communication |
| `fsdp_low_precision_all_gather_mode` | string | `on-demand` | FSDP low-precision all-gather mode. `on-demand` and `all` are supported |

### 2. Core Parameter Descriptions

#### ✅quant_recipe

The format of `quant_recipe` is as follows:

```python
<scaling_strategy>_<scaling_granularity>[-blocksize0-blocksize1-blocksize2]_<inputs_dtype>_<weight_dtype>_<grads_dtype>
```

| Field | Description |
|------|------|
| `scaling_strategy` | Scaling strategy, such as `dynamic` and `delayed` |
| `scaling_granularity` | Scaling granularity, such as `mx` (only supported), `per_tensor`, and `per_channel` |
| `blocksize0-blocksize1-blocksize2` | Optional, block size (only for block quantization) |
| `inputs_dtype`/`weight_dtype`/`grads_dtype` | Data types of inputs, weights, and gradients, such as `E4M3` and `E5M2` |

#### Predefined Recipe Example

- `mxfp8`: `dynamic_mx-1-1-32_E4M3_E4M3_E4M3`
→ Supports the MX quantization strategy and is suitable for most scenarios.

> ⚠️ Currently, only the `mxfp8` scaling strategy is supported. More strategies and recipes will be supported in the future.

#### ✅quant_format

The data type format of quantization, which specifies the numeric format after quantization, such as `e4m3`. It is case-insensitive.

#### ✅block_size

The number of elements quantized at a time. The default value is 32. This controls the grouping granularity during quantization. A smaller block size provides finer-grained quantization and may better preserve precision.

> ⚠️ Only a quantization block size of `32` is currently supported.

#### ✅quant_apply_modules

Specifies the layers or modules to which quantization is applied. Wildcards are supported.

**Example:**

```python
["model.layers.{*}"]                # Apply to all Transformer layers
["model.layers.0.self_attn"]        # Apply to the self-attention module of layer 0
```

#### ✅quant_ignored_modules

Specifies the list of submodules to which quantization is not applied. Wildcards are supported.

```python
["*lm_head"]       # Do not apply quantization to the lm_head module
["*gate"]          # Do not apply quantization to the gate part of the MLP
```

#### ✅converters

Specify the list of quantization converters to use. The following types are currently supported:

- `quantize.linear.mx`: Applies MX quantization to standard linear layers (such as FFN and Attention).
- `quantize.moe.mx`: MX quantization dedicated to the GMM in the expert modules of MoE models.
- `quantize.linear.w8a16`: QAT W8A16 linear layer quantization, where weights are fake-quantized to FP8 e4m3, activations remain in high precision, and STE is used in the backward pass. For details, see [QAT W8A16 Quantization Training Guide](qat_w8a16.md).

> 💡 In MoE models, `quantize.linear.mx` and `quantize.moe.mx` can be used simultaneously.
> 💡 When using `quantize.linear.w8a16`, set `quant_recipe` to `w8a16`. For details, see [QAT W8A16 Quantization Training Guide](qat_w8a16.md).

#### ✅enable_fsdp_low_precision_all_gather

Specifies whether to enable the low-precision all-gather mode of FSDP. When enabled, during forward/backward propagation, FSDP performs all-gather operations on parameters using low-precision weights (such as MXFP8), significantly reducing communication overhead and memory usage.

While low-precision training is enabled, this mode can be further enabled to maximize efficiency gains.

#### ✅fsdp_low_precision_all_gather_mode

Specifies the communication mode for low-precision all-gather:

| Mode | Description |
|------|------|
| `on-demand` | Communicates only the weights required for the current computation during forward or backward propagation |
| `all` | Communicates all weights during both forward and backward propagation |

> ⚠️ If recomputation is enabled, the system automatically handles it in a recomputation-aware manner in `on-demand` mode (the `_recompute_aware` logic is built into the code) to ensure computation consistency.
>
> ⚠️ In `all` mode, all-gathering all weights doubles the communication volume, and the communication time shows no significant change compared with bf16. In addition, because extra parameters such as scaling factors need to be communicated, memory usage increases slightly.

### 3. Example Script

Below is an example launch script that demonstrates how to configure quantization parameters and low-precision communication:

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

To enable low-precision training and communication, simply add the quantization-related parameters of `quantization_plan` under the `training` field on top of the existing training configuration file. For a configuration file example, see `examples/qwen3vl/qwen3vl_30B_config_v1_A5.yaml`.

## Notes

- ⚠️ Currently, low-precision training is supported only on 950-based machines. Machines such as 910B&C are not supported.
- ⚠️ Low-precision training may cause precision loss and degrade model performance. This is not a framework issue; use with caution.
