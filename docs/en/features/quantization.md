# MindSpeed MM FSDP2 Backend Low-Precision Training Guide

## Introduction

This guide helps users implement low-precision training (such as mxfp8) based on the FSDP2 backend within the MindSpeed MM framework, improving training efficiency and memory utilization. By configuring `QuantizationRecipe` and the low-precision all-gather mode, communication overhead and memory usage can be significantly reduced while maintaining model accuracy, making it suitable for large-model training scenarios.

## How to Use

### 1. Parameter Overview

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `recipe_name` | str | mxfp8 (required) | The quantization recipe name to use, which also serves as the identifier for enabling quantization |
| `apply_modules` | str | 'model.layers.{*}' | The layers or modules to which quantization is applied |
| `ignored_modules` | str | 'lm_head', 'gate' | The list of submodules to which quantization is not applied |
| `quant_converters` | str | 'quantize.linear.mx', 'quantize.moe.mx' | The list of quantization converters to use, which quantize the linear layers and the GMM in MoE respectively |
| `enable_fsdp_low_precision_all_gather` | bool | `True` | Whether to enable low-precision communication |
| `fsdp_low_precision_all_gather_mode` | str | 'on-demand' | FSDP low-precision all-gather, aggregating forward or backward weights on demand |

### 2. Core Parameter Descriptions

#### ✅recipe_name

The format of `recipe_name` is as follows:

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

- `mxfp8`: `dynamic_MX-1-1-32_E4M3_E4M3_E4M3`

→ Supports the MX quantization strategy and is suitable for most scenarios.

> ⚠️ Currently, only the `mxfp8` scaling strategy is supported. More strategies and recipes will be supported in the future.

#### ✅apply_modules

Specifies the layers or modules to which quantization is applied. Wildcards are supported.
**Example:**

```python
'model.layers.{*}'          # Apply to all Transformer layers.
'model.layers.0.self_attn' # Apply to the self-attention module of layer 0.
```

#### ✅ignored_modules

Specifies the list of submodules to which quantization is not applied. Wildcards are supported.

```python
'*q_proj'        # Do not apply quantization to all q_proj submodules.
'*gate'          # Do not apply quantization to the gate part in mlp.
```

#### ✅quant_converters

Specifies the quantization converters to use. The following types are currently supported:

- `quantize.linear.mx`: Applies MX quantization to standard linear layers (such as FFN and Attention).
- `quantize.moe.mx`: Specifically applies MX quantization to GMM in MoE model's expert modules.

> 💡 In MoE models, `quantize.linear.mx` and `quantize.moe.mx` can be used simultaneously.

#### ✅enable_fsdp_low_precision_all_gather

Specifies whether to enable the low-precision all-gather mode of FSDP. When enabled, during forward/backward propagation, FSDP performs all-gather operations on parameters using low-precision weights (such as mxfp8), significantly reducing communication overhead and memory usage.

While low-precision training is enabled, this mode can be further enabled to maximize efficiency gains.

#### ✅fsdp_low_precision_all_gather_mode

Specifies the communication mode for low-precision all-gather:

| Mode | Description |
|------|------|
| `on-demand` | Communicates only the weights required for the current computation during forward or backward propagation |
| `all` | Communicates all weights during both forward and backward propagation |

> ⚠️ If recomputation is enabled, the system automatically switches to the `all` mode to ensure computational consistency.
> ⚠️ In the `all` mode, communicating all weights via all-gather doubles the communication volume, and the communication time shows no significant improvement compared with bf16. In addition, memory usage increases slightly because essential parameters such as scaling factors must also be communicated.

### 3. Example Script

Below is an example launch script that demonstrates how to configure quantization parameters and low-precision communication:

```yaml
training:
  quantization_plan:
    recipe_name: mxfp8
    apply_modules: ['model.layers.{*}']
    ignored_modules: ['*lm_head', '*gate']
    quant_converters: ['quantize.linear.mx', 'quantize.moe.mx']
    enable_fsdp_low_precision_all_gather: True
    fsdp_low_precision_all_gather_mode: 'on-demand'
```

You only need to add the quantization-related parameters from `quantization_plan` under the `training` field on top of the original training configuration file to enable low-precision training and communication.

## Notes

- ⚠️ Currently, low-precision training is supported only on 950 machines. Machines such as 910B&C are not supported.
- ⚠️ Low-precision training may cause precision loss and degrade model performance. This is not a framework issue; use with caution.
  