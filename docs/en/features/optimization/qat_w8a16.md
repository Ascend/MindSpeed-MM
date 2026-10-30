# MindSpeed MM FSDP2 Backend QAT W8A16 Quantization Training Guide

## Introduction

This guide describes how to train with the QAT W8A16 quantization capability of FSDPTurbo on the FSDP2 backend within the MindSpeed MM framework.

W8A16 quantizes weights to FP8 e4m3 (per-32-block MX shared exponent format) and keeps activations at high precision, using the straight-through estimator (STE) for backpropagation. Unlike MXFP8, W8A16 **does not touch** the FSDP low-precision all-gather communication path (it does not introduce the `PreQuantWeight`/`PostQuantWeight` tensor subclasses), so it composes transparently with FSDP2 sharding in MindSpeed MM without additional communication configuration.

### How It Works

The W8A16 converter is registered automatically by FSDPTurbo. When `ParallelApplier.apply_quantization_modules` in MindSpeed MM imports `fsdp_turbo.quantization.converter.model_converter.build_model_converter`, initialization of the FSDPTurbo `quantization` package automatically executes `w8a16_converter.py`, which registers the following converter in the global registry:

- `quantize.linear.w8a16`: Replaces `nn.Linear` in the model with `W8A16Linear`; weights undergo FP8 e4m3 per-32-block quantization through `w8a16_fake_quant`, activations remain at high precision, and STE is used in the backward pass.

You only need to set `quant_recipe` to `w8a16` and specify the converter name in `converters` to enable it. No additional plugin or bridging module is required.

## Prerequisites

W8A16 quantization depends on the FSDPTurbo package, which must be installed before use:

```bash
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
pip install -e .
```

> ⚠️ Quantization conversion is triggered only when the W8A16 converter is actually selected in the configuration.

## How to Use

### 1. Parameter Description

W8A16 quantization reuses the `training.quantization_plan` configuration. The related parameters are as follows:

| Parameter | Type  | Description |
|------|------|------|
| `quant_recipe` | str |  Set to `w8a16` in W8A16 scenarios, serving as the enabling flag for quantization |
| `quant_apply_modules` | List[str] | Layers or modules to which quantization is applied; wildcards are supported |
| `quant_ignored_modules` | List[str] | List of submodules to which quantization is not applied |
| `converters` | List[str]  | Quantization converters to use; configure `quantize.linear.w8a16` in W8A16 scenarios |

> 💡 `enable_fsdp_low_precision_all_gather` and `fsdp_low_precision_all_gather_mode` are MXFP8-specific parameters and are not used by W8A16, so no configuration is required.

### 2. Core Parameter Description

#### quant_recipe

Set it to `w8a16` to enable the quantization flow. When `quant_recipe` is non-empty, `apply_quantization_modules` in MindSpeed MM triggers `build_model_converter` to build and execute the converter.

> 💡 The value of `quant_recipe` itself is not parsed as a specific quantization format; the actual quantization behavior is determined by the converter names in the `converters` list.

#### quant_apply_modules

Specifies the layers or modules to which W8A16 quantization is applied; wildcards are supported. It typically matches the decoder layer prefix (containing `layers.\d+`), and all `nn.Linear` modules within the layer are quantized.

```yaml
quant_apply_modules:
  - "model.language_model.layers.{*}"
```

> 💡 Modules that are not under `layers.\d+`, such as `lm_head`/`embed_tokens`, are skipped automatically.

#### quant_ignored_modules

Specifies the list of submodules to which quantization is not applied; wildcards are supported. List specific submodules here if they need to be excluded. An empty value is used by default.

```yaml
quant_ignored_modules: []
```

#### converters

Specifies the W8A16 quantization converter to use. The following type is currently supported:

- `quantize.linear.w8a16`: Suitable for ordinary linear layers (such as FFN and Attention); replaces `nn.Linear` with `W8A16Linear`.

### 3. Enabling Steps

Add `quantization_plan` under the `training` field of the training configuration, set `quant_recipe` to `w8a16`, and specify `converters` to enable W8A16 quantization:

```yaml
training:
  quantization_plan:
    quant_recipe: w8a16              # Enable the quantization flow
    quant_apply_modules:
      - "model.language_model.layers.{*}"
    quant_ignored_modules: []
    converters:
      - "quantize.linear.w8a16"
```

### 4. Complete Example

The following is a complete configuration example that enables W8A16 linear layer quantization for the Qwen3.5-27B model:

```yaml
training:
  micro_batch_size: 2
  gradient_accumulation_steps: 1
  # ... other training parameters ...
  use_deter_comp: false
  # QAT W8A16 quantization configuration: weights are fake-quantized to FP8 e4m3 (per-32-block MX shared exponent),
  # activations remain at high precision, and the straight-through estimator (STE) is used in the backward pass.
  quantization_plan:
    quant_recipe: w8a16              # Enable the quantization flow
    quant_apply_modules:
      - "model.language_model.layers.{*}"
    quant_ignored_modules: []
    converters:
      - "quantize.linear.w8a16"
  plugin:
    - mindspeed_mm/fsdp/models/qwen3_5
    - mindspeed_mm/fsdp/data/datasets/huggingface
```

## Notes

- ⚠️ Make sure FSDPTurbo is installed before use (`pip install -e FSDPTurbo`); otherwise a `ModuleNotFoundError` is raised when the converter is instantiated.
- ⚠️ When `quant_recipe` is set to `w8a16`, `quantize.linear.w8a16` must be configured in `converters`; both are required.
- ⚠️ W8A16 is a fake-quant training path that does not modify the FSDP low-precision all-gather communication path, so `enable_fsdp_low_precision_all_gather`/`fsdp_low_precision_all_gather_mode` have no effect on it and no configuration is required.
- ⚠️ Features related to low-precision training are supported only on 950-based machines; machines such as 910B/C are not supported.
- ⚠️ Low-precision training may cause precision loss and degrade model performance; this is not a problem of the framework itself, and it is recommended to use it with caution.
