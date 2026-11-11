# Parameter LR/WD Tuning (Parameter-Level Learning Rate and Weight Decay Tuning)

## Applicable Backends

Mcore

## Problem Analysis

Multimodal large models involve the fusion of multiple modalities. For example, a VLM model includes various modules such as a Visual Encoder and a Language Decoder. Different modules (e.g., visual feature extraction layers, language attention layers, bias/normalization layers) exhibit varying sensitivities to Learning Rate (LR) and Weight Decay (WD). A globally uniform LR/WD configuration cannot accommodate the differentiated optimization needs of each module, potentially leading to underfitting or overfitting in some modules, which affects the model's multimodal alignment and final performance.

## Solution

A keyword-driven LR scaling and WD exclusion feature for parameter names is introduced:

1. Keyword matching on parameter names is supported to exclude specified parameters from weight decay, avoiding the optimization deviation of specific parameters (such as norm and bias) caused by WD;
2. Supports applying a custom LR multiplier to specified parameters (e.g., the vision module of a VLM) by matching parameter names via keywords, enabling fine-grained parameter-level learning rate control.
3. All matching logic uses case-insensitive string matching.

## How to Enable

This feature is disabled by default. To enable it, add the following arguments to the model launch training script as need. Example:

```shell
GPT_ARGS="
    ...
    --weight-decay-exclude-modules norm bias \
    --lr-scale-modules vision \
    --lr-mult 0.5 \
"
```

### Parameter Description

| Parameter | Description |
|------|------|
| `--weight-decay-exclude-modules` | List of strings that excludes the weight decay of specified parameters by parameter name keywords (multiple keywords are supported) |
| `--lr-scale-modules` | List of strings that specifies the parameters whose learning rate is scaled, by parameter name keywords (multiple keywords are supported) |
| `--lr-mult` | Learning rate scaling multiplier, used together with `--lr-scale-modules` (for example, 0.5) |

## Notes

### Known Constraints

- Keyword matching is case-insensitive substring containment matching (not exact matching and not a regular expression); an overly broad keyword also affects other parameters whose names contain that substring.
- `--lr-mult` takes effect only when `--lr-scale-modules` matches parameters; setting it alone has no effect on any parameter.
