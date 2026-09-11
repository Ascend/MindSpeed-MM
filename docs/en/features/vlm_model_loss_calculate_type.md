# VLM Loss Calculation Types

## Problem Analysis

Vision-Language Models (VLMs) typically use cross-entropy loss as their training objective. However, an issue exists in the implementation of current mainstream model repositories (such as Hugging Face Transformers): when the global batch size is fixed, using different combinations of `micro_batch_size` and gradient accumulation steps (e.g., `micro_batch_size=32`, `grad_acc_steps=2` vs. `micro_batch_size=16`, `grad_acc_steps=4`) leads to significant differences in the loss curve and final values during model convergence, even with identical hyperparameters and training data. For related discussion, refer to [gradient](https://unsloth.ai/blog/gradient).

> Note: Transformers has fixed this issue in some models. However, in models like Qwen2.5-VL and Qwen3-VL, the problem persists due to incorrect parameter passing in related code or explicitly not computing `num_token_in_batch`.

## Solution

For VLMs, MindSpeed MM provides three Loss calculation methods. Assuming a training configuration of `micro_batch_size=2`, `grad_acc_steps=2`, using two cards for training (i.e., `DP=2`), the steps for each of the three loss computation methods are as follows:

### 1. Default Method

Consistent with the Transformers implementation. The calculation process is as follows:

- **Step 1**: Average the cross-entropy loss over valid tokens within a micro-batch.
- **Step 2**: Average over the gradient accumulation dimension.
- **Step 3**: Average over the data parallel (DP) domain.

![default](../../../sources/images/vlm_model_loss_calculate_type/default_en.png)

### 2. Calculate Per Sample Loss

The calculation process is as follows:

- **Step 1**: Average the cross-entropy loss over valid tokens within a single sample.
- **Step 2**: Average over the micro-batch dimension.
- **Step 3**: Average over the gradient accumulation dimension.
- **Step 4**: Average over the DP domain.

![sample_level](../../../sources/images/vlm_model_loss_calculate_type/sample_level_en.png)

### 3. Calculate Per Token Loss

The calculation process is as follows:

- Directly accumulate the cross-entropy loss over all valid tokens in the global batch.
- Divide the final result by the total number of valid tokens in the global batch.

![token-level](../../../sources/images/vlm_model_loss_calculate_type/token_level_en.png)

## How to Use

### Megatron Backend

For models using `pretrain_vlm.py` as the entry point, enable loss calculation as follows:

### 1. Default Calculation Method

In the model training script, **do not enable** any of the following parameters:

- `--calculate-per-sample-loss` (for per-sample loss)
- `--calculate-per-token-loss` (for per-token loss)

### 2. Calculate Per Sample Loss

Enable the following parameter in the model training script:

```shell
GPT_ARGS="
    ...
    --calculate-per-sample-loss \
"
```

### 3. Calculate Per Token Loss

Enable the following parameter in the model training script:

```shell
GPT_ARGS="
    ...
    --calculate-per-token-loss \
"
```

### FSDP2 Backend

The FSDP2 backend configures the loss calculation method via the `loss_type` field under `loss_cfg`. It supports the following three values (the configuration approach is the same for both backends):

- `default`: Default computation method.
- `per_sample_loss`: Computes loss at the sample level.
- `per_token_loss`: Computes loss at the token level.

#### Native FSDP2 (Recommended)

Set the loss_type in the `features.loss_cfg` section of the model YAML configuration file:

```yaml
features:
  loss_cfg:
    loss_type: default   # Options: default/per_sample_loss/per_token_loss
```

#### Megatron-Based FSDP2 (Transitional, to be Deprecated)

For models whose training entry point is `pretrain_transformers.py`, add the following field to model.json:

```json
"loss_cfg": {
    "loss_type": "default/per_sample_loss/per_token_loss"
}
```

### Notes

1. When using the Megatron backend, `--calculate-per-sample-loss` and `--calculate-per-token-loss` cannot be used simultaneously.
2. Choosing an inappropriate loss computation method may significantly impact downstream task evaluation. Users should select the appropriate method based on the distribution of samples in the training dataset. If the training dataset has an uneven sample distribution — some samples have very long responses while others have very short responses, or even just a single token — computing loss at the token level will give more weight to samples with more target tokens, introducing imbalance across samples and causing longer outputs to be trained more thoroughly.
