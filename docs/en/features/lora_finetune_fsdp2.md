# LoRA Fine-Tuning on the FSDP2 Backend [Experimental Feature]

LoRA (Low-Rank Adaptation) is an efficient model fine-tuning method. By adding low-rank matrices to the weights, making the fine-tuning process more lightweight and saving computational resources and storage space.

> **Status**: [Experimental Feature]
> MindSpeed MM natively supports LoRA fine-tuning on the FSDP2 backend, without relying on the Megatron parallel framework. You can complete LoRA fine-tuning tasks using a simpler YAML configuration approach.

## Principle Overview

The core idea of LoRA is to decompose the model's parameter updates into a low-rank form. The specific steps are as follows:

- **Decomposing Weight Updates**: In traditional fine-tuning methods, the model's weights are updated directly. LoRA, by contrast, introduces two low-rank matrices $A$ and $B$ as a replacement for the weight matrix in each layer:

$$
W' = W + A \cdot B
$$

Here, $W'$ is the updated weight, $W$ is the original weight, and $A$ and $B$ are the low-rank matrices to be learned.

- **Reducing Parameter Count**: Because $A$ and $B$ have low ranks, the number of required parameters is significantly reduced, saving storage and computation costs.

## Enabling LoRA Fine-tuning

In the FSDP2 backend, LoRA fine-tuning is configured through the `training.lora` field in the YAML configuration file, without requiring additional command-line arguments in the launch script.

### Configuration Example

Add the `lora` configuration under the `training` field in the model's YAML configuration file (such as `examples/qwen3_5/qwen3_5_4B_config.yaml`):

```yaml
training:
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  lr: 1.0e-4
  train_iters: 100
  save_interval: 20
  save: ./save_path
  # ... other training parameters

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

### Parameter Description

| Parameter| Type | Default Value | Description                                                                                                                                                               |
| :--- | :--- | :--- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `enable` | bool | `false` | Whether to enable LoRA fine-tuning                                                                                                                                                     |
| `rank` | int | `8` | Dimension of the LoRA low-rank matrix. A lower rank value updates fewer parameters, reducing computation and memory consumption                                                                                                                     |
| `alpha` | int | `16` | Controls the influence ratio of LoRA weights on the original weights; a higher value means greater influence. Generally keep `α/r` at 2                                                                                                                     |
| `target_modules` | List[str] | `["q_proj", "k_proj", "v_proj"]` | Module names or wildcard patterns to which LoRA is added                                                                                                                                            |
| `dropout` | float | `0.0` | Dropout ratio of the LoRA layer, in the range `[0, 1)`                                                                                                                                 |
| `init_lora_weights` | bool \| str | `True` | Weight initialization method. `True`; `False`; or one of the following string values: `"gaussian"`, `"eva"`, `"olora"`, `"pissa"`, `"pissa_niter_[number of iters]"`, `"corda"`, `"loftq"`, `"orthogonal"` |
| `pretrained_lora_path` | str | `null` | Path to pretrained LoRA weights (optional), supporting `.safetensors` and `.pt/.bin` formats                                                                                                              |

### `target_modules` Configuration

`target_modules` supports two matching modes:

- **Exact match**: Directly specify the module name. For example, `"q_proj"` matches all modules ending with `q_proj`.
- **Wildcard match**: Use `{*}` as a wildcard. For example, `"model.language_model.layers.{*}.self_attn.q_proj"` matches all layers such as `layers.0` and `layers.1`.

Taking the Qwen3.5 model as an example, a common `target_modules` configuration is:

**LoRA fine-tuning only on Attention modules**:

```yaml
target_modules:
  - "model.language_model.layers.{*}.self_attn.q_proj"
  - "model.language_model.layers.{*}.self_attn.k_proj"
  - "model.language_model.layers.{*}.self_attn.v_proj"
  - "model.language_model.layers.{*}.self_attn.o_proj"
```

**LoRA fine-tuning only on MLP modules**:

```yaml
target_modules:
  - "model.language_model.layers.{*}.mlp.gate_proj"
  - "model.language_model.layers.{*}.mlp.up_proj"
  - "model.language_model.layers.{*}.mlp.down_proj"
```

**LoRA fine-tuning on both Attention and MLP modules**:

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

## Loading Pretrained LoRA Weights

To load pretrained LoRA weights for resumable training, configure the `pretrained_lora_path` parameter:

```yaml
training:
  lora:
    enable: true
    pretrained_lora_path: ./save_path/iter_xxx  # Replace with the LoRA weight save path.
```

## Weight Saving

### Saving Only LoRA Weights

During training, only the LoRA adapter weights are saved in the safetensors format. The saved file structure is as follows:

```bash
save_path/
├── lora_adapter.safetensors
└── ...
```

## Starting Training

After configuration is complete, use the same launch script as for full fine-tuning:

```shell
bash examples/qwen3_5/finetune_qwen3_5_xxB.sh
```

Once training starts, a LoRA configuration summary is automatically printed, including information such as the number of matched modules and the number of trainable parameters.

## Merging LoRA Weights into Hugging Face Weights

```bash
cd checkpoint/common
python merge_lora_safetensors_to_base.py \
    --base_hf_dir ./Qwen3.5-27B \
    --lora_safetensors ./save_path/lora_adapter_iteration_10.safetensors \
    --save_merged_hf_dir ./merged_qwen3_5_27B_lora
```

## Resuming Training from Checkpoint for LoRA

When resuming training from checkpoint, the `load` path in the YAML configuration file must point to the checkpoint path saved by the previous training run. The previous training run must have `no_save_optim` and `no_save_rng` set to `false.` To resume training, set `no_load_optim` and `no_load_rng` to `false` to restore the optimizer state. After resume training is complete, use the weight conversion script to merge the LoRA weights into Hugging Face weights.

## Notes

- **Dependency installation**: FSDP2 LoRA fine-tuning depends on the `peft` library. Ensure it is installed by running `pip install peft`.
- **Frozen modules**: After LoRA fine-tuning is enabled, the base model parameters are automatically frozen, and only the LoRA adapter parameters participate in training.
- **Precision handling**: LoRA parameters are automatically converted to `float32` precision for training to ensure training stability.
- **Weight validation**: When training starts, LoRA weights are automatically validated to check for NaN or Inf values.
- **Distributed training**: In the FSDP2 distributed training environment, LoRA weight saving automatically handles DTensor sharding without additional configuration.
- **Differences from the Megatron backend**: The FSDP2 backend uses YAML configuration instead of command-line parameters (such as `--lora-r`, `--lora-alpha`, etc.)

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
