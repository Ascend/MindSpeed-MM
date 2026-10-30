# LoRA Fine-Tuning on the FSDP2 Backend

Low-Rank Adaptation (LoRA) is an efficient model fine-tuning method. By adding low-rank matrices to the weights, making the fine-tuning process more lightweight and saving computational resources and storage space.

MindSpeed MM natively supports LoRA fine-tuning on the FSDP2 backend. It does not depend on the Megatron parallel framework, and LoRA fine-tuning tasks can be completed with a simpler YAML configuration.

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

Add the `lora` configuration under the `training` field of the model YAML configuration file (such as `examples/qwen3_5/qwen3_5_35B_config.yaml`):

```yaml
training:
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  lr: 1.0e-4
  train_iters: 100
  save_interval: 20
  save: ./save_path
  # ... Other training parameters

  lora:
    enable: true
    rank: 8
    alpha: 16
    # all-linear: automatically expands all nn.Linear layers in the model (recommended; no need to write module names manually)
    # Use model.freeze to exclude components (see the target_modules configuration notes below)
    target_modules: all-linear
    dropout: 0.0
    lora_save_only: false
    pretrained_lora_path: null
```

### Parameter Description

| Parameter| Type | Default Value | Description                                                                                                                                                               |
| :--- | :--- | :--- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `enable` | bool | `false` | Whether to enable LoRA fine-tuning                                                                                                                                                     |
| `rank` | int | `8` | Dimension of the LoRA low-rank matrix. A lower rank value updates fewer parameters, reducing computation and memory consumption                                                                                                                     |
| `alpha` | int | `16` | Controls the influence ratio of LoRA weights on the original weights; a higher value means greater influence. Generally keep `α/r` at 2                                                                                                                     |
| `target_modules` | str \| List[str] | `["q_proj", "k_proj", "v_proj"]` | Names of the modules to which LoRA is added, or a wildcard pattern, or the special keyword `all-linear` |
| `dropout` | float | `0.0` | Dropout ratio of the LoRA layer, in the range `[0, 1)`                                                                                                                                 |
| `lora_save_only` | bool | `false` | When `true`, only the standalone LoRA safetensors and the adapter configuration file are exported; when `false`, the full model weights are saved according to the save type: for the DCP format, the base weights and the LoRA weights are saved, and for the HF format, the weights fused from the base weights and the LoRA weights are saved. |
| `pretrained_lora_path` | str | `null` | Path to pretrained LoRA weights (optional), supporting `.safetensors` and `.pt/.bin` formats                                                                                                              |
| `disable_peft_moe_conversion` | bool | `true` | Disables the automatic redirection of `target_modules→target_parameters` performed by PEFT for `gate_proj`/`up_proj`/`down_proj` of MoE models, so that LoRA is applied to the `nn.Linear` of `shared_expert` instead of the `nn.Parameter` of the routed experts. Only relevant to MoE models |

### `target_modules` Configuration

`target_modules` supports three matching modes:

#### 1. The `all-linear` Keyword (Recommended)

Set `target_modules` to `all-linear` (either as a string or as a list element), and the framework automatically scans all `nn.Linear` layers in the model and injects LoRA, without writing module names manually.

```yaml
lora:
  target_modules: all-linear
```

**Excluding multimodal components**: `all-linear` matches all `nn.Linear` layers by default, including ViT and aligner. If you only want to inject LoRA into the language model, use `model.freeze` to freeze components; modules covered by `model.freeze` do not get LoRA injected:

```yaml
model:
  freeze:
    - model.visual        # Exclude the entire vision tower (including merger/aligner); LoRA is only applied to language_model
    # - model.visual.blocks   # Exclude only the vit blocks and keep LoRA on aligner(merger)
lora:
  target_modules: all-linear
```

> MoE models: `all-linear` matches the `nn.Linear` layers of `shared_expert` (`gate_proj`/`up_proj`/`down_proj`). With `disable_peft_moe_conversion=true`, PEFT is prevented from redirecting these layers to the routed expert layers.

#### 2. Exact Matching

Specify module names directly. For example, `"q_proj"` matches all modules ending with `q_proj`:

```yaml
target_modules:
  - "q_proj"
  - "k_proj"
```

#### 3. Wildcard Matching

Use `{*}` as a wildcard. For example, `"model.language_model.layers.{*}.self_attn.q_proj"` matches all layers such as `layers.0` and `layers.1`, which is suitable for scenarios that require precise control over the injection scope:

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
  - "model.language_model.layers.{*}.mlp.shared_expert.gate_proj"
  - "model.language_model.layers.{*}.mlp.shared_expert.up_proj"
  - "model.language_model.layers.{*}.mlp.shared_expert.down_proj"
```

## Loading Pretrained LoRA Weights

To load pretrained LoRA weights for resumable training, configure the `pretrained_lora_path` parameter:

```yaml
training:
  lora:
    enable: true
    pretrained_lora_path: ./save_path/iter_00000xx/lora_adapter_iteration_xx.safetensors
```

## Weight Saving

Select the save scope through `training.lora.lora_save_only`; the default value is `false`. After weight saving is enabled, there are the following three save modes:

| Save Mode | Key Configuration | Saved Content | Use Cases |
| :--- | :--- | :--- | :--- |
| Save LoRA only | `lora_save_only: true` | Save the LoRA weight safetensors and `adapter_config.json` in the weight save directory; the base weights and training states such as optimizer and RNG are not saved | Distributing or reusing the adapter, and offline fusion with the base model later |
| Save a DCP checkpoint | `lora_save_only: false`, `save_format: dcp` | Save the unfused base weights and LoRA weights; states such as optimizer and RNG are controlled by the existing save parameters | Checkpoint resumption |
| Save an HF checkpoint | `lora_save_only: false`, `save_format: hf` | Fuse according to `base + (alpha / rank) * B @ A` and save standard HF weights, without keeping separate LoRA parameters | Inference, deployment, or releasing the complete model |

## Starting Training

After configuration is complete, use the same launch script as for full fine-tuning:

```shell
bash examples/qwen3_5/finetune_qwen3_5_xxB.sh
```

Once training starts, a LoRA configuration summary is automatically printed, including information such as the number of matched modules and the number of trainable parameters.

## Merging LoRA Weights into HuggingFace Weights

After the `lora_save_only=true` parameter is enabled, only the LoRA weights are saved. You can merge them with the original HF weights offline using the following script:

```bash
cd checkpoint/common
python merge_lora_safetensors_to_base.py \
    --base_hf_dir ./Qwen3.5-27B \
    --lora_safetensors ./save_path/iter_00000xx/lora_adapter_iteration_xx.safetensors \
    --save_merged_hf_dir ./merged_qwen3_5_27B_lora
```

## Resuming Training from Checkpoint for LoRA

Complete checkpoint resumption requires setting `save_format=dcp` and `lora.lora_save_only=false` when saving the last time, and setting `no_save_optim` and `no_save_rng` to `false`. When restoring, point `load` to the checkpoint root directory that contains `latest_checkpointed_iteration.txt`, and set `no_load_optim` and `no_load_rng` to `false`. The DCP weights contain the complete base weights and LoRA weights, so there is no need to set `pretrained_lora_path` again.

## Models Supported by LoRA Fine-Tuning

qwen3vl, qwen3.5, qwen3.6, qwen3.8, qwen3omni

## Notes

- **Dependency installation**: FSDP2 LoRA fine-tuning depends on the `peft` library. Ensure it is installed by running `pip install peft`.
- **Frozen modules**: After LoRA fine-tuning is enabled, the base model parameters are automatically frozen, and only the LoRA adapter parameters participate in training.
- **Precision handling**: LoRA parameters are automatically converted to `float32` precision for training to ensure training stability.
- **Weight validation**: When training starts, LoRA weights are automatically validated to check for NaN or Inf values.
- **Distributed training**: In the FSDP2 distributed training environment, LoRA weight saving automatically handles DTensor sharding without additional configuration.
- **Differences from the Megatron backend**: The FSDP2 backend uses YAML configuration instead of command-line parameters (such as `--lora-r`, `--lora-alpha`, etc.)
- **Fine-tuning of MoE routed expert parameters**: the routed expert parameters of MoE models are of the `nn.Parameter` type, and LoRA fine-tuning is currently not supported for them. This feature is under development.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
