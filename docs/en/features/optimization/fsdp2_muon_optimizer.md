# FSDP2 Muon Optimizer

## Applicable Backends

FSDP2 + Mcore

## Background and Challenges

Muon (Momentum Orthogonalized by Newton-Schulz) is a class of optimizers designed for matrix parameters. It orthogonalizes the update direction of two-dimensional weight matrices and can serve as an optimization choice in addition to AdamW.

The main advantage of Muon is that it leverages the matrix structure of hidden-layer weights in neural networks to impose an orthogonalization constraint on the momentum update direction, making the update direction of two-dimensional weight matrices closer to a well-conditioned spectral norm update. In some public experiments, Muon has demonstrated better sample efficiency and computational efficiency, that is, achieving a comparable loss with less training time or fewer FLOPs. However, the actual benefit still depends on the model architecture, batch size, learning rate, and training stage, and needs to be validated against the specific task.

Public use cases include:

- [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) trains with Muon/MuonClip at the 1T MoE scale;
- [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/README_CN.md) is trained with the Muon optimizer, and Muon is recommended for continued training or LoRA fine-tuning;
- [NVIDIA NeMo-RL](https://docs.nvidia.com/nemo/rl/latest/guides/muon-optimizer.html) provides examples of using Muon in the Qwen3-235B-A22B SFT and Qwen2.5-7B DAPO scenarios.

The implementation in this repository partially references the [MoonshotAI/Moonlight example Muon](https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py) and [KellerJordan/Muon](https://github.com/KellerJordan/Muon/blob/master/muon.py), and adds DTensor shard aggregation and re-sharding adaptation for the new FSDP2 backend on top of their implementation approach.

## Solution

Under the FSDP2 backend, the Muon optimizer first splits parameter groups based on parameter names and shapes:

- Two-dimensional matrix parameters whose names do not end with `.bias` and do not match the Muon fallback rules are updated using Muon;
- Parameters whose names contain `embedding`, `embed_tokens`, `output_layer`, or `lm_head` fall back to AdamW;
- All other parameters automatically fall back to the AdamW update logic;
- Existing configurations such as learning rate, weight decay, and no-decay grouping are preserved.

The Muon update process is as follows:

1. Use SGD momentum to accumulate the gradient direction for Muon parameters;
2. Convert the update direction to bfloat16 and perform approximate orthogonalization through Newton-Schulz iterations;
3. Apply weight decay to the weights, and then apply the orthogonalized update.

In FSDP2 scenarios, parameters may be DTensor shards. Before computing orthogonalized updates, Muon aggregates the update directions of sharded parameters into a replicated form; after the computation is complete, it re-shards them according to the original DTensor placements, ensuring that the optimizer updates remain consistent with the FSDP2 parameter layout.

## Usage

### FSDP2 Backend

To enable it, set `optimizer` to `muon` in the `training` section of the FSDP2 YAML configuration, and append the other Muon parameters (`matched_adamw_rms`, `muon_momentum`, `ns_steps`) as needed:

```yaml
training:
  lr: 1.0e-5
  weight_decay: 0
  optimizer: muon        # Default value is adamw
  matched_adamw_rms: 0.2
  muon_momentum: 0.95
  ns_steps: 5
  # Optional: append parameter name keywords that need to fall back to AdamW in the service model, beyond the built-in fallback rules.
  # For example, custom_head makes custom_head.weight use the AdamW fallback.
  muon_fallback_param_keywords:
    - custom_head
    - router
```

#### Parameter Description

| Parameter | Type | Default Value | Description |
| --- | --- | --- | --- |
| `training.optimizer` | string | `adamw` | Optimizer type. Valid values are `adamw`, `adamw_swap`, and `muon`. |
| `training.matched_adamw_rms` | float | `0.2` | Controls how closely the magnitude of the Muon update matches the RMS of the AdamW update. |
| `training.muon_momentum` | float | `0.95` | Momentum coefficient of the SGD momentum inside Muon. |
| `training.ns_steps` | int | `5` | Number of Newton-Schulz orthogonalization iterations. A larger number makes the orthogonalization more thorough, but the overhead also increases. |
| `training.muon_fallback_param_keywords` | list[str] | Built-in `embedding`, `embed_tokens`, `output_layer`, `lm_head` | Parameter name keywords to be additionally handled by the AdamW fallback logic. They are appended on top of the built-in rules. Qwen/HuggingFace-style `embed_tokens` and `lm_head` fall back by default, so no separate configuration is required for Qwen3. |
| `training.lr` | float | `5e-5` | Base learning rate. For Muon parameters, the update magnitude is adjusted on top of the base learning rate based on `matched_adamw_rms` and the matrix shape. |
| `training.weight_decay` | float | `0.0` | Weight decay coefficient. |

### Mcore Backend

Enable it with `--optimizer muon`, and set `"muon_optimizer": true` in the `patch` section of model.json (otherwise the native Megatron optimizer factory reports `muon optimizer is not supported.`).

#### Parameter Description

| Parameter | Type | Default Value | Description |
| --- | --- | --- | --- |
| `muon_optimizer` (model.json `patch` section) | bool | `false` | Wiring switch for enabling Muon on the Mcore side. |
| `--matched-adamw-rms` | float | `0.2` | Same meaning as `matched_adamw_rms` on the FSDP2 side. |
| `--muon-momentum` | float | `0.95` | Same meaning as `muon_momentum` on the FSDP2 side. |
| `--ns-steps` | int | `5` | Same meaning as `ns_steps` on the FSDP2 side. |

#### Example Script

See `examples/wan2.2/5B/t2v/pretrain.sh` (see the `muon_optimizer` field in `examples/wan2.2/5B/t2v/pretrain_model.json`, which defaults to `false`).

## Notes

### Known Constraints

1. Muon applies only to two-dimensional matrix parameters that meet the conditions; all other parameters automatically use the AdamW fallback logic, so parameters do not need to be split manually.
2. FSDP2 sharded parameters are temporarily aggregated before the Muon orthogonalization computation and re-sharded afterwards; this process incurs additional communication and computation overhead.
3. `ns_steps` can be adjusted based on training stability and performance requirements. A smaller value can be used for a quick trial run, and for formal training it is recommended to validate it against the loss curve and throughput.
4. `matched_adamw_rms` affects the magnitude of the Muon update; when changing the learning rate, it is recommended to also observe how this parameter affects convergence.
5. On the Mcore side, it cannot be used together with `--use-distributed-optimizer` (the framework performs validation, and enabling both causes an error).
