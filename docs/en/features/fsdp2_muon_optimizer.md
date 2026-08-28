# FSDP2 Muon Optimizer

Muon (Momentum Orthogonalized by Newton-Schulz) is a class of optimizers designed for matrix parameters. It orthogonalizes the update direction of two-dimensional weight matrices and can serve as an optimization alternative to AdamW.

The main advantage of Muon is that it leverages the matrix structure of hidden-layer weights in neural networks to impose an orthogonalization constraint on the momentum update direction, making the update direction of two-dimensional weight matrices closer to a well-conditioned spectral norm update. In some public experiments, Muon has demonstrated better sampling efficiency and computational efficiency, that is, achieving a comparable loss with less training time or fewer FLOPs. However, the actual benefit still depends on the model architecture, batch size, learning rate, and training stage, and needs to be validated against the specific task.

Public use cases include:

- [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) trains with Muon/MuonClip at the 1T MoE scale.
- The [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) technical report presents the Muon Optimizer training algorithm for DeepSeek-V4.
- [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/README_CN.md) is trained with the Muon optimizer, and Muon is recommended for continued training or LoRA fine-tuning.
- [NVIDIA NeMo-RL](https://docs.nvidia.com/nemo/rl/latest/guides/muon-optimizer.html) provides examples of using Muon in the Qwen3-235B-A22B SFT and Qwen2.5-7B DAPO scenarios.
  
The implementation in this repository partially references the [MoonshotAI/Moonlight example Muon](https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py) and [KellerJordan/Muon](https://github.com/KellerJordan/Muon/blob/master/muon.py), and adds DTensor shard aggregation and re-sharding adaptation for the new FSDP2 backend on top of their implementation approach.

## Update Process

Under the FSDP2 backend, the Muon optimizer first shards parameter groups based on parameter names and shapes:

- Two-dimensional matrix parameters whose names do not end with `.bias`, do not contain `embedding`, and do not contain `output_layer` are updated using Muon.
- Other parameters automatically fall back to the AdamW update logic.
- Existing configurations such as learning rate, weight decay, and `no decay` grouping are preserved.

The Muon update process is as follows:

1. Use SGD momentum to accumulate the gradient direction for Muon parameters.
2. Convert the update direction to bfloat16 and perform approximate orthogonalization through Newton-Schulz iterations.
3. Apply weight decay to weights, and then apply the orthogonalized update.

In FSDP2 scenarios, parameters may be sharded as DTensors. Before computing orthogonalized updates, Muon aggregates the update directions of sharded parameters into a replicated form. After computation is complete, it reshards the updates according to the original DTensor placements, ensuring that the optimizer updates remain consistent with the FSDP2 parameter layout.

## Usage

In the FSDP2 YAML configuration, set `training.optimizer` to `muon` to enable Muon.

```yaml
training:
  lr: 1.0e-5
  weight_decay: 0
  optimizer: muon
  matched_adamw_rms: 0.2
  muon_momentum: 0.95
  ns_steps: 5
```

## Parameter Details

- **`optimizer`**
  - Description: Selects the optimizer type.
  - Values: `adamw` or `muon`.

- **`matched_adamw_rms`**
  - Description: Controls how closely the Muon update magnitude matches the AdamW update RMS.
  - Default value: `0.2`.

- **`muon_momentum`**
  - Description: The momentum coefficient for Muon's internal SGD momentum.
  - Default value: `0.95`.

- **`ns_steps`**
  - Description: Number of Newton-Schulz orthogonalization iterations.
  - Default value: `5`.
  - Note: More steps yield more thorough orthogonalization but increase computational overhead.

- **`lr`**
  - Description: Base learning rate.
  - Note: Muon parameters adjust the update magnitude based on the base learning rate, `matched_adamw_rms`, and matrix shape.

- **`weight_decay`**
  - Description: Weight decay coefficient.

## Notes

1. Muon only applies to two-dimensional matrix parameters that meet the conditions; all other parameters automatically use the AdamW logic. No manual parameter splitting is required.
2. In FSDP2, sharded parameters are temporarily aggregated before Muon orthogonalization computation and reshared afterward. This process introduces additional communication and computational overhead.
3. `ns_steps` can be adjusted based on training stability and performance requirements. Smaller values can be used for fast validation runs; for formal training, it is recommended to verify the impact using loss curves and throughput metrics.
4. `matched_adamw_rms` affects the Muon update magnitude. When modifying the learning rate, it is recommended to monitor its impact on convergence simultaneously.
   