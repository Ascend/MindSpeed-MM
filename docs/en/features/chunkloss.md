# ChunkLoss

## Background and Challenges

When training multimodal understanding models, the output dimension of `lm_head` (i.e., `vocab_size`) is typically much larger than the model's hidden dimension `hidden_size`. Traditional loss computation requires explicitly constructing an intermediate `logits` tensor of shape `[bs, seq, vocab_size]`, which leads to a significant memory usage peak. This peak becomes more pronounced as the vocabulary size or sequence length increases. Furthermore, in dynamic shape scenarios, this operation tends to cause large memory fragmentation, further exacerbating memory management overhead.

## Solution

By chunking the sequence dimension, the loss calculation is split into multiple consecutive chunks of length `sub_seq`. After completing the forward computation of each chunk, the corresponding backward propagation is executed immediately, thereby avoiding the need to retain `logits` for the entire sequence simultaneously. In this way, at any given moment, only `logits` of length `sub_seq` need to be cached, significantly reducing memory usage.

## How to Use

ChunkLoss currently only supports the FSDP2 backend, with two configuration options: native FSDP2 (recommended) and Megatron-based FSDP2 (transitional, to be phased out). ChunkLoss does not change the loss calculation method and can be used together with the default, per-sample, or per-token loss modes. For details on these three calculation methods [VLM Model Loss Calculation](vlm_model_loss_calculate_type.md).

### Native FSDP2 (Recommended)

Enable ChunkLoss in the `features` section of the model YAML configuration file:

```yaml
features:
  enable_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    chunk_size: 1024
```

`enable_chunk_loss` (static chunking) and `enable_dynamic_chunk_loss` (dynamic chunking) are mutually exclusive, corresponding to different chunk size parameters in `chunkloss_plan`. Do not use them together:

- `enable_chunk_loss`: Enables static chunking for ChunkLoss, splitting by fixed chunk size. Default is `false`. The chunk size is specified by `chunkloss_plan.chunk_size`.
- `enable_dynamic_chunk_loss`: Enables dynamic chunking for ChunkLoss, with adaptive chunking based on total size. Default is `false`. The total size is specified by `chunkloss_plan.total_chunk_size`.
- `chunkloss_plan`:
  - `apply_module`: The module to which ChunkLoss is applied. Default is `lm_head`.
  - `chunk_size`: Chunk size (in tokens) for static chunking. Default is `1024` (only effective when `enable_chunk_loss` is enabled).
  - `total_chunk_size`: Total token upper limit for a single computation in dynamic chunking. Default is `4096` (only effective when `enable_dynamic_chunk_loss` is enabled; chunk size is automatically derived from batch size).

For reference, see `examples/qwen3_5/qwen3_5_4B_config.yaml`.

### Megatron-Based FSDP2 (Transitional, to be Phased Out)

> Megatron-based FSDP2 is a transitional solution and will be gradually phased out. For new models, please prioritize native FSDP2.

In the understanding model configuration file `model.json` that supports ChunkLoss, configure it via the `loss_cfg` field. An example is as follows:

```json
"loss_cfg": {
    "compute_mode": "default",
    "chunk_size": 1024
}
```

- `compute_mode`:
  - Set to `"default"` to use the original loss calculation method.
  - Set to `"chunk"` to enable the static chunking mode, which divides the sequence into fixed-length chunks for loss computation.
  - Set to `"dynamic_chunk"` to enable the dynamic chunking mode, which adaptively adjusts the chunk size.
- `chunk_size`:
  - When `compute_mode` is set to `"chunk"`: Specifies the maximum length of each subsequence after sequence chunking (i.e., the number of tokens per chunk).
  - When `compute_mode` is set to `"dynamic_chunk"`: Specifies the maximum total computation size as (`subsequence length × batch_size`), used to constrain the total compute for dynamic chunking and prevent memory overflow.

By properly configuring `chunk_size`, memory usage can be effectively controlled while ensuring training correctness.

## Effectiveness

After enabling the ChunkLoss feature in a multimodal understanding model and setting an appropriate `chunk_size`, the peak memory usage can be significantly reduced while maintaining the same loss curve.
