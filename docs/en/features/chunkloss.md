# ChunkLoss

## Background and Challenges

When training multimodal understanding models, the output dimension of `lm_head` (i.e., the vocabulary size `vocab_size`) is typically much larger than the model's hidden dimension `hidden_size`. Traditional loss computation requires explicitly constructing an intermediate `logits` tensor of shape `[bs, seq, vocab_size]`, which leads to a significant memory usage peak. This peak becomes more pronounced as the vocabulary size or sequence length increases. Furthermore, in dynamic shape scenarios, this operation tends to cause large memory fragmentation, further exacerbating memory management overhead.

## Solution

By chunking the sequence dimension, the loss calculation is split into multiple consecutive chunks of length `sub_seq`. After completing the forward computation of each chunk, the corresponding backward propagation is executed immediately, thereby avoiding the need to retain `logits` for the entire sequence simultaneously. In this way, at any given moment, only `logits` of length `sub_seq` need to be cached, significantly reducing memory usage.

## How to Use

The currently supported loss calculation formulas for understanding models in MindSpeed MM are detailed in [this document](vlm_model_loss_calculate_type.md). The ChunkLoss feature currently supports the default mode, per-sample loss computation, and per-token loss computation.

For each understanding model configuration file (`model.json`) that supports chunkLoss, related settings can be configured via the `loss_cfg` field, as shown in the following example:

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
