# ChunkLoss

## Applicable Backends

FSDP2 + Mcore

## Background and Challenges

When training multimodal understanding models, the output dimension of `lm_head` (that is, the vocabulary size `vocab_size`) is usually much larger than the hidden dimension `hidden_size` of the model. The traditional loss computation requires explicitly constructing an intermediate logits tensor of shape `[bs, seq, vocab_size]`, which causes a significant memory peak, and the peak becomes more obvious as the vocabulary or the sequence grows. In addition, in dynamic shape scenarios, this operation easily causes large memory fragments, further aggravating the burden of memory management.

## Solution

By chunking the sequence dimension, the loss computation is split into multiple consecutive sub-segments of length `sub_seq`. After the forward computation of each sub-segment is complete, the corresponding backward propagation is executed immediately, which avoids keeping the logits of the entire sequence at the same time. In this way, at any moment, only logits of length `sub_seq` need to be cached at most, significantly reducing the memory peak.

## How to Use

ChunkLoss supports the FSDP2 family and the Mcore backend. The FSDP2 family has two configuration approaches: native FSDP2 (recommended) and Megatron-based FSDP2 (Megatron-FSDP2, transitional, to be phased out); the Mcore backend is currently provided by the Qwen3.5 model and is enabled through the `use_chunk_loss`/`chunk_size` fields in the model config (see `examples/mcore/qwen3_5/` for examples), and is not covered in this document. ChunkLoss does not change how the loss is computed and can be used with the default, per-sample loss, and per-token loss modes; for details on these three computation modes, see [VLM Model Loss Calculation](./vlm_model_loss_calculate_type.md).

### Native FSDP2 (Recommended)

Enable ChunkLoss in the `features` section of the model YAML configuration file:

```yaml
features:
  enable_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    chunk_size: 1024
```

### Parameter Description

| Configuration Item | Type | Default Value | Meaning |
| --- | --- | --- | --- |
| `enable_chunk_loss` | bool | `false` | Enables static chunking for ChunkLoss, splitting by a fixed chunk size; the chunk size is specified by `chunkloss_plan.chunk_size`. |
| `enable_dynamic_chunk_loss` | bool | `false` | Enables dynamic chunking for ChunkLoss, with adaptive chunking based on the total size; the total size is specified by `chunkloss_plan.total_chunk_size`. |
| `chunkloss_plan.apply_module` | string | `lm_head` | Module to which ChunkLoss is applied (the module must be an `nn.Linear`; otherwise an error is reported). |
| `chunkloss_plan.chunk_size` | int | `1024` | Size of each chunk (number of tokens) for static chunking; effective only for `enable_chunk_loss`. |
| `chunkloss_plan.total_chunk_size` | int | `4096` | Upper limit of the total tokens per computation for dynamic chunking; effective only for `enable_dynamic_chunk_loss`, and the chunk size is derived automatically from the batch size. |

For reference, see `examples/qwen3_5/qwen3_5_4B_config.yaml`.

### Megatron-Based FSDP2 (Transitional, to Be Phased Out)

> Megatron-based FSDP2 is a transitional solution and will be gradually phased out. For new models, prioritize native FSDP2.

In the `model.json` configuration file of understanding models that support ChunkLoss, set it through the `loss_cfg` field, as shown in the following example:

```json
"loss_cfg": {
    "compute_mode": "chunk",
    "chunk_size": 1024
}
```

- `compute_mode`:
  - Setting it to `"default"` means the original loss computation method is used;
  - Setting it to `"chunk"` enables the ChunkLoss static chunking feature, which splits the sequence into fixed-length chunks before computing the loss;
  - Setting it to `"dynamic_chunk"` enables the ChunkLoss dynamic chunking feature, which adaptively adjusts the chunk size.
- `chunk_size`:
  - When `compute_mode` is set to `"chunk"`: specifies the maximum length of each subsequence after the sequence is chunked (that is, the number of tokens each chunk contains);
  - When `compute_mode` is set to `"dynamic_chunk"`: specifies the maximum length of "each subsequence length × batch size (batch_size)" (used to constrain the total computation of dynamic chunking and prevent memory overflow).

By properly configuring `chunk_size`, memory usage can be effectively controlled while ensuring training correctness.

## Skipping Ignored Tokens (SFT Sparse Scenario)

In scenarios such as SFT, a large number of tokens in `shift_labels` are often set to `ignore_index` (prompt segments, padding, and so on). Their contribution to the loss and gradients is **always 0**, yet by default they are still projected to `lm_head` together with the remaining tokens, wasting compute and activation memory for nothing. ChunkLoss drops these tokens **before the `lm_head` projection**, thereby saving projection computation and activation memory in proportion to the masked ratio (the more masked tokens, the greater the gain). Both computation paths support this:

- ChunkLoss (`chunk_loss`): reassembles the complete `shift_labels`, filters out invalid tokens, and then re-chunks before computing;
- CCE (`chunk_loss_cce_fused`): filters out `ignore_index` rows before entering the vocab-tile streaming kernel.

**This optimization has no numerical impact on training and requires no additional switch**: it takes effect automatically when the conditions are met, and its loss and gradients are element-wise equivalent to the dense path without skipping (only floating-point noise at the level of fp32/bf16 summation order exists).

**Conditions for taking effect** (all must be met):

- The computation mode is `default` or `per_token_loss`; both are "the sum of CE over all valid tokens ÷ a scalar alpha (the total number of valid tokens)", so the total loss is independent of whether masked tokens are dropped, and they can therefore be dropped safely.
- **Some** (neither all nor none) tokens in the sequence are masked.
- **Any batch size works**: the scalar-sum reduction is independent of the batch structure, and the valid tokens of a multi-sample batch are flattened into a single dense sequence and then re-chunked, so the result remains element-wise consistent with the dense path.

**Cases that fall back to the dense path** (the behavior is exactly the same as when this optimization is not enabled, only without the additional gain):

- The computation mode is `per_sample_loss`: it divides **each sample by its own alpha** (the number of valid tokens of that sample × batch) before summing, so the contribution of a token depends on which sample it belongs to; dropping masked tokens, however, requires flattening the whole batch into a single sequence, which loses the sample boundaries and makes it impossible to divide by the respective alphas again, so this optimization is not applied (corresponding to `reduction != "sum"` or a non-scalar `alpha`);
- No token in the sequence is masked, or all tokens are masked.

## Notes

### Known Constraints

- Do not use `enable_chunk_loss` and `enable_dynamic_chunk_loss` together: the framework performs no conflict verification, and if both are enabled, the model side is enabled with static chunking while the chunk size for loss computation is derived from the dynamic chunking parameters, resulting in inconsistent behavior.
