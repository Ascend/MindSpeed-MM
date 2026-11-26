# ChunkLoss

## Applicable Backends

FSDP2 + Mcore

## Background and Challenges

When training multimodal understanding models, the output dimension of `lm_head` (that is, the vocabulary size `vocab_size`) is usually much larger than the hidden dimension `hidden_size` of the model. The traditional loss computation requires explicitly constructing an intermediate logits tensor of shape `[bs, seq, vocab_size]`, which causes a significant memory peak, and the peak becomes more obvious as the vocabulary or the sequence grows. In addition, in dynamic shape scenarios, this operation easily causes large memory fragments, further aggravating the burden of memory management.

### Limitations of the Existing Approach (legacy)

`impl_type=legacy` (the default) reduces memory by "chunking the sequence dimension and computing gradients manually inside forward". Each chunk still materializes a logits tensor of shape `[chunk, vocab_size]`, and its computation and gradient backward rely on the three-layer call stack of `ChunkLoss`, `calculate_lm_loss`, and `fixed_cross_entropy`, with backward replaying the computation graph inside each chunk. When the vocabulary is extremely large (such as 248320 in Qwen3.5) or the sequence is very long, the logits of a single chunk still occupy considerable memory, and the multi-layer calls add further overhead.

## Solution

### Sequence-Dimension Chunking (legacy)

By chunking the sequence dimension, the loss computation is split into multiple consecutive sub-segments of length `sub_seq`. After the forward computation of each sub-segment is complete, the corresponding backward propagation is executed immediately, which avoids keeping the logits of the entire sequence at the same time. In this way, at any moment, only logits of length `sub_seq` need to be cached at most, significantly reducing the memory peak.

### CCE Fused Operator (impl_type: cce)

`impl_type=cce` follows the Cut Cross-Entropy idea. It performs tile-based streaming chunking along the **vocabulary dimension** together with an online softmax, and the forward pass never materializes the full `[N, V]` logits (it keeps only the activations required for backward recomputation), thereby reducing the memory peak and the multi-layer call overhead at the same time.

- **Two-level chunking**: the outer level splits the sequence into segments of `seq_chunk_size` (optional, reused from `chunk_size`, where `None` means a single segment without splitting); the inner level splits the vocabulary into tiles of size `vocab_tile_size` and processes them one by one.
- **Dual-stream pipelining**: the Cube stream performs the `hidden @ head` matrix multiplication while the Vector stream performs the tile-level online softmax and accumulation, overlapping the computation through a sliding window (NSLOT rotation) to improve hardware utilization.
- **Backward recomputation**: the required gradients are recomputed from the activations cached in forward, so there is no need to additionally store the full logits.
- **Lighter implementation**: the three-layer call stack of `ChunkLoss` + `calculate_lm_loss` + `fixed_cross_entropy` is replaced directly by the fused operator `chunk_loss_cce_fused`, which also runs normally in environments without Triton.

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
| `chunkloss_plan.chunk_size` | int | `1024` | Size of each chunk (number of tokens) for static chunking; effective only for `enable_chunk_loss`. With `impl_type=legacy`, `None` automatically takes effect as `1024`; with `impl_type=cce`, this field is reused as the outer sequence split size (`seq_chunk_size`), and `None` or `0` means a single chunk without splitting. |
| `chunkloss_plan.total_chunk_size` | int | `4096` | Upper limit of the total tokens per computation for dynamic chunking; effective only for `enable_dynamic_chunk_loss`, and the chunk size is derived automatically from the batch size. |
| `chunkloss_plan.impl_type` | str | `legacy` | Implementation type: `legacy` (the default) or `cce`. When set to `cce`, `vocab_tile_size` takes effect. |
| `chunkloss_plan.vocab_tile_size` | int | `4096` | Vocabulary tile size for CCE (effective only with `impl_type=cce`). |

For reference, see `examples/qwen3_5/qwen3_5_4B_config.yaml`.

To enable the CCE fused operator, set `impl_type: cce` in `chunkloss_plan`:

```yaml
features:
  enable_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    impl_type: cce
    vocab_tile_size: 4096
    chunk_size: 1024
```

Parameters related to `impl_type=cce`:

- `enable_chunk_loss`: CCE depends on this master switch, which must be `true`; otherwise configuration validation fails (see the normalization logic of `chunkloss_plan`).
- `impl_type`: the implementation type, which can be `legacy` (the default) or `cce`. Once set to `cce`, `vocab_tile_size` takes effect.
- `vocab_tile_size`: the vocabulary tile size of CCE; `4096` by default (effective only with `impl_type=cce`). `4096` is recommended to balance the number of tiles against the per-tile kernel overhead, or use `8192` to halve the number of tiles and reduce host-side event overhead (at the cost of doubling slot memory).
- `chunk_size`: under `impl_type=cce` this is reused as the outer sequence split size (`seq_chunk_size`), where `None` or `0` means a single chunk without splitting (effective only with `enable_chunk_loss`).

CCE introduces no separate launch script; simply enable it in the config loaded by the standard launch script, as in `examples/qwen3_5/finetune_qwen3_5_35B.sh`.

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

### Results

In multimodal understanding models, properly setting `chunk_size` significantly lowers the memory peak while keeping the loss curve unchanged. The gains become more pronounced once `impl_type=cce` is enabled.

- **Memory peak**: in large-vocabulary, long-sequence scenarios, CCE saves about 63% of memory compared with the Triton-based chunking path.
- **Per-step latency**: in large-vocabulary scenarios, CCE is about 12% faster than the Triton-based chunking path.
- **Numerical accuracy**: 7/7 unit tests pass, with `loss_rel < 1e-4`, a relative error of `grad_hidden` below `1e-3`, and a relative error of `grad_weight` below `5e-2`, all consistent with the baseline.

CCE currently supports only the default and per-token loss granularities. `per_sample_loss` is not yet supported (a `NotImplementedError` is raised).
