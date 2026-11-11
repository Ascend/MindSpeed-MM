# Act Stash (Activation Stash)

> Act Stash is a tenant on top of the [Swap Core](swap_core.md) foundation, and its interaction with the foundation is limited to the `put`/`pop` interfaces; the physical behavior of swap-in and swap-out is determined by the shared `swap_plan` configuration. For the behavior and conventions of the foundation, see [Swap Core](swap_core.md). This document describes only the configuration and semantics of Act Stash itself.

## Applicable Backends

FSDP2

## Background and Challenges

In long-sequence and large-model training, activation memory is one of the main bottlenecks. Even with recomputation enabled, tensors saved by autograd, such as checkpoint boundary inputs, still reside in on-chip memory until the backward pass. As the number of layers and the sequence length grow, this footprint becomes significant and weakens the memory benefit of recomputation.

## Solution

Act Stash is the `saved_tensors_hooks` implementation of activation offload (`activation_offload_plan.impl: "stash"`): pack is `put` and unpack is `pop`, so activations are swapped out when they are saved in the forward pass and fetched back on demand in the backward pass, and the tenant never touches any physical detail of swapping. It is another form of the existing per-layer activation offload in the repository (the legacy implementation, that is, [Async Activation Offload](./async_activation_offload.md)), and the two are mutually exclusive. A fetch miss always raises an error immediately: autograd guarantees that every packed tensor is unpacked, so a miss is always an anomaly and never produces incorrect values. The fetch order in the backward pass is approximately the reverse of the swap-out order in the forward pass, which naturally matches the default policy of the base, so tuning is usually unnecessary.

### Scope and Eligibility Filtering

- **Scope**: when enabled together with non-reentrant checkpoint, the scope equals the checkpoint boundary inputs plus the tensors saved by autograd outside the checkpoint, which is complementary to Op Replay (operator outputs inside the checkpoint) without overlap. **When recomputation is not enabled**, the scope becomes all tensors saved by autograd in the wrapped modules, and the swap-out traffic and page-locked memory usage increase accordingly.
- **Eligibility filtering**: only device (NPU/CUDA) tensors, non-parameter tensors (and their views), tensors with a strided layout, and tensors with non-empty storage are swapped out; CPU tensors, model parameters, and tensors with non-strided layouts such as sparse tensors are passed through unchanged and do not enter the cache.
- **kwargs blind spot**: non-reentrant checkpoint only passes top-level positional arguments through `save_for_backward` (the packing hook channel); tensors in named kwargs and containers are held by reference by the checkpoint frame and reside in HBM until the backward pass. You can use `recompute_plan.flatten_inputs: true` (disabled by default) to flatten the complete `(args, kwargs)` input tree into deduplicated positional-argument leaves that are included in the scope: keyword-only arguments, `**kwargs` contents, and tensors inside containers (list/dict) are all covered; the checkpoint rebuilds the original call structure internally and then executes the forward pass, so the semantics are preserved (objects aliased across slots share the same leaf).

## Use Cases

- **Enabled**: when recomputation is already enabled but activations saved by autograd, such as checkpoint boundary inputs, still reside in HBM (the footprint is significant as the number of layers and the sequence length grow); or when recomputation is not enabled and all tensors saved by autograd in the wrapped modules need to be swapped out (the swap-out traffic and page-locked memory usage increase accordingly, so `capacity_mb` must be planned accordingly). For the interaction with the model-level `skip_*_recompute` switches, see "Known Constraints".
- **Not enabled**: when `capacity_mb` is not less than the total swap-out volume there is no memory benefit (for the criterion, see "Known Constraints"); also for short sequences and scenarios with low activation pressure.

## How to Use

Enabling conditions: `enable_activation_offload: true`, `activation_offload_plan.apply_modules` is explicitly configured (defaults to `None`), `impl` is `"stash"`, and  `swap_plan` can be omitted (default values are used).

Minimal configuration example:

```yaml
features:
  enable_activation_offload: true
  activation_offload_plan:
    impl: stash
    apply_modules:
      - model.language_model.layers.{*}
  swap_plan:
    capacity_mb: 1024   # For the memory benefit criterion, see "Known Constraints"
```

### Parameter Description

| Configuration Item | Type | Default | Meaning and Value Constraints |
| --- | --- | --- | --- |
| `enable_activation_offload` | bool | false | Whether to enable activation offload. |
| `activation_offload_plan.apply_modules` | List[str] / null | None | Modules to which activation offload is applied (pattern matching, such as `model.language_model.layers.{*}`). Defaults to `None` and must be configured explicitly. |
| `activation_offload_plan.impl` | string | "legacy" | Implementation; choose one of two options: `"legacy"` = per-layer hidden_states swap-out (self-managed streams and prefetch heuristics, that is, the path of [Async Activation Offload](./async_activation_offload.md)); `"stash"` = Act Stash (the `saved_tensors_hooks` tenant, where all physical behavior is determined by `swap_plan`). An invalid value fails the startup verification with an error. |
| `recompute_plan.flatten_inputs` | bool | false | Flattens the complete `(args, kwargs)` input tree at the checkpoint boundary into deduplicated positional-argument leaves, so that kwargs and tensors inside containers are included in the Act Stash scope; takes effect only for non-reentrant. For the semantics, see [Scope and Eligibility Filtering](#scope-and-eligibility-filtering). |

### How to Set `capacity_mb`

Swap-out occurs in per-layer chunks, where **chunk = the swap-out volume of a single layer**, which is approximately the size of the checkpoint boundary inputs of each layer when recomputation is enabled: `micro_batch × seq × hidden × bytes` (2 for BF16). The following uses the criterion for a single Act Stash tenant; when enabled together with Op Replay, both share the same `swap_plan` budget, so plan the capacity according to the sum of their requirements (the upper bound is the sum of their total swap-out volumes). See the tenant list and selection guidance in [Swap Core](swap_core.md). The valid range and the recommended criterion are as follows:

- **Lower bound = 1 chunk**: below 1 chunk, each put must wait for its own just-issued copy to complete before it can proceed, so swap-out and computation are fully serialized (each step pays about `number of layers × single-chunk copy time` for nothing), which is worse than not enabling the feature.
- **Full hiding condition**: `capacity_mb ≥ chunk × ⌈single-chunk copy time/single-layer forward time⌉`, rounded **up**, because the capacity takes effect at chunk granularity: a ratio of 1.2 requires 2 chunks, and with only 1.2 chunks configured the window is counted as 1 chunk and waiting still occurs. When the copy is faster than the computation of one layer (common), **1 chunk is enough for full hiding**, and increasing it further does not improve the step time.
- **Upper bound = total swap-out volume** (chunk × number of covered layers); once reached, nothing is released within a step and there is no memory benefit (see "Known Constraints"). **Memory benefit ≈ total swap-out volume − `capacity_mb`**: between the hiding lower bound and the total volume, each additional 1 MB of `capacity_mb` reduces the benefit by 1 MB, so simply take the minimum value that "just hides" the swap-out.

**Example** (Qwen3.5 4B, hidden=2560, BF16, seq 16K, micro_batch=1): chunk ≈ 16384 × 2560 × 2B ≈ 80 MiB; with recomputation enabled for all 32 layers, the total swap-out volume is ≈ 2.5GiB. If the measured single-chunk copy time is less than the single-layer forward time (common), setting `capacity_mb` to 2 chunks (160 MiB, leaving some headroom for jitter) is enough for full hiding, and the memory benefit is ≈ 2.3 GiB. It is recommended to measure the copy bandwidth on the target machine to confirm.

**Tuning signal**: if the increase in step time after enabling is close to 0, the swap-out is already fully hidden; if there is obvious degradation, first check whether `capacity_mb` is lower than 1 chunk.

## Notes

### Known Constraints

- **Interaction with skip recomputation operators**: the outputs of the model-level switches `skip_flash_attn_recompute`/`skip_gdn_recompute` (for models such as Qwen3.5) are still released by the legacy offload protocol (not through Swap Core), and Act Stash does not cover this group of operators. `skip_*_recompute: false` is a recommended pairing, so that Op Replay covers the same group of operators through Swap Core.
- **Combining with recomputation requires `use_reentrant: false`**: the coverage semantics are defined by non-reentrant checkpoint, and the reentrant combination is not in the verified scope.
- **The memory benefit depends on the hard upper limit**: to save memory, `capacity_mb` must be less than the total swap-out volume (for the mechanism of the physical peak of the cache pool and the release timing, see the configuration mechanism illustration in [Swap Core](swap_core.md)). `keep_rate` only adjusts the copy timing and the fetch hit rate and is unrelated to the memory benefit.
- **In-place modification is prohibited**: tensors swapped out through Act Stash (including checkpoint boundary inputs and, after `flatten_inputs` is enabled, the flattened `kwargs` and tensors inside containers) must not be modified in place before they are fetched back in the backward pass; violating this does not raise an error but silently produces incorrect gradients. For the full contract, see the usage conventions in [Swap Core](swap_core.md). If modification is really necessary, call `clone()` first.
