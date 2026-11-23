# FSDP2 Muon Optimizer

## Applicable Backends

FSDP2 + Mcore

## Background and Challenges

Muon (Momentum Orthogonalized by Newton-Schulz) is a class of optimizers designed for matrix parameters. It orthogonalizes the update direction of two-dimensional weight matrices and can serve as an optimization choice in addition to AdamW.

The main advantage of Muon is that it leverages the matrix structure of hidden-layer weights in neural networks to impose an orthogonalization constraint on the momentum update direction, making the update direction of two-dimensional weight matrices closer to a well-conditioned spectral norm update. In some public experiments, Muon has demonstrated better sample efficiency and computational efficiency, that is, achieving a comparable loss with less training time or fewer FLOPs. However, the actual benefit still depends on the model architecture, batch size, learning rate, and training stage, and needs to be validated against the specific task.

Public use cases include:

- [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) trains with Muon/MuonClip at the 1T MoE scale;
- The [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) technical report presents the Muon Optimizer training algorithm for DeepSeek-V4.
- [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/README_CN.md) is trained with the Muon optimizer, and Muon is recommended for continued training or LoRA fine-tuning;
- [NVIDIA NeMo-RL](https://docs.nvidia.com/nemo/rl/latest/guides/muon-optimizer.html) provides examples of using Muon in the Qwen3-235B-A22B SFT and Qwen2.5-7B DAPO scenarios.

The implementation in this repository partially references the [MoonshotAI/Moonlight example Muon](https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py) and [KellerJordan/Muon](https://github.com/KellerJordan/Muon/blob/master/muon.py), and adds support for DTensor-sharded parameters in the FSDP2 backend. The code is in `mindspeed_mm/optimizer/muon/`, and parameter selection is in `mindspeed_mm/fsdp/optimizer/muon_routing.py`.

## Solution

Under the FSDP2 backend, the Muon optimizer first splits the parameter groups into Muon parameters and AdamW fallback parameters:

- Two-dimensional weights use Muon, except those listed below.
- These parameters use the AdamW fallback:
  - biases (names ending in `.bias`);
  - parameters of `nn.Embedding` modules and of modules named `lm_head` or `output_layer`;
  - parameters whose names contain `embedding`, `embed_tokens`, `output_layer`, `lm_head`, or a keyword from `muon_fallback_param_keywords` (case-insensitive);
  - two-dimensional weights with a dimension of size 1: orthogonalizing a vector only normalizes it;
  - all parameters that are not two-dimensional (unless `muon_per_expert` selects them, see below).
- Existing settings such as learning rate, weight decay, `no decay` grouping and `lr_scaling_plan` are kept for each split group. Set `no_decay_params` or `no_decay_modules` to exclude specific parameters or module types from weight decay. Both lists default to empty.

The Muon update process is as follows:

1. Update the momentum buffer, an exponential moving average of the gradient (Nesterov momentum by default).
2. Orthogonalize the update with Newton-Schulz iterations:
   - For local computation with 5 or 10 steps on a matrix whose long side is more than 1.5 times the short side, Gram Newton-Schulz is used in FP16. It iterates on the small Gram matrix X·Xᵀ and recomputes it from X at fixed iterations to limit rounding error.
   - Otherwise, standard Newton-Schulz is used, in BF16.
   - The normalizing norm is computed in FP32.
   - The update stays in this compute precision (BF16 or FP16), including when it is exchanged between ranks, because upcasting a BF16 or FP16 result adds no precision. The final addition to an FP32 parameter is computed in FP32.
   - On NPU, eligible shapes use CATLASS SYRK when `torch_catlass` provides kernels for Ascend 950PR&950DT products. Other shapes use PyTorch. Missing or unsupported CATLASS kernels trigger a warning on NPU.
3. Apply weight decay to the weights, then apply the orthogonalized update scaled by `lr`, `matched_adamw_rms` and the matrix shape.

### Sharded parameters

In FSDP2, parameters are sharded as DTensors. Each eligible row-sharded 2-D weight is assigned to one "owner" rank using a Newton-Schulz cost estimate. The owner gathers the matrix with an all-to-all, orthogonalizes it, and returns each rank's rows. The cost estimate accounts for the Standard and Gram paths, Gram resets, CATLASS dispatch and per-call overhead. Matrices are exchanged in waves. A wave can overlap computation on another wave, depending on the workload and backend. Before the exchange, one all-reduce per participating process group checks that ranks agree on parameter identities, shapes, gradient presence and scheduling settings, and verifies local row layouts. This check assumes matching parameter groups and mesh membership across ranks.

Per-head and per-expert updates can use separate owner exchanges for their head or expert blocks. Column-sharded per-head weights that qualify for distributed Gram Newton-Schulz stay sharded. Each rank computes its partial Gram matrix in FP32 and the ranks sum the partial matrices in FP32. Unsupported layouts fall back to DTensor redistribution.

The whole-matrix owner exchange accepts these environment variables:

- `MINDSPEED_MM_MUON_OWNER_WAVE_BYTES`: maximum bytes of full matrices that one owner gathers in a wave. Default: 1 GiB. A single larger matrix gets a wave of its own.
- `MINDSPEED_MM_MUON_OWNER_MAX_INFLIGHT`: gather-window size and pending-scatter limit, each independently. Default: 2.
- `MINDSPEED_MM_MUON_OWNER_SCATTER_COMM`: set to `1` to send results on a separate process group when the owner group spans all ranks. Default: `0`.

The wave budget counts gathered input matrices, not total optimizer memory. Packed exchange buffers, retained updates, stacked owner batches and Newton-Schulz workspaces all add to memory usage. Increasing the number of waves in flight can increase memory without improving optimizer time. To evaluate overlap and owner balance, record gather/NS/scatter intervals, exposed waits, total optimizer time, peak memory and per-rank owner compute times at matched settings.

### Layout selection

`Shard(d)` below names a **tensor** dimension; the mesh dimension identifies the ranks sharing that shard. For per-expert weights, the tensor shape is `[experts, rows, columns]`.

| Parameter layout | Execution path | Configuration guidance |
| --- | --- | --- |
| Whole matrix, 2-D, one plain `Shard(0)` with other mesh dimensions replicated | Owner all-to-all when the gradient's local rows match the expected row band on every rank | Use contiguous row shards; the mesh dimension's name is unrestricted. |
| Per-head, one plain `Shard(0)`, with all row boundaries between heads | Local NS on complete heads | Making the FSDP shard count divide the head count is a sufficient rule. The actual check also accepts uneven or empty bands whose endpoints preserve complete heads. |
| Per-head, one plain `Shard(1)`, qualifying head shape and step count | Distributed Gram NS with FP32 partial-Gram reduction | A single head must satisfy `columns > 1.5 * head_rows`, with 5 or 10 NS steps. |
| Per-head, one plain `Shard(0)` or `Shard(1)`, remaining supported cases | Per-head owner all-to-all | Row boundaries may split heads; nonqualifying column shards also use this path. |
| Per-expert, optional `Shard(0)` on the mesh dimension named `ep`, and one plain `Shard(1)` on another mesh dimension | Per-expert owner all-to-all over the row-sharding group | Keep experts independent under EP and shard each expert's rows under eFSDP. This layout still communicates for row-sharded experts. |
| Other layouts, including multiple moving shards and unsupported eHSDP layouts | All-gather/redistribute fallback | Fast paths reject `Partial` placements and `Shard` subclasses. Fallback compatibility depends on DTensor supporting the requested redistribution. |

The per-expert implementation recognizes the mesh name `ep` explicitly. Renaming it or leaving mesh dimensions unnamed can cause expert shards to be gathered as well. In the fallback, a recognized EP expert shard remains sharded while matrix dimensions become replicated. A single-rank per-head shard is computed locally.

### Per-head and per-expert Muon

Model modules register layouts with `register_muon_layout` in `mindspeed_mm/fsdp/optimizer/muon_metadata.py`. Register the parameter on the module that directly owns it. The FSDP2 optimizer uses these declarations when `muon_per_head` or `muon_per_expert` is enabled. Each enabled mode requires at least one eligible trainable annotation. Other eligible matrices still use whole-matrix Muon.

The Kimi K3 modules declare these update blocks, which also apply when Kimi K3.1 Pro reuses the corresponding modules.

- Delta attention divides each of the separate Q, K and V projections into equal, contiguous head blocks along the row dimension.
- MLA divides the query projection (`q_proj` or `q_b_proj`) and `kv_b_proj` per head. Each `kv_b_proj` head keeps its K and V rows together.
- Routed experts use one matrix per expert for each of `gate_up_proj` and `down_proj`. Gate and up remain combined in `gate_up_proj`.

### QK-Clip

QK-Clip rescales attention weights when the observed logits exceed a threshold. When `qk_clip_threshold` is set, the Kimi MLA attention layers record the maximum pre-softmax logit of each head during training. After the optimizer update, for every head whose maximum exceeded the threshold, with `gamma = qk_clip_threshold / max_logit`:

- the query content weights are scaled by `gamma ** qk_clip_alpha`;
- the key content weights are scaled by `gamma ** (1 - qk_clip_alpha)`;
- the query rotary-position weights are scaled by `gamma`.

Under Ulysses or KV all-gather context parallelism, each rank records the heads it computes in a full-width vector. The maxima are then combined across the ranks of the query weight's device mesh. QK-Clip requires `optimizer: muon`, `muon_per_head`, and a model hook that binds the query and key weights and supplies metrics. The Kimi MLA eager and NPU flash-attention paths supply these metrics. The Ascend C sparse attention path in Kimi K3.1 Pro's DSA does not supply them.

For Ulysses, local heads occupy a contiguous segment of that vector and unowned heads contribute `-inf`. Packed-sequence padding is excluded before reduction. The query weight's mesh must cover the ranks contributing query tokens and heads so that each head's maximum is global. The metric placement and reduction are already implemented; end-to-end model validation is still needed for each Ulysses/ring, packed-sequence and activation-recompute configuration. Auxiliary maxima and sums kept for the attention backward pass must be retained even when clipping is disabled.

### Semantics and numerical differences

| Change | Effect |
| --- | --- |
| Owner gather/compute/scatter at matched coefficient, step and block settings | Changes where NS executes. Data exchange preserves tensor values; batched kernels can still differ in rounding. |
| Gram reformulation, symmetric half-matrix products and fused polynomial evaluation | Algebraically equivalent in exact arithmetic, with finite-precision differences from FP16/BF16 and operation ordering. |
| Coefficient schedule or iteration count | Changes the finite-iteration update algorithm. Polar Express remains the default; the earlier implementation used the single Keller triple. Select `ns_coefficients: keller` explicitly to compare the old coefficient choice. |
| Per-head/per-expert orthogonalization | Changes the blocks optimized independently, and therefore the update direction. |
| QK-Clip | Adds logit-dependent rescaling of query/key weights after the optimizer update. A metric or control validation failure leaves the weights and saved optimizer state unchanged. |

The momentum buffer changes from a running sum to an EMA. With the same gradients and momentum, the EMA equals the running sum multiplied by `1 - momentum` in exact arithmetic. Scaling the input by this positive factor preserves its direction before epsilon normalization. Epsilon and floating-point rounding can change the computed update. Compare long-run loss at matched coefficients, steps, optimizer layout and kernel settings to assess training accuracy.

### Numerical rules and provenance

| Rule | Basis and validation scope |
| --- | --- |
| Gram polynomial reformulation and symmetric products | Iterating on `X X^T` allows polynomial factors of the same Gram matrix to be composed before applying them to `X`. These factors commute in exact arithmetic. Finite precision still requires numerical checks. |
| `long_side > 1.5 * short_side` | A dispatch heuristic motivated by rectangular versus square-product cost. It is not a universal hardware crossover. |
| Gram restarts and 5/10-step restriction | The numerical restart policy uses boundaries from `[2, 5, 7]` that precede the final step. Five steps reset after iteration 2. Ten steps reset after iterations 2, 5 and 7. Calibrate restarts separately for ordinary GEMM and symmetry-enforcing SYRK. |
| CATLASS row thresholds 32/256/512, batch threshold 32 and column thresholds 8192/16384 | A cost model indicates where computing one triangle should win; the values themselves were calibrated by benchmarking one CATLASS build, so another build may need other values. The polynomial path additionally requires rows divisible by 32 and a contiguous Gram tensor. |

Threshold measurements should report exact source commits, NPU model, CANN, PyTorch, `torch_npu` and `torch_catlass` versions, shapes, dtypes, coefficients, steps, warmup/repetitions and timing method. Repeat the comparison on the intended deployment build before treating these heuristics as optimal. Historical calibration and profiling results do not establish long-run training accuracy for a newer source or training base.

Ten-step product orientation: Newton-Schulz uses two separate products. `gram_product` computes `X @ X^T` for rectangular `X`, where the transpose is structural. `symmetric_product` multiplies two square matrices that are symmetric in exact arithmetic. Using `A @ B^T` instead of `A @ B` for the symmetric product is algebraically equivalent but amplifies the antisymmetric rounding drift `A @ (B^T - B)`, which compounds across iterations.

The Torch kernels evaluate the symmetric product as `A @ B`. The CATLASS SYRK kernel uses its own `A @ B^T` form, which is safe because its polynomial mirrors the lower triangle and keeps the operands exactly symmetric. The two backends evaluate different finite-precision forms of the same exact product.

### Megatron portability

The Megatron optimizer adapter constructs the same `Muon` class for ordinary parameter groups. Coefficient schedules and local NS/CATLASS helpers can therefore be reused there. The FSDP2 owner exchange and per-head/per-expert routing rely on DTensor layouts and the FSDP model builder; they are not automatically available for Megatron TP shards.

A Megatron port must identify global matrix shapes and shard orientation, provide global normalization and the correct process groups, reproduce parameter-block routing, and map optimizer/checkpoint state. Distributed Gram expects complete rows over local column slices. It sums squared local norms in FP32 for normalization and sums the FP32 partial Gram matrices at the start of each segment. QK-Clip additionally needs attention metric collection, global head/token maxima and bindings to the corresponding query/key weights. These adapters require separate TP/CP validation.

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
  ns_coefficients: polar_express
```

For Kimi K3, per-head and per-expert Muon and QK-Clip can be added:

```yaml
training:
  optimizer: muon
  muon_per_head: true
  muon_per_expert: true
  qk_clip_threshold: 100.0
  qk_clip_alpha: 0.5
```

#### Parameter Description

| Parameter | Type | Default Value | Description |
| --- | --- | --- | --- |
| `training.optimizer` | string | `adamw` | Optimizer type. Valid values are `adamw`, `adamw_swap`, and `muon`. |
| `training.matched_adamw_rms` | float | `0.2` | Controls how closely the magnitude of the Muon update matches the RMS of the AdamW update. |
| `training.muon_momentum` | float | `0.95` | Momentum coefficient of the SGD momentum inside Muon. |
| `training.ns_steps` | int | `5` | Number of Newton-Schulz orthogonalization iterations. A larger number makes the orthogonalization more thorough, but the overhead also increases. |
| `training.muon_fallback_param_keywords` | list[str] | `null` | Extra name keywords for AdamW fallback, added to the built-in rules. |
| `training.lr` | float | `5e-5` | Base learning rate. For Muon parameters, the update magnitude is adjusted on top of the base learning rate based on `matched_adamw_rms` and the matrix shape. |
| `training.weight_decay` | float | `0.0` | Weight decay coefficient. |

#### Additional FSDP2 Parameters

- **`ns_coefficients`**
  - Description: Newton-Schulz coefficient schedule: one `(a, b, c)` triple per iteration. If the schedule is shorter than `ns_steps`, its last triple is repeated.
  - Values:
    - `polar_express` (default): the five-step schedule from the [Polar Express paper](https://arxiv.org/abs/2505.16932).
    - `keller`: Keller Jordan's original single triple `(3.4445, -4.7750, 2.0315)`.
    - `you`: Jiacheng You's [five-step schedule](https://x.com/YouJiacheng/status/1905861218138804534).

- **`muon_per_head`**
  - Description: Orthogonalize each head block of weights with a model-provided per-head layout.
  - Default value: `false`.

- **`muon_per_expert`**
  - Description: Orthogonalize each expert matrix of weights with a model-provided per-expert layout.
  - Default value: `false`.

- **`qk_clip_threshold`**
  - Description: Finite, non-negative limit for the maximum per-head attention logit. Requires `muon_per_head` and a compatible metric provider.
  - Default value: `null` (disabled).

- **`qk_clip_alpha`**
  - Description: Value in `[0, 1]` that splits the QK-Clip scale between queries (`gamma ** alpha`) and keys (`gamma ** (1 - alpha)`).
  - Default value: `0.5`.

### Mcore Backend

Enable it with `--optimizer muon` and set `"muon_optimizer": true` in the `patch` section of model.json. Mcore Muon is incompatible with `--use-distributed-optimizer`. The FSDP2 parameter routing rules above apply only to FSDP2.

#### Parameter Description

| Parameter | Type | Default Value | Description |
| --- | --- | --- | --- |
| `muon_optimizer` (model.json `patch` section) | bool | `false` | Wiring switch for enabling Muon on the Mcore side. |
| `--matched-adamw-rms` | float | `0.2` | Same meaning as `matched_adamw_rms` on the FSDP2 side. |
| `--muon-momentum` | float | `0.95` | Same meaning as `muon_momentum` on the FSDP2 side. |
| `--ns-steps` | int | `5` | Same meaning as `ns_steps` on the FSDP2 side. |

#### Example Script

Use `examples/wan2.2/5B/t2v/pretrain.sh` as a starting point. Set the Muon optimizer flag and enable `muon_optimizer` in `examples/wan2.2/5B/t2v/pretrain_model.json`.

## Notes

1. In FSDP2, eligible two-dimensional matrix parameters use Muon. Per-expert annotations also select three-dimensional expert weights. Other groups inside the Muon optimizer use its AdamW fallback.
2. In FSDP2, sharded parameters are exchanged between ranks for orthogonalization, as described above. The owner wave budget is an input-gather budget, not a bound on peak optimizer memory.
3. `ns_steps` can be adjusted based on training stability and performance requirements. Smaller values can be used for fast validation runs; for formal training, it is recommended to verify the impact using loss curves and throughput metrics.
4. `matched_adamw_rms` affects the Muon update magnitude. When modifying the learning rate, it is recommended to monitor its impact on convergence simultaneously.
5. Loading a supported legacy Muon checkpoint converts `muon_buffer` to `momentum_buffer` using `1 - momentum` with the saved momentum setting. If both keys are present, the current buffer takes precedence. FSDP2 distributed checkpoint loading reads legacy keys through Muon's checkpoint hooks and restores defaults for known newer group fields. Strict loading still requires fields such as the learning rate. CPU tests cover plain and flattened MultiOptimizer round trips and the next optimizer step. Multi-rank EP training resume remains unverified.
6. QK-Clip validates metrics and controls before updating parameters. An invalid or missing metric may require a fresh forward pass before retrying. Valid metrics are cached before threshold and alpha validation, so correcting either control can allow a retry without another forward pass. Groups with a zero learning rate skip clipping and leave recorded metrics available for the next step. Distributed ranks must use the same clipping controls.
7. With NPU flash attention, QK-Clip leaves padded query rows out of the per-head maxima, using the `[batch, seq]` padding mask that the kernel itself does not receive. This covers right-padded batches, packed batches with tail padding (including chunked packed attention) and cached queries. The supported padded paths pass the causal mask without the key-padding mask, so Kimi training with QK-Clip rejects left or internal padding. Packed statistics are reduced in each sequence's head order. Padded batches with ring or local-query KV all-gather context parallelism raise an error. Square padding masks are recognized for metrics, but the pre-existing kernel-mask dispatch for `batch == seq` remains unsupported.
