# Memory Tuning

Last updated: 12/11/2025. Author: LinMingZhe

Memory tuning is an indispensable part of large model training. As model and data scale continue to grow, the corresponding tuning capability becomes essential. This document aims to help you overcome various out-of-memory (OOM) issues.

The "memory" discussed here borrows the concept of Video Random Access Memory (VRAM). It actually refers to the on-chip memory of the NPU, sometimes also called device memory, as opposed to host memory.

## Memory Data Collection

### Memory Snapshot

A memory snapshot is a state log of the native torch memory manager. It associates the lifecycle of each tensor and segment inside the torch memory pool with the runtime stack, and it also exposes other information managed by the memory manager.
Ascend supports the memory snapshot feature, which is very useful for general memory tuning.

#### Memory Snapshot for Built-in Models

The suite integrates the Ascend memory snapshot collection tool to analyze model runtime behavior. All built-in models have been adapted; you only need to modify the `mindspeed_mm/tools/tools.json` file for the change to take effect.

For models that reuse the training pipeline, only the configuration needs to be modified. The supported configuration items are as follows.

```json5
{
  "memory_profile": {
    "enable": false,    // Switch for the memory collection feature
    "start_step": 0,    // Step at which recording starts. The value is the starting point of the training steps; 0 represents the initialization process
    "end_step": 2,      // Step at which recording ends. The value is the starting point of the training steps; 0 represents the initialization process
    "save_path": "./memory_snapshot",  // Path for saving snapshot files
    "dump_ranks": [     // List of ranks to record snapshots for, starting from 0
      0
    ],
    "stacks": "all",    // Stack information recording. Options: python/all
    "max_entries": null // Maximum number of records; null means no limit
  }
}
```

#### Memory Snapshot for a Custom Training Pipeline

For a standalone training pipeline, refer to the following code to adapt the training script to use the profiler feature. The parameter configuration is the same as above.

```python
from megatron.training import get_args
from mindspeed_mm.tools.mem_profiler import memory_profiler

args = get_args()                                   # Get the configuration
memory_profiler.reset(args.mm.tool.memory_profile)  # Refresh the profiler state with the configuration
training_preparation()                              # Run the training preparation code
while iteration < args.train_iters:                 # Main training loop
    memory_profiler.step()                          # Call the profiler to record one iteration
    train_one_step()                                # Train one iteration
memory_profiler.stop()                              # Stop collection
```

#### Memory Snapshot for Arbitrary Scripts

For scripts that do not have a typical training structure, or for local manual debugging, you can call the basic functions directly.

```python
code_not_record()
from mindspeed_mm.tools.mem_profiler import _record
_record()
code_to_record()
```

Dumping and starting the recording can happen in different files.

```python
code_to_record()
from mindspeed_mm.tools.mem_profiler import _dump, _stop
_dump()
_stop()
```

#### OOM Snapshot

If an OOM occurs during training, the program most likely exits before dumping. With the following environment variables, memory snapshots can be collected for the entire run.

```shell
export OOM_SNAPSHOT_ENABLE=1            # Switch for OOM snapshot generation
export OOM_SNAPSHOT_PATH="/home/usr/"   # Save path
```

#### Using Memory Snapshots

After the dump completes, a `pickle` file whose name starts with `snapshot_` is generated in the output directory. You can view the memory snapshot visually on the [torch page](https://pytorch.org/memory_viz).
Combined with the basic memory knowledge and analysis methods described below, you can start memory tuning.

## Memory Analysis

### Memory Composition and Characteristics

The prerequisite for memory analysis is understanding the computation in a model training process and the memory space that the computation requires, from both the time and space dimensions.
The following uses a typical multimodal model training process as an example to break down the memory composition.

- Static memory: memory occupied by tensors whose lifecycle may span, or be equivalent to spanning, multiple training steps

  - Model weights: created during model initialization and, by default, resident in memory for a long time. Includes encoder weights, transformer weights, and so on.
  - Model gradients: created during the backward computation of the model; every parameter involved in the computation generates a corresponding gradient. They can be cleared after the optimizer finishes. Because some backends create gradient buffers for all model weights and hold them persistently, they are classified as static memory.
  - Optimizer states: for the common Adam optimizer, two state parameters are created for each trainable parameter.

- Dynamic memory: memory occupied by tensors that are created in each step and whose lifecycle is significantly shorter than a step.

  - Input data: the multimodal data that the training model learns from.
  - Local variables in the encode stage: created during encoding; after the latent is generated, all local variables can be released. (When the encoder does not participate in training.)
  - Local variables in the transformer forward stage: created during forward computation. Some computations do not save local variables and release them promptly after allocation, while some computations save local variables for backward propagation.
  - Local variables in the transformer backward stage: consume the local variables saved during the forward pass, and other local variables produced during the backward pass are released promptly.
  - Local variables in the optimizer computation stage: consume the gradients produced by the backward pass, and the resulting activations are negligible.

The breakdown above maps to corresponding types in a snapshot of a specific model.
However, every model and training job is different. After different optimization and distributed features are applied, the composition of the same model also changes, so analyze it together with snapshots.
The analysis approach is to find the lifecycle on device and the actual computation that each type of tensor participates in, gain a general understanding, and then analyze further.

### Bottleneck Analysis

Memory usage is subject to the weakest-link effect: once a single point during training exceeds the device resource limit, the entire training process becomes unusable.
Therefore, bottleneck analysis is the first step of memory tuning, that is, finding the memory allocation state at peak memory usage and analyzing the tuning headroom from there.

Bottleneck analysis roughly involves the following steps.

- Find memory peaks and spikes and treat them as key observation points.
- For each observation point, analyze the memory distribution when the local peak occurs, that is, the composition of static memory and the composition of dynamic memory.
- Identify the types of tensors that occupy the most memory, or those whose lifecycle is clearly too long.
- For each type of tensor, pick a representative and analyze its size, allocation timing, and release timing to check whether they meet expectations.
  \- If they do not meet expectations, analyze how the tensor is created and referenced, such as stack references and container references, to prevent an excessively long lifecycle.
  \- If they meet expectations, analyze which tuning methods to apply first; see the next chapter.

### Tuning Philosophy

In memory tuning, the tuning philosophy is shared, and the specific methods below essentially apply the following principles.

- **Partition computation across devices**: spread computations that do not depend on each other across devices, and the tensors they depend on can also be spread across devices according to the computation distribution. Example features include DP, PP, TP, and SP.
- **Offload unused tensors**: offload tensors unrelated to the current computation and load them back when needed. Example features include activation offloading, FSDP offloading, and encoder offloading.
- **Serial computation**: serialize part of the computation to reduce the size of the tensors that a single computation depends on. Example features include gradient accumulation, chunked loss computation, and VAE tiling.
- **Recomputation**: release intermediate computation results directly, keep only a small amount of data, and recompute them when needed. Example features include recomputation.

## Memory Tuning Methods

This chapter describes memory tuning methods for MM, focusing on tuning ideas. For the usage and constraints of specific features in built-in models, the model readme and the corresponding feature documents take precedence.
Memory tuning is also a balancing act: memory gains usually come with overhead, so the actual tuning process requires weighing the pros and cons and focusing on the main contradiction.
For new model development, you can prioritize the tuning methods that are easy to adapt and necessary according to the actual situation.

### Data Parallelism (DP)

Dynamic memory is directly related to the MBS. Therefore, by increasing the number of NPUs and spreading multiple data samples across different devices for computation, dynamic memory usage can be reduced.
At the same time, you can exploit the dependency of the computation process on static memory to manage static memory at a fine granularity, performing operations such as aggregation when needed, which reduces static memory usage.
Combining the two produces DDP and FSDP, corresponding to ZeRO1 and ZeRO3.

#### DDP

The suite's DDP uses the Megatron version of the distributed optimizer, which can shard the optimizer states in static memory.

```shell
--use-distributed-optimizer
```

#### FSDP

The suite supports the FSDP2 feature. See [FSDP2 Usage Instructions](../../features/parallel/fsdp2.md) to use FSDP2.
When this feature is enabled, static memory can be fully sharded by default.

### Static Memory Tuning

#### FSDP Offloading

Offloading is very common among memory tuning methods and will be mentioned several times later. The FSDP described above actually already includes an advanced offloading feature, which can be enabled through configuration.

For internal models, modifying the yaml file takes effect directly, offloading parameters, gradients, and optimizer states to CPU memory.

```yaml
offload_to_cpu: True
```

> **Note**
>
> When `offload_to_cpu=True` is set, the communication group must be set to a dual backend in the entry script, that is: `--distributed-backend npu:hccl,cpu:gloo`

For details about the feature, see the [official documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.CPUOffloadPolicy).

#### Encoder Offloading

For generative models, the encoder often does not participate in training, so you can exploit the temporal locality of parameters: offload the weights while the backbone model is computing, and use caching to amortize the transfer overhead.

In MM built-in models, this is enabled by configuring the offload interval, and it applies to models based on SoraModel, such as wan2.2.

```json
"encoder_offload_interval": 8,
```

- Gain: during non-encoder stages, the encoder weights are released from static memory
- Cost: additional weight transfer overhead and cache management overhead

#### Model Parallelism

Tensor parallelism (TP) and pipeline parallelism (PP) are common model parallelism strategies. They directly shard weights across different devices, which reduces static memory usage, and achieve algorithmic equivalence through communication and other operations.
On top of PP, algorithms such as VPP reduce the bubbles produced by PP and improve computing efficiency.
Built-in MM models such as Qwen2.5VL and Open-Sora Plan1.3 support enabling model sharding. For details, see the corresponding model readme and feature documents.

- TP

  - Gain: shards the model and reduces the computation load per device
  - Cost: high communication overhead; every MLP/Attention needs to communicate activations

- PP

  - Gain: shards the model, reduces activations, and spreads the computation across devices
  - Cost: activation communication exists between every two PP stages, and bubbles exist, which hurts computing performance

#### Expert Parallelism

Expert parallelism (EP) shards the experts of an MoE model in MoE scenarios and is a special form of model parallelism.

- Gain: memory savings, because only part of the weights is stored on each rank.
- Cost: performance loss due to memory transfer overhead and activation communication overhead. There is also the challenge of expert balance.

### Dynamic Memory Tuning

#### Recomputation

In the forward and backward computation stages of backbone transformer models such as the DiT of generative models and the decoder of understanding models, saving activation intermediate results is a common bottleneck.
Activation recomputation saves only a few checkpoints and recomputes the intermediate variables from those checkpoints during the backward pass, which avoids saving most intermediate results until backward and greatly reduces activations.

For FSDP2 models, recomputation takes effect through yaml configuration. For usage, see [FSDP2 Usage Instructions](../../features/parallel/fsdp2.md).

> - `recompute_modules`
>     - Description: configures activation recomputation to trade computation for memory
>     - Configuration format: same as `sub_modules_to_wrap`
>     - Constraints: conflicts with the full recomputation feature in Megatron, which must be disabled

For non-FSDP2 models, recomputation can be enabled through args configuration.

```shell
GPT_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 40 \
"
```

- Gain: significantly reduces the dynamic memory peak of the forward and backward computation of the backbone model.
- Cost: additional computation overhead; in full recomputation scenarios, the forward computation volume doubles.

#### Gradient Accumulation

When the micro-batch size increases, a step can train on multiple data samples, and activation usage grows accordingly.
Gradient accumulation splits one global batch into multiple micro-batches, processes only one micro-batch per step, and accumulates the gradients.
This achieves the per-step training data volume of a larger MBS while using a smaller MBS.

You can control this by changing GRAD_ACC_STEP and MBS in the launch script parameters.

```shell
MBS=1
GRAD_ACC_STEP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))
# ...
--micro-batch-size ${MBS} \
--global-batch-size ${GBS}
```

- Gain: reduces activation memory usage.
- Cost: the computation volume issued at one time becomes smaller, and more communication is introduced in FSDP scenarios.

#### Sequence Parallelism: TP-SP/Ulysses/ring/USP

Sequence is an important concept in transformers. The sequence dimension is interdependent in attention computation, but independent in token-level computation of the MLP or element-level computation.
Exploiting this locality, the industry has designed various sequence parallelism algorithms, and MM has adapted the mainstream ones.

- TP-SP

  - Gain: partitions the sequence of computation processes such as LayerNorm and Dropout between MLP/attention modules, which reduces dynamic memory usage in the partitioned stages and eliminates redundant computation.
  - Cost: TP must be enabled first.

- Ulysses sequence parallelism

  - Gain: partitions computation and activation memory across devices at the partitioned stages according to the heads in attention computation and the sequence dimension of other computations, reducing dynamic memory usage.
  - Cost: increases communication volume; the CP size must be divisible by the number of heads.

- RingAttention sequence parallelism

  - Gain: holds part of Q and passes part of KV around in a ring while computing, spreading dynamic memory and computation across the partitioned stages.
  - Cost: increases communication volume.

- USP hybrid sequence parallelism (Ulysses + RingAttention)

  - Gain: spreads dynamic memory and computation across the partitioned stages, and is a compromise between the high performance of Ulysses and the flexible partitioning of ring.
  - Cost: increases communication volume.

Multiple sequence parallelism algorithms can be selected according to the actual situation and combined with other partitioning strategies. Considering factors such as actual throughput, the situation can be fairly complex.
Some studies already provide recommendations. The following quotes some viewpoints from the [USP paper](https://arxiv.org/pdf/2405.07719); see the original paper for detailed analysis.

> - It is recommended to use Unified-SP instead of SP-Ring and SP-Ulysses, because it provides the functions of both and brings additional advantages
> - It is recommended to use DP first, and consider whether to enable SP only when the batch size is insufficient for partitioning.
> - When using SP, always pair it with ZeRO-1/2; ZeRO-3 and Offload can also be further enabled to trade communication overhead for memory savings.
> - At large scale, the communication overhead of SP is better than that of TP-SP; enabling GQA can further reduce the communication cost of SP.
> - Switching from TP-SP to SP does not increase the sequence length during training. SP+ZeRO3 can achieve a sequence length close to that of TP-SP.
> - It is recommended to adopt a higher degree of SP parallelism: when the number of attention heads is limited, a larger ring dimension can be set to split long sequences across more computing devices for training, which is a unique advantage that TP-SP cannot achieve.

#### VAE Sequence Parallelism

In the encoder stage of generative models, the main dynamic memory bottleneck is the VAE encoding of video data, especially when processing videos with high resolution and a large number of frames.
For mainstream VAEs, you can exploit the computational locality of convolutions together with communication algorithms to share the computation of one data sample across multiple devices.
However, because each model is implemented differently, the specific algorithms also differ. Built-in MM models such as OpenSoraPlan1.3 and CogVideoX have adapted partitioning algorithms. See the readme for how to enable them and their constraints.

- Gain: reduces the dynamic memory bottleneck in the VAE encode stage.
- Cost: increases communication. Some algorithms are constrained by actual parameters such as the number of frames.

#### VAE Tiling

Another way to address the VAE activation bottleneck is to enable tiling. Mature tiling algorithms already exist in the industry.
VAE tiling saves memory by splitting the image into multiple smaller overlapping tiles and processing one tile at a time instead of processing the whole image at once.

> **Note**
>
> Tiling does not significantly degrade generation quality, but it is not equivalent to the original computation.

Built-in MM models can enable tiling in `model.json`.

```json5
{
    "ae": {
        "enable_tiling": true # "use_tiling" for some models
    }
}
```

- Gain: significantly reduces the dynamic memory bottleneck in the VAE encode stage.
- Cost: the computation is not equivalent, the issued computations are small and numerous, and may include overlapping tiles, so computation time usually increases.

#### Activation Offloading

Activation offloading is also a common practice. Usually, the intermediate results of recomputation, or activations with a relatively long lifecycle such as caches, are offloaded.

The simplest way is to implement offloading directly with `to`.

```python
for cache in caches:
    for k in cache:
        cache[k] = cache[k].to("cpu")
for i in range(len(caches)):
    for k in caches[i]:
        caches[i][k] = caches[i][k].to("npu")
    run_your_code(**caches[i])
```

We have made asynchronous optimizations for offloading in some scenarios. See the async offload feature document.

> **Note**
>
> Note that torch's tensor lifecycle management reuses Python's reference management, so when offloading you must ensure that the references to the offloaded tensor have been cleared; otherwise the space will not be reclaimed.

#### Chunked Loss Computation

In a typical autoregressive language model, before loss computation the LM Head maps each token to the vocabulary size, and then softmax computes the probability of each word, which is used to compute the loss value against the ground truth.
This computation produces a large number of activations, especially when both the vocabulary and the sequence length are large.
MM applies column sharding to the LM Head and uses algorithms such as dynamic softmax on activations to compute the loss.
Users can also use the ChunkLoss feature, which exploits the independence of loss across the sequence dimension to compute the loss in chunks along the sequence dimension and accumulate it, further reducing memory usage.

- Partitioning along the vocabulary dimension.

  - Gain: reduces activation memory usage; the larger the TP, the greater the gain.
  - Cost: effective only when TP is enabled.

- Computing and accumulating the loss in chunks along the sequence dimension.

  - Gain: significantly reduces activation memory usage in the loss computation stage.
  - Cost: the issued computations become fragmented, reducing hardware utilization.

## Reference
