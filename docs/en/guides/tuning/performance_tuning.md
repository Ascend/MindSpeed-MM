# Performance Tuning

Performance tuning is a critical part of training. Reasonable performance tuning can significantly improve model training efficiency and reduce resource consumption. This document introduces the best practices and methods of MindSpeed MM in performance tuning. First, the methods for collecting performance data are introduced, followed by the performance analysis process, and finally the commonly used performance tuning methods.

In this document, performance refers to the time required for a machine (GPU, NPU, or other platforms) to complete one end-to-end training pass under a specified model and input data. Considering the differences in training data volume and training epochs across different models, the performance defined here is the time required to complete the training of one batch. By end-to-end here, we usually mean the process of completing a single training step of an artificial intelligence model. In other words, both the measurement and optimization of performance discussed in this document are considered from the perspective of the model.

For a batch, the time mainly consists of the following parts:

Total time per batch = Data loading time + Model forward/backward time + Optimizer time + Model post-processing time + Communication time + Scheduling time

The details are described as follows:

- Data loading time: the time taken by the model to load the data it requires (such as images, videos, and text), including the time to read data from hardware storage devices into CPU, preprocess the data on CPU (encoding/decoding and other operations), and place the CPU data onto the device.
- Model forward and backward time: the time of the forward and backward processes of the deep learning model, namely the forward and backward processes, including the time for forward data computation and backward data differentiation.
- Optimizer time: the time for updating model parameters.
- Model post-processing time: the time after the optimizer update, including data post-processing or some necessary synchronization operations, which usually depends on model-specific operations.
- Communication time: the communication time between cards within a single node and between nodes across multiple nodes. Due to PyTorch's special mechanism, when communication and computation can run in parallel, this represents the communication time that is not masked by computation.
- Scheduling time: the time required for the model to go from CPU instructions to invoking the Kernel on the NPU side.

## Performance Data Collection

MindSpeed MM provides a profiling tool that supports both static collection and dynamic collection modes for collecting performance data during model training. The configuration entries differ between the two training backends:

- **FSDP2 backend**: Configured in the `tools` section of the training YAML. `tools.profile` is used for performance data collection (fields such as `enable`, `profile_type`, and `ranks`), and `tools.memory_profile` is used for memory snapshot collection (fields such as `enable`, `start_step`, `end_step`, and `save_path`). Refer to the `tools` section in each FSDP2 model sample YAML (for example, `examples/qwen3vl/qwen3vl_30B_config_v1.yaml`).
- **Mcore (Megatron) backend**: Configure the collection switches and parameters through `mindspeed_mm/tools/tools.json`. For detailed usage, see [Profiling Collection Tool](../../tools/tools.md#profiling-tool);

When profiling data, it is recommended to record the following key contextual information at the same time to facilitate subsequent analysis:

- Model name, parameter count, and distributed strategy (TP/PP/DP/EP/SP settings)
- Training configuration (batch size, sequence length, gradient accumulation steps)
- NPU model, cluster scale, and CANN/TorchNPU version numbers
- Whether to enable features such as recomputation and activation offloading

## Performance Analysis Process

### Performance Analysis Tools

After the performance data is collected, you can use [MindStudio Insight](https://www.hiascend.com/document/detail/en/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/en/user_guide/overview.md) to visualize and analyze the performance data and locate performance bottlenecks.

MindStudio Insight is a performance analysis tool provided by Ascend that supports multi-dimensional analysis of profiled data, including:

- Operator time consumption analysis: identify operators with long execution time and locate computation bottlenecks
- Communication time consumption analysis: analyze the time ratio between communication and computation, and optimize communication strategies
- Memory analysis: view memory usage and identify memory bottlenecks
- Pipeline analysis: analyze the bubble ratio of pipeline parallelism

#### Multi-Rank Timeline Alignment

During multi-device collection, each rank's timeline is recorded independently. In multi-node scenarios, inconsistent clocks across machines may cause the relative positions of the timelines of different ranks to be misaligned, which affects judgments in cross-rank analysis (such as fast/slow ranks and communication waiting); this is a display issue when analyzing the collected data and does not affect actual training. MindStudio Insight already implements automatic alignment of relative timeline positions for single-device, cluster, and multi-model scenarios; if alignment is still inaccurate in multi-node scenarios, you can use its single-device or swimlane time calibration function (set an offset in the timeline view to shift the corresponding rank/swimlane left or right) to manually align the time base of each rank. For details, see the "Timeline - Time Alignment for Single Device and Swimlane" subsection in the [MindStudio Insight System Tuning Guide](https://www.hiascend.com/document/detail/en/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/en/user_guide/system_tuning.md).

### Performance Breakdown Analysis Method

After obtaining the profile data, it is recommended to perform a systematic breakdown along the following dimensions:

1. **Computation vs. Communication vs. Free ratio**: First examine the proportions of computation, communication, and Free in the overall time. A high Free ratio usually indicates a host-bound or scheduling problem. In this case, open the **dispatch curve** (Flow Events) in the MindStudio Insight timeline: the periods where the curve is steep or shows obvious gaps are often where the host fails to dispatch in time and NPU waits idle. By comparing these positions, the dispatch bottleneck can be located and targeted optimization can be performed (see [Scheduling Dispatch Optimization](#scheduling-dispatch-optimization)). A high communication ratio requires checking the computation-communication overlap.
2. **Fast/slow rank investigation**: Compare the time consumption distribution across ranks. If a certain rank is significantly slower than others in communication or dispatch, it is necessary to investigate whether it is dispatch-bound, load-imbalanced, or a hardware issue.
3. **Operator-level breakdown**: Break down the computation time by operator type (Cube/Vector/AI CPU) and identify the operator type with the largest share. For Cube operators (MatMul/FA and so on), low MFU indicates insufficient compute density; an excessively high share of Vector operators indicates room for fusion; the presence of AI CPU operators means that those operators fall back to CPU execution and are usually slow, so they should be eliminated or replaced with NPU operators.
4. **Submodule MFU analysis**: Calculate the MFU separately for model submodules (Attention, MLP, etc.) to precisely locate the computational bottleneck.

#### Communication Time Analysis

Analyze the proportion of communication time in the total time, focusing on the portion that is not masked by computation, which is the primary target for optimization. If communication is not sufficiently overlapped with computation, the computation-communication masking can be improved by adjusting the prefetch timing (initiating AllGather earlier).

#### Slow/Fast Rank Troubleshooting

During large-scale cluster training, slow/fast ranks (some ranks being significantly slower than others) are a common performance bottleneck. The troubleshooting approach is as follows: (1) run the single-node script on all servers and compare performance data; (2) gradually scale up from a small cluster and use binary search to locate the slow node; (3) profile cluster data to identify the rank with the longest communication wait. Common root causes include: dispatch-bound (increased ACL interface latency on a particular rank), load imbalance (multimodal data with varying image sizes causing large differences in the ViT portion), network congestion (improper cross-block networking or HCCL algorithm configuration), and host resource contention (non-training processes on the node occupying CPU).

When locating the issue from the communication timeline, first distinguish whether the elapsed time is **actual communication transfer** or **synchronization waiting**: if communication operators such as AllGather/ReduceScatter contain a long `notify_wait`, it indicates that this rank is already ready and is waiting for a slower rank. The root cause is usually on the slow rank (abnormal Free caused by dispatch or memory defragmentation) rather than on this rank. Adjusting the communication prefetch on this rank is unlikely to help; instead, the dispatch or memory defragmentation issue on the slow rank should be investigated and eliminated first.

In addition, **memory defragmentation** is also a common cause of prolonged Free. When memory usage is too high, the allocator must first defragment existing memory and consolidate a contiguous block before completing the allocation. This process is costly and appears on the timeline as a noticeable and prolonged Free, significantly slowing down training. During troubleshooting, it is recommended to enable memory collection together with profiling (under FSDP2, set `enable: true` in the `tools.memory_profile` section of the YAML). If the overall memory usage shows obvious rises and falls (sudden spikes and drops), it usually indicates that memory defragmentation has been triggered. Since defragmentation has a significant impact on performance, memory usage should be reduced, for example by lowering the sequence length or increasing the number of ranks.

#### MFU Decomposition Analysis

Model FLOPs Utilization (MFU) refers to the ratio of the FLOPs actually achieved during training to the theoretical peak FLOPs of the hardware. It reflects the degree to which hardware computing power is effectively utilized and is a commonly used metric for measuring computational efficiency. By decomposing MFU by model submodule (such as Attention and MLP) and by stage (forward, backward, and recomputation), the computational bottleneck can be precisely located.

The analysis steps are as follows:

1. **Calculate theoretical FLOPs**: Based on the model structure and input shape, calculate the theoretical FLOPs of each module separately.
2. **Obtain actual elapsed time**: Extract the actual elapsed time of each module in the forward and backward stages from the profile data.
3. **Calculate the submodule MFU**: `submodule MFU = submodule FLOPs / (submodule time × theoretical compute capacity)`.
4. **Compare and locate the bottleneck**: Compare the MFU values of each submodule to identify modules with low utilization.

In practice, the MFU of different submodules often has structural causes: Attention layers consist mainly of dense matrix multiplications (such as Flash Attention), so their MFU is usually high; MoE expert layers contain non-computational operations such as permute/unpermute, so their MFU is usually low. When interpreting the breakdown results, the key is to distinguish "structurally low" (for example, MoE routing itself contains a large amount of non-computational overhead and is hard to change) from "optimizable low" (for example, fusable Vector operators, redundant Casts, and inefficient memory access). Only the latter are optimization targets. The purpose of optimization is to eliminate this waste and improve real performance with appropriate means, not to inflate the MFU number itself.

## Performance Tuning Methods

MindSpeed MM provides a variety of performance tuning features, and you can choose an appropriate strategy based on the actual scenario. For detailed usage of each feature, see the corresponding document in the [Feature List](../../features/feature_list.md). Some features are bound to a specific training backend, so confirm the backend in use before applying them.

### Common Features of Both Backends

The following framework features are supported by both FSDP2 and Mcore.

#### Asynchronous Activation Offloading

Offload activations to the host side and use an asynchronous mechanism so that the copies are hidden by computation, reducing peak memory. For details, see [Async Activation Offload](../../features/memory/async_activation_offload.md)

#### ChunkLoss

Compute the loss in chunks along the sequence dimension to avoid keeping the logits of the entire sequence in memory at the same time, reducing peak memory. For details, see [ChunkLoss](../../features/optimization/chunkloss.md)

#### Recomputation

During backward propagation, some intermediate results from the forward pass are recomputed, trading computation time for memory space. This is suitable for scenarios where memory is tight and the MBS or sequence length needs to be increased.

#### Fused Operator

Multiple consecutive small operators are fused into a single large operator (such as RotaryMul, RmsNorm, and SwiGlu) to reduce kernel launch overhead and memory bandwidth consumption. The use of fused operators is automatically applied by the framework, requiring no manual configuration by users.

#### CPU Core Binding

TorchNPU provides the `CPU_AFFINITY_CONF` environment variable to control CPU-side core binding and reduce context switches and cross-NUMA latency: `0` or unset means disabled; `1` enables coarse-grained core binding; `2` enables fine-grained core binding. After core binding is enabled, you can optionally define custom binding ranges for specific NPU devices, for example `export CPU_AFFINITY_CONF=1,npu0:192-215,npu1:216-239,npu2:0-23,npu3:24-47`. For details, see [Environment Variable Reference](https://www.hiascend.com/document/detail/en/Pytorch/730/comref/EnvVariables/docs/en/environment_variable_reference/CPU_AFFINITY_CONF.md).

#### Scheduling Dispatch Optimization

When host-side operator dispatch becomes a bottleneck (large Free segments in profiling with no tasks executing on the CPU side), the following TorchNPU environment variables can be tried.

| Environment Variable | Purpose | Recommended Value |
|----------|------|--------|
| `TASK_QUEUE_ENABLE` | Controls the operator dispatch queue optimization level | `2` (enable level-2 optimization) |
| `ACLNN_CACHE_LIMIT` | Upper limit on the number of cached operators | `100000` |
| `PYTORCH_NPU_ALLOC_CONF` | Controls the cache allocator behavior | `expandable_segments:True` (enable virtual memory) |
| `ASCEND_LAUNCH_BLOCKING` | Whether to force synchronous operator dispatch (for debugging) | Use `0` or leave unset for normal training; temporarily set to `1` when locating the faulty operator or obtaining an accurate stack trace (this synchronously blocks dispatch and slows down training, and should be reverted after debugging) |

#### Python GC Issues

Python's garbage collection (GC) mechanism may cause training performance jitter. When periodic performance degradation is observed and profiling shows large segments of Free, you can try adjusting the GC threshold with `gc.set_threshold(700, 10, 10000)`, proactively calling `gc.collect()` after saving checkpoints, and in extreme scenarios, attempting `gc.disable()` while being mindful of the memory leak risk.

### FSDP2 Backend Features

The following methods apply only to the pluggable FSDP2 backend (entry: `mindspeed_mm/fsdp/train/trainer.py`).

#### Device Memory Optimization

Device memory is a key constraint on training throughput. Freeing device memory allows larger data scales (MBS or sequence length), thereby achieving better compute-communication overlap. The following methods can all be used to reduce peak memory usage.

- **ChunkMBS**: Combines recomputation and async activation offload to perform fine-grained chunking along the batch dimension, so that the forward and backward computation of multiple micro-chunks can be completed after a single parameter unshard, reducing the number of communication operations and improving communication-computation overlap. On the Qwen3.5 35B model, the measured end-to-end performance gain is about 5%. For details, see [ChunkMBS](../../features/optimization/chunkmbs.md). Using ChunkMBS requires enabling both recomputation and async activation offload at the same time, and its `apply_modules` must be covered by those of the two features.
- **async_activation_offload (async activation offload)**: Enabled in FSDP2 through `enable_activation_offload`; for the principle and details, see [Async Activation Offload](../../features/memory/async_activation_offload.md).

#### Host Memory Optimization

- **Meta Device initialization**: When disabled, multiple ranks consume a large amount of host memory; when enabled, an empty structure is first constructed on the meta device, and each rank initializes only its own shard, thereby saving host memory. To enable this feature, set `init_model_with_meta_device: true` in the YAML configuration.

#### Communication and Computation Overlap

- **Sequence parallelism**: Ulysses context parallelism, configured through `parallel.ulysses_parallel_size` in the training YAML; for the principle, see [Unaligned Ulysses CP](../../features/parallel/unaligned_ulysses_cp.md)
- **Prefetch Configuration**: Communication and computation overlap is achieved by initiating AllGather in advance, controlled by `num_to_forward_prefetch` (number of forward prefetch layers; default: 0) and `num_to_backward_prefetch` (number of backward prefetch layers; default: 1) (both defined in `mindspeed_mm/fsdp/params/parallel_args.py`). Two points require attention:
  - When Expert Parallelism (EP) is enabled, the AllGather of forward prefetch may contend for bandwidth with the All2All of the EP domain. Whether the number of prefetch layers is reasonable should be checked.
  - The prefetch order is determined by the configuration order of `fsdp_plan.apply_modules`, and **must be consistent with the forward execution order of the model**; otherwise, the wrong layers will be prefetched and the overlap will be ineffective.
- **reshard_after_forward configuration**: Controls whether FSDP2 reshard parameters after forward computation completes. `true` (default value, ZeRO3 mode): releases the gathered parameters immediately after forward completes, saving memory, but the backward pass requires another AllGather; `false` (ZeRO2 mode): retains the parameters after forward completes, saving one AllGather communication in the backward pass, while memory usage increases accordingly (the increment varies with model size, selected modules, and number of cards). In scenarios where ViT module communication volume is small and memory is sufficient, the ViT portion can be set to `reshard_after_forward: false` to reduce backward communication overhead. The related parameter definition is located in `mindspeed_mm/fsdp/params/parallel_args.py`.

#### Other Optimizations

- **Recomputation (recompute)**: Enabled through the `recompute` parameter, with `recompute_plan.apply_modules` specifying the modules to recompute (module path matching is used, supporting exact paths, wildcards, and regular expressions, down to the instance level). For details, see [FSDP2 Migration Guide](../development/fsdp2_model_migration_guide.md).
- **fully_shard granularity optimization**: By default, FSDP2 performs fully_shard separately for each block, resulting in multiple small parameter communications. For submodules with a large number of parameters (such as MoE experts), you can use `parallel.fsdp_plan.apply_modules` to specify separate sharding for them, reducing the number of communication operator invocations. For details, see [FSDP2 Migration Guide](../development/fsdp2_model_migration_guide.md).
- **Cast optimization**: The model may contain unnecessary precision conversion (Cast) operations, mainly involving: (1) RMSNorm already uses FP32 high-precision implementation internally, so no additional conversion to FP32 is needed externally; (2) the TorchNPU plugin has fixed the related bug of MoE routing (Unpermute), and the input now supports the combination where output is BF16 and `routing_weight` is FP32, so no additional Cast is needed. On the Qwen3.5 MoE model, ablating the Cast at RMSNorm + Unpermute yields about 5% performance gain; ablating the Cast at RMSNorm + Unpermute + RMSNormGated yields about 10% performance gain. The gain depends on the model structure, and precision alignment should be verified before making changes.

### Mcore (Megatron) Backend Features

- **Heterogeneous parallelism**: Supports combinations of parallel strategies across different dimensions. For details, see [Heterogeneous Parallelism](../../features/parallel/hetero_parallel.md)
- **Sequence parallelism**: Reduces the computation per device by partitioning the sequence dimension, supporting algorithms such as Ulysses, RingAttention, and USP. For details, see [DiT USP](../../features/parallel/dit_usp.md) and [DiT Ring Attention](../../features/parallel/dit_ring_attention.md)
- **Fused operators and communication hiding**: Fused operators such as rms_norm/swiglu/flash attention, together with Megatron weight update communication hiding. For details, see [Migration Tuning Guide](../development/model-migration.md#performance-tuning)
