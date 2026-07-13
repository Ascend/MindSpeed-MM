# Common Options in MindSpeed MM Scripts

- [Common Options in MindSpeed MM Scripts](#common-options-in-mindspeed-mm-scripts)
  - [Parameter Annotations Under GPT\_ARGS](#parameter-annotations-under-gpt_args)
    - [General Parameters](#general-parameters)
    - [Memory Optimization](#memory-optimization)
      - [Recomputation](#recomputation)
      - [FSDP2](#fsdp2)
    - [Acceleration](#acceleration)
  - [Parameter Annotations Under MOE\_ARGS](#parameter-annotations-under-moe_args)
  - [Parameter Annotations Under OUTPUT\_ARGS](#parameter-annotations-under-output_args)
  - [Environment Variables](#environment-variables)

## Parameter Annotations Under GPT_ARGS

### General Parameters

| Parameter Name | Type/Value | Description |
|--------|----------|------|
| `--micro-batch-size` | Integer (from `${MBS}`) | The number of samples directly processed by a single GPU in one forward/backward pass should be adapted to the memory limit within a single NPU. This directly affects GPU memory capacity. |
| `--global-batch-size` | Integer (from `${GBS}`) | Total number of samples across all devices used for one model parameter update. |
| `--num-workers` | Non-negative integer | TNumber of subprocesses launched for the data loading part in PyTorch. Setting it too high consumes CPU resources, while setting it too low causes the model to wait too long for data loading. |
| `--seq_length` | Integer | Sequence length, representing the number of tokens in a single sample that the model can process at one time. Note that this function is disabled when `--variable-seq-lengths` is enabled. Sequence length determines the range of contextual information that can be captured; longer sequences can capture longer relations but significantly increase computational complexity and memory consumption. |
| `--normalization` | `RMSNorm` |Use RMSNorm. Recommended to be used with `--use-fused-rmsnorm`. |
| `--swiglu` | `store_true` | Use the SwiGLU activation function. Recommended to be used with `--use-fused-swiglu`. |
| `--lr-warmup-fraction` | Float (0 to 1) | Proportion of the learning rate's "warm-up" phase to the total number of steps. |
| `--weight-decay-exclude-modules` | List of strings | Excludes specific parameters from weight decay by configuring parameter name keywords (multiple allowed). See [this document](../features/parameter_lr_wd_tuning.md) for detailed information. |
| `--lr-scale-modules` | List of strings | Scales the learning rate for specific parameters by configuring parameter name keywords (multiple allowed). See [this document](../features/parameter_lr_wd_tuning.md) for detailed information. |
| `--clip-grad` | Float (defaulted to 1) | A non-zero value enables this feature that limits weights in the optimizer to prevent excessive loss fluctuation. |
| `--seed` | Integer | Random seed. |
| `--bf16` | `store_true` | Train using the torch.bfloat16 format, significantly reducing memory consumption. |
| `--load` | String | Model weight path. Fill in according to the instructions in each example. |
| `--variable-seq-lengths` | `store_true` | Enables variable sequence lengths. |
| `--calculate-per-sample-loss` | - | Calculates loss at the per-sample granularity. See [this document](../features/vlm_model_loss_calculate_type.md) for detailed information. |
| `--calculate-per-token-loss` | - | Calculates loss at the per-token granularity. See [Detailed Information](../features/vlm_model_loss_calculate_type.md). |
| `--ckpt-format` | `torch_dcp` | Uses DCP format when saving data. See [this document](../features/fsdp2.md) for detailed information. |
| `--init-model-with-meta-device` | - | Initializes the model using FSDP2's meta; currently only supported by the Qwen3VL model. For detailed usage, please refer to the specific model's README in the `examples` directory. |

### Memory Optimization

| Parameter Name | Type/Value | Description |
|--------|----------|------|
| `--tensor-model-parallel-size` | Non-zero integer (defaulted to 1, from `${TP}`) | Sets the tensor parallelism size. It shards model weights across multiple devices for computation, reducing memory usage per device but introducing additional communication overhead. |
| `--pipeline-model-parallel-size` | Non-zero integer (defaulted to 1, from `${PP}`) | Sets the pipeline parallelism size. It distributes model computation across multiple devices by stages, reducing memory usage per device but increasing communication time and potentially causing idle time (bubble) on some devices. |
| `--context-parallel-size` | Non-zero integer (defaulted to 1, from `${CP}`) | Sets the sequence parallelism size. It shards data along the sequence dimension. It is primarily used for long-sequence tasks to reduce memory usage per device, but introduces extra communication overhead that impacts performance. |
| `--context-parallel-algo` | string | Selects the CP algorithm, including `ulysses_cp_algo`, `hybrid_cp_algo`, and `megatron_cp_algo`. See [this document](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/ulysses-context-parallel.md) for detailed information. |
| `--expert-model-parallel-size` | Non-zero integer (defaulted to 1, from `${EP}`) | Sets the expert parallelism size in MoE networks. It distributes experts across different devices for computation. It is primarily used to address the issue where a single device's memory cannot hold all experts, but may cause imbalanced expert load and low computational efficiency. |
| `--use-distributed-optimizer` | `store_true` | Distributed optimizer. It shards the optimizer state across devices for independent computation and storage. Enabling this significantly reduces memory consumption and improves computational resource utilization. |

#### Recomputation

See [this document](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1//docs/en/features/recomputation.md) for detailed information.

| Parameter Name | Type/Value | Description |
|--------|----------|------|
| `--recompute-granularity` | `full` | Only `full` is supported to enable full recomputation. |
| `--recompute-method` | `block` or `uniform` | Configures the recomputation mode.<br>- `uniform`: Evenly divides transformer layers into groups, with the group size specified by `--recompute-num-layers`, storing inputs and activations per group.<br>- `block`: Applies recomputation to the first `--recompute-num-layers` transformer layers, skipping the remaining layers. |
| `--recompute-num-layers` | Integer | Configures the number of layers for recomputation. Its effect depends on the setting of `--recompute-method`. |

#### FSDP2

> **Note**: When FSDP2 is enabled, all Megatron sharding strategies and recomputation configurations must be disabled.

| Parameter Name | Type/Value | Description |
|--------|----------|------|
| `--fsdp2-config-path` | string | Path to the FSDP2 configuration file. |
| `--use-cpu-initialization` | - | Uses CPU for weight initialization; must be enabled. |

### Acceleration

| Parameter Name | Type/Value | Description |
|--------|----------|------|
| `--use-fused-swiglu` | - | Enables the relevant fusion operator; effective only when using SwiGLU. |
| `--use-fused-rmsnorm` | - | Enables the relevant fusion operator; effective only when using RMSNorm. |
| `--overlap-grad-reduce`/`--overlap-param-gather` | - | Overlaps communication for weight updates; effective only when `--use-distributed-optimizer` is enabled. See [this document](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1//docs/en/features/async-ddp-param-gather.md) for more details. |

## Parameter Annotations Under MOE_ARGS

| Parameter Name | Type/Value | Description |
|--------|----------|------|
| `--moe-token-dispatcher-type` | String (defaulted to `allgather`) | Selects the communication method for dispatching tokens in the MoE network. If expert parallelism is enabled, `alltoall` is recommended. |
| `--moe-permute-fusion` | - | Enables the permute and unpermute fusion operator to accelerate computation. |

## Parameter Annotations Under OUTPUT_ARGS

| Parameter Name | Type/Value | Description |
|--------|----------|------|
| `--save` | String (from `SAVE_PATH`) | Weight saving path.<br>**Note**: Weights are saved only when this parameter is configured. |
| `--ckpt-format` | `torch` or `torch_dcp` | Weight saving format. It is recommended to prioritize using [`torch_dcp`](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/fsdp2.md).<br><br>**Note**:<br>1. When FSDP2 is used for model training, only the `torch_dcp` configuration is supported.<br>2. Setting `--ckpt-format` to `torch_dcp` under `OUTPUT_ARGS` has the same effect as enabling `--ckpt-format torch_dcp` under `GPT_ARGS`; choose either one. |

## Environment Variables

Detailed explanations for all environment variables can be found by searching on the [Ascend official website](https://www.hiascend.com/document/detail/en/canncommercial/83RC1/maintenref/envvar/envref_07_0121.html). The following only shows those commonly used within the MindSpeed MM suite.

| Environment Initialization Script | Description |
|-----------------------------------------|--------------------------------------------------------------------|
| `source /usr/local/Ascend/cann/set_env.sh`| CANN installation path; must be configured |
| `source /usr/local/Ascend/nnal/atb/set_env.sh` | NNAL installation path |

| Environment Variable                                                                                                                                  | Description | Value Description |
|---------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `CUDA_DEVICE_MAX_CONNECTIONS`                                                                                                           | Controls the number of parallel device connections on the host in multi-GPU systems | Must be configured as an integer in the value range `[1, 32]`; set to `1` when sequence parallelism is enabled. |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | Specifies whether to enable log printing.                                                          | `0`: Disable.<br>`1`: Enable.                                                                   |
| `ASCEND_GLOBAL_LOG_LEVEL`     | Sets the log level for application logs and the log level for each module; only supports debug logs.                             | `0`: DEBUG level<br>`1`: INFO level<br>`2`: WARNING level<br>`3`: ERROR level<br>`4`: NULL level; no log output |
| `TASK_QUEUE_ENABLE`           | Controls the level of `task_queue` operator dispatch queue optimization.                                    | `0`: Disable.<br>`1`: Enable Level 1 optimization.<br>`2`: Enable Level 2 optimization.                                              |
| `COMBINED_ENABLE`             | Sets the combined flag. Set to `0` to disable this feature; set to `1` to enable, used for optimizing non-contiguous two-operator combination.| `0`: Disable.<br>`1`: Enable.                                                                           |
| `CPU_AFFINITY_CONF`           | Controls the processor affinity of CPU-side operator tasks, i.e., sets task core binding.                                    | Set to `0` or not set: Indicates core binding is not enabled.<br>`1`: Indicates coarse-grained core binding is enabled.<br>`2`: Indicates fine-grained core binding is enabled.                                     |
| `HCCL_CONNECT_TIMEOUT`        | Limits the timeout waiting period for socket connection establishment between different devices.                                  | Must be configured as an integer in the value range `[120,7200]` (unit:s). The default value is `120`.                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | Controls the behavior of the cache allocator.                                                          | `expandable_segments:<value>`: Enables expandable segments of the memory pool, i.e., virtual memory characteristics.                                            |
| `HCCL_EXEC_TIMEOUT`           | Controls the synchronization wait time during execution between devices. Within this configured time, each device process waits for other devices to perform communication synchronization.         | Must be configured as an integer in the value range `[68,17340]` (unit: s). The default value is `1800`.                                                    |
| `ACLNN_CACHE_LIMIT`           | Configures the number of operator information entries cached on the host side by the single-operator execution API.                                  | Must be configured as an integer in the value range `[1, 10,000,000]`. The default value is `10000`.                                                    |
| `TOKENIZERS_PARALLELISM`      | Controls the behavior of the tokenizer in Hugging Face's transformers library in a multi-threading environment    | `False`: Disable parallel tokenization.<br>`True`: Enable parallel tokenization.                                                            |
| `MULTI_STREAM_MEMORY_REUSE`   | Configures whether multi-stream memory reuse is enabled. | `0`: Disable multi-stream memory reuse.<br>`1`: Enable multi-stream memory reuse.                                                               |
| `NPU_ASD_ENABLE`   | Controls whether to enable the feature value detection function of Ascend Extension for PyTorch | Set to `0` or not set: Disable feature value detection.<br>`1`: Enable feature value detection and print only abnormal logs, without alarms.<br>`2`: Enable feature value detection and print alarms.<br>`3`: Enable feature value detection and print alarms, as well as process data in device-side info level logs. |
| `ASCEND_LAUNCH_BLOCKING`   | Controls whether to enable synchronous mode during operator execution. | `0`: Execute operators asynchronously.<br>`1`: Force operators to run in synchronous mode.                                                               |
| `NPUS_PER_NODE`               | Configures the number of NPUs used on a compute node.                                                  | Integer value (e.g., `1`, `8`, etc.)                                                                            |
| `ASCEND_RT_VISIBLE_DEVICES`   | Specifies the device(s) visible to the current process, supporting specifying one or multiple device IDs at a time. | A combination of numbers, with multiple device IDs separated by commas. |
