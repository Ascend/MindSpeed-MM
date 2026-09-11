# FSDP2

## Background and Challenges

PyTorch's Fully Sharded Data Parallelism (FSDP) aims to provide a high-performance implementation in eager mode with communication bucketing and communication/computation overlap. The API represents communication buckets by flattening and concatenating a set of parameters into a `FlatParameter`. However, this `FlatParameter` design makes it difficult to apply differentiated operations (such as parameter freezing or precision conversion) to individual parameters within a bucket, compromising compositional flexibility. It also complicates internal implementation (e.g., the state dictionary logic spans thousands of lines of code and requires additional communication).

## Solution

To address the above limitations, FSDP2 removes the `FlatParameter` and instead uses `DTensor` sharded along dimension 0 to represent sharded parameters. This enables convenient operations on individual parameters, communication-free sharded state dictionaries, and a simplified initialization flow.

MindSpeed MM provides two FSDP2 usage paths:

- **Native FSDP2 (Recommended)**: Runs with a dedicated training entry point and a single YAML configuration file, without relying on Megatron command-line arguments. This is the recommended approach for new models.
- **Megatron-based FSDP2 (not recommended, to be deprecated)**: Reuses the Megatron training entry point, enabled via command-line switches. This path is intended only for legacy models during a transitional period. It will be deprecated and no longer maintained in the future. For new models, please use native FSDP2 exclusively.

## Native FSDP2 (Recommended)

### How to Use

The training entry point for native FSDP2 is `mindspeed_mm/fsdp/train/trainer.py` driven by a single YAML configuration file. The launch script must first set  `export NON_MEGATRON=true`, then use `torchrun` to start the process, passing the configuration file path as the sole argument to the entry script:

```shell
export NON_MEGATRON=true

torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    ${config_path}
```

`config_path` points to the model's YAML configuration file. See `examples/qwen3_5/finetune_qwen3_5_4B.sh` and  `examples/qwen3_5/qwen3_5_4B_config.yaml` for reference.

The configuration file follows a six-section structure, with each section responsible for the following:

| Section | Purpose |
| --- | --- |
| `parallel` | Parallelism and sharding strategies (FSDP sharding, tensor parallelism, sequence parallelism, expert parallelism) |
| `model` | Model source, attention implementation, fusion operators, etc. |
| `data` | Dataset, preprocessing, and DataLoader |
| `features` | Loss, recompute, activation offload, Chunk Loss, and other optimization features |
| `training` | Optimizer, learning rate, iteration steps, weight loading/saving, etc. |
| `tools` | Profiling, memory analysis, and other tools |

For details on the fields in each configuration section, refer to the example configuration  `examples/qwen3_5/qwen3_5_4B_config.yaml` and [FSDP2 Developer Migration Guide](fsdp2_developer_migration_guide.md).

### Weight Conversion

Native FSDP2 saves weights in DCP format. When initializing the model with meta device (`training.init_model_with_meta_device: true`), DCP weights must be loaded. You can first use `mm-convert` to convert Hugging Face weights to DCP, and point `training.load` to the conversion output (the parent directory of the `release` folder):

```shell
mm-convert GenericDCPConverter hf_to_dcp \
    --hf_dir ckpt/hf_path/xxx \
    --dcp_dir ckpt/dcp_path/xxx
```

For descriptions of exporting Hugging Face weights (`dcp_to_hf`), full parameters, and model-specific converters, see [Weight Conversion](../pytorch/weight_conversion.md).

### Important Notes

1. The launch script must set `export NON_MEGATRON=true`; otherwise, the operator adaptations required by native FSDP2 will not be enabled.
2. Native FSDP2 and Megatron-based FSDP2 use different configuration systems: the latter uses Megatron command-line arguments plus an additional `fsdp2_config.yaml`, while the former uses a six-section YAML. The fields from the two systems are not interchangeable.

## Megatron-Based FSDP2 (Not Recommended, to be Deprecated)

> Megatron-based FSDP2 is no longer recommended and will be deprecated in the future. For new models, please use native FSDP2 and avoid this path.

This mode reuses the Megatron training entry point (`pretrain_*.py`), enabled via the command-line switch `--use-torch-fsdp2`, with sharding-related parameters provided in a separate `fsdp2_config.yaml`.

### How to Use

To use this feature, pass the following command-line arguments to the entry script:

```shell
export CUDA_DEVICE_MAX_CONNECTIONS=2 # Cannot be set to 1
--use-torch-fsdp2 \
--fsdp2-config-path ./fsdp2_config.yaml \
--ckpt-format torch_dcp \
--untie-embeddings-and-output-weights \
# Distributed optimizer cannot be enabled
```

#### Parameter Details

The configuration items of `fsdp2_config.yaml` are as follows:

- `sharding_size`
  - Description: Controls the model parallelism size for tensor sharding; defaulted to `1`.
  - Values:
    - `"auto"`: Automatically determines the optimal sharding size based on the number of available devices.
    - Integer: Specifies the size of the sharding group.

- `sub_modules_to_wrap`
  - Description: Specifies the sub-modules to use FSDP for parameter sharding.
  - Configuration format
    - Use the full module path separated by dots.
    - Start from the first-level submodule of the model (excluding the outermost model variable name)
    - Support exact paths and pattern matching.
  - Examples:
    - `model.model.deepstack_merger_list.{*}`
    - `model.model.language_model.layers.{0-20,22-40}`
    - `model.lm_head`

- `ignored_modules`
  - Description: List of module classes to exclude from FSDP management
  - Configuration format: Same as `sub_modules_to_wrap`.

- `recompute_modules`
  - Description: Configures activation recomputation (trading compute for memory).
  - Configuration format: Same as `sub_modules_to_wrap`.
  - Constraint: Conflicts with Megatron's full recomputation feature; must disable Megatron recomputation when using this.

- `use_reentrant`
  - Description: Selects the checkpointing implementation type (reentrant or not); defaulted to `True`.
  - Values: `True` or `False`

- `reshard_after_forward`
  - Description: Controls whether to reshard parameters after forward pass.
  - Values:
    - `True`: Reshard immediately after forward; all-gather again in backward (saves memory).
    - `False`: Keep gathered parameters after forward; no all-gather in backward (saves communication but consumes more memory).

- `param_dtype`
  - Description: Data type for parameter storage and computation.
  - Values: `"bf16"`, `"fp16"`, `"fp32"`

- `reduce_dtype`
  - Description: Data type for gradient reduction operations.
  - Values: `"bf16"`, `"fp16"`, `"fp32"`

- `output_dtype`
  - Description: Data type for forward outputs.
  - Values: `"bf16"`, `"fp16"`, `"fp32"`

- `cast_forward_inputs`
  - Description: Controls automatic type conversion of forward propagation inputs
  - Values: `True` or `False`

- `num_to_forward_prefetch`
  - Description: Number of subsequent layers to prefetch parameters for during forward propagation

- `num_to_backward_prefetch`
  - Description: Number of subsequent layers to prefetch parameters during backward propagation

- `offload_to_cpu`
  - Description: Specifies whether to offload parameters, gradients, and optimizer states to CPU memory; defaults to `False`.
  - Value: `True` or `False`

- `pin_memory`
  - Description: Specifies whether to pin CPU memory to improve data transfer efficiency. This only takes effect when `offload_to_cpu` is enabled.
  - Value: `True` or `False`

#### Configuration Example

```shell
sharding_size: auto
sub_modules_to_wrap:
  - "text_decoder.output_layer"
  - "text_decoder.embedding"
  - "text_decoder.rotary_pos_emb"
  - "text_decoder.decoder.layers.{*}"
param_dtype: "bf16"
reduce_dtype: "fp32"
cast_forward_inputs: True
ignored_modules:
  - "image_encoder"
recompute_modules:
  - "text_decoder.decoder.layers.{*}"
num_to_forward_prefetch: 2
num_to_backward_prefetch: 2
offload_to_cpu: False
```

### Performance Impact

For Llama-7B, FSDP2 achieves higher MFU compared to FSDP1, reducing peak memory by 7% while maintaining the same loss curve.

### Notes

1. When enabling FSDP2 training, the distributed optimizer and its related configurations must be disabled.

2. When enabling FSDP2 training, the model weight save format `ckpt-format` only supports `torch_dist` or `torch_dcp`.

   - When configured as `torch_dist`, the model must implement the `sharded_state_dict()` method by inheriting from `MegatronModule` or through customization; at the same time, it must be ensured that the 0-dimension size of all weights in the model is greater than or equal to `sharding_size`.

   - When configured as `torch_dcp`, the model needs to implement the `state_dict_for_save_checkpoint()` method by inheriting from `MegatronModule` or through customization. The returned weight dictionary must be consistent with the return value of `model.state_dict()`.

3. When enabling FSDP2 training, disable recomputation-related configurations, including: `--recompute-granularity`, `--recompute-method`, and `--recompute-num-layers`, etc.

4. When setting `offload_to_cpu=True`, configure the communication group in the entry script as a dual backend, i.e.: `--distributed-backend npu:hccl,cpu:gloo`.

5. For training models with extremely large parameter counts, it is recommended to enable `--init-model-with-meta-device` and `--no-initialization` to effectively avoid memory overflow caused by loading the full model parameters at once, while significantly reducing the waiting time during the model initialization phase.

6. The mixed precision of FSDP2 is configured and takes effect in the YAML file. `--bf16` is no longer necessary and conflicts with resumable training. If enabled, it must be used together with `--no-save-optim` and `--no-load-optim`.
To align with the computation behavior of `--bf16`, we have added the `--downcast-to-bf16` option, which adds weight downcast during the weight loading phase to maintain computation consistency when `--bf16` is disabled.
Here, the FSDP2 mixed precision is to keep the precision of the loaded weights unchanged. It is recommended to use this default behavior to avoid precision loss.

7. If `--untie-embeddings-and-output-weights=True` is set, for models that originally use weight tying, this configuration will disable the weight tying mechanism. The current framework does not support this scenario, requiring users to manually copy the `lm_head` and `embeddings` during weight conversion. Please note that the model structure may change as a result.
