# FSDP2

## Background and Challenges

PyTorch's Fully Sharded Data Parallel (FSDP) aims to provide a high-performance implementation in eager mode with communication bucketing and communication/computation overlap. The API represents communication buckets by flattening and concatenating a set of parameters into a `FlatParameter`. However, this `FlatParameter` design makes it difficult to apply differentiated operations (such as parameter freezing or precision conversion) to individual parameters within a bucket, compromising compositional flexibility. It also complicates internal implementation (e.g., the state dictionary logic spans thousands of lines of code and requires additional communication).

## Solution

To address the above limitations, FSDP2 removes the `FlatParameter` and instead uses `DTensor` sharded along dimension 0 to represent sharded parameters. This enables convenient operations on individual parameters, communication-free sharded state dictionaries, and a simplified initialization flow.

## How to Use

In MindSpeed, the entry point for FSDP2 is a configuration file. By generating the configuration file and passing it as a command-line argument, you can use this feature.

```shell
export CUDA_DEVICE_MAX_CONNECTIONS=2 # Cannot be set to 1
--use-torch-fsdp2 \
--fsdp2-config-path ./fsdp2_config.yaml \
--ckpt-format torch_dcp \
--untie-embeddings-and-output-weights \
# Note that the distributed optimizer cannot be enabled
```

### Parameter Details

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
  - Description: Controls when to reshard parameters.
  - Values:
    - `True`: Reshards parameters immediately after the forward pass (saves memory, ZeRO3).
    - `False`: Keeps parameters gathered until backward propagation (better performance, ZeRO2).

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

### Configuration Example

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

## Effectives

For Llama-7B, FSDP2 achieves higher MFU compared to FSDP1, reducing peak memory by 7% while maintaining the same loss curve.

## Notes

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
