# Self-Forcing

<p align="left">
</p>

- [Self-Forcing](#self-forcing)
  - [Model Overview](#model-overview)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Setup](#environment-setup)
  - [Training](#training)
  - [Inference](#inference)
  - [Environment Variable Declaration](#environment-variable-declaration)
- [Reference](#reference)
  - [Public Address Statement](#public-address-statement)

## Model Overview

The core technical approach of Self-Forcing is to abandon the traditional "teacher forcing" method when training autoregressive video diffusion models. Instead, it enables the model to predict the next frame based on its own previously generated, noisy frames, thereby simulating the real inference generation process and achieving complete autoregressive self-unfolding training. Its key innovation lies in fundamentally solving the "exposure bias" problem by forcing the model to learn how to correct from its own imperfectly generated context and introducing a holistic distribution matching loss to optimize the global quality of the entire generated sequence, thus significantly improving generation consistency and stability. The primary application is to achieve high-quality, high-frame-rate, and temporally consistent real-time video generation.

## Version Description

### Reference Implementation

  ```shell
  url=https://github.com/guandeh17/Self-Forcing
  commit_id=33593df3e81fa3ec10239271dd2c100facac6de1
  ```

### Changelog

2025.11.12: Initial release of Self-Forcing

## Environment Setup

It is recommended to use the matching environment version for model development.

For Ascend basic software installation, please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md).

> Python 3.10 and torch and torch_npu 2.7.1 are recommended.

1. Execute the following commands in the working directory.

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    cd MindSpeed-MM
    git clone https://github.com/guandeh17/Self-Forcing.git
    bash examples/self_forcing/replace_npu_patch.sh
    cd Self-Forcing
    ```

2. Install dependencies.

    ```shell
    pip install -r requirements.txt
    ```

3. Download weights.

    ```shell
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
    huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
    huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
    huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
    ```

## Training

1. Start training.

    ```shell
   torchrun --nnodes=1 --nproc_per_node=8 \
      train.py \
      --config_path configs/self_forcing_dmd.yaml \
      --logdir logs/self_forcing_dmd \
     --disable-wandb
   ```

## Inference

  1. Start inference.

      ```shell
      python inference.py \
          --config_path configs/self_forcing_dmd.yaml \
          --output_folder videos/self_forcing_dmd \
          --checkpoint_path checkpoints/self_forcing_dmd.pt \
          --data_path prompts/MovieGenVideoBench_extended.txt \
          --use_ema
      ```

## Environment Variable Declaration

| Environment Variable                          | Description                                                                 | Value Description                                                                                                               |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | Specifies whether to enable log printing.                                                          | `0`: Disable.<br>`1`: Enable.                                                                   |
| `ASCEND_GLOBAL_LOG_LEVEL`     | Sets the log level for application logs and the log level for each module; only supports debug logs.                             | `0`: DEBUG level<br>`1`: INFO level<br>`2`: WARNING level<br>`3`: ERROR level<br>`4`: NULL level; no log output |
| `TASK_QUEUE_ENABLE`           | Controls the level of `task_queue` operator dispatch queue optimization.                                    | `0`: Disable.<br>`1`: Enable Level 1 optimization.<br>`2`: Enable Level 2 optimization.                                              |
| `COMBINED_ENABLE`             | Sets the combined flag. Set to `0` to disable this feature; set to `1` to enable, used for optimizing non-contiguous two-operator combination.| `0`: Disable.<br>`1`: Enable.                                                                           |
| `CPU_AFFINITY_CONF`           | Controls the processor affinity of CPU-side operator tasks, i.e., sets task core binding.                                    | Set to `0` or not set: Indicates core binding is not enabled.<br>`1`: Indicates coarse-grained core binding is enabled.<br>`2`: Indicates fine-grained core binding is enabled.                                     |
| `HCCL_CONNECT_TIMEOUT`        | Limits the timeout waiting period for socket connection establishment between different devices.                                  | Must be configured as an integer in the value range `[120,7200]` (unit:s). The default value is `120`.                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | Controls the behavior of the cache allocator.                                                          | `expandable_segments:<value>`: Enables expandable segments of the memory pool, i.e., virtual memory characteristics.                                            |
| `HCCL_EXEC_TIMEOUT`           | Controls the synchronization wait time during execution between devices. Within this configured time, each device process waits for other devices to perform communication synchronization.         | Must be configured as an integer in the value range `[68,17340]` (unit: s). The default value is `1800`.                                                    |
| `ACLNN_CACHE_LIMIT`           | Configures the number of operator information entries cached on the host side by the single-operator execution API.                                  | Must be configured as an integer in the value range `[1, 10,000,000]`. The default value is `10000`.                                                    |
| `TOKENIZERS_PARALLELISM`      | Controls the behavior of the tokenizer in Hugging Face's transformers library in a multi-threading environment    | `False`: Disable parallel tokenization.<br>`True`: Enable parallel tokenization.                                               |
| `OMP_NUM_THREADS`             | Sets the number of threads used during execution.    |      Must be configured as an integer.                                                  |

# Reference

## Public Address Statement

For the public URLs involved in the code, refer to [Public Address Statement](../../docs/en/public_address_statement.md).
