# Diffusers

<p align="left">
</p>

- [Diffusers](#diffusers)
- [FLUX-Kontext](#flux-kontext)
  - [Model Introduction](#model-introduction)
  - [Version Notes](#version-notes)
    - [Reference Implementation](#reference-implementation)
    - [Change Log](#change-log)
  - [Fine-tuning](#fine-tuning)
    - [Environment Setup](#environment-setup)
  - [Fine-tuning](#fine-tuning-1)
    - [Performance](#performance)
      - [Throughput](#throughput)
  - [Inference](#inference)
    - [Environment Setup and Execution](#environment-setup-and-execution)
    - [Performance](#performance-1)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Reference](#reference)
    - [Public Address Statement](#public-address-statement)

<a id="jump1"></a>

# FLUX-Kontext

## Model Introduction

[FLUX.1-Kontext-dev](https://bfl.ai/models/flux-kontext) is a generative model based on FLUX, representing the current state-of-the-art in context-aware image generation and editing technology. It can combine text and images to achieve precise and coherent generation results.

## Version Notes

### Reference Implementation

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=c222570a9b47901266fecf34222f540870c3bb1b
  ```

### Change Log

2025.09.08: Initial release of FLUX-Kontext

## Fine-tuning

### Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md)

1. Install the software and driver.

    ```bash
    # Install torch and torch_npu by following the installation guide mentioned above.

    # Modify the environment variable paths in the shell script to the actual paths. Example:
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. Clone the repository to the local server.

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    ```

3. Set the model.

3.1 Download [FLUX-Kontext Reference Implementation](https://github.com/huggingface/diffusers). Execute the following commands in the model root directory to install the dependencies required by the corresponding PyTorch version of the model.

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout c222570
    cp -r ../MindSpeed-MM/examples/diffusers/flux-kontext/* ./examples/dreambooth
    cp ../MindSpeed-MM/sources/images/flux_cat.png ./examples/dreambooth
    ```

Main code path:

    ```shell
    code_path=examples/dreambooth/
    ```

3.2 Install the required dependency libraries.

    ```shell
    pip install -e .
    pip install -r examples/dreambooth/mm_requirements_kontext.txt # Install the corresponding dependencies.
    ```

<a id="jump2"></a>

## Fine-tuning

1. Prepare the fine-tuning dataset.

    - Download and decompress the [kontext-community/relighting](https://huggingface.co/datasets/kontext-community/relighting) dataset, and set the `dataset_name` parameter in the following startup shell script to the absolute path of the local dataset.

    ```shell
    dataset_name="kontext-community/relighting" # Dataset path
    ```

   - The format of the `kontext-community/relighting` dataset is as follows:

    ```shell
    relighting
    ├── .gitattributes
    ├── README.md
    └── data
          └── train-00000-of-00001.parquet
    ```

2. Configure the FLUX-Kontext fine-tuning script.

    When connected to the internet, the fine-tuning model can be downloaded using the following steps. Without network access, you can visit the Hugging Face official website to manually download [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).

    ```bash
    export model_name="black-forest-labs/FLUX.1-Kontext-dev" # Pre-trained model path
    ```

    After obtaining the corresponding fine-tuning model, set the `model_name` parameter in the following shell script to the absolute path of the local pre-trained model.

    ```shell
    model_name="black-forest-labs/FLUX.1-Kontext-dev" # Pre-trained model path
    batch_size=2
    max_train_steps=5000
    mixed_precision="bf16" # Mixed precision
    resolution=1024
    config_file="bf16_accelerate_config.yaml"

    # accelerate launch --config_file ${config_file} \ directory
    --dataloader_num_workers=0 \ # Please adjust num workers based on system configuration and data size
    ```

3. modify code files.

    1. Open the `train_dreambooth_lora_flux_kontext.py` file.

        ```shell
        cd examples/dreambooth/ # Navigate from the diffusers directory to the dreambooth directory.
        vim train_dreambooth_lora_flux_kontext.py # Open the Python file.
        ```

        - Add code near line 71 in the import section, above the line `if is_wandb_available():`.

        ```python
        # Add the code near line 71 of train_dreambooth_lora_flux_kontext.py.
        from transformer_patches import apply_patches
        apply_patches()

        if is_wandb_available(): # Original code
          import wandb
        ```

        - Modify `batch_sample` before `train_dataloader`, change `drop_last` of the dataset to `False` around line 1645.

        ```python
        # Modify drop_last to False:
        batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=False)
        # batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=True) # Original code
        ```

    2. (Optional) On Ubuntu systems, add `accelerator.print("")` around line 1701.

        ```python
        if global_step >= args.max_train_steps: # Original code
          break
        accelerator.print("") # Add
        ```

    3. (Optional) Multi-node execution

        Modify the `config` file:

        ```bash
        vim bf16_accelerate_config.yaml
        ```

        Uncomment `deepspeed_multinode_launcher`, `main_process_ip`, and `main_process_port` in the file to enable them.

        ```shell
            zero_stage: 2
          #  deepspeed_multinode_launcher: standard
          # main_process_ip: localhost  # Master node IP
          # main_process_port: 6000     # Master node port
          machine_rank: 0             # Current machine rank
          num_machines: 1             # Total number of machines
          num_processes: 8            # Total number of cards
        ```

        For two-machine execution:
        - Set `main_process_ip` and `main_process_port` in the YAML files of both machines to the same master node and port.
        - Set `machine_rank: 0` on one node and `machine_rank: 1` on the other.
        - Set `num_machines: 2` and `num_processes: 16` on both machines.

4. Start the FLUX-Kontext LoRA fine-tuning script.

    The `train_dreambooth_lora_flux_kontext` script is provided for this task, supporting multi-card training.

    Launch the fine-tuning script:

    ```shell
    bash finetune_kontext_dreambooth_lora_deepspeed_bf16.sh
    ```

### Performance

#### Throughput

Performance comparison of FLUX on the **Ascend chip** and **benchmark chip**:

| Chip | Chip Count | Task | FPS | batch_size | AMP_Type | Resolution | DeepSpeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | FLUX-Kontext LoRA fine-tuning | 1.97 | 2 | bf16 | 1024 | ✔ |
| Competitor A | 8p | FLUX-Kontext LoRA fine-tuning | 2.00 | 2 | bf16 | 1024 | ✔ |

## Inference

### Environment Setup and Execution

See the "fine-tuning" section.

```shell
cd examples/dreambooth/ # Navigate from the diffusers directory to the dreambooth directory
```

FLUX-Kontext model inference:

```shell
vim infer_kontext_text2img_bf16.py # Enter the Python file for running inference
```

  1. Modify the path.

      ```python
      MODEL_PATH = "black-forest-labs/FLUX.1-Kontext-dev"  # FLUX model path
      ```

  2. Run the code.

      ```shell
      python infer_kontext_text2img_bf16.py
      ```

LoRA fine-tuned FLUX-Kontext model inference:

  ```shell
  vim infer_kontext_text2img_lora_bf16.py
  ```

1. Modify the path.

      ```python
      MODEL_PATH = "black-forest-labs/FLUX.1-Kontext-dev"  # Flux Model Path
      LORA_WEIGHTS = "./logs/pytorch_lora_weights.safetensors"  # LoRA Weight Path
      ```

2. Run the code.

      ```shell
      python infer_kontext_text2img_lora_bf16.py
      ```

<a id="jump3"></a>

### Performance

| Chip | Chip Count |     Task     |  E2E (it/s)  |  AMP_Type | Torch_Version |
|:---:|:---:|:----------:|:-----:|:---:|:---:|
| Atlas 900 A2 PODc |8p |  LoRA Text-to-Image  | 1.04 | bf16 | 2.7.1 |
| Competitor A | 8p |  LoRA Text-to-Image  | 1.04 | bf16 | 2.7.1 |

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

## Reference

### Public Address Statement

For public addresses referenced in the code, see [Public Address Statement](../../../docs/en/public_address_statement.md).
