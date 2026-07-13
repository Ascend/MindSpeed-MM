# Diffusers

<p align="left">
</p>

- [Diffusers](#diffusers)
- [FLUX2](#flux2)
  - [Model Introduction](#model-introduction)
  - [Environment Setup](#environment-setup)
  - [Fine-tuning T2I](#fine-tuning-t2i)
    - [Prerequisites](#prerequisites)
    - [Performance](#performance)
      - [Throughput](#throughput)
  - [Fine-tuning Img2Img](#fine-tuning-img2img)
    - [Prerequisites](#prerequisites-1)
    - [Performance](#performance-1)
      - [Throughput](#throughput-1)
  - [Inference](#inference)
    - [Environment Setup and Execution](#environment-setup-and-execution)
    - [Text-to-Image Inference](#text-to-image-inference)
    - [I2I Inference](#i2i-inference)
    - [Performance](#performance-2)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Reference](#reference)
    - [Public Address Statement](#public-address-statement)

<a id="jump1"></a>

# FLUX2

## Model Introduction

[FLUX.2 dev](https://blackforestlabs.ai/announcing-black-forest-labs/) is a generative model based on Rectified Flow Transformers.

- Reference implementation

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=29a930a
  ```

## Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md).

1. Install the software and driver.

    ```bash
    # Install torch and torch_npu by following the installation guide mentioned above.

    # Modify the environment variable paths in the shell script to the actual paths. Example:
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. Clone the repository to the local server.

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    cd MindSpeed-MM
    ```

3. Set the model.

    3.1 Download the [FLUX2 reference implementation](https://github.com/huggingface/diffusers) and execute the following commands in the model's root directory to install the dependencies required for the corresponding PyTorch version of the model.

    Main code path:

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2 Install dependencies and enter the code path.

    ```shell
    bash examples/diffusers/flux2/install.sh
    cd ../diffusers/examples/dreambooth
    ```

<a id="jump2"></a>

## Fine-tuning T2I

<a id="jump3"></a>

### Prerequisites

1. Prepare the fine-tuning dataset.

    - Download and decompress the [3d-icon](https://huggingface.co/datasets/linoyts/3d_icon) dataset and set the `dataset_name` parameter in the `finetune_t2i_flux2_dreambooth_lora_fsdp_bf16.sh` script to the absolute path of the local dataset.

    Open the script:

    ```shell
    vim finetune_t2i_flux2_dreambooth_lora_fsdp_bf16.sh
    ```

    ```shell
    dataset_name="linoyts/3d_icon" # Dataset path
    ```

   - The `3d_icon dataset` format is as follows:

    ```shell
    3d_icon
    ├── metadata.jsonl
    ├── README.MD
    ├── gitattributes
    ├── 00.jpg
    ├── 01.jpg
    ├── ...jpg
    └── 22.jpg
    ```

    > **Note**:
    >The training script for this dataset is provided only for reference.
    >

    - If using your own fine-tuning dataset, modify `dataset_name` in the shell script:

    ```shell
    dataset_name="/path/customized_datasets" # Dataset path
    ```

    In the shell script's `accelerate launch` directory (around line 58), modify `dataset_name=$dataset_name`, change `instance_prompt` to a prompt matching your dataset, and modify `caption_column` to the matching name in your dataset. If using the 3dicon dataset, no modification is needed.

    ```shell
    # Example
    accelerate launch --config_file ${config_file} \
      ./train_dreambooth_lora_flux2.py \
      --pretrained_model_name_or_path=$model_name  \
      --dataset_name=$dataset_name \
      --caption_column="prompt" \
      --instance_prompt="a prompt that is suitable for your own dataset" \
    ```

2. Configure the FLUX2 fine-tuning script.

    When connected to the internet, the fine-tuning model can be downloaded via the following steps. Without network access, you can visit the Hugging Face official website to download [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev).

    After obtaining the model, set the `model_name` parameter in the fine-tuning script to the absolute path of the local pre-trained model.

    Open the script:

    ```shell
    vim finetune_t2i_flux2_dreambooth_lora_fsdp_bf16.sh
    ```

    ```shell
    model_name="black-forest-labs/FLUX.2-dev" # Pre-trained model path
    batch_size=1
    max_train_steps=5000
    mixed_precision="bf16"
    resolution=1024
    gradient_accumulation_steps=1
    config_file="${mixed_precision}_accelerate_config.yaml"

    # accelerate launch --config_file ${config_file} \ directory
    --dataloader_num_workers=0 \ # Adjust num_workers based on system configuration and data size.
    ```

3. Modify code files.

    ```shell
    vim train_dreambooth_lora_flux2.py # Navigate to the Python file.
    ```

    1. (Optional) On Ubuntu systems, add `accelerator.print("")` near line 1879.

        ```python
        if global_step >= args.max_train_steps: # Original Code
          break
        accelerator.print("") # Add
        ```

    2. (Optional) If the machine is not connected to the internet, you need to modify `save_model_card`:
        Delete `save_model_card` or place it under the `args.push_to_hub` directory:

        ```python
        elif args.bnb_quantization_config_path:
            quant_training = "BitsandBytes" # Original Code
        if args.push_to_hub:
            save_model_card(
                (args.hub_model_id or Path(args.output_dir).name) if not args.push_to_hub else repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                validation_prompt=validation_prompt,
                repo_folder=args.output_dir,
                quant_training=quant_training,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            ) # Original Code

        ```

    3. (Optional) Multi-node execution

         Modify the `config` file:

        ```bash
        vim bf16_accelerate_config.yaml
        ```

        Uncomment `main_process_ip` and `main_process_port` in the file for use.

        ```shell
          main_process_ip: localhost  # Main node IP
          main_process_port: 6000     # Main node port
          machine_rank: 0             # Rank of the current machine
          num_machines: 1             # Total number of machines
          num_processes: 8            # Total number of cards
        ```

4. Start Flux2 T2I fine-tuning script.

    The `dreambooth_lora_flux2_t2i` script is provided for this task, supporting multi-card training.

    Start the Flux2 T2I dreambooth_lora fine-tuning script:

    ```shell
    bash finetune_t2i_flux2_dreambooth_lora_fsdp_bf16.sh
    ```

<a id="jump4"></a>

### Performance

#### Throughput

Performance comparison of FLUX on the **Ascend chip** and **benchmark chip**:

| Chip | Chip Count |     Task     |  FPS  | batch_size | Resolution | AMP_Type | Torch_Version | FSDP2 |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Full-parameter fine-tuning for FLUX |  1.28  | 1 | 1024 | bf16 | 2.7.1 | ✔ |
| Competitor A | 8p | Full-parameter fine-tuning for FLUX  |  1.24 | 1 | 1024 | bf16 | 2.7.1 | ✔ |

<a id="jump5"></a>

## Fine-tuning Img2Img

<a id="jump6"></a>

### Prerequisites

1. Prepare the fine-tuning dataset.

    - Download and decompress the [kontext-community/relighting](https://huggingface.co/datasets/kontext-community/relighting) dataset, and set the `dataset_name` parameter to the absolute path of the local dataset in the following startup shell script.

    ```shell
    vim finetune_i2i_flux2_dreambooth_lora_fsdp_bf16.sh
    ```

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

2. Configure the FLUX2 fine-tuning script.

    When connected to the internet, the fine-tuning model can be downloaded through the following steps. Without network access, you can visit the Hugging Face official website to download [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev).

    ```bash
    export model_name="black-forest-labs/FLUX.2-dev" # Pre-trained model path
    ```

    After obtaining the corresponding fine-tuning model, set the `model_name` parameter to the absolute path of the local pre-trained model in the following shell script for starting fine-tuning.

    ```shell
    model_name="black-forest-labs/FLUX.2-dev" # Pre-trained model path
    batch_size=1
    max_train_steps=5000
    mixed_precision="bf16"
    resolution=1024
    gradient_accumulation_steps=1
    config_file="${mixed_precision}_accelerate_config.yaml"

    # accelerate launch --config_file ${config_file} \ directory
    --dataloader_num_workers=0 \ # Adjust num_workers based on system configuration and data size.
    ```

3. Modify the code file.

    ```shell
    vim train_dreambooth_lora_flux2_img2img.py # Enter the Python file.
    ```

    1. (Optional) On Ubuntu systems, add `accelerator.print("")` near line 1796.

        ```python
        if global_step >= args.max_train_steps: # Original code
          break
        accelerator.print("") # Add
        ```

    2. (Optional) If the machine is not connected to the internet, delete `save_model_card` or place it under the `args.push_to_hub` directory.

        ```python
        validation_prompt = args.validation_prompt if args.validation_prompt else args.final_validation_prompt # Original Code
        if args.push_to_hub:
            save_model_card(
                (args.hub_model_id or Path(args.output_dir).name) if not args.push_to_hub else repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                validation_prompt=validation_prompt,
                repo_folder=args.output_dir,
                fp8_training=args.do_fp8_training,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            ) # Original Code

        ```

    3. (Optional) Multi-node execution

         Modify the `config` file:

        ```bash
        vim bf16_accelerate_config.yaml

        ```

        Uncomment `main_process_ip` and `main_process_port` in the file for use.

        ```shell
          main_process_ip: localhost  # Main node IP
          main_process_port: 6000     # Main node port
          machine_rank: 0             # Rank of the current machine
          num_machines: 1             # Total number of machines
          num_processes: 8            # Total number of cards
        ```

4. Start the FLUX2 Img2Img fine-tuning script.

    The `dreambooth_lora_flux2_i2i` fine-tuning script is provided, supporting multi-card training.

    Start `dreambooth_lora_flux2_i2i`:

    ```shell
    bash finetune_i2i_flux2_dreambooth_lora_fsdp_bf16.sh
    ```

<a id="jump7"></a>

### Performance

#### Throughput

Performance comparison of FLUX on the **Ascend chip** and **benchmark chip**:

| Chip | Chip Count |     Task     |  FPS  | batch_size | Resolution | AMP_Type | Torch_Version | FSDP2 |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | FLUX full-parameter fine-tuning  |  0.61  | 1 | 1024 | bf16 | 2.7.1 | ✔ |
| Competitor A | 8p | FLUX full-parameter fine-tuning  |  0.6 | 1 | 1024 | bf16 | 2.7.1 | ✔ |

## Inference

### Environment Setup and Execution

  See [Environment Setup](#environment-setup).

<a id="jump8"></a>

### Text-to-Image Inference

Enter the Python file for running the T2I inference task:

```shell
vim infer_flux2_text2img.py
```

1. Modify the path.

    ```python
    MODEL_PATH = "black-forest-labs/FLUX.2-dev"  # FLUX model path
    ```

    If using LoRA fine-tuned FLUX2 model for inference, modify the `LORA_WEIGHTS` parameter:

    ```python
    LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA weight path
    ```

2. Run the code.

    - Since `accelerate` is required for distributed inference, set `--num_processes` to `Number of cards` and `num_machines` to `Number of machines`, etc.

    ```shell
    accelerate launch --num_processes=4 infer_flux2_text2img.py # Distributed inference on a single machine with four cards
    ```

<a id="jump9"></a>

### I2I Inference

Enter the Python file for running the I2I inference task:

```shell
vim infer_flux2_img2img.py
```

1. Modify the path.

    ```python
    MODEL_PATH = "black-forest-labs/FLUX.2-dev"  # FLUX model path
    IMAGE = "./infer_result/flux2.fsdp_ulysses4.png"  # Path of the image to be edited
    PROMPT = "Change the crab to a dog"  # Prompt required for the editing task
    ```

    If using LoRA fine-tuned FLUX2 model for inference, modify the `LORA_WEIGHTS` parameter:

    ```python
    LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA weights path
    ```

2. Run the code.

    - Since accelerate is required for distributed inference, set `--num_processes` to `Number of cards` and `num_machines` to `Number of machines`, etc.

    ```shell
    accelerate launch --num_processes=4 infer_flux2_img2img.py # Distributed inference on a single machine with four cards
    ```

<a id="jump10"></a>

### Performance

| Chip | Chip Count |     Task     |  E2E (it/s)  |  AMP_Type | Torch_Version |
|:---:|:---:|:----------:|:-----:|:---:|:---:|
| Atlas 900 A2 PODc |8p |  Text-to-Image  | 1.14 | bf16 | 2.7.1 |
| Competitor A | 8p |  Text-to-Image  | 1.05 | bf16 | 2.7.1 |
| Atlas 900 A2 PODc |8p |  Image-to-Image  | 1.14 | bf16 | 2.7.1 |
| Competitor A | 8p |  Image-to-Image  | 1.04 | bf16 | 2.7.1 |

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
| `MULTI_STREAM_MEMORY_REUSE`   | Configures whether multi-stream memory reuse is enabled. | `0`: Disable multi-stream memory reuse.<br>`1`: Enable multi-stream memory reuse.                                                                |

## Reference

### Public Address Statement

For public IP addresses used in the code, refer to [Public Address Statement](../../../docs/en/public_address_statement.md).
