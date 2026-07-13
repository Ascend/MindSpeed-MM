# Diffusers

<p align="left">
</p>

- [Diffusers](#diffusers)
- [HiDream-I1](#hidream-i1)
  - [Model Introduction](#model-introduction)
  - [Version Notes](#version-notes)
    - [Reference Implementation](#reference-implementation)
    - [Change Log](#change-log)
  - [Fine-tuning](#fine-tuning)
    - [Environment Setup](#environment-setup)
  - [Fine-tuning](#fine-tuning-1)
    - [Fine-tuning Performance](#fine-tuning-performance)
      - [Throughput](#throughput)
  - [Inference](#inference)
    - [Environment Setup and Execution](#environment-setup-and-execution)
    - [Inference Performance](#inference-performance)
    - [Environment Variable Declaration](#environment-variable-declaration)
  - [Reference](#reference)
    - [Public Address Statement](#public-address-statement)

<a id="jump1"></a>

# HiDream-I1

## Model Introduction

HiDream-I1 is an open-source image generation model that utilizes a dual-stream decoupled sparse DiT with a dynamic MoE architecture. In the first stage, image and text are processed by two independent encoders, followed by processing with a single-stream sparse DiT using dynamic MoE.

## Version Notes

### Reference Implementation

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=d72184eba358b883d7186a0a96dedd8118fcb72a
  ```

### Change Log

2025.06.25: Initial release of HiDream-I1

## Fine-tuning

### Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md).

1. Install the software and driver.

    ```bash
    # python3.10
    conda create -n test python=3.10
    conda activate test

    # Install torch and torch_npu. Select the torch, torch_npu, and apex packages that match the corresponding Python version and x86 or arm architecture.
    pip install torch-2.7.1-cp310-cp310m-manylinux2014_aarch64.whl
    pip install torch_npu-2.7.1*-cp310-cp310m-linux_aarch64.whl

    # For apex for Ascend, refer to https://gitcode.com/Ascend/apex.
    # It is recommended to compile and install from the original repository.

    # Modify the environment variable paths in the shell script to the actual paths. Example:
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. Clone the repository to the local server.

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    ```

3. Set the model.

3.1 Download the [HiDream Reference Implementation](https://github.com/huggingface/diffusers) and execute the following command in the model root directory to install the dependencies required for the corresponding PyTorch version of the model.

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout d72184e
    cp -r ../MindSpeed-MM/examples/diffusers/hidream/* ./examples/dreambooth
    ```

Main code path:

    ```shell
    code_path=examples/dreambooth/
    ```

3.2 Install other required dependency libraries.

    ```shell
    pip install -e .
    pip install -r examples/dreambooth/mm_requirements_hidream.txt # Install the corresponding dependencies.
    ```

<a id="jump2"></a>

## Fine-tuning

1. Prepare the fine-tuning dataset.

    - Download and decompress the [3d-icon](https://huggingface.co/datasets/linoyts/3d_icon) dataset, and set the `dataset_name` parameter in the following startup shell script to the absolute path of the local dataset.

    ```shell
    dataset_name="linoyts/3d_icon" # Dataset path
    ```

   - The format of the `3d_icon` dataset is as follows:

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
    >The training script for this dataset is provided only as a reference example.
    >

    - If using your own fine-tuning dataset, you need to modify `dataset_name` in the shell script:

    ```shell
    dataset_name="/path/customized_datasets" # Dataset path
    ```

    In the shell script's `accelerate launch` directory (around line 58), modify `dataset_name=$dataset_name`, change `instance_prompt` to a prompt matching your dataset, and modify `caption_column` to the matching name for your dataset. If using the `3d_icon` dataset, no modifications are needed.

    ```shell
    # Example
    accelerate launch --config_file ${config_file} \
      ./train_dreambooth_lora_hidream.py \
      --pretrained_model_name_or_path=$model_name  \
      --pretrained_tokenizer_4_name_or_path=$pretrained4_path \
      --pretrained_text_encoder_4_name_or_path=$pretrained4_path \
      --dataset_name=$dataset_name \
      --caption_column="prompt" \
      --instance_prompt="a prompt that is suitable for your own dataset" \
    ```

2. Configure the LoRA fine-tuning script.

    When connected to the internet, the fine-tuning model can be downloaded through the following steps. Without network access, you can visit the HuggingFace official website to download [HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) and [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

    ```shell
    model_name="HiDream-ai/HiDream-I1-Dev" # Pre-trained model path
    pretrained4_path="meta-llama/Meta-Llama-3.1-8B-Instruct" # Fourth pre-trained path
    ```

    After obtaining the corresponding fine-tuning model, set the `model_name` parameter to the absolute path of the local pre-trained model in the shell script that launches fine-tuning below.

    ```shell
    model_name="HiDream-ai/HiDream-I1-Full"
    pretrained4_path="meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset_name="linoyts/3d_icon"
    batch_size=8
    num_processors=8
    max_train_steps=5000
    mixed_precision="bf16"
    resolution=512
    gradient_accumulation_steps=1
    config_file="bf16_accelerate_config.yaml"

    # In the accelerate launch --config_file ${config_file} \ directory
    --dataloader_num_workers=0 \ # Adjust num_workers based on the system configuration and data size.
    ```

3. Modify the code file.

    1. Open the `train_dreambooth_lora_hidream.py` file.

        ```shell
        cd examples/dreambooth/ # Enter the dreambooth directory from the diffusers directory.
        vim train_dreambooth_lora_hidream.py # Enter the Python file.
        ```

        - Add code around line 71 above `if is_wandb_available():`.

        ```python
        # Add code to train_dreambooth_lora_hidream.py （around line 71）.
        from transformer_patches import apply_patches
        apply_patches()

        if is_wandb_available(): # Original code
          import wandb
        ```

    2. (Optional) On Ubuntu systems, add `accelerator.print("")` around line 1701. If not added, the terminal display may not update synchronously.

        ```python
        if global_step >= args.max_train_steps: # Original code
          break
        accelerator.print("") # Add
        ```

    3. (Optional) If the machine is not connected to the internet, you need to delete `save_model_card` or place it under the `args.push_to_hub` directory.

        ```python
        validation_prompt = args.validation_prompt if args.validation_prompt else args.final_validation_prompt # Original code

        if args.push_to_hub:
            save_model_card(
                (args.hub_model_id or Path(args.output_dir).name) if not args.push_to_hub else repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                validation_prompt=validation_prompt,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            ) # Original code

        ```

    4. (Optional) Multi-node execution

         Modify the `config` file:

        ```bash
        vim bf16_accelerate_config.yaml
        ```

        Uncomment `deepspeed_multinode_launcher`, `main_process_ip`, and `main_process_port` in the file to enable them.

        ```shell
            zero_stage: 2
            deepspeed_multinode_launcher: standard
          main_process_ip: localhost  # Main node IP
          main_process_port: 6000     # Main node port
          machine_rank: 0             # Rank of the current machine
          num_machines: 1             # Total number of machines
          num_processes: 8            # Total number of cards
        ```

4. Start the HiDream fine-tuning script.

    The `dreambooth_lora_hidream` fine-tuning script is provided, which supports multi-card training.

    Start `dreambooth_lora_hidream`:

    ```shell
    bash finetune_hidream_dreambooth_lora_deepspeed_bf16.sh
    ```

### Fine-tuning Performance

#### Throughput

Performance comparison of HiDream on the **Ascend chip** and **benchmark chip**:

| Chip | Chip Count | Task | FPS | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | HiDream full-parameter fine-tuning | 18.37 | 8 | bf16 | 2.6 | ✔ |
| Competitor A | 8p | HiDream full-parameter fine-tuning | 19.61 | 8 | bf16 | 2.6 | ✔ |

## Inference

### Environment Setup and Execution

See the "fine-tuning" section.

```shell
cd examples/dreambooth/ # Enter the dreambooth directory from the diffusers directory.
```

HiDream model inference:

Modify `PROMPTS` and parameters required for inference.

```shell
vim prompt_utils.py
```

Modify the weight path required by the inference file.

```shell
vim infer_hidream_text2img_bf16.py # Enter the Python file for running inference
```

- Modify the path.

  ```python
  MODEL_PATH = "HiDream-ai/HiDream-I1-Full"  # Model path for HiDream
  FORTH_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Pre-trained model path for tokenizer & text encoder
  ```

Run the code.

```shell
python infer_hidream_text2img_bf16.py
```

LoRA fine-tuned HiDream model inference:

Modify `PROMPTS` and parameters required for inference.

```shell
vim prompt_utils.py
```

Modify the weight paths required for the inference file.

```shell
vim infer_hidream_text2img_lora_bf16.py
```

- Modify paths.

  ```python
  MODEL_PATH = "HiDream-ai/HiDream-I1-Full"  # Model path for HiDream
  FORTH_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Pre-trained model path for tokenizer & text encoder
  OUTPUT_PATH = "./infer_result"  # Output path
  ```

Run the code.

```shell
python infer_hidream_text2img_lora_bf16.py
```

<a id="jump3"></a>

### Inference Performance

| Chip | Chip Count | Task | E2E (it/s) | AMP_Type | Torch_Version |
|:---:|:---:|:----------:|:-----:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Text-to-Image | 1.27 | bf16 | 2.6 |
| Competitor A | 8p | Text-to-Image | 1.88 | bf16 | 2.6 |

### Environment Variable Declaration

| Environment Variable | Description | Value Description |
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

For public IP addresses used in the code, refer to [Public Address Statement](../../../docs/en/public_address_statement.md).
