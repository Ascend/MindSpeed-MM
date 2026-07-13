# Diffusers

<p align="left">
</p>

- [Diffusers](#diffusers)
- [Qwen Image](#qwen-image)
  - [Model Introduction](#model-introduction)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Change Log](#change-log)
  - [Fine-tuning](#fine-tuning)
    - [Environment Setup](#environment-setup)
  - [Fine-tuning](#fine-tuning-1)
  - [Inference](#inference)
    - [Environment Setup and Run](#environment-setup-and-run)
    - [Environment Variable Declaration](#environment-variable-declaration)
  - [Reference](#reference)
    - [Public Address Statement](#public-address-statement)

<a id="jump1"></a>

# Qwen Image

## Model Introduction

Qwen Image is a multimodal image generation model built upon the MMDiT diffusion backbone and the Qwen2.5-VL text encoder. Its core structural innovation lies in adopting multi-scale rotary position embedding (MSRoPE) to resolve the confusion between text and image positions. Functionally, it features excellent multilingual text rendering (especially Chinese) and precise image editing, while also delivering powerful image generation capabilities.

## Version Description

### Reference Implementation

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=7a2b78bf0f788d311cc96b61e660a8e13e3b1e63
  ```

### Change Log

2025.09.08: Initial release of Qwen Image

## Fine-tuning

### Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md).

1. Install the software and driver.

    ```bash
    # Create and activate a Python environment, and install torch and torch_npu. Please refer to the Installation Guide above.

    # Modify the environment variable paths in the shell script to the actual paths. Example:
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. Clone the repository to the local server.

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    ```

3. Set the model.

    3.1 Download the [Qwen Image Reference Implementation](https://github.com/huggingface/diffusers) and execute the following command in the model's root directory to install the dependencies required by the corresponding PyTorch version.

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout 7a2b78b
    cp -r ../MindSpeed-MM/examples/diffusers/qwen_image/* ./examples/dreambooth
    ```

    Main code path:

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2 Install other required dependency libraries.

    ```shell
    pip install -e .
    pip install -r examples/dreambooth/requirements_qwen_image.txt # Install the corresponding dependencies.
    ```

<a id="jump2"></a>

## Fine-tuning

1. Prepare the fine-tuning dataset.

    - Download and extract the [pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions/tree/main) dataset, and set the `dataset_name` parameter to the absolute path of the local dataset in the following startup shell script.

    ```shell
    dataset_name="pokemon-blip-captions" # Dataset path
    ```

   - The format of the `pokemon-blip-captions` dataset is as follows:

    ```shell
    pokemon-blip-captions
    ├── dataset_infos.json
    ├── README.md
    └── data
          └── train-001.parquet
    ```

    > **Description:**
    > The training process script for this dataset is provided only as a reference example.
    >

    - If using your own fine-tuning dataset, modify `dataset_name` in the shell script:

    ```shell
    dataset_name="/path/customized_datasets" # Dataset path
    ```

    In the shell script's `accelerate launch` directory (around line 40), modify `dataset_name=$dataset_name`, change `instance_prompt` to the prompt matching your own dataset, and modify `caption_column` to the name matching your dataset:

    ```shell
    # Example
    accelerate launch --config_file $config_file \
      ./examples/dreambooth/train_dreambooth_lora_qwen_image.py \
      --pretrained_model_name_or_path=$model_name  \
      --dataset_name=$dataset_name \
      --caption_column="text" \
      --instance_prompt="a photo of pokemon" \
    ```

2. Configure the LoRA fine-tuning script.

    When connected to the internet, the fine-tuning model can be downloaded via the following steps. Without network access, you can visit the Hugging Face official website to download [Qwen Image](https://huggingface.co/Qwen/Qwen-Image).

    ```shell
    model_name="Qwen/Qwen-Image" # Pre-trained model path
    ```

    After obtaining the corresponding fine-tuned model, set the `model_name` parameter to the absolute path of the local pre-trained model in the following shell script for starting fine-tuning. If necessary, modify the configuration in accelerate, i.e., `bf16_accelerate_config.yaml`, according to the DeepSpeed distributed training configuration. Set the `config_file` parameter to the absolute path of this YAML file.

    ```shell
    model_name="Qwen/Qwen-Image"
    dataset_name="pokemon-blip-captions"
    batch_size=8
    num_processors=8
    max_train_steps=5000
    mixed_precision="bf16"
    resolution=512
    gradient_accumulation_steps=1
    config_file="./examples/dreambooth/bf16_accelerate_config.yaml"

    # accelerate launch --config_file $config_file \ directory
    --dataloader_num_workers=0 \ # Please adjust num_workers based on the system configuration and data size.
    ```

3. modify code files.

    1. (Optional) If the machine is not connected to the internet, delete `save_model_card` or place it under the `args.push_to_hub` directory.

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
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            ) # Original Code

        ```

    2. (Optional) Multi-node execution

         Modify the `config` file:

        ```bash
        vim ./examples/dreambooth/bf16_accelerate_config.yaml
        ```

        Uncomment `deepspeed_multinode_launcher`, `main_process_ip`, and `main_process_port` in the file for use.

        ```shell
            zero_stage: 2
            deepspeed_multinode_launcher: standard
          main_process_ip: localhost  # Main node IP
          main_process_port: 6000     # Main node port
          machine_rank: 0             # Rank of the current machine
          num_machines: 1             # Total number of machines
          num_processes: 8            # Total number of cards
        ```

4. Start the Qwen Image fine-tuning script.

    The `dreambooth_lora_qwen_image` fine-tuning script is provided, which supports multi-card training.

    Start `dreambooth_lora_qwen_image`:

    ```shell
    bash examples/dreambooth/finetune_qwen_image_dreambooth_lora_deepspeed_bf16.sh
    ```

## Inference

### Environment Setup and Run

  See the "fine-tuning" section.

DREAMBOOTH fine-tuned Qwen Image model inference:

```shell
vim ./examples/dreambooth/infer_qwen_image_text2img_bf16.py # Enter the Python file for running inference.
```

  1. Modify the path.

      ```python
      model_name="Qwen/Qwen-Image" # Pre-trained Qwen Image model
      output_path = "./infer_result"  # Inference result output folder
      ```

  2. Modify the prompt as needed.

      ```shell
      prompt = "In a forest, a cute fawn is bending down to drink water, with a wooden board next to it that reads 'MindSpeed-MM'."
      ```

  3. Run the code.

      ```shell
      python examples/dreambooth/infer_qwen_image_text2img_bf16.py
      ```

LoRA fine-tuned Qwen Image inference:

  ```shell
  vim ./examples/dreambooth/infer_qwen_image_text2img_lora_bf16.py
  ```

  1. Modify the path.

      ```python
      model_name="Qwen/Qwen-Image" # Pre-trained Qwen Image model
      lora_path = "qwen_image_lora" # Folder containing trained lora weights
      output_path = "./infer_result"  # Inference result output folder
      ```

  2. Modify the prompt as needed.

      ```shell
      prompt = "In a forest, a cute fawn is bending down to drink water. Next to the fawn, there is a wooden board with the words 'MindSpeed-MM' written on it."
      ```

  3. Run the code.

      ```shell
      python examples/dreambooth/infer_qwen_image_text2img_lora_bf16.py
      ```

<a id="jump3"></a>

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

For public IP addresses used in the code, see [Public Address Statement](../../../docs/en/public_address_statement.md)
