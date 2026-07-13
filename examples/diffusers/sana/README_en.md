# Diffusers

<p align="left">
</p>

- [Diffusers](#diffusers)
- [SANA](#sana)
  - [Model Introduction](#model-introduction)
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

# SANA

## Model Introduction

SANA is a text-to-image model jointly launched by NVIDIA, the Massachusetts Institute of Technology, and Tsinghua University. By utilizing a deep compression autoencoder, a Linear DiT, and a Decoder Only small language model, it can efficiently generate high-definition images up to 4096x4096 resolution.

- Reference implementation

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=cd0a4a82cf8625b96e2889afee2fce5811b35c05
  ```

## Fine-tuning

### Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md)).

1. Install the software and driver.

    ```bash
    # python3.10
    conda create -n test python=3.10
    conda activate test

    # Install torch and torch_npu. Ensure you select the torch, torch_npu, and apex packages that correspond to your Python version and architecture (x86 or arm).
    pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
    pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

    # For apex for Ascend, refer to https://gitcode.com/Ascend/apex.
    # It is recommended to compile and install components from the source repository.

    # Modify the environment variable paths in the shell script to the actual paths. Example:
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. Clone the repository to the local server.

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    ```

3. Set the model.

    3.1 Download [SANA Reference Implementation](https://github.com/huggingface/diffusers) and execute the following command in the model root directory to install the dependencies required for the corresponding PyTorch version of the model.

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout cd0a4a82cf8625b96e2889afee2fce5811b35c05
    cp -r ../MindSpeed-MM/examples/diffusers/sana/* ./examples/dreambooth
    ```

    Main code path:

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2 Install the required dependency libraries.

    ```shell
    pip install -e .
    vim examples/dreambooth/requirements_sana.txt #Modify versions: torch==2.7.1, accelerate==1.10.0, transformers==4.47.1, add deepspeed==0.17.2, peft==0.7.1
    pip install -r examples/dreambooth/requirements_sana.txt # Install the corresponding dependencies
    ```

<a id="jump2"></a>

## Fine-tuning

1. Prepare the fine-tuning dataset.

    - Download and decompress the [pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions/tree/main) dataset, and set the `dataset_name` parameter to the absolute path of the local dataset in the following startup shell script.

    ```shell
    dataset_name="pokemon-blip-captions" # Dataset Path
    ```

   - The format of the `pokemon-blip-captions` dataset is as follows:

    ```shell
    pokemon-blip-captions
    ├── dataset_infos.json
    ├── README.MD
    └── data
          └── train-001.parquet
    ```

    > **Note**:
    > The training process script for this dataset is provided only as a reference example.
    >

    - If using your own fine-tuning dataset, you need to modify `dataset_name` in the shell script:

    ```shell
    dataset_name="/path/customized_datasets" # Dataset Path
    ```

    In the shell script, under the `accelerate launch` directory (around line 70), modify `dataset_name=$dataset_name`, and change `instance_prompt` and `validation_prompt` to prompts that match your own dataset.

    ```shell
    # Example
    accelerate launch --config_file ${config_file} \
      ./train_dreambooth_lora_sana.py \
      --pretrained_model_name_or_path=$model_name  \
      --dataset_name=$dataset_name \
      --instance_prompt="a prompt that is suitable for your own dataset" \
      --validation_prompt="a validation prompt based on your own dataset" \
    ```

2. Configure the LoRA fine-tuning script.

    With the Internet connected, the fine-tuning model can be downloaded through the following steps. Without network access, you can visit the Hugging Face official website to download [SANA 4K](https://huggingface.co/Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers).

    ```shell
    export model_name="Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers" # Pre-trained model path
    ```

    After obtaining the corresponding fine-tuning model, set the `model_name` parameter to the absolute path of the local pre-trained model in the following shell script to start fine-tuning.

    ```shell
    model_name="Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers" # Pre-trained model path
    dataset_name="pokemon-blip-captions"
    batch_size=8
    num_processors=8
    max_train_steps=5000
    mixed_precision="bf16"
    resolution=1024
    gradient_accumulation_steps=1
    config_file="bf16_accelerate_config.yaml"

    # accelerate launch --config_file ${config_file} \ directory
    --dataloader_num_workers=8 \ # Adjust num_workers based on the system configuration and data size.
    ```

3. Modify code files.

    1. Open the `train_dreambooth_lora_sana.py` file.

        ```shell
        cd examples/dreambooth/ # Enter the dreambooth directory from the diffusers directory.
        vim train_dreambooth_lora_sana.py # Enter the Python file.
        ```

        - Add code above `if is_wandb_available():` (near line 69).

        ```python
        # Add code near line 69 in train_dreambooth_sana.py.
        from patch_sana import create_save_model_hook, create_load_model_hook

        if is_wandb_available(): # Original code
          import wandb
        ```

        Modify the VAE `dtype` to `BF16` near line 986.

        ```python
        vae.to(dtype=torch.bfloat16)
        # vae.to(dtype=torch.float32) # Original code
        transformer.to(accelerator.device, dtype=weight_dtype) # Original code
        # Gemma2 is particularly suited for bfloat16.
        text_encoder.to(dtype=torch.bfloat16) # Original code
        ```

    2. (Optional) On Ubuntu systems, add `accelerator.print("")` near line 1701.

        ```python
        if global_step >= args.max_train_steps: # Original code
          break
        accelerator.print("") # Add
        ```

    3. (Optional) Save model checkpoints.

        ```shell
        --checkpointing_steps=5001 \ # Set the number of steps to 5001.
        ```

        Modify the code if checkpoint saving is required.

        ```shell
        vim examples/dreambooth/train_dreambooth_lora_sana.py #(Near line 1788)
        ```

        - Add `DistributedType` after `from accelerate import Accelerator` (around line 31).
        - After `if accelerator.is_main_process`, add `or accelerator.distributed_type == DistributedType.DEEPSPEED` (around line 1431), and after `if args.checkpoints_total_limit is not None`, add `and accelerator.is_main_process`.

        ```python
        from accelerate import Accelerator, DistributedType
        # from accelerate import Accelerator # Original code

        if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
        # if accelerator.is_main_process: # Original code
          if global_step % args.checkpointing_steps == 0:  # Original code; no modification needed
            if args.checkpoints_total_limit is not None and accelerator.is_main_process: # Add
        ```

        The LoRA task requires to call the patch task to save weights.
        Locate the code `accelerator.register_save_state_pre_hook(save_model_hook)` in the `train_dreambooth_lora_sana.py` file and modify it (around line 1088). Copy and paste the following code:

        ```python
        # Add
        save_Model_Hook = create_save_model_hook(
              accelerator=accelerator,
              unwrap_model=unwrap_model,
              transformer=transformer,
        )
        load_Model_Hook = create_load_model_hook(
              accelerator=accelerator,
              unwrap_model=unwrap_model,
              transformer=transformer,
              args=args,
        )
        accelerator.register_save_state_pre_hook(save_Model_Hook) # Modify
        accelerator.register_load_state_pre_hook(load_Model_Hook) # Modify
        # accelerator.register_save_state_pre_hook(save_model_hook) # Original code
        # accelerator.register_load_state_pre_hook(load_model_hook) # Original code
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

4. Launch the SANA fine-tuning script.

    The `sana_dreambooth_lora` fine-tuning script is provided, which supports multi-card training.

    Command:

    ```shell
    bash finetune_sana_dreambooth_lora_deepspeed_bf16.sh
    ```

### Performance

#### Throughput

Performance comparison of SANA on the **Ascend Chip** and **benchmark Chip**:

| Chip | Chip Count |     Task     |  FPS  | batch_size | AMP_Type | Torch_Version | DeepSpeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | SANA full-parameter fine-tuning  |  28.7  |     8      | bf16 | 2.1 | ✔ |
| Competitor A | 8p | SANA full-parameter fine-tuning  | 32.8  |     8      | bf16 | 2.1 | ✔ |

## Inference

### Environment Setup and Execution

  See the "fine-tuning" section.

```shell
cd examples/dreambooth/ # Enter the dreambooth directory from the diffusers directory.
```

SANA model inference:

```shell
vim infer_sana_text2img_bf16.py # Enter the Python file for running inference.
```

  1. Modify the path.

      ```python
      MODEL_PATH = "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers"  # SANA model path
      ```

  2. Run the code.

      ```shell
      python infer_sana_text2img_bf16.py
      ```

  LoRA fine-tuned SANA model inference:

  ```shell
  vim infer_sana_text2img_lora_bf16.py
  ```

  1. Modify the path.

      ```python
      MODEL_PATH = "./SANA"  # SANA model path
      LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA weight path
      ```

  2. Run the code.

      ```shell
      python infer_sana_text2img_lora_bf16.py
      ```

  Distributed inference:

  ```shell
  vim infer_sana_text2img_distrib.py
  ```

- Set `model_path` to the model weight path or the fine-tuned weight path.
- For LoRA fine-tuning, set `lora_weights` to the LoRA weight path.

    ```python
    model_path = "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers"  # Model Weight / Fine-tuned weight path
    lora_weights = "/pytorch_lora_weights.safetensors"  # LoRA weight path
    ```

- Execute the distributed inference script.

  - Since `accelerate` is used for distributed inference, set `--num_processes` to `Number of cards` and `num_machines` to `Number of machines`, etc.

  ```shell
  accelerate launch --num_processes=4 infer_sana_text2img_distrib.py # Distributed inference on a single machine with four cards
  ```

<a id="jump3"></a>

### Performance

| Chip | Chip Count |     Task     |  E2E (it/s)  |  AMP_Type | Torch_Version |
|:---:|:---:|:----------:|:-----:|:---:|:---:|
| Atlas 900 A2 PODc |8p |  Text-to-Image  | 0.84 | bf16 | 2.1 |
| Competitor A | 8p |  Text-to-Image  | 1.04 | bf16 | 2.1 |

### Environment Variable Declaration

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

For public IP addresses used in the code, refer to [Public Address Statement](../../../docs/en/public_address_statement.md).
