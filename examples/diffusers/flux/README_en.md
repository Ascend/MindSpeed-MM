# Diffusers

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-05-26T09:05:13.589Z pushedAt=2026-05-26T09:14:15.827Z -->
<p align="left">
</p>

- [Diffusers](#diffusers)
- [FLUX](#flux)
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

# FLUX

## Model Introduction

[FLUX.1 dev](https://blackforestlabs.ai/announcing-black-forest-labs/) is a generative model based on Rectified Flow Transformers.

- Reference implementation

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=a98a839de75f1ad82d8d200c3bc2e4ff89929081
  ```

## Fine-tuning

### Environment Setup

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md)

1. Install the software and driver.

    ```bash
    # Install torch and torch_npu by following the installation guide mentioned above.

    # Modify the environment variable paths in the shell script to the actual paths. Example:
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. Clone the repository to the local server..

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    ```

3. Set the model.

    3.1 Download [FLUX reference implementation](https://github.com/huggingface/diffusers) and execute the following command in the model root directory to install the dependencies matching the corresponding PyTorch version of the model.

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout a98a839de75f1ad82d8d200c3bc2e4ff89929081
    cp -r ../MindSpeed-MM/examples/diffusers/flux/* ./examples/dreambooth
    ```

    Main code path:

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2 Install other required dependencies.

    ```shell
    pip install -e .
    vim examples/dreambooth/requirements_flux.txt #Modify versions: torch==2.7.1, accelerate==1.7.0, transformers==4.47.1, add deepspeed==0.17.2, peft==0.7.1
    pip install -r examples/dreambooth/requirements_flux.txt # Install the corresponding dependencies.
    ```

<a id="jump2"></a>

## Fine-tuning

1. Prepare the fine-tuning dataset.

    - Download and decompress the [pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions/tree/main) dataset, and set the `dataset_name` parameter to the absolute path of the local dataset in the following launch shell script.

    ```shell
    dataset_name="pokemon-blip-captions" # Dataset path
    ```

   - The format of the `pokemon-blip-captions` dataset is as follows:

    ```shell
    pokemon-blip-captions
    ├── dataset_infos.json
    ├── README.MD
    └── data
          └── train-001.parquet
    ```

    - For a training dataset containing only images, such as the [dog dataset](https://huggingface.co/datasets/diffusers/dog-example) used by the DeepSpeed script, move the `dog` folder to the `examples/dreambooth/` directory.

    ```shell
    input_dir="dog" # Dataset Path
    ```

    ```shell
    dog
    ├── alvan-nee-*****.jpeg
    ├── alvan-nee-*****.jpeg
    ```

    > **Note**:
    > The training script for this dataset is provided for reference only.
    >

    - If using your own fine-tuning dataset, modify `input_dir` to `dataset_name` in the shell script:

    ```shell
    dataset_name="/path/customized_datasets" # Dataset path
    ```

    In the shell script under the `accelerate launch` directory (around line 70), modify `instance_data_dir=$instance_dir \` to `dataset_name=$dataset_name`, and change `instance_prompt` and `validation_prompt` to prompts that match your own dataset:

    ```shell
    # Example
    accelerate launch --config_file ${config_file} \
      ./train_dreambooth_flux.py \
      --pretrained_model_name_or_path=$model_name  \
      --dataset_name=$dataset_name \
      --instance_prompt="a prompt that is suitable for your own dataset" \
      --validation_prompt="a validation prompt based on your own dataset" \
    ```

2. Configure the FLUX fine-tuning script.

    When connected to the internet, the fine-tuning model can be downloaded via the following steps. Without network access, you can visit the Hugging Face official website to manually download [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).

    ```bash
    export model_name="black-forest-labs/FLUX.1-dev" # Pre-trained model path
    ```

    After obtaining the corresponding fine-tuning model, set `model_name` in the following shell script for launching fine-tuning to the absolute path of the local pre-trained model.

    ```shell
    model_name="black-forest-labs/FLUX.1-dev" # Pre-trained model path
    batch_size=16
    max_train_steps=5000
    mixed_precision="bf16" # Mixed precision
    resolution=256
    config_file="bf16_accelerate_config.yaml"

    # Under the accelerate launch --config_file ${config_file} \ directory
    --dataloader_num_workers=0 \ # Adjust num_workers based on system configuration and data size.
    ```

3. modify code files..

    1. In the `src/diffusers/models/embeddings.py` file, find `freqs_dtype = torch.float32 if is_mps else torch.float64` around line 813 under `class FluxPosEmbed(nn.Module):` and modify it to `freqs_dtype = torch.float32`.

        ```shell
        # Change to freqs_dtype = torch.float32.
        vim src/diffusers/models/embeddings.py
        ```

        ```python
        freqs_dtype = torch.float32 # around line 813
        # freqs_dtype = torch.float32 if is_mps else torch.float64 # Original code
        ```

    2. Open the `train_dreambooth_flux.py` or `train_dreambooth_lora_flux_advanced.py` file.

        ```shell
        cd examples/dreambooth/ # Enter the dreambooth directory from the diffusers directory.
        vim train_dreambooth_flux.py # Enter the Python file.
        # For FLUX LoRA, first enter the advanced_diffusion_training directory.
        vim ../advanced_diffusion_training/train_dreambooth_lora_flux_advanced.py # Enter the Python file.
        ```

        - Add code above the import line `if is_wandb_available():` (around line 62).

        ```python
        # Add code around line 62 of train_dreambooth_flux.py.
        from patch_flux import TorchPatcher, config_gc, create_save_model_hook
        TorchPatcher.apply_patch()
        config_gc()

        if is_wandb_available(): # Original code
          import wandb
        ```

        - In `log_validation`, modify `pipeline = pipeline.to(accelerator.device)`, around line 171 in `train_dreambooth_flux.py`

        ```python
        # Modify the pipeline to:
        pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
        # pipeline = pipeline.to(accelerator.device) # Original code
        ```

    3. (Optional) On Ubuntu systems, add `accelerator.print("")` around line 1701.

        ```python
        if global_step >= args.max_train_steps: # Original Code
          break
        accelerator.print("") # Add
        ```

    4. (Optional) Save model checkpoints.

        Since the model size is large, if `checkpointing_steps` is not needed, set it to a value greater than `max_train_steps` to avoid enabling it.

        ```shell
        --checkpointing_steps=50000 \ # Set the number of steps to 50000.
        ```

        To save checkpoints, please modify the code:

        ```shell
        vim train_dreambooth_flux.py #(around line 1669)
        vim ../advanced_diffusion_training/train_dreambooth_lora_flux_advanced.py #(around line 2322)
        ```

        - In the import section at the top of the file, add `DistributedType` after `from accelerate import Accelerator` (around line 30).
        - After `if accelerator.is_main_process`, add `or accelerator.distributed_type == DistributedType.DEEPSPEED` (around lines 1669/2322), and after `if args.checkpoints_total_limit is not None`, add `and accelerator.is_main_process`.

        ```python
        from accelerate import Accelerator, DistributedType
        # from accelerate import Accelerator # Original Code

        if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
        # if accelerator.is_main_process: # Original code
          if global_step % args.checkpointing_steps == 0:  # Original code; no modifications needed
            if args.checkpoints_total_limit is not None and accelerator.is_main_process: # Add
        ```

        The LoRA tasks require the patch task to be called to save weights. To do so,
        locate the code `accelerator.register_save_state_pre_hook(save_model_hook)` in the `train_dreambooth_lora_flux_advanced.py` file and modify it (around line 1712) by copying and pasting the following code:

        ```python
        # Add
        save_Model_Hook = create_save_model_hook(
              accelerator=accelerator,
              unwrap_model=unwrap_model,
              transformer=transformer,
              text_encoder_one=text_encoder_one,
              args=args,
              weight_dtype=weight_dtype
        )
        accelerator.register_save_state_pre_hook(save_Model_Hook) # Modify
        # accelerator.register_save_state_pre_hook(save_model_hook) # Original code
        accelerator.register_load_state_pre_hook(load_model_hook) # Original code; no modification needed.
        ```

        Modify the shell script:

        ```shell
        export HCCL_CONNECT_TIMEOUT=1200 # Significantly increase HCCL_CONNECT_TIMEOUT (e.g., to 5000)
        export HCCL_EXEC_TIMEOUT=17000
        --checkpointing_steps=50000 \ # Set the number of steps to 50000.
        ```

    5. (Optional) Multi-node execution

        Modify the `config` file:

        ```bash
        vim bf16_accelerate_config.yaml
        ```

        Uncomment `deepspeed_multinode_launcher`, `main_process_ip`, and `main_process_port` in the file for use.

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

4. Launch the FLUX fine-tuning script.

    The `flux_dreambooth` and `flux_dreambooth_lora` fine-tuning scripts are provided for this task, supporting multi-card training.

    Launch the `flux_dreambooth` script:

    ```shell
    bash finetune_flux_dreambooth_deepspeed_bf16.sh
    ```

    Launch `flux_dreambooth_lora` script:

    ```shell
    bash finetune_flux_dreambooth_lora_deepspeed_bf16.sh
    ```

### Performance

#### Throughput

Performance comparison of FLUX on the **Ascend chip** and **benchmark chip**:

| Chip | Chip Count |     Task     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Full-parameter fine-tuning for FLUX  |  55.23  |     16      | bf16 | 2.1 | ✔ |
| Competitor A | 8p | Full-parameter fine-tuning for FLUX  |  53.65 |     16      | bf16 | 2.1 | ✔ |

## Inference

### Environment Setup and Execution

  See the "fine-tuning" section.

```shell
cd examples/dreambooth/ # Enter the dreambooth directory from the diffusers directory.
```

FLUX model inference:

```shell
vim infer_flux_text2img_bf16.py # Enter the Python file for running inference.
```

  1. Modify the path.

      ```python
      MODEL_PATH = "/black-forest-labs/FLUX.1-dev"  # FLUX model path.
      ```

  2. Run the code.

      ```shell
      python infer_flux_text2img_bf16.py
      ```

  DreamBooth fine-tuned FLUX model inference:

  ```shell
  vim infer_flux_text2img_dreambooth_bf16.py
  ```

  1. Modify the path.

      ```python
      MODEL_PATH = "./output_FLUX_dreambooth"  # Path to save the DreamBooth fine-tuned model
      ```

  2. Run the code.

      ```shell
      python infer_flux_text2img_dreambooth_bf16.py
      ```

LoRA fine-tuned FLUX model inference:

  ```shell
  vim infer_flux_text2img_lora_bf16.py
  ```

1. Modify paths.

      ```python
      MODEL_PATH = "./FLUX"  # FLUX model path
      LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA weight path
      ```

2. Run the code.

      ```shell
      python infer_flux_text2img_lora_bf16.py
      ```

Distributed inference:

  ```shell
  vim infer_flux_text2img_distrib.py
  ```

- Set the model weight path `model_path` to the model weight path or the fine-tuned weight path.
- For LoRA fine-tuning, set `lora_weights` to the LoRA weight path.

    ```python
    model_path = "/black-forest-labs/FLUX.1-dev"  # Model weight/Fine-tuned weight path
    lora_weights = "/pytorch_lora_weights.safetensors"  # LoRA weight path
    ```

- Launch the distributed inference script

  - Since `accelerate` is used for distributed inference,  `--num_processes` can be set to `Number of cards`, `num_machines` can be set to `Number of machines`, etc.

  ```shell
  accelerate launch --num_processes=4 infer_flux_text2img_distrib.py # Distributed inference on a single machine with four GPUs
  ```

<a id="jump3"></a>

### Performance

| Chip | Chip Count | Task | E2E (it/s) | AMP_Type | Torch_Version |
|:---:|:---:|:----------:|:-----:|:---:|:---:|
| Atlas 900 A2 PODc |8p | Text-to-Image | 1.16 | bf16 | 2.1 |
| Competitor A | 8p | Text-to-Image | 1.82 | bf16 | 2.1 |
| Atlas 900 A2 PODc |8p | Text-to-Image fine-tuning | 1.12 | bf16 | 2.1 |
| Competitor A | 8p | Text-to-Image fine-tuning | 1.82 | bf16 | 2.1 |

## Environment Variable Declaration

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

For public address references in the code, see [Public Address Statement](../../../docs/en/public_address_statement.md).
