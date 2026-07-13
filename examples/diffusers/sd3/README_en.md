# Diffusers

<p align="left">
</p>

- [Diffusers](#diffusers)
- [Stable Diffusion 3 \& Stable Diffusion 3.5](#stable-diffusion-3--stable-diffusion-35)
  - [Model Introduction](#model-introduction)
  - [Fine-tuning](#fine-tuning)
    - [Environment Setup](#environment-setup)
    - [Fine-tuning](#fine-tuning-1)
    - [Performance](#performance)
      - [Throughput](#throughput)
  - [Inference](#inference)
    - [Environment Setup and Execution](#environment-setup-and-execution)
  - [Evaluation Using Baseline Dataset](#evaluation-using-baseline-dataset)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Reference](#reference)
    - [Public Address Statement](#public-address-statement)

<a id="jump1"></a>

# Stable Diffusion 3 & Stable Diffusion 3.5

## Model Introduction

Diffusion Models are a type of generative models capable of producing a wide variety of high-resolution images. Diffusers is a model suite released by Hugging Face, serving as the go-to library for state-of-the-art pre-trained diffusion models for generating images, audio, and even 3D structures of molecules. The suite includes various models based on diffusion models and provides implementations for training and inference across a range of downstream tasks.

- Reference implementation

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=5f724735437d91ed05304da478f3b2022fe3f6fb
  ```

## Fine-tuning

### Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md).

1. Install the software and driver.

    ```bash
    # python3.10
    conda create -n test python=3.10
    conda activate test

    # Install torch and torch_npu, paying attention to selecting the torch, torch_npu, and apex packages corresponding to the Python version and x86 or arm architecture.
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

    3.1 Download [SD3/SD3.5 Reference Implementation](https://github.com/huggingface/diffusers) or [Implementation Adapted for Ascend AI Processors](https://gitcode.com/Ascend/ModelZoo-PyTorch.git) or execute the following command in the model root directory to install the dependencies required for the corresponding PyTorch version of the model.

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout 5f724735437d91ed05304da478f3b2022fe3f6fb
    cp -r ../MindSpeed-MM/examples/diffusers/sd3 ./sd3
    ```

    Main code path:

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2 Install the required dependency libraries.

    ```shell
    pip install -e .
    vim examples/dreambooth/requirements_sd3.txt #Modify versions: torch==2.7.1, accelerate==1.10.0, transformers==4.44.0, add deepspeed==0.17.2, peft==0.7.1
    pip install -r examples/dreambooth/requirements_sd3.txt # Install the corresponding dependencies.
    ```

<a id="jump2"></a>

### Fine-tuning

1. Prepare the fine-tuning dataset.

    Download and extract the [pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions/tree/main) dataset, and set the `dataset_name` parameter to the absolute path of the local dataset in the following startup shell script.

    ```shell
    vim sd3/finetune_sd3_dreambooth_deepspeed_**16.sh
    vim sd3/finetune_sd3_dreambooth_fp16.sh
    ```

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

    - For training datasets containing only images, such as the [dog dataset](https://huggingface.co/datasets/diffusers/dog-example) used by non-DeepSpeed scripts, set the `input_dir` parameter to the absolute path of the local dataset.

    ```shell
    input_dir="dog" # Dataset Path
    ```

    ```shell
    dog
    ├── alvan-nee-*****.jpeg
    ├── alvan-nee-*****.jpeg
    ```

    > **Note**:
    >The training process script for this dataset is provided only as a reference example.
    >

2. Configure the SD3/SD3.5 fine-tuning script.

    SD3: Visit the official Hugging Face website to download the [sd3-medium model](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main).

    ```bash
    export model_name="stabilityai/stable-diffusion-3-medium-diffusers" # Pre-trained model path
    ```

    SD3.5: Visit the official Hugging Face website to download the [sd3.5-large model](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main).

    ```bash
    export model_name="stabilityai/stable-diffusion-3.5-large" # Pre-trained model path
    ```

    After obtaining the corresponding fine-tuned model, set the `model_name` parameter in the following shell startup script to the absolute path of the local pre-trained model. SD3 and SD3.5 share the same script.

    ```shell
    scripts_path="./sd3" # Model root directory (model folder name)
    model_name="stabilityai/stable-diffusion-3-medium-diffusers" # Pre-trained model path (for SD3)
    dataset_name="pokemon-blip-captions"
    batch_size=4
    num_processors=8 # Number of cards (used for FPS calculation; be the same as the value in the yaml file)
    max_train_steps=2000
    mixed_precision="bf16" # Mixed precision
    resolution=1024
    config_file="${scripts_path}/${mixed_precision}_accelerate_config.yaml"

    # Under the accelerate launch --config_file ${config_file} \ directory
    --dataloader_num_workers=0 \ # Please adjust num_workers based on the system configuration and data size.
    ```

    Dataset selection: If you choose the [dataset provided in the source repository](https://huggingface.co/datasets/diffusers/dog-example), modify two instances of `dataset_name` to `input_dir`.

    ```shell
    input_dir="dog"

    # accelerator Modify --dataset_name=#dataset_name
    --instance_data_dir=$input_dir
    ```

    | Dataset | Path Configuration | accelerate Configuration |
    |:----------:|:----------:|:----------:|
    | dog | input_dir="dog" | --instance_data_dir=$input_dir; --instance_prompt="A photo of sks dog" |
    | pokemon | dataset_name="pokemon-blip-captions" | --dataset_name=$dataset_name --caption_column="text"; --instance_prompt="A photo of pokemon" |

    Modify the path for `deepspeed_config_file` in `fp16_accelerate_config.yaml`:

    ```shell
    vim sd3/fp16_accelerate_config.yaml
    # Modify:
    deepspeed_config_file: ./sd3/deepspeed_fp16.json # DeepSpeed JSON file path
    ```

3. (Optional) For Ubuntu systems, add `accelerator.print("")` near the lines indicated by `___CODE_BLOCK_PLACEHOLDER___421705` and `___CODE_BLOCK_PLACEHOLDER___431861`.

    ```shell
    vim examples/dreambooth/train_dreambooth_sd3.py
    # Or
    vim examples/dreambooth/train_dreambooth_lora_sd3.py
    ```

    Example:

    ```python
    if global_step >= args.max_train_steps: # Original code
      break
    accelerator.print("") # Add
    ```

4. Modify the code if you need to save checkpoints.

    ```shell
    vim examples/dreambooth/train_dreambooth_sd3.py
    ```

    - Add `DistributedType` after `from accelerate import Accelerator` (around line 30).
    - Add `or accelerator.distributed_type == DistributedType.DEEPSPEED` after `if accelerator.is_main_process` (around line 1681 for dreambooth), and add `and accelerator.is_main_process` after `if args.checkpoints_total_limit is not None`.

    ```python
    from accelerate import Accelerator, DistributedType
    # from accelerate import Accelerator # Original Code
    from accelerate.logging import get_logger # Original Code

    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
    # if accelerator.is_main_process: # Around line 1681/1833 of the original code
      if global_step % args.checkpointing_steps == 0:  # Original code: No modification
        if args.checkpoints_total_limit is not None and accelerator.is_main_process: # Add
    ```

5. Modify the file.

    ```shell
    vim examples/dreambooth/train_dreambooth_sd3.py
    # Or
    vim examples/dreambooth/train_dreambooth_lora_sd3.py
    ```

Modify `pipeline = pipeline.to(accelerator.device)` in `log_validation`. It is located near line 174 in `train_dreambooth_sd3.py` and near line 198 in `train_dreambooth_lora_sd3.py`.

    ```python
    # Modify pipeline to:
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    # pipeline = pipeline.to(accelerator.device) # Original code
    ```

6. Execute the SD3 fine-tuning script.

The **8-card** training scripts for DreamBooth and DreamBooth+LoRA using **mixed-precision fp16** and **mixed-precision bf16** are provided, with and without DeepSpeed distributed training.

    ```shell
    bash sd3/finetune_sd3_dreambooth_deepspeed_**16.sh #Fine-tuning with DeepSpeed and DreamBooth
    bash sd3/finetune_sd3_dreambooth_lora_deepspeed_fp16.sh #Fine-tuning with DeepSpeed and DreamBooth (SD3.5)
    bash sd3/finetune_sd3_dreambooth_fp16.sh #Fine-tuning with DreamBooth without DeepSpeed
    bash sd3/finetune_sd3_dreambooth_lora_fp16.sh #Fine-tuning with DreamBooth+LoRA without DeepSpeed (SD3)
    ```

### Performance

#### Throughput

Performance comparison of SD3 on **Ascend chips** and **benchmark chips**:

| Chip | Chip Count |     Task     |  FPS  | batch_size | AMP_Type | Resolution | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | DreamBooth full-parameter fine-tuning  |   16.09 |     4      | bf16 | 1024 | 2.1 | ✔ |
| Competitor A | 8p | DreamBooth full-parameter fine-tuning  |  16.01 |     4      | bf16 | 1024 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | DreamBooth full-parameter fine-tuning |  15.16 |     4      | fp16 | 1024 | 2.1 | ✔ |
| Competitor A | 8p | DreamBooth full-parameter fine-tuning |   15.53 |     4      | fp16 | 1024 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p | DreamBooth full-parameter fine-tuning | 3.11  | 1 | fp16 | 1024 | 2.1 | ✘ |
| Competitor A | 8p | DreamBooth full-parameter fine-tuning | 3.71 | 1 | fp16 | 1024 | 2.1 | ✘ |
| Atlas 900 A2 PODc |8p | DreamBooth+LoRA | 108.8 | 8 | fp16 | 1024 | 2.1 | ✘ |
| Competitor A | 8p | DreamBooth+LoRA | 110.69 | 8 | fp16 | 1024 | 2.1 | ✘ |

Performance comparison of SD3.5 on **Ascend chips** and **benchmark chips**:

| Chip | Chip Count |     Task     |  FPS  | batch_size | AMP_Type | Resolution | Torch_Version | deepspeed | gradient checkpointing |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | DreamBooth full-parameter fine-tuning  |   26.24 |     8      | bf16 | 512 | 2.1 | ✔ | ✔ |
| Competitor A | 8p | DreamBooth full-parameter fine-tuning  |  28.33 |     8      | bf16 | 512 | 2.1 | ✔ | ✔ |
| Atlas 900 A2 PODc | 8p | DreamBooth+LoRA |  47.93 |     8      | fp16 | 512 | 2.1 | ✔ | ✘ |
| Competitor A | 8p | DreamBooth+LoRA |   47.95 |     8      | fp16 | 512 | 2.1 | ✔ | ✘ |

## Inference

### Environment Setup and Execution

See the "fine-tuning" section.

Inference script:

  For the image-to-image inference script, you need to prepare an [image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png) first. Then, modify the pre-trained model path and the local image loading path in the inference script.

  ```shell
  cd sd3/ # # Enter the sd3 directory.
  ```

SD3/SD3.5 model inference:

```shell
vim infer_sd3_text2img.py # Enter the Python file for running T2I inference.
# or
vim infer_sd3_img2img.py # Enter the Python file for running I2I inference.
```

  1. Modify the path.

      ```python
      MODEL_PATH = "stabilityai/stable-diffusion-3.5-large"  # The path can be the SD3/SD3.5 model weight path or the output model path after DreamBooth Fine-tuning.
      DTYPE = torch.float16 # Mixed precision mode can be selected.
      ```

  2. Run the code.

      ```shell
      python infer_sd3_text2img.py  # Single-card text-to-image inference
      python infer_sd3_img2img.py   # Single-card image-to-image inference
      ```

  LoRA fine-tuned SD3 model inference:

  ```shell
  vim infer_sd3_text2img_lora.py
  ```

  1. Modify paths.

      ```python
      MODEL_PATH = "stabilityai/stable-diffusion-3.5-large"  # The path can be the SD3/SD3.5 model weight path or the model output path after DreamBooth fine-tuning.
      LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA weight path
      ```

  2. Run the code.

      ```shell
      python infer_sd3_text2img_lora.py
      ```

  Distributed inference:

  ```shell
  vim infer_sd3_text2img_distrib.py
  ```

- Modify `model_path` to the model weight path or the fine-tuning weight path.
- For LoRA fine-tuning, modify `lora_weights` to the LoRA weight path.

  ```python
  model_path = "stabilityai/stable-diffusion-3.5-large"  # Model weight/Fine-tuning weight path
  lora_weights = "/pytorch_lora_weights.safetensors"  # LoRA weight path
  ```

- Launch the distributed inference script.

  - Since `accelerate` is used for distributed inference, the `config` can be set with: `--num_processes=Number of Cards`, `num_machines=Number of Machines`, etc.

  ```shell
  accelerate launch --num_processes=4 infer_sd3_text2img_distrib.py # Distributed inference on a single machine with four cards
  ```

## Evaluation Using Baseline Dataset

## Environment Variable Declaration

| Environment Variable            | Description                                                                 | Value Description                                                                                                               |
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

For the public IP addresses involved in the code, refer to [Public Address Statement](../../../docs/en/public_address_statement.md).
