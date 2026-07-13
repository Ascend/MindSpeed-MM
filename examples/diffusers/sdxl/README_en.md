# Diffusers

<p align="left">
</p>

- [Diffusers](#diffusers)
- [Stable Diffusion XL (SDXL)](#stable-diffusion-xl-sdxl)
  - [Model Introduction](#model-introduction)
  - [Pre-training](#pre-training)
    - [Environment Setup](#environment-setup)
    - [Pre-training](#pre-training-1)
    - [Performance](#performance)
      - [Throughput](#throughput)
  - [Fine-tuning](#fine-tuning)
    - [Environment Setup](#environment-setup-1)
      - [LoRA Fine-tuning - Dataset](#lora-fine-tuning---dataset)
      - [ControlNet Fine-tuning - Dataset](#controlnet-fine-tuning---dataset)
      - [Full-parameter Fine-tuning - Dataset](#full-parameter-fine-tuning---dataset)
      - [Obtain Pre-trained Models](#obtain-pre-trained-models)
    - [Fine-tuning](#fine-tuning-1)
    - [Performance](#performance-1)
  - [Inference](#inference)
    - [Environment Setup and Execution](#environment-setup-and-execution)
    - [Performance](#performance-2)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Reference](#reference)
    - [Public Address Statement](#public-address-statement)

<a id="jump1"></a>

# Stable Diffusion XL (SDXL)

## Model Introduction

Diffusion Models are a type of generative models that can generate a wide variety of high-resolution images. Diffusers is a model suite released by Hugging Face, serving as the premier library for state-of-the-art pre-trained diffusion models used to generate images, audio, and even 3D structures of molecules. The suite includes various models based on diffusion models and provides implementations for training and inference across a range of downstream tasks.

- Reference implementation

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=5956b68a6927126daffc2c5a6d1a9a189defe288
  ```

## Pre-training

### Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md).

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

    3.1 Download the [SDXL Reference Implementation](https://github.com/huggingface/diffusers) or execute the following command in the model root directory to install the dependencies required by the corresponding PyTorch version for the model.

    ```shell
    git clone https://github.com/huggingface/diffusers.git -b v0.30.0
    cd diffusers
    git checkout 5956b68a6927126daffc2c5a6d1a9a189defe288
    cp -r ../MindSpeed-MM/examples/diffusers/sdxl ./sdxl
    ```

    Main code path:

    ```shell
    code_path=examples/text_to_image/
    ```

    3.2 Install `{task_pretrain/train}_sdxl_deepspeed_{mixed_precision_fp16/bf16}.sh`.

    Move `collect_dataset.py`, `pretrain_model.py`, `train_text_to_image_sdxl_pretrain.py`, and `patch_sdxl.py` to the `examples/text_to_image/` path.

    ```shell
    # Example: Modify the .py file name to perform four tasks.
    cp ./sdxl/train_text_to_image_sdxl_pretrain.py ./examples/text_to_image/
    ```

    3.3 Install other required dependency libraries.

    ```shell
    pip install -e .
    vim examples/text_to_image/requirements_sdxl.txt #Modify version: torch==2.7.1
    pip install -r examples/text_to_image/requirements_sdxl.txt # Install the corresponding dependencies from the original diffusers repository.
    pip install -r sdxl/requirements_sdxl_extra.txt #Install the corresponding dependencies for SDXL.
    ```

<a id="jump2"></a>

### Pre-training

1. Prepare the pre-training dataset.

    Download and extract the `laion_sx` dataset (this dataset has been decommissioned; other datasets can be used as alternatives) and the [pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions/tree/main) dataset. In the startup shell script below, set the `dataset_name` parameter to the absolute path to the local dataset.

    Modify `dataset_name` in `pretrain_sdxl_deepspeed_**16.sh` to the absolute path of `laion_sx`.

    ```shell
    vim sdxl/pretrain_sdxl_deepspeed_**16.sh
    ```

    Modify `dataset_name` in `train_sdxl_deepspeed_**16.sh` to the absolute path of `pokemon-blip-captions`.

    ```shell
    vim sdxl/train_sdxl_deepspeed_**16.sh
    ```

    The format of the `laion_sx` dataset is as follows:

    ```shell
    laion_sx
    ├── 000000000.jpg
    ├── 000000000.json
    ├── 000000000.txt
    ```

    The format of the `pokemon-blip-captions` dataset is as follows:

    ```shell
    pokemon-blip-captions
    ├── dataset_infos.json
    ├── README.MD
    └── data
          └── train-001.parquet
    ```

    > **Note**:
    >The training script for this dataset is provided as a reference example only.
    >

2. Configure the SDXL pre-training script and pre-trained model.

    When connected to the internet, the pre-trained model can be downloaded using the following steps. Without network access, visit the Hugging Face official website to manually download the [sdxl-base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and the [sdxl-vae model](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).

    ```bash
    export model_name="stabilityai/stable-diffusion-xl-base-1.0" # Pre-trained model path
    export vae_name="madebyollin/sdxl-vae-fp16-fix" # VAE model path
    ```

    After obtaining the corresponding pre-trained model, set the `model_name` parameter to the absolute path of the local pre-trained model and the `vae_name` parameter to the absolute path of the local VAE model.

    ```bash
    scripts_path="./sdxl" # Model root directory (model folder name)
    model_name="stabilityai/stable-diffusion-xl-base-1.0" # Pre-trained model path
    vae_name="madebyollin/sdxl-vae-fp16-fix" # VAE model path
    dataset_name="laion_sx" # Dataset Path
    batch_size=4
    max_train_steps=2000
    mixed_precision="bf16" # Mixed precision
    resolution=1024
    config_file="${scripts_path}/pretrain_${mixed_precision}_accelerate_config.yaml"

    # Under the accelerate launch *** \ directory
    --dataloader_num_workers=8 \ # Adjust num_workers based on the system configuration and data size.
    ```

    Modify the path of `train_text_to_image_sdxl_pretrain.py` under the `accelerate` configuration in the bash file (the default path is `diffusers/sdxl/`).

    ```bash
    accelerate launch --config_file ${config_file} \
      ${scripts_path}/train_text_to_image_sdxl_pretrain.py \  #If the model root directory is sdxl, no modification is needed.
    ```

    Modify the path of `deepspeed_config_file` in `pretrain_fp16_accelerate_config.yaml`:

    ```bash
    deepspeed_config_file: ./sdxl/deepspeed_fp16.json # DeepSpeed JSON file path
    ```

    Modify the `examples/text_to_image/train_text_to_image_sdxl.py` file:

    ```bash
    vim examples/text_to_image/train_text_to_image_sdxl.py
    ```

    1. Modify the version at line 58:

        ```python
        # Change the minimum version from 0.31.0 to 0.30.0
        check_min_version("0.30.0")
        ```

    2. Add code at line 59:

        ```python
        from patch_sdxl import TorchPatcher, config_gc
        TorchPatcher.apply_patch()
        config_gc()
        ```

        <a id="jump2.1"></a>

    3. (Optional)For Ubuntu systems, add `accelerator.print("")` near line 1216:

        ```python
        if global_step >= args.max_train_steps:
          break
        accelerator.print("")
        ```

    4. For FPS printing method, refer to `train_text_to_image_sdxl_pretrain.py`.

3. Start the SDXL pre-training script.

    The **8-card** training scripts for both **mixed precision fp16** and **mixed precision bf16** are provided, with DeepSpeed used for distributed training.

    The **pretrain** model is mainly responsible for the text-to-image training in the second stage.
    The **train** model is mainly responsible for the text-to-image training in the first stage.

    ```shell
    bash sdxl/pretrain_sdxl_deepspeed_**16.sh
    bash sdxl/train_sdxl_deepspeed_**16.sh
    ```

### Performance

#### Throughput

Performance comparison of SDXL on **Ascend chips** and **benchmark chips**:

| Chip | Chip Count | Task | FPS | batch_size | AMP_Type | Torch_Version | DeepSpeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Competitor A | 8p | SDXL_train_bf16 | 30.65 | 4 | bf16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_train_bf16 | 29.92 | 4 | bf16 | 2.1 | ✔ |
| Competitor A | 8p | SDXL_train_fp16 | 30.23 | 4 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_train_fp16 | 28.51 | 4 | fp16 | 2.1 | ✔ |
| Competitor A | 8p | SDXL_pretrain_bf16 | 21.14 | 4 | bf16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_pretrain_bf16 | 19.79 | 4 | bf16 | 2.1 | ✔ |
| Competitor A | 8p | SDXL_pretrain_fp16 | 20.77 | 4 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_pretrain_fp16 | 19.67 | 4 | fp16 | 2.1 | ✔ |

## Fine-tuning

<a id="jump3"></a>

### Environment Setup

#### LoRA Fine-tuning - Dataset

   > **Note**:
   > For environment setup and required dataset (`pokemon-blip-captions`), see [Pre-training](#pre-training).
   >

  ```shell
  sdxl/finetune_sdxl_lora_deepspeed_fp16.sh
  ```

#### ControlNet Fine-tuning - Dataset

   Download the [fill50k](https://huggingface.co/datasets/fusing/fill50k) dataset, set the `dataset_name` parameter in the startup shell script below to the absolute path of the local dataset, and modify the `fill50k.py` file.

   ```shell
   sdxl/finetune_sdxl_controlnet_deepspeed_fp16.sh
   ```

   The DeepSpeed version needs to be changed to `0.14.4`.

   Modify `controlnet/train_controlnet_sdxl.py` as follows, appending `trust_remote_code=True`.

   ```shell
   dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            trust_remote_code=True
          )
   ```

   > **Note**:
   >Modify lines 57 to 59 in the `fill50k.py` file under the dataset directory. Example:
>
   > ```python
   > metadata_path = "dataset_path/fill50k/train.jsonl"
   > images_dir = "dataset_path/fill50k"
   > conditioning_images_dir = "dataset_path/fill50k"
   >```
>
   The `fill50k` dataset format is as follows:

   ```shell
   fill50k
   ├── images
   ├── conditioning_images
   ├── train.jsonl
   └── fill50k.py
   ```

   > **Note**:
   >The training process script for this dataset is provided only as a reference example.

#### Full-parameter Fine-tuning - Dataset

   > **Note**:
   > The dataset is the same as that for LoRA fine-tuning. Please refer to [LoRA Fine-tuning - Dataset](#lora-fine-tuning---dataset).
   >

#### Obtain Pre-trained Models

   Obtain the [sdxl-base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and the [sdxl-vae model](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).

   After obtaining the corresponding pre-trained models, set the `model_name` parameter to the absolute path of the local pre-trained model and the `vae_name` parameter to the absolute path of the local `vae` model in the Controlnet fine-tuning shell script.

   ```shell
   sdxl/finetune_sdxl_controlnet_deepspeed_fp16.sh
   ```

   In the LoRA fine-tuning and full-parameter fine-tuning shell scripts, set the `model_name` parameter to the absolute path of the local pre-trained model.

   ```shell
   sdxl/finetune_sdxl_deepspeed_fp16.sh
   sdxl/finetune_sdxl_lora_deepspeed_fp16.sh
   ```

   > **Note**:
   > The pre-trained model is the same as the one used for pre-training. Please refer to [Pre-training](#pre-training).
   >

<a id="jump3.1"></a>

### Fine-tuning

   (Optional) For Ubuntu systems, add `accelerator.print("")` in `examples/text_to_image/train_text_to_image_lora_sdxl.py` and `examples/controlnet/train_controlnet_sdxl.py`.

   > **Note**:
   > Add the code near line 1235 in `train_text_to_image_lora_sdxl` and near line 1307 in `train_controlnet_sdxl`.
   >

  LoRA weight saving for resumable inference:

   To save `Lora_weights` in checkpointing steps, add the following code (method same as the patch modification in SDXL pre-training):

   ```python
  from patch_sdxl import save_Lora_Weights
  ```

  And near line 1227 in `train_text_to_image_lora_sdxl.py`, add `save_Lora_Weights(unwrap_model, unet, text_encoder_one, text_encoder_two, args.train_text_encoder, save_path)` below `accelerator.save_state(save_path)` as follows:

  ```python
  accelerator.save_state(save_path)
  save_Lora_Weights(unwrap_model, unet, text_encoder_one, text_encoder_two, args.train_text_encoder, save_path)
  logger.info(f"Saved state to {save_path}")
  ```

   Run the fine-tuning script:

    ```shell
    # Single-node 8-card fine-tuning
    # The images required by finetune_sdxl_controlnet_deepspeed_fp16.sh can be downloaded using the following commands.
    # wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
    # wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
    bash sdxl/finetune_sdxl_controlnet_deepspeed_fp16.sh      #   sdxl_controlnet fp16 (8-card DeepSpeed training)
    bash sdxl/finetune_sdxl_lora_deepspeed_fp16.sh            # sdxl_lora fp16 (8-card DeepSpeed training)
    bash sdxl/finetune_sdxl_deepspeed_fp16.sh        # sdxl_finetune fp16 (8-card DeepSpeed training)
    ```

<a id="jump3.2"></a>

### Performance

| Chip | Chip Count | Task | FPS | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Competitor A | 8p | LoRA | 31.74 | 7 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p | LoRA | 26.40 | 7 | fp16 | 2.1 | ✔ |
| Competitor A | 8p | ControlNet | 32.44 | 5 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p | ControlNet | 29.98 | 5 | fp16 | 2.1 | ✔ |
| Competitor A | 8p | Fine-tuning | 164.66 | 24 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p | Fine-tuning | 166.71 | 24 | fp16 | 2.1 | ✔ |

**Note**: The FPS in fine-tuning tasks tends to fluctuate more significantly when the heat sink performance is weak.

## Inference

### Environment Setup and Execution

  See the "fine-tuning" section.

Inference scripts:

- Script configuration for single-node single-card inference
  - `sdxl/sdxl_text2img_lora_infer.py`
    - Set `model_path` to the output directory of LoRA fine-tuning, i.e., `output_path` specified by yourself in `sdxl_text2img_lora_deepspeed.sh`
    - `stabilityai/stable-diffusion-xl-base-1.0`: Visit the Hugging Face official website to [download](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) it.
  - `sdxl/sdxl_text2img_controlnet_infer.py`
    - Set `base_model_path` to `stabilityai/stable-diffusion-xl-base-1.0`. You can visit the Hugging Face official website to [download](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) it.
    - Set `controlnet_path` to the output path of the ControlNet fine-tuning result, i.e., `output_path` specified by yourself in `sdxl_text2img_controlnet_deepspeed.sh`.
  - `sdxl/sdxl_text2img_infer.py`
    - Set `/diffusion/sdxl/pretrained/` to `stabilityai/stable-diffusion-xl-base-1.0`. You can visit the Hugging Face official website to [download](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) it.
    - `/diffusion/sdxl/pretrained/` can also be configured as the output path of the fine-tuning result, i.e., `output_path` specified in the fine-tuning script.
  - `sdxl/sdxl_img2img_infer.py`
    - Set `MODEL_NAME` to `stabilityai/stable-diffusion-xl-base-1.0`. You can visit the Hugging Face official website to [download](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) it.
    - Set `VAE_NAME` to `madebyollin/sdxl-vae-fp16-fix`. You can visit the Hugging Face official website to [download](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) it.
    - `Intel/dpt-hybrid-midas`: You can visit the Hugging Face official website to [download](https://huggingface.co/Intel/dpt-hybrid-midas) it.
    - `diffusers/controlnet-depth-sdxl-1.0-small`: You can visit the Hugging Face official website to [download](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-small) it.

- Execution the inference scripts.

  ```shell
  python sdxl/sdxl_text2img_lora_infer.py        # Text-to-image LoRA fine-tuning with mixed precision fp16
  python sdxl/sdxl_text2img_controlnet_infer.py  # ControlNet fine-tuning with mixed precision fp16
  python sdxl/sdxl_text2img_infer.py             # Text-to-image full-parameter Fine-tuning with mixed precision fp16
  python sdxl/sdxl_img2img_infer.py              # Image-to-image fine-tuning with mixed precision fp16
  ```

Distributed inference:

- Modify the `sdxl/sdxl_text2img_distrib_infer.py` file.

  ```shell
  vim sdxl/sdxl_text2img_distrib_infer.py
  ```

- Modify `model_path` to the model weight path or the fine-tuned weight path.
- For LoRA fine-tuning, modify `lora_weights` to the LoRA weight path.

  ```python
  model_path = "/stabilityai/stable-diffusion-xl-base-1.0"  # Model weight/Fine-tuning weight path
  lora_weights = "/pytorch_lora_weights.safetensors"  # LoRA weight path
  ```

- Execute the distributed inference script.

  - Since `accelerate` is used for distributed inference, `config` can be set with: `--num_processes=number of cards`, `num_machines=number of machines`, etc.

  ```shell
  accelerate launch --num_processes=4 sdxl/sdxl_text2img_distrib_infer.py # Distributed inference on a single machine with four cards
  ```

<a id="jump4"></a>

### Performance

| Chip | Chip Count | Task | E2E (it/s) | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:---:|:---:|:---:|
| Competitor A | 1p | Text-to-Image LoRA | 1.45 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | Text-to-Image LoRA | 2.61 | fp16 | 2.1 | ✔ |
| Competitor A | 1p | Text-to-Image ControlNet | 1.41 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 1p | Text-to-Image ControlNet | 2.97 | fp16 | 2.1 | ✔ |
| Competitor A | 1p | Text-to-Image full-parameter fine-tuning | 1.55 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 1p | Text-to-Image full-parameter fine-tuning | 3.02 | fp16 | 2.1 | ✔ |
| Competitor A | 1p | Image-to-Image | 3.56 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 1p | Image-to-Image | 3.94 | fp16 | 2.1 | ✔ |

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

For public address involved in the code, refer to [Public Address Statement](../../../docs/en/public_address_statement.md).
