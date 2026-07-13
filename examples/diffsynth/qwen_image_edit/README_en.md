# Diffsynth-Studio

<p align="left">
</p>

- [Diffsynth-Studio](#diffsynth-studio)
- [Qwen Image Edit](#qwen-image-edit)
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
    - [Environment Variable Declaration](#environment-variable-declaration)
  - [Reference](#reference)
    - [Public Address Statement](#public-address-statement)

<a id="jump1"></a>

# Qwen Image Edit

## Model Introduction

Qwen Image Edit is an image editing functionality extended from the Qwen Image base model, which introduces an input image editing mechanism to enable modification of existing images. Qwen Image Edit inherits the complete DiT architecture QwenImageDit and other core components such as QwenImageTextEncoder and QwenImageVAE from Qwen Image. It uses Qwen2VLProcessor to process editing instructions containing images and text, employing a special prompt template to jointly encode the image to be edited and text instructions. The core structural innovation of the Qwen Image series lies in using MSRoPE to resolve the confusion between text and image positions. Functionally, it features excellent multilingual text rendering (especially Chinese) and precise image editing, while also possessing powerful image generation capabilities.

## Version Notes

### Reference Implementation

  ```shell
  url=https://github.com/modelscope/Diffsynth-Studio
  commit_id=084bc2fc78422fd15b37f7a8db02ad924eaf2917
  ```

### Change Log

2025.11.18: Initial release of Qwen Image Edit

## Fine-tuning

### Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md)

1. Install the software and driver.

    ```bash
    # Create and activate a Python environment, then install torch and torch_npu. Please refer to the Installation Guide above.

    # Modify the environment variable paths in the shell script to the actual paths. Example:
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. Clone the repository to the local server..

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    ```

3. Set the model.

    3.1 Download the Qwen Image Edit project [GitHub Reference Implementation](https://github.com/modelscope/Diffsynth-Studio) and execute the following command in the model root directory to install the dependencies required for the corresponding PyTorch version.

    ```shell
    git clone https://github.com/modelscope/DiffSynth-Studio.git
    cd DiffSynth-Studio
    git checkout 084bc2f
    bash ../MindSpeed-MM/examples/diffsynth/qwen_image_edit/replace_npu_patch.sh
    ```

    3.2 Install other required dependency libraries.

    ```shell
    pip install -e .
    pip install -r requirements.txt # Install the corresponding dependencies
    ```

<a id="jump2"></a>

## Fine-tuning

1. Download weights.

   You can visit the ModelScope official website to download the complete [Qwen Image Edit Model Weights](https://modelscope.cn/models/Qwen/Qwen-Image-Edit).

2. Prepare the fine-tuning dataset.

   Download the [UltraEdit dataset](https://huggingface.co/datasets/BleachNick/UltraEdit). This dataset is large in scale, so you can download just one parquet file. The parquet format data needs to be converted into the format specified below. You can also build your own dataset, and the constructed dataset should meet the following requirements:

    - The dataset folder should be organized according to the following standard format:

      ```shell
      dataset_name
      ├── metadata_edit.csv
      └── images
            ├── edited_00010000.jpg
            ├── source_00010000.jpg
            └── ......
      ```

      > **Note:**
      >`dataset_name` represents the name of the dataset.
      >`metadata_edit.csv` is the structured data index/configuration file, which uses a table format to record information such as data paths, labels, metadata, and pairing relationships.
      >`images` is the folder for storing images, and all images are placed in this folder.
      >

    - An example of how `metadata_edit.csv` records information for a pair of input images required for an image editing task is as follows:

      | image | edit_image | prompt | sample_id |
      | :----: | :----: | :----: | :----: |
      | images/source_00010000.jpg | images/edited_00010000.jpg | transform the bird into a butterfly | 10000 |
      | images/source_00010001.jpg | images/edited_00010001.jpg | change the cat's fur color to orange | 10001 |

      > **Note:**
      >`image` is the image before editing, with the filename starting with `source_` followed by the image sequence number. `edit_image` is the image after editing, with the filename starting with `edited_` followed by the image sequence number.
      >`prompt` is the image editing instruction.
      >`sample_id` is the image sequence number. A pair of `image` and `edit_image` shares the same image sequence number.
      >

3. Configure the LoRA fine-tuning script.

   After downloading the weights and dataset, you can modify the model fine-tuning shell script `train_qwen_image_edit_lora.sh` according to the actual storage paths.

   - Modify `transformer_path`, `text_encoder_path`, `model_paths`, `tokenizer_path`, and `processor_path` according to the weight paths.
    `model_paths` stores the weight paths for the three model components: transformer, text_encoder, and vae. For weights saved using multiple shard files, you need to enclose them in `[]` and separate them with `,` and newline characters. An example of the modified `model_paths` parameter is as follows:

      ```shell
      transformer_path="Qwen/Qwen-Image-Edit/transformer"
      text_encoder_path="Qwen/Qwen-Image-Edit/text_encoder"
      model_paths='[
          [
              "'"${transformer_path}"'/diffusion_pytorch_model-00001-of-00005.safetensors",
              "'"${transformer_path}"'/diffusion_pytorch_model-00002-of-00005.safetensors",
              "'"${transformer_path}"'/diffusion_pytorch_model-00003-of-00005.safetensors",
              "'"${transformer_path}"'/diffusion_pytorch_model-00004-of-00005.safetensors",
              "'"${transformer_path}"'/diffusion_pytorch_model-00005-of-00005.safetensors"
          ],
          [
              "'"${text_encoder_path}"'/model-00001-of-00004.safetensors",
              "'"${text_encoder_path}"'/model-00002-of-00004.safetensors",
              "'"${text_encoder_path}"'/model-00003-of-00004.safetensors",
              "'"${text_encoder_path}"'/model-00004-of-00004.safetensors"
          ],
          "/path/Qwen-Image-Edit/vae/diffusion_pytorch_model.safetensors"
      ]'
      ```

      `tokenizer_path` stores the tokenizer, and `processor_path` stores the processor:

      ```shell
      tokenizer_path="Qwen/Qwen-Image-Edit/tokenizer"
      processor_path="Qwen/Qwen-Image-Edit/processor"
      ```

   - Modify `dataset_base_path` and `dataset_metadata_path` according to the dataset path.
      `dataset_base_path` stores `dataset_name` of the constructed dataset, and `dataset_metadata_path` stores the CSV file path:

      ```shell
      dataset_base_path="/path/dataset"
      dataset_metadata_path="/path/dataset/metadata_edit.csv"
      ```

   - Modify the `accelerate` configuration according to the distributed training setup
      `accelerate_config.yaml`. By default, a single-node 8-card setup with the ZeRO Stage 2 sharding strategy is used for training. If you need to change the LoRA training configuration, please modify it in `./examples/qwen_image/model_training/lora/accelerate_config.yaml`, and place the relative path of this file after the `config_file` parameter.

   - Control training steps based on dataset size, `dataset_repeat`, and `num_epochs`.
      `dataset_repeat` refers to the number of times when samples in the dataset are repeated, while `num_epochs` refers to the number of training epochs. `dataset_repeat` should be set to a smaller value to prevent out-of-memory issues during training. The number of training steps for Qwen Image Edit is calculated using the following formula:

      ```shell
      training_steps = len(dataloader) × num_epochs
      ```

4. Modify the code file.

    1. Patch applying based on NPU characteristics

      Open the `train.py` file:

        ```shell
        vim ./examples/qwen_image/model_training/train.py # Enter the Python file
        ```

        - Import section (insert code before line 4)

        ```python
        # Add code
        from diffsynth.models.qwen_image_edit_patch import apply_patches
        apply_patches()

        from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig # Original code
        ```

    2. (Optional) Multi-node execution

        Modify the `config` file:

        ```bash
        vim ./examples/qwen_image/model_training/lora/accelerate_config.yaml
        ```

        Uncomment `deepspeed_multinode_launcher`, `main_process_ip`, and `main_process_port` in the file for use. The main node IP must be consistent across all machines forming the multi-node setup. Modify `num_machines` and `num_processes` according to actual usage needs. The CANN version must be consistent across all machines.

        ```shell
            zero_stage: 2
            deepspeed_multinode_launcher: standard
          main_process_ip: localhost  # Main node IP
          main_process_port: 6000     # Main node port
          machine_rank: 0             # Rank of the current machine
          num_machines: 1             # Total number of machines
          num_processes: 8            # Total number of cards
        ```

5. Start the Qwen Image Edit fine-tuning script.

    The `train_qwen_image_edit_lora` fine-tuning script is provided for this task, supporting multi-card training.

    Start the Qwen Image Edit LoRA fine-tuning script:

    ```shell
    bash examples/qwen_image/model_training/lora/train_qwen_image_edit_lora.sh
    ```

### Fine-tuning Performance

#### Throughput

Performance comparison of Qwen Image Edit on the  **Ascend chip** and **benchmark chip**:

| Chip | Card Count | Task | FPS | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Qwen Image Edit-LoRA fine-tuning | 20.59 | 8 | bf16 | 2.7.1 | ✔ |
| Competitor A | 8p | Qwen Image Edit-LoRA fine-tuning | 17.47 | 8 | bf16 | 2.7.1 | ✔ |

## Inference

### Environment Setup and Execution

See the "Fine-tuning" section.

Qwen Image Edit inference:

    ```shell
    vim ./examples/qwen_image/model_inference/inference_qwen_image_edit_bf16.py # Enter the Python file for running inference.
    ```

1. Modify the paths.

    ```python
    transformer_path = "Qwen/Qwen-Image-Edit/transformer"
    transformer_files = "${transformer_path}/diffusion_pytorch_model*.safetensors"

    text_encoder_path = "Qwen/Qwen-Image-Edit/text_encoder"
    text_encoder_files = "${text_encoder_path}/model*.safetensors"

    vae_file = "Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors"

    tokenizer_file = "Qwen/Qwen-Image/tokenizer"

    processor_file = "Qwen/Qwen-Image/processor"
    ```

    If `transformer_files` and `text_encoder_files` are sharded multi-file weights, they need to be enclosed in `[]` and separated by `,` and newline characters.

    ```python
    transformer_path = "Qwen/Qwen-Image-Edit/transformer"
    transformer_files = [
                "${transformer_path}/diffusion_pytorch_model-00001-of-00005.safetensors",
                "${transformer_path}/diffusion_pytorch_model-00002-of-00005.safetensors",
                "${transformer_path}/diffusion_pytorch_model-00003-of-00005.safetensors",
                "${transformer_path}/diffusion_pytorch_model-00004-of-00005.safetensors",
                "${transformer_path}/diffusion_pytorch_model-00005-of-00005.safetensors"
            ]

    text_encoder_path = "Qwen/Qwen-Image-Edit/text_encoder"
    text_encoder_files = [
                "${text_encoder_path}/model-00001-of-00004.safetensors",
                "${text_encoder_path}/model-00002-of-00004.safetensors",
                "${text_encoder_path}/model-00003-of-00004.safetensors",
                "${text_encoder_path}/model-00004-of-00004.safetensors"
            ]
    ```

2. Create an inference result path.

    ```shell
    mkdir -p inference
    ```

3. Run the code.

    ```shell
    # Modify the ascend-toolkit path according to the actual situation.
    source /usr/local/Ascend/cann/set_env.sh
    python examples/qwen_image/model_inference/inference_qwen_image_edit_bf16.py
    ```

LoRA fine-tuned Qwen Image Edit inference:

    ```shell
    vim ./examples/qwen_image/model_inference/inference_qwen_image_edit_lora_bf16.py
    ```

1. Modify the paths.

    ```python
    transformer_path = "Qwen/Qwen-Image-Edit/transformer"
    transformer_files = "${transformer_path}/diffusion_pytorch_model*.safetensors"

    text_encoder_path = "Qwen/Qwen-Image-Edit/text_encoder"
    text_encoder_files = "${text_encoder_path}/model*.safetensors"

    vae_file = "Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors"

    tokenizer_file = "Qwen/Qwen-Image/tokenizer"

    processor_file = "Qwen/Qwen-Image/processor"

    lora_path = "Qwen-Image-LoRA/model.safetensors"
    ```

    `lora_path` is the absolute path where the LoRA weights are stored; other weights are the same as above.

2. Create an inference result path.

    ```shell
    mkdir -p inference
    ```

3. Run the code.

    ```shell
    # Modify the ascend-toolkit path according to the actual situation.
    source /usr/local/Ascend/cann/set_env.sh
    python examples/qwen_image/model_inference/inference_qwen_image_edit_lora_bf16.py
    ```

<a id="jump3"></a>

### Environment Variable Declaration

| Environment Variable              | Description                                                                 | Value Description                                                                                                               |
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

The code involves public URL references. See [Public Address Statement](../../../docs/en/public_address_statement.md).
