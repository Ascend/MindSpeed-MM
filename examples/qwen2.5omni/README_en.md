# Qwen2_5_Omni Usage Guide

<p align="left">
</p>

## Contents

- [Qwen2\_5\_Omni Usage Guide](#qwen2_5_omni-usage-guide)
  - [Contents](#contents)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Model Introduction](#model-introduction)
  - [Environment Installation](#environment-installation)
    - [1. Environment Preparation](#1-environment-preparation)
    - [2. Environment Setup](#2-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion (hf2mm)](#2-weight-conversion-hf2mm)
    - [3. Weight Conversion (mm2hf)](#3-weight-conversion-mm2hf)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Video and Audio Datasets](#1-video-and-audio-datasets)
    - [2. Pure Text or Mixed Data with and without Images (Taking LLaVA-Instruct-150K as an Example)](#2-pure-text-or-mixed-data-with-and-without-images-taking-llava-instruct-150k-as-an-example)
      - [2.1 Image-Text Dataset Download (Using the COCO2017 Dataset as an Example)](#21-image-text-dataset-download-using-the-coco2017-dataset-as-an-example)
      - [3.2 Loading Image-Text Datasets](#32-loading-image-text-datasets)
      - [2.3 Modify Model Configuration](#23-modify-model-configuration)
  - [Fine-tuning](#fine-tuning)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
  - [Heterogeneous Parallel Fine-tuning](#heterogeneous-parallel-fine-tuning)
    - [1. Prerequisites](#1-prerequisites-1)
    - [2. Parameter Configuration](#2-parameter-configuration-1)
    - [3. Start Heterogeneous Parallel Fine-tuning](#3-start-heterogeneous-parallel-fine-tuning)
  - [Feature Usage](#feature-usage)
    - [LoRA Fine-tuning](#lora-fine-tuning)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Precautions](#precautions)

## Version Description

### Reference Implementation

```shell
url=https://github.com/hiyouga/LLaMA-Factory.git
commit_id=52f2565
# transformers Version
url=https://github.com/huggingface/transformers.git
commit_id=7bb619d
```

### Changelog

2025.06.05: Initial support for the Qwen2.5-Omni model

## Model Introduction

Qwen 2.5-Omni is an end-to-end multimodal large language model designed to perceive various modalities including text, images, audio, and video, while generating text and natural speech responses in a streaming manner.

**Reference Implementation**

```bash
https://github.com/hiyouga/LLaMA-Factory
commit id: 52f25651a2016ddede2283be17cf40c2c1b906ed
```

<a id="jump1"></a>

## Environment Installation

<a id="jump1.1"></a>

### 1. Environment Preparation

It is recommended to use the matching environment version during model development.

Refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md) to complete the Ascend software installation.

<a id="jump1.2"></a>

### 2. Environment Setup

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
mkdir logs data ckpt
# Install the acceleration library.
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
# Install MindSpeed and its dependencies.
pip install -e .
cd ..
# Install MindSpeed MM and its dependencies.
pip install -e .
# Install librosa for audio parsing.
pip install librosa

```

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the corresponding model weights from the Hugging Face library:

- Model address: [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/tree/main)

 Save the downloaded model weights to the local `ckpt/hf_path/Qwen2.5-Omni-7B` directory.

<a id="jump2.2"></a>

### 2. Weight Conversion (hf2mm)

MindSpeed MM has modified the structure names of some original networks. Use the `mm-convert` tool to convert the original pre-trained weights. This tool enables bidirectional conversion between HuggingFace weights and MindSpeed MM weights, as well as re-sharding of PP weights. For more details, refer to [Weight Conversion Tool Usage](../../docs/en/features/mm_convert.md).

```bash

# 7b
mm-convert  Qwen2_5_OmniConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-Omni-7B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-Omni-7B" \
  --cfg.parallel_config.llm_pp_layers [[11,17]] \
  --cfg.parallel_config.vit_pp_layers [[32,0]] \
  --cfg.parallel_config.audio_pp_layers [[32,0]] \
  --cfg.parallel_config.tp_size 1

# Where:
# mm_dir: Directory for saving converted weights
# hf_dir: Hugging Face weight directory
# llm_pp_layers: Number of LLM layers partitioned across each card. Note that this must be consistent with `pipeline_num_layers` configured in `model.json`.
# vit_pp_layers: Number of ViT layers partitioned across each card. Note that it must be consistent with `pipeline_num_layers` configured in `model.json`.
# audio_pp_layers: Number of audio layers partitioned across each card. Note that it must be consistent with `pipeline_num_layers` configured in `model.json`.
# tp_size: Number of TP partitions. Note that it must be consistent with the configuration in the fine-tuning startup script.
```

<a id="jump2.3"></a>

### 3. Weight Conversion (mm2hf)

MindSpeed MM has modified some structural names of the original network. After fine-tuning, if you need to convert the weights back to the Hugging Face format, you can use the `mm-convert` weight conversion tool to convert the fine-tuned weights, modifying the weight names to match the original network.

```bash
mm-convert  Qwen2_5_OmniConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-Omni-7B" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-Omni-7B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-Omni-7B" \
  --cfg.parallel_config.llm_pp_layers [11,17] \
  --cfg.parallel_config.vit_pp_layers [32,0] \
  --cfg.parallel_config.audio_pp_layers [32,0] \
  --cfg.parallel_config.tp_size 1
# Where:
# save_hf_dir: Directory where mm-to-hf weights are saved after MindSpeed MM fine-tuning is complete.
# mm_dir: Directory where the fine-tuned weights are saved.
# hf_dir: Hugging Face weight directory
# llm_pp_layers: Number of LLM layers partitioned across each card. Note that this must be consistent with `pipeline_num_layers` configured in `model.json`.
# vit_pp_layers: Number of ViT layers partitioned across each card. Note that it must be consistent with `pipeline_num_layers` configured in `model.json`.
# audio_pp_layers: Number of audio layers partitioned across each card. Note that it must be consistent with `pipeline_num_layers` configured in `model.json`.
# tp_size: Number of TP partitions. Note that it must be consistent with the configuration in the fine-tuning startup script.
```

If you need to train with the converted model, synchronously modify the `LOAD_PATH` parameter in `examples/qwen2.5omni/finetune_qwen2_5_omni_7b.sh`. This path points to the converted or partitioned weights. Ensure it is distinguished from the original weight path `ckpt/hf_path/Qwen2.5-Omni-7B`.

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-Omni-7B"
```

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Video and Audio Datasets

The video dataset is sourced from [llamafactory] <https://github.com/hiyouga/LLaMA-Factory/tree/main/data>.

The videos are sourced from `mllm_video_demo`. When using them, you need to place this demo file into your own data folder, and also place the `mllm_video_audio_demo.json` from llamafactory into your own data folder.

<a id="jump3.2"></a>

### 2. Pure Text or Mixed Data with and without Images (Taking LLaVA-Instruct-150K as an Example)

#### 2.1 Image-Text Dataset Download (Using the COCO2017 Dataset as an Example)

(1) Download the [COCO2017 dataset](https://cocodataset.org/#download) and extract it to the `./data/COCO2017` folder within the project directory.

(2) Obtain the description file for the image dataset ([LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)) and download it to the `./data/` path.

(3) Run the data conversion script: `examples/qwen2vl/llava_instruct_2_mllm_demo_format.py`.

   ```shell
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

Currently, it supports reading multiple datasets separated by `,` (do not add spaces). To do so, modify `dataset_param->basic_parameters->dataset` in `data.json`: Change `"./data/mllm_format_llava_instruct_data.json"` to `"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"`.

Also note the configuration of `dataset_param->basic_parameters->max_samples` in `data.json` limits the data reading to only `max_samples` entries, allowing for quick function verification. For formal training, you can remove this parameter to read all the data.

The framework now supports pure text/mixed data (mixed training with and without image data).

During data construction, for data containing images, retain the `image` key.

```python
{
  "id": your_id,
  "image": your_image_path,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

During data construction, for pure text data, remove the  `image` key.

```python
{
  "id": your_id,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

#### 3.2 Loading Image-Text Datasets

Then, modify the dataset path in `data.json` according to the actual situation, including the `model_name_or_path`, `dataset_dir`, and `dataset` fields, and modify the `images`, `videos`, and `audios` fields in `attr`.

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./Qwen2.5-Omni-7B",
            ...
        },
        "basic_parameters": {
            ...
            "dataset_dir": "./data",
            "dataset": "./data/mllm_format_llava_instruct_data.json",
            "cache_dir": "./data/cache_dir",
            ...
        },
        ...
        "attr": {
            "system": null,
            "images": "images",
            "videos": null,
            "audios": null,
            ...
        },
    },
    ...
}
```

#### 2.3 Modify Model Configuration

In `model.json`, modify `img_context_token_id` as follows:

```shell
"img_context_token_id": 151655
```

Note that the `image_token_id` and `img_context_token_id` parameters serve different purposes. The former is fixed and is the token ID that identifies an image, used in `qwen2_5_omni_get_rope_index` to calculate the number of images in a sequence with image-text input. The latter is the token ID that identifies visual content, used to mark the position of visual tokens in the forward pass, so it needs to be modified accordingly based on the input.

<a id="jump4"></a>

## Fine-tuning

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the preliminary prerequisites, including: **Environment Installation**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data.json` according to the actual situation, including fields such as `model_name_or_path`, `dataset_dir`, and `dataset`.

Taking Qwen2.5Omni-7B as an example, make the following modifications to `data.json`. Note that the weight path specified by `model_name_or_path` is the weight path before conversion.

**Note: Do not configure the same mount directory for `cache_dir` across multiple machines to avoid conflicts caused by writing to the same file**.

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2.5-Omni-7B",
            ...
        },
        "basic_parameters": {
            ...
            "dataset_dir": "./data",
            "dataset": "./data/mllm_format_llava_instruct_data.json",
            "cache_dir": "./data/cache_dir",
            ...
        },
        ...
    },
    ...
}
```

[Model Saving, Loading, and Logging Configuration]

Configure the parameters in `examples/qwen2.5omni/finetune_qwen2_5_omni_7b.sh` according to the actual situation, including the load and save paths, as well as the save interval `--save-interval` (Note: Distributed optimizer files are large and saving them takes a long time, so please set the save interval carefully).

```shell
...
# Load Path
LOAD_PATH="ckpt/mm_path/Qwen2.5-Omni-7B"
# Save Path
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # Do not load the optimizer state. Remove this if loading is required.
    --no-load-rng \  # Do not load the random number state. Remove this if loading is required.
    --no-save-optim \  # Do not save the optimizer state. Remove this if saving is required.
    --no-save-rng \  # Do not save the random number state. Remove this if saving is required.
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # Logging Interval
    --save-interval 5000 \  # Saving Interval
    ...
    --log-tps \  # Adding this parameter enables printing the average sequence length of the language module at each step during training, and calculating the throughput in tokens per second after training ends.
"
```

If you need to load the weights, optimizer states, etc. for a specific iteration, set `LOAD_PATH` to `"save_dir"`, and modify the content of the `latest_checkpointed_iteration.txt` file to the specified iteration count (not supported now).

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Single-Node Running Configuration]

Configure the parameters in `examples/qwen2.5omni/finetune_qwen2_5_omni_7b.sh` as follows:

```shell
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

Note that when PP is enabled, `pipeline_num_layer` configured for `vision_encoder` and `text_decoder` in `model.json` control their respective PP partitioning strategies. For pipeline parallelism, `vision_encoder` must be processed before `text_decoder`.
For example, the default values for the 7B model are `[32,0,0,0]` and `[1,10,10,7]`. This means that within the PP domain, the 32 layers of `vision_encoder`, followed by 1 layer of `text_decoder`, are placed on the first card; the next 10 layers of `text_decoder` are placed on the second card; the next 10 layers of `text_decoder` are placed on the third card; and the next 7 layers of `text_decoder` are placed on the fourth card. The layers of `text_decoder` cannot be placed before layers of `vision_encoder` are fully placed (for example, the configuration `[30,2,0,0]` and `[1,10,10,7]` is incorrect).

Also note that if all parameters on a certain card are frozen, it will result in no gradients (for example, when `vision_encoder` is frozen with `[30,2,0,0]` and `[0,11,10,7]` configured for PP). In this case, you need to add `--enable-dummy-optimizer` to the `GPT_ARGS` parameter in `finetune_qwen2_5_omni_7b.sh.sh`. For more details, refer to the [dummy_optimizer](../../docs/en/features/dummy_optimizer.md).

[Recomputation Configuration (Optional)]

To enable ViT recomputation, add the following three recomputation-related parameters in the `vision_encoder` part of `model.json`.

```json
{
  "model_id": "qwen2_5vl",
  "img_context_token_id": 151655,
  "vision_start_token_id": 151652,
  "image_encoder": {
    "vision_encoder": {
      "recompute_granularity": "full",
      "recompute_method": "uniform",
      "recompute_num_layers": 1
    }
  }
}
```

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Taking Qwen2.5Omni-7B as an example, start the fine-tuning training task.
Differences in loss calculation methods can have varying impacts on training effectiveness. Before starting the training task, please refer to the documentation on loss calculation and select an appropriate loss calculation method. For details, see [vlm_model_loss_calculate_type.md](../../docs/en/features/vlm_model_loss_calculate_type.md).

```shell
bash examples/qwen2.5omni/finetune_qwen2_5_omni_7b.sh
```

<a id="jump5"></a>

## Heterogeneous Parallel Fine-tuning

<a id="jump5.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the prerequisites, including: **Environment Installation**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. Refer to the corresponding sections for details.

The parameters related to weight conversion need to be modified according to the configured heterogeneous parallelism settings (currently, only DP and TP heterogeneous parallelism is supported). For example, when the Vit module and Audio module are not partitioned, and the LLM module is partitioned by `TP4`, the weight conversion script commands are as follows:

```bash
# 7b
mm-convert  Qwen2_5_OmniConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-Omni-7B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-Omni-7B" \
  --cfg.parallel_config.llm_pp_layers [[28]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.audio_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 4 \
  --cfg.parallel_config.vit_tp_size 1 \
  --cfg.parallel_config.audio_tp_size 1

# Where:
# mm_dir: Directory where the fine-tuned weights are saved.
# hf_dir: Hugging Face weight directory
# llm_pp_layers: Number of LLM layers partitioned across each card. Note that this must be consistent with `pipeline_num_layers` configured in `model.json`.
# vit_pp_layers: Number of ViT layers partitioned across each card. Note that it must be consistent with `pipeline_num_layers` configured in `model.json`.
# audio_pp_layers: Number of audio layers partitioned across each card. Note that it must be consistent with `pipeline_num_layers` configured in `model.json`.
# tp_size: Number of TP partitions. Note that it must be consistent with the configuration in the fine-tuning startup script.
# vit_tp_size: Number of TP partitions on the ViT. If not configured, the ViT uses the default value.
# audio_tp_size:  Number of TP partitions on the audio. If not configured, vit uses the default value.
```

<a id="jump5.2"></a>

### 2. Parameter Configuration

Refer to the "fine-tuning" section to configure the data directory, model saving and loading, and other settings. When configuring `examples/qwen2.5omni/finetune_qwen2_5_omni_7b.sh`, you need to add `--hetero-parallel` to enable heterogeneous parallel training.

The parallel configurations for llm, vit, and audio are all defined in the `model_7b.json` file, and the parallel configurations in `examples/qwen2.5omni/finetune_qwen2_5_omni_7b.sh` need to be all set to `1`. The GBS for vit, audio, and llm are consistent, and attention should be paid to the MBS configuration of llm. To ensure that vit and audio have equal amounts of data for computation, the value of llm's `DP * MBS` must be divisible by the DP of vit and the DP of audio. For example, when the vit and audio modules use full DP and llm uses TP4 partitioning, the MBS of llm needs to be set to a multiple of 4 to meet this condition.

```shell
TP=1
PP=1
CP=1
MBS=4
...
GPT_ARGS="
    ...
    --hetero-parallel \  # Enable Heterogeneous Parallelism Training
    ...
"
```

```json
{
    "image_encoder": {
        "vision_encoder": {},
        "vision_projector": {},
        "tp":1,
        "pp":1,
        "cp":1
    },
    "audio_encoder": {
        "audio_encoder": {},
        "tp":1,
        "pp":1,
        "cp":1
    },
    "text_decoder": {
        "tp":4,
        "pp":1,
        "cp":1
    }
}
```

<a id="jump5.3"></a>

### 3. Start Heterogeneous Parallel Fine-tuning

Take Qwen2.5Omni-7B as an example to start the heterogeneous parallel fine-tuning training task.

```shell
bash examples/qwen2.5omni/finetune_qwen2_5_omni_7b.sh
```

<a id="jump7"></a>

## Feature Usage

<a id="jump7.1"></a>

### LoRA Fine-tuning

LoRA is a general capability of the framework and is currently supported. Refer to the [LoRA Feature Guide](../../docs/en/features/lora_finetune.md).

<a id="jump8"></a>

## Environment Variable Declaration

| Environment Variable                      | Description                                                                 | Value Description                                                                                         |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | Specifies whether to enable log printing.                                                          | `0`: Disable.<br>`1`: Enable.                                                                   |
| `ASCEND_GLOBAL_LOG_LEVEL`     | Sets the log level for application logs and the log level for each module; only supports debug logs.                             | `0`: DEBUG level<br>`1`: INFO level<br>`2`: WARNING level<br>`3`: ERROR level<br>`4`: NULL level; no log output |
| `TASK_QUEUE_ENABLE`           | Controls the level of `task_queue` operator dispatch queue optimization.                                    | `0`: Disable.<br>`1`: Enable Level 1 optimization.<br>`2`: Enable Level 2 optimization.                                              |
| `COMBINED_ENABLE`             | Sets the combined flag. Set to `0` to disable this feature; set to `1` to enable, used for optimizing non-contiguous two-operator combination.| `0`: Disable.<br>`1`: Enable.                                                                           |
| `CPU_AFFINITY_CONF`           | Controls the processor affinity of CPU-side operator tasks, i.e., sets task core binding.                                    | Set to `0` or not set: Indicates core binding is not enabled.<br>`1`: Indicates coarse-grained core binding is enabled.<br>`2`: Indicates fine-grained core binding is enabled.                                     |
| `HCCL_CONNECT_TIMEOUT`        | Limits the timeout waiting period for socket connection establishment between different devices.                                  | Must be configured as an integer in the value range `[120,7200]` (unit:s). The default value is `120`.                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | Controls the behavior of the cache allocator.                                                          | `expandable_segments:<value>`: Enables expandable segments of the memory pool, i.e., virtual memory characteristics.                                            |
| `HCCL_EXEC_TIMEOUT`           | Controls the synchronization wait time during execution between devices. Within this configured time, each device process waits for other devices to perform communication synchronization.         | Must be configured as an integer in the value range `[68,17340]` (unit: s). The default value is `1800`.                                                    |
| `ACLNN_CACHE_LIMIT`           | Configures the number of operator information entries cached on the host side by the single-operator execution API.                                  | Must be configured as an integer in the value range `[1, 10,000,000]`. The default value is `10000`.                                                    |
| `TOKENIZERS_PARALLELISM`      | Controls the behavior of the tokenizer in Hugging Face's transformers library in a multi-threading environment    | `False`: Disable parallel tokenization.<br>`True`: Enable parallel tokenization.                                                            |
| `MULTI_STREAM_MEMORY_REUSE`   | Configures whether multi-stream memory reuse is enabled. | `0`: Disable multi-stream memory reuse.<br>`1`: Enable multi-stream memory reuse.                                                               |
| `NPU_ASD_ENABLE`   | Controls whether to enable the feature value detection function of Ascend Extension for PyTorch | Set to `0` or not set: Disable feature value detection.<br>`1`: Enable feature value detection and print only abnormal logs, without alarms.<br>`2`: Enable feature value detection and print alarms.<br>`3`: Enable feature value detection and print alarms, as well as process data in device-side info level logs. |
| `ASCEND_LAUNCH_BLOCKING`   | Controls whether to enable synchronous mode during operator execution. | `0`: Execute operators asynchronously.<br>`1`: Force operators to run in synchronous mode.                                                               |
| `NPUS_PER_NODE`               | Configures the number of NPUs used on a compute node.                                                  | Integer value (e.g., `1`, `8`, etc.)                                                                            |

<a id="jump9"></a>

## Precautions

1. In `finetune_xx.sh`, parameters related to the model structure do not take effect. The configuration of parameters with the same name in `examples/qwen2.5omni/model_xb.json` takes precedence. Training-related parameters that are not part of the model structure should be modified in `finetune_xx.sh`.
2. When you change the training parameter `MBS` in `finetune_xx.sh`, it is recommended to adjust the `--num-workers` parameter accordingly to ensure efficient matching between data loading and model computation settings. Otherwise, training performance may fluctuate or degrade. Typically, it is recommended to set `--num-workers` to a value not less than `MBS`.
