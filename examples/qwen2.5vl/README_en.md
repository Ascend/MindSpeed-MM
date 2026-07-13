# Qwen2_5_VL User Guide

<p align="left">
</p>

## Contents

- [Qwen2\_5\_VL User Guide](#qwen2_5_vl-user-guide)
  - [Contents](#contents)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Installation](#environment-installation)
    - [1. Environment Preparation](#1-environment-preparation)
    - [2. Environment Setup](#2-environment-setup)
  - [Weight Download and Offline Conversion](#weight-download-and-offline-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion (hf2mm)](#2-weight-conversion-hf2mm)
    - [3. Weight Conversion (mm2hf)](#3-weight-conversion-mm2hf)
    - [4. Re-partitioning Weights After Training](#4-re-partitioning-weights-after-training)
    - [5. LoRA Weight Conversion (LoRA-hf2mm)](#5-lora-weight-conversion-lora-hf2mm)
    - [6. LoRA Weight Conversion (LoRA-mm2hf)](#6-lora-weight-conversion-lora-mm2hf)
  - [Weight Download and Online Loading](#weight-download-and-online-loading)
    - [1. Weight Download](#1-weight-download-1)
    - [2. Online Loading](#2-online-loading)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download (Taking the COCO2017 Dataset as an Example)](#1-dataset-download-taking-the-coco2017-dataset-as-an-example)
    - [2. Pure Text or Mixed Image-Text Training Data (Using LLaVA-Instruct-150K as an Example)](#2-pure-text-or-mixed-image-text-training-data-using-llava-instruct-150k-as-an-example)
  - [Fine-tuning](#fine-tuning)
    - [Long Sequence](#long-sequence)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
    - [4. FSDP2 Training](#4-fsdp2-training)
  - [Inference](#inference)
    - [1. Parameter Configuration](#1-parameter-configuration)
    - [2. Start Inference](#2-start-inference)
  - [Qwen2.5-VL Video Understanding](#qwen25-vl-video-understanding)
    - [1. Load the Video Dataset](#1-load-the-video-dataset)
    - [2. Modify Model Configuration](#2-modify-model-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning-1)
  - [Evaluation](#evaluation)
    - [Dataset Preparation](#dataset-preparation)
    - [Parameter Configuration](#parameter-configuration)
    - [Start Evaluation](#start-evaluation)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Notes](#notes)

## Version Description

### Reference Implementation

```shell
url=https://github.com/hiyouga/LLaMA-Factory.git
commit_id=52f2565
# transformers version
url=https://github.com/huggingface/transformers.git
commit_id=fa56dcc
```

### Changelog

2025.03.26: Initial support for Qwen2.5-VL model
2025.05.29: Synchronized data processing modifications from the open-source repository

<a id="jump1"></a>

## Environment Installation

<a id="jump1.1"></a>

### 1. Environment Preparation

It is recommended to use the matching environment version for model development.

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
git checkout 69f41000786438204b5f2ffdb788c055788f7378
# Install MindSpeed and its dependencies.
pip install -e .
cd ..
# Install MindSpeed MM and its dependencies.
pip install -e .
```

<a id="jump2"></a>

## Weight Download and Offline Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the corresponding model weights from the Hugging Face library:

- [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main)
- [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)
- [Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/tree/main)
- [Qwen2.5-VL-72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/tree/main)

Save the downloaded model weights to the local `ckpt/hf_path/Qwen2.5-VL-7B-Instruct` directory.

<a id="jump2.2"></a>

### 2. Weight Conversion (hf2mm)

MindSpeed MM modifies the structure names of some original networks. Use the `mm-convert` tool to convert the original pre-trained weights. This tool enables mutual conversion between Hugging Face weights and MindSpeed MM weights, as well as re-partitioning of PP weights. Refer to [Weight Conversion Tool Usage](../../docs/en/features/mm_convert.md).

```bash
# 3b
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[36]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 1

# 7b
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[12,16]] \
  --cfg.parallel_config.vit_pp_layers [[32,0]] \
  --cfg.parallel_config.tp_size 1

# 32b
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-32B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-32B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[1,9,9,9,9,9,9,9]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0,0,0,0,0]] \
  --cfg.parallel_config.tp_size 2

# 72b
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-72B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-72B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[6,11,11,11,11,11,11,8]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0,0,0,0,0]] \
  --cfg.parallel_config.tp_size 2

# 7b: Weight conversion using a model structure consistent with Hugging Face
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[12,16]] \
  --cfg.parallel_config.vit_pp_layers [[32,0]] \
  --cfg.parallel_config.tp_size 1 \
  --cfg.common_model_config.enable_canonical_hf_struct true
# Where:
# mm_dir: Directory for saving converted weights
# hf_dir: Hugging Face weight directory
# llm_pp_layers: Number of LLM layers partitioned across each card. Ensure this is consistent with `pipeline_num_layers` configured in `model.json`.
# vit_pp_layers: Number of ViT layers partitioned across each card. Ensure this is consistent with `pipeline_num_layers` configured in `model.json`.
# tp_size: Number of TP partitions. Ensure this is consistent with the configuration in the fine-tuning launch script.
# enable_canonical_hf_struct: Whether to adopt a model structure consistent with Hugging Face (no QKV fusion or MLP fusion in the LLM). It is recommended to enable this for LoRA fine-tuning.
```

<a id="jump2.3"></a>

### 3. Weight Conversion (mm2hf)

MindSpeed-MM has modified some structure names of the original network. After fine-tuning, if you need to convert the weights back to the Hugging Face format, you can use the `mm-convert` tool to convert the fine-tuned weights, changing the weight names to be consistent with the original network.

```bash
mm-convert  Qwen2_5_VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-7B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 1

# Adopt a model structure consistent with Hugging Face
mm-convert  Qwen2_5_VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-7B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 1 \
  --cfg.common_model_config.enable_canonical_hf_struct true
# Where:
# save_hf_dir: Directory where mm-to-hf weights are saved after MM fine-tuning is complete.
# mm_dir: Directory where the fine-tuned weights are saved.
# hf_dir: Hugging Face weight directory
# llm_pp_layers: Number of LLM layers partitioned across each card. Note that this must be consistent with `pipeline_num_layers` configured in `model.json`.
# vit_pp_layers: Number of ViT layers partitioned across each card. Note that it must be consistent with `pipeline_num_layers` configured in `model.json`.
# tp_size: Number of TP partitions. Note that it must be consistent with the configuration in the fine-tuning startup script.
# enable_canonical_hf_struct: Whether to adopt a model structure consistent with Hugging Face (no QKV fusion or MLP fusion in LLM). It is recommended to enable this for LoRA fine-tuning.
```

If you need to train with the converted model, synchronously modify the `LOAD_PATH` parameter in `examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`. This path points to the converted or partitioned weights. Ensure it is distinguished from the original weight path `ckpt/hf_path/Qwen2.5-VL-7B-Instruct`.

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
```

<a id="jump2.4"></a>

### 4. Re-partitioning Weights After Training

In the weight download and conversion section, weights are partitioned by PP and TP. After fine-tuning, if you need to re-partition the weights, you can use the `mm-convert` tool to partition the fine-tuned weights.

Note: VPP partitioning is not currently supported.

```bash
mm-convert  Qwen2_5_VLConverter resplit \
  --cfg.source_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.target_dir "ckpt/mm_resplit_pp/Qwen2.5-VL-7B-Instruct" \
  --cfg.source_parallel_config.llm_pp_layers [12,16] \
  --cfg.source_parallel_config.vit_pp_layers [32,0] \
  --cfg.source_parallel_config.tp_size 1 \
  --cfg.target_parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.target_parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.target_parallel_config.tp_size 2
# Where:
# source_dir: Directory where the fine-tuned weights are saved
# target_dir: Directory where you want to save the re-partitioned weights by PP
# source_parallel_config.llm_pp_layers: PP configuration for the LLM during fine-tuning
# source_parallel_config.vit_pp_layers: PP configuration for the ViT during fine-tuning
# source_parallel_config.tp_size: TP configuration during fine-tuning
# target_parallel_config.llm_pp_layers: Desired number of layers for re-partitioning the LLM module
# target_parallel_config.vit_pp_layers: Desired number of layers for re-partitioning the ViT module
# target_parallel_config.tp_size: Desired TP configuration (tp_size cannot exceed num_key_value_heads in the original config.json.)
```

<a id="jump2.5"></a>

### 5. LoRA Weight Conversion (LoRA-hf2mm)

MindSpeed-MM modifies the structure names of the LoRA network. Use the `mm-convert` tool to convert the LoRA pre-trained weights. This tool enables mutual conversion between Hugging Face LoRA weights and MindSpeed MM LoRA weights, as well as re-partitioning of PP weights.

```bash
# LoRA weight conversion for 7b using a model structure consistent with Hugging Face
mm-convert  Qwen2_5_VLConverter lora_hf_to_mm \
 --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct-lora" \
 --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct-lora" \
 --cfg.parallel_config.llm_pp_layers [[12,16]] \
 --cfg.parallel_config.vit_pp_layers [[32,0]] \
 --cfg.parallel_config.tp_size 1 \
 --cfg.common_model_config.enable_canonical_hf_struct true \
 --cfg.common_model_config.model_prefix "base_model.model." \
    --cfg.common_model_config.new_transformers_weight_key true
# Where:
# mm_dir: Directory to save the converted LoRA weights
# hf_dir: Directory to save the Hugging Face LoRA weights
# llm_pp_layers: Number of LLM layers partitioned per device. Ensure this is consistent with the `pipeline_num_layers` configured in `model.json`.
# vit_pp_layers: Number of ViT layers partitioned per device. Ensure this is consistent with the `pipeline_num_layers` configured in `model.json`.
# tp_size: The number of TP partitions. Ensure it matches the configuration in the fine-tuning startup script.
# enable_canonical_hf_struct: Whether to adopt a model structure consistent with Hugging Face (no QKV fusion or MLP fusion in the LLM). This must be enabled for LoRA weight conversion scenarios.
# model_prefix: Removes the prefix ("base_model.model.") generated by PEFT wrapping in Hugging Face weights.
# new_transformers_weight_key: Whether to use Hugging Face weights with the new Qwen2.5-VL weight names.
```

Note: For LoRA weight conversion, set `enable_canonical_hf_struct` to true.

<a id="jump2.6"></a>

### 6. LoRA Weight Conversion (LoRA-mm2hf)

MindSpeed-MM has modified the structure names of the LoRA network. After fine-tuning, if you need to convert the LoRA weights back to the Hugging Face format, you can use the `mm-convert` weight conversion tool to convert the fine-tuned LoRA weights, modifying the weight names to be consistent with the original network.

```bash
# LoRA weight conversion for 7b using a model structure consistent with Hugging Face
mm-convert  Qwen2_5_VLConverter lora_mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-7B-Instruct-lora/" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct-lora/" \
  --cfg.parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 1 \
  --cfg.common_model_config.enable_canonical_hf_struct true \
  --cfg.common_model_config.model_prefix "base_model.model." \
  --cfg.common_model_config.new_transformers_weight_key true
# Where:
# save_hf_dir: Directory for converting LoRA weights back to the Hugging Face model format after fine-tuning.
# mm_dir: Directory where the fine-tuned LoRA weights are saved.
# llm_pp_layers: Number of LLM layers partitioned per device. Ensure this is consistent with the `pipeline_num_layers` configured in `model.json`.
# vit_pp_layers: Number of ViT layers partitioned per device. Ensure this is consistent with the `pipeline_num_layers` configured in `model.json`.
# tp_size: The number of TP partitions. Ensure it matches the configuration in the fine-tuning startup script.
# enable_canonical_hf_struct: Whether to adopt a model structure consistent with Hugging Face (no QKV fusion or MLP fusion in the LLM). This must be enabled for LoRA weight conversion scenarios.
# model_prefix: Remove the prefix ("base_model.model.") generated by PEFT wrapping in Hugging Face weights.
# new_transformers_weight_key: Whether to use Hugging Face weights with the new Qwen2.5VL weight names.
```

Note: For LoRA weight conversion, set `enable_canonical_hf_struct` to `true`.

<a id="jump3"></a>

## Weight Download and Online Loading

<a id="jump3.1"></a>

### 1. Weight Download

Verified models (currently only supports weights loading via TP partitioning) and their weight download links:

- [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main)
- [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)
- [Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/tree/main)

Save the downloaded model weights to the local `ckpt/hf_path/Qwen2.5-VL-7B-Instruct` directory.

<a id="jump3.2"></a>

### 2. Online Loading

If you need to use online weight loading mode for model training, simply assign the downloaded Hugging Face original weights to the `LOAD_PATH` parameter in `examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`.

```shell
LOAD_PATH="ckpt/hf_path/Qwen2.5-VL-7B-Instruct"
```

Also, set `bridge_patch` to `true` in `examples/qwen2.5vl/model_7b.json`.

```shell
    "patch": {
        "bridge_patch": true
    }
```

<a id="jump4"></a>

## Dataset Preparation and Processing

<a id="jump4.1"></a>

### 1. Dataset Download (Taking the COCO2017 Dataset as an Example)

(1) You need to download the [COCO2017 dataset](https://cocodataset.org/#download) and extract it to the `./data/COCO2017` folder under the project directory.

(2) Obtain the description file for the image dataset ([LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)) and download it to the `./data/` path.

(3) Run the data conversion script `examples/qwen2vl/llava_instruct_2_mllm_demo_format.py`. The reference data directory structure after conversion is as follows:

   ```shell
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

Currently, reading multiple datasets separated by `,` (do not add spaces) is supported. To do so, modify `dataset_param->basic_parameters->dataset` in `data.json`:
Change `"./data/mllm_format_llava_instruct_data.json"` to `"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"`.

Also note the `dataset_param->basic_parameters->max_samples` parameter in `data_*b.json` limits the data reading to only `max_samples` entries for quick function verification. For formal training, you can remove this parameter to read all data.

<a id="jump4.2"></a>

### 2. Pure Text or Mixed Image-Text Training Data (Using LLaVA-Instruct-150K as an Example)

This framework now supports pure text/mixed data (mixed training with and without image data).

When constructing the data, for entries containing images, you need to retain the `image` key.

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

When constructing data, for pure text data, remove the `image` key.

```python
{
  "id": your_id,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

<a id="jump5"></a>

## Fine-tuning

### Long Sequence

In multimodal understanding tasks, when the training data contains long videos or high-resolution multi-image sequences, the training task may consume excessive memory due to long sequence lengths, and the default partitioning configuration may not be suitable. This section provides the configuration for long sequence scenarios to support training.

Remove `--swap-attention \` from `finetune_qwen2_5_vl_72b.sh`, change `TP=2` to `TP=8`, `PP=8` to `PP=4`, `CP=1` to `CP=4`, `GRAD_ACC_STEP=96` to `GRAD_ACC_STEP=1`, `--seq-length 1024` to `--seq-length 131072`, and `--context-parallel-algo ulysses_cp_algo` to `--context-parallel-algo megatron_cp_algo`.

In `data_72b.json`, change `"video_max_pixels": 16384` to `"video_max_pixels": 262144`, `"video_fps": 2.0` to `"video_fps": 60.0`, `"video_maxlen": 64` to `"video_maxlen": 768`, `"images": "images"` to `"images": null`, and `"videos": null` to `"videos": "videos"`.

In `model_72b.json`, change `"pipeline_num_layers": [32, 0, 0, 0, 0, 0, 0, 0]` to `"pipeline_num_layers": [32, 0, 0, 0]`, `"pipeline_num_layers": [6, 11, 11, 11, 11, 11, 11, 8]` to `"pipeline_num_layers": [6, 25, 25, 24]`, and `"max_position_embeddings": 128000` to `"max_position_embeddings": 131072`, and add `"recompute_granularity": "full",`, `"recompute_method": "uniform",`, and `"recompute_num_layers": 1,`.

| **Training Data Configuration** | **Model Scale** | **Cluster Scale** | **Model and Partitioning Configuration** | **Performance Data** |
| --------------- | ----------- | ----------- | ----------- | ------------ |
| "video_max_pixels":262144,<br>"video_fps":60.0,<br>"video_maxlen":768,<br>"seq-length":131072 | 72B | 8*8(A3) | TP8<br>PP4(vit pp_layers:[32,0,0,0], llm pp_layers:[6,25,25,24])<br>CP4(context-parallel-algo:megatron_cp_algo)<br>text_decoder full recompute:<br> &ensp; "recompute_granularity": "full",<br> &ensp; "recompute_method": "uniform",<br> &ensp; "recompute_num_layers": 1 |End-to-end TPS: 1105.175|

<a id="jump5.1"></a>

### 1. Prerequisites

Before configuring the script, complete the following prerequisites: **Environment Installation**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. Refer to the corresponding sections for details.

<a id="jump5.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data.json` according to the actual situation, including fields such as `model_name_or_path`, `dataset_dir`, and `dataset`.

Example: If the data and its corresponding JSON are both in the `/home/user/data/` directory, where the JSON directory is `/home/user/data/video_data_path.json`, the configuration is as follows:
Set `dataset_dir` to `/home/user/data/` and `dataset` to `./data/video_data_path.json`
Note that `dataset` must be configured as a relative path here.

Taking Qwen2.5VL-7B as an example, make the following modifications to `data.json`. Note that the weight path specified by `model_name_or_path` is the weight path before conversion.

**Note: Do not configure `cache_dir` to the same mounted directory on multiple machines to avoid conflicts caused by writing to the same file**.

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2.5-VL-7B-Instruct",
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

If a communication TIMEOUT occurs when a large amount of data is loaded, you can add the `dataset_param.basic_parameters.preprocess_on_fly` field in `data_xxb.json` and set it to `true`.

[Model Saving, Loading, and Logging Configuration]

Configure the parameters in `examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh` according to the actual situation, including load and save paths, and the save interval `--save-interval` (Note: Distributed optimizer files are large and saving then is time-consuming, so set the save interval cautiously).

```shell
...
# Load Path
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
# Save Path
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # Do not load optimizer state. Remove this if loading is required.
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
    --log-tps \  # Adding this parameter enables printing the average sequence length of the language module at each step during training, and calculates the throughput in tokens per second after training completes.
"
```

If you need to load weights, optimizer states, etc., from a specific iteration, set `LOAD_PATH` to `"save_dir"`, and modify the `latest_checkpointed_iteration.txt` file content to the specified iteration count
(not supported now).

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Single-Node Execution Configuration]

Configure the parameters in `examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh` as follows:

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

Also note that if all parameters on a certain card are frozen, it will result in no gradients (for example, when `vision_encoder` is frozen with `[30,2,0,0]` and `[0,11,10,7]` configured for PP). In this case, you need to add `--enable-dummy-optimizer` to the `GPT_ARGS` parameter in `finetune_qwen2_5_vl_7b.sh`. For more details, refer to the [dummy_optimizer](../../docs/en/features/dummy_optimizer.md).

[ViT Module Recomputation Configuration (Optional)]

When ViT training is enabled, you can enable recomputation to reduce memory usage (note that this will impact performance). (ViT is frozen by default; to enable it, set `"vision_encoder": {"freeze": false}` in the `model.json` file.)

To enable ViT recomputation, add recomputation-related parameters to the `vision_encoder` section in `model.json`.
The `recompute_granularity` parameter can be used to configure as `full` or `selective`.

1. `full` mode

    All components in the TransformerLayer (layernorm, attention, mlp) undergo recomputation, and the number of layers to recompute can be configured.

    - `recompute_method`: Controls the method for calculating the number of recomputation layers. Valid values are `uniform` (uniform recomputation) or `block` (block-based recomputation).
    - `recompute_num_layers`: Controls the number of layers to recompute, specifying the quantity of layers that need recomputation.

    An example configuration is as follows:

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

2. `selective` mode

    Only recomputes the core_attention component within the attention of the TransformerLayer.

    An example configuration is as follows:

    ```json
    {
      "model_id": "qwen2_5vl",
      "img_context_token_id": 151655,
      "vision_start_token_id": 151652,
      "image_encoder": {
        "vision_encoder": {
          "recompute_granularity": "selective"
        }
      }
    }
    ```

[Hugging Face Equivalent Model Structure Configuration (Optional)]

The Qwen2.5-VL model structure under the Megatron framework differs from the Hugging Face implementation, which may impact training results.

Enabling this feature allows training with a model structure fully consistent with Hugging Face. It is recommended to enable this feature in LoRA fine-tuning scenarios. For a detailed introduction, refer to [canonical_model.md](../../docs/en/features/canonical_model.md).

How to enable:
Enable `canonical_model` in `model_xxb.json`

```json
{
  "model_id": "qwen2_5vl",
  "img_context_token_id": 151655,
  "vision_start_token_id": 151652,
  "image_encoder": {
    "vision_encoder": {
      "model_id": "qwen2vit",
      "canonical_model": true,
      ...
    },
    ...
  },
  "text_decoder": {
    "model_id": "qwen2_5_lm",
    "canonical_model": true,
    ...
  },
  ...
}
```

[LoRA Fine-tuning (Optional)]

LoRA is a general capability of the framework and is currently supported. For parameter introduction, please refer to the [LoRA Feature Guide](../../docs/en/features/lora_finetune.md).

To enable LoRA fine-tuning, add LoRA parameters to the startup script `examples/qwen2.5vl/finetune_qwen2_5_vl_xxb.sh`. The relevant configuration modifications are as follows:

```shell
LORA_ARGS="
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0 \
    --lora-target-modules linear_proj linear_fc2 linear_qkv q_proj k_proj v_proj gate_proj up_proj \
"

torchrun $DISTRIBUTED_ARGS pretrain_vlm.py \
    ...
    $LORA_ARGS \
    ...
```

Here, the `lora-target-modules` parameter must be selected based on the model structure. When the Hugging Face equivalent model structure configuration feature is not enabled, an example configuration for this parameter is as follows:

`--lora-target-modules linear_proj linear_fc2 linear_qkv linear_fc1 \`

If enabled, the `lora-target-modules` parameter must be replaced as follows based on the fine-tuning module:

| Module | Original Parameter | Replacement |
|------------|------------|-----------------------|
| `ViT/LLM` | `linear_fc1` | `gate_proj` `up_proj` |
| `LLM` | `linear_qkv` | `q_proj` `k_proj` `v_proj` |

An example configuration is:

(1) LoRA fine-tuning for the ViT module only:

`--lora-target-modules linear_proj linear_fc2 linear_qkv gate_proj up_proj \`

(2) LoRA fine-tuning for the LLM module only:

`--lora-target-modules linear_proj linear_fc2 q_proj k_proj v_proj gate_proj up_proj \`

(3) LoRA fine-tuning for both the ViT and LLM modules simultaneously:

`--lora-target-modules linear_proj linear_fc2 linear_qkv q_proj k_proj v_proj gate_proj up_proj \`

**Note: To enable the Hugging Face equivalent model structure configuration, set the `enable_canonical_hf_struct` parameter to `true` during weight conversion.**

To load LoRA pre-trained weights, add the LoRA pre-trained weight path in the startup script `examples/qwen2.5vl/finetune_qwen2_5_vl_xxb.sh` and modify `GPT_ARGS`. The relevant configuration changes are as follows:

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-32B-Instruct"
LORA_PATH="ckpt/mm_path/Qwen2.5-VL-32B-Instruct-lora"

# Replace the original `--load $LOAD_PATH \` with `--load-base-model $LOAD_PATH \`.
GPT_ARGS="
 ...
    --load-base-model $LOAD_PATH \
    --load $LORA_PATH \
 ...
"
```

<a id="jump5.3"></a>

### 3. Start Fine-tuning

Take Qwen2.5VL-7B as an example to start the fine-tuning training task.
Differences in loss calculation methods can have varying impacts on training effectiveness. Before starting the training task, please refer to the documentation on loss calculation and select an appropriate loss calculation method. For details, see [vlm_model_loss_calculate_type.md](../../docs/en/features/vlm_model_loss_calculate_type.md).

```shell
bash examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
```

<a id="jump5.4"></a>

### 4. FSDP2 Training

Currently, Qwen2.5VL-72B uses FSDP2 training, and the MFU has reached over 30%.

Reference configurations of 32K long-sequnece training on videos:

  - `model_72b.json`

    ```json
    "max_position_embeddings": 32768,
    ```

  - `data_72b.json`

    ```json
    "video_max_pixels": 262144,
    "video_min_pixels": 0,
    "video_fps": 60.0,
    "video_maxlen": 192
    ```

  - `finetune_qwen2_5_vl_72b_fsdp.sh`

    ```shell
    CP=4
    --seq-length 32768 \
    ```

The current FSDP2 configuration file is located at `examples/qwen2.5vl/fsdp2_config.yaml`. For an introduction to related parameters, refer to the [FSDP2 Feature Guide](../../docs/en/features/fsdp2.md).

Execute the FSDP2 training script:

```shell
bash examples/qwen2.5vl/finetune_qwen2_5_vl_72b_fsdp.sh
```

<a id="jump6"></a>

## Inference

<a id="jump6.1"></a>

### 1. Parameter Configuration

Modify the path configurations in `examples/qwen2.5vl/inference_qwen2_5_vl_7b.json` and `examples/qwen2.5vl/inference_qwen2_5_vl_7b.sh` according to the actual situation, including the tokenizer loading path `from_pretrained`. Note:

(1) The path configured for `tokenizer/from_pretrained` is the original Qwen2.5-VL-7B-Instruct path downloaded from Hugging Face.

(2) The `LOAD_PATH` in the shell file is the model path after weight conversion (supporting PP partitioning).

<a id="jump6.2"></a>

### 2. Start Inference

```shell
bash examples/qwen2.5vl/inference_qwen2_5_vl_7b.sh
```

<a id="jump7"></a>

## Qwen2.5-VL Video Understanding

<a id="jump7.1"></a>

### 1. Load the Video Dataset

The video dataset is sourced from [llamafactory](<https://github.com/hiyouga/LLaMA-Factory/tree/main/data>).

The video is sourced from `mllm_demo_data`. When using it, place the demo data into your own data folder, and also place the `mllm_video_demo.json` from llamafactory into your own data folder.

Then, modify the dataset paths in `data.json` according to the actual situation, including the `model_name_or_path`, `dataset_dir`, and `dataset` fields. Also, modify the `images` and `videos` fields in `attr`.

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./Qwen2.5-VL-7B-Instruct",
            ...
        },
        "basic_parameters": {
            ...
            "dataset_dir": "./data",
            "dataset": "./data/mllm_video_demo.json",
            "cache_dir": "./data/cache_dir",
            ...
        },
        ...
        "attr": {
            "system": null,
            "images": null,
            "videos": "videos",
            ...
        },
    },
    ...
}
```

<a id="jump7.2"></a>

### 2. Modify Model Configuration

In `model.json`, change `img_context_token_id` as follows.

```shell
"img_context_token_id": 151656
```

Note: `img_context_token_id` is the token ID that identifies visual content. It is used to mark the position of visual tokens in the forward pass, so it needs to be modified accordingly based on the input.

<a id="jump7.3"></a>

### 3. Start Fine-tuning

Take Qwen2.5VL-7B as an example to start the fine-tuning training task.

```shell
bash examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
```

<a id="jump8"></a>

## Evaluation

<a id="jump8.1"></a>

### Dataset Preparation

The current model supports evaluation on four datasets: AI2D (test), ChartQA (test), DocVQA (val), and MMMU (val).

- [MMMU_DEV_VAL](https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv)
- [DocVQA_VAL](https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv)
- [AI2D_TEST](https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv)
- [ChartQA_TEST](https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv)

<a id="jump8.2"></a>

### Parameter Configuration

To perform evaluation, pass the name and path of the dataset to be evaluated to `examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.json`. The fields that need to be changed are:

- Change `from_pretrained` in `tokenizer` to the Hugging Face weights of Qwen2.5-VL. Refer to the link above to download and pass them in.
- Set `dataset_path` to the local path of the evaluation dataset mentioned above.
- Set `evaluation_dataset` to the name of the evaluation dataset. The available names are (`ai2d_test`, `mmmu_dev_val`, `docvqa_val`, `chartqa_test`). Note that this must correspond to the dataset path above.
- Set `result_output_path` to the output path for the evaluation results. Note that before each evaluation, delete the evaluation files previously saved in this path.

```json
    "tokenizer": {
        "from_pretrained": "./Qwen2.5-VL-7B-Instruct",

    },
    "dataset_path": "./AI2D_TEST.tsv",
    "evaluation_dataset":"ai2d_test",
    "evaluation_model":"qwen2_vl_7b",
    "result_output_path":"./evaluation_outputs/"

```

After modifying `examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.json`, pass the path of the JSON file to the `MM_MODEL` field in `examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh`.

Also, pass the model after the weight conversion mentioned above to the `LOAD_PATH` field in `examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh`.

```shell
MM_MODEL=examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.json
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"

```

To enable multi-card DP evaluation, change the number of NPUs.

```shell
NPUS_PER_NODE=8
```

<a id="jump8.3"></a>

### Start Evaluation

Evaluation requires additional Python packages. Install them using the following command:

```shell
pip install -e ".[evaluate]"
```

<a id="jump8.4"></a>
Start the shell to begin evaluation.

```shell
# Execute in the MindSpeed-MM directory:
bash examples/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh
```

Evaluation results will be output to the `result_output_path` directory, generating the following result files:

- `*.xlsx` file: contains detailed information such as the prediction result and answer for each question.
- `\*.csv` file: contains statistical accuracy and other data.

<a id="jump9"></a>

## Environment Variable Declaration

| Environment Variable | Description | Value Description |
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

<a id="jump10"></a>

## Notes

1. In `finetune_xx.sh`, parameters related to the model structure do not take effect. The configuration of parameters with the same name in `examples/mindspore/qwen2.5vl/model_xb.json` takes precedence. Training-related parameters that are not part of the model structure should be modified in `finetune_xx.sh`.
2. If an Out Of Memory error occurs during sing-card 3B-sized model training, you can use multiple cards in parallel and enable the distributed optimizer for training.
3. When `use_remove_padding` is set to `true` in `model.json`, the attention_mask shape is currently fixed to `[2048, 2048]` in `examples/mindspore/qwen2vl/dot_product_attention.py`. If you need to change it, please refer to the [FlashAttentionScore](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0027.html) in the Ascend community.
