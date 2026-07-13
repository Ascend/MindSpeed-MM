# Qwen2_VL Usage Guide

<p align="left">
</p>

## Contents

- [Qwen2\_VL Usage Guide](#qwen2_vl-usage-guide)
  - [Contents](#contents)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Installation](#environment-installation)
    - [1. Environment Preparation](#1-environment-preparation)
    - [2. Environment Setup](#2-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion (hf2mm)](#2-weight-conversion-hf2mm)
    - [3. Converting Trained Weights Back to Hugging Face Format](#3-converting-trained-weights-back-to-hugging-face-format)
    - [4. Re-shard Weights After Training](#4-re-shard-weights-after-training)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download (Taking the COCO2017 Dataset as an Example)](#1-dataset-download-taking-the-coco2017-dataset-as-an-example)
    - [2. Pure Text or Mixed Training Data with and without Images (Taking LLaVA-Instruct-150K as an Example)](#2-pure-text-or-mixed-training-data-with-and-without-images-taking-llava-instruct-150k-as-an-example)
  - [Fine-tuning](#fine-tuning)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
  - [Inference](#inference)
    - [1. Prerequisites](#1-prerequisites-1)
    - [2. Parameter Configuration](#2-parameter-configuration-1)
    - [3. Start Inference](#3-start-inference)
  - [Qwen2VL Video Understanding Support](#qwen2vl-video-understanding-support)
    - [1. Loading Video Datasets](#1-loading-video-datasets)
    - [2. Modify Model Configuration](#2-modify-model-configuration)
    - [3. Qwen2VL Supporting Video Inference](#3-qwen2vl-supporting-video-inference)
  - [Qwen2VL Supporting DPO Algorithm](#qwen2vl-supporting-dpo-algorithm)
    - [1. Dataset Preparation and Processing (Taking RLHF-V as an Example)](#1-dataset-preparation-and-processing-taking-rlhf-v-as-an-example)
    - [2. Parameter Configuration](#2-parameter-configuration-2)
    - [3. Start DPO Task](#3-start-dpo-task)
  - [Evaluation](#evaluation)
    - [Dataset Preparation](#dataset-preparation)
    - [Parameter Configuration](#parameter-configuration)
    - [Start Evaluation](#start-evaluation)
  - [Feature Usage](#feature-usage)
    - [LoRA Fine-tuning](#lora-fine-tuning)
    - [Unaligned CP Partitioning](#unaligned-cp-partitioning)
    - [Unaligned SP Partitioning](#unaligned-sp-partitioning)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Notes](#notes)

## Version Description

### Reference Implementation

```shell
url=https://github.com/hiyouga/LLaMA-Factory.git
commit_id=52f2565
```

### Changelog

- Oct 21, 2024: Initial support for Qwen2-VL model
- Mar 26, 2025: Synchronized open-source repository data template modifications
- May 29, 2025: Synchronized data processing modifications in the open-source repository

<a id="jump1"></a>

## Environment Installation

<a id="jump1.1"></a>

### 1. Environment Preparation

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md) to complete the Ascend software installation.

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
```

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the corresponding model weights from the Hugging Face library:

- [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/main)

- [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/tree/main)

- [Qwen2-VL-72B](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct/tree/main)

 Save the downloaded model weights to the local `ckpt/hf_path/Qwen2-VL-*B-Instruct` directory. (`*` indicates the corresponding size.)

<a id="jump2.2"></a>

### 2. Weight Conversion (hf2mm)

MindSpeed MM has modified some of the original network structure names. Use the `mm-convert` tool to convert the original pre-trained weights. This tool enables mutual conversion between HuggingFace weights and MindSpeed MM weights, as well as re-sharding of PP weights. Refer to [Weight Conversion Tool Usage](../../docs/en/features/mm_convert.md) for more details.

```bash
# 2b
mm-convert  Qwen2VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2-VL-2B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-2B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[28]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 1

# 7b
mm-convert  Qwen2VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
  --cfg.parallel_config.tp_size 1

# 7b vpp
mm-convert  Qwen2VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2-VL-7B-Instruct-vpp" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[0,0,0,1],[4,4,4,4],[4,3,2,2]] \
  --cfg.parallel_config.vit_pp_layers [[10,10,10,2],[0,0,0,0],[0,0,0,0]] \
  --cfg.parallel_config.tp_size 1

# 72b
mm-convert  Qwen2VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2-VL-72B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-72B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[5,11,11,11,11,11,11,9]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0,0,0,0,0]] \
  --cfg.parallel_config.tp_size 2
# Where:
# mm_dir: Directory to save the converted weights
# hf_dir: Hugging Face weight directory
# llm_pp_layers: Number of LLM layers split on each card. Note that this must be consistent with pipeline_num_layers configured in model.json.
# vit_pp_layers: Number of ViT layers split on each card. Note that this must be consistent with pipeline_num_layers configured in model.json.
# tp_size: Number of TP partitions. Ensure it matches the configuration in the fine-tuning startup script.
```

If you need to train with the converted model, synchronously modify the `LOAD_PATH` parameter in `examples/qwen2vl/finetune_qwen2vl_7b.sh`. This path should point to the converted or sharded weights. Ensure it is distinguished from the original weight path `ckpt/hf_path/Qwen2-VL-7B-Instruct`.

```shell
LOAD_PATH="ckpt/mm_path/Qwen2-VL-7B-Instruct"
```

<a id="jump2.3"></a>

### 3. Converting Trained Weights Back to Hugging Face Format

MindSpeed MM modifies the structure names of some original network layers. After fine-tuning, if you need to convert the weights back to the Hugging Face format, you can use the `mm-convert` tool to transform the fine-tuned weights, renaming them to match the original network structure.

```bash
mm-convert  Qwen2VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2-VL-7B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 1
# Where:
# save_hf_dir: Directory to save weights converted to the Hugging Face format after MindSpeed MM fine-tuning.
# mm_dir: Directory to save the fine-tuned weights
# hf_dir: Hugging Face weight directory
# llm_pp_layers: Number of LLM layers split on each card. Note that this must be consistent with pipeline_num_layers configured in model.json.
# vit_pp_layers: Number of ViT layers split on each card. Note that this must be consistent with pipeline_num_layers configured in model.json.
# tp_size: Number of TP partitions. Ensure it matches the configuration in the fine-tuning startup script.
```

<a id="jump2.4"></a>

### 4. Re-shard Weights After Training

The Weight Download and Conversion section performs PP and TP sharding on the weights. After fine-tuning, if you need to re-shard the weights, you can use the `mm-convert` weight conversion tool to shard the fine-tuned weights.

```bash
mm-convert  Qwen2VLConverter resplit \
  --cfg.source_dir "ckpt/mm_path/Qwen2-VL-7B-Instruct" \
  --cfg.target_dir "ckpt/mm_resplit_pp/Qwen2-VL-7B-Instruct" \
  --cfg.source_parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.source_parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.source_parallel_config.tp_size 1 \
  --cfg.target_parallel_config.llm_pp_layers [4,24] \
  --cfg.target_parallel_config.vit_pp_layers [32,0] \
  --cfg.target_parallel_config.tp_size 1
# Where:
# source_dir: Directory to save fine-tuned weights
# target_dir: Directory where you want to save weights after PP re-partitioning
# source_parallel_config.llm_pp_layers: PP configuration of the LLM during fine-tuning
# source_parallel_config.vit_pp_layers: PP configuration of the ViT during fine-tuning
# source_parallel_config.tp_size: TP configuration during fine-tuning
# target_parallel_config.llm_pp_layers: Desired number of re-sharded layers for the LLM module
# target_parallel_config.vit_pp_layers: Desired number of re-sharded layers for the ViT module
# target_parallel_config.tp_size: Desired TP configuration (tp_size cannot exceed num_key_value_heads in the original config.json.)
```

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download (Taking the COCO2017 Dataset as an Example)

(1) Download the [COCO2017 dataset](https://cocodataset.org/#download) and extract it to the `./data/COCO2017` folder under the project directory.

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

Currently, reading multiple datasets separated by `,` (do not add spaces) is supported. To do so, modify `dataset_param->basic_parameters->dataset` in `data.json`:
Change `"./data/mllm_format_llava_instruct_data.json"` to `"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"`.

Also, note the configuration of `dataset_param->basic_parameters->max_samples` in `data.json`. It will limit the data read to only `max_samples` entries, which allows for quick feature verification. For formal training, you can remove this parameter to read all the data.

<a id="jump3.2"></a>

### 2. Pure Text or Mixed Training Data with and without Images (Taking LLaVA-Instruct-150K as an Example)

This framework now supports pure text/mixed data (mixed training data with and without images).

When constructing data, retain the `image` key-value pair for data containing images.

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

Remove the `image` key-value pair for pure text data.

```python
{
  "id": your_id,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

<a id="jump4"></a>

## Fine-tuning

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the preliminary preparations, including: **Environment Installation**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. For details, please refer to the corresponding chapters.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data.json` according to the actual situation, including fields such as `model_name_or_path`, `dataset_dir`, and `dataset`.

Example: If the data and its corresponding JSON files are both in the `/home/user/data/` directory, and the JSON directory is `/home/user/data/video_data_path.json`, the configuration is as follows:
Set `dataset_dir` to `/home/user/data/` and
`dataset` to `./data/video_data_path.json`.
Note that `dataset` must be configured as a relative path in this case.

Taking Qwen2VL-7B as an example, make the following modifications to `data.json`. Note that the weight path specified by `model_name_or_path` is the weight path before conversion.

**Note: Do not configure `cache_dir` to the same mounted directory across multiple machines to avoid conflicts caused by writing to the same file**.

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2-VL-7B-Instruct",
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

If you need to load large amounts of data, you can use streaming loading by modifying the `sampler_type` field in `data.json` and adding the `streaming` field. (Note: Once streaming loading is enabled, only `num_workers=0` is currently supported, meaning data is processed by a single process. This may cause performance fluctuations, and the resumable training feature will not be supported.)

```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            ...
            "streaming": true
            ...
        },
        ...
    },
    "dataloader_param": {
        ...
        "sampler_type": "stateful_distributed_sampler",
        ...
    }
}
```

If you need to calculate the validation loss, you need to modify the `eval-interval` and `eval-iters` parameters in the shell script; and add fields within `basic_parameters` in `data.json`.

For non-streaming data, there are two methods: (1) Add the `val_dataset` path based on the actual situation; (2) Add the `val_rate` field to split the training dataset.

For streaming data, only adding the `val_dataset` field is supported.

```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            ...
            "val_dataset": "./data/val_dataset.json",
            "val_max_samples": null,
            "val_rate": 0.1,
            ...
        },
        ...
    },
   ...
    }
}
```

[Model Saving, Loading, and Logging Configuration]

Configure the parameters of `examples/qwen2vl/finetune_qwen2vl_7b.sh` according to the actual situation, including loading, saving paths, and the save interval `--save-interval` (Note: Distributed optimizer files are large and saving them is time-consuming, so please set the saving interval cautiously).

```shell
...
# Load Path
LOAD_PATH="ckpt/mm_path/Qwen2-VL-7B-Instruct"
# Save Path
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # Do not load optimizer state; remove if loading is required.
    --no-load-rng \  # Do not load random number state; remove if loading is required.
    --no-save-optim \  # Do not save optimizer state; remove if saving is required.
    --no-save-rng \  # Do not save random number state; remove if saving is required.
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # Logging Interval
    --save-interval 5000 \  # Save Interval
    ...
    --log-tps \  # Add this parameter to enable printing the average sequence length of the language module at each step during training, and to calculate the throughput in tokens per second after training.
"
```

If you need to load the weights, optimizer states, etc., from a specific iteration, set `LOAD_PATH` to `"save_dir"`, and modify the `latest_checkpointed_iteration.txt` file content to the specified iteration count (not supported now).

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Single-node Running Configuration]

Configure the parameters in `examples/qwen2vl/finetune_qwen2vl_7b.sh` as follows:

```shell
# Modify the ascend-toolkit path according to your actual setup.
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

Also note that if all parameters on a certain card are frozen, it will result in no gradients (for example, when `vision_encoder` is frozen with `[30,2,0,0]` and `[0,11,10,7]` configured for PP). In this case, you need to add `--enable-dummy-optimizer` to the `GPT_ARGS` parameter in `finetune_qwen2vl_7b.sh`. For more details, refer to the [dummy_optimizer](../../docs/en/features/dummy_optimizer.md).

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Take Qwen2VL-7B as an example to start the fine-tuning training task.
Differences in loss calculation methods can have varying impacts on training effectiveness. Before starting the training task, please refer to the documentation on loss calculation and select an appropriate loss calculation method. For details, see  [vlm_model_loss_calculate_type.md](../../docs/en/features/vlm_model_loss_calculate_type.md).

```shell
bash examples/qwen2vl/finetune_qwen2vl_7b.sh
```

<a id="jump5"></a>

## Inference

<a id="jump5.1"></a>

### 1. Prerequisites

The following procedure is performed on the basis of the fine-tuning environment. That is, environment setup, weight download and conversion, and other instructions have been completed. Currently, only inference with PP partitioning is supported.

Additional installation:

```shell
pip install qwen_vl_utils
```

Note: If using the original weights downloaded from Hugging Face, weight conversion is required. During the weight conversion step, set the PP partitioning parameters according to specific requirements.

Note: If you use weights saved in MindSpeed MM, no conversion is needed and they can be loaded directly (ensure partitioning consistency with the training phase).

<a id="jump5.2"></a>

### 2. Parameter Configuration

Modify the path configurations in `examples/qwen2vl/inference_qwen2vl_7b.json` and `examples/qwen2vl/inference_qwen2vl_7b.sh` according to your actual situation, including the tokenizer loading path `from_pretrained` and the image processor path `image_processer_path`. Note:

(1) The path configured for `tokenizer/from_pretrained` is the original Qwen2-VL-7B-Instruct path downloaded from Hugging Face.

(2) The `LOAD_PATH` in the shell file is the model path after weight conversion (supporting PP partitioning).

<a id="jump5.3"></a>

### 3. Start Inference

```shell
bash examples/qwen2vl/inference_qwen2vl_7b.sh
```

Note: For single-card inference, FA must be enabled; otherwise, insufficient memory errors may occur. The switch `--use-flash-attn` is enabled by default, so ensure the FA enabling steps are completed. If using multi-card inference, you need to adjust the corresponding PP parameters and the `NPUS_PER_NODE` (number of NPUs used). Taking PP4 as an example, modify the shell parameters as follows:

```shell
NPUS_PER_NODE=4 # The number of available cards must be greater than PP*TP*CP
PP=4 #PP Parallelism Parameters
```

<a id="jump6"></a>

## Qwen2VL Video Understanding Support

<a id="jump6.1"></a>

### 1. Loading Video Datasets

The video dataset is sourced from [llamafactory](https://github.com/hiyouga/LLaMA-Factory/tree/main/data).

The video is sourced from `mllm_demo_data`. When using it, you need to place this demo data into your own data folder, and also place the `mllm_video_demo.json` from llamafactory into your own data file.

Example: `data_72b_video.json`.

<a id="jump6.2"></a>

### 2. Modify Model Configuration

Taking 72b as an example, you need to modify `model_72b.json`:

```shell
"img_context_token_id": 151656
```

<a id="jump6.3"></a>

### 3. Qwen2VL Supporting Video Inference

Taking the 7B model inference as an example, modify `inference_qwen2vl_7b.json`.

```shell
"img_context_token_id": 151656
```

Modify `prompts` to add a description of the video.

```shell
"prompts": "Describe this video and keep it within 100 words."
```

For inference supporting video, change `image_path` to `video_path`, and change the original image loading path to the video path.

Video data sample:
<https://github.com/hiyouga/LLaMA-Factory/blob/main/data/mllm_demo_data/1.mp4>

Simultaneous use of `image_path` and `video_path` is currently not supported, and mixed inference with images and videos is not supported.

<a id="jump7"></a>

## Qwen2VL Supporting DPO Algorithm

**Currently, only 72B LoRA is supported.**

**The procedure for environment setup, weight download, and weight conversion is the same as that in the fine-tuning section.**

<a id="jump7.1"></a>

### 1. Dataset Preparation and Processing (Taking RLHF-V as an Example)

- Download the dataset: [RLHF-V](https://huggingface.co/datasets/llamafactory/RLHF-V)

- Process the dataset: In the `examples/qwen2vl/rlhfv_2_sharegpt_demo_format.py` file, modify the three paths described below, and then run the script.

  ```python
  # Set this to the path where images are saved
  IMAGE_FOLDER = Path("./data/rlhf_v_images/res")
  # Set this to the path of the processed JSON file
  OUTPUT_JSON_PATH = "./data/rlhf-v.json"
  # Set it to the dataset path downloaded from Hugging Face.
  DATASET_NAME = "./data/datasets/rlhf-v"
  ```

<a id="jump7.2"></a>

### 2. Parameter Configuration

- `data_72b_dpo.json`

  For parameter descriptions, see the fine-tuning section.

  Modify the dataset paths in `data.json` according to the actual situation, including fields such as `model_name_or_path`, `dataset_dir`, and `dataset`.

  For example: place the downloaded weights in `./ckpt/hf_path/Qwen2-VL-72B-Instruct`, and the processed dataset in `./data/rlhf-v.json`.

  The parameter settings in `data_72b_dpo.json` are as follows:

  ```json
      ......
   "dataset_param": {
          "dataset_type": "huggingface",
          "preprocess_parameters": {
              "model_name_or_path": "./ckpt/hf_path/Qwen2-VL-72B-Instruct",
              ......
          },
          "basic_parameters": {
              "template": "qwen2vl",
              "dataset_dir": "./data",
              "dataset": "./data/rlhf-v.json",
              ......
          },
        ......
  ......
  ```

- `model_72b.json`

  For parameter descriptions, see the fine-tuning section.

  Taking a single machine with 8 cards as an example, `pipeline_num_layers` for `vision_encoder` and `text_decoder` in `model_72b.json` needs to be adjusted to:

  ```json
  {
  ...
      "image_encoder": {
          "vision_encoder": {
              "model_id": "qwen2vit",
              "num_layers": 32,

              ...

              "pipeline_num_layers": [32, 0, 0, 0],

              ...
          },
   ...
      },
      "text_decoder": {
          "model_id": "qwen2lm",
          "kv_channels": 128,
          "num_layers": 80,
          "pipeline_num_layers": [17, 21, 22, 20],
          ...
  }
  ...
  ```

- `finetune_qwen2vl_72b_dpo.sh`

   For parameter descriptions, see the fine-tuning section.

  The following introduces the parameter descriptions for DPO:

  | Parameter           | Description                                                         |
  | ------------------- | ------------------------------------------------------------ |
  | `dpo-beta`            | Regularization parameter, balancing reward score and KL divergence. Defaults to `0.1`.                    |
  | `dpo-loss-type`       | Loss calculation method, supporting sigmoid (original DPO scheme). Other methods such as hinge and ipo are not supported as they are unverified. |
  | `dpo-label-smoothing` | Smoothing parameter for loss calculation considering sample noise. The value ranges from 0 to 0.5, and defaults to `0.0`.  |
  | pref-ftx            | Multiplier used when adding SFT loss to DPO loss. Defaults to `0.0`.                    |
  | `ref-model`           | Weight path of the reference model. Resumable training is currently not supported.                     |

<a id="jump7.3"></a>

### 3. Start DPO Task

```shell
bash examples/qwen2vl/finetune_qwen2vl_72b_dpo.sh
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

To perform evaluation, you need to pass the name and path of the dataset to be evaluated into `examples/qwen2vl/evaluate_qwen2vl_7b.json`. The fields that need to be changed are:

- `from_pretrained` in `tokenizer` should be the Hugging Face weights of Qwen2-VL. Refer to the link above to download and pass it.
- `dataset_path` is the local path to the evaluation dataset mentioned above.
- `evaluation_dataset` is the name of the evaluation dataset. The available names are (`ai2d_test`, `mmmu_dev_val`, `docvqa_val`, `chartqa_test`). Note that this must correspond to the dataset path above.
- `result_output_path` is the output path for the evaluation results. Note that before each evaluation, you must delete the evaluation files previously saved in this path.

```json
    "tokenizer": {
        "from_pretrained": "./Qwen2-VL-7B-Instruct",

    },
    "dataset_path": "./AI2D_TEST.tsv",
    "evaluation_dataset":"ai2d_test",
    "evaluation_model":"qwen2_vl_7b",
    "result_output_path":"./evaluation_outputs/"

```

After modifying `examples/qwen2vl/evaluate_qwen2vl_7b.json`, you need to pass the path of the JSON file into the `MM_MODEL` field in `examples/qwen2vl/evaluate_qwen2vl_7b.sh`.

You also need to pass the model obtained after the weight conversion mentioned above into the `LOAD_PATH` field in `examples/qwen2vl/evaluate_qwen2vl_7b.sh`.

```shell
MM_MODEL=examples/qwen2vl/evaluate_qwen2vl_7b.json
LOAD_PATH="./qwen_7b_pp1/Qwen2-VL-7B-Instruct"

```

Evaluation supports multi-card DP evaluation, requiring to change the number of NPUs.

```shell
NPUS_PER_NODE=1
```

<a id="jump8.3"></a>

### Start Evaluation

Evaluation requires additional Python packages. Use the following command to install them.

```shell
pip install -e ".[evaluate]"
```

Start the shell to begin evaluation.

```shell
bash examples/qwen2vl/evaluate_qwen2vl_7b.sh
```

The evaluation results will be output to the `result_output_path` path, generating the following result files:

- `*.xlsx` file: contains outputs detailed information such as the predicted results and answers for each question.
- `*.csv` file: contains outputs statistical data such as accuracy.

<a id="jump9"></a>

## Feature Usage

<a id="jump9.1"></a>

### LoRA Fine-tuning

LoRA is a general capability of the framework and is currently supported. Please refer to the [LoRA Feature Guide](../../docs/en/features/lora_finetune.md).

<a id="jump9.2"></a>

### Unaligned CP Partitioning

For an introduction to non-uniform CP and how to enable it, refer to [unaligned_ulysses_cp](../../docs/en/features/unaligned_ulysses_cp.md).

<a id="jump9.3"></a>

### Unaligned SP Partitioning

For an introduction to non-uniform SP and how to enable it, refer to [unaligned_sequence_parallel](../../docs/en/features/unaligned_sequence_parallel.md).

<a id="jump10"></a>

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

<a id="jump11"></a>

## Notes

1. In `finetune_xx.sh`, parameters related to the model structure do not take effect. The configuration of parameters with the same name in `examples/qwen2vl/model_xb.json` shall prevail. Training-related parameters not related to the model structure should be modified in `finetune_xx.sh`.
