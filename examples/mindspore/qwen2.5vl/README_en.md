# Qwen2.5VL (MindSpore Backend) Usage Guide

<p align="left">
</p>

## Contents

- [Qwen2.5VL (MindSpore Backend) Usage Guide](#qwen25vl-mindspore-backend-usage-guide)
  - [Contents](#contents)
  - [Environment Setup](#environment-setup)
    - [1. Repository Cloning and Environment Setup](#1-repository-cloning-and-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion (hf2mm)](#2-weight-conversion-hf2mm)
    - [3. Weight Conversion (mm2hf)](#3-weight-conversion-mm2hf)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download (Using the COCO2017 Dataset as an Example)](#1-dataset-download-using-the-coco2017-dataset-as-an-example)
    - [2. Pure Text or Mixed Training Data with and without Images (Taking LLaVA-Instruct-150K as an Example)](#2-pure-text-or-mixed-training-data-with-and-without-images-taking-llava-instruct-150k-as-an-example)
  - [Fine-tuning](#fine-tuning)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
  - [Inference](#inference)
    - [1. Prerequisites](#1-prerequisites-1)
    - [2. Parameter Configuration](#2-parameter-configuration-1)
    - [3. Start Inference](#3-start-inference)
  - [Qwen2.5VL-based Video Understanding](#qwen25vl-based-video-understanding)
    - [1. Load the Video Dataset](#1-load-the-video-dataset)
    - [2. Modify Model Configuration](#2-modify-model-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning-1)
  - [Evaluation](#evaluation)
    - [1. Dataset Preparation](#1-dataset-preparation)
    - [2. Parameter Configuration](#2-parameter-configuration-2)
    - [3. Start Evaluation](#3-start-evaluation)
  - [Feature Usage Introduction](#feature-usage-introduction)
    - [LoRA Fine-tuning](#lora-fine-tuning)
    - [Swap Attention](#swap-attention)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Notes](#notes)

<a id="jump1"></a>

## Environment Setup

The dependencies for the MindSpeed MM MindSpore backend are listed in the table below. For installation steps, refer to the [Installation Guide](../../../docs/en/mindspore/install_guide.md).

| Dependency        |  Version                                                                                                                             |
| -------------------------- |-------------------------------------------------------------------------------------------------------------------------------|
| Ascend NPU Driver & Firmware | In-development |
| Ascend CANN                | In-development                                           |
| MindSpore                  | 2.7.2         |
| Python                     | >=3.10       |

<a id="jump1.1"></a>

### 1. Repository Cloning and Environment Setup

For the MindSpeed MindSpore backend, the Ascend community provides a one-click conversion tool, MindSpeed-Core-MS, designed to help users automatically clone the relevant code repositories and perform one-click adaptation of Torch code. This allows users to start model training with a single click in the Huawei MindSpore + CANN environment without additional manual adaptation. Before performing the one-click conversion, users need to clone the relevant code repositories and set up the environment.

```shell
# Create a Conda environment.
conda create -n test python=3.10
conda activate test

# Use environment variables.
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# Install the MindSpeed-Core-MS conversion tool.
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.4.0

# Use the internal scripts of MindSpeed-Core-MS to automatically pull the relevant code repositories, adapt them in one click, and configure the environment.
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm

mkdir ckpt
mkdir data
mkdir logs
```

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the corresponding model weights from the Hugging Face library:

- Model address: [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main)
- Model address: [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)
- Model address: [Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/tree/main)
- Model address: [Qwen2.5-VL-72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/tree/main)

Save the downloaded model weights to the local `ckpt/hf_path/Qwen2.5-VL-*B-Instruct` directory (`*` indicates the corresponding size).

<a id="jump2.2"></a>

### 2. Weight Conversion (hf2mm)

MindSpeed MM has modified the structure names of some original networks. Use the `mm-convert` tool to convert the original pre-trained weights. This tool enables mutual conversion between Hugging Face weights and MindSpeed MM weights, as well as the re-partitioning of PP weights. For details, refer to the [Weight Conversion Tool Usage](../../../docs/en/features/mm_convert.md). **Note that currently, under the MindSpore backend, the converted weights cannot be used for training on the Torch backend**.

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
  --cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
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
# Where:
# mm_dir: Directory for saving converted weights
# hf_dir: Hugging Face weight directory
# llm_pp_layers: Number of LLM layers partitioned across each card. Ensure this is consistent with `pipeline_num_layers` configured in `model.json`.
# vit_pp_layers: Number of ViT layers partitioned across each card. Ensure this is consistent with `pipeline_num_layers` configured in `model.json`.
# tp_size: Number of TP partitions. Ensure this is consistent with the configuration in the fine-tuning launch script.
```

If you need to use the converted model for training, synchronously modify the `LOAD_PATH` parameter in `examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`. This path points to converted or partitioned weights. Be sure to distinguish it from the original weight path `ckpt/hf_path/Qwen2.5-VL-7B-Instruct`.

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
```

<a id="jump2.3"></a>

### 3. Weight Conversion (mm2hf)

MindSpeed MM has modified the names of some original network structures. After fine-tuning, if you need to convert the weights back to the Hugging Face format, you can use the `mm-convert` weight conversion tool to convert the fine-tuned weights, changing the weight names to match the original network.

```bash
mm-convert  Qwen2_5_VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-7B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 1
# Where:
# save_hf_dir: Directory where mm-to-hf weights are saved after MM fine-tuning is complete.
# mm_dir: Directory where the fine-tuned weights are saved.
# hf_dir: Hugging Face weight directory
# llm_pp_layers: Number of LLM layers partitioned across each card. Note that this must be consistent with `pipeline_num_layers` configured in `model.json`.
# vit_pp_layers: Number of ViT layers partitioned across each card. Note that it must be consistent with `pipeline_num_layers` configured in `model.json`.
# audio_pp_layers: Number of audio layers partitioned across each card. Note that it must be consistent with `pipeline_num_layers` configured in `model.json`.
# tp_size: Number of TP partitions. Note that it must be consistent with the configuration in the fine-tuning startup script.
```

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download (Using the COCO2017 Dataset as an Example)

(1) Users need to download the [COCO2017 dataset](https://cocodataset.org/#download) and extract it to the `./data/COCO2017` folder under the project directory (using the current directory `MindSpeed-MM/` as an example);

(2) Obtain the image dataset description file ([LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)) and download it to the `./data/` path.

(3) Run the data conversion script `python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py`. The file `mllm_format_llava_instruct_data.json` will be generated in the `./data` path (if this file already exists, please remove it or rename it as a backup first).

   ```shell
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

Currently, reading multiple datasets separated by `,` (do not add spaces) is supported. To do so, modify `dataset_param->basic_parameters->dataset` in `data_*b.json` (`*` indicates the corresponding model size):
Change `"./data/mllm_format_llava_instruct_data.json"` to `"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"`.

Also note the `dataset_param->basic_parameters->max_samples` parameter in `data_*b.json` limits the data reading to only `max_samples` entries for quick function verification. For formal training, you can remove this parameter to read all data.

<a id="jump3.2"></a>

### 2. Pure Text or Mixed Training Data with and without Images (Taking LLaVA-Instruct-150K as an Example)

This framework now supports pure text/mixed data (mixed training data with and without images).

When constructing data, for data containing images, retain the `image` key.

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

<a id="jump4"></a>

## Fine-tuning

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the script, you must complete the preliminary preparations, including: **Environment Setup**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data_*b.json` according to the actual situation, including fields such as `model_name_or_path`, `dataset_dir`, and `dataset`.

Taking Qwen2.5VL-7B as an example, make the following modifications to `data_7b.json`. Note that the weight path specified by `model_name_or_path` is the path before weight conversion.

**Note: Do not configure the same mount directory for `cache_dir` on multiple machines to avoid conflicts caused by writing to the same file**.

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
}
```

If you need to load a large amount of data, you can use streaming. Modify the `sampler_type` field in `data_7b.json` and add the `streaming` field. (Note: Once streaming is enabled, only `num_worker=0` is currently supported, meaning data is processed by a single process. This may cause performance fluctuations, and the resumable training feature will not be supported.)

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

If you need to calculate validation loss, modify the `eval-interval` and `eval-iters` parameters in the shell script; and add the following fields within `basic_parameters` in `data_7b.json`.
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

Configure the parameters of `examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh` according to the actual situation, including load and save paths, and the save interval `--save-interval` (Note: Distributed optimizer files are large and saving them is time-consuming, so set the save interval carefully).

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
    --no-load-rng \  # Do not load random number state. Remove this if loading is required.
    --no-save-optim \  # Do not save optimizer state. Remove this if saving is required.
    --no-save-rng \  # Do not save random number state. Remove this if saving is required.
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # Logging Interval
    --save-interval 5000 \  # Save Interval
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

[Single-node Execution Configuration]

Configure the parameters in `examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh` as follows:

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

Also note that if all parameters on a certain card are frozen, it will result in no gradients (for example, when `vision_encoder` is frozen with `[30,2,0,0]` and `[0,11,10,7]` configured for PP). In this case, you need to add `--enable-dummy-optimizer` to the `GPT_ARGS` parameter in `finetune_qwen2_5_vl_7b.sh`. For more details, refer to the [dummy_optimizer](../../../docs/en/features/dummy_optimizer.md).

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Take Qwen2.5VL-7B as an example to start the fine-tuning training task.

```shell
bash examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
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

Note: If using weights saved in MindSpeed-MM, no conversion is needed and they can be loaded directly (ensure the partitioning configuration matches that used during training).

<a id="jump5.2"></a>

### 2. Parameter Configuration

Modify the path configurations in `examples/mindspore/qwen2.5vl/inference_qwen2_5_vl_7b.json` and `examples/mindspore/qwen2.5vl/inference_qwen2_5_vl_7b.sh` according to your actual situation, including the tokenizer loading path `from_pretrained` and the image processor path `image_processer_path`.

Note:
(1) The path configured for `tokenizer/from_pretrained` is the original `Qwen2.5-VL-7B-Instruct` path downloaded from Hugging Face.

(2) The path for `LOAD_PATH` in the shell file is the model path after weight conversion (supporting PP partitioning).

<a id="jump5.3"></a>

### 3. Start Inference

```shell
bash examples/mindspore/qwen2.5vl/inference_qwen2_5_vl_7b.sh
```

Note: For single-card inference, FA must be enabled; otherwise, insufficient memory errors may occur. The switch `--use-flash-attn` is enabled by default, so ensure the FA enabling steps are completed. If using multi-card inference, you need to adjust the corresponding PP parameters and the `NPUS_PER_NODE` (number of NPUs used). Taking PP4 as an example, modify the shell parameters as follows:

```shell
NPUS_PER_NODE=4 # The number of available NPUs must be greater than PP*TP*CP
PP=4 # Pipeline parallelism size
```

<a id="jump6"></a>

## Qwen2.5VL-based Video Understanding

<a id="jump6.1"></a>

### 1. Load the Video Dataset

The video dataset is sourced from [llamafactory](https://github.com/hiyouga/LLaMA-Factory/tree/main/data), and the videos are sourced from `mllm_demo_data`. When using it, you need to place this data into your own data folder, and also place the `mllm_video_demo.json` from llamafactory into your own data folder.

Then, modify the dataset path in `data_*b.json` according to the actual situation, including the `model_name_or_path`, `dataset_dir`, and `dataset` fields, and modify the `images` and `videos` fields in `attr`.

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

<a id="jump6.2"></a>

### 2. Modify Model Configuration

In `model_xxx.json`, modify `img_context_token_id` as shown below:

```shell
"img_context_token_id": 151656
```

Note that the `image_token_id` and `img_context_token_id` parameters serve different purposes. The former is fixed and is the token ID that identifies images, used in `qwen2_5_vl_get_rope_index` to calculate the number of images in the sequence for image-text inputs. The latter is the token ID that identifies visual content, used to mark the position of visual tokens in the forward pass, so it needs to be modified accordingly based on the input.

<a id="jump6.3"></a>

### 3. Start Fine-tuning

Taking Qwen2.5VL-7B as an example, start the fine-tuning training task:

```shell
bash examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
```

<a id="jump7"></a>

## Evaluation

<a id="jump7.1"></a>

### 1. Dataset Preparation

The current model supports evaluation on four datasets: AI2D (test), ChartQA (test), DocVQA (val), and MMMU (val).

- [MMMU_DEV_VAL](https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv)
- [DocVQA_VAL](https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv)
- [AI2D_TEST](https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv)
- [ChartQA_TEST](https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv)

<a id="jump7.2"></a>

### 2. Parameter Configuration

To perform evaluation, you need to modify the name and path of the dataset to be evaluated in `examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.json`. The fields that need to be changed are:

- `from_pretrained` in `tokenizer` is the Qwen2.5-VL weights from Hugging Face. Refer to the link above to download and pass them.
- `dataset_path` is the local path of the evaluation dataset mentioned above.
- `evaluation_dataset` is the name of the evaluation dataset. Available names are (`ai2d_test`, `mmmu_dev_val`, `docvqa_val`, `chartqa_test`). Note that iIt must correspond to the dataset path mentioned above.
- `result_output_path` is the output path for the evaluation results. Note that before each evaluation, you must delete the evaluation files previously saved in this path.

```json
    "tokenizer": {
        "from_pretrained": "./Qwen2.5-VL-7B-Instruct",

    },
    "dataset_path": "./AI2D_TEST.tsv",
    "evaluation_dataset":"ai2d_test",
    "evaluation_model":"qwen2_vl_7b",
    "result_output_path":"./evaluation_outputs/"

```

After modifying `examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.json`, you need to configure the path of this file in the `MM_MODEL` field of `examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh`.

You also need to configure the model weight path after weight conversion in the `LOAD_PATH` field of `examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh`.

```shell
MM_MODEL=examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.json
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
```

The current evaluation also supports multi-card mode, which requires changing the NPU count.

```shell
NPUS_PER_NODE=1
```

<a id="jump7.3"></a>

### 3. Start Evaluation

Evaluation requires additional Python packages. Install them using the following command:

```shell
pip install -e ".[evaluate]"
```

Start the shell to begin evaluation:

```shell
bash examples/mindspore/qwen2.5vl/evaluate_qwen2_5_vl_7b.sh
```

Evaluation results will be output to the `result_output_path` path. The output result files include:

- `*.xlsx` file, which contains detailed information such as the prediction results and answers for each question.
- `*.csv` file, which contains statistical data such as accuracy.

<a id="jump8"></a>

## Feature Usage Introduction

<a id="jump8.1"></a>

### LoRA Fine-tuning

LoRA is a general framework capability and is currently supported. Please refer to the [LoRA Feature Guide](../../../docs/en/features/lora_finetune.md).

### Swap Attention

For scenarios where recomputation is not required, enabling only `--swap-attention` can save memory with almost no performance loss, supporting configurations for larger models. Please refer to the memory saving section in the [Swap Attention Feature Guide](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/swap_attention.md).

<a id="jump9"></a>

## Environment Variable Declaration

`ASCEND_RT_VISIBLE_DEVICES`: Specifies the index value of the NPU device.

`ASCEND_SLOG_PRINT_TO_STDOUT`: Controls whether log printing is enabled. `0` disables log printing, and `1` enables log printing.

`ASCEND_GLOBAL_LOG_LEVEL`: Sets the log level for application logs and module logs supporting only debug logs.
`0` corresponds to the DEBUG level, `1` to the INFO level, `2` to the WARNING level, `3` to the ERROR level, and `4` to the NULL level, with no log output.

`HCCL_CONNECT_TIMEOUT`: Limits the timeout waiting time for socket connection establishment between different devices. It must be configured as an integer in the value range [120, 7200]. The default value is 120, in seconds.

`HCCL_EXEC_TIMEOUT`: Controls the synchronous waiting time during execution between devices. Within this configured time, each device process waits for other devices to execute communication synchronization.

`ASCEND_LAUNCH_BLOCKING`: Controls whether to enable synchronous mode during operator execution. `0` executes operators in asynchronous mode, and `1` forces operators to run in synchronous mode.

`MS_DEV_HOST_BLOCKING_RUN`: Controls whether dynamic graph operators are dispatched in a single thread. `0` indicates multi-threaded dispatch, and `1` indicates single-threaded dispatch.

`MS_DEV_LAUNCH_BLOCKING`: Controls whether operators are dispatched synchronously. `0` indicates asynchronous dispatch, and `1` indicates single-threaded dispatch with stream synchronization.

`ACLNN_CACHE_LIMIT`: Configures the number of operator information entries cached on the host side for single-operator execution APIs.

`TOKENIZERS_PARALLELISM`: Controls the behavior of the tokenizer in Hugging Face's transformers library in a multi-threaded environment.

`NPUS_PER_NODE`: Configures the number of NPUs used on a single compute node.

<a id="jump10"></a>

## Notes

1. In `finetune_xx.sh`, parameters related to the model structure do not take effect. The configuration of parameters with the same name in `examples/mindspore/qwen2.5vl/model_xb.json` takes precedence. Training-related parameters that are not part of the model structure should be modified in `finetune_xx.sh`.
2. If an Out Of Memory error occurs during sing-card 3B-sized model training, you can use multiple cards in parallel and enable the distributed optimizer for training.
3. When `use_remove_padding` is set to `true` in `model.json`, the attention_mask shape is currently fixed to `[2048, 2048]` in `examples/mindspore/qwen2vl/dot_product_attention.py`. If you need to change it, please refer to the [FlashAttentionScore](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0027.html) in the Ascend community.
