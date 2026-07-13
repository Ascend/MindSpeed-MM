# Qwen2_5_Omni (MindSpore Backend) User Guide

<p align="left">
</p>

## Contents

- [Qwen2\_5\_Omni (MindSpore Backend) User Guide](#qwen2_5_omni-mindspore-backend-user-guide)
  - [Contents](#contents)
  - [Environment Setup](#environment-setup)
    - [1. Repository Cloning and Environment Setup](#1-repository-cloning-and-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion (hf2mm)](#2-weight-conversion-hf2mm)
    - [3. Weight Conversion (mm2hf)](#3-weight-conversion-mm2hf)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download (Using the COCO2017 Dataset as an Example)](#1-dataset-download-using-the-coco2017-dataset-as-an-example)
    - [2. Pure Text or Mixed Data with and without Images (Taking LLaVA-Instruct-150K as an Example)](#2-pure-text-or-mixed-data-with-and-without-images-taking-llava-instruct-150k-as-an-example)
    - [3. Video and Audio Datasets](#3-video-and-audio-datasets)
      - [3.1 Loading the Video Dataset](#31-loading-the-video-dataset)
      - [3.2 Modify Model Configuration](#32-modify-model-configuration)
  - [Fine-tuning](#fine-tuning)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
  - [Environment Variables Declaration](#environment-variables-declaration)
  - [Notes](#notes)

<a id="jump1"></a>

## Environment Setup

The dependencies for the MindSpeed MM MindSpore backend are listed in the table below. For installation steps, refer to the [Installation Guide](../../../docs/en/mindspore/install_guide.md).

| Dependency         |        Version                                                      |
| ---------------- | ------------------------------------------------------------ |
| Ascend NPU Driver & Firmware  | [In-development Version](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.3.RC1&driver=Ascend+HDK+25.3.RC1) |
| Ascend CANN        | [In-development Version](https://www.hiascend.com/zh/developer/download/community/result?module=cann) |
| MindSpore        | [2.7.2](https://www.mindspore.cn/install/en)         |
| Python           | >=3.9                                                        |
|transformers     |      [v4.53.0](https://github.com/huggingface/transformers/tree/v4.53.0)    |
|mindspore_op_plugin | [In-development Version](https://gitee.com/mindspore/mindspore_op_plugin) |

<a id="jump1.1"></a>

### 1. Repository Cloning and Environment Setup

For the MindSpeed MindSpore backend, the Ascend community provides a one-click model deployment tool, MindSpeed-Core-MS, designed to help users automatically pull relevant code repositories and perform one-click adaptation of torch code. This allows users to initiate model training with a single click in the Huawei MindSpore + CANN environment without additional manual adaptation. Before using the one-click deployment, users need to pull the relevant code repositories and set up the environment.

```shell
# Create a conda environment.
conda create -n test python=3.10
conda activate test

# Use environment variables.
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# Install MindSpeed-Core-MS for one-click deployment.
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master

# Use the internal script of MindSpeed-Core-MS to automatically pull the relevant code repositories and adapt them with one click.
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm
pip install transformers==4.53.0
# Install librosa for audio parsing.
pip install librosa

# Pull and install mindspore_op_plugin.
git clone https://gitee.com/mindspore/mindspore_op_plugin.git
cd mindspore_op_plugin
bash build.sh
source env.source
cd ..

mkdir ckpt
mkdir data
mkdir logs
```

> Note
>
> [mindspore_op_plugin](https://gitee.com/mindspore/mindspore_op_plugin) is an operator plugin library of MindSpore. It quickly supplements CPU/GPU operator functionality by directly calling the ATen operator in libtorch. It is currently an **experimental feature** and is **restricted for use** only with this model.

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the corresponding model weights from the Hugging Face library:

- Model address: [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/tree/main)

 Save the downloaded model weights to the local `ckpt/hf_path/Qwen2.5-Omni-7B` directory.

<a id="jump2.2"></a>

### 2. Weight Conversion (hf2mm)

MindSpeed MM has modified the structure names of some original networks. Use the `mm-convert` tool to convert the original pre-trained weights. This tool enables mutual conversion between Hugging Face weights and MindSpeed MM weights, as well as the re-sharding of PP weights. Refer to [Weight Conversion Tool Usage](../../../docs/en/features/mm_convert.md) for specific usage of this tool. **Note that currently under the MindSpore backend, the converted weights cannot be used for training on the Torch backend**.

> Note
>
> Weight conversion depends on mindspore_op_plugin. Please ensure this software is installed. For usage instructions, refer to the [op_plugin CPU Operator Development Guide](https://gitee.com/mindspore/mindspore_op_plugin/wikis/op_plugin%20CPU%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97).

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

If you need to train with the converted model, synchronously modify the `LOAD_PATH` parameter in `examples/mindspore/qwen2.5omni/finetune_qwen2_5_omni_7b.sh`. This path is for the converted or partitioned weights. Note that it should be distinguished from the original weight path `ckpt/hf_path/Qwen2.5-Omni-7B`.

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-Omni-7B"
```

<a id="jump2.3"></a>

### 3. Weight Conversion (mm2hf)

MindSpeed MM has modified some of the original network's structure names. After fine-tuning, if you need to convert the weights back to the Hugging Face format, you can use the `mm-convert` weight conversion tool to convert the fine-tuned weights, changing the weight names to match the original network.

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

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download (Using the COCO2017 Dataset as an Example)

(1) Users need to download the [COCO2017 dataset](https://cocodataset.org/#download) and extract it into the `./data/COCO2017` folder within the project directory.

(2) Obtain the description file for the image dataset ([LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)) and download it to the `./data/` path;

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

Currently, reading multiple datasets separated by `,` (do not add spaces) is supported. To do so, modify
`dataset_param->basic_parameters->dataset` in `data.json`:
Change `./data/mllm_format_llava_instruct_data.json` to `./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json`.

Also, note the configuration of `dataset_param->basic_parameters->max_samples` in `data.json`, which will limit the data read to only `max_samples` entries. This allows for quick function verification. For formal training, you can remove this parameter to read all the data.

<a id="jump3.2"></a>

### 2. Pure Text or Mixed Data with and without Images (Taking LLaVA-Instruct-150K as an Example)

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

<a id="jump3.3"></a>

### 3. Video and Audio Datasets

#### 3.1 Loading the Video Dataset

The video dataset is sourced from [llamafactory](<https://github.com/hiyouga/LLaMA-Factory/tree/main/data>).

The videos are sourced from `mllm_video_demo`. When using them, you need to place it into your own data folder, and also place `mllm_video_audio_demo.json` from llamafactory into your own data file.

Afterwards, modify the dataset path in `data.json` according to the actual situation, including the `model_name_or_path`, `dataset_dir`, and `dataset` fields, and modify the `images` and `videos` fields in `attr`.

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
            "dataset": "./data/mllm_video_audio_demo.json",
            "cache_dir": "./data/cache_dir",
            ...
        },
        ...
        "attr": {
            "system": null,
            "images": null,
            "videos": "videos",
            "audios": "audios",
            ...
        },
    },
    ...
}
```

#### 3.2 Modify Model Configuration

In `model.json`, modify `img_context_token_id` as follows:

```shell
"img_context_token_id": 151656
```

Note that the `image_token_id` and `img_context_token_id` parameters serve different purposes. The former is fixed and is the token ID that identifies an image, used in `qwen2_5_omni_get_rope_index` to calculate the number of images in a sequence with image-text input. The latter is the token ID that identifies visual content, used to mark the position of visual tokens in the forward pass, so it needs to be modified accordingly based on the input.

<a id="jump4"></a>

## Fine-tuning

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the preliminary preparations, including: **Environment Setup**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. For details, please refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data.json` according to the actual situation, including fields such as `model_name_or_path`, `dataset_dir`, and `dataset`.

Taking Qwen2.5Omni-7B as an example, make the following modifications to `data.json` as follows. Note that the weight path specified by `model_name_or_path` is the path before weight conversion.

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

Configure the parameters in `examples/mindspore/qwen2.5omni/finetune_qwen2_5_omni_7b.sh` according to the actual situation, including the load and save paths, as well as the save interval `--save-interval` (Note: Distributed optimizer files are large and saving them takes a long time, so please set the save interval carefully).

```shell
...
# Load Path
LOAD_PATH="ckpt/mm_path/Qwen2.5-Omni-7B"
# Save Path
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # Do not load optimizer state; remove this if loading is required.
    --no-load-rng \  # Do not load random number state; remove this if loading is required.
    --no-save-optim \  # Do not save optimizer state; remove this if saving is required.
    --no-save-rng \  # Do not save random number state; remove this if saving is required.
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # Logging Interval
    --save-interval 5000 \  # Save Interval
    ...
    --log-tps \  # Adding this parameter enables printing the average sequence length of the language module per step during training, and calculating the throughput in tokens per second after training ends.
"
```

If you need to load the weights, optimizer states, etc. for a specific iteration, set `LOAD_PATH` to `"save_dir"`, and modify the content of the `latest_checkpointed_iteration.txt` file to the specified iteration count.

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Single-Node Execution Configuration]

Configure the parameters in `examples/mindspore/qwen2.5omni/finetune_qwen2_5_omni_7b.sh` as follows:

```shell
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
export LOCAL_WORLD_SIZE=8
export MS_NODE_TIMEOUT=1800
```

Note that when PP is enabled, `pipeline_num_layer` configured for `vision_encoder` and `text_decoder` in `model.json` control their respective PP partitioning strategies. For pipeline parallelism, `vision_encoder` must be processed before `text_decoder`.
For example, the default values for the 7B model are `[32,0,0,0]` and `[1,10,10,7]`. This means that within the PP domain, the 32 layers of `vision_encoder`, followed by 1 layer of `text_decoder`, are placed on the first card; the next 10 layers of `text_decoder` are placed on the second card; the next 10 layers of `text_decoder` are placed on the third card; and the next 7 layers of `text_decoder` are placed on the fourth card. The layers of `text_decoder` cannot be placed before layers of `vision_encoder` are fully placed (for example, the configuration `[30,2,0,0]` and `[1,10,10,7]` is incorrect).

Also note that if all parameters on a certain card are frozen, it will result in no gradients (for example, when `vision_encoder` is frozen with `[30,2,0,0]` and `[0,11,10,7]` configured for PP). In this case, you need to add `--enable-dummy-optimizer` to the `GPT_ARGS` parameter in `finetune_qwen2_5_omni_7b.sh`. For more details, refer to the [dummy_optimizer](../../../docs/en/features/dummy_optimizer.md).

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Take Qwen2.5Omni-7B as an example to start the fine-tuning training task.
Differences in loss calculation methods can have varying impacts on training effectiveness. Before starting the training task, please review the documentation on loss calculation and select an appropriate loss calculation method. For details, see [vlm_model_loss_calculate_type.md](../../../docs/en/features/vlm_model_loss_calculate_type.md).

```shell
bash examples/mindspore/qwen2.5omni/finetune_qwen2_5_omni_7b.sh
```

<a id="jump5"></a>

## Environment Variables Declaration

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

<a id="jump6"></a>

## Notes

1. In `finetune_xx.sh`, parameters related to the model structure do not take effect. The configuration of parameters with the same name in `examples/mindspore/qwen2.5omni/model_xb.json` shall prevail. Non-model-structure training-related parameters should be modified in `finetune_xx.sh`.
