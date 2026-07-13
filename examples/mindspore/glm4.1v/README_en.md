# GLM4.1V User Guide

<p align="left">
</p>

## Contents

- [GLM4.1V User Guide](#glm41v-user-guide)
  - [Contents](#contents)
  - [Environment Setup](#environment-setup)
    - [1. Repository Cloning](#1-repository-cloning)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion (hf2mm)](#2-weight-conversion-hf2mm)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download (using the COCO2017 dataset as an example)](#1-dataset-download-using-the-coco2017-dataset-as-an-example)
  - [Fine-tuning](#fine-tuning)
    - [1. Preparation](#1-preparation)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Precautions](#precautions)

<a id="jump1"></a>

## Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md).

<a id="jump1.1"></a>

### 1. Repository Cloning

```shell
# Install MindSpeed-Core-MS for one-click deployment.
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.5.0

# Install the specified version of transformers.
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 4a79bf947d0614d2a023b9137a32cf754ac241fe
pip install -e .

# Use the internal scripts of MindSpeed-Core-MS to automatically pull the relevant code repositories and adapt them with one click.
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm

mkdir logs
mkdir data
mkdir ckpt
```

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the corresponding model weights from the Hugging Face repository:

- Model address: [GLM4.1V-9B](https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking)

 Save the downloaded model weights to the local `ckpt/hf_path/GLM4.1V-9B-Instruct` directory.

<a id="jump2.2"></a>

### 2. Weight Conversion (hf2mm)

MindSpeed MM has modified some of the original network structure names. Use the `mm-convert` tool to convert the original pre-trained weights. This tool enables bidirectional conversion between HuggingFace weights and MindSpeed MM weights, as well as the re-slicing of PP weights. Refer to the [Weight Conversion Tool Usage](../../../docs/en/features/mm_convert.md).

```bash
# 9B PP1
mm-convert GlmConverter hf_to_mm \
  --cfg.mm_dir "pretrained/GLM4.1V-9B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/GLM4.1V-9B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[40]] \
  --cfg.parallel_config.vit_pp_layers [[24]] \
  --cfg.trust_remote_code True

# 9B PP4
mm-convert GlmConverter hf_to_mm \
  --cfg.mm_dir "pretrained/GLM4.1V-9B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/GLM4.1V-9B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[5,12,12,11]] \
  --cfg.parallel_config.vit_pp_layers [[24,0,0,0]] \
  --cfg.trust_remote_code True


# Where:
# mm_dir: directory to save the converted weights
# hf_dir: Hugging Face weights directory
# llm_pp_layers: number of LLM layers split across each device. Note that this must be consistent with the `pipeline_num_layers` configured in `model.json`
# vit_pp_layers: number of ViT layers split across each device. Note that this must be consistent with the `pipeline_num_layers` configured in `model.json`
# tp_size: number of TP partitions. Ensure it matches the configuration in the fine-tuning startup script
```

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download (using the COCO2017 dataset as an example)

(1) Users need to download the [COCO2017 dataset](https://cocodataset.org/#download) and extract it to the `./data/COCO2017` folder under the project directory.

(2) Obtain the image dataset description file ([LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)) and download it to the `./data/` path.

(3) Run the data conversion script `python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py`.

   ```shell
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

Currently, reading multiple datasets separated by `,` (note: do not add spaces) is supported. The configuration method is to modify `dataset_param->basic_parameters->dataset` in `data.json` from `"./data/mllm_format_llava_instruct_data.json"` to `"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"`.

Also, note the configuration of `dataset_param->basic_parameters->max_samples` in `data.json`. It limits the data reading to only `max_samples` entries, which allows for quick function verification. For formal training, you can remove this parameter to read all the data.

## Fine-tuning

Since the official repository has not yet open-sourced the fine-tuning code and scripts, the current implementation is based on a reference implementation for qwen2.5vl. The official fine-tuning feature will be rolled out subsequently.

<a id="jump4.1"></a>

### 1. Preparation

Before configuring the scripts, you need to complete the preliminary preparations, including: **environment setup**, **weight download and conversion**, and **dataset preparation and processing**. For details, please refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data.json` according to the actual situation, including fields such as `model_name_or_path`, `dataset_dir`, and `dataset`.

Taking GLM4.1V-9B as an example, make the following modifications to `data.json`. Note that the weight path for `model_name_or_path` is the weight path before conversion.

Note: Do not configure `cache_dir` to the same mounted directory across multiple machines to avoid conflicts caused by writing to the same file.

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/GLM4.1V-9B-Instruct",
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

Configure the parameters in `examples/glm4.1v/finetune_glm4.1v_9b.sh` according to your actual situation, including the load and save paths, as well as the save interval `--save-interval` (Note: Distributed optimizer files are large and saving them takes a long time, so please set the save interval cautiously).

```shell
...
# Load Path
LOAD_PATH="ckpt/mm_path/GLM4.1V-9B-Instruct"
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
    --log-interval 1 \  # Log interval
    --save-interval 5000 \  # Save Interval
    ...
    --log-tps \  # Adding this parameter enables printing the average sequence length of the language module at each step during training, and calculates the throughput in tokens per second after training is completed.
"
```

If you need to load the weights, optimizer, and other states from a specific iteration, set `LOAD_PATH` to `"save_dir"`, and modify the content of the `latest_checkpointed_iteration.txt` file to the specified iteration number (this feature is coming soon).

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Single-Node Running Configuration]

Configure the parameters in `examples/experimental/glm4.1v/finetune_glm4.1v_9b.sh` as follows:

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

Note that when PP is enabled, the `pipeline_num_layer` parameters configured for `vision_encoder` and `text_decoder` in `model.json` control their respective PP partitioning strategies. For pipeline parallelism, the `vision_encoder` must be processed first, followed by the `text_decoder`.

For example, the default values for the 9B model are `[24,0,0,0]` and `[5,12,12,11]`, which means that within the PP domain, the first card places 24 layers of `vision_encoder` and then 5 layers of `text_decoder`, the second card places the next 12 layers of `text_decoder`, the third card places the next 12 layers of `text_decoder`, and the fourth card places the next 11 layers of `text_decoder`. The layers of `text_decoder` cannot be placed before those of `vision_encoder` are fully placed (for example, the configuration `[22,2,0,0]` and `[5,12,12,11]` is incorrect).

Also note that if all parameters on a certain card are frozen, it will result in no gradients (for example, when `vision_encoder` is frozen with `[24,0,0,0]` and `[5,12,12,11]` configured for PP). In this case, you need to add `--enable-dummy-optimizer` to the `GPT_ARGS` parameter in `finetune_glm4.1v_9b.sh`. Refer to the [dummy_optimizer feature guide](../../../docs/en/features/dummy_optimizer.md).

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Take GLM4.1V-9B as an example to start the fine-tuning training task.
Differences in loss calculation methods can have varying impacts on training results. Before starting the training task, please refer to the documentation on loss calculation and select an appropriate loss calculation method. For details, see [vlm_model_loss_calculate_type.md](../../../docs/en/features/vlm_model_loss_calculate_type.md).

```shell
bash examples/experimental/glm4.1v/finetune_glm4.1v_9b.sh
```

<a id="jump6"></a>

## Environment Variable Declaration

| Environment Variable          | Description                                                                 | Value Description                                                                                         |
|-------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
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

<a id="jump7"></a>

## Precautions

1. In `finetune_xx.sh`, parameters related to the model structure do not take effect. The configuration of parameters with the same name in `examples/glm4.1v/model_xb.json` shall prevail. Training-related parameters that are not part of the model structure should be modified in `finetune_xx.sh`.
