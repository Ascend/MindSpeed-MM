# InternVL2.5 Usage Guide (MindSpore as Backend)

<p align="left">
</p>

## Contents

- [InternVL2.5 Usage Guide (MindSpore as Backend)](#internvl25-usage-guide-mindspore-as-backend)
  - [Contents](#contents)
  - [Environment Setup](#environment-setup)
    - [1. Repository Pull and Environment Setup](#1-repository-pull-and-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion](#2-weight-conversion)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download](#1-dataset-download)
  - [Fine-tuning](#fine-tuning)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-Tuning](#3-start-fine-tuning)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Precautions](#precautions)

<a id="jump1"></a>

## Environment Setup

The dependencies for the MindSpeed MM MindSpore backend are listed in the following table. For installation steps, refer to the [Installation Guide](../../../docs/en/mindspore/install_guide.md).

| Dependency         |                                                                                                                                 |
| ---------------- |---------------------------------------------------------------------------------------------------------------------------------|
| Ascend NPU Driver & Firmware  | [In-development version](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.3.RC1&driver=Ascend+HDK+25.3.RC1) |
| Ascend CANN        | [8.3.rc1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1)                                |
| MindSpore        | [2.7.2](https://www.mindspore.cn/install/en)                                                                                      |
| Python           | >=3.9                                                                                                                           |
|transformers     | [v4.53.0](https://github.com/huggingface/transformers/tree/v4.53.0)                                                             |

<a id="jump1.1"></a>

### 1. Repository Pull and Environment Setup

The model is deployed with one click using MindSpeed-Core-MS, which automatically pulls the relevant code repositories and performs one-click adaptation of the torch code. Before the one-click deployment, users need to pull the relevant code repositories and perform environment setup.

```shell
# Create a conda environment.
conda create -n test python=3.10
conda activate test

# Use environment variables.
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Modify the ascend-toolkit path based on the actual situation.
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# Install MindSpeed-Core-MS for one-click deployment.
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master

# Use the internal script of MindSpeed-Core-MS to automatically pull the relevant code repositories and adapt them with one click.
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm

mkdir ckpt
mkdir data
mkdir logs
```

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download open-source model weights from websites such as Hugging Face.

- [InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B)

- [InternVL2_5-78B](https://huggingface.co/OpenGVLab/InternVL2_5-78B)

Save the model weights in the `raw_ckpt` directory, for example, `raw_ckpt/InternVL2_5-78B`.

<a id="jump2.2"></a>

### 2. Weight Conversion

MindSpeed MM has modified some of the original network structure names. Use the `mm-convert` tool to convert the original pre-trained weights. This tool implements the conversion between HuggingFace weights and MindSpeed MM weights, as well as weight sharding for PP (Pipeline Parallelism).

For detailed usage of the `mm-convert` tool, refer to [Weight Conversion Tool Usage](../../../docs/en/features/mm_convert.md).

```bash
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh

# 4B
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2_5-4B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2_5-4B" \
  --cfg.parallel_config.llm_pp_layers [[36]] \
  --cfg.parallel_config.vit_pp_layers [[24]] \
  --cfg.trust_remote_code True

# 78B
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2_5-78B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2_5-78B" \
  --cfg.parallel_config.llm_pp_layers [[0,3,6,6,6,6,6,6,6,6,6,6,5,5,5,2]] \
  --cfg.parallel_config.vit_pp_layers [[45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] \
  --cfg.trust_remote_code True

# Where:
# mm_dir: Directory to save the converted weights
# hf_dir: HuggingFace weight directory
# llm_pp_layers: Number of LLM layers partitioned on each card. Ensure this is consistent with pipeline_num_layers configured in model.json.
# vit_pp_layers: Number of ViT layers partitioned on each card. Ensure this is consistent with pipeline_num_layers configured in model.json.
# trust_remote_code: To ensure code security, the default value of trust_remote_code is False. Users need to set it to True and ensure the security of the models and data they download.
```

Synchronously modify the `LOAD_PATH` parameter in `examples/mindspore/internvl2.5/finetune_internvl2.5_*b.sh`. This path points to the converted or sharded weights. Note that it should be distinguished from the original weights `raw_ckpt/InternVL2_5-*B`.

Take `InternVL2_5-78B` as an example:

```shell
LOAD_PATH="pretrained/InternVL2_5-78B"
```

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download

[Image Data]

You need to obtain and decompress the [InternVL-Finetune](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data) dataset into the `dataset/playground` directory. Taking the ai2d dataset as an example, the decompressed data structure is as follows:

   ```shell
   $playground
   ├── data
       ├── ai2d
           ├── abc_images
           ├── images
   ├── opensource
       ├── ai2d_train_12k.jsonl
   ```

[Video Data]

To train with videos, you can construct a video dataset by referring to [Video Dataset Construction](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data).

It also depends on the Decord library for video reading. The installation method for Decord is as follows:

X86:

```bash
pip install decord==0.6.0
```

Arm:

For installation via `apt`, please [refer to the link](https://github.com/dmlc/decord).

For `yum` installation, please [refer to the script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

<a id="jump4"></a>

## Fine-tuning

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the following prerequisites: **environment setup**, **weight download and conversion**, and **dataset preparation and processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset path in `data.json` according to the actual situation, including the `from_pretrained`, `data_path`, `data_folder`, and other fields.

Taking InternVL2_5-78B as an example, make the following modifications to `data_78B.json`. Note that the weight path in `tokenizer_config` is the weight path before conversion.

```json
{
  "dataset_param": {
      ...
      "basic_parameters": {
          "data_path": "dataset/playground/opensource/ai2d_train_12k.jsonl",
          "data_folder": "dataset/playground/data/ai2d"
      },
      ...
      "tokenizer_config": {
          ...
          "from_pretrained": "raw_ckpt/InternVL2_5-78B",
          ...
      },
      ...
  },
  ...
}
```

[Model Saving, Loading, and Logging Configuration]

Configure the parameters of `examples/internvl2.5/finetune_internvl2.5_xx.sh` according to the actual situation, including the loading and saving paths and the saving interval `--save-interval` (Note: Distributed optimizer files are large and saving them takes a long time, so please set the saving interval carefully).

Taking InternVL2.5-78B as an example:

```shell
...
# Loading path
LOAD_PATH="ckpt/InternVL2_5-78B"
# Saving path
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # Do not load optimizer state. Remove this if loading is required
    --no-load-rng \  # Do not load random number state. Remove this if loading is required
    --no-save-optim \  # Do not save optimizer state. Remove this if saving is required
    --no-save-rng \  # Do not save the random number state. Remove this if saving is required.
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # Logging interval
    --save-interval 5000 \  # Save interval
    ...
    --log-tps \  # Add this parameter to enable printing the average sequence length of the language module at each step during training, and to calculate the throughput in tokens per second after training completes.
"
```

If you need to load the weights, optimizer states, etc., from a specific iteration, set `LOAD_PATH` to `"save_dir"`, and modify the `latest_checkpointed_iteration.txt` file to specify the iteration count.

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Single-Node Execution Configuration]

Configure the parameters in `examples/internvl2.5/finetune_internvl2.5_xx.sh` as follows.

```shell
  # Modify the ascend-toolkit path according to the actual situation.
  source /usr/local/Ascend/cann/set_env.sh
  NPUS_PER_NODE=8
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

### 3. Start Fine-Tuning

Take InternVL2_5-78B as an example to start the fine-tuning training task.

```shell
bash examples/mindspore/internvl2.5/finetune_internvl2.5_78B.sh
```

<a id="jump5"></a>

## Environment Variable Declaration

| Environment Variable          | Description                                                                 | Value Description                                                                                         |
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

<a id="jump6"></a>

## Precautions

1. When using the PP strategy for multi-node training, the process may hang. Refer to [this PR](https://gitcode.com/Ascend/MindSpeed/pulls/1627/files) for the fix.
