# InternVL3 User Guide (MindSpore as Backend)

<p align="left">
</p>

## Contents

- [InternVL3 User Guide (MindSpore as Backend)](#internvl3-user-guide-mindspore-as-backend)
  - [Contents](#contents)
  - [Environment Setup](#environment-setup)
    - [1. Repository Cloning and Environment Setup](#1-repository-cloning-and-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion](#2-weight-conversion)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download](#1-dataset-download)
  - [Fine-tuning](#fine-tuning)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
  - [Environment Variables Declaration](#environment-variables-declaration)
  - [Notes](#notes)

<a id="jump1"></a>

## Environment Setup

The dependency compatibility for the MindSpeed MM MindSpore backend is shown in the following table. For installation steps, refer to the [Installation Guide](../../../docs/en/mindspore/install_guide.md).

| Dependency         |                                                                                                                                 |
| ---------------- |---------------------------------------------------------------------------------------------------------------------------------|
| Ascend NPU Driver/Firmware  | In-development |
| Ascend CANN        | 8.3.rc1                                |
| MindSpore        | [2.7.2](https://www.mindspore.cn/install/en)                                                                                      |
| Python           | >=3.9   |
|transformers     | [v4.53.0](https://github.com/huggingface/transformers/tree/v4.53.0)                                                             |

<a id="jump1.1"></a>

### 1. Repository Cloning and Environment Setup

The model can be deployed with one-click using MindSpeed-Core-MS, which automatically pulls the relevant code repositories and performs one-click adaptation of the torch code. Before the one-click deployment, users need to pull the relevant code repositories and set up the environment.

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

# Use the internal scripts of MindSpeed-Core-MS to automatically pull the relevant code repositories and adapt them with one click.
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

- [InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B)
- [InternVL3-78B](https://huggingface.co/OpenGVLab/InternVL3-78B)

Save the model weights in the `raw_ckpt` directory, for example, `raw_ckpt/InternVL3-8B`.

<a id="jump2.2"></a>

### 2. Weight Conversion

MindSpeed MM has modified some of the original network structure names. Use the `mm-convert` tool to convert the original pre-trained weights. This tool implements the conversion between HuggingFace weights and MindSpeed MM weights, as well as weight sharding for PP (Pipeline Parallelism).

For detailed usage of the `mm-convert` tool, refer to [Weight Conversion Tool Usage](../../../docs/en/features/mm_convert.md).

```bash
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh

# 8B
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL3-8B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL3-8B" \
  --cfg.parallel_config.llm_pp_layers [[6,8,8,6]] \
  --cfg.parallel_config.vit_pp_layers [[24,0,0,0]] \
  --cfg.trust_remote_code True

# 78B
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL3-78B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL3-78B" \
  --cfg.parallel_config.llm_pp_layers [[40,40]] \
  --cfg.parallel_config.vit_pp_layers [[45,0]] \
  --cfg.parallel_config.tp_size 8 \
  --cfg.trust_remote_code True

# Where:
# mm_dir: Directory to save the converted weights
# hf_dir: HuggingFace weight directory
# llm_pp_layers: Number of LLM layers split on each card. Note that it must be consistent with pipeline_num_layers configured in model.json
# vit_pp_layers: Number of ViT layers split across each card. Note that this must be consistent with pipeline_num_layers configured in model.json.
# trust_remote_code: To ensure code security, the trust_remote_code configuration defaults to False. Users need to set it to True and ensure the security of the models and data they download.
```

Synchronously modify the `LOAD_PATH` parameter in `examples/mindspore/internvl3/finetune_internvl3_*b.sh`. This path points to the converted or split weights. Note that it should be distinguished from the original weights `raw_ckpt/InternVL3-*B`.

Take `InternVL3-8B` as an example:

```shell
LOAD_PATH="pretrained/InternVL3-8B"
```

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download

[Image Data]

Users need to obtain and decompress the [InternVL-Finetune](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data) dataset into the `dataset/playground` directory. The decompressed data structure is as follows:

   ```shell
   $playground
   ├── data
       ├── ai2d
           ├── abc_images
           ├── images
       ├── coco
           ├── train2017
       ├── docvqa
           ├── train
           ├── test
           ├── val
       ├──...
   ├── opensource
       ├── ai2d_train_12k.jsonl
       ├── sharegpt4v_instruct_gpt4-vision_cap100k.jsonl
       ├── chartqa_train_18k.jsonl
       ├── ...
   ```

[Video Data]

To train with videos, you can construct your own video dataset by referring to [Video Dataset Construction](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data).

The Decord library is also required for reading videos. The installation method for Decord is as follows:

X86:

```bash
pip install decord==0.6.0
```

Arm:

For installation via `apt`, please [refer to the link](https://github.com/dmlc/decord).

For installation via `yum`, please [refer to the script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

<a id="jump4"></a>

## Fine-tuning

### 1. Prerequisites

Before configuring the script, you need to complete the following prerequisites: **Environment Setup**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data.json` according to your actual situation, including fields such as `from_pretrained`, `data_path`, and `data_folder`.

Taking InternVL3-8B as an example, make the following modifications to `data_8B.json`. Note that the weight path in `tokenizer_config` is the path before weight conversion.

```json
{
  "dataset_param": {
      ...
      "basic_parameters": {
          "data_path": "dataset/playground/opensource/sharegpt4v_instruct_gpt4-vision_cap100k.jsonl",
          "data_folder": "dataset/playground/data"
      },
      ...
      "tokenizer_config": {
          ...
          "from_pretrained": "raw_ckpt/InternVL3-8B",
          ...
      },
      ...
  },
  ...
}
```

[Model Saving, Loading, and Logging Configuration]

Configure the parameters in `examples/mindspore/internvl3/finetune_internvl3_xx.sh` according to your actual situation, including the load path, save path, and save interval `--save-interval` (Note: Distributed optimizer files are large and saving them is time-consuming, so please set the save interval carefully).

Taking InternVL3-8B as an example:

```shell
...
# Load Path
LOAD_PATH="ckpt/InternVL3-8B"
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

If you need to load weights, optimizer states, etc., from a specific iteration, set `LOAD_PATH` to `"save_dir"`, and modify the `latest_checkpointed_iteration.txt` file content to the specified iteration count.

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Single-Node Execution Configuration]

Configure the parameters in `examples/mindspore/internvl3/finetune_internvl3_xx.sh` as follows:

```shell
  # Modify the ascend-toolkit path according to the actual situation
  source /usr/local/Ascend/cann/set_env.sh
  NPUS_PER_NODE=8
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

[Model Parallelism Configuration]

InternVL involves unaligned TP sharding. If TP sharding is enabled, the following parameter needs to be added. For feature description, see the [feature guide](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/unaligned_linear.md).

```shell
--unaligned-linear \
```

To enable TP-SP, add the following parameters:

```shell
--unaligned-linear \
--sequence-parallel \
```

To enable CP, add the following parameter:

```shell
--context-parallel-algo megatron_cp_algo \
```

To enable PP, add the following parameter:

```shell
--variable-seq-lengths \
```

To enable VPP, add the following parameter (*N* is  the number of VPP partitions). For feature details, see the [feature guide](../../../docs/en/features/virtual_pipeline_parallel.md).

```shell
--virtual-pipeline-model-parallel-size N \
```

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Take InternVL3-8B as an example to start the fine-tuning training task.
Differences in loss calculation methods can have varying impacts on training results. Before starting the training task, please review the documentation on loss calculation and select an appropriate loss calculation method. For details, see [vlm_model_loss_calculate_type.md](../../../docs/en/features/vlm_model_loss_calculate_type.md).

```shell
bash examples/mindspore/internvl3/finetune_internvl3_8B.sh
```

<a id="jump5"></a>

## Environment Variables Declaration

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

## Notes

1. When using the PP strategy for multi-node training, the process may hang. You can refer to [this PR](https://gitcode.com/Ascend/MindSpeed/pulls/1627/files) for a fix.
