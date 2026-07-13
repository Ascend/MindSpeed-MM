# InternVL3 User Guide

<p align="left">
</p>

## Contents

- [InternVL3 User Guide](#internvl3-user-guide)
  - [Contents](#contents)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Setup](#environment-setup)
    - [1. Repository Cloning](#1-repository-cloning)
    - [2. Environment Setup](#2-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion](#2-weight-conversion)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download](#1-dataset-download)
  - [Fine-tuning](#fine-tuning)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
  - [Inference](#inference)
    - [1. Prerequisites](#1-prerequisites-1)
    - [2. Parameter Configuration](#2-parameter-configuration-1)
    - [3. Start Inference](#3-start-inference)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Notes](#notes)

## Version Description

### Reference Implementation

```shell
url=https://github.com/OpenGVLab/InternVL.git
commit_id=d779db3
```

### Changelog

- 2025.04.15: Support for online inference of InternVL3
- 2025.07.09: Support for InternVL3-78B, InternVL3-8B fine-tuning

<a id="jump1"></a>

## Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md).

<a id="jump1.1"></a>

### 1. Repository Cloning

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
mkdir logs
mkdir dataset
mkdir ckpt
```

<a id="jump1.2"></a>

### 2. Environment Setup

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# Install torch and torch_npu. Ensure you select the torch, torch_npu, and apex packages that correspond to your Python version and architecture (x86 or arm).
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

# For apex for Ascend, please refer to https://gitcode.com/Ascend/apex.
# It is recommended to compile and install from the source repository.

# Install the acceleration library.
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# Checkout commit from MindSpeed core_r0.12.1
git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
pip install -r requirements.txt
pip3 install -e .
cd ..
# Install other required dependency libraries.
pip install -e .
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

MindSpeed MM has modified the structure names of some original networks. Use the `mm-convert` tool to convert the original pre-trained weights. This tool implements the conversion between Hugging Face weights and MindSpeed MM weights, as well as weight sharding for pipeline parallelism (PP).

For detailed usage of the `mm-convert` tool, please refer to [Weight Conversion Tool Usage](../../docs/en/features/mm_convert.md).

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
# mm_dir: Directory to save the converted model
# hf_dir: Hugging Face weight directory
# llm_pp_layers: Number of layers split per card for the LLM. Note that this must be consistent with the `pipeline_num_layers` configured in `model.json`
# vit_pp_layers: Number of layers split per card for the ViT. Note that this must be consistent with the `pipeline_num_layers` configured in `model.json`
# trust_remote_code: To ensure code security, `trust_remote_code` is configured to `False` by default. Users need to set it to `True` and ensure the security of the model and data they have downloaded.
```

Synchronously modify the `LOAD_PATH` parameter in `examples/internvl3/finetune_internvl3_*b.sh`. This path points to the converted or sharded weights. Be sure to distinguish it from the original weights in `raw_ckpt/InternVL3-*B`.

Taking `InternVL3-8B` as an example:

```shell
LOAD_PATH="pretrained/InternVL3-8B"
```

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download

[Image Data]

Users must obtain and extract the [InternVL-Finetune](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data) dataset to the `dataset/playground` directory. The extracted data structure is as follows:

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

To use videos for training, please refer to [Video Dataset Construction](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data) to build your own video dataset.

It also depends on the Decord library to read videos. The installation method for Decord is as follows:

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

Before configuring the script, you need to complete the preliminary preparations, including: **Environment Setup**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. For details, please refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data.json` according to the actual situation, including fields such as `from_pretrained`, `data_path`, and `data_folder`.

Taking InternVL3-8B as an example, make the following modifications to `data_8B.json`. Note that the weight path for `tokenizer_config` is the weight path before conversion.

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

Configure the parameters in `examples/internvl3/finetune_internvl3_xx.sh` according to the actual situation, including the load path, save path, and `--save-interval` (Note: Saving distributed optimizer files takes a long time, so please set the save interval carefully). Taking InternVL3-8B as an example:

```shell
...
# Load Path
LOAD_PATH="ckpt/InternVL3-8B"
# Save Path
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # Do not load optimizer state; remove this line if loading is required.
    --no-load-rng \  # Do not load random number state; remove this if loading is required.
    --no-save-optim \  # Do not save optimizer state; remove this if saving is required.
    --no-save-rng \  # Do not save random number state; remove this if saving is required.
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # Log Interval
    --save-interval 5000 \  # Save Interval
    ...
    --log-tps \  # Adding this parameter enables printing the average sequence length of the language module at each step during training, and calculating the throughput in tokens per second after training completes.
"
```

If you need to load the weights, optimizer states, etc., for a specific iteration, set `LOAD_PATH` to `"save_dir"`, and modify the `latest_checkpointed_iteration.txt` file to contain the specific iteration number.

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Single-Node Execution Configuration]

Configure the parameters in `examples/internvl3/finetune_internvl3_xx.sh` as follows.

```shell
  # Modify the ascend-toolkit path according to your actual situation
  source /usr/local/Ascend/cann/set_env.sh
  NPUS_PER_NODE=8
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

[Model Parallelism Configuration]

InternVL involves non-aligned TP partitioning. To enable TP partitioning, the following parameters must be added. For more information, please refer to te [feature guide](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/unaligned_linear.md).

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

To enable VPP, add the following parameter (where *N* is the number of VPP partitions). For more information, please refer to te [feature guide](../../docs/en/features/virtual_pipeline_parallel.md).

```shell
--virtual-pipeline-model-parallel-size N \
```

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Take InternVL3-8B as an example to start the fine-tuning training task.
Differences in loss calculation methods can have varying impacts on training effectiveness. Before starting the training task, please refer to the documentation on loss calculation and select an appropriate loss calculation method. For more information, see [vlm_model_loss_calculate_type.md](../../docs/en/features/vlm_model_loss_calculate_type.md).

```shell
bash examples/internvl3/finetune_internvl3_8B.sh
```

<a id="jump5"></a>

## Inference

<a id="jump5.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the preliminary preparations, including environment setup and weight download and conversion. For details, please refer to the corresponding sections. Currently, 8B single-card inference is supported.

The command for inference weight conversion is as follows:

```shell
# Modify the ascend-toolkit path according to your actual situation
source /usr/local/Ascend/cann/set_env.sh

# 8B
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL3-8B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL3-8B" \
  --cfg.parallel_config.llm_pp_layers [[28]] \
  --cfg.parallel_config.vit_pp_layers [[24]] \
  --cfg.trust_remote_code True
# trust_remote_code: To ensure code security, the configuration `trust_remote_code` defaults to `False`. Users need to set it to `True` and ensure the security of the models and data they download.
```

<a id="jump5.2"></a>

### 2. Parameter Configuration

[Parameter Configuration]

Modify the `inference_8B.json` file, including fields such as `infer_data_type`, `file_path`, `prompts`, `from_pretrained`, and the tokenizer's `from_pretrained`.

[Single-Image Inference]

Taking InternVL3-8B as an example, modify the corresponding parameters in `inference_8B.json` according to the actual situation. Note that the weight path for `tokenizer_config` is the weight path before conversion.

```json
{
    "infer_data_type": "image",
    "file_path": "./examples/internvl3/view.jpg",    # Enter the image path according to the actual situation
    "prompts": "Please describe the image shortly.", # Enter the prompt according to the actual situation (both Chinese and English are supported)
    "model_id": "InternVLPipeline",
    "from_pretrained": "./pretrained/InternVL3-8B/release/mp_rank_00/model_optim_rng.pt", # Note that the path should point to the .pt file
    ...
    "tokenizer":{
        ...
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "raw_ckpt/InternVL3-8B",
        ...
    },
    ...
}
```

[Video Inference]

Taking InternVL3-8B as an example, modify the corresponding parameters in `inference_8B.json` according to your actual situation. Note that the weight path for `tokenizer_config` is the weight path before conversion.

Inference demo video: [red-panda](https://huggingface.co/OpenGVLab/InternVL2-8B/blob/main/examples/red-panda.mp4).

```json
{
    "infer_data_type": "video",
    "file_path": "examples/internvl3/red-panda.mp4",    # Enter the video path according to your actual situation.
    "prompts": "Please describe the video shortly.", # Enter the prompt according to your actual situation (supports both Chinese and English).
    "model_id": "InternVLPipeline",
    "from_pretrained": "./pretrained/InternVL3-8B/release/mp_rank_00/model_optim_rng.pt", # Note that the path should point to the .pt file.
    ...
    "tokenizer":{
        ...
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "raw_ckpt/InternVL3-8B",
        ...
    },
    ...
}
```

[Startup Script Configuration]
Modify the `inference_internvl.sh` script according to your actual situation.

```shell
# Modify the ascend-toolkit path according to your actual situation.
source /usr/local/Ascend/cann/set_env.sh
...
MM_MODEL="./examples/internvl3/inference_8B.json"
```

<a id="jump5.3"></a>

### 3. Start Inference

```shell
bash examples/internvl3/inference_internvl.sh
```

<a id="jump6"></a>

## Environment Variable Declaration

| Environment Variable          | Description                                                         | Value Description                                                                                         |
|-------------------------------|---------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
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
| `NPUS_PER_NODE`               | Configures the number of NPUs used on a compute node.                                                  | Integer value (e.g., `1`, `8`, etc.)                                                                                                                       |

<a id="jump7"></a>

## Notes

1. When using PP strategy for multi-node training, the process may hang. Please refer to [this PR](https://gitcode.com/Ascend/MindSpeed/pulls/1627/files) for a fix.
