# InternVL2.5 User Guide

<p align="left">
</p>

## Contents

- [InternVL2.5 User Guide](#internvl25-user-guide)
  - [Contents](#contents)
  - [Version Notes](#version-notes)
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
    - [1. Preparation](#1-preparation)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
  - [Inference](#inference)
    - [1. Preparation](#1-preparation-1)
    - [2. Parameter Configuration](#2-parameter-configuration-1)
    - [3. Launch Inference](#3-launch-inference)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Notes](#notes)

## Version Notes

### Reference Implementation

```shell
url=https://github.com/OpenGVLab/InternVL.git
commit_id=2d57e21
```

### Changelog

2025.02.20: Initial release of the InternVL2.5 model

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

# Install torch and torch_npu. Ensure you select the torch, torch_npu, and apex packages that correspond to your Python version and architecture (x86 or ARM).
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

# For apex for Ascend, refer to https://gitcode.com/Ascend/apex.
# It is recommended to compile and install from the original repository.

# Install the acceleration library.
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
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

- [InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B)

- [InternVL2_5-78B](https://huggingface.co/OpenGVLab/InternVL2_5-78B)

Save the model weights in the `raw_ckpt` directory, for example, `raw_ckpt/InternVL2_5-78B`.

<a id="jump2.2"></a>

### 2. Weight Conversion

MindSpeed MM modifies the structure names of some original networks. Use the `mm-convert` tool to convert the original pre-trained weights. This tool implements the conversion between Hugging Face weights and MindSpeed MM weights, as well as weight sharding for pipeline parallelism (PP).

For detailed usage of the `mm-convert` tool, refer to [Weight Conversion Tool usage](../../docs/en/features/mm_convert.md).

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
# hf_dir: Hugging Face weight directory
# llm_pp_layers: Number of LLM layers partitioned per card. Note that this must be consistent with the `pipeline_num_layers` configured in `model.json`.
# vit_pp_layers: Number of ViT layers partitioned per card. Note that this must be consistent with the `pipeline_num_layers` configured in `model.json`.
# trust_remote_code: To ensure code security, the configuration `trust_remote_code` defaults to `False`. Users need to set it to `True` and ensure the security of the models and data they download.
```

Synchronously modify the `LOAD_PATH` parameter in `examples/internvl2.5/finetune_internvl2.5_*b.sh`. This path points to the converted or sharded weights. Note that it should be distinguished from the original weights `raw_ckpt/InternVL2_5-*B`.

Take `InternVL2_5-78B` as an example.

```shell
LOAD_PATH="pretrained/InternVL2_5-78B"
```

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download

Image data:

Users need to obtain and decompress the [InternVL-Finetune](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data) dataset into the `dataset/playground` directory. Taking the `ai2d` dataset as an example, the directory structure after decompression is as follows:

   ```shell
   $playground
   ├── data
       ├── ai2d
           ├── abc_images
           ├── images
   ├── opensource
       ├── ai2d_train_12k.jsonl
   ```

Video data:

To train with video data, you can construct your own video dataset by referring to [Video Dataset Construction](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data).

The Decord library is also required for reading videos. The installation methods for Decord are as follows:

X86:

```bash
pip install decord==0.6.0
```

Arm:

For installation via `apt`, please [refer to this link](https://github.com/dmlc/decord).

For `yum` installation, please [refer to this script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

<a id="jump4"></a>

## Fine-tuning

<a id="jump4.1"></a>

### 1. Preparation

Before configuring the script, you need to complete the preliminary preparations, including: **Environment Setup**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data.json` according to the actual situation, including fields such as `from_pretrained`, `data_path`, and `data_folder`.

Taking InternVL2_5-78B as an example, make the following modifications to `data_78B.json`. Note that the weight path for `tokenizer_config` is the weight path before conversion.

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

Configure the parameters of `examples/internvl2.5/finetune_internvl2.5_xx.sh` according to the actual situation, including the load and save paths, as well as `--save-interval` (Note: Saving distributed optimizer files is time-consuming; please set the save interval cautiously).

Taking InternVL2.5-78B as an example:

```shell
...
# Load Path
LOAD_PATH="ckpt/InternVL2_5-78B"
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
    --log-tps \  # Enabling this parameter allows printing the average sequence length of the language module at each step during training, and calculating the throughput in tokens per second after training completes.
"
```

If you need to load the weights, optimizer states, etc., for a specific iteration count, set `LOAD_PATH` to `"save_dir"`, and modify the content of the `latest_checkpointed_iteration.txt` file to the specified iteration count.

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Single-Node Execution Configuration]

Configure the parameters in `examples/internvl2.5/finetune_internvl2.5_xx.sh` as follows:

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

### 3. Start Fine-tuning

Take InternVL2_5-78B as an example to start the fine-tuning training task.

```shell
bash examples/internvl2.5/finetune_internvl2.5_78B.sh
```

<a id="jump5"></a>

## Inference

<a id="jump5.1"></a>

### 1. Preparation

Before configuring the script, you need to complete the preliminary preparations, including environment setup, weight download, and weight conversion. For details, refer to the corresponding sections. (Currently, only 4B single-card inference is supported.)

The inference weight conversion command is as follows:

```shell
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh

# 4B
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2_5-4B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2_5-4B" \
  --cfg.parallel_config.llm_pp_layers [[36]] \
  --cfg.parallel_config.vit_pp_layers [[24]] \
  --cfg.trust_remote_code True
# trust_remote_code: To ensure code security, trust_remote_code is configured to False by default. Users need to set it to True and ensure the security of the models and data they download.
```

<a id="jump5.2"></a>

### 2. Parameter Configuration

[Parameter Configuration]

Modify the `inference_*B.json` file, including fields such as `infer_data_type`, `file_path`, `prompts`, `from_pretrained`, and the tokenizer's `from_pretrained`.

[Single-Image Inference]

Taking InternVL2_5-4B as an example, modify the corresponding parameters in `inference_4B.json` according to the actual situation. Note that the weight path for `tokenizer_config` is the weight path before conversion.

```json
{
    "infer_data_type": "image",
    "file_path": "./examples/internvl2.5/view.jpg",    # Enter the image path according to the actual situation.
    "prompts": "Please describe the image shortly.", # Enter the prompt according to the actual situation (supports both Chinese and English).
    "model_id": "InternVLPipeline",
    "from_pretrained": "./pretrained/InternVL2_5-4B/release/mp_rank_00/model_optim_rng.pt", # Note that the path must point to the .pt file.
    ...
    "tokenizer":{
        ...
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "raw_ckpt/InternVL2_5-4B",
        ...
    },
    ...
}
```

[Video Inference]

Taking InternVL2_5-4B as an example, modify the corresponding parameters in `inference_4B.json` according to the actual situation. Note that the weight path for `tokenizer_config` is the weight path before conversion.

Inference demo video: [red-panda](https://huggingface.co/OpenGVLab/InternVL2-8B/blob/main/examples/red-panda.mp4)

```json
{
    "infer_data_type": "video",
    "file_path": "examples/internvl2.5/red-panda.mp4",    # Enter the video path according to the actual situation.
    "prompts": "Please describe the video shortly.", # Enter the prompt based on the actual situation (both Chinese and English are supported)
    "model_id": "InternVLPipeline",
    "from_pretrained": "./pretrained/InternVL2_5-4B/release/mp_rank_00/model_optim_rng.pt", # Note that the path must point to the .pt file
    ...
    "tokenizer":{
        ...
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "raw_ckpt/InternVL2_5-4B",
        ...
    },
    ...
}
```

[Launch Script Configuration]
Modify the `inference_internvl.sh` script according to the actual situation.

```shell
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
...
MM_MODEL="./examples/internvl2.5/inference_4B.json"
```

<a id="jump5.3"></a>

### 3. Launch Inference

```shell
bash examples/internvl2.5/inference_internvl.sh
```

<a id="jump6"></a>

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
| `NPUS_PER_NODE`               | Configures the number of NPUs used on a compute node.                                                  | Integer value (e.g., `1`, `8`, etc.)                                                                                                                  |

<a id="jump7"></a>

## Notes

1. When using the PP strategy for multi-node training, a hang may occur. You can refer to [this PR](https://gitcode.com/Ascend/MindSpeed/pulls/1627/files) for a fix.
