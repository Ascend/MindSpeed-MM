# OpenSora2.0 User Guide

<p align="left">
</p>

## Contents

- [OpenSora2.0 User Guide](#opensora20-user-guide)
  - [Contents](#contents)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Installation](#environment-installation)
    - [1. Repository Cloning](#1-repository-cloning)
    - [2. Environment Setup](#2-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion](#2-weight-conversion)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
  - [Pre-training](#pre-training)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Pre-training](#3-start-pre-training)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Description

### Reference Implementation

```shell
url=https://github.com/hpcaitech/Open-sora.git
commit_id=d0cd5ac
```

### Changelog

2025.06.25: Initial support for Open-sora 2.0 T2V

<a id="jump1"></a>

## Environment Installation

It is recommended to use the matching environment version during model development.

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

# For apex for Ascend, refer to https://gitcode.com/Ascend/apex.
# It is recommended to compile and install from the source repository.

# Modify the environment variable paths in the shell script to the actual paths. Example:
source /usr/local/Ascend/cann/set_env.sh

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

# Specify the av version.
pip install av==16.1.0

```

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the open-source model weights from Hugging Face.

- [OpenSoraV2 Model](https://huggingface.co/hpcai-tech/Open-Sora-v2/blob/main/Open_Sora_v2.safetensors)
- [vae model](https://huggingface.co/hpcai-tech/Open-Sora-v2/blob/main/hunyuan_vae.safetensors)
- [T5 model](https://huggingface.co/hpcai-tech/Open-Sora-v2/tree/main/google)
- [Clip model](https://huggingface.co/hpcai-tech/Open-Sora-v2/tree/main/openai)

<a id="jump2.2"></a>

### 2. Weight Conversion

Weight conversion is required for the [OpenSoraV2 model]. Run the weight conversion script:

```shell
mm-convert OpenSoraConverter hf_to_mm \
  --cfg.source_path <OpenSoraV2 mode> \
  --cfg.target_path <Path to the converted OpenSoraV2 model>
```

<a id="jump3"></a>

## Dataset Preparation and Processing

Users need to prepare their own training dataset, which requires providing a corresponding collection of sliced videos (datasets) and a CSV file. The CSV file should be named `train_data.csv` and used as the `data_path` for model input.

The dataset directory structure is as follows:

   ```shell
   train_data.csv
   datasets
   ├── video1990_scene-4.mp4
   ├── video1990_scene-5.mp4
   ├── video1991_scene-1.mp4
   ...
   ```

The format of the CSV file content is as follows:

   ```shell
   path,text,num_frames,height,width,aspect_ratio,resolution,fps
   ./datasets/pexels_45k/popular_3/853857_scene-0_cut-border.mp4,"an aerial view of a large...",330.0,1036.0,1102.0,0.94010889292196,1141672.0,30.0
   ```

   Note: The path field in the CSV file needs to be filled with the relative or absolute path of sliced videos. If it is a relative path, the parent path needs to be supplemented in the `data_folder` field in the `data.json` file.

<a id="jump4"></a>

## Pre-training

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the following prerequisites: **Environment Installation**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

The default configuration has been tested. Users can modify the following content according to their own environment.

| Configuration File                                                   |     Field to be Modified      | Instruction                                           |
| -------------------------------------------------------- | :-----------------: | :-------------------------------------------------- |
| examples/opensora2.0/data.json                           |  basic_parameters   | `data_path` points to the CSV file, and `data_folder` (optional) is the path prefix for the dataset of sliced videos.  |
| examples/opensora2.0/pretrain_model.json           |  text_encoder  | Configures the paths for two text encoders: `"from_pretrained": "Open-Sora-v2/google/t5-v1_1-xxl"` and `"from_pretrained": "Open-Sora-v2/openai/clip-vit-large-patch14"` |
| examples/opensora2.0/pretrain_model.json           |       ae       | Configures the VAE model path `"from_pretrained": "Open-Sora-v2/hunyuan_vae.safetensors"`       |
| examples/opensora2.0/pretrain_opensora2_0.sh       |    NPUS_PER_NODE    | Number of NPUs per node                                      |
| examples/opensora2.0/pretrain_opensora2_0.sh       |       NNODES        | Number of nodes                                            |
| examples/opensora2.0/pretrain_opensora2_0.sh       |      LOAD_PATH      | Path to the pre-training weights after weight conversion                          |
| examples/opensora2.0/pretrain_opensora2_0.sh       |      SAVE_PATH      | Path to weights saved during training                            |

[Dataset Bucket Configuration]

`bucket_config (dict)`: A dictionary containing the bucket configuration.

The dictionary should use the following format:

```json
"bucket_config": {
    "256px": {"1": [1.0, 3], "125": [1.0, 2], "129": [1.0, 1]},
    "720p": {"100": [0.5, 1]}
}
```

Example:

`256px` indicates a video with a resolution of 256*256 pixels.

`720p` indicates a video with an aspect ratio of 16:9 and a height of 720 pixels.

In `{"100": [0.5, 1]}`, `100` indicates the number of video frames, `0.5` is the video sampling probability (a float between 0 and 1), and `1` is the batch_size for the current video specification.

[Parallel Parameter Configuration]

Due to the large scale of the OpenSora2.0 model parameters, a single machine cannot run the complete model, so the default configuration has integrated `layer_zero`.

+ Introduction to LayerZeRO

  - Use Case: When the model parameter scale is large and a single card cannot accommodate the complete model, you can enable LayerZeRO to reduce static memory.

  - Enabling Method: Add `--layerzero` and `--layerzero-config $LAYERZERO_CONFIG` to the `GPT_ARGS` in `examples/opensora2.0/pretrain_opensora2_0.sh`.

  - Suggestion: It is recommended to set `zero3_size` in `examples/opensora2.0/zero_config.yaml` to the number of cards on a single machine.

  - Training Weight Post-processing: When training with this feature is performed, the saved weights need to be post-processed using the following conversion script before they can be used for inference:

  ```bash
  # Modify the ascend-toolkit path according to the actual situation.
  source /usr/local/Ascend/cann/set_env.sh
  # Replace your_mindspeed_path and your_megatron_path with the paths of the previously downloaded MindSpeed and megatron respectively.
  export PYTHONPATH=$PYTHONPATH:<your_mindspeed_path>
  export PYTHONPATH=$PYTHONPATH:<your_megatron_path>
  # input_folder is the path where Layer Zero training saves weights, and output_folder is the path for the output Megatron format weights
  mm-convert OpenSoraConverter layerzero_to_mm \
      --cfg.source_path <./save_ckpt/opensora2/> \
      --cfg.target_path <./save_ckpt/opensora2_megatron_ckpt/>
  ```

<a id="jump4.3"></a>

### 3. Start Pre-training

```shell
bash examples/opensora2.0/pretrain_opensora2_0.sh
```

<a id="jump5"></a>

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
| `NPUS_PER_NODE`               | Configures the number of NPUs used on a compute node.                                                  | Integer value (e.g., `1`, `8`, etc.)                                                                                                                       |
