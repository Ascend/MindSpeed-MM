# Whisper User Guide

<p align="left">
</p>

## Contents

- [Whisper User Guide](#whisper-user-guide)
  - [Contents](#contents)
  - [Environment Setup](#environment-setup)
    - [1. Repository Cloning](#1-repository-cloning)
    - [2. Environment Setup](#2-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion](#2-weight-conversion)
  - [Dataset Preparation](#dataset-preparation)
    - [1. Dataset Download](#1-dataset-download)
  - [Pre-training](#pre-training)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Pre-training](#3-start-pre-training)
    - [4. Model Inference and Weight Conversion](#4-model-inference-and-weight-conversion)
  - [Environment Variable Declaration](#environment-variable-declaration)

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
```

<a id="jump1.2"></a>

### 2. Environment Setup

```bash
    # python3.10
    conda create -n test python=3.10
    conda activate test

    # Install torch and torch_npu. Ensure you select the torch, torch_npu, and apex packages that correspond to your Python version and architecture (x86 or arm).
    pip install torch-2.7.1-cp310-cp310-manylinux2014_aarch64.whl
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

    # Install other dependency libraries.
    pip install librosa datasets==3.6.0
    conda install -c conda-forge libsndfile
    pip install -e .
```

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download open-source model weights from websites such as Hugging Face.

- [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3/tree/main)

The Whisper large-v3 model weight structure is as follows:

   ```shell
   $whisper-large-v3
   ├── config.json
   ├── pytorch_model.bin
   ├── tokenizer.json
   └── ...
   ```

<a id="jump2.2"></a>

### 2. Weight Conversion

MindSpeed MM has modified some of the original network structure names, so you need to use the following script to convert the downloaded pre-training weights.

```python
import torch

pretrained_checkpoint = torch.load("your pretrained ckpt path", map_location="cpu")
new_checkpoint = {}
for key in pretrained_checkpoint.keys():
    model_key = key.replace("q_proj", "proj_q")
    model_key = model_key.replace("k_proj", "proj_k")
    model_key = model_key.replace("v_proj", "proj_v")
    model_key = model_key.replace("out_proj", "proj_out")
    new_checkpoint[model_key] = pretrained_checkpoint[key]

torch.save(new_checkpoint, "whisper.pth")
```

<a id="jump3"></a>

## Dataset Preparation

<a id="jump3.1"></a>

### 1. Dataset Download

Download the `mozilla-foundation/common_voice_11_0` dataset yourself. The data structure is as follows:

   ```shell
   $common_voice_11_0
   ├── audio
   ├── ├── hi
   ├── ├── ├── train
   ├── ├── ├── ├── hi_train_0.tar
   ├── ├── ├── test
   ├── ├── ├── ...
   ├── ├── en
   ├── ├── ...
   ├── transcript
   ├── ├── hi
   ├── ├── ├── train.tsv
   ├── ├── ├── test.tsv
   ├── ├── ├── ...
   ├── ├── en
   ├── ├── ...
   ├── common_voice_11_0.py
   ├── count_n_shard.py
   └── ...
   ```

<a id="jump4"></a>

## Pre-training

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the preliminary preparations, including: **Environment Setup**, **Weight Download and Conversion**, **Dataset Preparation**, etc. For details, please refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

Modify the weight and dataset paths in `model.json` and `data.json` according to the actual situation, including fields such as `ckpt_path`, `dataset_name_or_path`, and `processor_name_or_path`.

[Single-Node Execution]

```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=localhost
    MASTER_PORT=29501
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

[Multi-Node Execution]

```shell
    # Configure distributed parameters according to the actual situation of the distributed cluster.
    GPUS_PER_NODE=8  #Number of NPUs per node
    MASTER_ADDR="your master node IP"  # All must be modified to the IP address of the master node (cannot be localhost).
    MASTER_PORT=29501
    NNODES=2  # Number of nodes in the cluster. Fill in based on the actual situation.
    NODE_RANK="current node id"  # RANK of the current node. Must be unique across nodes. The rank of the master node is 0, and ranks of other nodes can be 1, 2, etc.
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

### 3. Start Pre-training

```shell
    bash examples/whisper/pretrain_whisper.sh
```

**Note**:

- For multi-node training, the pre-training script must be started simultaneously on multiple terminals (the pre-training script on each terminal differs only in the `NODE_RANK` parameter; all other parameters are identical).
- If multi-node training is enabled, training data must be prepared on each node.

<a id="jump5"></a>

### 4. Model Inference and Weight Conversion

Currently, MindSpeed-MM does not provide inference code for the whisper model. You need to convert the trained model back to the Hugging Face format for inference. An example of the conversion script is as follows:

```python
import torch
import mindspeed.megatron_adaptor

pretrained_checkpoint = torch.load("your trained ckpt path/model_optim_rng.pt", map_location="cpu")
pretrained_checkpoint = pretrained_checkpoint['model']

new_checkpoint = {}
for key in pretrained_checkpoint.keys():
    if key == "proj_out.weight":
        model_key = key
    else:
        model_key = key.replace("proj_q", "q_proj")
        model_key = model_key.replace("proj_k", "k_proj")
        model_key = model_key.replace("proj_v", "v_proj")
        model_key = model_key.replace("proj_out", "out_proj")
    new_checkpoint[model_key] = pretrained_checkpoint[key]

torch.save(new_checkpoint, "whisper_hf.bin")
```

**Note**:

- This conversion script must be executed in the root directory of MindSpeed-MM.
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
| `NPUS_PER_NODE`               | Configures the number of NPUs used on a compute node.                                                  | Integer value (e.g., `1`, `8`, etc.)                                                                            |
