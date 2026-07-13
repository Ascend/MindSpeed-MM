# OpenSoraPlan1.3.1 VAE User Guide

<p align="left">
</p>

## Contents

- [OpenSoraPlan1.3.1 VAE User Guide](#opensoraplan131-vae-user-guide)
  - [Contents](#contents)
  - [Environment Installation](#environment-installation)
    - [1. Repository Cloning](#1-repository-cloning)
    - [2. Environment Setup](#2-environment-setup)
    - [3. Decord Setup](#3-decord-setup)
  - [Weight Downloading](#weight-downloading)
    - [1. Weight Downloading and Saving](#1-weight-downloading-and-saving)
  - [Dataset Preparation](#dataset-preparation)
    - [1. Dataset Downloading](#1-dataset-downloading)
  - [Pre-training](#pre-training)
    - [1. Preparation](#1-preparation)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Pre-training](#3-start-pre-training)
  - [Environment Variable Declaration](#environment-variable-declaration)

<a id="jump1"></a>

## Environment Installation

It is recommended to use the matching environment version during model development.

Please refer to [Installation Guide](../../docs/en/pytorch/install_guide.md).

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

    # Install torch and torch_npu, making sure to select the torch, torch_npu, and apex packages corresponding to the Python version and x86 or arm architecture.
    pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
    pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

    # apex for Ascend reference https://gitcode.com/Ascend/apex.
    # It is recommended to compile and install from the original repository.

    # Modify the environment variable paths in the shell script to the actual paths. Example:
    source /usr/local/Ascend/cann/set_env.sh

    # Install the acceleration library.
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed
    # checkout commit from MindSpeed core_r0.12.1
    git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
    pip install -r requirements.txt
    pip install -e .
    cd ..

    # Install the required dependency libraries.
    pip install -e .
```

<a id="jump1.3"></a>

### 3. Decord Setup

[X86 Installation]

```bash
pip install decord==0.6.0
```

[Arm Installation]

For `apt` installation, please [refer to the link](https://github.com/dmlc/decord).

For `yum` installation, please [refer to the script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

<a id="jump2"></a>

## Weight Downloading

<a id="jump2.1"></a>

### 1. Weight Downloading and Saving

Download the pre-trained **visual perception** models.

- [vgg_lpips](https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1): Download the `vgg.pth` weights to the `MindSpeed-MM/.cache/lpips/` directory.

- [vgg16](https://download.pytorch.org/models/vgg16-397923af.pth): Download the `vgg16-397923af.pth` weights to the user directory `~/.cache/torch/hub/checkpoints/`;

<a id="jump3"></a>

## Dataset Preparation

<a id="jump3.1"></a>

### 1. Dataset Downloading

Download and decompress the [pixabay_v2](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/pixabay_v2_tar) dataset. The data structure is as follows:

   ```shell
   $pixabay_v2
   ├── folder_01
   ├── ├── video0.mp4
   ├── ├── video1.mp4
   ├── ├── ...
   ├── folder_02
   ├── folder_03
   └── ...
   ```

<a id="jump4"></a>

## Pre-training

<a id="jump4.1"></a>

### 1. Preparation

Before configuring the script, you need to complete the prerequisite preparations, including: **Environment Installation**, **Weight Downloading and Saving**, **Dataset Preparation**. For details, please refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

The weight and dataset paths in `model.json` and `data.json` need to be modified according to the actual situation, including the `from_pretrained`, `perceptual_from_pretrained`, and `video_folder` fields.

[Single-node Running Configuration]

```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=localhost
    MASTER_PORT=29505
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

[Multi-node Running Execution]

```shell
    # Configure distributed parameters according to the actual situation of the distributed cluster.
    GPUS_PER_NODE=8  # Number of cards per node
    MASTER_ADDR="your master node IP"  # All need to be modified to the IP address of the master node (cannot be localhost).
    MASTER_PORT=29505
    NNODES=2   # Number of nodes in the cluster. Fill in according to the actual situation.
    NODE_RANK="current node id"  # RANK of the current node; cannot be repeated across multiple nodes. The rank of the master node is 0, and ranks of other nodes can be 1, 2, etc.
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

### 3. Start Pre-training

```shell
    bash examples/vae/pretrain_vae.sh
```

**Note**:

- For multi-node training, the pre-training script must be started simultaneously in multiple terminals (the pre-training script in each terminal differs only in the `NODE_RANK` parameter; all other parameters are the same)
- If multi-node training is used, training data and model weights need to be prepared on each node

<a id="jump5"></a>

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
