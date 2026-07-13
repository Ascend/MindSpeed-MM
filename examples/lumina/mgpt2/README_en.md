# Lumina-mGPT2 User Guide

- [Lumina-mGPT2 User Guide](#lumina-mgpt2-user-guide)
  - [Version Notes](#version-notes)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Setup](#environment-setup)
    - [Repository Cloning](#repository-cloning)
    - [Environment Setup](#environment-setup-1)
    - [Decord Setup](#decord-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [Weight Conversion](#weight-conversion)
  - [Pre-training](#pre-training)
    - [Data Preprocessing](#data-preprocessing)
    - [Feature Extraction](#feature-extraction)
      - [Preparation](#preparation)
      - [Parameter Configuration](#parameter-configuration)
      - [Start Feature Extraction](#start-feature-extraction)
    - [Training](#training)
      - [Preparation](#preparation-1)
      - [Parameter Configuration](#parameter-configuration-1)
      - [Start Training](#start-training)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Notes

### Reference Implementation

T2I fine-tuning task:

```shell
url=https://github.com/Alpha-VLLM/Lumina-mGPT-2.0
commit_id=978feb32473b57b79ea6a709687d01107e630478
```

### Changelog

2025.08.15: Initial release of the Lumina-mGPT2 fine-tuning task.

## Environment Setup

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md).

### Repository Cloning

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
```

### Environment Setup

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# Install torch and torch_npu. Ensure you select the torch, torch_npu, and apex packages corresponding to your Python version and architecture (x86 or ARM).
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
git checkout e92252f4f1b7cbd78868922e6fe5659f8b762bf8
pip install -r requirements.txt
pip install -e .
cd ..

# Install other required dependency libraries.
pip install -e .
```

### Decord Setup

X86:

```bash
pip install decord==0.6.0
```

Arm:

For installation via `apt`, please [refer to the link](https://github.com/dmlc/decord).

For installation via `yum`, please [refer to the script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

## Weight Download and Conversion

|   Model   |   Download Link   |
| ---- | ---- |
|   Lumina-mGPT2 7B  |   <https://huggingface.co/Alpha-VLLM/Lumina-mGPT-2.0/tree/main>   |
|  MoVQGAN  |  <https://huggingface.co/ai-forever/MoVQGAN/resolve/main/movqgan_270M.ckpt>    |

### Weight Conversion

You need to perform weight conversion on the downloaded Lumina-mGPT2 model weights. Run the weight conversion script:

```shell
mm-convert LuminaConverter hf_to_mm \
 --cfg.source_path <./Alpha-VLLM/Lumina-mGPT-2.0/> \
 --cfg.target_path <./Lumina/Lumina-mGPT-2.0-mm-convert/> \
```

The parameters of the weight conversion script are described as follows:

| Parameter | Meaning | Default Value |
|:------------|:----|:----|
| --cfg.source_path | Original weight path | / |
| --cfg.target_path | Path to save the converted weights | / |

## Pre-training

### Data Preprocessing

Process the data into the following format:

```bash
</data/hunyuanvideo/dataset>
  ├──data.json
  ├──images
  │  ├──image0001.jpg
  │  ├──image0002.png
```

Where, `images/` stores the images, and `data.json` contains all image-text pair information in the dataset. A specific example is as follows:

```json
[
    {
        "file": "images/image0001.jpg",
        "prompt": "Image discrimination1."
    },
    {
        "file": "images/image0002.jpg",
        "prompt": "Image discrimination2."
    },
    ......
]
```

### Feature Extraction

#### Preparation

Before starting, ensure that environment preparation, model weights, and dataset preprocessing are complete.

#### Parameter Configuration

Check whether configurations such as the model weight path, dataset path, and extracted feature save path are complete.

| Configuration File                                             |       Field to be Modified        | Instructions                                            |
| ------------------------------------------------------------ | :-------------------: | :-------------------------------------------------- |
| examples/lumina/mgpt2/feature_extract/data.json              |         path          | Path to the `data.json` file        |
| examples/lumina/mgpt2/feature_extract/data.json              |     from_pretrained    | Modify it to the path corresponding to the downloaded Lumina mGPT2 weights.     |
| examples/lumina/mgpt2/feature_extract/model.json              |    from_pretrained    | Modify it to the path corresponding to the downloaded MoVQGAN weights. |
| examples/lumina/mgpt2/feature_extract/feature_extraction.sh  |     NPUS_PER_NODE     | Number of NPUs                                                |
| mindspeed_mm/tools/tools.json                                |       save_path       | Save path for the extracted features                                |

#### Start Feature Extraction

```bash
bash examples/lumina/mgpt2/feature_extract/feature_extraction.sh
```

### Training

#### Preparation

Before starting, ensure that environment setup, model weight download, and feature extraction have been completed.

#### Parameter Configuration

Check whether the model weight path, parallel parameter configuration, etc., have been completed.

| Configuration File | Field to be Modified | Instructions |
| ---------------------------------------------------------- | :-----------------: | :-------------------------------------------------- |
| examples/lumina/mgpt2/feature_data.json | basic_parameters | Dataset path; configure `path` to the file path of the extracted features. |
| examples/lumina/mgpt2/model.json | vocabulary_map_path | Vocabulary file path; configure it to the path corresponding to the downloaded original Lumina mGPT2 weights. |
| examples/lumina/mgpt2/pretrain.sh | NPUS_PER_NODE | Number of NPUs per node |
| examples/lumina/mgpt2/pretrain.sh | NNODES | Number of nodes |
| examples/lumina/mgpt2/pretrain.sh | LOAD_PATH | Pre-training weight path after weight conversion |
| examples/lumina/mgpt2/pretrain.sh | SAVE_PATH | Path for saving weights during training |

[Parallel Parameter Configuration]

When adjusting model parameters or token sequence length, enable the following parallel strategies based on the actual situation and determine the optimal parallel strategy through debugging.

- fsdp1

  - Use case: When the model parameter scale is large and a single card cannot accommodate the complete model, enable fsdp1 to reduce memory usage.

  - Enablement method: Add fsdp1 configuration information in `examples/lumina/mgpt2/model.json`.

  - Limitation: This feature is currently incompatible with model sharding. When enabling this feature, `TP` and `PP` must be set to `1`.

> ⚠️ Currently this feature is not adapted for CP and TPSP.

#### Start Training

```bash
bash examples/lumina/mgpt2/pretrain.sh
```

## Environment Variable Declaration

| Environment Variable                      | Description                                                                 | Value Description                                                                                         |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
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
