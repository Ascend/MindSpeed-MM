# DeepSeek-OCR-2 User Guide

<p align="left">
</p>

## Contents

- [DeepSeek-OCR-2 User Guide](#deepseek-ocr-2-user-guide)
  - [Contents](#contents)
  - [Introduction](#introduction)
    - [Reference Implementation](#reference-implementation)
    - [Change Log](#change-log)
  - [Environment Installation](#environment-installation)
    - [1. Repository Cloning](#1-repository-cloning)
    - [2. Environment Installation](#2-environment-installation)
  - [Weight Download](#weight-download)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
  - [Training](#training)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Start Training](#2-start-training)
  - [Environment Variable Declaration](#environment-variable-declaration)

<a id="jump0"></a>

## Introduction

[DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2) is a vision-language model launched by the DeepSeek team, focused on efficiently processing long text content through optical compression technology. The model innovatively introduces DeepEncoder V2, an LLM-based image causal inference encoder, paired with the DeepSeek3B-MoE decoder, achieving efficient utilization of visual tokens. It performs exceptionally well on complex layout tasks (formulas/tables), achieving a recognition accuracy of up to 91% on OmniDocBench v1.5, a 3.73% improvement over DeepSeek-OCR.

### Reference Implementation

```shell
url=https://github.com/deepseek-ai/DeepSeek-OCR-2/
commit_id=c548f2e60d409fc1d92208472c0f2389998af632
```

### Change Log

2026.01.31: Established the fixed-size image training pipeline for DeepSeek-OCR-2.

<a id="jump1"></a>

## Environment Installation

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/installation.md)

> Note: Python version 3.10 is recommended, and torch and torch_npu version 2.7.1 are recommended.

<a id="jump1.1"></a>

### 1. Repository Cloning

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git

cd MindSpeed-MM
```

<a id="jump1.2"></a>

### 2. Environment Installation

```bash
# Install MindSPeed MM Dependencies
pip install -r examples/deepseekocr2/requirements.txt
```

<a id="jump2"></a>

## Weight Download

Download the open-source model weights from websites such as Hugging Face:

- [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)

<a id="jump3"></a>

## Dataset Preparation and Processing

DeepSeek-OCR-2 does not open-source its dataset. Here, we use the [CC-OCR dataset](https://huggingface.co/datasets/wulipc/CC-OCR) as an example.

1. Download the dataset and place it in the `./data/` folder.
2. Run the data conversion script `python examples/deepseekocr/convert_ccocr_to_dsvlocr.py`.
After preprocessing, the generated data format is as follows:

```json
{
      'id': i,
      'conversations': [
          {
              "role": "<|User|>",
              "content": "Free OCR.",
              "images": [f"{save_file}"]
          },
          {
              "role": "<|Assistant|>",
              "content": answer
          }
      ]
}
    ...
```

The example of data paths are as follows:

```json
$playground
|--data
  |--CC-OCR
  |--convert
      |--*jpg
      ···
  |--output.jsonl
```

<a id="jump4"></a>

## Training

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the scripts, you need to complete the preliminary preparations by referring to the corresponding sections, including: environment installation, weight download, and dataset preparation and processing.

<a id="jump4.2"></a>

### 2. Start Training

1. Taking the fine-tuning task for image-text understanding as an example, you can modify the configuration of the [launch script](../../examples/deepseekocr/finetune_ocr.sh) based on your actual situation. The following configurations must be modified:

    ``` shell
    DATA_PATH="./data/output.jsonl" # Dataset files
    DATA_DIR="./data" # Directory for dataset dependencies such as images and text files
    LOAD_PATH="./ckpt/deepseek-ai/DeepSeek-OCR-2" # Path to weights downloaded from Hugging Face
    ```

2. Modify the `NNODES` and `NPUS_PER_NODE` configurations as required. For example, set `NNODES` to `1` and `NPUS_PER_NODE` to `8`.

3. To ensure code security, `trust_remote_code` is configured to `False` by default. You need to enable `--trust-remote-code` in the startup script and ensure the security of the downloaded model and data.

4. After completing the modifications mentioned above, you can start the script to begin training:

    ```bash
    bash examples/deepseekocr2/finetune_ocr2.sh
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
| `NPUS_PER_NODE`               | Configures the number of NPUs used on a compute node.                                                  | Integer value (e.g., `1`, `8`, etc.)
