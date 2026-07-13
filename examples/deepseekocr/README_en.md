# DeepSeek-OCR User Guide

<p align="left">
</p>

## Contents

- [DeepSeek-OCR User Guide](#deepseek-ocr-user-guide)
  - [Contents](#contents)
  - [Introduction](#introduction)
    - [Reference Implementation](#reference-implementation)
    - [Change Log](#change-log)
  - [Environment Installation](#environment-installation)
    - [1. Repository Clone](#1-repository-clone)
    - [2. Environment Setup](#2-environment-setup)
  - [Weight Download](#weight-download)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
  - [Training](#training)
    - [1. Preparation](#1-preparation)
    - [2. Start Training](#2-start-training)
  - [Environment Variable Declaration](#environment-variable-declaration)

<a id="jump0"></a>

## Introduction

[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) is a vision-language model launched by the DeepSeek team, focusing on efficiently processing long text content through optical compression technology. The model consists of a DeepEncoder encoder and a DeepSeek3B-MoE decoder, which can significantly reduce activation memory and the number of visual tokens while maintaining high-resolution input. The model achieves an OCR accuracy of 97% at a 10x compression ratio and maintains 60% accuracy at a 20x compression ratio. DeepSeek-OCR supports multiple resolution modes, suitable for multilingual document processing. It can parse complex content such as charts and chemical formulas, providing an efficient solution for large-scale document processing.

### Reference Implementation

```shell
url=https://github.com/deepseek-ai/DeepSeek-OCR/
commit_id=e4ac34e1e59b891163fb9325480fbedec865e1f0
```

### Change Log

2025.11.04: Established the DeepSeek-OCR fixed-size image training pipeline.

<a id="jump1"></a>

## Environment Installation

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/installation.md)

> Note: Python version 3.10 is recommended, and torch and torch_npu version 2.7.1 are recommended.

<a id="jump1.1"></a>

### 1. Repository Clone

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git

cd MindSpeed-MM
```

<a id="jump1.2"></a>

### 2. Environment Setup

```bash
# Install python3.10.
conda create -n test python=3.10
conda activate test

# Install torch and torch_npu.
pip install torch-2.7.1-cp310-cp310-*.whl
pip install torch_npu-2.7.1*.manylinux2014_aarch64.whl

# Install MindSpeed MM dependencies.
pip install -r examples/deepseekocr/requirements.txt

```

<a id="jump2"></a>

## Weight Download

Download open-source model weights from websites such as Hugging Face

- [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

<a id="jump3"></a>

## Dataset Preparation and Processing

DeepSeekOCR has not open-sourced its dataset. Here we take the [CC-OCR dataset](https://huggingface.co/datasets/wulipc/CC-OCR) as an example.
Download the dataset and place it in the `./data/` folder. Then, run the data conversion script `python examples/deepseekocr/convert_ccocr_to_dsvlocr.py`. After preprocessing, the generated data format is as follows:

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
    ... ...
   ```

The example of data paths is as follows:

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

### 1. Preparation

Before configuring the scripts, you need to complete the preliminary preparations by referring to the corresponding sections, including: environment installation, weight download, and dataset preparation and processing.

<a id="jump4.2"></a>

### 2. Start Training

1. Taking the fine-tuning task of image-text understanding as an example, you can modify the configuration of the [startup script](../../examples/deepseekocr/finetune_ocr.sh) according to the actual situation. The following configurations must be modified:

    ``` shell
    DATA_PATH="./data/output.jsonl" # Dataset files
    DATA_DIR="./data" # Directory where the dataset depends on files such as images and text
    LOAD_PATH="./ckpt/deepseek-ai/DeepSeek-OCR" # Path to the weights downloaded from Hugging Face
    ```

2. Modify the `NNODES` and `NPUS_PER_NODE` configurations as required. For example, for a single A2 server, you can set `NNODES` to `1` and `NPUS_PER_NODE` to `8`;

3. To ensure code security, `trust_remote_code` is set to `False` by default. You need to enable `--trust-remote-code` in the startup script and ensure the security of the downloaded model and data.

4. After the above points have been modified, you can start the script to begin training:

    ```bash
    bash examples/deepseekocr/finetune_ocr.sh
    ```

<a id="jump6"></a>

## Environment Variable Declaration

| Environment Variable                      | Description                                                                 | Description                                                                                         |
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
