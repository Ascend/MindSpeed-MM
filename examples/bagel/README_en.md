# Bagel Usage Guide

<p align="left">
</p>

## Contents

- [Bagel Usage Guide](#bagel-usage-guide)
  - [Contents](#contents)
  - [Version Notes](#version-notes)
    - [Reference Implementation](#reference-implementation)
  - [Environment Installation](#environment-installation)
    - [1. Environment Setup](#1-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
    - [2. Weight Conversion (hf2mm)](#2-weight-conversion-hf2mm)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download](#1-dataset-download)
  - [Fine-tuning](#fine-tuning)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Notes

### Reference Implementation

Training:

```bash
url=https://github.com/bytedance-seed/BAGEL
commit_id = 57c390
```

<a id="jump1"></a>

## Environment Installation

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md) to complete the Ascend software installation.
> Python version 3.10 is recommended, and torch and torch_npu version 2.7.1 are recommended.

<a id="jump1.1"></a>

### 1. Environment Setup

Pull the MindSpeed MM code repository and enter the root directory of the repository:

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

For X86 architecture machines, execute the following command:

```bash
bash scripts/install.sh --arch x86 --msid 93c45456c7044bacddebc5072316c01006c938f9
```

For Arm architecture machines, execute the following command:

```bash
bash scripts/install.sh --arch arm --msid 93c45456c7044bacddebc5072316c01006c938f9
```

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the corresponding model weights from the Hugging Face library:

- Model address: [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT/tree/main);

Save the downloaded model weights to the local `ckpt/hf_path/BAGEL-7B-MoT` directory.

<a id="jump2.2"></a>

### 2. Weight Conversion (hf2mm)

The Bagel model requires weight conversion for the downloaded weights. Run the weight conversion script:

```bash
# Bagel
mm-convert BagelConverter hf_to_mm \
 --cfg.source_path <./ckpt/hf_path/BAGEL-7B-MoT/> \
 --cfg.target_path <./ckpt/mm_path/BAGEL-7B-MoT/> \
```

The parameter descriptions for the weight conversion script are as follows:

| Parameter         | Meaning                          | Default Value                                                |
| :---------------- | :------------------------------- | :----------------------------------------------------------- |
| --cfg.source_path | Original weight path             | /                                                            |
| --cfg.target_path | Path to save converted or split weights | /                                                            |

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download

```bash

<https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip>

```

Process the data into the following format:

```bash

</dataset>
bagel_example
├── t2i/                           # text-to-image (parquet)
└── vlm/
    ├── images/                    # JPEG / PNG frames
    └── llava_ov_si.jsonl          # vision‑language SFT conversations
```

If you need to add your own dataset, please process the data into the same format as above.

<a id="jump4"></a>

## Fine-tuning

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the preliminary preparations, including: environment installation, weight download and conversion, and dataset preparation and processing. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

Data directory configuration:

Modify the dataset path in `data.json` according to the actual situation. Here, `num_files` indicates the number of t2i data files, which must be an integer multiple of the number of cards, and `num_total_samples` is the total number of data samples.

```json

    "t2i": {
      "data_dir": "data/t2i",
      "num_files": 8,
      "num_total_samples": 800
    },
    "llava_ov":{
      "data_dir": "data/vlm/images",
      "jsonl_path": "data/vlm/llava_ov_si.jsonl",
      "num_total_samples": 2000
    },
    ......
```

Weight path configuration:

| Configuration File                                                |   Modification Field  | Modification Instructions                                |
|-----------------------------------------------------| :---: |:------------------------------------|
| examples/bagel/data.json             |      model_path       | Modify to the path corresponding to the downloaded tokenizer weights           |
| examples/bagel/model.json             | from_pretrained | Modify to the weight path after weight conversion             |

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Before starting, ensure that environment preparation, model weight download, and weight conversion are complete.

Parallelization parameter configuration:

- FSDP2

  - Use Scenario: When the model parameter scale is large, enabling FSDP2 can reduce static memory.

  - Enabling Method: Add `--use-torch-fsdp2`, `--fsdp2-config-path ${fsdp2_config}`, `--untie-embeddings-and-output-weights`, and `--ckpt-format torch_dcp` to `GPT_ARGS` in `examples/bagel/finetune_bagel.sh`. For the `fsdp2_config` configuration, please refer to: [FSDP2 Description](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/fsdp2.md)

Start training:

 ```bash
 bash examples/bagel/finetune_bagel.sh
 ```

## Environment Variable Declaration

| Environment Variable            | Meaning                                                       | Value Description                                                                                                              |
|-------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | Specifies whether to enable log printing.                                    | `0`: Disable log screen printing.<br>`1`: Enable log screen. printing                                                                                        |
| `ASCEND_GLOBAL_LOG_LEVEL`     | Sets the log level for application logs and module logs; only supports debug logs. | `0`: DEBUG level<br>`1`: INFO level<br>`2`: WARNING level<br>`3`: ERROR level<br>`4`: NULL level, no log output                      |
| `TASK_QUEUE_ENABLE`           | Controls the level of `task_queue` operator dispatch queue optimization | `0`: Disable.<br>`1`: Enable Level 1 optimization.<br>`2`: Enable Level 2 optimization.                                                                  |
| `COMBINED_ENABLE`             | Sets the combined flag. Set to `0` to disable this feature; set to `1` to enable, used for optimizing non-consecutive two-operator combination. | `0`: Disable.<br>`1`: Enable.                                                                                      |
| `CPU_AFFINITY_CONF`           | Controls the processor affinity of CPU-side operator tasks, i.e., sets task core binding | `0` or not set: Core binding is not enabled.<br>`1`: Coarse-grained core binding is enabled<br>`2`: Fine-grained core binding is enabled.                                                          |
| `HCCL_CONNECT_TIMEOUT`        | Limits the timeout waiting time for socket connection establishment between different devices. | Must be configured as an integer in the value range `[120,7200]` (unit: s). The default value is `120`.                                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | Controls the behavior of the caching allocator.                    | `expandable_segments:<value>`: Enables the expandable segments of the memory pool, i.e., virtual memory characteristics.                                                                 |
| `HCCL_EXEC_TIMEOUT`           | Controls the synchronous waiting time during inter-device execution. Within this configured time, each device process waits for other devices to execute communication synchronization. | Must be configured as an integer in the value range `[68,17340]` (unit: s). The default value is `1800`.                                                                     |
| `ACLNN_CACHE_LIMIT`           | Configures the number of operator information entries cached on the host side for single-operator execution APIs | Must be configured as an integer in the value range `[1, 10,000,000]`. The default value is `10000`.                                                                         |
| `TOKENIZERS_PARALLELISM`      | Controls the behavior of the tokenizer in Hugging Face's transformers library in multi-threading environments | `False`: Disable parallel tokenization.<br>`True`: Enable parallel tokenization.                                                                               |
| `MULTI_STREAM_MEMORY_REUSE`   | Configures whether multi-stream memory reuse is enabled.           | `0`: Disable multi-stream memory reuse.<br>`1`: Enable multi-stream memory reuse.                                                                                    |
| `NPU_ASD_ENABLE`              | Controls whether to enable the feature value detection function of Ascend Extension for PyTorch. | `0` or not set: Disable feature value detection.<br>`1`: Enable feature value detection. Only print abnormal logs, without alarm.<br>`2`: Enable feature value detection and print alarms.<br>`3`: Enable feature value detection and print alarm, as well as process data in device-side info level logs. |
| `ASCEND_LAUNCH_BLOCKING`      | Controls whether to enable synchronous mode during operator execution. | `0`: Execute operator asynchronously.<br>`1`: Force operators to run in synchronous mode.    |
| `NPUS_PER_NODE`               | Configures the number of NPUs used on a compute node.              | Integer (e.g., `1`, `8`, etc.)
