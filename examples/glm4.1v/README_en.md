# Glm4.5v User Guide

<p align="left">
</p>

## Contents

- [Glm4.5v User Guide](#glm45v-user-guide)
  - [Contents](#contents)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Change Log](#change-log)
  - [Environment Installation](#environment-installation)
    - [1. Environment Preparation](#1-environment-preparation)
    - [2. Environment Setup](#2-environment-setup)
  - [Weights Download and Conversion](#weights-download-and-conversion)
    - [1. Weights Download](#1-weights-download)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download (Taking the COCO2017 Dataset as an Example)](#1-dataset-download-taking-the-coco2017-dataset-as-an-example)
    - [2. Pure Text or Mixed Training Data with and without Images (Taking LLaVA-Instruct-150K as an Example)](#2-pure-text-or-mixed-training-data-with-and-without-images-taking-llava-instruct-150k-as-an-example)
  - [Fine-tuning](#fine-tuning)
    - [1. Preparation](#1-preparation)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
    - [4. hf Weights Conversion](#4-hf-weights-conversion)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Notes](#notes)

## Version Description

### Reference Implementation

```shell
url=https://github.com/huggingface/transformers.git
commit_id=8cb5963
```

### Change Log

2025.11.29: Initial support for the Glm4.5v model

<a id="jump1"></a>

## Environment Installation

<a id="jump1.1"></a>

### 1. Environment Preparation

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md) to complete the Ascend software installation.
> Python 3.10 is recommended, and torch and torch_npu 2.7.1 are recommended.

‼️The acceleration features of MoE depend on newer versions of torch_npu and CANN. The following versions are recommended.

- [CANN](https://www.hiascend.com/document/detail/en/canncommercial/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)
- [torch_npu](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/insg_0004.html)

<a id="jump1.2"></a>

### 2. Environment Setup

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
mkdir logs data ckpt

# Install the acceleration library.
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout d76dbddd4517d48a2fc1cd494de8b9a6cfdbfbab

# Install MindSpeed and dependencies.
pip install -e .
cd ..
# Install MindSpeed MM and dependencies.
pip install -e .

# Install the latest version of transformers (supporting the glm4.5v model).
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 8cb5963
pip install -e .
```

<a id="jump2"></a>

## Weights Download and Conversion

<a id="jump2.1"></a>

### 1. Weights Download

Download the corresponding model weights from the Hugging Face library:

- Model address: [GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V).

 Save the downloaded model weights to the local `ckpt/hf_path/GLM-4.5V` directory. (`*` indicates the corresponding size.)

To ensure the network runs smoothly, MindSpeed MM has modified the structure names of the experts in the MoE, requiring the conversion of the original pre-training weights:

```bash
mm-convert ExpertMergeDcpConverter hf_to_dcp --hf_dir "ckpt/hf_path/GLM-4.5V" --save_dir "ckpt/mm_path/GLM-4.5V"
```

Due to the large parameter size of glm4.5v, `meta init` must be used to initialize and load the weights.
The `--init-model-with-meta-device` parameter needs to be added to `GPT_ARGS` in `examples/glm4.5v/finetune_glm4_5v106B.sh`. It has been included in the default script.
Additionally, when using `meta init` to initialize and load weights, `LOAD_PATH` in `examples/glm4.5v/finetune_glm4_5v106B.sh` needs to be set `ckpt/mm_path/GLM-4.5V`, the path where the weights are saved after conversion.

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download (Taking the COCO2017 Dataset as an Example)

(1) Users need to download the [COCO2017](https://cocodataset.org/#download) dataset and extract it to the `./data/COCO2017` folder under the project directory.

(2) Obtain the image dataset description file ([LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)) and download it to the `./data/` path.

(3) Run the data conversion script `python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py`. After conversion, the data directory structure is as follows:

   ```shell
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

Currently, reading multiple datasets separated by `,` (note: do not add spaces) is supported. This is implemented by the `dataset_param->basic_parameters->dataset` configuration in `data_106B.json`: change
`/data/mllm_format_llava_instruct_data.json` to `.//data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json`.

Also, note the configuration of `dataset_param->basic_parameters->max_samples` in `data_106B.json`. It limits data reading to only `max_samples` entries, which allows for quick function verification. For formal training, you can remove this parameter to read all the data.

<a id="jump3.2"></a>

### 2. Pure Text or Mixed Training Data with and without Images (Taking LLaVA-Instruct-150K as an Example)

This framework now supports pure text/mixed data (mixed training with and without image data).

During data construction, for data containing images, the `image` key-value pair must be retained.

```python
{
  "id": your_id,
  "image": your_image_path,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

During data construction, for pure text data, the `image` key-value pair can be removed.

```python
{
  "id": your_id,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

<a id="jump4"></a>

## Fine-tuning

<a id="jump4.1"></a>

### 1. Preparation

Before configuring the script, you need to complete the preliminary preparations, including: **Environment Installation**, **Weights Download and Conversion**, and **Dataset Preparation and Processing**. For details, please refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

Data directory configuration:

Modify the dataset paths in `data_106B.json` according to the actual situation, including `model_name_or_path`, `dataset_dir`, and `dataset`.

If the data and its corresponding JSON are both in the `/home/user/data/` directory, and the JSON directory is `/home/user/data/video_data_path.json`, set `dataset_dir` to `/home/user/data/` and
`dataset` to `./data/video_data_path.json`
Note that `dataset` needs to be configured as a relative path.

The modification example is as follows. Note that the weight path for `model_name_or_path` is the weight path before conversion, i.e., the original HF weights path.

**Note**: Do not configure `cache_dir` to the same mount directory across multiple nodes to avoid conflicts caused by writing to the same file.

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/GLM-4.5V",
            ...
        },
        "basic_parameters": {
            ...
            "dataset_dir": "./data",
            "dataset": "./data/mllm_format_llava_instruct_data.json",
            "cache_dir": "./data/cache_dir",
            ...
        },
        ...
    },
    ...
}
```

If you need to load large batches of data, you can enable streaming loading by modifying the `sampler_type` field in `data_106B.json` and adding the `streaming` field.

Note: After streaming loading is enabled, only `num_workers=0` is supported, meaning data is processed by a single process. This may cause performance fluctuations, and the resumable training feature will not be supported.

```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            ...
            "streaming": true
            ...
        },
        ...
    },
    "dataloader_param": {
        ...
        "sampler_type": "stateful_distributed_sampler",
        ...
    }
}

```

Module freezing configuration:

Currently, freezing of the `vision encoder`, `vision projector`, `text decoder`, and `lm head` modules is supported. Among these, `vision encoder` and `vision projector` are in a frozen state by default during training.

Modify whether each module is frozen by configuring the `freeze` field of each module in the `model.json` file.

Model saving, loading, and logging configuration:

Configure the parameters of `examples/glm4.5v/finetune_glm4_5v_106B.sh` according to the actual situation, including loading and saving paths, and `--save-interval` (Note: It takes a long time for distributed optimizer to save large files, so please set the save interval carefully).

```shell
...
# Weight load path for resumable training
LOAD_PATH="./ckpt/save_dir/GLM-4.5V"
# Save path
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # Do not load the optimizer state; remove this if loading is required
    --no-load-rng \  # Do not load the random number state; remove this if loading is required
    --no-save-optim \  # Do not save optimizer state; remove this if saving is required.
    --no-save-rng \  # Do not save random number state; remove this if saving is required.
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # Logging interval
    --save-interval 5000 \  # Save interval
    --save $SAVE_PATH \ # Save path
"
```

Configure the `init_from_hf_path` parameter in `examples/glm4.5v/model_106B.json` as required. This parameter indicates the load path for the initial weights.
Configure the `image_encoder.vision_encoder.freeze`, `image_encoder.vision_projector.freeze`, and `text_decoder.freeze` parameters in `examples/glm4.5v/model_106B.json` as required. These parameters represent whether to freeze the `vision model`, `vision_projector`, and `text_decoder` modules, respectively.

The current network layer counts in `examples/glm4.5v/model_106B.json` are invalid configurations that have not been verified. If you need to reduce layers, please modify the relevant configuration files in the original hf path.
To run model training faster on the NPU, the MoE module can enable NPU-friendly fusion operators. That is, configure `"use_npu_fused_moe"` to `true` in `examples/glm4.5v/model_106B.json`. By default, this parameter is set to `true`.

Single-node execution configuration:
A single node can only run with reduced layers, which can be used for model debugging. For running a complete model, please use the multi-node configuration.
Configure the parameters in `examples/glm4.5v/model_106B.sh` as follows:

```shell
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
NPUS_PER_NODE=16  # Can run with reduced layers on a single node
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

Multi-node execution configuration:

Configure the parameters of `examples/glm4.5v/model_106B.sh` as follows:

```shell
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Configure distributed parameters according to the actual situation of the distributed cluster.
NPUS_PER_NODE=16  # Number of cards per node. Set the number as required.
MASTER_ADDR="your master node IP"  # All need to be modified to the IP address of the master node (cannot be localhost)
MASTER_PORT=6000
NNODES=8  # Number of nodes in the cluster. Set the number as required.
NODE_RANK="current node id"  # Rank of the current node; cannot be duplicated across multiple nodes. The rank of the master node is 0, and ranks of other nodes can be 1, 2, etc.
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Starting a fine-tuning training task requires confirming the loss calculation method.
Differences in loss calculation methods can have varying impacts on training effectiveness. Before starting the training task, please review the documentation on loss calculation and select an appropriate loss calculation method. For more details, see [vlm_model_loss_calculate_type.md](../../docs/en/features/vlm_model_loss_calculate_type.md).

```shell
bash examples/glm4.5v/model_106B.sh
```

<a id="jump4.4"></a>

### 4. hf Weights Conversion

After training is complete, convert the weights saved in the `save_dir` directory to Hugging Face format.

```shell
mm-convert ExpertMergeDcpConverter dcp_to_hf --hf_dir "ckpt/hf_path/GLM-4.5V" --dcp_dir "save_dir/iter_000xx" --save_dir "ckpt/dcp_to_hf/GLM-4.5V"
```

`--hf_dir` indicates the path to the original Hugging Face weights, `--dcp_dir` indicates the save path for the fine-tuned weights, `iter_000xx` indicates the weights saved at step xx, and `--save_dir` indicates the save path for the converted weights.

After completing the weights conversion, you can use the relevant libraries for inference.

<a id="jump10"></a>

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

<a id="jump11"></a>

## Notes
