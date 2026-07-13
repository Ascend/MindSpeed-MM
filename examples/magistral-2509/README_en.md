# Magistral-Small-2509 User Guide

<p align="left">
</p>

## Contents

- [Magistral-Small-2509 User Guide](#magistral-small-2509-user-guide)
  - [Contents](#contents)
  - [Version Note](#version-note)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Installation](#environment-installation)
    - [1. Environment Setup](#1-environment-setup)
    - [2. Environment Setup](#2-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [Weight Download](#weight-download)
    - [Weight Conversion](#weight-conversion)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download (Using COCO2017 as an Example)](#1-dataset-download-using-coco2017-as-an-example)
  - [Fine-tuning](#fine-tuning)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
    - [4. Start Inference](#4-start-inference)
  - [LoRA Fine-tuning](#lora-fine-tuning)
    - [LoRA Weight Conversion](#lora-weight-conversion)
    - [Parameter Configuration](#parameter-configuration)
    - [Start LoRA Fine-tuning](#start-lora-fine-tuning)
    - [Weight Merging After LoRA Fine-tuning](#weight-merging-after-lora-fine-tuning)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Note

### Reference Implementation

```shell
url=https://github.com/hiyouga/LLaMAFactory
commit_id=68119e5
```

### Changelog

2026.1.19: Initial support for the Magistral-Small-2509 model

<a id="jump1"></a>

## Environment Installation

<a id="jump1.1"></a>

### 1. Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md) to complete the Ascend software installation.
> Python version 3.10 is recommended, and torch and torch_npu version 2.7.1 are recommended.

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
# checkout commit from MindSpeed core_r0.12.1
git checkout d76dbddd4517d48a2fc1cd494de8b9a6cfdbfbab

# Install MindSpeed and its dependencies.
pip install -e .
cd ..
# Install MindSpeed MM and its dependencies.
pip install -e .

```

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### Weight Download

Download open-source model weights from websites such as Hugging Face.

- [Magistral-Small-2509](https://huggingface.co/mistralai/Magistral-Small-2509)

Save the model weights in the `ckpt/hf_path/` directory, for example, `ckpt/hf_path/Magistral-Small-2509`.

<a id="jump2.2"></a>

### Weight Conversion

```shell
mm-convert Mistral3Converter hf_to_dcp --hf_dir "ckpt/hf_path/Magistral-Small-2509" --dcp_dir "ckpt/convert_path/Magistral-Small-2509"
```

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download (Using COCO2017 as an Example)

(1) Users need to download the [COCO2017 dataset](https://cocodataset.org/#download) and extract it to the `./data/COCO2017` folder within the project directory.

(2) Obtain the description file of the image dataset ([LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)) and download it to the `./data/` path.

(3) Run the data conversion script `python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py`. The reference data directory structure after conversion is as follows:

   ```shell
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

## Fine-tuning

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the prerequisite preparations, including: **Environment Setup**, **Weight Download**, and **Dataset Preparation and Processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset path in `data.json` according to the actual situation, including the `model_name_or_path`, `dataset_dir`, `dataset`, and other fields.
`model_name_or_path` is the path to the original model.

[Model Path Configuration]

Modify the weight path in `model.json` according to the actual situation, including `init_from_hf_path`, which is the path to the original model.

[Model Saving, Loading, and Logging Configuration]

Configure the parameters in `examples/magistral-2509/finetune_magistral_2509.sh` according to the actual situation, including the load path, save path, and save interval `--save-interval` (Note: Saving distributed optimizer files takes a long time, so please set the save interval carefully).

Taking Magistral-Small-2509 as an example:

```shell
...

# Load path: path after weight conversion.
LOAD_PATH="ckpt/convert_path/Magistral-Small-2509"
# Save path
SAVE_PATH="Magistral-Small-2509_finetune_result"
...
GPT_ARGS="
    ...
    --no-load-optim \  # Do not load optimizer state; remove it if loading is required.
    --no-load-rng \  # Do not load random number state; remove it if loading is required.
    --no-save-optim \  # Do not save optimizer state; remove it if saving is required.
    --no-save-rng \  # Do not save random number state; remove it if saving is required.
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # Logging Interval
    --save-interval 5000 \  # Save Interval
    ...
    --log-tps \  # Adding this parameter enables printing the average sequence length of the language module at each step during training, and calculating the throughput in tokens per second after training ends.
"
```

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Single-node Running Configuration]

Configure the parameters in `examples/magistral-2509/finetune_magistral_2509.sh` as follows.

```shell
  # Modify the ascend-toolkit path according to the actual situation
  source /usr/local/Ascend/cann/set_env.sh
  NPUS_PER_NODE=8 # Can run on a single A2 node
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

[Multi-node Running Configuration]

Configure the parameters in `examples/magistral-2509/finetune_magistral_2509.sh` as follows.

```shell
  # Modify the ascend-toolkit path according to the actual situation.
  source /usr/local/Ascend/cann/set_env.sh
  # Configure distributed parameters based on the actual situation of the distributed cluster.
  GPUS_PER_NODE=8  # Number of cards per node. Fill in according to the actual situation.
  MASTER_ADDR="your master node IP"  # All need to be modified to the IP address of the master node (cannot be localhost).
  MASTER_PORT=6000
  NNODES=2  # Number of nodes in the cluster. Fill in according to the actual situation.
  NODE_RANK="current node id"  # RANK of the current node. Cannot be duplicated across multiple nodes. The rank of the master node is 0, and ranks of other nodes can be 1, 2, etc.
  WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

(Optional) [Data Packing]
To enable Data Packing, set `packing` to `true` in `data.json`, and adjust `SEQ_LEN` in `finetune_magistral_2509.sh` as needed.

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Take Magistral-Small-2509 as an example to start the fine-tuning training task.

```shell
bash examples/magistral-2509/finetune_magistral_2509.sh
```

<a id="jump4.4"></a>

### 4. Start Inference

After training is complete, convert the weights saved in the `SAVE_PATH` directory to the Hugging Face format.

```shell
mm-convert Mistral3Converter dcp_to_hf  \
  --load_dir "Magistral-Small-2509_finetune_result/iter_000xx" \
  --save_dir "ckpt/dcp_to_hf/Magistral-Small-2509" \
  --model_assets_dir "ckpt/hf_path/Magistral-Small-2509"
```

Where, `iter_000xx` represents the weights saved at step *xx*, `--save_dir` specifies the save path for the converted weights, and `--model_assets_dir` is the path to the original Hugging Face weights.

After completing the weight conversion, you can use the transformers library for inference.

<a id="jump5"></a>

## LoRA Fine-tuning

Similar to model fine-tuning, weight conversion and script configuration are required.

### LoRA Weight Conversion

```shell
mm-convert Mistral3Converter hf_to_dcp --hf_dir "ckpt/hf_path/Magistral-Small-2509" --dcp_dir "ckpt/convert_path/Magistral-Small-2509-lora-base" --is_lora_base true
```

### Parameter Configuration

LoRA fine-tuning uses the `finetune_magistral_2509_lora.sh` script. The configurations for data, model paths, etc., are the same as those set for fine-tuning.

For the first LoRA fine-tuning, set `LOAD_PATH` in `finetune_magistral_2509_lora.sh` to the converted weight path, such as ckpt/`convert_path/Magistral-Small-2509-lora-base`. If you want to continue LoRA training, add the `--load-base-model` parameter to `GPT_ARGS in finetune_magistral_2509_lora.sh`. And configure `--load-base-model` to the weight obtained from the "LoRA weight conversion" section, and configure `LOAD_PATH` to the weights saved from the previous LoRA training.

### Start LoRA Fine-tuning

```shell
bash examples/magistral-2509/finetune_magistral_2509_lora.sh
```

### Weight Merging After LoRA Fine-tuning

```shell
mm-convert Mistral3Converter merge_mm_lora_dcp_weight_to_base_hf \
  --base_hf_dir ckpt/hf_path/Magistral-Small-2509 \ # Original Weight Path
  --lora_dcp_dir /path/to/Magistral-Small-2509_lora_finetune_result \ # LoRA Weight Save Path
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \ # LoRA Computation Module
  --save_merged_hf_dir /path/to/save  # Save Path to Merged Weights
```

<a id="jump6"></a>

## Environment Variable Declaration

The following lists common environment variables. For details, see [Detailed Variable Declaration](../../docs/en/pytorch/args_readme.md).

| Environment Variable          | Description                                                                 | Value Description                                                                                         |
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
