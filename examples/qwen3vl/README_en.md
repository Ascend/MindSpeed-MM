# Qwen3_VL Usage Guide

<p align="left">
</p>

## Contents

- [Qwen3\_VL Usage Guide](#qwen3_vl-usage-guide)
  - [Contents](#contents)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Installation](#environment-installation)
    - [1. Environment Preparation](#1-environment-preparation)
    - [2. Environment Setup](#2-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download (Taking the COCO2017 Dataset as an Example)](#1-dataset-download-taking-the-coco2017-dataset-as-an-example)
    - [2. Mixed Dataset Processing (Using LLaVA-Instruct-150K as an Example)](#2-mixed-dataset-processing-using-llava-instruct-150k-as-an-example)
  - [Fine-tuning](#fine-tuning)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
    - [4. Start Inference](#4-start-inference)
  - [PMCC (Privacy and Model Confidential Computing)](#pmcc-privacy-and-model-confidential-computing)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Description

### Reference Implementation

```shell
url=https://github.com/huggingface/transformers.git
commit_id=c0dbe09
```

### Changelog

2025.09.28: Initial support for Qwen3-VL model

<a id="jump1"></a>

## Environment Installation

<a id="jump1.1"></a>

### 1. Environment Preparation

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md) to complete the Ascend software installation.

> Python 3.10 is recommended, and torch and torch_npu 2.7.1 are recommended.

‼️The acceleration features of the MoE part depend on newer versions of torch_npu and CANN. The following versions are recommended:

- [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)
- [torch_npu](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/insg_0004.html)

<a id="jump1.2"></a>

### 2. Environment Setup

⚠️ If you have previously used other models in MindSpeed MM, it is strongly recommended to switch to a new working directory and create a new Conda environment to avoid potential risks caused by version inconsistencies of some third-party libraries.

Clone the MindSpeed MM repository and enter its root directory:

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
bash scripts/install.sh --megatron --msid 96bc0a3bf3398bf45ac26e0bded95ee174ac449b && pip install -r examples/qwen3vl/requirements.txt
```

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the corresponding model weights from the Hugging Face library:

- [Qwen3-VL-*B](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe)

 Save the downloaded model weights to the local `ckpt/hf_path/Qwen3-VL-*B-Instruct` directory. (`*` indicates the corresponding size.)

If using FSDP2's `meta init` to initialize the model, you need to complete the following weight conversion first.

```bash
mm-convert Qwen3VLConverter hf_to_dcp \
  --hf_dir Qwen3-VL-xxB \
  --dcp_dir Qwen3-VL-xxB-dcp

# The converted directory structure is:
# ———— Qwen3-VL-xxB-dcp
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

And in `examples/qwen3vl/qwen3vl_full_sft_xxB.yaml`, set `init_model_with_meta_device` to `true` under `gpt_args`, and also modify `MM_MODEL_LOAD_PATH` in that yaml to the converted dcp weight path (write to the parent directory of the `release` folder, such as `Qwen3-VL-xxB-dcp`).

Note: For Qwen3VL-30B and Qwen3VL-235B models, you must use `meta init` to load weights. The repository has `init_model_with_meta_device` enabled by default.

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download (Taking the COCO2017 Dataset as an Example)

(1) Download the [COCO2017 dataset](https://cocodataset.org/#download) and extract it to the `./data/COCO2017` folder under the project directory.

(2) Obtain the image dataset description file ([LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)) and download it to the `./data/` path.

(3) Run the data conversion script: `examples/qwen2vl/llava_instruct_2_mllm_demo_format.py`. The reference data directory structure after conversion is as follows:

   ```shell
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

Currently, it supports reading multiple datasets separated by `,` (do not add spaces). To do so, modify `DATASET_PATH` in `qwen3vl_full_sft_xxB.yaml`: Change `./data/mllm_format_llava_instruct_data.json` to `./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json`.

Also note the configuration of `dataset_param->basic_parameters->max_samples` in `qwen3vl_full_sft_xxB.yaml` limits the data reading to only `max_samples` entries, allowing for quick function verification. For formal training, you can remove this parameter to read all the data.

<a id="jump3.2"></a>

### 2. Mixed Dataset Processing (Using LLaVA-Instruct-150K as an Example)

This framework now supports pure text/mixed data (mixed training with image and non-image data).

When constructing data, for data containing images, retain the `image` key-value pair.

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

When constructing data, for pure text data, remove the `image` key-value pair.

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

### 1. Prerequisites

Before configuring the script, you need to complete the preliminary preparations, including: **Environment Installation**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Model Category Configuration]

Currently, the nothink model is fine-tuned by default. If you want to fine-tune the Qwen3-VL-thinking model, please set the `template` in `qwen3vl_full_sft_xxB.yaml` to `qwen3_vl`, and set `enable_thinking` to `true`.

[Data Directory Configuration]

Modify the dataset paths in `qwen3vl_full_sft_xxB.yaml` according to the actual situation, including the `model_name_or_path`, `dataset_dir`, `dataset` and other fields.

Example: If the data and its corresponding json are both in the `/home/user/data/` directory, and the json directory is `/home/user/data/video_data_path.json`, set `dataset_dir` to `/home/user/data/`
and `dataset` to `./data/video_data_path.json.`
Note that `dataset` needs to be configured as a relative path here.

Taking Qwen3VL-xxB as an example, make the following modifications to `qwen3vl_full_sft_xxB.yaml`. Note that the weight path for `model_name_or_path` is the weight path before conversion, i.e., the original HF weight path.

**Note: Do not configure `cache_dir` to the same mounted directory on multiple machines to avoid conflicts caused by writing to the same file.**

```yaml
HF_MODEL_LOAD_PATH: &HF_MODEL_LOAD_PATH ./ckpt/hf_path/Qwen3-VL-8B-Instruct
DATASET_PATH: &DATASET_PATH ./data/mllm_format_llava_instruct_data.json
data:
  dataset_param:
    dataset_type: huggingface
    preprocess_parameters:
      model_name_or_path: *HF_MODEL_LOAD_PATH

    basic_parameters:
      dataset_dir: ./data
      dataset: *DATASET_PATH
      cache_dir: ./data/cache_dir
```

If you need to load a large amount of data, you can use streaming loading by modifying the `sampler_type` field in `qwen3vl_full_sft_xxB.yaml` and adding the `streaming` field. (Note: Once streaming loading is enabled, only `num_workers=0` is currently supported, meaning data is processed by a single process. This may cause performance fluctuations, and resumable training will not be supported.)

```yaml
data:
  dataset_param:
    basic_parameters:
      streaming: true
  dataloader_param:
      sampler_type: stateful_distributed_sampler
```

[Module Freezing Configuration]

Currently, freezing of the `vision encoder`, `vision projector`, `text decoder`, and `lm head` modules is supported. Among these, `vision encoder` and `vision projector` are frozen by default during training.

Modify whether each module is frozen by configuring the `freeze` field under the `model` field in the `qwen3vl_full_sft_xxB.yaml` file.

[MoE Acceleration Configuration]

Enabling MoE fusion can improve model training performance. To do so, modify the `use_npu_fused_moe` field to `true` in the `qwen3vl_full_sft_xxB.yaml` file.

Note: The FusedMoE feature depends on a newer version. For version details, see [Environment Preparation](#1-environment-preparation).

[MoE Expert Parallelism Configuration]

Enabling MoE expert parallelism can effectively reduce peak memory usage. Currently, when enabling expert parallelism, MoE fusion acceleration must be set first, i.e., modify the `use_npu_fused_moe` field to `true` in the `qwen3vl_full_sft_xxB.yaml` file.
The enabling method for Expert Parallelism is to set `expert_parallel_size > 1` in the `fsdp2_config.yaml` file. For example:

```yaml
expert_parallel_size: 16
```

Note: The number of expert parallelisms must be divisible by the number of model experts.

[Sequence Parallelism Configuration]

Ulysses is currently supported. During training with long sequences, the CP feature needs to be enabled by setting `context_parallel_size > 1` in `qwen3vl_full_sft_xxB.yaml`. For example:

```yaml
gpt_args:
  context_parallel_size: 4
```

[Attention Configuration]

- Whether to compute AttnMask is determined by the `is_causal` field in the `qwen3vl_full_sft_xxB.yaml` file.
When set to `true`, it calculates using a causal mask. When set to `false`, a complete attention mask is created. It is recommended to enable this for long sequences to save memory.

- For attn_implementation and layout configurations, different attention implementation methods can be selected for the vision and text modules. Specifically, this is done by modifying the `attn_implementation` field in the `qwen3vl_full_sft_xxB.yaml` file. The currently supported configurations are shown in the table below.

  | Module | Supported FA and layout | Supported CP Types |
  | --- | --- | --- |
  | ViT | `flash_attention_2`: `TND` | ulysses, ring, usp |
  | ViT | `flash_attention_2`: ``BNSD`` | ulysses |
  | ViT | `sdpa`: ``BNSD`` | ulysses |
  | LLM | `flash_attention_2`: `TND` | ulysses, ring, usp |
  | LLM | `flash_attention_2`: `BNSD` | ulysses, ring, usp |
  | LLM | `flash_attention_2`: `BSND` | ulysses |
  | LLM | `sdpa`: `BNSD` | ulysses |

[synchronize_per_layer Configuration]

During training with FSDP2, there may be OOM issues caused by memory not being released in time. Enabling `synchronize_per_layer` forces synchronization for each transformer layer, alleviating the issue of delayed memory release caused by multi-stream reuse and reducing some memory usage.
The enabling method is to modify the `synchronize_per_layer` field in the `qwen3vl_full_sft_xxB.yaml` file. It is currently set to `true` by default.

[activation_offload Configuration]

Using `activation_offload` moves the activations of checkpoints generated during recomputation to the host, and asynchronously transfers them from the host to the device during the backward pass, reducing device activation memory usage. The configuration method is to set the `activation_offload` field to `True` in `qwen3vl_full_sft_xxB.yaml`.

[FSDP2 offload_to_cpu Configuration]

Setting `offload_to_cpu` to `True` in the `fsdp2_config.yaml` allows offloading parameters, gradients, and optimizer states to CPU memory, further reducing memory usage. However, the training speed will be relatively slower, so it is not recommended to enable this when memory is sufficient.
For a detailed feature description, please refer to [FSDP2](../../docs/en/features/fsdp2.md).
When enabling this feature, you also need to configure `distributed_backend: npu:hccl,cpu:gloo` in the `gpt_args` configuration item within the `qwen3vl_full_sft_xxB.yaml` file to enable dual communication backends.

[chunkloss Configuration]

Refer to [chunk loss](../../docs/en/features/chunkloss.md).

[Load Balancing Loss Configuration]

Supports customizing the coefficient of the auxiliary loss (`aux_loss`) for expert load balancing in MoE models. This is set via `router_aux_loss_coef` in `qwen3vl_full_sft_xxB.yaml`, and defaults to `0.0`, meaning this loss is not calculated.

[Model Saving, Loading, and Logging Configuration]

Configure the parameters in `qwen3vl_full_sft_xxB.yaml` according to the actual situation, including the loading and saving paths, as well as the save interval `save_interval`. (Note: Distributed optimizer files are large and saving them takes a long time, so please set the save interval carefully.)

```yaml
# Loading path for converted dcp weights or checkpoint resumption weights
MM_MODEL_LOAD_PATH: &MM_MODEL_LOAD_PATH ./ckpt/save_dir/Qwen3-VL-xxB-Instruct
SAVE_PATH: &SAVE_PATH save_dir
gpt_args:
  ## training:
  no_load_optim: true  # Do not load the optimizer state; remove this if loading is required
  no_load_rng: true  # Do not load the random number state; remove this if loading is required
  no_save_optim: true  # Do not save the optimizer state; remove this if saving is required
  no_save_rng: true  # Do not save the random number state; remove this if saving is required.

  ## save_and_logging:
  log_interval: 1  # Logging Interval
  save_interval: 10000   # Save Interval
  save: *SAVE_PATH  # Save Path
```

Configure the `init_from_hf_path` parameter in `qwen3vl_full_sft_xxB.yaml` according to the actual situation. This parameter represents the loading path for the initial weights.

Configure the `image_encoder.vision_encoder.freeze`, `image_encoder.vision_projector.freeze`, and `text_decoder.freeze` parameters in `qwen3vl_full_sft_xxB.yaml` according to the actual situation. These parameters represent whether to freeze the vision model module, the projector module, and the language model module, respectively.

Note that the number of network layers in the current `qwen3vl_full_sft_xxB.yaml` are invalid configurations that have not been verified. If you need to reduce layers, please modify the relevant configuration files under the original HF path.

[Single-Node Running Configuration]

Configure the parameters in `examples/qwen3vl/finetune_qwen3vl_xxB.sh` as follows:

```shell
# Modify the ascend-toolkit path based on the actual situation.
source /usr/local/Ascend/cann/set_env.sh
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

[LoRA Fine-tuning (Optional)]

LoRA is a general capability of the framework. Currently, it supports LoRA fine-tuning for the 30B language module. For parameter introduction, please refer to the [LoRA Feature Guide](../../docs/en/features/lora_finetune.md).

In the LoRA fine-tuning scenario, you need to first complete the following weight conversion for the original weights.

```bash
mm-convert Qwen3VLConverter hf_to_dcp \
  --hf_dir Qwen3-VL-30B-A3B-Instruct \
  --dcp_dir Qwen3-VL-30B-A3B-Instruct-dcp \
  --is_lora_base true

# The converted directory structure is:
# ———— Qwen3-VL-30B-A3B-Instruct-dcp
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

If you need to load LoRA pre-trained weights, you need to first complete the following weight conversion for the LoRA weights.

```bash
mm-convert Qwen3VLConverter lora_hf_to_dcp \
  --hf_dir Qwen3-VL-30B-A3B-Instruct-lora \
  --dcp_dir Qwen3-VL-30B-A3B-Instruct-lora-dcp

# The converted directory structure is:
# ———— Qwen3-VL-30B-A3B-Instruct-lora-dcp
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

And add the LoRA pre-trained weight path in `examples/qwen3vl/qwen3vl_lora_sft_30B.yaml`. The related configuration modifications are as follows:

```shell
MM_MODEL_LOAD_PATH: &MM_MODEL_LOAD_PATH ./ckpt/mm_path/Qwen3-VL-30B-A3B-Instruct
LORA_MODEL_LOAD_PATH: &LORA_MODEL_LOAD_PATH ./ckpt/mm_path/Qwen3-VL-30B-A3B-Instruct-lora

...
# Original load: *MM_MODEL_LOAD_PATH needs to be replaced with load_base_model: *MM_MODEL_LOAD_PATH
load: *LORA_MODEL_LOAD_PATH
load_base_model: *MM_MODEL_LOAD_PATH
...
```

Run the following command to perform LoRA fine-tuning:

```shell
bash examples/qwen3vl/finetune_lora_qwen3vl_30B.sh
```

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Take Qwen3VL-xxB as an example to start the fine-tuning training task.
Differences in loss calculation methods can have varying impacts on training effectiveness. Before starting the training task, please refer to the documentation on loss calculation and select an appropriate loss calculation method. For details, see [vlm_model_loss_calculate_type.md](../../docs/en/features/vlm_model_loss_calculate_type.md).

You can switch between different loss calculation methods by modifying the `loss_type` field in the `qwen3vl_full_sft_xxB.yaml` file.

```shell
bash examples/qwen3vl/finetune_qwen3vl_xxB.sh
```

**Optimization Feature**:

- ChunkLoss: Refer to [ChunkLoss](../../docs/en/features/chunkloss.md) to enable this feature and optimize memory usage for long sequences.

<a id="jump4.4"></a>

### 4. Start Inference

After training is completed, take Qwen3VL-xxB as an example, convert the weights saved in the `save_dir` directory to the Hugging Face format.

```shell
mm-convert Qwen3VLConverter dcp_to_hf \
  --load_dir save_dir/iter_000xx/ \
  --save_dir save_dir/iter_000xx_hf/ \
  --model_assets_dir ./ckpt/Qwen3-VL-xxB-Instruct \
  --to_bf16 False \
```

`iter_000xx` represents the weights saved at step *xx*, `--save_dir` specifies the save path for the converted weights, `--model_assets_dir` is the path to the original HuggingFace weights, and `--to_bf16` indicates whether to convert the weight data type from fp32 to bf16.

After completing the weight conversion, you can refer to the following tutorial to perform inference using the transformers library.

```shell
This script is intended solely to provide a convenient inference tool for testing training effectiveness. It does not guarantee inference performance.
Tutorials:：
1. Configure MODEL_PATH, MODEL_TYPE, and DATA_JSON_PATH according to your own local paths.
2. Navigate to the MindSpeed MM directory via `cd`.
3. Source your CANN environment path.
4.  Manually specify which cards to use by exporting ASCEND_RT_VISIBLE_DEVICES. Otherwise, the script may fail to automatically detect multiple cards and encounter out-of-memory errors during execution.
5. Execute `examples/qwen3vl/inference_demo.py`.
```

[Multi-node Execution Configuration]

To initiate multi-node training, modify the `MASTER_ADDR`, `NODE_ADDR`, `NNODES`, and `NODE_RANK` variables in the startup script.

``` shell
MASTER_ADDR: Master node IP
NODE_ADDR: IP of the current node
NODE_RANK: Node rank
NNODES: Number of nodes
```

<a id="jump5"></a>

## PMCC (Privacy and Model Confidential Computing)

PMCC is a privacy computing solution provided by Ascend, designed to protect model weights and data privacy during model training. When fine-tuning the Qwen3VL-32B model, if you need to enable the PMCC feature, you must first install the PMCC component.

```python
pip install ai_asset_obfuscate
pip install opencv-python
pip install pandas==2.3.3
```

Start PMCC weight encryption and data preprocessing encryption with the following commands:

```bash
# Encrypt HF model weights.
python mindspeed_mm/tools/pmcc/pmcc_qwen3vl.py \
    --obf-type model \
    --hf-model-path "/data/ckpt/Qwen3-VL-32B-Instruct/" \
    --obf-seed "22222222222222222222222222222222" \
    --model-save-path "/data/pmcc/obf_hf_ckpt/" \
    --device-id 0 1 2 3 4 5 6 7

# Encrypt dataset.
python mindspeed_mm/tools/pmcc/pmcc_qwen3vl.py \
    --obf-type data \
    --hf-model-path "/data/ckpt/Qwen3-VL-32B-Instruct/" \
    --obf-seed "22222222222222222222222222222222" \
    --src-json-path "/data/dataset/llava_instruct_150k.json" \
    --src-img-dir "/data/dataset/COCO2017/train2017" \
    --obf-json-path "/data/pmcc/obf_json_2000.json" \
    --obf-img-dir "/data/pmcc/obf_images" \
    --data-limit 2000

# Convert encrypted HF model weights to DCP format.
mm-convert Qwen3VLConverter hf_to_dcp \
    --hf_dir /data/pmcc/obf_hf_ckpt \
    --dcp_dir /data/pmcc/obf_dcp_ckpt
```

After completing model and data encryption, and converting encrypted HF weights to DCP format, modify the `HF_MODEL_LOAD_PATH`, `MM_MODEL_LOAD_PATH`, `DATASET_PATH`, and `DATASET_DIR` in the `qwen3vl_full_sft_32B.yaml` file to the encrypted HF weight path, DCP weight path, encrypted dataset JSON path, and dataset folder path, respectively. Modify the `use_pmcc_data` parameter to `true` to enable PMCC data loading.

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
