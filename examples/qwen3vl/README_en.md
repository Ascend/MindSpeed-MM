# Qwen3-VL User Guide

<p align="left">
</p>

## Contents

- [Version Description](#version-description)
  - [Reference Implementation](#reference-implementation)
  - [Change Log](#change-log)
- [Environment Installation](#environment-installation)
  - [Environment Preparation](#1-environment-preparation)
  - [Environment Setup](#2-environment-setup)
- [Weight Download and Conversion](#weight-download-and-conversion)
  - [Weight Download](#1-weight-download)
- [Dataset Preparation and Processing](#dataset-preparation-and-processing)
- [Fine-Tuning](#fine-tuning)
  - [Preparation](#1-preparation)
  - [Parameter Configuration](#2-parameter-configuration)
  - [Start Fine-Tuning](#3-start-fine-tuning)
  - [Start Inference](#4-start-inference)
- [Privacy and Model Confidential-Computing](#privacy-and-model-confidential-computing)
- [Environment Variable Declaration](#environment-variable-declaration)

## Version Description

### Reference Implementation

```shell
url=https://github.com/huggingface/transformers.git
commit_id=c0dbe09
```

### Change Log

2025.09.28: First support for the Qwen3-VL model

---
<a id="jump1"></a>

## Environment Installation

<a id="jump1.1"></a>

### 1. Environment Preparation

[The accompanying environment version is recommended for model development.]

For Ascend software installation, see the [Installation Guide](../../docs/en/guides/installation/install_guide.md).

<a id="jump1.2"></a>

### 2. Environment Setup

⚠️If you have used other models of MindSpeed-MM before, you are strongly advised to switch to a new working directory and create a new Conda environment, to avoid the risks caused by inconsistent versions of some third-party libraries.

Clone the MindSpeed MM repository and go to the root directory of the repository:

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
bash scripts/install.sh --megatron --msid 96bc0a3bf3398bf45ac26e0bded95ee174ac449b && pip install -r examples/qwen3vl/requirements.txt
```

---

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the corresponding model weights from the Hugging Face (HF) library:

> [!NOTE]
>
> If you cannot access the HuggingFace community download resources properly, it is recommended to download from ModelScope. Pay attention to the correctness and security of the files to be downloaded.

- Model address: [Qwen3-VL-*B](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe).

 Save the downloaded model weights to the local `ckpt/hf_path/Qwen3-VL-*B-Instruct` directory. (   indicates the corresponding size.)

If you use the meta init of FSDP2 to initialize the model, complete the following weight conversion first:

```bash
mm-convert Qwen3VLConverter hf_to_dcp \
  --hf_dir Qwen3-VL-xxB \
  --dcp_dir Qwen3-VL-xxB-dcp

# The converted directory structure is as follows:
# ———— Qwen3-VL-xxB-dcp
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

In addition, set `init_model_with_meta_device` to `true` in `gpt_args` of `examples/qwen3vl/qwen3vl_full_sft_xxB.yaml`, and change `MM_MODEL_LOAD_PATH` in the yaml to the converted dcp weight path (pointing to the directory one level above the `release` folder, such as `Qwen3-VL-xxB-dcp`).

Note that for the Qwen3VL-30B and Qwen3VL-235B models, the weights must be loaded by means of meta init, and `init_model_with_meta_device` is enabled by default in the repository.

---
<a id="jump3"></a>

## Dataset Preparation and Processing

- Training with a **real dataset**: see [Building Data for VL Models–Using a Real Dataset](../../docs/en/features/data/building_data_for_VLModel.md#real-data) (download COCO2017 → download the LLaVA-Instruct-150K annotations → run the conversion script to generate `mllm_format_llava_instruct_data.json`).
- Functional/performance testing with **mock data**: see [Building Data for VL Models–Using Mock Data](../../docs/en/features/data/building_data_for_VLModel.md#mock-data).

## Fine-Tuning

<a id="jump4.1"></a>

### 1. Preparation

Before configuring the scripts, complete the preparatory work, including **environment installation**, **weight download and conversion**, and **dataset preparation and processing**. For details, see the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

**Model Category Configuration**
The nothink model is fine-tuned by default. To fine-tune the qwen3-VL-thinking model, set `template` in the configuration file `qwen3vl_full_sft_xxB.yaml` to `qwen3_vl`,
and set `enable_thinking` to `true`.

**Data Directory Configuration**

Modify the dataset paths in `qwen3vl_full_sft_xxB.yaml` according to the actual situation, including fields such as `model_name_or_path`, `dataset_dir`, and `dataset`.

Example: if the data and its corresponding json are both in the `/home/user/data/` directory, and the json file is `/home/user/data/video_data_path.json`, the configuration is as follows:
`dataset_dir` is set to `/home/user/data/`, and 
`dataset` is set to `./data/video_data_path.json`.
Note that `dataset` must be set to a relative path here.

> **Note**: At runtime, `dataset_dir` is concatenated with the relative path of the media file in the annotation JSON (such as `./train2017/000000033471.jpg`). Therefore, `dataset_dir` must be set to **the parent directory of the directory where the media files are actually located** (for example, if the images are under `train2017/`, `dataset_dir` should point to the parent directory of `train2017/`), otherwise a `FileNotFoundError` is reported because the media file cannot be found. For details, see [VL Model Data Preparation](../../docs/en/features/data/building_data_for_VLModel.md).

Taking Qwen3VL-xxB as an example, make the following modifications to `qwen3vl_full_sft_xxB.yaml`. Note that the weight path of `model_name_or_path` is the path before conversion, that is, the original HF weight path.

**Note that `cache_dir` must not be configured as the same mounted directory on multiple nodes, to avoid conflicts caused by writing to the same file**.

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

If you need to load a large amount of data, you can use streaming loading. Modify the `sampler_type` field in `qwen3vl_full_sft_xxB.yaml` and add the `streaming` field. (Note: with streaming loading, currently only `num_workers=0` is supported, data is processed by a single process, performance may fluctuate, and resumable training is not supported.)

```yaml
data:
  dataset_param:
    basic_parameters:
      streaming: true
  dataloader_param:
      sampler_type: stateful_distributed_sampler
```

**Module Freezing Configuration**

Freezing is currently supported for the vision encoder, vision projector, text decoder, and lm head modules. The vision encoder and vision projector modules are frozen by default during training;

you can change whether each module is frozen by configuring the `freeze` field of each module under the `model` field in `qwen3vl_full_sft_xxB.yaml`.

**MoE Acceleration Configuration**

Enabling MoE fusion can improve model training performance. To enable it, change the `use_npu_fused_moe` field in `qwen3vl_full_sft_xxB.yaml` to `true`

Note: the FusedMoE feature depends on a newer version. For the download link and installation method of the new version, see the [Environment Preparation](#1-environment-preparation) section.

**MoE Expert Parallelism Configuration**

Enabling MOE expert parallelism can effectively reduce peak memory. To enable expert parallelism, enable MOE fusion acceleration first, that is, change the `use_npu_fused_moe` field in `qwen3vl_full_sft_xxB.yaml` to `true`.
To enable expert parallelism, set `expert_parallel_size` > 1 in the `fsdp2_config.yaml` file, for example:

```yaml
expert_parallel_size: 16
```

Note: the degree of expert parallelism must be divisible by the number of experts in the model.

**Sequence Parallelism Configuration**

Ulysses sequence parallelism is currently supported. For long sequence training, the CP feature needs to be enabled by setting `context_parallel_size` > 1 in `qwen3vl_full_sft_xxB.yaml`, for example:

```yaml
gpt_args:
  context_parallel_size: 4
```

**Attention Configuration**

- Whether to compute `AttnMask`:
  The configuration method is to modify the `is_causal` field in the `qwen3vl_full_sft_xxB.yaml` file.
  As for `causal_mask`, when set to `true`, the computation follows the causal mask; when set to `false`, a complete attention mask is created. For long sequences, it is recommended to enable it to save memory.

- `attn_implementation` and layout configuration: 
  Different attention implementations can be selected for the vision and text modules. Specifically, modify the `attn_implementation` field in `qwen3vl_full_sft_xxB.yaml`. The currently supported combinations are listed in the following table.

  | Module| Supported FA and layout | Supported CP types |
  | --- | --- | --- |
  | ViT | `flash_attention_2`: `TND` | ulysses, ring, usp |
  | ViT | `flash_attention_2`: ``BNSD`` | ulysses |
  | ViT | `sdpa`: ``BNSD`` | ulysses |
  | LLM | `flash_attention_2`: `TND` | ulysses, ring, usp |
  | LLM | `flash_attention_2`: `BNSD` | ulysses, ring, usp |
  | LLM | `flash_attention_2`: `BSND` | ulysses |
  | LLM | `sdpa`: `BNSD` | ulysses |

**synchronize_per_layer Configuration**
When training with FSDP2, memory may not be released in time and OOM may occur. You can enable `synchronize_per_layer` to force each transformer layer to synchronize, which mitigates the delayed memory release caused by multi-stream reuse and reduces memory usage to some extent.
To enable it, modify the `synchronize_per_layer` field in the `qwen3vl_full_sft_xxB.yaml` file. It is set to `true` by default.

**activation_offload Configuration**
activation_offload can move the activations of the checkpoints generated during recomputation to the host, and asynchronously transfer them from the host back to the device during the backward pass, reducing the device memory occupied by activations. To configure it, set the `activation_offload` field to `True` in `qwen3vl_full_sft_xxB.yaml`.

**FSDP2 offload_to_cpu Configuration**
Setting `offload_to_cpu` to `True` in `fsdp2_config.yaml` can offload parameters, gradients, and optimizer states to CPU memory, further reducing memory usage. However, training becomes relatively slower at the same time, so it is not recommended to enable it when memory is sufficient.
For the feature description, see [FSDP2](../../docs/en/features/parallel/fsdp2.md).
When this function is enabled, you also need to configure `distributed_backend: npu:hccl,cpu:gloo` in the `gpt_args` configuration item of `qwen3vl_full_sft_xxB.yaml` to enable the dual communication backend.

**ChunkLoss Configuration**
See the [ChunkLoss documentation](../../docs/en/features/optimization/chunkloss.md).

**Load Balancing Loss Configuration**
The coefficient of `aux_loss` for expert load balancing in the moe model can be customized through `router_aux_loss_coef` in `qwen3vl_full_sft_xxB.yaml`. The default value is `0.0`, which means the loss is not computed.

**Model Saving/Loading and Log Configuration**

Configure the parameters of `qwen3vl_full_sft_xxB.yaml` according to the actual situation, including the loading and saving paths and the save interval `save_interval` (note: the files saved by the distributed optimizer are large and take a long time, so set the save interval carefully).

```yaml
# Path for loading the converted dcp weights or the resumable training weights
MM_MODEL_LOAD_PATH: &MM_MODEL_LOAD_PATH ./ckpt/save_dir/Qwen3-VL-xxB-Instruct
SAVE_PATH: &SAVE_PATH save_dir
gpt_args:
  ## training:
  no_load_optim: true  # Do not load the optimizer state; remove this line if you need to load it
  no_load_rng: true  # Do not load the random number state; remove this line if you need to load it
  no_save_optim: true  # Do not save the optimizer state; remove this line if you need to save it
  no_save_rng: true  # Do not save the random number state; remove this line if you need to save it

  ## save_and_logging:
  log_interval: 1  # Log interval
  save_interval: 10000   # Save interval
  save: *SAVE_PATH  # Save path
```

Configure the `init_from_hf_path` parameter in `qwen3vl_full_sft_xxB.yaml` according to the actual situation. This parameter indicates the loading path of the initial weights.
Configure the `image_encoder.vision_encoder.freeze`, `image_encoder.vision_projector.freeze`, and `text_decoder.freeze` parameters in `qwen3vl_full_sft_xxB.yaml` according to the actual situation. These parameters indicate whether to freeze the vision model module, the projector module, and the language model module, respectively.
Note: the number of network layers in the current `qwen3vl_full_sft_xxB.yaml` is an invalid configuration that has not passed verification. If you need to reduce the number of layers, modify the related configuration files in the original HF path.

**Single-Node Running Configuration**

Configure the parameters of `examples/qwen3vl/finetune_qwen3vl_xxB.sh` as follows:

```shell
# Modify the ascend-toolkit path according to the actual situation
source /usr/local/Ascend/cann/set_env.sh
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

**LoRA Fine-Tuning (Optional)**

LoRA is a general capability of the framework. LoRA fine-tuning of the language module of the 30B model is currently supported. For parameter descriptions, see the [LoRA Feature Documentation](../../docs/en/features/training_mode/lora_finetune.md).

For LoRA fine-tuning, complete the following weight conversion on the original weights first:

```bash
mm-convert Qwen3VLConverter hf_to_dcp \
  --hf_dir Qwen3-VL-30B-A3B-Instruct \
  --dcp_dir Qwen3-VL-30B-A3B-Instruct-dcp \
  --is_lora_base true

# The converted directory structure is as follows:
# ———— Qwen3-VL-30B-A3B-Instruct-dcp
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

If you need to load LoRA pre-trained weights, complete the following weight conversion on the LoRA weights first:

```bash
mm-convert Qwen3VLConverter lora_hf_to_dcp \
  --hf_dir Qwen3-VL-30B-A3B-Instruct-lora \
  --dcp_dir Qwen3-VL-30B-A3B-Instruct-lora-dcp

# The converted directory structure is as follows:
# ———— Qwen3-VL-30B-A3B-Instruct-lora-dcp
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

In addition, add the LoRA pre-trained weight path in `examples/qwen3vl/qwen3vl_lora_sft_30B.yaml`. The related configuration modifications are as follows:

```shell
MM_MODEL_LOAD_PATH: &MM_MODEL_LOAD_PATH ./ckpt/mm_path/Qwen3-VL-30B-A3B-Instruct
LORA_MODEL_LOAD_PATH: &LORA_MODEL_LOAD_PATH ./ckpt/mm_path/Qwen3-VL-30B-A3B-Instruct-lora

...
# The original load: *MM_MODEL_LOAD_PATH needs to be replaced with load_base_model: *MM_MODEL_LOAD_PATH
load: *LORA_MODEL_LOAD_PATH
load_base_model: *MM_MODEL_LOAD_PATH
...
```

Run the following command to perform LoRA fine-tuning:

```shell
bash examples/qwen3vl/finetune_lora_qwen3vl_30B.sh
```

<a id="jump4.3"></a>

### 3. Start Fine-Tuning

Taking Qwen3VL-xxB as an example, start the fine-tuning training task as follows.
Differences in the loss computation method have different impacts on the training effect. Before starting the training task, see the documentation on loss computation and choose an appropriate loss computation method (see  [vlm_model_loss_calculate_type.md](../../docs/en/features/optimization/vlm_model_loss_calculate_type.md)).
You can switch between different loss computation methods by modifying the `loss_type` field in `qwen3vl_full_sft_xxB.yaml`.

```shell
bash examples/qwen3vl/finetune_qwen3vl_xxB.sh
```

**Optimization Features:**

- ChunkLoss: see [ChunkLoss](../../docs/en/features/optimization/chunkloss.md) to enable this feature and reduce the memory usage of long sequences.

---

<a id="jump4.4"></a>

### 4. Start Inference

After training is complete, taking Qwen3VL-xxB as an example, convert the weights saved in the `save_dir` directory to the HF format:

```shell
mm-convert Qwen3VLConverter dcp_to_hf \
  --load_dir save_dir/iter_000xx/ \
  --save_dir save_dir/iter_000xx_hf/ \
  --model_assets_dir ./ckpt/Qwen3-VL-xxB-Instruct \
  --to_bf16 False \
```

Here, `iter_000xx` indicates the weight saved at step `xx`, `--save_dir` indicates the save path of the converted weights, `--model_assets_dir` indicates the path of the original HF weights, and `--to_bf16` indicates whether the weight data type is converted from FP32 to BF16.

After the weight conversion is complete, you can use the transformers library for inference by referring to the following tutorial.

```shell
This script only provides a convenient inference tool for testing the training effect and does not guarantee inference performance
Tutorial:
Step 1: Configure MODEL_PATH, MODEL_TYPE, and DATA_JSON_PATH according to your own paths
Step 2: cd to the MindSpeed-MM path
Step 3: source the user's cann path
Step 4: You must use export ASCEND_RT_VISIBLE_DEVICES to manually specify which devices to use; otherwise, execution may encounter OOM because multiple devices cannot be automatically identified
Step 5: Run python examples/qwen3vl/inference_demo.py
```

**Multi-Node Running Configuration**

To start multi-node training, modify the `MASTER_ADDR`, `NODE_ADDR`, `NNODES`, and `NODE_RANK` variables in the startup script:

``` shell
MASTER_ADDR: IP address of the master node
NODE_ADDR: IP address of the local node
NODE_RANK: Index of the node
NNODES: Total number of nodes
```

---

<a id="jump5"></a>

## Privacy and Model Confidential Computing

Privacy and Model Confidential Computing (PMCC) is a privacy computing solution provided by Ascend, used to protect model weights and data privacy during model training. To enable the PMCC function when fine-tuning the Qwen3VL-32B model, first install the PMCC component in the Ascend AI software stack.

```python
pip install ai_asset_obfuscate
pip install opencv-python
pip install pandas==2.3.3
```

Start PMCC weight encryption and data preprocessing encryption with the following commands:

```bash
# Encrypt the hf model weights
python mindspeed_mm/tools/pmcc/pmcc_qwen3vl.py \
    --obf-type model \
    --hf-model-path "/data/ckpt/Qwen3-VL-32B-Instruct/" \
    --obf-seed "22222222222222222222222222222222" \
    --model-save-path "/data/pmcc/obf_hf_ckpt/" \
    --device-id 0 1 2 3 4 5 6 7

# Encrypt the dataset
python mindspeed_mm/tools/pmcc/pmcc_qwen3vl.py \
    --obf-type data \
    --hf-model-path "/data/ckpt/Qwen3-VL-32B-Instruct/" \
    --obf-seed "22222222222222222222222222222222" \
    --src-json-path "/data/dataset/llava_instruct_150k.json" \
    --src-img-dir "/data/dataset/COCO2017/train2017" \
    --obf-json-path "/data/pmcc/obf_json_2000.json" \
    --obf-img-dir "/data/pmcc/obf_images" \
    --data-limit 2000

# Convert the encrypted hf model weights to the dcp format
mm-convert Qwen3VLConverter hf_to_dcp \
    --hf_dir /data/pmcc/obf_hf_ckpt \
    --dcp_dir /data/pmcc/obf_dcp_ckpt
```

After the model and data encryption is complete and the encrypted HF weights are converted to the DCP format, set `HF_MODEL_LOAD_PATH`, `MM_MODEL_LOAD_PATH`, `DATASET_PATH`, and `DATASET_DIR` in `qwen3vl_full_sft_32B.yaml` to the encrypted HF weight path, the DCP weight path, the encrypted dataset json path, and the dataset folder path, respectively, and set the `use_pmcc_data` parameter to true to enable PMCC data loading.

---

<a id="jump10"></a>

## Environment Variable Declaration

| Environment Variable | Description | Value Description |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | Whether to enable log printing | `0`: Disable log printing to the screen<br>`1`: Enable log printing to the screen |
| `ASCEND_GLOBAL_LOG_LEVEL` | Sets the log level of application logs and the log level of each module. Only debug logs are supported | `0`: DEBUG level<br>`1`: INFO level<br>`2`: WARNING level<br>`3`: ERROR level<br>`4`: NULL level, no logs are output |
| `TASK_QUEUE_ENABLE` | Controls the level of the task_queue operator delivery queue optimization | `0`: Disabled<br>`1`: Enable Level 1 optimization<br>`2`: Enable Level 2 optimization |
| `COMBINED_ENABLE` | Sets the combined flag. Setting it to 0 disables this function; setting it to 1 enables it, which is used to optimize scenarios of two non-consecutive combined operators | `0`: Disabled<br>`1`: Enabled |
| `CPU_AFFINITY_CONF` | Controls the processor affinity of CPU-side operator tasks, that is, binds tasks to cores | `0` or not set: Core binding is not enabled<br>`1`: Enable coarse-grained core binding<br>`2`: Enable fine-grained core binding |
| `HCCL_CONNECT_TIMEOUT` | Limits the timeout waiting time of the socket connection establishment process between different devices | Must be configured as an integer in the range `[120,7200]`, the default value is `120`, in `s` |
| `PYTORCH_NPU_ALLOC_CONF` | Controls the behavior of the caching allocator | `expandable_segments:<value>`: Enables the expandable segments function of the memory pool, that is, the virtual memory feature |
| `HCCL_EXEC_TIMEOUT` | Controls the synchronous waiting time during execution between devices. Within this configuration time, each device process waits for other devices to complete communication synchronization | Must be configured as an integer in the range `[68,17340]`, the default value is `1800`, in `s` |
| `ACLNN_CACHE_LIMIT` | Configures the number of operator information entries cached on the Host side by the single-operator execution API | Must be configured as an integer in the range `[1, 10,000,000]`, the default value is `10000` |
| `TOKENIZERS_PARALLELISM` | Controls the behavior of the tokenizer in the HF transformers library in a multi-threaded environment | `False`: Disable parallel tokenization<br>`True`: Enable parallel tokenization |
| `MULTI_STREAM_MEMORY_REUSE` | Configures whether multi-stream memory reuse is enabled | `0`: Disable multi-stream memory reuse<br>`1`: Enable multi-stream memory reuse |
| `NPU_ASD_ENABLE` | Controls whether to enable the feature value detection function of TorchNPU | `0` or not set: Feature value detection is disabled<br>`1`: Enable feature value detection, only exception logs are printed without alarms<br>`2`: Enable feature value detection and raise alarms<br>`3`: Enable feature value detection and raise alarms, and the process data is also recorded in info-level logs on the device side |
| `ASCEND_LAUNCH_BLOCKING` | Controls whether to enable the synchronous mode when operators are executed | `0`: Execute in asynchronous mode<br>`1`: Force operators to run in synchronous mode |
| `NPUS_PER_NODE` | Configures the number of NPUs used on a compute node | Integer value (such as `1`, `8`, etc.) |

---
