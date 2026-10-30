# Qwen3_5 User Guide

<p align="left">
</p>

## Contents

- [Version Notes](#version-notes)
  - [Reference Implementation](#reference-implementation)
  - [Change History](#change-history)
- [Environment Installation](#environment-installation)
  - [Environment Preparation](#1-environment-preparation)
  - [Environment Setup](#2-environment-setup)
  - [Install the Matching Version of Triton-Ascend](#3-install-the-matching-version-of-triton-ascend)
  - [Install fla-npu for AscendC Adaptation](#4-install-fla-npu-for-ascendc-adaptation)
- [Weight Download and Conversion](#weight-download-and-conversion)
  - [Weight Download](#1-weight-download)
  - [Weight Loading](#2-weight-loading)
  - [Weight Saving](#3-weight-saving)
- [Dataset Preparation and Processing](#dataset-preparation-and-processing)
- [Fine-Tuning](#fine-tuning)
  - [Preparations](#1-preparations)
  - [Configuring Parameters](#2-configuring-parameters)
  - [Starting Fine-Tuning](#3-starting-fine-tuning)
  - [LoRA Fine-Tuning](#4-lora-fine-tuning)
- [Environment Variable Declaration](#environment-variable-declaration)
- [Notes](#notes)

## Version Notes

### Reference Implementation

```shell
url=https://github.com/huggingface/transformers.git
commit_id=fc91372
```

### Change History

2026.02.10: First support for the Qwen3.5 model

Environment change history:

Latest: 2026.8.24. To adapt to CANN 9.1.0 and torch 2.10.0, the reference version of Triton-Ascend is changed to 3.2.2, and the reference version of fla-npu remains v26.6.0

---
<a id="jump1"></a>

## Environment Installation

<a id="jump1.1"></a>

### 1. Environment Preparation

[It is recommended to use the matching environment version during model development]

For CANN-related configuration (driver, firmware, and Toolkit package), see [Installation Guide](../../docs/en/guides/installation/install_guide.md).

<a id="jump1.2"></a>

### 2. Environment Setup

Clone the MindSpeed MM repository and go to the repository root directory (if you need to use an earlier version of Python, see the cloning method in the [Installation Guide](../../docs/en/guides/installation/install_guide.md)):

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

If you need to manually install dependencies, such as freely specified ones, see the manual installation procedure in the [Installation Guide](../../docs/en/guides/installation/install_guide.md) (note: Qwen3.5 does not require installing the Megatron-LM library described in that procedure).

Run the following command for one-click installation:

```bash
bash scripts/install.sh --msbranch master && pip install transformers==5.2.0
```

The torch and torch_npu libraries are installed automatically here, and their versions can be selected through parameters (default: 2.10.0). Example:

```bash
bash scripts/install.sh --msbranch master --torchversion 2.10.0 && pip install transformers==5.2.0
```

Select a compatible PyTorch/torch_npu version by referring to [Version Compatibility Notes](../../docs/en/release_notes_mm.md) and [Triton-Ascend 3.2.2 Compatible Versions](https://github.com/triton-lang/triton-ascend/releases)

> [!NOTE]
> The torch_npu version installed by the current one-click installation script strictly matches torch (without a post suffix). It is recommended to install the torch version with the suffix according to the compatible versions in the link; otherwise, the subsequent environment check script may not pass.

### 3. Install the Matching Version of Triton-Ascend

To install the matching version of Triton-Ascend, see the "[Installing Triton-Ascend via pip](https://triton-ascend.readthedocs.io/en/latest/installation_guide.html)" section in the Triton-Ascend documentation to obtain the installation command for the matching version.

The following installation command is for reference:

```shell
# Note: For triton-ascend 3.2.0 and earlier, Triton-Ascend and Triton cannot coexist. You need to uninstall the community Triton first, and then install Triton-Ascend.
pip install triton-ascend==3.2.2 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

> [!NOTE]
> You may encounter some compatibility issues with triton-ascend==3.2.2. If the problem cannot be resolved after installing triton-ascend==3.2.2, try falling back to triton-ascend==3.2.1. Tests show that 3.2.1 can run the complete procedure in a Python 3.12 environment.

### 4. Install fla-npu for AscendC Adaptation

Clone the flash-linear-attention-npu repository and go to the repository root directory:

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu -b v26.6.0
cd flash-linear-attention-npu
```

Installation steps: see the README of the fla-npu repository [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu/blob/v26.6.0/README.md).

> **Note:** Make sure `gawk` and `cmake` are installed on the operating system; otherwise, the subsequent installation will fail. You can install them with the following commands:

```shell
# Ubuntu/Debian
apt-get update
apt-get install gawk cmake
# openEuler/CentOS/RHEL
yum update
yum install gawk cmake
```

The following installation commands are recommended:

```shell
# source the actual CANN path
source /usr/local/Ascend/cann/set_env.sh

python -m pip install -r requirements.txt
python scripts/check_npu_env.py --build-only

# --soc must be set to the chip type of the current machine {ascend910b/ascend910_93/ascend950}
FLA_NPU_SOC=ascend910_93 python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
```

Check whether fla_npu is installed successfully:

```bash
# Displays True
python -c "import fla_npu; import torch; print(hasattr(torch.ops.npu, 'npu_chunk_fwd_o'))"

# Displays Packaged wheel API check passed.
python scripts/check_packaged_wheel_api.py

# Displays flash-linear-attention-npu 26.6.0
pip list | grep fla
```

---

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the corresponding model weights from the Hugging Face library.

> [!NOTE]
>
> If you cannot access the Hugging Face community download resources, it is recommended to download from ModelScope. Pay attention to the correctness and security of the files to be downloaded.

- Model address: [Qwen3.5-*B](https://huggingface.co/collections/Qwen/qwen35).

 Save the downloaded model weights to the local `ckpt/hf_path/xxxxxxx` directory. (`*` indicates the corresponding size)

<a id="jump2.2"></a>

### 2. Weight Loading

Loading Hugging Face weights or DCP weights is supported. Configure the weight type to load in the `training->load_format` field of `xxx_config.yaml`; `hf`, `dcp`, and `auto` are supported. When set to `auto`, the weight type is determined based on the weight file format.

If you need to load DCP weights, first convert the HF weights to DCP weights based on the model configuration as follows:

```bash
mm-convert Qwen35Converter hf_to_dcp \
--hf_dir ckpt/hf_path/xxxxxxx \
--dcp_dir ckpt/dcp_path/xxxxxxx \
--num_workers 0

# Where:
# hf_dir: Hugging Face weight directory
# dcp_dir: directory for saving the converted DCP weights
# num_workers: number of parallel worker threads; 0 means serial execution. If storage I/O performance allows, increase the concurrency to improve conversion efficiency. The recommended value is 4.

# The converted directory structure is as follows:
# ———— xxxxxxx
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

In `xxx_config.yaml`, set the `init_model_with_meta_device` parameter to `True`, and change the `load` parameter to the converted DCP weight path (the directory one level above the `release` folder).
Note: If the MoE model does not support MTP, you can set `mtp_num_hidden_layers` in `ckpt/hf_path/xxxxxxx/config.json` to 0 before running the `mm-convert` weight conversion, to skip the MTP expert weight merging and shorten the conversion time. For example, the 397B model can be about 5 minutes faster.

<a id="jump2.3"></a>

### 3. Weight Saving

MindSpeed MM supports saving weights in Hugging Face format and DCP format. Configure the weight save type in the `training->save_format` field of `xxx_config.yaml`; `hf`, `dcp`, and `auto` are supported:

(1) When `save_format` is set to `auto`, the saved weight type is consistent with the loaded weight type.

(2) When `save_format` is set to `hf`, the saved weight files are converted to the safetensors format.

**Note: The hf save format supports saving weights only. It does not support saving optimizer states or random number states. If you need resumable training, save the weights in dcp format**

(3) When `save_format` is set to `dcp`, you can use the following command to convert the DCP weights back to HF weights:

```bash
# A sample directory structure of the DCP weights to be converted is as follows:
# ———— xxxxxxx
#   |—— release
#   |—— latest_checkpointed_iteration.txt

mm-convert Qwen35Converter dcp_to_hf \
--save_hf_dir ckpt/save_hf_path/Qwen3.5-xxB-hf-save \
--dcp_dir ./save_path/iter_000xx \
--origin_hf_dir ckpt/hf_path/Qwen3.5-xxB \
--to_bf16 false \
--num_workers 0

# Where:
# save_hf_dir: directory for saving the converted Hugging Face weights
# dcp_dir: directory of the saved DCP weights; `iter_000xx` indicates the weights saved at step xx
# origin_hf_dir: directory of the original Hugging Face weights
# to_bf16: whether to convert the weight data type from fp32 to bf16
# num_workers: number of parallel worker threads; 0 means serial execution. If storage I/O performance allows, increase the concurrency to improve conversion efficiency. The recommended value is 4.
```

Note: If MTP is not enabled for the model (that is, the `mtp_num_layers` field under model in `xxx_config.yaml` is set to 0 or is not configured), the converted weights do not contain the MTP layer weights by default. You can set `--keep_origin_mtp_weights true` to keep the MTP layer weights.

---
<a id="jump3"></a>

## Dataset Preparation and Processing

- Training with **real datasets**: see [Data Construction for VL Models · Using Real Data](../../docs/en/features/data/building_data_for_VLModel.md#real-data) (download COCO2017 → download the LLaVA-Instruct-150K annotations → run the conversion script to generate `mllm_format_llava_instruct_data.json`).
- Functional/performance testing with **mock data**: see [Data Construction for VL Models · Using Mock Data](../../docs/en/features/data/building_data_for_VLModel.md#mock-data).

## Fine-Tuning

<a id="jump4.1"></a>

### 1. Preparations

Before configuring the scripts, complete the preparatory work, including **environment installation**, **weight download and conversion**, and **dataset preparation and processing**. For details, see the corresponding sections.

<a id="jump4.2"></a>

### 2. Configuring Parameters

[Data Directory Configuration]

Modify the dataset paths in `xxx_config.yaml` as required, including fields such as `model_name_or_path`, `dataset_dir`, and `dataset`.

Example: If both the data and the corresponding JSON file are in the `/home/user/data/` directory, where the JSON file is `/home/user/data/video_data_path.json`, configure as follows:
`dataset_dir` is set to `/home/user/data/`;
`dataset` is set to `./data/video_data_path.json`.
Note that `dataset` must be configured as a relative path here.
**On multiple nodes, do not configure the same mount directory for `cache_dir`, to avoid conflicts caused by writing to the same file**.

[Module Freezing Configuration]

Custom frozen modules are supported. Configure the modules to freeze in the model->freeze field of `xxx_config.yaml` to freeze the corresponding modules.

[Model Saving/Loading and Log Information Configuration]

Configure the `training` parameters of `xxx_config.yaml` as required, including the save path and the save intervals `save` and `save_interval`.
Configure the `init_from_hf_path` parameter in `xxx_config.yaml` as required. This parameter indicates the loading path of the initial weights.

[Ulysses-CP Configuration]

Configure `ulysses_parallel_size` in `xxx_config.yaml` as required to adjust the parallelism degree of Ulysses-CP. (Ulysses-CP is not enabled when `ulysses_parallel_size` is 1.)

**Note that when Ulysses-CP is enabled, set `attn_implementation` in `xxx_config.yaml` to `flash_attention_2`**.

[EP Configuration]

Configure `expert_parallel_size` in `xxx_config.yaml` as required (note that this applies to MoE models only).

You can choose a more appropriate `ep_plan.dispatcher` based on `expert_parallel_size`. It is recommended to select `allgather` for `dispatcher` when `expert_parallel_size` is smaller than `topk`, and `alltoall` when `expert_parallel_size` is larger than `topk`.

[MoE Aux Loss Configuration]

For MoE models, if you need to add `router_aux_loss` on top of the cross-entropy loss during training so that the expert load distribution is balanced, configure the `features.loss_cfg.router_aux_loss_coef` field in `xxx_config.yaml`. This field is the coefficient of the load balancing loss.

[MTP Configuration]
The current model supports configuring the MTP module. Set the `mtp_num_layers` field under model in `xxx_config.yaml` to `1` (default value: `0`). The `mtp_loss_scaling_factor` field is also configurable, with a default of `0.1`.
Note: The MTP layer of Qwen3.5 currently supports only one layer.

[Performance Optimization Configuration]

- Recomputation
  - Configure it in `features.recompute`: `true` enables it, `false` disables it, and it is enabled by default.
  - When enabled, it reduces device memory usage
- [ChunkLoss](../../docs/en/features/optimization/chunkloss.md)
  - Configure it in `features.enable_chunk_loss`: `true` enables it and `false` disables it
  - `features.chunkloss_plan.chunk_size` means that, when the loss is computed, the sequence dimension is split into small chunks of size (`chunk_size`) for computation.
  - When enabled, it greatly reduces device memory spikes during loss computation and lowers overall device memory usage
- [async activation offload](../../docs/en/features/memory/async_activation_offload.md)
  - Configure it in `features.enable_activation_offload`: `true` enables it and `false` disables it.
  - When enabled, the activations at the recomputation entry are asynchronously offloaded to the host side, which further reduces device memory usage when recomputation is enabled.
- [ChunkMBS](../../docs/en/features/optimization/chunkmbs.md)
  - Configure it in `features.enable_chunk_mbs`: `true` enables it and `false` disables it.
  - `features.chunkmbs_plan.chunk_mbs` indicates the `micro_batch_size` of a single computation after partitioning.
  - Enabling this feature requires enabling the recomputation and async activation offload features at the same time. It increases the computation density of each unshard in FSDP2 and improves overall network throughput.
- Selective recomputation
  - When recomputation is enabled, you can skip the GDN recomputation of the linear attention layers or the flash attention recomputation of the full attention layers, and asynchronously offload the intermediate tensors that are saved, which reduces the amount of computation and improves training throughput without changing device memory usage.
  - Configure `model.skip_gdn_recompute` to specify whether to skip the GDN recomputation of the linear attention layers. `true` skips it and `false` does not skip it.
  - Configure `model.skip_flash_attn_recompute` to specify whether to skip the flash attention recomputation of the full attention layers. `true` skips it and `false` does not skip it.
  - Enabling this feature requires enabling the recomputation and async activation offload features at the same time.
- `gdn_implementation` and `causal_conv1d_implementation`
  - `gdn_implementation` and `causal_conv1d_implementation` support the `eager`, `triton`, and `ascendc` configurations respectively. `ascendc` provides the best performance and requires the fla_npu library.
  - When `gdn_implementation` is set to `ascendc`, `causal_conv1d_implementation` supports only `triton` and `ascendc`, which prevents layout mismatch between operators

[Text-Only SFT]

When using text-only data to perform SFT on Qwen3.5, the `text_only` collator can avoid fake modality injection and unnecessary vision tower computation.

Configure the following in `xxx_config.yaml`:

```yaml
data:
  dataloader_param:
    collate_param:
      collator_id: text_only
      ignore_pad_token_for_loss: true

model:
  freeze:
    - model.visual

features:
  recompute_plan:
    apply_modules:
      - model.language_model.layers.{*}
  activation_offload_plan:
    apply_modules:
      - model.language_model.layers.{*}
```

Notes:

- The input data must be text only and must not contain non-empty `images`, `videos`, or `audios`.
- The vision tower must be frozen (`model.freeze: [model.visual]`).
- Keep the vision modules in `fsdp_plan.apply_modules` to maintain the sharding and checkpoint topology.
- Remove the vision modules from recompute/offload and keep only the language model layers.
- To train the vision tower or use multimodal data, switch `collator_id` back to `qwen3vl`.
- FSDP2 text-only single-round training has been verified for Qwen3.5-9B, Qwen3.6-27B, and Qwen3-VL-30B-A3B-Instruct in the CANN 9.1.0 and PyTorch 2.7.1 environment.
- Qwen3.5-9B has also been verified for text-only single-round training in the CANN 9.1.0 and PyTorch 2.10.0 environment.

[Single-Node Configuration]
Take the Qwen3.5 model as an example:
Configure the parameters of `examples/qwen3_5/finetune_qwen3_5.sh` as follows.

```shell
# Modify the ascend-toolkit path as required
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
```

[Multi-Node Configuration]
To start multi-node training, modify the `MASTER_ADDR`, `NODE_ADDR`, `NNODES`, and `NODE_RANK` variables in the startup script.

``` shell
MASTER_ADDR: IP address of the master node
NODE_ADDR: IP address of the local node
NODE_RANK: index of the node
NNODES: total number of nodes
```

---

<a id="jump4.3"></a>

### 3. Starting Fine-Tuning

Differences in loss computation methods affect training results differently. Before starting a training job, see the documentation on loss computation and select an appropriate loss computation method (see [vlm_model_loss_calculate_type.md](../../docs/en/features/optimization/vlm_model_loss_calculate_type.md)).
You can configure the `loss_type` described in the above document in the `model` parameter of `xxx_config.yaml`.

```shell
bash examples/qwen3_5/finetune_qwen3_5_xxB.sh
```

<a id="jump4.4"></a>

### 4. LoRA Fine-Tuning

Set `training.lora.enable` to `true`, configure the remaining parameters as needed, and run LoRA fine-tuning with the same startup script used for full-parameter fine-tuning.

For more detailed LoRA configuration and parameter descriptions, see [LoRA Fine-Tuning](../../docs/en/features/training_mode/lora_finetune_fsdp2.md).

[Recommendations for Adjusting the Parallel Strategy]

After LoRA is enabled, all base model parameters are frozen and only the LoRA adapter parameters participate in training. The number of trainable parameters drops significantly compared with full-parameter fine-tuning, and the device memory usage of gradients and optimizer states decreases accordingly. Therefore, it is recommended to re-evaluate the parallel partitioning strategy in the `parallel` module based on the actual size of the trainable parameters, instead of reusing the full-parameter fine-tuning configuration. For example, `fully_shard_parallel_size` in the YAML configuration file does not need to remain `auto` (that is, partitioning parameters across all devices); you can explicitly specify a smaller sharding degree based on device memory conditions to reduce the communication overhead introduced by parameter sharding. For MoE models, `EP` can be set to a very low value or even disabled during LoRA fine-tuning depending on the actual situation.

<a id="jump10"></a>

## Environment Variable Declaration

| Environment Variable | Description | Value Description |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | Whether to enable log printing | `0`: disable screen log printing<br>`1`: enable screen log printing |
| `ASCEND_GLOBAL_LOG_LEVEL` | Sets the log level of application logs and the log level of each module; only debug logs are supported | `0`: DEBUG level<br>`1`: INFO level<br>`2`: WARNING level<br>`3`: ERROR level<br>`4`: NULL level, no logs are output |
| `TASK_QUEUE_ENABLE` | Controls the level of the task_queue operator delivery queue optimization | `0`: disable<br>`1`: enable Level 1 optimization<br>`2`: enable Level 2 optimization |
| `COMBINED_ENABLE` | Sets the combined flag. Setting it to 0 disables this function; setting it to 1 enables it, which optimizes scenarios with two non-consecutive operators combined | `0`: disable<br>`1`: enable |
| `CPU_AFFINITY_CONF` | Controls the processor affinity of CPU-side operator tasks, that is, binds tasks to cores | Set to `0` or not set: core binding disabled<br>`1`: coarse-grained core binding enabled<br>`2`: fine-grained core binding enabled |
| `HCCL_CONNECT_TIMEOUT` | Limits the timeout for socket connection establishment between different devices | Must be an integer in the range `[120,7200]`. The default value is `120`, in `s` |
| `PYTORCH_NPU_ALLOC_CONF` | Controls the behavior of the caching allocator | `expandable_segments:<value>`: enables the expandable segments feature of the memory pool, that is, the virtual memory feature |
| `HCCL_EXEC_TIMEOUT` | Controls the synchronous wait time during execution between devices. Within this configured time, each device process waits for other devices to complete communication synchronization | Must be an integer in the range `[68,17340]`. The default value is `1800`, in `s` |
| `ACLNN_CACHE_LIMIT` | Configures the number of operator information entries cached on the host side by the single-operator execution API | Must be an integer in the range `[1, 10,000,000]`. The default value is `10000` |
| `TOKENIZERS_PARALLELISM` | Controls the behavior of the tokenizer in the Hugging Face transformers library in a multi-threaded environment | `False`: disable parallel tokenization<br>`True`: enable parallel tokenization |
| `MULTI_STREAM_MEMORY_REUSE` | Configures whether multi-stream memory reuse is enabled | `0`: disable multi-stream memory reuse<br>`1`: enable multi-stream memory reuse |
| `NPU_ASD_ENABLE` | Controls whether the eigenvalue detection function of TorchNPU is enabled | Set to `0` or not set: eigenvalue detection disabled<br>`1`: enable eigenvalue detection, printing only abnormal logs without alarms<br>`2`: enable eigenvalue detection with alarms<br>`3`: enable eigenvalue detection with alarms, and record process data in device-side info-level logs |
| `ASCEND_LAUNCH_BLOCKING` | Controls whether the synchronous mode is used for operator execution | `0`: execute asynchronously<br>`1`: force operators to run in synchronous mode |
| `NPUS_PER_NODE` | Configures the number of NPUs used on a compute node | Integer value (such as `1`, `8`) |

---
<a id="jump11"></a>

## Notes

1. During processor loading, compatibility issues with the version of the `mistral_common` third-party library may cause the processor to be not found, and then the training exits with an error. You can resolve this as follows:
   - Uninstall the `mistral_common` third-party library: `pip uninstall -y mistral_common`
   - Upgrade the `mistral_common` third-party library to the latest version: `pip install --upgrade mistral_common`
