# Wan2.1 User Guide

- [Wan2.1 User Guide](#wan21-user-guide)
  - [Version Note](#version-note)
    - [Reference Implementation](#reference-implementation)
  - [Supported Tasks](#supported-tasks)
  - [Environment Setup](#environment-setup)
    - [Repository Pull and Environment Setup](#repository-pull-and-environment-setup)
    - [Decord Setup](#decord-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [Diffusers Weight Download](#diffusers-weight-download)
    - [Weight Conversion](#weight-conversion)
  - [Pre-training](#pre-training)
    - [Data Preprocessing](#data-preprocessing)
    - [Training](#training)
      - [Prerequisites](#prerequisites)
      - [Parameter Configuration](#parameter-configuration)
  - [LoRA Fine-tuning](#lora-fine-tuning)
    - [Prerequisites](#prerequisites-1)
    - [Parameter Configuration](#parameter-configuration-1)
    - [Start Fine-tuning](#start-fine-tuning)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Note

### Reference Implementation

T2V/I2V LoRA fine-tuning task:

```shell
url=https://github.com/modelscope/DiffSynth-Studio.git
commit_id=03ea278
```

## Supported Tasks

| Model Size | Task Type | Pre-training | LoRA Fine-tuning | Online T2V Inference | Online I2V Inference | Online FLF2V Inference | Online V2V Inference |
|------|:----:|:----|:-------|:-|:-----|:-----|:-|
| 1.3B | t2v  | ✔ | ✔ |  |  |  |  |
| 1.3B | i2v  | ✔ |  |  |  |  |  |

## Environment Setup

The dependencies suite for the MindSpeed-MM MindSpore backend are shown in the table below. For installation steps, refer to the [Installation Guide](../../../docs/en/mindspore/install_guide.md).

| Dependency         |                                    Version                          |
| ---------------- | ------------------------------------------------------------ |
| Ascend NPU Driver & Firmware  | In-development |
| Ascend CANN        | In-development |
| MindSpore        | [2.7.0](https://www.mindspore.cn/install/en)         |
| Python           | >=3.9                                                        |
|mindspore_op_plugin | [In-development](https://gitee.com/mindspore/mindspore_op_plugin) |

<a id="jump1.1"></a>

### Repository Pull and Environment Setup

For the MindSpeed MindSpore backend, the Ascend community provides a one-click setup tool, MindSpeed-Core-MS, designed to help users automatically pull relevant code repositories and perform one-click adaptation of torch code. This allows users to initiate model training with a single click in the Huawei MindSpore + CANN environment without additional manual adaptation. Before performing the one-click setup, users need to pull the relevant code repositories and set up the environment.

```shell
# Create a conda environment.
conda create -n test python=3.10
conda activate test

# Use environment variables.
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# Install the MindSpeed-Core-MS tool.
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.5.0

# Use the internal scripts of MindSpeed-Core-MS to automatically pull the relevant code repositories, perform one-click adaptation, and configure the environment.
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm

pip install transformers==4.51.0
pip install diffusers==0.30.3

# Pull and install mindspore_op_plugin.
git clone https://gitee.com/mindspore/mindspore_op_plugin.git
cd mindspore_op_plugin
bash build.sh
pip install output/xxx.whl
source env.source
cd ..

mkdir ckpt
mkdir data
mkdir logs
```

> Note
>
> [mindspore_op_plugin](https://gitee.com/mindspore/mindspore_op_plugin) is an operator plugin library for MindSpore. It quickly supplements CPU/GPU operator functionality by directly calling ATen operators in libtorch. It is currently an **experimental feature** and is **restricted for use** only in this model.
>
> For more details, please refer to the [op_plugin CPU Operator Development Guide](https://gitee.com/mindspore/mindspore_op_plugin/wikis/op_plugin%20CPU%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97).

### Decord Setup

X86:

```bash
pip install decord==0.6.0
```

Arm:

For installation via `apt`, please [refer to the link](https://github.com/dmlc/decord).

For installation via `yum`, please [refer to the script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

## Weight Download and Conversion

### Diffusers Weight Download

|   Model   |   Hugging Face Download Link   |
| ---- | ---- |
|   T2V-1.3B   |   <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>   |

### Weight Conversion

You need to perform weight conversion on the `transformer` part of the downloaded Wan2.1 model weights by running the weight conversion script:

```shell
mm-convert WanConverter hf_to_mm \
 --cfg.source_path <./weights/Wan-AI/Wan2.1-{T2V/I2V}-1.3B-Diffusers/transformer/> \
 --cfg.target_path <./weights/Wan-AI/Wan2.1-{T2V/I2V}-1.3B-Diffusers/transformer/> \
 --cfg.target_parallel_config.pp_layers <pp_layers>
```

The parameters of the weight conversion script are described as follows:

| Parameter        | Meaning                                  | Default Value                                                |
| :--------------- | :--------------------------------------- | :----------------------------------------------------------- |
| `--cfg.source_path` | Original weight path                     | /                                                            |
| `--cfg.target_path` | Path to save converted or partitioned weights  | /                                                            |
| `--pp_layers`      | Number of PP/VPP layers                  | When PP is enabled, using PP and VPP requires specifying the number of layers at each stage and conversion. The default is `[]`, meaning not used. |

If you need to convert weights back to the Hugging Face format, run the weight conversion script:

**Note**: If training with LayerZeRO is performed, you must first perform [Training Weight Post-processing](#jump1), and then proceed with the following operations:

```shell
mm-convert WanConverter mm_to_hf \
 --cfg.source_path <path for your saved weight/> \
 --cfg.target_path <./converted_weights/Wan-AI/Wan2.1-{T2V/I2V}-1.3B-Diffusers/transformer/> \
 --cfg.hf_dir <weights/Wan-AI/Wan2.1-{T2V/I2V}-1.3B-Diffusers/transformer/>
```

The parameters of the weight conversion script are described as follows:

| Parameter | Meaning | Default Value |
|:------------------|:----|:----|
| `--cfg.source_path` | Path to the weights saved by MindSpeed MM | / |
| `--cfg.target_path` | Path to the converted Hugging Face weights | / |
| `--cfg.hf_dir` | Path to the original Hugging Face weights, from which the original Hugging Face configuration files are obtained | / |

## Pre-training

### Data Preprocessing

Process the data into the following format:

```bash
</dataset>
  ├──data.json
  ├──videos
  │  ├──video0001.mp4
  │  ├──video0002.mp4
```

`videos/` stores the videos, and `data.json` contains all the video-text pair information in the dataset. A specific example is as follows:

```json
[
    {
        "path": "videos/video0001.mp4",
        "cap": "Video discrimination1.",
        "num_frames": 81,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 832
        }
    },
    {
        "path": "videos/video0002.mp4",
        "cap": "Video discrimination2.",
        "num_frames": 81,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 832
        }
    },
    ......
]
```

Modify the `examples/mindsporewan2.1/feature_extract/data.txt` file, where each line represents a dataset. The first parameter indicates the path to the data folder, and the second parameter indicates the path to the `data.json` file, separated by `,`

### Training

#### Prerequisites

Before starting, please confirm that environment preparation, model weight download, and feature extraction have been completed.

#### Parameter Configuration

Check that the model weight path, parallel parameter configuration, etc. are complete.

| Configuration File | Field to be Modified| Instruction |
| --- | :---: | :--- |
| examples/mindsporewan2.1/{model_size}/{task}/feature_data.json | basic_parameters | Dataset path. Set `data_path` and `data_folder` to the file path and directory of the extracted features, respectively. |
| examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh | NPUS_PER_NODE | Number of NPUs per node |
| examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh | NNODES | Number of nodes |
| examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh | LOAD_PATH | Pre-training weight path after weight conversion |
| examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh | SAVE_PATH | Weight path saved during training |
| examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh | CP | CP size during training (recommended to adjust it according to the resolution set during training) |

[Parallel Parameter Configuration]

When adjusting model parameters or video sequence length, the following parallel strategies need to be enabled based on actual conditions, and the optimal parallel strategy should be determined through debugging.

- CP: Sequence Parallelism

  - Usage Scenario: When the video sequence (resolution × number of frames) is large, it can be enabled to reduce memory usage.

  - Enablement Method: Set CP to a value greater than 1 in the script, e.g., `CP=2`;

  - Constraints: The number of heads must be divisible by CP (see `num_heads` configured in `examples/mindsporewan2.1/{model_size}/{task}/pretrain_model.json`).

  - The default mode is Ulysses.

  - DiT-RingAttention: See [DiT-RingAttention](../../../docs/en/features/dit_ring_attention.md).

  - DiT-USP: Ulysses + RingAttention. For details, see [DiT-USP](../../../docs/en/features/dit_usp.md).

  - FPDT (Fully Pipelined Distributed Transformer): Ulysses Offload. For details, see [FPDT](../../../docs/en/features/fpdt.md).

- layer_zero

  - Usage Scenario: When the model parameter scale is large and a single card cannot accommodate the complete model, you can enable layerZeRO to reduce static memory.

  - Enablement Method: Add `--layerzero` and `--layerzero-config ${layerzero_config}` to `GPT_ARGS` in `examples/mindsporewan2.1/{model_size}/{task}/pretrain.sh`.

  <a id="jump1"></a>
  - Training Weight Post-processing: When training with this feature, the saved weights need to be post-processed using the following conversion script before they can be used for inference:

    ```bash
    # Modify the ascend-toolkit path according to the actual situation.
    source /usr/local/Ascend/cann/set_env.sh
    mm-convert WanConverter layerzero_to_mm \
     --cfg.source_path <./save_ckpt/wan2.1/> \
     --cfg.target_path <./save_ckpt/wan2.1_megatron_ckpt/>
    ```

- PP: Pipeline Parallelism

  Currently, the predictor model can be partitioned into pipelines.

  - Usage Scenario: When the model parameters are large, they can be partitioned and parallelized through pipelining to reduce training memory usage.

  - Enablement Method:
    - Modify the `pipeline_num_layers` field in the `pretrain_model.json` file, which is of type list. The length of this list is the number of pipeline ranks, and each value represents the number of layers in `rank_i`. For example, `[7, 8, 8, 7]` means there are 4 pipeline stages, each accommodating 7/8 DiT layers. Note that the sum of all values in the list should equal the total `num_layers` field. In addition, the stage with `pp_rank==0` accommodated `text_encoder` and `ae` in addition to the DiT layers, so the number of DiT layers in the 0th stage can be reduced as appropriate. Ensure that the PP parameter configuration is consistent with the parameter configuration during model conversion.
    - Additionally, when using PP, enable the following parameters in `GPT_ARGS`:

    ```shell
    PP = 4 # PP > 1 Enabled
    GPT_ARGS="
    --optimization-level 2 \
    --use-multiparameter-pipeline-model-parallel \  # Must be enabled when using PP or VPP
    --variable-seq-lengths \  # Enable as needed. This configuration is required for dynamic shape training. Do not add it for static shape training.
    “
    ```

- VPP: Virtual Pipeline Parallelism

    Currently, the predictor model can be partitioned into virtual pipelines.

  - Usage Scenario: Further split pipelines to reduce pipeline bubbles through virtualization.
  - Enablement Method:
    - To enable VPP, change the one-dimensional `pipeline_num_layers` array in the `pretrain_model.json` file into a two-dimensional array, where the first dimension represents the number of virtual pipelines, and the second dimension represents the number of pipeline stages. For example, `[[3, 4, 4, 4], [3, 4, 4, 4]]` indicates that the first dimension has two arrays, meaning `VP=2`, and the second dimension has 4 stages, meaning `pp=3` or `pp=4`.
    - The following variables need to be modified in `pretrain.sh`. Note that VPP only takes effect when `PP` is greater than 1:

    ```shell
    PP=4
    VP=2

    GPT_ARGS="
      --pipeline-model-parallel-size ${PP} \
      --virtual-pipeline-model-parallel-size ${VP} \
      --optimization-level 2 \
      --use-multiparameter-pipeline-model-parallel \  # Must be enabled to use PP or VPP.
      --variable-seq-lengths \   # Enable as needed. This configuration is required for dynamic shape training but should not be added for static shape training.
    ”
    ```

## LoRA Fine-tuning

### Prerequisites

The procedure for data processing, feature extraction, weight download, and weight conversion is the same as that in the "pre-training" section.

### Parameter Configuration

For details, see the "pre-training" section. In addition, LoRA fine-tuning parameters need to be modified.

| Configuration File                                             |        Field to be Modified         | Instruction                         |
|--------------------------------------------------|:-------------------:|:-----------------------------|
| examples/mindsporewan2.1/{model_size}/{task}/finetune_lora.sh |       lora-r        | Dimension of the LoRA update matrix                  |
| examples/mindsporewan2.1/{model_size}/{task}/finetune_lora.sh |     lora-alpha      | Controls the degree to which the decomposed matrices influence the original matrix. |
| examples/mindsporewan2.1/{model_size}/{task}/finetune_lora.sh | lora-target-modules | List of modules to which LoRA is applied                  |

### Start Fine-tuning

```bash
bash examples/mindsporewan2.1/{model_size}/{task}/finetune_lora.sh
```

After fine-tuning is complete, you can use the weight conversion tool to merge the trained LoRA weights with the original weights.

```bash
mm-convert WanConverter merge_lora_to_base \
 --cfg.source_path <./converted_weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer/> \
 --cfg.target_path <./converted_weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer_merge/> \
 --cfg.lora_path <lora_save_path> \
 --lora_alpha 64 \
 --lora_rank 64
```

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
