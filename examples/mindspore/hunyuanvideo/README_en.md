# HunyuanVideo User Guide

- [HunyuanVideo User Guide](#hunyuanvideo-user-guide)
  - [Version Notes](#version-notes)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Setup](#environment-setup)
    - [Repository Pull and Environment Setup](#repository-pull-and-environment-setup)
    - [Decord Setup](#decord-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [TextEncoder Download](#textencoder-download)
    - [HunyuanVideoDiT and VAE Download](#hunyuanvideodit-and-vae-download)
    - [Weight Conversion](#weight-conversion)
  - [Pre-training](#pre-training)
    - [Data Preprocessing](#data-preprocessing)
    - [Training](#training)
      - [Prerequisites](#prerequisites)
      - [Parameter Configuration](#parameter-configuration)
      - [Start Training](#start-training)
      - [Weight Post-processing](#weight-post-processing)
  - [I2V LoRA Fine-tuning](#i2v-lora-fine-tuning)
    - [Prerequisites](#prerequisites-1)
      - [Weight Conversion](#weight-conversion-1)
      - [Parameter Configuration](#parameter-configuration-1)
    - [Start LoRA Fine-tuning](#start-lora-fine-tuning)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Notes

### Reference Implementation

T2V task:

```shell
url=https://github.com/hao-ai-lab/FastVideo
commit_id=a33581186973e6d7355f586fa065b6abb29b97fb
```

I2V and I2V LoRA fine-tuning task:

```shell
url=https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V
commit_id=2766232ceaafeb680ca32fe0a7e9735c04b561d4
```

### Changelog

2025.06.07: Synchronized key parameter modifications from the FastVideo original repository for the T2V task, setting the default value of the `embedded_guidance_scale` parameter to `1`.

2025.04.27: Initial release of the HunyuanVideo I2V task and I2V LoRA fine-tuning task.

2025.02.20: Initial release of HunyuanVideo T2V.

## Environment Setup

The dependencies for the MindSpeed-MM MindSpore backend are listed in the table below. For installation steps, refer to the [Installation Guide](../../../docs/en/mindspore/install_guide.md).

| Dependency         |                                                                                                                                   |
| ---------------- |-----------------------------------------------------------------------------------------------------------------------------------|
| Ascend NPU Driver & Firmware  | In-development |
| Ascend CANN        | In-development                                               |
| MindSpore        | [2.7.1](https://www.mindspore.cn/install/en)                                                                                        |
| Python           | >=3.9                                                                                                                             |
|mindspore_op_plugin | [In-development](https://gitee.com/mindspore/mindspore_op_plugin)                                                                           |

<a id="jump1.1"></a>

### Repository Pull and Environment Setup

For the MindSpeed MindSpore backend, the Ascend community provides a one-click setup tool, MindSpeed-Core-MS, designed to help users automatically pull relevant code repositories and perform one-click adaptation of torch code, thereby enabling users to start model training with one click in the Huawei MindSpore + CANN environment without additional manual adaptation development. Before performing the one-click setup, users need to pull the relevant code repositories and set up the environment.

```shell
# Create a conda environment.
conda create -n test python=3.10
conda activate test

# Using environment variables
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# Install MindSpeed-Core-MS Launcher.
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.5.0

# Use the internal script of MindSpeed-Core-MS to automatically pull the relevant code repositories, perform one-click adaptation, and configure the environment.
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm
#MM Version
cd MindSpeed-MM
git checkout 2.3.0
git checkout 4da05733e49e9f2b47ad48d7c488af0975033a34
cd ..

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

Note

- [mindspore_op_plugin](https://gitee.com/mindspore/mindspore_op_plugin) is an operator plugin library for MindSpore. It quickly supplements CPU/GPU operator functionality by directly calling ATen operators in libtorch. It is currently an **experimental feature** and is **restricted for use** only with this model.

- For a tutorial on using op_plugin, please refer to the [op_plugin CPU Operator Development Guide](https://gitee.com/mindspore/mindspore_op_plugin/wikis/op_plugin%20CPU%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97).

### Decord Setup

X86:

```bash
pip install decord==0.6.0
```

Arm:

For installation via `apt`, please [refer to the link](https://github.com/dmlc/decord).

For installation via `yum`, please [refer to the script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

## Weight Download and Conversion

### TextEncoder Download

+ [llava-llama-3-8b](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers)
+ [clip-vit-large](https://huggingface.co/openai/clip-vit-large-patch14)

### HunyuanVideoDiT and VAE Download

+ [tencent/HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)
+ [tencent/HunyuanVideo-I2V](https://huggingface.co/tencent/HunyuanVideo-I2V)

The downloaded weight structures are as follows:

```shell
HunyuanVideo
  ├──README.md
  ├──hunyuan-video-t2v-720p
  │  ├──transformers
  │  │  ├──mp_rank_00_model_states.pt
  │  ├──vae
  │  │  ├──config.json
  │  │  ├──pytorch_model.pt
```

```shell
  HunyuanVideo-I2V
    ├──README.md
    ├──hunyuan-video-i2v-720p
    │  ├──transformers
    │  │  ├──mp_rank_00_model_states.pt
    │  ├──vae
    │  ├──lora
    │  │  ├──embrace_kohaya_weights.safetensors
    │  │  ├──hair_growth_kohaya_weights.safetensors
```

Among them, `HunyuanVideo/hunyuan-video-t2v-720p/transformers` and `HunyuanVideo-I2V/hunyuan-video-i2v-720p/transformers` are the weights for the transformer part, `HunyuanVideo/hunyuan-video-t2v-720p/vae` and `HunyuanVideo-I2V/hunyuan-video-i2v-720p/vae` are the weights for the VAE part, and `HunyuanVideo-I2V/hunyuan-video-i2v-720p/lora` is the LoRA weight.

### Weight Conversion

The T2V task requires weight conversion for the `llava-llama3-8b` model. Run the weight conversion script:

```shell
mm-convert HunyuanVideoConverter --version t2v t2v_text_encoder \
 --cfg.source_path <llava-llama-3-8b> \
 --cfg.target_path <llava-llama-3-8b-text-encoder-tokenizer> \
```

Weight conversion needs to be performed separately for the transformer parts of HunyuanVideo-t2v and i2v. Run the weight conversion scripts:

```shell
mm-convert HunyuanVideoConverter --version t2v source_to_mm \
 --cfg.source_path <hunyuan-video-t2v-720p/transformers/mp_rank_00/model_states.pt> \
 --cfg.target_path <./ckpt/hunyuanvideo> \
 --cfg.target_parallel_config.tp_size=<tp_size>
```

```bash
mm-convert HunyuanVideoConverter --version i2v source_to_mm \
 --cfg.source_path <hunyuan-video-i2v-720p/transformers/mp_rank_00/model_states.pt> \
 --cfg.target_path <./ckpt/hunyuanvideo> \
```

You need to convert the LoRA weights for HunyuanVideo-i2v by running the weight conversion script:

```bash
mm-convert HunyuanVideoConverter --version i2v-lora source_to_mm \
 --cfg.source_path <hunyuan-video-i2v-720p/lora/embrace_kohaya_weights.safetensors> \
 --cfg.target_path <./ckpt/hunyuanvideo-i2v-lora>
```

The parameters of the weight conversion script are described as follows:

| Parameter | Meaning | Default Value |
|:------------|:----|:----|
| --version | Different tasks | Supports `t2v`, `i2v`, `i2v-lora`. Defaults to `t2v` |
| --cfg.source_path | Original weight path | / |
| --cfg.target_path | Path to save the converted weights | / |
| --cfg.target_parallel_config.tp_size | Partitions weights by `tp_size` | 1 |

## Pre-training

### Data Preprocessing

Process the data into the following format:

```bash
</data/hunyuanvideo/dataset>
  ├──data.json
  ├──videos
  │  ├──video0001.mp4
  │  ├──video0002.mp4
```

Where, `videos/` stores the videos, and `data.json` contains all video-text pair information in the dataset. A specific example is as follows:

```json
[
    {
        "path": "videos/video0001.mp4",
        "cap": "Video discrimination1.",
        "num_frames": 93,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 848
        }
    },
    {
        "path": "videos/video0002.mp4",
        "cap": "Video discrimination2.",
        "num_frames": 93,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 848
        }
    },
    ......
]
```

Modify the `examples/mindspore/hunyuanvideo/feature_extract/data.txt` file, where each line represents a dataset. The first parameter indicates the path to the data folder, and the second parameter indicates the path to the `data.json` file, separated by `,`

### Training

#### Prerequisites

Before starting, please confirm that the environment setup and model weight download are complete.

#### Parameter Configuration

Check that the model weight path, parallel parameter configuration, etc. are complete

| Configuration File                                                   |     Field to Be Modified | Instructions                                            |
| ---------------------------------------------------------- | :-----------------: | :-------------------------------------------------- |
| examples/mindspore/hunyuanvideo/{task_name}/feature_data.json        | basic_parameters   | Dataset path. Set `data_path` and `data_folder` to the file path and directory of the extracted features, respectively. |
| examples/mindspore/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |    NPUS_PER_NODE    | Number of cards per node                                      |
| examples/mindspore/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |       NNODES        | Number of nodes                                            |
| examples/mindspore/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |      LOAD_PATH      | Pre-training weight path after weight conversion                          |
| examples/mindspore/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |      SAVE_PATH      | Path to weights saved during training                            |
| examples/mindspore/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |         TP          | TP size during training (recommended to adjust it according to the resolution set during training)   |
| examples/mindspore/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |         CP          | CP size during training (recommended to adjust it according to the resolution set during training)   |
| examples/mindspore/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh | --sequence-parallel | Enable TP-SP; defaults to enabled.                                 |

[Parallel Parameter Configuration]

When adjusting model parameters or video sequence length, you need to enable the following parallelism strategies based on actual conditions and determine the optimal parallelism strategy through debugging.

+ CP: sequence parallelism; currently supports Ulysses, RingAttention, and USP.

  - Usage Scenario: When the video sequence (resolution x number of frames) is large, it can be enabled to reduce memory usage.
  - Enablement Method: Set CP to a value greater than 1 in the startup script, e.g., `CP=2`.
    - Defaults to Ulysses.
    - For RingAttention, please [refer to its feature  documentation](../../../docs/en/features/dit_ring_attention.md).
    - DiT-USP: For hybrid DiT-USP parallelism (Ulysses + RingAttention), please [refer to its feature documentation](../../../docs/en/features/dit_usp.md).
  - Constraints:
    - When Ulysses is used, the number of heads must be divisible by TP * CP (configured in `examples/mindspore/hunyuanvideo/{task_name}/model_hunyuanvideo.json`; defaults to `24`).
    - When RingAttention or USP is used, CP cannot exceed the number of NPUs on a single compute node (see `NPUS_PER_NODE`).

+ TP: model tensor parallelism

  - Usage Scenario: When the model parameter scale is large and a single card cannot accommodate the complete model, enabling TP can reduce static memory and runtime memory.

  - Enablement Method: Set TP to a value greater than 1 in the startup script, e.g., `TP=8`.

  - Constraints: The number of heads must be divisible by TP * CP (configured in `examples/mindspore/hunyuanvideo/{task_name}/model_hunyuanvideo.json`; defaulted to `24`.)

+ TP-SP

  - Usage Scenario: On the basis of TP, additionally shard the sequence dimension of the LayerNorm and Dropout modules to reduce dynamic memory.

  - Enablement Method: Set `--sequence-parallel` in `GPT_ARGS`.

  - Suggestion: It is recommended to enable this setting synchronously when TP is enabled.

+ Selective recomputation + FA activation offloading

  - If the GPU memory is relatively sufficient, you can enable selective recomputation (FA does not perform recomputation) to improve throughput. It is recommended to simultaneously enable FA activation offloading to asynchronously offload FA activations to the CPU.

  - In `examples/mindspore/hunyuanvideo/{task_name}/model_hunyuanvideo.json`, `attention_async_offload` indicates whether to enable FA activation offloading, which is enabled by default.

  - In `examples/mindspore/hunyuanvideo/{task_name}/model_hunyuanvideo.json`, `double_stream_full_recompute_layers` and `single_stream_full_recompute_layers` indicate the number of layers in the model's `double_stream_block` and `single_stream_block` that undergo full recomputation. These two parameters can be gradually reduced until the GPU memory is fully utilized.

> ⚠️**HunyuanVideo I2V currently does not support CP and TP-SP.**

#### Start Training

```bash
bash examples/mindspore/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh
```

#### Weight Post-processing

If `TP>1` during training, the trained weights need to be merged. The merged weights can then be used for inference.

```bash
mm-convert HunyuanVideoConverter --version t2v source_to_mm \
 --cfg.source_path <./save_ckpt/hunyuanvideo> \
 --cfg.target_path <./save_ckpt_merged/hunyuanvideo> \
 --cfg.target_parallel_config.tp_size=<target_tp_size>
```

## I2V LoRA Fine-tuning

### Prerequisites

Before configuring the script, ensure that the environment setup is complete.

#### Weight Conversion

The transformer part of hunyuanvideo-i2v requires weight conversion. Run the weight conversion script:

```bash
mm-convert HunyuanVideoConverter --version i2v source_to_mm \
 --cfg.source_path <hunyuan-video-i2v-720p/transformers/mp_rank_00/model_states.pt> \
 --cfg.target_path <./ckpt/hunyuanvideo> \
```

#### Parameter Configuration

The default configuration has been tested. You can modify the following content according to your own environment.

1. Weight Configuration

  After the weight conversion is complete, according to the actual task, add the actual path of the converted weights to the `LOAD_PATH="your_converted_dit_ckpt_dir"` variable in the startup script file (`examples/mindspore/hunyuanvideo/i2v/pretrain_hunyuanvideo_lora.sh`), for example, `LOAD_PATH="./ckpt/hunyuanvideo-i2v"`, where `./ckpt/hunyuanvideo-i2v` is the actual path of the converted weights. The full path filled in the `LOAD_PATH` variable must be correct. An incorrect path will cause the weights to fail to load, but the run will not report an error.
  Fill in the path in the `SAVE_PATH` variable as needed to save the trained LoRA weights.

### Start LoRA Fine-tuning

```shell
bash examples/mindspore/hunyuanvideo/i2v/pretrain_hunyuanvideo_lora.sh
```

After training is complete, the saved weights only include the LoRA fine-tuning part. If you need to merge them into the original weights, you can execute the following script to complete the merging (the configuration is for reference only):

```bash
mm-convert HunyuanVideoConverter --version i2v merge_lora_to_base \
 --cfg.source_path <'converted_transformer'>
 --cfg.target_path <'merged_weight_dir'>
 --cfg.lora_path <'converterd_lora_dir'>
 --lora-alpha 64 \
 --lora-rank 64
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
