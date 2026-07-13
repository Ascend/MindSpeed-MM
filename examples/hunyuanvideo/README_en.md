# HunyuanVideo User Guide

- [HunyuanVideo User Guide](#hunyuanvideo-user-guide)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Change Log](#change-log)
  - [Environment Setup](#environment-setup)
    - [Repository Cloning](#repository-cloning)
    - [Environment Setup](#environment-setup-1)
    - [Decord Setup](#decord-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [TextEncoder Download](#textencoder-download)
    - [HunyuanVideoDiT and VAE Download](#hunyuanvideodit-and-vae-download)
    - [Weight Conversion](#weight-conversion)
  - [Pre-training](#pre-training)
    - [Data Preprocessing](#data-preprocessing)
    - [Feature Extraction](#feature-extraction)
      - [Preparation](#preparation)
      - [Parameter Configuration](#parameter-configuration)
      - [Start Feature Extraction](#start-feature-extraction)
    - [Training](#training)
      - [Preparation](#preparation-1)
      - [Parameter Configuration](#parameter-configuration-1)
      - [Start Training](#start-training)
      - [Weight Post-processing](#weight-post-processing)
  - [I2V LoRA Fine-tuning](#i2v-lora-fine-tuning)
    - [Preparation](#preparation-2)
      - [Weight Conversion](#weight-conversion-1)
      - [Feature Extraction](#feature-extraction-1)
      - [Parameter Configuration](#parameter-configuration-2)
    - [Start LoRA Fine-tuning](#start-lora-fine-tuning)
  - [Inference](#inference)
    - [Preparation](#preparation-3)
    - [Parameter Configuration](#parameter-configuration-3)
    - [Start Inference](#start-inference)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Description

### Reference Implementation

T2V task:

```shell
url=https://github.com/hao-ai-lab/FastVideo
commit_id=a33581186973e6d7355f586fa065b6abb29b97fb
```

I2V and I2V LoRA fine-tuning tasks:

```shell
url=https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V
commit_id=2766232ceaafeb680ca32fe0a7e9735c04b561d4
```

### Change Log

2025.06.07: Synchronized key parameter modifications from the source FastVideo repository for the T2V task, setting the default value of the `embedded_guidance_scale` parameter to `1`.

2025.04.27: Initial release of the HunyuanVideo I2V task and I2V LoRA fine-tuning task.

2025.02.20: Initial release of HunyuanVideo T2V.

## Environment Setup

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md).

### Repository Cloning

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
```

### Environment Setup

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# Install torch and torch_npu. Ensure you select the torch, torch_npu, and apex packages corresponding to your Python version and architecture (x86 or arm).
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

# For apex for Ascend, refer to https://gitcode.com/Ascend/apex.
# It is recommended to compile and install from the original repository.

# Modify the environment variable paths in the shell script to the actual paths. Example:
source /usr/local/Ascend/cann/set_env.sh

# Install the acceleration library.
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout 6aff65eba929b4f39848a5153ac455467d0b0f9e
pip install -r requirements.txt
pip install -e .
cd ..

# Install other required dependency libraries.
pip install -e .
```

### Decord Setup

X86:

```bash
pip install decord==0.6.0
```

Arm:

For installation via `apt`, please [click this link](https://github.com/dmlc/decord).

For installation via `yum`, please [refer to this script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

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

`HunyuanVideo/hunyuan-video-t2v-720p/transformers` and `HunyuanVideo-I2V/hunyuan-video-i2v-720p/transformers` are the transformer weights, `HunyuanVideo/hunyuan-video-t2v-720p/vae` and `HunyuanVideo-I2V/hunyuan-video-i2v-720p/vae` are the VAE weights, and `HunyuanVideo-I2V/hunyuan-video-i2v-720p/lora` is the LoRA weights.

### Weight Conversion

For T2V tasks, the `llava-llama3-8b` model requires weight conversion. Run the weight conversion script:

```shell
mm-convert HunyuanVideoConverter --version t2v t2v_text_encoder \
 --cfg.source_path <llava-llama-3-8b> \
 --cfg.target_path <llava-llama-3-8b-text-encoder-tokenizer>
```

Weight conversion needs to be performed separately for the transformer parts of HunyuanVideo-T2V and HunyuanVideo-I2V. Run the weight conversion script:

```shell
mm-convert HunyuanVideoConverter --version t2v source_to_mm \
 --cfg.source_path <hunyuan-video-t2v-720p/transformers/mp_rank_00/model_states.pt> \
 --cfg.target_path <./ckpt/hunyuanvideo> \
 --cfg.target_parallel_config.tp_size=<tp_size>
```

```bash
mm-convert HunyuanVideoConverter --version i2v source_to_mm \
 --cfg.source_path <hunyuan-video-i2v-720p/transformers/mp_rank_00/model_states.pt> \
 --cfg.target_path <./ckpt/hunyuanvideo>
```

Weight conversion needs to be performed for the LoRA weights of HunyuanVideo-i2v. Run the weight conversion script:

```bash
mm-convert HunyuanVideoConverter --version i2v-lora source_to_mm \
 --cfg.source_path <hunyuan-video-i2v-720p/lora/embrace_kohaya_weights.safetensors> \
 --cfg.target_path <./ckpt/hunyuanvideo-i2v-lora>
```

The parameter descriptions for the weight conversion script are as follows.

| Parameter | Meaning | Default Value |
|:------------|:----|:----|
| --version | Different tasks | Supports `t2v`, `i2v`, `i2v-lora`; defaulted to `t2v`. |
| --cfg.source_path | Source weight path | / |
| --cfg.target_path | Converted weight save path | / |
| --cfg.target_parallel_config.tp_size | Partitions weights by `tp_size`. | 1 |

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

Where, `videos/` stores the videos, and `data.json` contains all the video-text pair information in the dataset. A specific example is as follows:

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

Modify the `examples/hunyuanvideo/feature_extract/data.txt` file, where each line represents a dataset. The first parameter indicates the path to the data folder, and the second parameter indicates the path to the `data.json` file, separated by `,`

### Feature Extraction

#### Preparation

Before starting, please ensure that the environment setup, model weight download and conversion, and dataset preprocessing have been completed.

#### Parameter Configuration

Ensure that parameters such as the model weight path, dataset path, and feature extraction save path are configured.

| Configuration File                                                     |       Field to be Modified      |  Instructions                                                                 |
| ------------------------------------------------------------ | :-------------------: | :----------------------------------------------------------------------- |
| examples/hunyuanvideo/feature_extract/data.json              |      num_frames       | Maximum number of frames; if exceeded, `num_frames` frames are randomly selected. |
| examples/hunyuanvideo/feature_extract/data.json              | max_height, max_width | Maximum height and width; if exceeded, CenterCrop to the maximum resolution. |
| examples/hunyuanvideo/feature_extract/data.json              |    from_pretrained    | Modify it to the path corresponding to the downloaded weights (including the Tokenizer). |
| examples/hunyuanvideo/feature_extract/feature_extraction.sh  |     NPUS_PER_NODE     | Number of NPUs                                                                 |
| examples/hunyuanvideo/feature_extract/model_hunyuanvideo.json |    from_pretrained    | Modify it to the path corresponding to the downloaded weights (including VAE and Text Encoder). |
| mindspeed_mm/tools/tools.json                                |       save_path       | Save path for the extracted features                                        |

#### Start Feature Extraction

```bash
bash examples/hunyuanvideo/feature_extract/feature_extraction.sh
```

### Training

#### Preparation

Before starting, please confirm that preparations for the environment setup, model weight download, and feature extraction have been completed.

#### Parameter Configuration

Ensure that items such as the model weight path and parallel parameter configuration are set.

| Configuration File                                                   |     Field to be Modified      |  Instructions                                            |
| ---------------------------------------------------------- | :-----------------: | :-------------------------------------------------- |
| examples/hunyuanvideo/{task_name}/feature_data.json        | basic_parameters   | Dataset path. Set `data_path` and `data_folder` to the file path and directory of the extracted features, respectively.
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |    NPUS_PER_NODE    | Number of NPUs per node                                      |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |       NNODES        | Number of nodes                                            |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |      LOAD_PATH      | Pre-training weight path after weight conversion                          |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |      SAVE_PATH      | Weight path saved during training                            |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |         TP          | TP size during training (recommended to adjust it according to the resolution set during training)   |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |         CP          | CP size during training (recommended to adjust it according to the resolution set during training)   |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh | --sequence-parallel | Enable TP-SP. It is enabled by default.                                 |

Parallel parameter configuration:

When you adjust model parameters or video sequence length, it is necessary to enable the following parallel strategies based on the actual situation and determine the optimal parallel strategy through debugging.

+ CP: sequence parallelism; currently supports Ulysses, RingAttention, and USP.

  - Usage Scenario: When the video sequence (resolution x number of frames) is large, it can be enabled to reduce memory usage.
  - Enablement Method: Set CP to a value greater than 1 in the startup script, e.g., `CP=2`.
    - Defaults to Ulysses.
    - For RingAttention, please [refer to its feature  documentation](../../docs/en/features/dit_ring_attention.md).
    - DiT-USP: For hybrid DiT-USP parallelism (Ulysses + RingAttention), please [refer to its feature documentation](../../docs/en/features/dit_usp.md).
  - Constraints:
    - When Ulysses is used, the number of heads must be divisible by TP * CP (configured in `examples/hunyuanvideo/{task_name}/model_hunyuanvideo.json`; defaults to `24`).
    - When RingAttention or USP is used, CP cannot exceed the number of NPUs on a single compute node (see `NPUS_PER_NODE`).

+ TP: model tensor parallelism

  - Usage Scenario: When the model parameter scale is large and a single card cannot accommodate the complete model, enabling TP can reduce static memory and runtime memory.

  - Enablement Method: Set TP to a value greater than 1 in the startup script, e.g., `TP=8`.

  - Constraints: The number of heads must be divisible by TP * CP (configured in `examples/hunyuanvideo/{task_name}/model_hunyuanvideo.json`; defaulted to `24`.)

+ TP-SP

  - Usage Scenario: On the basis of TP, additionally shard the sequence dimension of the LayerNorm and Dropout modules to reduce dynamic memory.

  - Enablement Method: Set `--sequence-parallel` in `GPT_ARGS`.

  - Suggestion: It is recommended to enable this setting synchronously when TP is enabled.

+ layer_zero

  - Usage Scenario: When the model parameter scale is large and a single card cannot accommodate the complete model, you can reduce static memory by enabling `layer_zero`.

  - Enablement Method: Add `--layerzero` and `--layerzero-config ${layerzero_config}` to `GPT_ARGS` in `examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh`.

  - Suggestion: This feature and TP are mutually exclusive. When this feature is enabled, TP must be set to 1. It is recommended to set `zero3_size` in the configuration file `examples/hunyuanvideo/zero_config.yaml` to the number of cards on a single node.

  - Training Weight Post-processing: When training with this feature, the saved weights need to be post-processed using the following conversion script before they can be used for inference:

    ```bash
    # Modify the ascend-toolkit path according to the actual situation.
    source /usr/local/Ascend/cann/set_env.sh
    mm-convert HunyuanVideoConverter --version t2v --layerzero_to_mm \
     --cfg.source_path <./save_ckpt/hunyuanvideo/> \
     --cfg.target_path <./save_ckpt/hunyuanvideo_megatron_ckpt/>
    ```

+ Selective recomputation + FA activation offloading

  - If the GPU memory is relatively sufficient, you can enable selective recomputation (FA does not perform recomputation) to improve throughput. It is recommended to simultaneously enable FA activation offloading to asynchronously offload FA activations to the CPU.

  - In `examples/hunyuanvideo/{task_name}/model_hunyuanvideo.json`, `attention_async_offload` indicates whether to enable FA activation offloading, which is enabled by default.

  - In `examples/hunyuanvideo/{task_name}/model_hunyuanvideo.json`, `double_stream_full_recompute_layers` and `single_stream_full_recompute_layers` indicate the number of layers in the model's `double_stream_block` and `single_stream_block` that undergo full recomputation. These two parameters can be gradually reduced until the GPU memory is fully utilized.

> ⚠️**HunyuanVideo I2V currently does not support CP and TP-SP.**

#### Start Training

```bash
bash examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh
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

### Preparation

Please ensure the environment setup is complete before configuring the script.

#### Weight Conversion

 The transformer part of HunyuanVideo-I2V needs weight conversion. Run the weight conversion script:

```bash
mm-convert HunyuanVideoConverter --version i2v source_to_mm \
 --cfg.source_path <hunyuan-video-i2v-720p/transformers/mp_rank_00/model_states.pt> \
 --cfg.target_path <./ckpt/hunyuanvideo> \
```

#### Feature Extraction

Please refer to the [Feature Extraction](#feature-extraction) section above, and modify the VAE weight path to that under the `hunyuan-video-i2v-720p` directory.

#### Parameter Configuration

The default configuration has been tested. You can modify the following content according to your own environment:

1. Weight Configuration

After the weight conversion is complete, fill in the actual path of the converted weights in the `LOAD_PATH="your_converted_dit_ckpt_dir"` variable in the startup script file (`examples/hunyuanvideo/i2v/pretrain_hunyuanvideo_lora.sh`) according to the actual task, for example, `LOAD_PATH="./ckpt/hunyuanvideo-i2v"`, where `./ckpt/hunyuanvideo-i2v` is the actual path of the converted weights. The complete path filled in the `LOAD_PATH` variable must be correct; an incorrect path will cause the weights to fail to load without triggering an error message during runtime.
Fill in the path in the `SAVE_PATH` variable as needed to save the trained LoRA weights.

### Start LoRA Fine-tuning

```shell
bash examples/hunyuanvideo/i2v/pretrain_hunyuanvideo_lora.sh
```

After training is complete, only weights of the LoRA fine-tuning part are saved. If you need to merge them into the original weights, you can execute the following script to complete the merge (the configuration is for reference only):

```bash
mm-convert HunyuanVideoConverter --version i2v merge_lora_to_base \
 --cfg.source_path <'converted_transformer'> \
 --cfg.target_path <'merged_weight_dir'> \
 --cfg.lora_path <'converterd_lora_dir'> \
 --lora-alpha 64 \
 --lora-rank 64
```

## Inference

### Preparation

Before starting, please confirm that the environment preparation and model weight download are complete.

### Parameter Configuration

Check if the model weight path, parallel parameters, and other configurations are complete.

| Configuration File                                           |               Field to Be Modified              |                 Instructions                 |
|---------------------------------------------------|:--------------------------------:|:-----------------------------------|
| examples/hunyuanvideo/{task_name}/inference_model.json |         from_pretrained          |            Modify it to the path corresponding to the downloaded weights (including VAE, Text Encoder).            |
| examples/hunyuanvideo/{task_name}/samples_prompts.txt |               File Content               |      You can customize your own prompts, one prompt per line.     |
| examples/hunyuanvideo/{task_name}/inference_model.json |  input_size |  Resolution of the generated video in the format of [t, h, w]. |
| examples/hunyuanvideo/{task_name}/inference_model.json |  save_path |  Save path for the generated video |
| examples/hunyuanvideo/{task_name}/inference_hunyuanvideo.sh |   LOAD_PATH | Weight path of the converted transformer part |

### Start Inference

```shell
bash examples/hunyuanvideo/{task_name}/inference_hunyuanvideo.sh
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
