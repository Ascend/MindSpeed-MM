# CogVideoX User Guide

<p align="left">
</p>

## Contents

- [CogVideoX User Guide](#cogvideox-user-guide)
  - [Contents](#contents)
  - [Version Notes](#version-notes)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Supported Task List](#supported-task-list)
  - [Environment Installation](#environment-installation)
    - [Repository Cloning](#repository-cloning)
    - [Environment Setup](#environment-setup)
    - [Decord Setup](#decord-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [VAE Download](#vae-download)
    - [Transformer File Download](#transformer-file-download)
    - [T5 Model Download](#t5-model-download)
    - [Weight Conversion](#weight-conversion)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
  - [Pre-training](#pre-training)
    - [Preparation](#preparation)
    - [Parameter Configuration](#parameter-configuration)
    - [Launch Pre-training](#launch-pre-training)
  - [Inference](#inference)
    - [Preparation](#preparation-1)
    - [Parameter Configuration](#parameter-configuration-1)
    - [Launch Inference](#launch-inference)
  - [LoRA Fine-tuning](#lora-fine-tuning)
    - [Preparation](#preparation-2)
    - [Parameter Configuration](#parameter-configuration-2)
    - [Launch LoRA Fine-tuning](#launch-lora-fine-tuning)
  - [Pre-trained Model Parameter Modification Example (15B)](#pre-trained-model-parameter-modification-example-15b)
    - [Model Parameter Modification](#model-parameter-modification)
    - [Startup Script Modification](#startup-script-modification)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Notes

### Reference Implementation

```shell
url=https://github.com/THUDM/CogVideo.git
commit_id=806a7f6
```

The reference implementation is the official open-source version from SAT. Since the default configuration of the source repository cannot achieve training on competing products, the reference implementation has adjusted the following configurations:

- Sequence length adjusted to 6976
- Optimizer adjusted to AdamW

### Changelog

2025.01.24: Initial release of CogVideoX 1.5

## Supported Task List

The following model task types are supported.

|      Model      | Task Type | Task List | Supported |
|:------------:|:----:|:----:|:-----:|
| CogVideoX-5B | t2v  | Pre-training  | ✔ |
| CogVideoX-5B | t2v  | Online inference | ✔ |
| CogVideoX-5B | i2v  | Pre-training  | ✔ |
| CogVideoX-5B | i2v  | Online inference | ✔ |

## Environment Installation

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md)

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

# Install torch and torch_npu, ensuring you select the torch, torch_npu, and apex packages corresponding to your Python version and x86 or arm architecture.
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

# For apex for Ascend, refer to https://gitcode.com/Ascend/apex
# It is recommended to compile and install from the source repository.

# Modify the environment variable path in the shell script to the actual path. The following is a reference path.
source /usr/local/Ascend/cann/set_env.sh

# Install acceleration libraries.
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
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

For `apt` installation, please [click this link](https://github.com/dmlc/decord).

For `yum` installation, please [refer to this script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

## Weight Download and Conversion

### VAE Download

+ [VAE download link](https://huggingface.co/zai-org/CogVideoX1.5-5B-SAT/tree/main/vae)

### Transformer File Download

+ [CogVideoX1.0-5B-t2v](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX1.0-5B-i2v](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)
+ [CogVideoX1.5-5B-t2v](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main/transformer_t2v)
+ [CogVideoX1.5-5B-i2v](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main/transformer_i2v)

### T5 Model Download

Only the [contents](https://huggingface.co/THUDM/CogVideoX-5b/tree/main) of the `tokenizer` and `text_encoder` directories need to be downloaded.

The pre-training weight structure is as follows:

   ```shell
   CogVideoX-5B
   ├── text_encoder
   │   ├── config.json
   │   ├── model-00001-of-00002.safetensors
   │   ├── model-00002-of-00002.safetensors
   │   └── model.safetensors.index.json
   ├── tokenizer
   │   ├── added_tokens.json
   │   ├── special_tokens_map.json
   │   ├── spiece.model
   │   └── tokenizer_config.json
   ├── transformer
   │   ├── 1000 (or 1)
   │   │   └── mp_rank_00_model_states.pt
   │   └── latest
   └── vae
       └── 3d-vae.pt
   ```

### Weight Conversion

For the weight conversion `source_path` parameter, please configure the path to the transformer weight file:

```bash
mm-convert CogVideoConverter --version <t2v or i2v> source_to_mm \
  --cfg.source_path <your source path> \
  --cfg.target_path <your target path> \
  --cfg.target_parallel_config.tp_size <tp_size> \
  --cfg.target_parallel_config.pp_layers <pp_layers>
```

Here, `tp_size` is the actual TP partitioning strategy, and the value of `--version` is `t2v` or `i2v`. When PP is enabled, the number of `--pp_layers` values equals the value of `PP`, and the sum of these parameters equals `num_layers`. For example: when `num_layers=42` and `PP=4`, `pp_layers` can be set to `[10,11,11,10]`.

The structure of the converted weights is as follows:

`TP=1`, `PP=1`:

```shell
CogVideoX-5B-Converted
├── release
│   └──mp_rank_00
│      └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```

`TP=2`, `PP=1`, `TP>2`:

```shell
CogVideoX-5B-Converted
├── release
│   ├──mp_rank_00
│   │    └──model_optim_rng.pt
│   └──mp_rank_01
│      └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```

`TP=1`, `PP=4`, `PP>1`, `TP>1`:

```shell
CogVideoX-5B-Converted
├── release
│   ├──mp_rank_00_000
│   │   └──model_optim_rng.pt
│   ├──mp_rank_00_001
│   │   └──model_optim_rng.pt
│   ├──mp_rank_00_002
│   │   └──model_optim_rng.pt
│   └──mp_rank_00_003
│       └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```

## Dataset Preparation and Processing

The dataset format should be as follows:

```shell
.
├── data.jsonl
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```

Each txt file shares the same name as the video and serves as the video's label. Videos and labels should have a one-to-one correspondence.

The content of the `data.jsonl` file is shown in the following example:

```shell
{"file": "dataPath/1.mp4", "captions": "Content from 1.txt"}
{...}
...
```

## Pre-training

### Preparation

Before configuring the scripts, you need to complete the preliminary preparations by referring to the corresponding sections, including: environment installation, weight download and cConversion, and dataset preparation and processing.

### Parameter Configuration

The startup files for the CogVideoX training phase are shell scripts, which are mainly divided into the following 4 types:

|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 |  pretrain_cogvideox_i2v.sh |pretrain_cogvideox_t2v.sh  |
| 1.5 | pretrain_cogvideox_i2v_1.5.sh |pretrain_cogvideox_t2v_1.5.sh |

The configuration files for the model parameters are as follows:

|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 |  model_cogvideox_i2v.json |model_cogvideox_t2v.json  |
| 1.5 | model_cogvideox_i2v_1.5.json |model_cogvideox_t2v_1.5.json |

The `data.json` file related to the training dataset is also required.

The default configuration has been tested, and you can modify the following content according to you own environment.

1. Weight Configuration

    Based on the actual task, add the actual path of the converted weights to the `LOAD_PATH="your_converted_dit_ckpt_dir"` variable in the startup script file (e.g., `pretrain_cogvideox_i2v.sh`), such as `LOAD_PATH="./CogVideoX-5B-Converted"`, where `./CogVideoX-5B-Converted` is the actual path of the converted weights, and its folder content structure is shown in the "Weight Conversion" section. The full path filled in the `LOAD_PATH` variable must be correct; an incorrect path will cause the weights to fail to load without triggering an error message during execution.

    Fill in the path in the `SAVE_PATH` variable as needed to save the trained weights.

2. Dataset Path Configuration

    Modify the dataset paths in `data.json` according to the actual situation, specifically `"data_path":"/data_path/data.jsonl"` and `"data_folder":"/data_path/"`. Replace `"/data_path/"` with the actual dataset path.

3. VAE and T5 Model Path Configuration

    Modify the actual paths of the VAE and T5 model files in the model parameter configuration file (e.g., `model_cogvideox_i2v.json`) and the `data.json` file according to the actual situation. The path field for the T5 file is `"from_pretrained": "5b-cogvideo/tokenizer"` and `"from_pretrained": "5b-cogvideo"` (replace `5b-cogvideo` with the actual path). The path field for the VAE model file is `"from_pretrained": "3d-vae.pt"` (replace `3d-vae.pt` with the actual path).

    When you need to offload VAE and T5, change the values of the `"load_video_features": false` and `"load_text_features": false` fields in the model parameter configuration file to `true`, respectively. Change the value of the `"use_feature_data"` field in `data.json` to `true`.

4. Partitioning Strategy Configuration

* When PP is enabled, add `--optimization-level 2 --use-multiparameter-pipeline-model-parallel` to the startup script file, and change the value of the `pipeline_num_layers` parameter in the model parameter configuration file from `null` to the actual values. For example, when `PP=4` and `num_layers=42`, `"pipeline_num_layers"` can be `[11, 10, 11, 10]`. The specific values are determined according to the actual PP partitioning strategy.

* When enabling VAE CP, modify the value of the keyword `cp_size` within the `ae` dictionary in the model parameter configuration file to the desired value. It is incompatible with Encoder-DP and has not been verified for effectiveness with LayerZeRO.

* When enabling SP, add the `--sequence-parallel` parameter to the startup script file.

* When enabling Encoder-DP, change the `enable_encoder_dp` parameter in [model_cogvideox_i2v_1.5.json](i2v_1.5/model_cogvideox_i2v_1.5.json) or [model_cogvideox_t2v_1.5.json](t2v_1.5/model_cogvideox_t2v_1.5.json) to `true`. Notice that this can only be enabled when CP/TP is enabled and both `load_video_features` and `load_text_features` are `false`. It is incompatible with PP scenarios, VAE-CP, and LayerZeRO.

* When enabling LayerZeRO, add the following parameters in [pretrain_cogvideox_t2v_1.5.sh](t2v_1.5/pretrain_cogvideox_t2v_1.5.sh) or [pretrain_cogvideox_i2v_1.5.sh](i2v_1.5/pretrain_cogvideox_i2v_1.5.sh). Notice that it is incompatible with the Encoder-DP feature, TP scenarios, and PP scenarios. Its effectiveness with VAE-CP has not been verified.

    ```shell
    --layerzero \
    --layerzero-config ./zero_config.yaml \
    ```

    The yaml file for the parameters is shown below:

    ```yaml
    zero3_size: 8
    transformer_layers:
    - mindspeed_mm.models.predictor.dits.sat_dit.VideoDiTBlock
    backward_prefetch: 'BACKWARD_PRE'
    param_dtype: "bf16"
    reduce_dtype: "fp32"
    forward_prefetch: True
    limit_all_gathers: True
    ignored_modules:
    - ae
    - text_encoder
    ```

    This feature is incompatible with TP. When enabled, TP must be set to `1`. When training with this feature, the saved weights need to be post-processed using the following conversion script before they can be used for inference:

    ```bash
    # Modify the ascend-toolkit path according to the actual situation.
    source /usr/local/Ascend/cann/set_env.sh
    # Replace your_mindspeed_path and your_megatron_path with the paths of the previously downloaded MindSpeed and megatron, respectively.
    export PYTHONPATH=$PYTHONPATH:<your_mindspeed_path>
    export PYTHONPATH=$PYTHONPATH:<your_megatron_path>
    # cfg.source_path is the path for saving weights from LayerZeRO training, and cfg.target_path is the path of output weights in megatron format.
    mm-convert CogVideoConverter layerzero_to_mm \
        --cfg.source_path ./save_ckpt/cogvideo/ \
        --cfg.target_path ./save_ckpt/cogvideo_megatron_ckpt/
    ```

The `head_dim` field in the model parameter configuration file is set to `64` by default for the original model. Adjusting this field to `128` will be more Ascend-friendly.

In the startup script shell, you can modify the number of running cards (`NNODES` is the number of nodes, `GPUS_PER_NODE` is the number of cards per node; multiplying them gives the total number of running cards):

    ```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=localhost
    MASTER_PORT=29501
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
    ```

### Launch Pre-training

t2v 1.0:

```shell
bash examples/cogvideox/t2v_1.0/pretrain_cogvideox_t2v.sh
```

t2v 1.5:

```shell
bash examples/cogvideox/t2v_1.5/pretrain_cogvideox_t2v_1.5.sh
```

i2v 1.0:

```shell
bash examples/cogvideox/i2v_1.0/pretrain_cogvideox_i2v.sh
```

i2v 1.5:

```shell
bash examples/cogvideox/i2v_1.5/pretrain_cogvideox_i2v_1.5.sh
```

## Inference

### Preparation

Before starting, please confirm that the environment setup and model weight download are complete.

### Parameter Configuration

The CogVideoX inference startup files are shell scripts, mainly divided into the following 4 types:

|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 |  inference_cogvideox_i2v.sh |inference_cogvideox_t2v.sh  |
| 1.5 | inference_cogvideox_i2v_1.5.sh |inference_cogvideox_t2v_1.5.sh |

The configuration files for model parameters are as follows:

|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 |  inference_model_i2v.json |inference_model_t2v.json  |
| 1.5 | inference_model_i2v_1.5.json |inference_model_t2v_1.5.json |

1. Weight Configuration

    You need to add the actual path of the converted weights to the `LOAD_PATH="your_converted_dit_ckpt_dir"` variable in the startup script file (e.g., `inference_cogvideox_i2v.sh`) according to the actual task situation, such as `LOAD_PATH="./CogVideoX-5B-Converted"`, where `./CogVideoX-5B-Converted` is the actual path of the converted weights, and its folder content structure is shown in the "Weight Conversion" section. The full path filled in the `LOAD_PATH` variable must be correct; an incorrect path will cause the weights to fail to load without triggering an error message during execution.

2. VAE and T5 Model Path Configuration

    Modify the actual paths of the VAE and T5 model files in the model parameter configuration file (e.g., `inference_model_i2v.json`) according to the actual situation. The path field for the T5 file is `"from_pretrained": "5b-cogvideo"` (replace `5b-cogvideo` with the actual path). The path field for the VAE model file is `"from_pretrained": "3d-vae.pt"` (replace `3d-vae.pt` with the actual path).

3. Prompt Configuration

    | t2v Prompt Configuration File                               |               Modification Field               |                Modification Notes                 |
    |----------------------------------------|:--------------------------------:|:-----------------------------------:|
    | examples/cogvideox/samples_prompts.txt |               File content               |      Custom prompt      |

    | i2v Prompt Configuration File                                   |               Modification Field               |       Modification Notes       |
    |--------------------------------------------|:--------------------------------:|:----------------:|
    | examples/cogvideox/samples_i2v_images.txt  |               File content               |       Image path       |
    | examples/cogvideox/samples_i2v_prompts.txt |               File content               |    Custom prompt     |

If you use the weights saved after training to change the model partitioning strategy for inference, you need to use the command for conversion. The `source_path` parameter for weight conversion should be configured to the save path used during training.

```bash
mm-convert CogVideoConverter --version <t2v or i2v> resplit \
  --cfg.source_path <your source path> \
  --cfg.target_path <your target path> \
  --cfg.target_parallel_config.tp_size <tp_size> \
  --cfg.target_parallel_config.pp_layers <pp_layers>
```

### Launch Inference

t2v 1.0:

```bash
bash examples/cogvideox/t2v_1.0/inference_cogvideox_t2v.sh
```

t2v 1.5:

```bash
bash examples/cogvideox/t2v_1.5/inference_cogvideox_t2v_1.5.sh
```

i2v 1.0:

```bash
bash examples/cogvideox/i2v_1.0/inference_cogvideox_i2v.sh
```

i2v 1.5:

```bash
bash examples/cogvideox/i2v_1.5/inference_cogvideox_i2v_1.5.sh
```

## LoRA Fine-tuning

### Preparation

Before configuring the script, ensure that the environment preparation is complete.

1. Weight Download and Conversion

    Model weight download links (containing model weights, tokenizer, and text_encoder):

    + [t2v](https://huggingface.co/THUDM/CogVideoX1.5-5B/tree/main)
    + [i2v](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V/tree/main)

    Weight conversion for LoRA fine-tuning requires the `mm-convert` command:

    ```bash
    mm-convert CogVideoConverter --version <t2v or i2v> hf_to_mm \
      --cfg.source_path <your source path> \
      --cfg.target_path <target path>
    ```

    VAE Weight Download

    + [VAE Download Link](https://huggingface.co/zai-org/CogVideoX1.5-5B-SAT/tree/main/vae)

2. Dataset Preparation and Processing

    [LoRA Dataset Download Link](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset)

    The original dataset does not include the `data.jsonl` file format required by the MM suite. You need to merge the `prompt.txt` and `videos.txt` files from the original dataset to generate the `data.jsonl` file.

    It is recommended to use the provided `cogvideox_lora_dataset_convert.py` script to complete the conversion:

    ```bash
    python examples/cogvideox/cogvideox_lora_dataset_convert.py --video_path '/data_path/videos.txt' --prompt_path '/data_path/prompt.txt' --output_path '/data_path/data.jsonl'
    ```

### Parameter Configuration

The startup files for the CogVideoX LoRA fine-tuning phase are shell scripts, which are mainly divided into the following two:

|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.5 | finetune_cogvideox_lora_i2v_1.5.sh |finetune_cogvideox_lora_t2v_1.5.sh |

The configuration files for model parameters are as follows:

|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.5 | model_cogvideox_i2v_1.5.json |model_cogvideox_t2v_1.5.json |

The `data.json` file related to the training dataset is also required.

The default configuration has been tested, and you can modify the following content according to your own environment:

1. Weight Configuration

    After the weight conversion is complete, add the actual path of the converted weights to the `LOAD_PATH="your_converted_dit_ckpt_dir"` variable in the startup script file (e.g., `finetune_cogvideox_lora_i2v_1.5.sh`) according to the actual task situation, such as `LOAD_PATH="./CogVideoX-5B-Converted"`, where `./CogVideoX-5B-Converted` is the actual path of the converted weights, and its folder content structure is shown in the "Weight Conversion" section. The full path filled in the `LOAD_PATH` variable must be correct; an incorrect path will cause the weights to fail to load without triggering an error message during execution.
    Fill in the path in the `SAVE_PATH` variable as needed to save the trained LoRA weights.

2. Dataset Path Configuration

    After preparing the dataset, modify the dataset paths in `data.json`, that is, `"data_path":"/data_path/data.jsonl"` and `"data_folder":"/data_path/"` as required. Replace `"/data_path/"` with the actual dataset path.

3. VAE and T5 Model Path Configuration

    Refer to the pre-training section.

4. Partitioning Strategy Configuration

    Refer to the pre-training section.

### Launch LoRA Fine-tuning

t2v 1.5:

```shell
bash examples/cogvideox/t2v_1.5/finetune_cogvideox_lora_t2v_1.5.sh
```

i2v 1.5:

```shell
bash examples/cogvideox/i2v_1.5/finetune_cogvideox_lora_i2v_1.5.sh
```

After training is completed, the saved weights are only the LoRA fine-tuning part. If you need to merge them into the original weights, execute the following script (the configuration is for reference only):

```bash
mm-convert CogVideoConverter --version <t2v or i2v> merge_lora_to_base \
  --cfg.source_path './converted_transformer' \
  --cfg.lora_path "./my_ckpt" \
  --cfg.target_path "./merge_base_lora_target" \
  --lora_rank 128 \
  --lora_alpha 64
```

## Pre-trained Model Parameter Modification Example (15B)

### Model Parameter Modification

The 15B parameter scale can be simulated by modifying configurations such as increasing the number of diffusion model layers. As shown below, modify the values of `"num_layers"`, `"num_heads"`, and `"head_dim"` under `"predictor"` in the model parameter configuration file (`model_cogvideox_i2v.json`).

```shell
"predictor": {
    "num_layers": 64,
    "num_heads": 32,
    "head_dim": 128,
    ...
}
```

### Startup Script Modification

Modify the `GPT_ARGS` parameters as follows, and adjust the distributed configuration in the startup script according to the actual resolution and number of frames (CP4 for single-node 16-card yields better results):

```shell
GPUS_PER_NODE=16
MASTER_ADDR={Master node IP}
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

TP=4
PP=1
CP=4
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

GPT_ARGS="
    --num-attention-heads 64 \
    --seq-length 32 \
    --max-position-embeddings 32 \
    --recompute-num-layers 64 \
    ...
"
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
