# CogVideoX User Guide (MindSpore as Backend)

<p align="left">
</p>

## Contents

- [CogVideoX User Guide (MindSpore as Backend)](#cogvideox-user-guide-mindspore-as-backend)
  - [Contents](#contents)
  - [Supported Task List](#supported-task-list)
  - [Environment Installation](#environment-installation)
    - [Repository Cloning and Environment Setup](#repository-cloning-and-environment-setup)
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
    - [Start Pre-training](#start-pre-training)
  - [Pre-trained Model Parameter Expansion Example (15B)](#pre-trained-model-parameter-expansion-example-15b)
    - [Model Parameter Modification](#model-parameter-modification)
    - [Startup Script Modification](#startup-script-modification)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Supported Task List

The following model tasks are supported.

|      Model      | Task Type | Task List | Supported |
|:------------:|:----:|:----:|:-----:|
| CogVideoX-5B | t2v  |Pre-training  | ✔ |
| CogVideoX-5B | i2v  |Pre-training  | ✔ |

## Environment Installation

The dependency compatibility for the MindSpeed MM MindSpore Backend is shown in the table below. For installation steps, refer to the [Installation Guide](../../../docs/en/mindspore/install_guide.md).

| Dependency         |                                                              |
| ---------------- | ------------------------------------------------------------ |
| Ascend NPU Driver & Firmware  | In-development |
| Ascend CANN        | In-development |
| MindSpore        | [2.7.0](https://www.mindspore.cn/install/en)         |
| Python           | >=3.9                                                        |

### Repository Cloning and Environment Setup

For the MindSpeed MindSpore Backend, the Ascend community provides a one-click conversion tool, MindSpeed-Core-MS, designed to help users automatically pull the relevant code repositories and perform one-click adaptation of torch code, thereby enabling users to start model training with one click in the Huawei MindSpore + CANN environment without additional manual adaptation. Before performing the one-click conversion, users need to pull the relevant code repositories and set up the environment.

```shell
# Create a conda environment.
conda create -n test python=3.10
conda activate test

# Use environment variables.
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# Install MindSpeed-Core-MS.
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.4.0

# Use the internal scripts of MindSpeed-Core-MS to automatically pull the relevant code repositories, perform one-click adaptation, and provide the configured environment.
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm

# Replace files in MindSpeed.
cd MindSpeed-MM
cp examples/cogvideox/dot_product_attention.py ../MindSpeed/mindspeed/core/transformer/dot_product_attention.py
mkdir ckpt
mkdir data
mkdir logs
```

### Decord Setup

X86:

```bash
pip install decord==0.6.0
```

Arm:

For installation via `apt`, please [refer to the link](https://github.com/dmlc/decord).

For installation via `yum`, please [refer to the script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

## Weight Download and Conversion

### VAE Download

+ [VAE Download Link](https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1)

### Transformer File Download

+ [CogVideoX1.0-5B-t2v](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX1.0-5B-i2v](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)
+ [CogVideoX1.5-5B-t2v](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main/transformer_t2v)
+ [CogVideoX1.5-5B-i2v](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main/transformer_i2v)

### T5 Model Download

Only the [contents](https://huggingface.co/THUDM/CogVideoX-5b/tree/main) of the `tokenizer` and `text_encoder` directories need to be downloaded.

The structure of the pre-training weights is as follows:

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

Set the `source_path` parameter to the transformer weight file.

```bash
python examples/cogvideox/cogvideox_sat_convert_to_mm_ckpt.py \
    --source_path <your source path> \
    --target_path <target path> \
    --task t2v \
    --tp_size 1 \
    --pp_size 10 11 11 10 \
    --num_layers 42 \
    --mode split
```

Here, the value after `--tp_size` is the actual TP strategy, and the value of `--task` is `t2v` or `i2v`.
When PP is enabled, the number after `--pp_size` equals the `PP` value, and the sum of these parameters equals the `--num_layers` parameter. For example: when `PP=4` and `--num_layers=4`, then `--pp_size=1 1 1 1`; when `PP=4` and `--num_layers=42`, then `--pp_size=10 11 11 10`.

The structure of the converted weights is as follows:

`TP=1` and `PP=1`:

```shell
CogVideoX-5B-Converted
├── release
│   └──mp_rank_00
│      └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```

`TP=2` and `PP=1` (situation for `TP>2` follows this pattern):

```shell
CogVideoX-5B-Converted
├── release
│   ├──mp_rank_00
│   │    └──model_optim_rng.pt
│   └──mp_rank_01
│      └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```

`TP=1` and `PP=4` (situation for `PP>1` and `TP>1` follows this pattern):

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

<a id="jump4"></a>

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

Each txt file shares the same name as its corresponding video and serves as the video's label. Videos and labels should have a one-to-one correspondence.

Example of the content of the `data.jsonl` file:

```shell
{"file": "dataPath/1.mp4", "captions": "Content from 1.txt"}
{...}
...
```

## Pre-training

### Preparation

Before configuring the script, you need to complete the preliminary preparations, including: [Environment Installation](#environment-installation), [Weight Download and Conversion](#weight-download-and-conversion), and [Dataset Preparation and Processing](#dataset-preparation-and-processing).

### Parameter Configuration

The startup files for the CogVideoX training phase are shell scripts, including the following 4 scripts:

| Version | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 | pretrain_cogvideox_i2v.sh | pretrain_cogvideox_t2v.sh |
| 1.5 | pretrain_cogvideox_i2v_1.5.sh | pretrain_cogvideox_t2v_1.5.sh |

The configuration files for model parameters are as follows:

| Version | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 | model_cogvideox_i2v.json | model_cogvideox_t2v.json |
| 1.5 | model_cogvideox_i2v_1.5.json | model_cogvideox_t2v_1.5.json |

The `data.json` file related to the training dataset is also required.

The default configuration has been tested, and you can modify the following content according to your own environment.

1. Weight Configuration

    According to the actual task, you need to add the actual path of the converted weights in the `LOAD_PATH="your_converted_dit_ckpt_dir"` variable within the startup script file (e.g., `pretrain_cogvideox_i2v.sh`), such as `LOAD_PATH="./CogVideoX-5B-Converted"`, where `./CogVideoX-5B-Converted` is the actual path of the converted weights. Its folder content structure is shown in the "Weight Conversion" section. The full path filled in the `LOAD_PATH` variable must be correct; an incorrect path will cause the weights to fail to load without triggering an error message during execution.

    Fill in the path in the `SAVE_PATH` variable as needed to save the trained weights.

2. Dataset Path Configuration

    Modify the dataset path in `data.json` according to the actual situation, specifically `"data_path":"/your_data_path/data.jsonl"` and `"data_folder":"/your_data_path/"`, replacing `"/your_data_path/"` with the actual dataset path.

3. VAE and T5 Model Path Configuration

    Modify the actual paths of the VAE and T5 model files in the model parameter configuration file (e.g., `model_cogvideox_i2v.json`) and the `data.json` file according to the actual situation. The path fields for the T5 file is `"from_pretrained": "5b-cogvideo/tokenizer"` and `"from_pretrained": "5b-cogvideo"`, and replace `5b-cogvideo` with the actual path. The path field for the VAE model file is `"from_pretrained": "3d-vae.pt"`, and replace `3d-vae.pt` with the actual path.

    When VAE and T5 need to be offloaded, change the values of the `"load_video_features": false` and `"load_text_features": false` fields in the model parameter configuration file to `true` respectively. Also, change the value of the `"use_feature_data"` field in `data.json` to `true`.

4. Partitioning Strategy Configuration

* When PP is enabled, add the `--optimization-level 2 --use-multiparameter-pipeline-model-parallel` parameters to the startup script, and change the value of the `pipeline_num_layers` parameter in the model parameter configuration file from `null` to the actual value. For example, when `PP=4` and `num_layers=42`, then `"pipeline_num_layers":[11, 10, 11, 10]`.The specific values are determined based on the actual PP partitioning strategy.

* When VAE CP is enabled, modify the value of the `cp_size` key within the `ae` dictionary in the model parameter configuration file to the desired value. It is incompatible with Encoder-DP, and its effect with hierarchical ZeRO has not been verified.

* When SP is enabled, add the `--sequence-parallel` parameter to the startup script.

* When Encoder-DP is enabled, change the `enable_encoder_dp` option in [model_cogvideox_i2v_1.5.json](i2v_1.5/model_cogvideox_i2v_1.5.json) or [model_cogvideox_t2v_1.5.json](t2v_1.5/model_cogvideox_t2v_1.5.json) to `true`.
  * Note: This can only be enabled when CP/TP is enabled, and `load_video_features` is `false` and `load_text_features` is `false`. It is incompatible with PP scenarios, VAE-CP, and hierarchical ZeRO.

* When hierarchical ZeRO is enabled, add the following parameters to [pretrain_cogvideox_t2v_1.5.sh](t2v_1.5/pretrain_cogvideox_t2v_1.5.sh) or [pretrain_cogvideox_i2v_1.5.sh](i2v_1.5/pretrain_cogvideox_i2v_1.5.sh).
  * Note: It is incompatible with Encoder-DP feature, TP scenario, and PP scenario. The effect with VAE-CP has not been verified.

  ```shell
  --layerzero \
  --layerzero-config ./zero_config.yaml \
  ```

The YAML file is shown below:

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

This feature is incompatible with TP. TP must be set to 1 when this feature is enabled. When training with this feature, the saved weights need to be post-processed using the following conversion script before they can be used for inference.

    ```bash
    # Modify the ascend-toolkit path according to the actual situation.
    source /usr/local/Ascend/cann/set_env.sh
    # Replace `your_mindspeed_path` and `your_megatron_path` with the specific paths of the MindSpeed and megatron repositories pulled earlier via the MindSpeed-Core-MS one-click script. These two paths are usually located in the corresponding subdirectories of the MindSpeed-Core-MS directory.
    export PYTHONPATH=$PYTHONPATH:<your_mindspeed_path>
    export PYTHONPATH=$PYTHONPATH:<your_megatron_path>
    # `input_folder` is the path where weights are saved during LayerZero training, and `output_folder` is the path to the output Megatron-format weights.
    python <your_mindspeed_path>/mindspeed/core/distributed/layerzero/state/scripts/convert_to_megatron.py --input_folder ./save_ckpt/hunyuanvideo/iter_000xxxx/ --output_folder ./save_ckpt/hunyuanvideo_megatron_ckpt/iter_000xxxx/ --prefix predictor
    ```

The `head_dim` field in the model parameter configuration file defaults to `64` in the original model. Adjusting this field to `128` will be more Ascend-friendly.

In the startup shell script, you can modify the number of running cards (`NNODES` is the number of nodes, `GPUS_PER_NODE` is the number of cards per node; their product is the total number of running cards):

```shell
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

### Start Pre-training

t2v 1.0:

```shell
bash examples/mindspore/cogvideox/t2v_1.0/pretrain_cogvideox_t2v.sh
```

t2v 1.5:

```shell
bash examples/mindspore/cogvideox/t2v_1.5/pretrain_cogvideox_t2v_1.5.sh
```

i2v 1.0:

```shell
bash examples/mindspore/cogvideox/i2v_1.0/pretrain_cogvideox_i2v.sh
```

i2v 1.5:

```shell
bash examples/mindspore/cogvideox/i2v_1.5/pretrain_cogvideox_i2v_1.5.sh
```

## Pre-trained Model Parameter Expansion Example (15B)

### Model Parameter Modification

By modifying configurations such as the number of diffusion model layers, a 15B parameter count can be simulated. As shown below, modify the values of `"num_layers"`, `"num_heads"`, and `"head_dim"` under `"predictor"` in the model parameter configuration file (`model_cogvideox_i2v.json`).

```shell
"predictor": {
    "num_layers": 64,
    "num_heads": 32,
    "head_dim": 128,
    ...
}
```

### Startup Script Modification

Modify the `GPT_ARGS` parameters as follows, and adjust the distributed configuration in the startup script according to the actual resolution and frame count (CP4 on a single server with16 cards yields better results):

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

`ASCEND_RT_VISIBLE_DEVICES`: Specifies the index value of the NPU device.

`ASCEND_SLOG_PRINT_TO_STDOUT`: Controls whether log printing is enabled. `0` disables log printing, and `1` enables log printing.

`ASCEND_GLOBAL_LOG_LEVEL`: Sets the log level for application logs and module logs supporting only debug logs.
`0` corresponds to the DEBUG level, `1` to the INFO level, `2` to the WARNING level, `3` to the ERROR level, and `4` to the NULL level, with no log output.

`HCCL_CONNECT_TIMEOUT`: Limits the timeout waiting time for socket connection establishment between different devices. It must be configured as an integer in the value range [120, 7200]. The default value is 120, in seconds.

`HCCL_EXEC_TIMEOUT`: Controls the synchronous waiting time during execution between devices. Within this configured time, each device process waits for other devices to execute communication synchronization.

`ASCEND_LAUNCH_BLOCKING`: Controls whether to enable synchronous mode during operator execution. `0` executes operators in asynchronous mode, and `1` forces operators to run in synchronous mode.

`MS_DEV_HOST_BLOCKING_RUN`: Controls whether dynamic graph operators are dispatched in a single thread. `0` indicates multi-threaded dispatch, and `1` indicates single-threaded dispatch.

`MS_DEV_LAUNCH_BLOCKING`: Controls whether operators are dispatched synchronously. `0` indicates asynchronous dispatch, and `1` indicates single-threaded dispatch with stream synchronization.

`ACLNN_CACHE_LIMIT`: Configures the number of operator information entries cached on the host side for single-operator execution APIs.

`TOKENIZERS_PARALLELISM`: Controls the behavior of the tokenizer in Hugging Face's transformers library in a multi-threaded environment.

`NPUS_PER_NODE`: Configures the number of NPUs used on a single compute node.
