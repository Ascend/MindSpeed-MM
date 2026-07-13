# StepVideo User Guide

<p align="left">
</p>

## Contents

- [StepVideo User Guide](#stepvideo-user-guide)
  - [Contents](#contents)
  - [Supported Task List](#supported-task-list)
  - [Environment Installation](#environment-installation)
    - [Repository Cloning](#repository-cloning)
      - [Environment Setup](#environment-setup)
      - [Decord Setup](#decord-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [Weight Download](#weight-download)
    - [Weight Conversion](#weight-conversion)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
  - [Pre-training](#pre-training)
    - [Preparation](#preparation)
    - [Feature Extraction](#feature-extraction)
    - [Parameter Configuration](#parameter-configuration)
    - [Start Pre-training](#start-pre-training)
  - [Inference](#inference)
    - [Preparation](#preparation-1)
    - [Parameter Configuration](#parameter-configuration-1)
    - [Start Inference](#start-inference)
  - [DPO Training](#dpo-training)
    - [Environment Preparation](#environment-preparation)
    - [Video Sample Generation](#video-sample-generation)
    - [Generate Preference Dataset](#generate-preference-dataset)
    - [Training Parameter Configuration](#training-parameter-configuration)
    - [Start DPO Training](#start-dpo-training)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Supported Task List

The following model task types are supported.

|    Model    | Task Type | Task List | Supported |
|:---------:|:----:|:----:|:-----:|
| StepVideo | t2v  |Pre-training  | ✔ |
| StepVideo | t2v  |Online Inference | ✔ |
| StepVideo | i2v  |Pre-training  | ✔ |
| StepVideo | i2v  |Online Inference | ✔ |

## Environment Installation

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

#### Environment Setup

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# Install torch and torch_npu. Ensure you select the torch, torch_npu, and apex packages that correspond to your Python version and architecture (x86 or arm).
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
git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
pip install -r requirements.txt
pip install -e .
cd ..

# Install other dependency libraries.
pip install -e .
```

#### Decord Setup

[X86 Installation]

```bash
pip install decord==0.6.0
```

[ARM Installation]

For installation via `apt`, please [refer to the link](https://github.com/dmlc/decord).

For installation via `yum`, please [refer to the script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

## Weight Download and Conversion

### Weight Download

[StepVideo-t2v](https://hf-mirror.com/stepfun-ai/stepvideo-t2v/tree/main): requires downloading VAE, transformer, text_encoder, and tokenizer.

[StepVideo-i2v](https://hf-mirror.com/stepfun-ai/stepvideo-ti2v/tree/main): requires downloading VAE, transformer, text_encoder, and tokenizer.

The pre-training/inference weight structure is as follows:

```shell
   stepvideo-ti2v/t2v
   ├── hunyuan_clip
   │   ├── clip_text_encoder
   │   │   ├── config.json
   │   │   └── pytorch_model.bin
   │   ├── tokenizer
   │   │   ├── special_tokens_map.json
   │   │   ├── tokenizer_config.json
   │   │   ├── vocab.txt
   │   │   └── vocab_org.txt
   ├── step_llm
   │   ├── config.json
   │   ├── model-00001-of-00009.safetensors
   │   ├── model-00002-of-00009.safetensors
   │   ├── ...
   │   ├── model-00009-of-00009.safetensors
   │   ├── model.safetensors.index.json
   │   └── step1_chat_tokenizer.model
   ├── transformer
   │   ├── config.json
   │   ├── diffusion_pytorch_model-00001-of-00006.safetensors
   │   ├── diffusion_pytorch_model-00002-of-00006.safetensors
   │   ├── ...
   │   ├── diffusion_pytorch_model-00006-of-00006.safetensors
   │   └── diffusion_pytorch_model.safetensors.index.json
   └── vae
       ├── vae.safetensors
       └── vae_v2.safetensors
```

### Weight Conversion

For the weight conversion `source_path` parameter, please configure the path to the transformer weight file:

```bash
mm-convert StepVideoConverter hf_to_mm \
  --cfg.source_path <your source path> \
  --cfg.target_path <your target path> \
  --cfg.target_parallel_config.tp_size <tp_size> \
  --cfg.target_parallel_config.pp_layers <pp_layers>
```

`tp_size` represents the number of TP partitions, and `pp_layers` represents the number of layers per stage after PP partitioning. For example, `48` means partitioning is not performed, and `[24,24]` means `PP=2`, with 24 layers per PP stage.

The converted weight structure is as follows:

When `TP=1`, `PP=1`:

```shell
StepVideo-Converted
├── release
│   └──mp_rank_00
│      └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```

When `TP=2`, `PP=1` (situation for `TP>2` follows this pattern):

```shell
StepVideo-Converted
├── release
│   ├──mp_rank_00
│   │    └──model_optim_rng.pt
│   └──mp_rank_01
│      └──model_optim_rng.pt
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

Example content of the `data.jsonl` file:

```shell
{"file": "dataPath/1.mp4", "captions": "Content from 1.txt"}
{...}
...
```

## Pre-training

### Preparation

Before configuring the script, you need to complete the preliminary preparations, including: **environment installation**, **weight download and conversion**, and **dataset preparation and processing**. For details, please refer to the corresponding chapters.

### Feature Extraction

1. Configure feature extraction parameters.

    Check whether configurations such as the model weight path, dataset path, and save path of extracted features are complete.

    | T2V Configuration File                                                     |       Field to be Modified        |  Instructions                                            |
    | ------------------------------------------------------------ | :-------------------: | :-------------------------------------------------- |
    | examples/stepvideo/feature_extract/data.json              |      basic_parameters   | Dataset path. Set `data_path` and `data_folder` to the file path and directory of `data.jsonl`, respectively. |
    | examples/stepvideo/feature_extract/data.json              |      num_frames        | Maximum number of frames. If exceeded, `num_frames` frames are randomly selected. Set the value to 102 for i2v and 136 for t2v. |
    | examples/stepvideo/feature_extract/data.json              |      tokenizer_config  | Tokenizer selection. Configure two tokenizer paths: `"from_pretrained": "/model_path/step_llm/step1_chat_tokenizer.model"` and `"from_pretrained": "/model_path/hunyuan_clip/tokenizer"`. |
    | examples/stepvideo/feature_extract/model_stepvideo.json   |      text_encoder    | Configure two text encoder paths: `"from_pretrained": "./weights/step_llm/"` and `"from_pretrained": "./weights/hunyuan_clip/clip_text_encoder"`. |
    | examples/stepvideo/feature_extract/model_stepvideo.json   |      ae              | Configure the VAE model path: `"from_pretrained": "./weights/vae/vae_v2.safetensors"` |
    | mindspeed_mm/tools/tools.json                             |      save_path       | Save path of extracted features                                |

    | I2V Configuration File                                                     |        Field to be Modified        |  Instructions                                            |
    | ------------------------------------------------------------ | :-------------------: | :-------------------------------------------------- |
    | examples/stepvideo/feature_extract/data_i2v.json              |      basic_parameters   | Dataset path. Set `data_path` and `data_folder` to the file path and directory of `data.jsonl`, respectively. |
    | examples/stepvideo/feature_extract/data_i2v.json              |      num_frames        | aximum number of frames. If exceeded, `num_frames` frames are randomly selected. Set the value to 102 for i2v and 136 for t2v. |
    | examples/stepvideo/feature_extract/data_i2v.json              |      tokenizer_config  | Tokenizer selection. Configure two tokenizer paths: `"from_pretrained": "/model_path/step_llm/step1_chat_tokenizer.model"` and `"from_pretrained": "/model_path/hunyuan_clip/tokenizer"` |
    | examples/stepvideo/feature_extract/model_stepvideo_i2v.json   |      text_encoder    | Configure two text encoder paths: `"from_pretrained": "./weights/step_llm/"` and `"from_pretrained": "./weights/hunyuan_clip/clip_text_encoder"`. |
    | examples/stepvideo/feature_extract/model_stepvideo_i2v.json   |      ae              | Configure the VAE model path: `"from_pretrained": "./weights/vae/vae_v2.safetensors"` |
    | mindspeed_mm/tools/tools.json                                 |      save_path       | Save path of extracted features                                |

2. Extract features.

    T2V execution command:

    ```bash
    bash examples/stepvideo/feature_extract/feature_extraction.sh
    ```

    I2V execution command:

    ```bash
    bash examples/stepvideo/feature_extract/feature_extraction_i2v.sh
    ```

### Parameter Configuration

The startup files for the StepVideo training phase are shell scripts, mainly divided into the following two types:

| I2V | T2V |
|:----:|:----:|
| pretrain_i2v.sh |pretrain_t2v.sh  |

The configuration files for model parameters are as follows:

| I2V | T2V |
|:----:|:----:|
| pretrain_i2v_model.json |pretrain_t2v_model.json  |

Also, the `data_static_resolution.json` file related to the training dataset is required.

The default configuration has been tested. You can modify the following content according to your own environment:

| Configuration File                                                   |      Field to be Modified      |  Instructions                                            |
| ---------------------------------------------------------- | :-----------------: | :-------------------------------------------------- |
| examples/stepvideo/{task_name}/data_static_resolution.json |  basic_parameters  | Dataset path. Set `data_path` and `data_folder` to the file path and directory of `data.jsonl`, respectively. |
| examples/stepvideo/{task_name}/pretrain_*.sh        |    NPUS_PER_NODE    | Number of cards per node                                      |
| examples/stepvideo/{task_name}/pretrain_*.sh        |       NNODES        | Number of nodes                                            |
| examples/stepvideo/{task_name}/pretrain_*.sh        |      LOAD_PATH      | Path to the pre-training weights after weight conversion                          |
| examples/stepvideo/{task_name}/pretrain_*.sh        |      SAVE_PATH      | Path to weights saved during training                            |
| examples/stepvideo/{task_name}/pretrain_*.sh        |         TP          | TP size during training (recommended to adjust it according to the resolution set during training)   |
| examples/stepvideo/{task_name}/pretrain_*.sh        |         CP          | CP size during training (recommended to adjust it according to the resolution set during training)   |

[Parallel Parameter Configuration]

When adjusting model parameters or video sequence length, you need to enable the following parallel strategies based on the actual situation and determine the optimal parallel strategy through debugging.

+ CP: Sequence parallelism; currently supports Ulysses.

  - Use Case: When the video sequence (resolution × number of frames) is large, it can be enabled to reduce memory usage.

  - How to Enable: Set CP to a value greater than 1 in the startup script, e.g., `CP=2`;

  - Limitation: The number of `num_attention_heads` must be divisible by TP*CP (configured in `examples/stepvideo/{task_name}/pretrain_xx_model.json`; defaulted to 48).

+ TP: Tensor Parallelism

  - Use Case: When the model parameter scale is large and a single card cannot accommodate the complete model, enabling TP can reduce static memory and runtime memory.

  - How to Enable: Set TP to a value greater than 1 in the startup script, e.g., `TP=8`.

  - Limitation: The number of `num_attention_heads` must be divisible by TP*CP (configured in `examples/stepvideo/{task_name}/pretrain_xx_model.json`; defaulted to 48).

### Start Pre-training

T2V:

```shell
bash examples/stepvideo/t2v/pretrain_t2v.sh
```

I2V:

```shell
bash examples/stepvideo/i2v/pretrain_i2v.sh
```

## Inference

### Preparation

Before starting, please confirm that the environment preparation and model weight download have been completed.

### Parameter Configuration

The StepVideo inference startup files are shell scripts, mainly divided into the following two:

| I2V | T2V |
|:----:|:----:|
| inference_i2v.sh |inference_t2v.sh |

The configuration files for model parameters are as follows:

| I2V | T2V |
|:----:|:----:|
| inference_i2v_model.json | inference_t2v_model.json  |

1. Weight Configuration

    Based on the actual task, add the actual path of the converted weights to the `LOAD_PATH="your_converted_dit_ckpt_dir"` variable in the startup script file (e.g., `inference_i2v.sh`). Note that the default inference configuration is `tp=4`. For example, `LOAD_PATH="./StepVideo-Converted"`, where `./StepVideo-Converted` is the actual path of the converted weights, and its folder content structure is shown in the "Weight Conversion" section. The full path filled in the `LOAD_PATH` variable must be correct; an incorrect path will cause the weights to fail to load without triggering an error message during execution.

2. VAE and T5 Model Path Configuration

    Modify the model parameter configuration file according to the actual situation. For example, in the `inference_i2v_model.json` file, configure two text encoder paths in the `text_encoder` field: `"from_pretrained": "./weights/step_llm/"` and `"from_pretrained": "./weights/hunyuan_clip/clip_text_encoder"`, and configure the VAE model path in the `ae` field: `"from_pretrained": "./weights/vae/vae_v2.safetensors"`.

    Configure the two tokenizer paths in the `tokenizer` field: `"from_pretrained": "/model_path/step_llm/step1_chat_tokenizer.model"`, and `"from_pretrained": "/model_path/hunyuan_clip/tokenizer"`.

3. Prompt Configuration

    | T2V Prompt Configuration File                      |               Field to be Modified               |                Instructions                 |
    |----------------------------------------|:--------------------------------:|:-----------------------------------:|
    | examples/stepvideo/t2v/samples_prompts.txt |               File content            |      Custom prompt      |

    | I2V Prompt Configuration File                                   |               Field to be Modified                |       Instructions       |
    |--------------------------------------------|:--------------------------------:|:----------------:|
    | examples/stepvideo/i2v/samples_i2v_images.txt  |               File content               |       Image path       |
    | examples/stepvideo/i2v/samples_i2v_prompts.txt |               File content               |    Custom prompt     |

    If you use the weights saved after training for inference, you need to use a script for conversion. For the weight conversion `source_path` parameter, please configure the save path used during training.

    ```bash
    mm-convert StepVideoConverter resplit \
    --cfg.source_path <your source path> \
    --cfg.target_path <your target path> \
    ```

### Start Inference

T2V:

```bash
bash examples/stepvideo/t2v/inference_t2v.sh
```

I2V:

```bash
bash examples/stepvideo/i2v/inference_i2v.sh
```

## DPO Training

Currently, only basic DPO training with T2V puncturing is supported, with more features to be refined later.

### Environment Preparation

1. Install VBench and its dependencies by referring to [VBench Evaluation](../../docs/zh/features/vbench-evaluate.md).
2. Download the [VBench T2V json](https://github.com/Vchitect/VBench/blob/master/vbench/VBench_full_info.json) to the MM code root path `"./vbench/VBench_full_info.json"`.

### Video Sample Generation

1. Modify the inference configuration file.

    | Parameter Configuration File                                                 |               Field to be Modified                |  Instructions                          |
    |------------------------------------------------------------|:--------------------------------:|:----------------------------------|
    | examples/stepvideo/{task_name}/inference_*_model.json      |         from_pretrained          | Modify it to the path corresponding to the downloaded weights (including VAE and text encoder). |
    | examples/stepvideo/{task_name}/inference_*_model.json      |  num_inference_videos_per_sample | Number of video samples generated per prompt. It is recommended that the value be at least greater than 2.         |
    | examples/stepvideo/{task_name}/inference_*_model.json        |  save_path | Save path for the generated videos                         |
    | examples/stepvideo/{task_name}/inference_*.sh              |   LOAD_PATH | Path to the converted transform weights              |

    | T2V Prompt Configuration File                      |                Field to be Modified              |                Instructions                 |
    |----------------------------------------|:--------------------------------:|:-----------------------------------:|
    | examples/stepvideo/t2v/samples_prompts.txt |               File content            |      Custom prompt      |

2. Start the inference process to generate video samples.

    ```shell
    bash examples/stepvideo/{task_name}/inference_{task_name}.sh
    ```

3. Delete `video_grid.mp4` in the video sample save path. The final number of video samples is `number of prompts * $num_inference_videos_per_sample`.

### Generate Preference Dataset

Execute the following command to score the generated video samples and generate a preference data file.

```bash
python examples/stepvideo/histogram_generator.py --prompt_file <prompt file path> --videos_path <video sample path> --num_inference_videos_per_sample <number of video samples generated per prompt>
```

The parameter descriptions for the preference dataset generation script are as follows:

| Parameter | Meaning | How to Configure |
|:------------|:----|:----|
| --prompt_file | Path to the prompt file | Make it consistent with `prompt` in the inference configuration file when generating video samples. |
| --videos_path | Path to the video samples | Make it consistent with `save_path` in the inference configuration file when generating video samples. |
| --num_inference_videos_per_sample | Number of video samples generated per prompt | Make it consistent with `num_inference_videos_per_sample` in the inference configuration file when generating video samples. |

After executing the script, a preference dataset file `data.jsonl` and a score probability histogram file `video_score_histogram.json` will be generated, defaulting to the same level as the video sample directory.

`data.jsonl` contains paired video preference data and text information, with a specific example as follows:

```json
[
    {
        "file": "video_0.mp4",
        "file_rejected": "video_2.mp4",
        "captions": "prompt1",
        "score": 0.646468401,
        "score_rejected": 0.5799660087
    },
    {
        "file": "video_4.mp4",
        "file_rejected": "video_5.mp4",
        "captions": "prompt2",
        "score": 0.7914018631,
        "score_rejected": 0.69968328357
    },
    ......
]
```

### Training Parameter Configuration

Before starting, please confirm that the environment preparation, model weight preparation, and preference data preparation have been completed.

1. Weight Configuration

    According to the actual task, add the actual path of the converted weights to the `LOAD_PATH="your_converted_dit_ckpt_dir"` variable in the startup script file (e.g., `posttrain_t2v_dpo.sh`), such as `LOAD_PATH="./StepVideo-Converted"`, where `./StepVideo-Converted` is the actual path of the converted weights, and its folder content structure is shown in the "Weight Conversion" section. The full path filled in the `LOAD_PATH` variable must be correct; an incorrect path will cause the weights to fail to load without triggering an error message.
    Fill in the path in the `SAVE_PATH` variable as needed to save the trained weights.

2. Preference Dataset Path Configuration

    Modify the preference dataset path in `data_dpo.json` according to the actual situation: replace `"data_path":"/data_path/data.jsonl"` with the actual path where `data.jsonl` is located, and replace `"data_folder":"/data_path/"` with the actual path where the video samples are located.

3. VAE, Text Encoder, and Tokenizer Path Configuration

    Modify the model parameter configuration file according to the actual situation. For example, in the `posttrain_*_model.json` file, configure the two text encoder paths in the `text_encoder` field: `"from_pretrained": "./weights/step_llm/"` and `"from_pretrained": "./weights/hunyuan_clip/clip_text_encoder"`. Configure the VAE model path in the `ae` field: `"from_pretrained": "./weights/vae/vae_v2.safetensors"`.
    In the `data_dpo.json` file, configure the two tokenizer paths in the `tokenizer_config` field: `"from_pretrained": "/model_path/step_llm/step1_chat_tokenizer.model"` and `"from_pretrained": "/model_path/hunyuan_clip/tokenizer"`.

4. DPO Parameter Configuration

    Modify the histogram file path in `posttrain_t2v_model.json` according to the actual situation, that is, configure the value of `histogram_path` to the path of the `video_score_histogram.json` file generated after executing the generation preference dataset script.

### Start DPO Training

```bash
bash examples/stepvideo/{task_name}/posttrain_*_dpo.sh
```

## Environment Variable Declaration

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
| `NPUS_PER_NODE`               | Configures the number of NPUs used on a compute node.                                                  | Integer value (e.g., `1`, `8`, etc.)                                                                            |
