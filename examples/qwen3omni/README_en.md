# Qwen3_Omni User Guide

<p align="left">
</p>

## Contents

- [Qwen3\_Omni User Guide](#qwen3_omni-user-guide)
  - [Contents](#contents)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Installation](#environment-installation)
    - [1. Environment Preparation](#1-environment-preparation)
    - [2. Environment Setup](#2-environment-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [1. Weight Download](#1-weight-download)
      - [Special Notes](#special-notes)
    - [2. Weight Conversion](#2-weight-conversion)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download (Using the COCO2017 Dataset as an Example)](#1-dataset-download-using-the-coco2017-dataset-as-an-example)
    - [2. Mixed Dataset Processing (Using LLaVA-Instruct-150K as an Example)](#2-mixed-dataset-processing-using-llava-instruct-150k-as-an-example)
  - [Fine-tuning](#fine-tuning)
    - [1. Preparation](#1-preparation)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
    - [4. Fine-tuning with Tool Call Data](#4-fine-tuning-with-tool-call-data)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Notes](#notes)

## Version Description

### Reference Implementation

```shell
url=https://github.com/huggingface/transformers.git
commit_id=7a833d1
```

### Changelog

2025.11.13: Initial support for Qwen3-Omni model

<a id="jump1"></a>

## Environment Installation

<a id="jump1.1"></a>

### 1. Environment Preparation

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md) to complete the Ascend software installation.
> Python 3.10 is recommended, and torch and torch_npu 2.7.1 are recommended.

The following versions are recommended:

- [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)
- [torch_npu](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/insg_0004.html)

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
git checkout d76dbddd

# Install MindSpeed and its dependencies.
pip install -e .
cd ..
# Install MindSpeed MM and its dependencies.
pip install -e .

# Install the latest version of transformers (supporting the qwen3omni model).
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 7a833d1
pip install -e .
pip install accelerate==1.11.0 librosa==0.11.0 datasets==4.0.0
```

<a id="jump2"></a>

## Weight Download and Conversion

<a id="jump2.1"></a>

### 1. Weight Download

Download the corresponding model weights from the Hugging Face library:

- Model address: [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/collections/Qwen/qwen3-omni)

Save the downloaded model weights to the local `ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct` directory.

#### Special Notes

After downloading the weights, modify the `ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct/config.json` file in the weight path, changing `enable_audio_output` from `true` to `false`.

<a id="jump2.2"></a>

### 2. Weight Conversion

When using multi-card fine-tuning, gradient communication issues may arise. MindSpeed MM has modified the MOE implementation in transformers, requiring the original pre-training weights to be converted:

```shell
mm-convert ExpertMergeDcpConverter hf_to_dcp \
  --hf_dir "ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct" \
  --save_dir "ckpt/convert_path/Qwen3-Omni-30B-A3B-Instruct"
```

And add the `--init-model-with-meta-device` parameter to `GPT_ARGS` in `examples/qwen3omni/finetune_qwen3omni.sh`.

After training is complete, the weights saved in the `SAVE_PATH` directory can be converted to HuggingFace format:

```shell
mm-convert ExpertMergeDcpConverter dcp_to_hf \
  --hf_dir "ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct" \
  --dcp_dir "save_dir/iter_000xx" \
  --save_dir "ckpt/dcp_to_hf/Qwen3-Omni-30B-A3B-Instruct"
```

`--hf_dir` specifies the path to the original Hugging Face weights, `--dcp_dir` specifies the save path for the fine-tuned weights (the `iter_000xx` in the path indicates the weights saved at step *xx*), and `--save_dir` specifies the save path for the converted HuggingFace-format weights.

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download (Using the COCO2017 Dataset as an Example)

(1) Download the [COCO2017 dataset](https://cocodataset.org/#download) and extract it to the `./data/COCO2017` folder under the project directory.

(2) Obtain the description file for the image dataset ([LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)) and download it to the `./data/` path.

(3) Run the data conversion script: `examples/qwen2vl/llava_instruct_2_mllm_demo_format.py`.

After conversion, the reference data directory structure is as follows:

   ```shell
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

Currently, it supports reading multiple datasets separated by `,` (do not add spaces). To do so, modify `dataset_param->basic_parameters->dataset` in `data.json`: Change `"./data/mllm_format_llava_instruct_data.json"` to `"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"`.

Also note the configuration of `dataset_param->basic_parameters->max_samples` in `data.json` limits the data reading to only `max_samples` entries, allowing for quick function verification. For formal training, you can remove this parameter to read all the data.

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

### 1. Preparation

Before configuring the script, complete the preliminary preparations, including: **Environment Installation**, **Weight Download and Conversion**, and **Dataset Preparation and Processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data.json` according to the actual situation, including fields such as `model_name_or_path`, `dataset_dir`, and `dataset`.

Example: If the data and its corresponding JSON are both in the `/home/user/data/` directory, and the JSON directory is `/home/user/data/video_data_path.json`, the configuration is as follows:
Set `dataset_dir` to `/home/user/data/`
and `dataset` to `./data/video_data_path.json`.
Note that `dataset` must be configured as a relative path in this case.

Taking Qwen3Omni as an example, make the following modifications to `data.json`. Note that the weight path for `model_name_or_path` is the weight path before conversion, i.e., the original Hugging Face weight path.

**Note: Do not configure the same mount directory for `cache_dir` across multiple machines to avoid conflicts caused by writing to the same file.**

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct",
            ...
        },
        "basic_parameters": {
            ...
            "dataset_dir": "./data",
            "dataset": "./data/mllm_format_llava_instruct_data.json",
            "cache_dir": "./data/cache_dir",
            ...
        },
        ...
    },
    ...
}
```

If you need to load a large amount of data, you can use streaming loading by modifying the `sampler_type` field in `data.json` and adding the `streaming` field. (Note: Once streaming loading is enabled, only `num_workers=0` is currently supported, meaning data is processed in a single process. This may cause performance fluctuations, and resumable training will not be supported.)

```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            ...
            "streaming": true
            ...
        },
        ...
    },
    "dataloader_param": {
        ...
        "sampler_type": "stateful_distributed_sampler",
        ...
    }
}

```

If you need to perform audio data training, you need to modify `attr`, set the `images` field to `null`, and configure the `audios` field. The input audio sampling rate can be configured via the `audio_sampling_rate` field, and it will be automatically resampled to 16kHz during training to adapt to Qwen3-Omni audio feature extraction.

```json
{
    "dataset_param": {
        ...
        "preprocess_parameters": {
            ...
            "audio_sampling_rate": 16000
            ...
        },
        ...
    },
    ...
    "attr": {
        ...
        "system": null,
        "images": null,
        "videos": null,
        "audios": "audios",
        ...
    }
}

```

If you need to support speech and video data and perform cross-modal fusion, you can set `use_audio_in_video` to `true`.

```json
{
    "dataset_param": {
        ...
        "preprocess_parameters": {
            ...
            "use_audio_in_video": true,
            ...
        },
        "attr": {
            ...
            "images": null,
            "videos": "videos",
            "audios": "audios",
            ...
        },
        ...
    },
    ...
}
```

If you encounter a communication TIMEOUT when loading a large amount of data, you can add the `dataset_param.basic_parameters.preprocess_on_fly` field in `data.json` and set it to `true`.

[Sequence Parallelism Configuration]

If the sequence length of the training data is long, it is recommended to set `TASK_QUEUE_ENABLE` to `1` in `examples/qwen3omni/finetune_qwen3omni.sh`, and adjust the `SEQ_LEN` parameter according to the actual scenario (for example, `262144`).

```shell
export TASK_QUEUE_ENABLE=1
SEQ_LEN=262144
```

Ulysses is currently supported. During training with long sequences, CP needs to be enabled by setting CP to a value greater than 1 in `examples/qwen3omni/finetune_qwen3omni.sh`. For example:

```shell
CP=4
```

By default, Ulysses is used for sequence parallelism.

```shell
    --context-parallel-algo ulysses_cp_algo
```

Note: If CP > 1, but the audio sequence length does not exceed the CP size, the AuT module does not support Ulysses.

[Attention Configuration]

attn_implementation and layout configuration:
  Currently, the audio, vision, and text modules support selecting different attention implementations. Specifically, modify the `attn_implementation` field in the `model.json` file.

  | Module | Supported FA and layout |
  | --- | --- |
  | AuT | `flash_attention_2`: `BNSD` |
  | AuT | `flash_attention_2`: `TND` |
  | AuT | `sdpa`: `BNSD` |
  | AuT | `eager`: `BNSD` |
  | ViT | `flash_attention_2`: `BNSD` |
  | ViT | `flash_attention_2`: `TND` |
  | ViT | `sdpa`: `BNSD` |
  | ViT | `eager`: `BNSD` |
  | LLM | `flash_attention_2`: `BNSD` |
  | LLM | `flash_attention_2`: `TND` |
  | LLM | `flash_attention_2`: `BSND` |
  | LLM | `sdpa`: `BNSD` |
  | LLM | `eager`: `BNSD` |

[activation_offload Configuration]

Using `activation_offload` moves the activations of checkpoints generated during recomputation to the host, and asynchronously transfers them from the host to the device in the backward pass, reducing device activation memory usage. To configure it, set the `activation_offload` field to `true` in `model.json`.

[chunkloss Configuration]

Refer to [chunk loss](../../docs/en/features/chunkloss.md).

[Model Saving, Loading, and Logging Configuration]

Configure the parameters in `examples/qwen3omni/finetune_qwen3omni.sh` according to the actual situation, including the load and save paths, as well as the save interval `--save-interval` (Note: Distributed optimizer files are large and saving them takes a long time, so please set the save interval carefully).

```shell
...
# Weight load path: converted weights
LOAD_PATH="./ckpt/convert_path/Qwen3-Omni-30B-A3B-Instruct"
# Save path
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # Do not load optimizer state; remove this if loading is required
    --no-load-rng \  # Do not load random number state; remove this if loading is required
    --no-save-optim \  # Do not save optimizer state; remove this if saving is required.
    --no-save-rng \  # Do not save random number state; remove this if saving is required.
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # Logging Interval
    --save-interval 5000 \  # Save Interval
    --save $SAVE_PATH \ # Save Path
"
```

Configure the `init_from_hf_path` parameter in `examples/qwen3omni/model.json` according to the actual situation. This parameter specifies the loading path for the initial weights.
Configure the `image_encoder.vision_encoder.freeze`, `image_encoder.vision_projector.freeze`, `audio_encoder.audio_encoder.freeze`, and `text_decoder.freeze` parameters in `examples/qwen3omni/model.json` according to the actual situation. These parameters respectively indicate whether to freeze the vision model module, the multi-model projector module, the audio model module, and the language model module.
Note that the network layer counts in the current `examples/qwen3omni/model.json` are invalid, because they have not been verified. If layer reduction is needed, please modify the relevant `config.json` under the original HF path.

[Single-node Execution Configuration]

Configure the parameters in `examples/qwen3omni/finetune_qwen3omni.sh` as follows:

```shell
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# A single node with 16 cards can run the full layers of the model.
NPUS_PER_NODE=16
# If you want to specify card 0, add `export ASCEND_RT_VISIBLE_DEVICES=0`
# And modify `NPUS_PER_NODE=1`.
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
# You can modify the number of steps to 5000.
--train-iters 5000
```

[Multi-node Running Configuration]

Configure the parameters of `examples/qwen3omni/finetune_qwen3omni.sh` as follows (defaults to a two-node running configuration in performance-related scenarios):

```shell
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Configure distributed parameters according to the actual situation of the distributed cluster.
export GLOO_SOCKET_IFNAME="Your SOCKET IFNAME" # Obtain via ifconfig.
# If the number of cards per node is greater than 8, you need to specify the devices. If specifying the first 8 cards, you can set as follows:
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPUS_PER_NODE=8 # Number of cards per node. Fill it in according to the actual situation. In performance-related scenarios, each node has 8 cards.
MASTER_ADDR=<master_ip_address> # All must be changed to the IP address of the master node (cannot be localhost).
MASTER_PORT=6000 # Keep consistent across all nodes.
NNODES=2 # Number of nodes in the cluster. Fill it in according to the actual situation.
NODE_RANK=0 # RANK of the current node; cannot be duplicated across multiple nodes. The rank of the master node is 0, and ranks of other nodes can be 1, 2, etc.
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
# The number of steps can be modified to 200.
--train-iters 200
```

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Take Qwen3-Omni as an example to start the fine-tuning training task.
Differences in loss calculation methods can have varying impacts on training effectiveness. Before starting the training task, please review the documentation on loss calculation and select an appropriate loss calculation method: [vlm_model_loss_calculate_type.md](../../docs/en/features/vlm_model_loss_calculate_type.md).

```shell
cd MindSpeed-MM/
bash examples/qwen3omni/finetune_qwen3omni.sh
```

<a id="jump4.5"></a>

### 4. Fine-tuning with Tool Call Data

- Data organization format:

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are a professional and efficient AI customer service agent. The current conversation time is: 2024-06-15 14:30:00 (Today is Saturday)."
        },
        {
            "role": "user",
            "content": "Hello, are you there?"
        },
        {
            "role": "assistant",
            "content": "Hello, is this Ms. Du?"
        },
        {
            "role": "user",
            "content": "Hello, yes, I am Ms. Du."
        },
        {
            "role": "assistant",
            "content": "Ms. Du, thank you for confirming. I am a staff member from the CDC. We have noticed that you have not yet received the new influenza vaccine. For your health and public health safety, we recommend that you get vaccinated as soon as possible."
        },
        {
            "role": "user",
            "content": "I am in good health right now and do not want to get vaccinated for the time being."
        },
        {
            "role": "assistant",
            "content": "I understand your concerns, but vaccination can not only effectively prevent infection but also protect your health and the health of those around you. Especially in the current pandemic environment, getting vaccinated is very important."
        },
        {
            "role": "user",
            "content": "I am worried about adverse reactions after vaccination. Is it really safe?"
        },
        {
            "role": "assistant",
            "content": "Your concern is understandable. The new influenza vaccine has undergone rigorous safety testing and has a high level of safety. There may be mild discomfort after vaccination, but most people get through it without any issues. In addition, we will ask you to remain under observation for 30 minutes after vaccination to ensure any adverse reactions are addressed promptly."
        },
        {
            "role": "user",
            "content": "Alright, hearing you say that, I agree to get vaccinated."
        },
        {
            "role": "assistant",
            "content": "Thank you very much for your understanding and support. Our vaccination hours are Monday to Saturday, from 9:00 AM to 12:00 PM and from 2:00 PM to 9:00 PM. You can choose a convenient time to come in for the vaccination."
        },
        {
            "role": "user",
            "content": "Can I choose Wednesday at 3:00 PM?"
        },
        {
            "role": "tool_call",
            "content": "<tool_call>\n{\"name\": \"register_vaccine_appointment\", \"arguments\": {\"appointment_time\": \"3:00 PM, Wednesday\"}}\n</tool_call>"
        },
        {
            "role": "tool_response",
            "content": "{\"status\": \"success\", \"message\": \"Appointment success\"}"
        },
        {
            "role": "assistant",
            "content": "Alright, your vaccination appointment has been registered for Wednesday at 3:00 PM. Please bring a valid ID and arrive at the designated vaccination site on time. Thank you for supporting the pandemic prevention and control efforts."
        },
        {
            "role": "user",
            "content": "Thank you for your hard work."
        },
        {
            "role": "assistant",
            "content": "You're welcome, it's our duty. Stay healthy, goodbye.",
            "id": "Customer Service.Polite Closing"
        }
    ],
    "audios": "/speeches/7_Katerina.wav",
    "tools": [
        "{\"type\": \"function\", \"function\": {\"name\": \"register_vaccine_appointment\", \"description\": \"Register a user's vaccination appointment\", \"parameters\": {\"type\": \"object\", \"properties\": {\"appointment_time\": {\"type\": \"string\", \"description\": \"The vaccination time selected by the user\"}}, \"required\": [\"appointment_time\"]}}}"
    ]
}
```

<font color='red'>Note: The data type of tools is `list[str]`.</font>

- Modify data.json

```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            "template": "qwen3_omni_nothink",
        },
        "attr": {
          ...
            "system_tag": "system",
            "formatting": "multimodal_tool"
        }
    },
    ...
}
```

<a id="jump10"></a>

## Environment Variable Declaration

| Environment Variable | Description | Value Description |
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

<a id="jump11"></a>

## Notes

‼️ During multi-card fine-tuning, gradient communication issues may occur. The MOE implementation in transformers needs to be rewritten. The rewriting method that involves weight conversion can achieve better performance, while other rewriting methods (e.g., having all experts participate in forward computation) yield poorer performance.

‼️ Two lines of code, `position_ids.to(torch.bfloat16)` and `self.rope_deltas = None`, are reserved in the project at the file path `mindspeed_mm/models/transformers/qwen3omni/modeling_qwen3_omni_moe.py`. **Their purpose is to maintain precision alignment with LLaMA Factory**. It is verified by actual testing that **removing these two lines of code** generally leads to better training results.
