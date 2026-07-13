# InternVL3.5 User Guide

<p align="left">
</p>

## Directory

- [InternVL3.5 User Guide](#internvl35-user-guide)
  - [Directory](#directory)
  - [Version Notes](#version-notes)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Installation](#environment-installation)
    - [1. Environment Preparation](#1-environment-preparation)
    - [2. Environment Setup](#2-environment-setup)
  - [Weights Download and Conversion](#weights-download-and-conversion)
    - [Weights Download](#weights-download)
      - [Special Note](#special-note)
    - [Weights Conversion](#weights-conversion)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [1. Dataset Download](#1-dataset-download)
  - [Fine-tuning](#fine-tuning)
    - [1. Preparation](#1-preparation)
    - [2. Parameter Configuration](#2-parameter-configuration)
    - [3. Start Fine-tuning](#3-start-fine-tuning)
    - [4. Start Inference](#4-start-inference)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Notes

### Reference Implementation

```shell
url=https://github.com/OpenGVLab/InternVL.git
commit_id=2410d1db
```

### Changelog

2025.10.21: Initial support for the InternVL3_5--30B-A3B-Instruct model

<a id="jump1"></a>

## Environment Installation

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md)

<a id="jump1.1"></a>

### 1. Environment Preparation

[It is recommended to use the matching environment version for model development]

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md) to complete the Ascend software installation.
> Python 3.10 is recommended, and torch and torch_npu 2.7.1 are recommended.

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
mkdir logs dataset ckpt

# Install the acceleration library.
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout d76dbddd4517d48a2fc1cd494de8b9a6cfdbfbab

# Install MindSpeed and its dependencies.
pip install -e .
cd ..
# Install MindSpeed-MM and its dependencies.
pip install -e .

# Install the new version of transformers (supports the InternVL3.5 model).
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout c0dbe09
pip install -e .

```

<a id="jump2"></a>

## Weights Download and Conversion

<a id="jump2.1"></a>

### Weights Download

Download open-source model weights from websites such as Hugging Face.

- [InternVL3_5-30B-A3B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-Instruct)

Save the model weights in the `ckpt/hf_path/` directory, for example, `ckpt/hf_path/InternVL3_5-30B-A3B-Instruct`.

#### Special Note

After downloading the weights, you need to modify the `ckpt/hf_path/InternVL3_5-30B-A3B-Instruct/modeling_internvl_chat.py` file in the weights path. Change `None` to `151671` at line 96, and add the `**kwargs` parameter to the `forward` function at line 112.

```python
    # Line 96
    self.img_context_token_id = 151671

    # Line 112
    def forward(
        ...
        return_dict: Optional[bool] = None,
        **kwargs
    )
```

<a id="jump2.2"></a>

### Weights Conversion

MindSpeed MM modifies the structure names of the original MoE network, so the original pre-trained weights need to be converted.

```shell
# Convert weights for the modified network structure.
mm-convert ExpertMergeDcpConverter hf_to_dcp --hf_dir "ckpt/hf_path/InternVL3_5-30B-A3B-Instruct" --save_dir "ckpt/convert_path/InternVL3_5-30B-A3B-Instruct"
```

<a id="jump3"></a>

## Dataset Preparation and Processing

<a id="jump3.1"></a>

### 1. Dataset Download

[Image Data]

Users need to obtain and decompress the [InternVL-Finetune](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data) dataset into the `dataset/playground` directory. The data structure after decompression is as follows:

   ```shell
   $playground
   ├── data
       ├── ai2d
           ├── abc_images
           ├── images
       ├── coco
           ├── train2017
       ├── docvqa
           ├── train
           ├── test
           ├── val
       ├──...
   ├── opensource
       ├── ai2d_train_12k.jsonl
       ├── sharegpt4v_instruct_gpt4-vision_cap100k.jsonl
       ├── chartqa_train_18k.jsonl
       ├── ...
   ```

[Video Data]

To use videos for training, refer to [Video Dataset Construction](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data) to build your own video dataset.

The Decord library is also required for reading videos. The installation method for Decord is as follows:

X86:

```bash
pip install decord==0.6.0
```

Arm：

For installation via `apt`, please [refer to the link](https://github.com/dmlc/decord).

For installation via `yum`, please [refer to the script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

<a id="jump4"></a>

## Fine-tuning

<a id="jump4.1"></a>

### 1. Preparation

Before configuring the script, you need to complete the preliminary preparations, including: **Environment Installation**, **Weights Download**, and **Dataset Preparation and Processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data.json` according to the actual situation, including fields such as `from_pretrained`, `data_path`, and `data_folder`.

Taking InternVL3_5-30B-A3B-Instruct as an example, make the following modifications to `data.json`. Note that the weights path for `tokenizer_config` is the original weights path.

```json
{
  "dataset_param": {
      ...
      "basic_parameters": {
          "data_path": "dataset/playground/opensource/sharegpt4v_instruct_gpt4-vision_cap100k.jsonl",
          "data_folder": "dataset/playground/data"
      },
      ...
      "tokenizer_config": {
          ...
          "from_pretrained": "./ckpt/hf_path/InternVL3_5-30B-A3B-Instruct",
          ...
      },
      ...
  },
  ...
}
```

[Model Path Configuration]

Modify the weight paths in `model.json` according to the actual situation, including fields such as `from_pretrained`, `data_path`, and `data_folder`.

Taking InternVL3_5-30B-A3B-Instruct as an example, make the following modifications to `model.json`, where the weight path for `init_from_hf_path` is the original weight path.

```json
{
    ...
    "init_from_hf_path": "./ckpt/hf_path/InternVL3_5-30B-A3B-Instruct",
    ...
}
```

[Model Saving, Loading, and Logging Configuration]

Configure the parameters of `examples/internvl3.5/finetune_internvl3_5.sh` according to the actual situation, including load and save paths, as well as `--save-interval` (Note: Saving distributed optimizer files are time-consuming, so please set the save interval cautiously).

 Taking InternVL3_5-30B-A3B-Instruct as an example:

```shell
...
# Load path: original weight path
HF_PATH="ckpt/hf_path/InternVL3_5-30B-A3B-Instruct"
# Load path: path after weights conversion
LOAD_PATH="ckpt/convert_path/InternVL3_5-30B-A3B-Instruct"
# Save path
SAVE_PATH="internvl35_finetune_result"
...
GPT_ARGS="
    ...
    --no-load-optim \  # Do not load optimizer state; remove this if loading is required.
    --no-load-rng \  # Do not load random number state; remove this if loading is required.
    --no-save-optim \  # Do not save optimizer state; remove this if saving is required.
    --no-save-rng \  # Do not save the random number state; remove this if saving is required.
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # Logging Interval
    --save-interval 5000 \  # Save Interval
    ...
    --log-tps \  # Add this parameter to enable printing the average sequence length of the language module at each step during training, and to calculate the throughput in tokens per second after training completes.
"
```

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Single-node Running Configuration]

Configure the parameters in `examples/internvl3.5/finetune_internvl3_5.sh` as follows.

```shell
  # Modify the ascend-toolkit path according to the actual situation.
  source /usr/local/Ascend/cann/set_env.sh
  NPUS_PER_NODE=16 # Can run on a single A3 server.
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

[Multi-node Running Configuration]

Configure the parameters in `examples/internvl3.5/finetune_internvl3_5.sh` as follows.

```shell
  # Modify the ascend-toolkit path according to the actual situation.
  source /usr/local/Ascend/cann/set_env.sh
  # Configure distributed parameters according to the actual distributed cluster situation.
  GPUS_PER_NODE=8  # Number of cards per node. Fill in according to the actual situation.
  MASTER_ADDR="your master node IP"  # All must be modified to the IP address of the master node (cannot be localhost).
  MASTER_PORT=6000
  NNODES=2  # Number of nodes in the cluster. Fill in according to the actual situation.
  NODE_RANK="current node id"  # RANK of the current node. Must be unique across multiple nodes. The rank is 0 for the master node, and ranks of other nodes can be 1, 2, etc.
  WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

### 3. Start Fine-tuning

Take InternVL3_5-30B-A3B-Instruct as an example to start the fine-tuning training task.

```shell
bash examples/internvl3.5/finetune_internvl3_5.sh
```

<a id="jump4.4"></a>

### 4. Start Inference

After training is complete, convert the weights saved in the `SAVE_PATH` directory to the Hugging Face format.

```shell
mm-convert ExpertMergeDcpConverter dcp_to_hf --hf_dir "ckpt/hf_path/InternVL3_5-30B-A3B-Instruct" --dcp_dir "internvl35_finetune_result/iter_000xx" --save_dir "ckpt/dcp_to_hf/InternVL3_5-30B-A3B-Instruct"
```

Where, `--hf_dir` indicates the path to the original Hugging Face weights, `--dcp_dir` indicates the save path of the fine-tuned weights, `iter_000xx` indicates the weights saved at step xx, and `--save_dir` indicates the save path for the converted weights.

After completing the weights conversion, you can use the transformers library for inference.

<a id="jump6"></a>

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
