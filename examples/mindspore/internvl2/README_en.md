# InternVL2 User Guide (MindSpore as Backend)

<p align="left">
</p>

## Contents

- [InternVL2 User Guide (MindSpore as Backend)](#internvl2-user-guide-mindspore-as-backend)
  - [Contents](#contents)
  - [Environment Setup](#environment-setup)
    - [Repository Pull and Environment Setup](#repository-pull-and-environment-setup)
  - [Weights Download and Conversion](#weights-download-and-conversion)
    - [Weights Download](#weights-download)
    - [Weights Conversion](#weights-conversion)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
    - [Dataset Download](#dataset-download)
  - [Fine-tuning](#fine-tuning)
    - [Prerequisites](#prerequisites)
    - [Parameter Configuration](#parameter-configuration)
    - [Start Fine-tuning](#start-fine-tuning)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Environment Setup

The dependency requirements for the MindSpeed MM MindSpore Backend are listed in the table below. For installation steps, refer to the [Installation Guide](../../../docs/en/mindspore/install_guide.md).

| Dependency         |                                                              |
| ---------------- | ------------------------------------------------------------ |
| Ascend NPU Driver & Firmware  | [Internal development version](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha)
| Ascend CANN        | [Internal development version](https://www.hiascend.com/zh/developer/download/community/result?module=cann) |
| MindSpore        | [2.7.1](https://www.mindspore.cn/install/en)         |
| Python           | >=3.9                                                        |

### Repository Pull and Environment Setup

For the MindSpeed MindSpore backend, the Ascend community provides a one-click conversion tool, MindSpeed-Core-MS, designed to help users automatically pull relevant code repositories and perform one-click adaptation of PyTorch code. This allows users to launch model training with one click in the Huawei MindSpore + CANN environment without additional manual adaptation. Before performing the one-click conversion, users need to pull the relevant code repositories and set up the environment.

```shell
# Create a conda environment.
conda create -n test python=3.10
conda activate test

# Use environment variables.
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# Install the MindSpeed-Core-MS conversion tool.
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.4.0

# Use the internal script of MindSpeed-Core-MS to automatically pull the relevant code repositories, adapt them with one click, and configure the environment.
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm

# Replace files in MindSpeed.
cd MindSpeed-MM
cp examples/internvl2/dot_product_attention.py MindSpeed/mindspeed/core/transformer/dot_product_attention.py
mkdir ckpt
mkdir data
mkdir logs
```

## Weights Download and Conversion

### Weights Download

Download open-source model weights from websites such as Hugging Face.

- [InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B/tree/main)
- [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main)
- [InternVL2-26B](https://huggingface.co/OpenGVLab/InternVL2-26B/tree/main)
- [InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B/tree/main)

Save the model weights in the `raw_ckpt` directory, for example, `raw_ckpt/InternVL2-8B`.

### Weights Conversion

MindSpeed MM has modified some of the original network structure names. Use the `mm-convert` tool to convert the original pre-trained weights. This tool implements the conversion between Hugging Face weights and MindSpeed MM weights, as well as weight sharding for PP (Pipeline Parallelism) and VPP (Virtual Pipeline Parallelism). For detailed VPP configuration, refer to [vpp feature description](../../../docs/en/features/virtual_pipeline_parallel.md).

For detailed usage of the `mm-convert` tool, refer to [Weights Conversion Tool Usage](../../../docs/en/features/mm_convert.md).
**Currently, the converted weights under the MindSpore backend cannot be used for training with the PyTorch backend**.

The MindSpore backend performs weight conversion on the device side by default, which poses an OOM risk for large-scale models. Therefore, you are advised to manually modify `MindSpeed-MM/checkpoint/convert_cli.py` and add the following code to set it to CPU-side weight conversion:

```python
import mindspore as ms
ms.set_context(device_target="CPU", pynative_synchronize=True)
import torch
torch.configs.set_pyboost(False)
```

Taking InternVL2-8B as an example:

```bash
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# 2B
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-2B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-2B" \
  --cfg.parallel_config.llm_pp_layers [[24]] \
  --cfg.parallel_config.vit_pp_layers [[24]] \
  --cfg.trust_remote_code True

# 8B
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-8B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-8B" \
  --cfg.parallel_config.llm_pp_layers [[6,9,9,8]] \
  --cfg.parallel_config.vit_pp_layers [[24,0,0,0]] \
  --cfg.trust_remote_code True

# 8B VPP
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-8B-vpp" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-8B" \
  --cfg.parallel_config.llm_pp_layers [[0,0,0,1],[4,4,4,4],[4,4,4,3]] \
  --cfg.parallel_config.vit_pp_layers [[6,7,7,4],[0,0,0,0],[0,0,0,0]] \
  --cfg.trust_remote_code True

# 76B
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-Llama3-76B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-Llama3-76B" \
  --cfg.parallel_config.llm_pp_layers [[0,0,0,1,5,6,7,7,7,7,7,7,7,7,6,6]] \
  --cfg.parallel_config.vit_pp_layers [[11,12,12,10,0,0,0,0,0,0,0,0,0,0,0,0]] \
  --cfg.trust_remote_code True

```

- Where:
- `mm_dir`: Directory to save the converted weights
- `hf_dir`: Hugging Face weights directory
- `llm_pp_layers`: Number of LLM layers split per card. Note that this must be consistent with the `pipeline_num_layers` configured in `model.json`.
- `vit_pp_layers`: Number of ViT layers split per card. Note that this must be consistent with the `pipeline_num_layers` configured in `model.json`.
- `trust_remote_code`: To ensure code security, `trust_remote_code` is configured as `False` by default. You need to set it to `True` and ensure the security of the downloaded model and data.

## Dataset Preparation and Processing

### Dataset Download

[Image Data]

Users need to obtain and extract the [InternVL-Finetune](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data) dataset into the `dataset/playground` directory. Taking the ai2d dataset as an example, the extracted data structure is as follows:

   ```shell
   $playground
   ├── data
       ├── ai2d
           ├── abc_images
           ├── images
   ├── opensource
       ├── ai2d_train_12k.jsonl
   ```

[Video Data]

To use videos for training, refer to [Video Dataset Construction](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data) to build your own video dataset.

## Fine-tuning

### Prerequisites

Before configuring the script, you need to complete the preliminary preparations, including: [Environment Setup](#environment-setup), [Weights Download and Conversion](#weights-download-and-conversion), and [Dataset Preparation and Processing](#dataset-preparation-and-processing).

### Parameter Configuration

[Data Directory Configuration]

Modify the dataset paths in `data.json` according to the actual situation, including fields such as `from_pretrained`, `data_path`, and `data_folder`.

Taking InternVL2-8B as an example, make the following modifications to `data_8B.json`. Note that the weight path for `tokenizer_config` is the weight path before conversion.

```json
{
  "dataset_param": {
      ...
      "basic_parameters": {
          "data_path": "dataset/playground/opensource/ai2d_train_12k.jsonl",
          "data_folder": "dataset/playground/data/ai2d"
      },
      ...
      "tokenizer_config": {
          ...
          "from_pretrained": "raw_ckpt/InternVL2-8B",
          ...
      },
      ...
  },
  ...
}
```

If you need to load large amounts of data, you can use streaming loading by modifying the `sampler_type` field in `data.json` and adding the `streaming` field. (Note: After streaming loading is enabled, only `num_worker=0` is currently supported, indicating single-process data processing is performed. This may cause performance fluctuations, and resumable training from checkpoints is not supported.)

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

If you need to calculate the validation loss, you also need to set the following parameters:

- Set the `eval-interval` and `eval-iters` parameters in the shell script.
- Add fields within `basic_parameters` in `data.json`:
  - For non-streaming data, there are two methods: Add the `val_dataset` path based on the actual situation; Add the `val_rate` field to split the training dataset;
  - For streaming data, only add the `val_dataset` field.

```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            ...
            "val_dataset": "./data/val_dataset.json",
            "val_max_samples": null,
            "val_rate": 0.1,
            ...
        },
        ...
    },
   ...
}
```

[Model Saving and Loading Configuration]

Configure the parameters of `examples/mindspore/internvl2/finetune_internvl2_xx.sh` based on the actual situation, including the load path, save path, and save interval `--save-interval` (Note: Distributed optimizer files are large and saving them is time-consuming, so please set the save interval carefully).

Taking InternVL2-8B as an example:

```shell
...
# Load Path
LOAD_PATH="ckpt/InternVL2-8B"
# Save Path
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # Do not load optimizer state. Remove this if loading is required
    --no-load-rng \  # Do not load random number state. Remove this if loading is required
    --no-save-optim \  # Do not save optimizer state. Remove this if saving is required
    --no-save-rng \  # Do not save the random number state. Remove this if saving is required.
    ...
"
...
OUTPUT_ARGS="
    ···
    --save-interval 5000 \  # Save Interval
    ...
    --log-tps \  # Add this parameter to enable printing the average sequence length of the language module at each step during training, and to calculate the throughput in tokens per second after training completes.
"
```

If you need to load the weights, optimizer states, etc., from a specific iteration, set `LOAD_PATH` to `"save_dir"`, and modify the `latest_checkpointed_iteration.txt` file to specify the iteration number.

```shell
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

[Logging Configuration]

Configure the parameters in `examples/mindspore/internvl2/finetune_internvl2_xx.sh` as follows:

```shell
OUTPUT_ARGS="
    ···
    --log-interval 1 \  # Logging Interval
    ...
"
```

[Single-Node Run Configuration]

Configure the parameters in `examples/mindspore/internvl2/finetune_internvl2_xx.sh` as follows:

```shell
  # Modify the ascend-toolkit path according to the actual situation.
  source /usr/local/Ascend/cann/set_env.sh
  NPUS_PER_NODE=8
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

### Start Fine-tuning

Take InternVL2-8B as an example to start the fine-tuning training task.

```shell
bash examples/mindspore/internvl2/finetune_internvl2_8B.sh
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
