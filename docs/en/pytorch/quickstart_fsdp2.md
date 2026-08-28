# Quick Start: Qwen3-VL Model Fine-Tuning

MindSpeed MM supports multimodal understanding models. Using the Qwen3-VL model as an example, this document introduces how to use MindSpeed MM and guides developers to quickly get started with efficient execution of preset models on Ascend NPUs.

## Multimodal Understanding Models

This document uses Qwen3-VL-30B as an example to guide users through fine-tuning a multimodal understanding model in a single-node scenario.

Fully Sharded Data Parallel 2 (FSDP2) fully shards parameters, gradients, and optimizer states along the data-parallel dimension, significantly reducing per-rank memory usage. It is decoupled from the model architecture and requires low adaptation cost for new models. In MindSpeed-MM, FSDP2 training has the following characteristics:

- **Trainer**: `mindspeed_mm/fsdp/train/trainer.py`, which can be launched with a single `torchrun` command in the startup script.
- **Centralized configuration**: A single YAML file is sufficient, divided into six segments: `parallel`, `data`, `model`, `features`, `training`, and `tools`.
- **Weight Format**: Uses PyTorch Distributed Checkpoint (DCP) format, combined with `meta init` to reduce peak memory usage.
- **Scalable**: On top of sharding, Ulysses context parallelism (CP) and MoE expert parallelism (EP) can be added.

> [!NOTE]
>
> MindSpeed MM supports the <term>Ascend 950 products</term>, <term>Atlas A3 training products</term>, and <term>Atlas A2 training products</term>, and requires the on-chip memory of a single NPU to be 64 GB or above.
>
> In the current example script, `NPUS_PER_NODE=16` indicates that 16 NPUs are used. If the actual count is lower than this, you may encounter OOM (Out of Memory) issues.
>
> If you use other models, refer to the actual parameters of the specific script.

Prerequisites for developers:

- Basic experience with PyTorch
- Basic Python development experience
- Basic understanding of FSDP

## Environment Preparation

Click [MindSpeed Quick Start](https://www.hiascend.com/en/developer/software/mindspeed/download) and follow the instructions to set up the environment. For detailed installation instructions, see [MindSpeed MM Installation Guide](install_guide.md).

Create the following directories under `MindSpeed-MM` to store logs, data, and weight files.

    ```bash
    mkdir logs
    mkdir data
    mkdir ckpt
    ```

## Weight Download and Conversion

1. Weight Download

   Download the [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) weights from Hugging Face.
    >[!NOTE]
    >
    > If you cannot access the HuggingFace community to download resources smoothly, it is recommended to download from ModelScope. Pay attention to the correctness and security of the files to be downloaded.

2. Weight File Storage

   Create the `ckpt/Qwen3-VL-30B-A3B-Instruct` directory and save the downloaded model weight file to that directory.

3. Weight Conversion

    You can use the `mm-convert` tool to convert the original weights. Run the following command to execute the tool:

    ```bash
    # Qwen3-VL-30B
    mm-convert GenericDCPConverter hf_to_dcp \
      --hf_dir ckpt/Qwen3-VL-30B-A3B-Instruct \
      --dcp_dir ckpt/Qwen3-VL-30B-A3B-Instruct-dcp
    ```

    **Table 1** Weight conversion tool parameter description

    |Parameter|Description|Required|Default|
    |-|-|-|-|
    |`GenericDCPConverter`|Qwen3-VL model conversion tool|Yes|/|
    |`hf_to_dcp`|Converts Hugging Face model weights to MindSpeed MM model weights|Yes|/|
    |`dcp_dir`|Directory for saving converted weights|Yes|/|
    |`hf_dir`|Directory of Hugging Face weights|Yes|/|

    > [!NOTE]
    > Qwen3_VL and Qwen2_VL share the same weight conversion logic. For more tool details, see [Weight Conversion Command-line Tool](../features/mm_convert.md).

    When `meta init` is enabled for initialization, DCP weights are required (**Qwen3-VL-30B/235B must use `meta init`, which is enabled by default in the repository**).
   After conversion, a `release/` folder and `latest_checkpointed_iteration.txt` are generated under `--dcp_dir`. Then enable `init_model_with_meta_device` in the configuration file and point `load` to this `dcp` directory (write to the parent directory of `release`, i.e., `ckpt/Qwen3-VL-30B-A3B-Instruct-dcp`).

## Dataset Preparation and Processing

1. Dataset Download

   Taking the COCO2017 dataset as an example, create the `data/COCO2017` directory, then download and extract the [COCO2017](https://cocodataset.org/#download) dataset.
    >[!NOTE]
    >
    > If you cannot access HuggingFace community resources smoothly, it is recommended to download from ModelScope instead. Pay attention to the correctness and security of the files to be downloaded.

2. Obtain the Dataset Description File

   Download the image dataset description file [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) from Hugging Face and save it to the `./data/` path.

3. Dataset Preprocessing

    Run the following data conversion script:

    ```python
    # This script is compatible with both Qwen2-VL and Qwen3-VL and can be run directly.
    python mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py
    ```

    After conversion, the reference data directory structure is as follows:

    ```bash
    ├── data
        ├── COCO2017
            ├── train2017
        ├── llava_instruct_150k.json
        ├── mllm_format_llava_instruct_data.json
        ...
    ```

## Parameter Configuration

You only need to edit **one** configuration file `examples/qwen3vl/qwen3vl_30B_config_v1.yaml`, which consists of six top-level segments: `parallel` (parallelism/sharding), `data` (data), `model` (model/freezing), `features` (loss, etc.), `training` (training hyperparameters/weight send and receive), and `tools` (tools).

### Required Fields

| Segment | Field | Change to |
|---|---|---|
| `data` | `model_name_or_path` | The **original HF weight** path before conversion, i.e. `./ckpt/Qwen3-VL-30B-A3B-Instruct` |
| `data` | `dataset_dir` | The dataset root directory, i.e. `./data` |
| `data` | `dataset` | The preprocessed `./data/mllm_format_llava_instruct_data.json` |
| `training` | `load` | The converted **DCP weight** path `./ckpt/Qwen3-VL-30B-A3B-Instruct-dcp` (commented out by default; uncomment and fill in when using meta init) |
| `training` | `init_model_with_meta_device` | `true` (enabled by default; required for 30B/235B) |
| `training` | `save` / `save_interval` | The weight save path and interval |
| `model` | `freeze` | The modules to freeze (optional) |
| `features` | `loss_type` | The loss computation method (`default`, etc.; see below) |

Data segment example (`model_name_or_path` uses the original Hugging Face path before conversion; on multi-node setups, do not use the same mounted directory for `cache_dir`):

```yaml
data:
  dataset_param:
    dataset_type: huggingface
    preprocess_parameters:
      model_name_or_path: &HF_MODEL_LOAD_PATH ./ckpt/Qwen3-VL-30B-A3B-Instruct
    basic_parameters:
      dataset_dir: ./data
      dataset: &DATASET_PATH ./data/mllm_format_llava_instruct_data.json
      cache_dir: ./data/cache_dir
```

### Sharding Strategy (`parallel` Segment, Preconfigured in the Sample)

The FSDP sharding configuration is in the `parallel` segment of this YAML file, and no modification is required for regular fine-tuning:

```yaml
parallel:
  tensor_parallel_size: 1
  fully_shard_parallel_size: auto    # FSDP size. auto means it is sets automatically based on the global number of devices.
  fsdp_plan:
    apply_modules:                   # Modules that require fully_shard.
      - model.visual.blocks.{*}
      - model.language_model.layers.{*}
      - lm_head
      # ... (see the sample file for the complete list)
    param_dtype: bf16
    reduce_dtype: fp32
  ulysses_parallel_size: 1           # Can be greater than 1 for long sequences to enable Ulysses CP.
  expert_parallel_size: 1            # MoE EP
```

## Starting Fine-tuning

1. Configure the startup script `examples/qwen3vl/finetune_qwen3vl_30B_v1.sh` according to the server scale.

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh

   NPUS_PER_NODE=16        # Number of NPUs per node (MindSpeed MM supports Ascend 950 products, Atlas A3 training products, and Atlas A2 training products, and requires the on-chip memory of a single NPU to be 64 GB or above. In the current example, NPUS_PER_NODE=16 indicates that 16 NPUs are required. If the actual occur is lower than this, OOM issues may occur.)
   MASTER_ADDR=localhost   # Change to the primary node IP for multi-node scenarios.
   MASTER_PORT=6000
   NNODES=1                # Change to the total number of nodes for multi-node scenarios.
   NODE_RANK=0             # For multi-node setups, change this to the local node rank.
   ```

2. Start fine-tuning.

   ```bash
   bash examples/qwen3vl/finetune_qwen3vl_30B_v1.sh
   ```

   Logs are output to `logs/`, and weights are saved to the directory specified by `training.save` in the YAML file.

>[!NOTE]
>
> **Loss calculation method**: The `loss_type` field in the `features` section defaults to `default`, which is suitable for in most cases. Adjust this field only if you need to customize normalization methods such as per-sample or per-token normalization.

## Post-processing

The training produces weights in DCP format. If you need to load them with HuggingFace/Transformers, you can first export them back to Hugging Face format.

The following is an example of `dcp2hf` conversion:

```bash
mm-convert GenericDCPConverter dcp_to_hf \
  --load_dir save_dir/release \
  --save_dir save_dir_hf \
  --model_assets_dir ./ckpt/Qwen3-VL-30B-A3B-Instruct
```

- `--load_dir`: the directory under the training save directory that stores the DCP shards (the `release` under the `training.save` path; fill in according to the actual save structure).
- `--save_dir`: the output directory for the exported Hugging Face weights.
- `--model_assets_dir`: original Hugging Face weight directory, used to copy assets such as `config`/`tokenizer`.

> For the complete list of parameters, refer to `mm-convert GenericDCPConverter dcp_to_hf -h`.

If you only want low-cost fine-tuning, you can use LoRA instead (see [LoRA Fine-tuning (FSDP2)](../features/lora_finetune_fsdp2.md)).
