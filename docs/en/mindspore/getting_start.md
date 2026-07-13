# Quick Start: Fine-tuning the Qwen2.5-VL-7B Model

MindSpeed MM, which uses MindSpore as its backend, supports several multimodal generation and understanding models. The following describes how to use the typical Qwen2.5-VL model with the MindSpore backend, help you quickly get started with running pre-built models efficiently on MindSpore + Ascend NPUs.

## Multimodal Understanding Models

Taking the Qwen2.5-VL-7B model as an example, the following introduces an efficient way to run multimodal understanding models.

### Environment Preparation

1. Install the model training environment based on the MindSpore framework and Python 3.10. For details, refer to the [Installation Guide](./install_guide.md).
2. Create the following directories under `MindSpeed-MM` to store logs, data, and weight files.

    ```bash
    mkdir logs
    mkdir data
    mkdir ckpt
    ```

### Weight Download and Conversion

1. Download weights.

    Download the corresponding model weights [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main) from the Hugging Face library.

2. Save weights.

   Create the `ckpt/hf_path/Qwen2.5-VL-7B-Instruct` directory and save the downloaded model weights into this directory.

3. Convert weights.

   MindSpeed MM has modified some of the original network structure names. Use the `mm-convert` tool to convert the original pre-trained weights.

    The following is an example of converting Hugging Face weights to MindSpeed MM weights:

    ```bash
    # 7b
    mm-convert  Qwen2_5_VLConverter hf_to_mm \
    --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
    --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
    --cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
    --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
    --cfg.parallel_config.tp_size 1
    ```

    **Table 1** Parameters

    |Parameter|Description|Required|Default Value|
    |-|-|-|-|
    |`mm_dir`|Directory to save the converted weights|Yes|/|
    |`hf_dir`|Hugging Face weight directory|Yes|/|
    |`llm_pp_layers`|Number of LLM layers partitioned across each card. Note that this must be consistent with `pipeline_num_layers` configured in `examples/qwen2.5vl/model_7b.json`|No|[1,10,10,7]|
    |`vit_pp_layers`|Number of ViT layers partitioned across each card. Note that this must be consistent with `pipeline_num_layers` configured in `examples/qwen2.5vl/model_7b.json`|No|[32,0,0,0]|
    |`tp_size`|Tensor parallelism size. Note that this must be consistent with the configuration in the fine-tuning startup script.|No|1|

    > [!NOTE]
    > The weight conversion logic for Qwen2_5_VL and Qwen2_VL remains consistent. Refer to [Weight Conversion Tool Usage](../features/mm_convert.md) for details.

### Data Preprocessing

1. Download a dataset (COCO2017 as an example).

   Create the `data/COCO2017` directory, then download and extract the [COCO2017](https://cocodataset.org/#download) dataset.

2. Obtain the dataset description file.

   Download the image dataset description file [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) from Hugging Face and save it to the `./data/` path.

3. Pre-process the dataset.

    Execute the following data conversion script:

    ```python
    python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py
    ```

The reference data directory structure after conversion is as follows:

    ```bash
    $playground
    ├── data
        ├── COCO2017
            ├── train2017

        ├── llava_instruct_150k.json
        ├── mllm_format_llava_instruct_data.json
        ...
    ```

> [!NOTE]
> Qwen2_5_VL and Qwen2_VL share the same data conversion logic, and the data conversion script from Qwen2_VL is used here.

### Fine-tuning

1. Configure the data directory.

    Configure the dataset path in `examples/qwen2.5vl/data_7b.json`. A configuration example is as follows:

    ```json
        {
            "dataset_param": {
                "dataset_type": "huggingface",
                "preprocess_parameters": {
                    "model_name_or_path": "./ckpt/hf_path/Qwen2.5-VL-7B-Instruct",
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

    **Table 2** Parameters

    |Parameter|Description|Value|
    |-|-|-|
    |`model_name_or_path`|Weights|`./ckpt/hf_path/Qwen2.5-VL-7B-Instruct`, consistent with `hf_config.hf_dir` in [Weight Download and Conversion](#weight-download-and-conversion).|
    |`dataset_dir`|Dataset directory|`./data`|
    |`dataset`|Dataset|`./data/mllm_format_llava_instruct_data.json`|

    > [!CAUTION]
    > To avoid conflicts caused by writing to the same file, do not configure the same mount directory (`cache_dir`) across multiple nodes.

2. Edit the fine-tuning script.

    ```shell
    vi examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
    ```

3. Configure model saving, loading, and logging information.

    ```bash
    ...
    # Load Path
    LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
    # Save Path
    SAVE_PATH="save_dir"
    ...
    GPT_ARGS="
        ...
        --no-load-optim \  # Do not load optimizer state; remove if loading is required
        --no-load-rng \  # Do not load random number state; remove if loading is required
        --no-save-optim \  # Do not save optimizer state; remove if saving is required
        --no-save-rng \  # Do not save the random number state; remove if saving is required.
        ...
    "
    ...
    OUTPUT_ARGS="
        --log-interval 1 \  # Logging Interval
        --save-interval 5000 \  # Saving Interval
        ...
        --log-tps \  # Add this parameter to enable printing the average sequence length of the language module at each step during training, and to calculate the throughput in tokens per second after training.
    "
    ```

    To load the weights, optimizer states, etc., from a specific iteration, set `LOAD_PATH` to `"save_dir"`, and modify the content of the `latest_checkpointed_iteration.txt` file to the specified iteration count.

    ```bash
    $save_dir
    ├── latest_checkpointed_iteration.txt
    ├── ...
    ```

    **Table 3** Parameters

    |Parameter|Description|Value|
    |-|-|-|
    |LOAD_PATH|Load path|/|
    |SAVE_PATH|Save path|/|
    |`--log-interval`|Logging interval|1|
    |`--save-interval`|Saving interval|5000|
    |`--no-load-optim`|Do not load optimizer state. Remove it if loading is required.|/|
    |`--no-load-rng`|Do not load random number state. Remove it if loading is required.|/|
    |`--no-save-optim`|Do not save optimizer state. Remove it if saving is required.|/|
    |`--no-save-rng`|Do not save random number state. Remove it if saving is required.|/|

    > [!NOTE]
    > The distributed optimizer files are large, resulting in longer processing times. Please set the save interval carefully.

4. Configure model runtime parameters.

    Complete the configuration of the model runtime parameters. A configuration example is as follows:

    ```bash
    # Modify the ascend-toolkit path according to the actual situation
    source /usr/local/Ascend/cann/set_env.sh
    NPUS_PER_NODE=8          # Use a single node with 8 NPUs
    MASTER_ADDR=localhost    # Use the local node IP in single-node scenario
    MASTER_PORT=29501        # The port number for this node is 29501
    NNODES=1                 # Configure based on the number of participating nodes; set to 1 for a single node
    NODE_RANK=0              # The RANK for a single node is 0
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
        ```

5. Start fine-tuning.

    After saving the fine-tuning script, start the fine-tuning task with the following command:

    ```shell
    bash examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
    ```

### Subsequent Procedure

MindSpeed MM modifies the structure names of some original networks. After fine-tuning, if you need to convert the weights back to the Hugging Face format, you can use the `mm-convert` tool to convert the fine-tuned weights and modify the weight names to be consistent with the original networks.

The following is a conversion example for `mm2hf`:

```bash
mm-convert  Qwen2_5_VLConverter mm_to_hf \
--cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-7B-Instruct" \
--cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
--cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
--cfg.parallel_config.llm_pp_layers [1,10,10,7] \
--cfg.parallel_config.vit_pp_layers [32,0,0,0] \
--cfg.parallel_config.tp_size 1
```

**Table 4** mm2hf parameters

|Parameter|Meaning|Required|Default Value|
|:----|:----|:----|:----|
|`Qwen2_5_VLConverter`|Qwen2.5-VL model conversion tool|Yes|/|
|`mm_to_hf`|Converts MindSpeed MM weights to Hugging Face weights|Yes|/|
|`save_hf_dir`|Directory to save Hugging Face weights after MindSpeed MM fine-tuning|Yes|/|
|`mm_dir`|Weight directory saved after fine-tuning|Yes|/|
|`hf_dir`|Hugging Face weight directory|Yes|/|
|`llm_pp_layers`|Number of LLM layers partitioned per card. Note that this must be consistent with `pipeline_num_layers` configured in `model.json`.|No|36|
|`vit_pp_layers`|Number of ViT layers partitioned per card. Note that this must be consistent with `pipeline_num_layers` configured in `model.json`.|No|32|
|`tp_size`|TP size. Note that this must be consistent with the configuration in the fine-tuning startup script.|No|1|

If you need to use the converted model for training, synchronously modify the `LOAD_PATH` parameter in `examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`. This path points to the converted or partitioned weights; be sure to distinguish it from the original weight path `ckpt/hf_path/Qwen2.5-VL-7B-Instruct`.

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
```
