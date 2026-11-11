# Quick Start: Fine-Tuning the Qwen2.5-VL Model and the Wan2.2 Model

MindSpeed MM supports both multimodal generation and multimodal understanding models. The following uses two typical models, Qwen2.5-VL (an understanding model) and Wan2.2 (a generation model), as examples to describe how to use MindSpeed MM and help developers quickly get started with running the built-in models efficiently on Ascend NPUs.

## Multimodal Understanding Model

This chapter uses Qwen2.5-VL-3B as an example to describe how to complete the fine-tuning of a multimodal understanding model in a single-node scenario.

### Environment Preparation

1. Install the model training environment based on the PyTorch framework and Python 3.12. For details, see [MindSpeed MM Installation Guide](../installation/install_guide.md).
2. Create the following directories under `MindSpeed-MM` to store logs, data, and weight files.

    ```bash
    mkdir logs
    mkdir data
    mkdir ckpt
    ```

> [!NOTE]
>
> MindSpeed MM supports <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 training products</term>, and <term>Atlas A2 training products</term>, and requires 64 GB or more on-chip memory per NPU.
>
> In the current example script, `NPUS_PER_NODE=8` means that 8 NPUs are required. If the actual configuration is lower than this, OOM issues may occur.

Developer Getting Started Prerequisites:

- Basic experience with PyTorch
- Basic experience with PyTorch
- Basic understanding of the Megatron-LM repository

### Weight Download and Conversion

1. Download weights.

   Download the corresponding model weights [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main) from Hugging Face.

   > [!NOTE]
   >
   > If you are unable to successfully download resources from the Hugging Face community, it is recommended to download them from ModelScope instead. Please ensure the correctness and security of the files to be downloaded.

2. Save weights.

   Create the `ckpt/hf_path/Qwen2.5-VL-3B-Instruct` directory and save the downloaded model weights into it.

3. Convert weights.

    MindSpeed MM modifies the structure names of some original networks, so you can use the `mm-convert` tool to convert the original weights. Run the tool with the following command:

    ```bash
    # Qwen2.5-VL-3B
    mm-convert  Qwen2_5_VLConverter hf_to_mm \
    --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
    --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
    --cfg.parallel_config.llm_pp_layers [[36]] \
    --cfg.parallel_config.vit_pp_layers [[32]] \
    --cfg.parallel_config.tp_size 1
    ```

    **Table 1** Weight conversion tool parameters

    |Parameter|Description|Required|Default Value|
    |-|-|-|-|
    |`Qwen2_5_VLConverter`|Qwen2.5-VL model conversion tool|Yes|/|
    |`hf_to_mm`|Convert Hugging Face weights to MindSpeed MM weights|Yes|/|
    |`mm_dir`|Save directory after conversion|Yes|/|
    |`hf_dir`|Hugging Face weight directory|Yes|/|
    |`llm_pp_layers`|Number of LLM layers partitioned per card. Note that this must be consistent with `pipeline_num_layers` configured in `examples/qwen2.5vl/model_3b.json`.|No|36|
    |`vit_pp_layers`|Number of ViT layers partitioned per card. Note that this must be consistent with `pipeline_num_layers` configured in `examples/qwen2.5vl/model_3b.json`.|No|32|
    |`tp_size`|TP size. Note that this must be consistent with the configuration in the fine-tuning startup script.|No|1|

    > [!NOTE]
    > Because Qwen2_5_VL and Qwen2_VL share the same weight conversion logic, see [Weight Conversion Command Line Tool](../../tools/mm_convert.md) for more tool details.

### Data Preprocessing

1. Download a dataset (COCO2017 as an example).

   Using the COCO2017 dataset as an example, create the `data/COCO2017` directory and then download and decompress the [COCO2017](https://cocodataset.org/#download) dataset.

2. Obtain the dataset description file.

   Download the image dataset description file [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) from Hugging Face and save it to the `./data/` path.

3. Pre-process data.

    Execute the following data conversion script:

    ```python
    python mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py
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
    >
    > The Qwen2_5_VL and Qwen2_VL share the same data conversion logic, so the data conversion script from Qwen2_VL is used to meet the requirements.

### Fine-tuning

1. Configure the data directory.

    Configure the dataset path in `examples/qwen2.5vl/data_3b.json`. A configuration example is as follows:

    ```json
        {
            "dataset_param": {
                "dataset_type": "huggingface",
                "preprocess_parameters": {
                    "model_name_or_path": "./ckpt/hf_path/Qwen2.5-VL-3B-Instruct",
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
    |`model_name_or_path`|Weights|`./ckpt/hf_path/Qwen2.5-VL-3B-Instruct`, consistent with `hf_config.hf_dir` in [Weight Download and Conversion](#weight-download-and-conversion).|
    |`dataset_dir`|Dataset directory|`./data`|
    |`dataset`|Dataset|`./data/mllm_format_llava_instruct_data.json`|

    > [!CAUTION]
    >
    > To avoid conflicts caused by writing to the same file, do not configure the same mount directory (`cache_dir`) across multiple nodes.

2. Edit the fine-tuning script.

    ```shell
    vi examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh
    ```

3. Configure model saving, loading, and logging information.

    Complete the configuration of the model load path, save path, and save interval (`--save-interval`). A configuration example is as follows:

    ```bash
    ...
    # Load path
    LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-3B-Instruct"
    # Save path
    SAVE_PATH="save_dir"
    ...
    GPT_ARGS="
        ...
        --no-load-optim \  # Do not load the optimizer state. Remove this if loading is required.
        --no-load-rng \  # Do not load the random number state. Remove this if loading is required.
        --no-save-optim \  # Do not save the optimizer state. Remove this if saving is required.
        --no-save-rng \  # Do not save the random number state. Remove this if saving is required.
        ...
    "
    ...
    OUTPUT_ARGS="
        --log-interval 1 \  # Log interval
        --save-interval 5000 \  # Save interval
        ...
        --log-tps \  # Adding this parameter enables printing the average sequence length of the language module at each step during training and calculating the tokens per second throughput after training ends.
    "
    ```

    **Table 3** Parameters

    |Parameter|Description|Value|
    |-|-|-|
    |`LOAD_PATH`|Load path|`ckpt/mm_path/Qwen2.5-VL-3B-Instruct`|
    |`SAVE_PATH`|Save Path|`save_dir`|
    |`--log-interval`|Log interval|1|
    |`--save-interval`|Save interval|5000|
    |`--no-load-optim`|Do not load optimizer state. Remove it if loading is required.|/|
    |`--no-load-rng`|Do not load random number state. Remove it if loading is required.|/|
    |`--no-save-optim`|Do not save optimizer state. Remove it if saving is required.|/|
    |`--no-save-rng`|Do not save random number state. Remove it if saving is required.|/|

    > [!NOTE]
    >
    > The distributed optimizer files are large, resulting in longer processing times. Please set the save interval carefully.

4. Configure model runtime parameters.

    Complete the configuration of the model runtime parameters. A configuration example is as follows:

    ```bash
    # Modify the ascend-toolkit path as required
    source /usr/local/Ascend/cann/set_env.sh
    NPUS_PER_NODE=8          # Use 8 NPUs on a single node
    MASTER_ADDR=localhost    # Use the local node IP for a single node
    MASTER_PORT=29501        # Port number of this node is 29501
    NNODES=1                 # Configure based on the number of participating nodes; set it to 1 for a single node
    NODE_RANK=0              # RANK is 0 for a single node
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
    ```

5. Start fine-tuning.

    After saving the fine-tuning script, start the fine-tuning task with the following command:

    ```shell
    bash examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh
    ```

### Subsequent Procedure

MindSpeed MM modifies the structure names of some original networks. After fine-tuning, if you need to convert the weights back to the Hugging Face format, you can use the `mm-convert` tool to convert the fine-tuned weights and modify the weight names to be consistent with the original networks.

The following is a conversion example for `mm2hf`:

```bash
mm-convert  Qwen2_5_VLConverter mm_to_hf \
--cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-3B-Instruct" \
--cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
--cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
--cfg.parallel_config.llm_pp_layers [36] \
--cfg.parallel_config.vit_pp_layers [32] \
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

If you need to use the converted model for training, synchronously modify the `LOAD_PATH` parameter in `examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh`. This path points to the converted or partitioned weights; be sure to distinguish it from the original weight path `ckpt/hf_path/Qwen2.5-VL-3B-Instruct`.

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-3B-Instruct"
```

## Multimodal Generation Model

This chapter uses Wan2.2-T2V-A14B as an example to describe how to complete the fine-tuning of a multimodal generation model in a single-node scenario.

### Environment Preparation

1. Install the model training environment based on the PyTorch framework and Python 3.12. For details, see [MindSpeed MM Installation Guide](../installation/install_guide.md).

2. Install other dependencies.

    ```bash
    # Install Diffusers from source
    pip install diffusers==0.33.1
    ```

3. Set up Decord.

    - X86

        ```bash
        pip install decord==0.6.0
        ```

    - Arm

        For installation via `apt`, please refer to [decord documentation](https://github.com/dmlc/decord).

        For installation via `yum`, please refer to the [build_manylinux2010.sh](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh) script.

### Weight Download and Conversion

1. Download weights.

    Download the model weights from the Hugging Face repository: [Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers).
    Create `weights/Wan2.2-T2V-A14B-Diffusers/` under MindSpeed-MM and save the downloaded model weights to this directory.
    > [!NOTE]
    >
    > If you cannot access the download resources of the Hugging Face community, it is recommended to download from ModelScope. Pay attention to the correctness and security of the files to be downloaded.

2. Convert weights.

    You need to convert the `transformer` part of the downloaded Wan2.2 model weights. Run the weight conversion tool:

    ```shell
    mm-convert WanConverter hf_to_mm \
    --cfg.source_path ./weights/Wan2.2-T2V-A14B-Diffusers/transformer/ \
    --cfg.target_path ./weights/Wan2.2-T2V-A14B-Diffusers/transformer_mm/
    ```

    > [!NOTE]
    >
    > The Hugging Face Diffusers weights contain two transformer weights, where the transformer folder corresponds to the high-noise (high) model weights and the transformer_2 folder corresponds to the low-noise (low) model weights

    To convert the weights to DCP format, refer to the following method:

    ```shell
    mm-convert WanConverter mm_to_dcp \
    --cfg.source_path ./weights/Wan2.2-T2V-A14B-Diffusers/transformer_mm/ \
    --cfg.target_path ./weights/Wan2.2-T2V-A14B-Diffusers/transformer_dcp/
    ```

    **Table 5** Weight conversion tool parameters

    | Parameter | Description |
    | :-- | :--- |
    |WanConverter|Wan2.2/Wan2.1 model conversion tool|
    | `hf_to_mm` | Converts Hugging Face weights to MindSpeed MM weights. |
    |hf_to_dcp|Convert Hugging Face model weights to DCP (FSDP2 distributed) model weights|
    | `source_path` | Original weight path |
    | `target_path` | Save path for converted or partitioned weights |

### Data Preprocessing

Create a `dataset` directory under `MindSpeed-MM`, then create a `videos` directory and a `data.json` file under `dataset`, and save the videos to be processed in `videos`. All video-text pair information of the dataset is stored in `data.json`. (You can use the [Open-Sora-Dataset pixabay_v2 dataset](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/pixabay_v2_tar); the corresponding data annotation file is [video_pixabay_65f_601513.json](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/blob/main/anno_jsons/video_pixabay_65f_601513.json). The format must be processed into the form shown in the example below.)

> [!NOTE]
>
> The complete dataset is large (1.22 TB). You can choose to download the data of some folders for testing, for example, only folder_01 to folder_04, which is about 100 GB. Note that you need to process the `video_pixabay_65f_601513.json` file and delete the data items of the videos that are not downloaded.

An example of the specific directory structure is as follows:

```bash
dataset
├──data.json
├──videos
│  ├──video0001.mp4
│  ├──video0002.mp4
```

An example of video-text pair information is as follows:

```json
[
    {
        "path": "videos/video0001.mp4",
        "cap": "Video discrimination1.",
        "num_frames": 81,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 832
        }
    },
    {
        "path": "videos/video0002.mp4",
        "cap": "Video discrimination2.",
        "num_frames": 81,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 832
        }
    }
]
```

|Parameter|Description|Default Value|
|-|-|-|
|`path`|Video storage path|/|
|`cap`|Video description|Configure it according to the actual situation.|
|`num_frames`|Maximum number of frames|81|
|`fps`|Video frame rate|Configure it according to the actual situation.|
|`height`|Video height|Configure it according to the actual situation.|
|`width`|Video width|Configure it according to the actual situation.|

### Parameter Configuration

1. Configure `data.txt`.

    Modify the `examples/wan2.2/data.txt` file, where each line represents a dataset, the first parameter represents the path of the data folder, and the second parameter represents the path of the `data.json` file, separated by `,`. Make the following modification:

    ```text
    ./dataset,./dataset/data.json
    ```

2. Configure `data.json`.

    Modify the `examples/wan2.2/A14B/t2v/data.json` file and configure the following parameters according to the actual situation:
    - `num_frames`: maximum number of frames. The default value is 49. If it is exceeded, `num_frames` frames are randomly selected.
    - `max_height`: maximum height. The default value is 480. If exceeded, it will be center-cropped to the maximum resolution.
    - `max_width`: maximum width, defaulting to 832. If exceeded, the image is center-cropped to the maximum resolution.
    - `from_pretrained`: path of the tokenizer weights. The reference path is "weights/Wan2.2-T2V-A14B-Diffusers/tokenizer".

    ```json
    "preprocess_parameters": {
        ......
        "num_frames": 81,
        "max_height": 480,
        "max_width": 832,
        ......
    "tokenizer_config":
    {
        ......
        "from_pretrained": "weights/Wan2.2-T2V-A14B-Diffusers/tokenizer",
        ......
    }
    }
    ```

3. Configure `model_t2v.json`.

    Modify the `examples/wan2.2/A14B/t2v/pretrain_model_high.json, *examples/wan2.2/A14B/t2v/pretrain_model_low.json*` file, where `from_pretrained` is the path of the downloaded weights, including vae and text_encoder. Modify the parameters according to the actual situation:

    ```json
    {
        "ae": {
            ......
            "from_pretrained": "weights/Wan2.2-T2V-A14B-Diffusers/vae",
            ......
        },
        "text_encoder": {
            ......
            "from_pretrained": "weights/Wan2.2-T2V-A14B-Diffusers/text_encoder"
        }
    }
    ```

4. Configure `pretrain_high.sh` and `pretrain_low.sh`.

    An example of a high-noise model fine-tuning script for a single node with 8 devices is as follows (`examples/wan2.2/A14B/t2v/pretrain_high.sh`). Note that you need to change LOAD_PATH to the actual path where the MM-format weights are stored and change `SAVE_PATH` to the path for saving the fine-tuned weights.

    ```sh
    #!/bin/bash
    # Modify the ascend-toolkit path as required
    source /usr/local/Ascend/cann/set_env.sh
    # This variable is only used to bypass the Megatron check and has no effect on NPUs
    export CUDA_DEVICE_MAX_CONNECTIONS=2 # When FSDP2 is enabled, the value cannot be set to 1
    export ASCEND_SLOG_PRINT_TO_STDOUT=0
    export ASCEND_GLOBAL_LOG_LEVEL=3
    export TASK_QUEUE_ENABLE=1
    export COMBINED_ENABLE=1
    export CPU_AFFINITY_CONF=1
    export HCCL_CONNECT_TIMEOUT=1200
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    NPUS_PER_NODE=8
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

    TP=1
    PP=1
    VP=1
    CP=1
    MBS=1
    GRAD_ACC_STEP=1
    DP=$(($WORLD_SIZE/$TP/$PP/$CP))
    GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

    MM_DATA="./examples/wan2.2/A14B/t2v/data.json"
    MM_MODEL="./examples/wan2.2/A14B/t2v/pretrain_model_high.json"
    MM_TOOL="./mindspeed_mm/tools/tools.json"
    LOAD_PATH="./weights/Wan2.2-T2V-A14B-Diffusers/transformer/"
    SAVE_PATH="path to save your high noise expert wandit weights"
    FSDP_CONFIG="./examples/wan2.2/A14B/fsdp2_config.yaml"

    DISTRIBUTED_ARGS="
        --nproc_per_node $NPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "

    GPT_ARGS="
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --virtual-pipeline-model-parallel-size ${VP} \
        --context-parallel-size ${CP} \
        --context-parallel-algo ulysses_cp_algo \
        --micro-batch-size ${MBS} \
        --global-batch-size ${GBS} \
        --num-workers 8 \
        --lr 1e-5 \
        --min-lr 1e-5 \
        --adam-beta1 0.9 \
        --adam-beta2 0.999 \
        --adam-eps 1e-8 \
        --lr-decay-style constant \
        --weight-decay 1e-2 \
        --lr-warmup-init 0 \
        --lr-warmup-iters 0 \
        --clip-grad 1.0 \
        --train-iters 5000 \
        --no-gradient-accumulation-fusion \
        --no-load-optim \
        --no-load-rng \
        --no-save-optim \
        --no-save-rng \
        --bf16 \
        --distributed-timeout-minutes 20 \
        --use-fused-rmsnorm \
        --use-torch-fsdp2 \
        --untie-embeddings-and-output-weights \
        --fsdp2-config-path ${FSDP_CONFIG} \
        --optimizer-selection fused_torch_adamw \
        --use-cpu-initialization \
    "

    MM_ARGS="
        --mm-data $MM_DATA \
        --mm-model $MM_MODEL \
        --mm-tool $MM_TOOL
    "

    OUTPUT_ARGS="
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 10000 \
        --eval-iters 10 \
        --load $LOAD_PATH \
        --save $SAVE_PATH \
        --ckpt-format torch_dcp \
    "

    logfile=wan_high_$(date +%Y%m%d)_$(date +%H%M%S)
    mkdir -p logs
    torchrun $DISTRIBUTED_ARGS pretrain_sora.py \
        $GPT_ARGS \
        $MM_ARGS \
        $OUTPUT_ARGS \
        --distributed-backend nccl \
        2>&1 | tee logs/train_${logfile}.log
    ```

### Training

Run the following command to start fine-tuning according to the script configured in the previous step:

```bash
bash examples/wan2.2/A14B/t2v/pretrain_high.sh
```

### Subsequent Procedure

If you need to convert weights back to Hugging Face format, run the weight conversion script:

```shell
mm-convert WanConverter mm_to_hf \
--cfg.source_path <path for your saved weight/> \
--cfg.target_path ./converted_weights/Wan2.2-T2V-A14B-Diffusers/transformer/ \
--cfg.hf_dir weights/Wan2.2-T2V-A14B-Diffusers/transformer/
```

>[!CAUTION]
>
> If training is based on Layer ZeRO, you must first convert back to the initial weights before performing the above operations.

## Reference

For more details about multimodal understanding models, see the [Qwen2_5_VL Usage Guide](https://gitcode.com/Ascend/MindSpeed-MM/tree/26.1.0/examples/qwen2.5vl/README.md).

For more details about multimodal generation models, see the [Wan2.2 Usage Guide](../../../../examples/wan2.2/README.md).
