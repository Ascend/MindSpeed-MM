# Quick Start

MindSpeed MM supports both multimodal generation and understanding models. The following sections use two typical models, Qwen2.5-VL (understanding model) and Wan2.1 (generation model), as examples to introduce the usage of MindSpeed MM, helping you to quickly get started with running preset models efficiently on Ascend NPUs.

## Multimodal Understanding Model

This chapter uses Qwen2.5-VL-3B as an example to describe how to complete the fine-tuning of a multimodal understanding model in a single-node scenario.

### Environment Preparation

1. Set up the model training environment based on the PyTorch framework and Python 3.10. For details, please refer to the [MindSpeed MM Installation Guide](./install_guide.md).
2. Create the following directories under `MindSpeed-MM` to store logs, data, and weight files.

    ```bash
    mkdir logs
    mkdir data
    mkdir ckpt
    ```

### Weight Download and Conversion

1. Download weights.

   Download the corresponding model weights [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main) from Hugging Face.

2. Save weights.

   Create the `ckpt/hf_path/Qwen2.5-VL-3B-Instruct` directory and save the downloaded model weights into it.

3. Convert weights.

   MindSpeed MM has modified the structure names of some original networks. You can use the `mm-convert` tool to convert the original pre-trained weights. Execute the following command to run the tool:

    ```bash
    # Qwen2.5-VL-3B
    mm-convert  Qwen2_5_VLConverter hf_to_mm \
    --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
    --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
    --cfg.parallel_config.llm_pp_layers [[36]] \
    --cfg.parallel_config.vit_pp_layers [[32]] \
    --cfg.parallel_config.tp_size 1
    ```

   **Table 1** Parameters

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
    > The weight conversion logic of Qwen2_5_VL and Qwen2_VL is consistent. For more tool details, see [Weight Conversion Tool Usage](../features/mm_convert.md).

### Data Preprocessing

1. Download a dataset (COCO2017 as an example).

    Create the `data/COCO2017` directory and then download and extract the [COCO2017](https://cocodataset.org/#download) dataset.

2. Obtain the dataset description file.

   Download the image dataset description file [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) from Hugging Face and save it to the `./data/` path.

3. Pre-process data.

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
    > To avoid conflicts caused by writing to the same file, do not configure the same mount directory (`cache_dir`) across multiple nodes.

2. Edit the fine-tuning script.

    ```shell
    vi examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh
    ```

3. Configure model saving, loading, and logging information.

    Complete the configuration of the model load path, save path, and save interval (`--save-interval`). A configuration example is as follows:

    ```bash
    ...
    # Load Path
    LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-3B-Instruct"
    # Save Path
    SAVE_PATH="save_dir"
    ...
    GPT_ARGS="
        ...
        --no-load-optim \  # Do not load optimizer state. Remove this if loading is required.
        --no-load-rng \  # Do not load random number state. Remove this if loading is required.
        --no-save-optim \  # Do not save optimizer state. Remove this if saving is required.
        --no-save-rng \  # Do not save the random number state. Remove this if saving is required.
        ...
    "
    ...
    OUTPUT_ARGS="
        --log-interval 1 \  # Log Interval
        --save-interval 5000 \  # Save Interval
        ...
        --log-tps \  # Adding this parameter enables printing the average sequence length of the language module at each step during training, and calculates the throughput in tokens per second after training ends.
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

This chapter uses Wan2.1-T2V-1.3B as an example to describe how to complete the pre-training of a multimodal generation model in a single-node scenario.

### Environment Preparation

1. Set up the model training environment based on the PyTorch framework and Python 3.10. For details, please refer to the [MindSpeed MM Installation Guide](./install_guide.md).

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

   Download the corresponding [Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tree/main) weights from Hugging Face.

2. Save weights.

   Create `weights/Wan2.1-T2V-1.3B-Diffusers/` under `MindSpeed-MM` and save the downloaded model weights to this directory.

3. Convert weights.

The `transformer` part of the downloaded Wan2.1 model weights needs to undergo weight conversion. Run the weight conversion tool:

    ```shell
    mm-convert WanConverter hf_to_mm \
    --cfg.source_path ./weights/Wan2.1-T2V-1.3B-Diffusers/transformer/ \
    --cfg.target_path ./weights/Wan2.1-T2V-1.3B-Diffusers/transformer_mm/
    ```

**Table 5** Parameters

| Parameter | Description |
| :-- | :--- |
| `WanConverter` | Wan2.1 model conversion tool |
| `hf_to_mm` | Converts Hugging Face weights to MindSpeed MM weights. |
| `source_path` | Original weight path |
| `target_path` | Save path for converted or partitioned weights |

### Data Preprocessing

Create a `dataset` directory under `MindSpeed-MM`, then create a `videos` directory and a `data.json` file inside `dataset`, and save the videos to be processed in `videos`. All video-text pair information in the dataset is saved in `data.json`.

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
    },
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

### Feature Extraction

1. Configure `data.txt`.

    Modify the `examples/wan2.1/feature_extract/data.txt` file, where each line represents a dataset. The first parameter indicates the path to the data folder, and the second parameter indicates the path to the `data.json` file, separated by `,`. Make the following modifications:

    ```text
    ./dataset,./dataset/data.json
    ```

2. Configure `data.json`.

    Modify the `examples/wan2.1/feature_extract/data.json` file and configure the following parameters based on the actual situation:
    - `num_frames`: maximum number of frames. The default value is 81. If exceeded, `num_frames` frames will be randomly selected.
    - `max_height`: maximum height. The default value is 480. If exceeded, it will be center-cropped to the maximum resolution.
    - `max_width`: maximum width, defaulting to 832. If exceeded, the image is center-cropped to the maximum resolution.
    - `from_pretrained`: path corresponding to the tokenizer weights, defaulting to `weights/Wan2.1-T2V-1.3B-Diffusers/tokenizer`.

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
        "from_pretrained": "weights/Wan2.1-T2V-1.3B-Diffusers/tokenizer",
        ......
    }
    }
    ```

3. Configure `model_t2v.json`.

    Modify the `examples/wan2.1/feature_extract/model_t2v.json` file, where `from_pretrained` is the path corresponding to the downloaded weights, including the vae and text_encoder. Modify the parameters according to the actual situation:

    ```json
    {
        "ae": {
            ......
            "from_pretrained": "weights/Wan2.1-T2V-1.3B-Diffusers/vae",
            ......
        },
        "text_encoder": {
            ......
            "from_pretrained": "weights/Wan2.1-T2V-1.3B-Diffusers/text_encoder"
        }
    }
    ```

4. Configure `tools.json`.

    Modify `mindspeed_mm/tools/tools.json`, where the `save_path` of `sorafeature` is the save path for the extracted features:

    ```json
        "sorafeature":{
        "save_path": "./sora_features"
    }
    ```

5. Configure the feature extraction script.

    Modify `NPUS_PER_NODE` in `examples/wan2.1/feature_extract/feature_extraction.sh`. The default value is 1; please change it to the actual number of NPUs used.

6. Start feature extraction.

    ```bash
    bash examples/wan2.1/feature_extract/feature_extraction.sh
    ```

### Training

1. Check parameter configuration.

    Confirm that the modifications for all fields in the configuration files listed in the table below have been completed.

    **Table 7** Fields

    | Configuration File   |      Field       | Modification Notes      |
    | --- | :---: | :--- |
    | examples/wan2.1/1.3b/t2v/data.txt    | File content  | Save path for extracted features |
    | examples/wan2.1/1.3b/t2v/feature_data.json   |   `from_pretrained`   | Modify to the path corresponding to the downloaded weights, consistent with that configured in [Weight Download and Conversion](#weight-download-and-conversion). |
    | examples/wan2.1/1.3b/t2v/pretrain.sh |    `NPUS_PER_NODE`    | Number of NPUs per node                                      |
    | examples/wan2.1/1.3b/t2v/pretrain.sh |       `NNODES`        | Number of nodes                                            |
    | examples/wan2.1/1.3b/t2v/pretrain.sh |      `LOAD_PATH`     | Path to the pre-training weights after weight conversion, consistent with that configured in [Weight Download and Conversion](#weight-download-and-conversion)                         |
    | examples/wan2.1/1.3b/t2v/pretrain.sh |      `SAVE_PATH`      | Path for weights saved during training                            |
    | examples/wan2.1/1.3b/t2v/pretrain.sh |         CP          | CP size during training (recommended to adjust based on the resolution set for training)   |

2. Start training.

    Modify the tokenizer weight path in `feature_data.json`.

    ```bash
    bash examples/wan2.1/1.3b/t2v/pretrain.sh
    ```

### Subsequent Procedure

If you need to convert weights back to Hugging Face format, run the weight conversion script:

```shell
mm-convert WanConverter mm_to_hf \
--cfg.source_path <path for your saved weight/> \
--cfg.target_path ./converted_weights/Wan2.1-T2V-1.3B-Diffusers/transformer/
--cfg.hf_dir weights/Wan2.1-T2V-1.3B-Diffusers/transformer/
```

>[!CAUTION]
>If training is based on Layer ZeRO, you must first convert back to the initial weights before performing the above operations.

## Reference

[Qwen2.5-VL User Guide](../../../examples/qwen2.5vl/README.md).

[Wan2.1 User Guide](../../../examples/wan2.1/README.md).
