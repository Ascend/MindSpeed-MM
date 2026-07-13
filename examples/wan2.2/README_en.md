# Wan2.2 User Guide

- [Wan2.2 User Guide](#wan22-user-guide)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Supported Task List](#supported-task-list)
  - [Environment Setup](#environment-setup)
    - [Repository Cloning](#repository-cloning)
    - [Environment Setup](#environment-setup-1)
    - [Decord Setup](#decord-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [Diffusers Weight Download](#diffusers-weight-download)
    - [Weight Conversion](#weight-conversion)
  - [Pre-training](#pre-training)
    - [Data Preprocessing](#data-preprocessing)
    - [Training](#training)
      - [Preparation](#preparation)
      - [Parameter Configuration](#parameter-configuration)
      - [Starting Training](#starting-training)
    - [LoRA Fine-tuning](#lora-fine-tuning)
      - [Preparation](#preparation-1)
      - [LoRA Weight Conversion (Optional)](#lora-weight-conversion-optional)
      - [Parameter Configuration](#parameter-configuration-1)
      - [LoRA Weight Loading (Optional)](#lora-weight-loading-optional)
      - [Starting Fine-tuning](#starting-fine-tuning)
  - [Inference](#inference)
    - [Preparation](#preparation-2)
    - [Parameter Configuration](#parameter-configuration-2)
    - [Launch Inference](#launch-inference)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Description

### Reference Implementation

Pre-training Task

5B:

```shell
url=https://github.com/modelscope/DiffSynth-Studio.git
commit_id=f0ea049
```

A14B:

```shell
url=https://github.com/modelscope/DiffSynth-Studio.git
commit_id=833ba1e
```

Inference:

```shell
url=https://github.com/huggingface/diffusers/tree/v0.35.1
```

### Changelog

2025.10.11: Initial support for Wan2.2 model

## Supported Task List

| Model Size | Task Type | Pre-training | Online T2V Inference | Online I2V Inference |
|------|:----:|:----|:-----|:-----|
| 5B | t2v  | ✔ | ✔ |  |
| 5B | ti2v  | ✔ |  | ✔ |
| A14B  | t2v  | ✔ | ✔ |  |
| A14B  | i2v  | ✔ |  | ✔ |

## Environment Setup

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md).

### Repository Cloning

Clone the MindSpeed MM repository and navigate to the root directory of the repository:

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

### Environment Setup

Execute the following command:

```bash
bash scripts/install.sh --megatron --msid 96bc0a3bf3398bf45ac26e0bded95ee174ac449b && pip install -r examples/wan2.2/requirements.txt
```

### Decord Setup

[X86 Installation]

```bash
pip install decord==0.6.0
```

[Arm Installation]

For installation via `apt`, please [refer to the link](https://github.com/dmlc/decord).

For installation via `yum`, please [refer to the script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

## Weight Download and Conversion

### Diffusers Weight Download

|   Model   |   Hugging Face Download Link   |
| ---- | ---- |
|   5B   |   <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers>   |
|  T2V-14B    |  <https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers>    |
|  I2V-14B  |   <https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers>   |

### Weight Conversion

The `transformer` weights of the downloaded Wan2.2 model need to be converted. Run the weight conversion script:

```shell
mm-convert WanConverter hf_to_mm \
 --cfg.source_path ./weights/Wan-AI/Wan2.2-{TI2V/T2V/I2V}-{5/A14}B-Diffusers/transformer* \
 --cfg.target_path ./weights/Wan-AI/Wan2.2-{TI2V/T2V/I2V}-{5/A14}B-Diffusers/transformer*
```

By further converting the weights to DCP format and loading the checkpoint in a distributed manner at startup, the peak memory pressure on the host side can be reduced (optional). The conversion command is as follows:

```shell
mm-convert WanConverter mm_to_dcp \
 --cfg.source_path ./weights/Wan-AI/Wan2.2-{TI2V/T2V/I2V}-{5/A14}B-Diffusers/transformer* \
 --cfg.target_path ./weights/Wan-AI/Wan2.2-{TI2V/T2V/I2V}-{5/A14}B-Diffusers/transformer*
```

The parameters of the weight conversion script are described below:

| Parameter         | Meaning                                  |
| :---------------- | :--------------------------------------- |
| --cfg.source_path | Original weight path                     |
| --cfg.target_path | Path to save the converted or sharded weights |

If you need to convert back to the Hugging Face format, run the weight conversion script:

**Note**: wan2.2 uses fsdp2 for training. You must first perform its [training weight post-processing](#jump1) before proceeding with the following operations:

```shell
mm-convert WanConverter mm_to_hf \
 --cfg.source_path path_for_your_saved_weight \
 --cfg.target_path ./converted_weights/Wan-AI/Wan2.2-{TI2V/T2V/I2V}-{5/A14}B-Diffusers/transformer* \
 --cfg.hf_dir weights/Wan-AI/Wan2.2-{TI2V/T2V/I2V}-{5/A14}B-Diffusers/transformer*
```

The parameters of the weight conversion script are described as follows:

| Parameter          | Description                                                                 |
|:-------------------|:----------------------------------------------------------------------------|
| --cfg.source_path  | Path where the MindSpeed MM weights are saved                               |
| --cfg.target_path  | Path for the converted Hugging Face weights                                 |
| --cfg.hf_dir       | Path to the original Hugging Face weights, from which the original Hugging Face configuration files are obtained |

**Note**: For the A14B model, the Hugging Face Diffusers weights contain two transformer weights. In the suffix, `transformer` corresponds to the high-noise (high) model, and `transformer_2` corresponds to the low-noise (low) model.

## Pre-training

### Data Preprocessing

Process the data into the following format:

```bash
</dataset>
  ├──data.json
  ├──videos
  │  ├──video0001.mp4
  │  ├──video0002.mp4
```

`videos/` stores the videos, and `data.json` contains all the video-text pair information in the dataset. A specific example is as follows:

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
    ......
]
```

Modify the `examples/wan2.2/data.txt` file, where each line represents a dataset. The first parameter indicates the path to the data folder, and the second parameter indicates the path to the `data.json` file, separated by `,`.

### Training

#### Preparation

Before starting, please confirm that environment preparation, model weight download, and data preprocessing have been completed.

#### Parameter Configuration

Verify that the dataset path, model weight path, parallel parameter configuration, etc., are completed.

| Configuration File   |      Field to be Modified       | Modification Notes      |
| --- | :---: | :--- |
| examples/wan2.2/{model_size}/{task}/data.json            |  from_pretrained  | Modify to the path corresponding to the downloaded tokenizer weights. |
| examples/wan2.2/{model_size}/{task}/pretrain_model*.json |  from_pretrained  | Modify to the path corresponding to the downloaded weights (including vae and text_encoder). |
| examples/wan2.2/{model_size}/{task}/pretrain*.sh         |    NPUS_PER_NODE  | Number of NPUs per node                                     |
| examples/wan2.2/{model_size}/{task}/pretrain*.sh         |       NNODES      | Number of nodes                                          |
| examples/wan2.2/{model_size}/{task}/pretrain*.sh         |      LOAD_PATH    | Path to the pre-trained weight after weight conversion                          |
| examples/wan2.2/{model_size}/{task}/pretrain*.sh         |      SAVE_PATH    | Path to weights saved during training                            |
| examples/wan2.2/{model_size}/{task}/pretrain*.sh         |        CP         | CP size during training (recommended to adjust it according to the resolution set during training)   |

**Note**:

1. When the current `LOAD_PATH` is invalid, MindSpeed will randomly initialize the model and train from scratch. To prevent loading failures, please pay attention to warning messages in the logs or verify the path validity yourself.
2. When using the resumable training feature, you need to remove the following configurations: `--downcast-to-bf16`, `--no-load-optim`, `--no-load-rng`, `--no-save-optim`, and `--no-save-rng`.

[Parallel Parameter Configuration]

- CP: Sequence Parallelism

  - Usage Scenario: When the video sequence (resolution × number of frames) is large, it can be enabled to reduce memory usage.

  - How to Enable: Set CP to a value greater than 1 in the script, e.g., `CP=2`;

  - Constraints: The number of heads must be divisible by CP (see `num_heads` configured in `examples/mindsporewan2.1/{model_size}/{task}/pretrain_model.json`).

  - The default mode is Ulysses.

  - DiT-RingAttention: See [DiT-RingAttention](../../docs/en/features/dit_ring_attention.md).

  - DiT-USP: Ulysses + RingAttention. For details, see [DiT-USP](../../docs/en/features/dit_usp.md).

  - Note: wan2.2 uses full attention, corresponding to `general`, i.e., `--attention-mask-type general`.

- FSDP2

  - Use Case: When the model parameter scale is large, enabling FSDP2 can reduce static memory.

  - How to Enable: Add `--use-torch-fsdp2`, `--fsdp2-config-path ${fsdp2_config}`, `--untie-embeddings-and-output-weights`, and `--ckpt-format torch_dcp` to `GPT_ARGS` in `examples/wan2.2/{model_size}/{task}/pretrain.sh`. For the `fsdp2_config` configuration, please refer to [FSDP2 Feature Guide](../../docs/en/features/fsdp2.md).

  <a id="jump1"></a>
  - Training Weight Post-processing: When training with this feature is performed, the saved weights must be post-processed using the following conversion script before they can be used for inference:

    ```bash
    # Weight path saved after training
    save_path="./wandit_weight_save"
    iter_dir="$save_path/iter_$(printf "%07d" $(cat $save_path/latest_checkpointed_iteration.txt))"
    # Target path for weight conversion
    convert_dir="./dcp_to_torch"
    mkdir -p $convert_dir/release/mp_rank_00
    cp $save_path/latest_checkpointed_iteration.txt $convert_dir/
    echo "release" > $convert_dir/latest_checkpointed_iteration.txt
    python -m torch.distributed.checkpoint.format_utils dcp_to_torch "$iter_dir" "$convert_dir/release/mp_rank_00/model_optim_rng.pt"
    ```

+ Encoder Interleaved Offload
  - Usage Scenario: In training scenarios where NPU memory is a bottleneck, multi-step training input data can be encoded at once, and then the encoder can be offloaded to CPU. This prevents the text encoder from needing to reside in memory permanently, thereby reducing memory usage.
  Therefore, online training can be done without increasing memory consumption, avoiding manual offline feature extraction. Both T2V and I2V tasks are supported.
  - How to Enable: In `xxx_model.json`, set `encoder_offload_interval > 1`. It is recommended to set it greater than 10 based on the actual scenario to amortize the performance overhead caused by offloading.
  - Constraints: When enabled, it is recommended to increase `num_worker` for optimal performance. Simultaneous activation with Encoder-DP is supported.

#### Starting Training

[5B]

```bash
bash examples/wan2.2/{model_size}/{task}/pretrain.sh
```

[A14B]

```bash
bash examples/wan2.2/{model_size}/{task}/pretrain_{type}.sh
```

### LoRA Fine-tuning

LoRA fine-tuning for the Wan2.2 A14B T2V model is currently supported in FSDP2 scenarios. Please follow the steps below to prepare.

#### Preparation

The procedure for data processing, weight download, and conversion is the same as that in the "pre-training" section.

#### LoRA Weight Conversion (Optional)

If you need to load LoRA pre-trained weights saved from Diffsynth, you must first perform the following weight conversion on the LoRA weights.

```bash
mm-convert WanConverter lora_hf_to_mm \
 --cfg.source_path ./weights/Wan-AI/Wan2.2-T2V-A14B-lora \
 --cfg.target_path ./weights/Wan-AI/Wan2.2-T2V-A14B-lora-mm
```

Then, convert the weights to DCP format.

```shell
mm-convert WanConverter mm_to_dcp \
 --cfg.source_path ./weights/Wan-AI/Wan2.2-T2V-A14B-lora-mm \
 --cfg.target_path ./weights/Wan-AI/Wan2.2-T2V-A14B-lora-dcp
```

The parameters for the weight conversion script are described below:

| Parameter         | Meaning                                    |
| :---------------- | :----------------------------------------- |
| --cfg.source_path | Original weight path                       |
| --cfg.target_path | Path to save the converted or sharded weights |

#### Parameter Configuration

The parameter configuration is the same as in that in the "training" section. The following parameters specific to LoRA fine-tuning are additionally required.

| Configuration File                                             |         Field to be Modified       | Modification Notes                         |
|--------------------------------------------------|:-------------------:|:-----------------------------|
| examples/wan2.2/A14B/t2v/finetune_lora_{low/high}.sh |       lora-r        | Dimension of the LoRA update matrix                  |
| examples/wan2.2/A14B/t2v/finetune_lora_{low/high}.sh |     lora-alpha      | Controls the degree to which the decomposed matrices influence the original matrix. |
| examples/wan2.2/A14B/t2v/finetune_lora_{low/high}.sh | lora-target-modules | List of modules to which LoRA is applied                  |

#### LoRA Weight Loading (Optional)

If you need to load LoRA pre-trained weights saved from Diffsynth, you must add the converted LoRA pre-trained weight path in the launch script `examples/wan2.2/A14B/t2v/finetune_lora_{low/high}.sh` and modify `GPT_ARGS`. The relevant configuration modifications are as follows:

```shell
LOAD_PATH="./weights/Wan-AI/Wan2.2-T2V-A14B-Diffusers/transformer/"
LORA_PATH="./weights/Wan-AI/Wan2.2-T2V-A14B-lora-dcp"

# The original --load $LOAD_PATH \ needs to be replaced with --load-base-model $LOAD_PATH \.
GPT_ARGS="
...
  --load-base-model $LOAD_PATH \
  --load $LORA_PATH \
...
"
```

#### Starting Fine-tuning

```bash
bash examples/wan2.2/A14B/t2v/finetune_lora_{low/high}.sh
```

After fine-tuning is complete, you must first perform [Training Weight Post-processing](#jump1) on the saved LoRA weights, and then use the weight conversion tool to merge the trained LoRA weights with the original weights.

```bash
mm-convert WanConverter merge_lora_to_base \
 --cfg.source_path <./converted_weights/Wan-AI/Wan2.2-T2V-14B-Diffusers/transformer*/> \
 --cfg.target_path <./converted_weights/Wan-AI/Wan2.2-T2V-14B-Diffusers/transformer_merge/> \
 --cfg.lora_path <lora_save_path> \
 --lora_alpha 32 \
 --lora_rank 32
```

The parameters for the weight merging script are described as follows:

| Parameter         | Meaning                                                              |
| :---------------- | :------------------------------------------------------------------- |
| --cfg.source_path | Original Weight Path                                                 |
| --cfg.target_path | Path to save the merged weights                                      |
| --cfg.lora_path   | Path to save the LoRA weights                                        |
| --lora_alpha      | Controls the degree to which the decomposed matrices influence the original matrix |
| --lora_rank       | Dimension of the LoRA update matrix                                  |

## Inference

### Preparation

Before starting, please confirm that the environment setup and model weight download have been completed.

### Parameter Configuration

Check whether the configuration of the model weight path, parallelism parameters, etc., is complete.

| Configuration File                                                     |         Field to be Modified        | Modification Notes                                        |
|----------------------------------------------------------|:-------------------:|:--------------------------------------------|
| examples/wan2.2/{model_size}/{task}/inference_model.json |   from_pretrained   | Modify to the path corresponding to the downloaded weights (including vae, tokenizer, text_encoder) |
| examples/wan2.2/samples_t2v_prompts.txt                  |        File content         | Prompts for the T2V inference task; customizable; one prompt per line             |
| examples/wan2.2/samples_i2v_prompts.txt                  |        File content         | Prompts for the I2V inference task; customizable; one prompt per line             |
| examples/wan2.2/samples_i2v_images.txt                   |        File content         | Path to the first frame image for the I2V inference task; customizable; one image path per line               |
| examples/wan2.2/{model_size}/{task}/inference_model.json |      save_path      | Save path to the generated video                                   |
| examples/wan2.2/{model_size}/{task}/inference_model.json |     input_size      | Resolution of the generated video in the format of [t, h, w]                      |
| examples/wan2.2/{model_size}/{task}/inference_model.json | low_noise_predictor | Weight path to the converted transformer_2 (low-noise) part; only applicable to the A14B model     |
| examples/wan2.2/{model_size}/{task}/inference.sh         |      LOAD_PATH      | Weight path for the converted transformer part; only applicable to the high-noise model for the A14B model       |

### Launch Inference

```shell
bash examples/wan2.2/{model_size}/{task}/inference.sh
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
