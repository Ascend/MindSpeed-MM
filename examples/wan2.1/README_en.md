# Wan2.1 Usage Guide

- [Wan2.1 Usage Guide](#wan21-usage-guide)
  - [Version Description](#version-description)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Supported Task List](#supported-task-list)
  - [Environment Installation](#environment-installation)
    - [Repository Pull](#repository-pull)
    - [Environment Setup](#environment-setup)
    - [Decord Setup](#decord-setup)
  - [Weight Download and Offline Conversion](#weight-download-and-offline-conversion)
    - [Diffusers Weight Download](#diffusers-weight-download)
    - [Weight Conversion](#weight-conversion)
  - [Weight Download and Online Loading](#weight-download-and-online-loading)
    - [Diffusers Weight Download](#diffusers-weight-download-1)
    - [Online Loading](#online-loading)
  - [Pre-training](#pre-training)
    - [Data Preprocessing](#data-preprocessing)
    - [Feature Extraction](#feature-extraction)
      - [Preparation](#preparation)
      - [Parameter Configuration](#parameter-configuration)
      - [Start Feature Extraction](#start-feature-extraction)
    - [Training](#training)
      - [Preparation](#preparation-1)
      - [Parameter Configuration](#parameter-configuration-1)
      - [Start Training](#start-training)
  - [LoRA Fine-tuning](#lora-fine-tuning)
    - [Preparation](#preparation-2)
    - [Parameter Configuration](#parameter-configuration-2)
    - [Start Fine-tuning](#start-fine-tuning)
  - [DPO Training](#dpo-training)
    - [Environment Preparation](#environment-preparation)
    - [Video Sample Generation](#video-sample-generation)
    - [Preference Dataset Generation](#preference-dataset-generation)
    - [Training Parameter Configuration](#training-parameter-configuration)
    - [Start DPO Training](#start-dpo-training)
  - [Inference](#inference)
    - [Preparation](#preparation-3)
    - [Parameter Configuration](#parameter-configuration-3)
    - [Start Inference](#start-inference)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Description

### Reference Implementation

T2V/I2V LoRA fine-tuning tasks:

```shell
url=https://github.com/modelscope/DiffSynth-Studio.git
commit_id=03ea278
```

FLF2V inference:

```shell
url=https://github.com/huggingface/diffusers.git
commit_id=f8d4a1e
```

### Changelog

2025.03.27: Initial support for Wan2.1 model

## Supported Task List

| Model Size | Task Type | Pre-training | LoRA Fine-tuning | Online T2V Inference | Online I2V Inference | Online FLF2V Inference | Online V2V Inference |
|------|:----:|:----|:-------|:-----|:-----|:-----|:-----|
| 1.3B | t2v  | ✔ | ✔ | ✔ |  |  | ✔ |
| 1.3B | i2v  | ✔ |  |  |  |  |  |
| 14B  | t2v  | ✔ | ✔ | ✔ |  |  | ✔ |
| 14B  | i2v  | ✔ | ✔ |  | ✔ |  |  |
| 14B  | flf2v|   |  |  |  | ✔ |  |

## Environment Installation

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide]../../docs/en/pytorch/install_guide.md).

### Repository Pull

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ../MindSpeed-MM
```

### Environment Setup

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# Install torch and torch_npu. Make sure to select the torch, torch_npu, and apex packages that match the corresponding Python version and x86 or ARM architecture.
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
git checkout 93c45456c7044bacddebc5072316c01006c938f9
pip install -r requirements.txt
pip install -e .
cd ..

# Install the required dependency libraries.
pip install -e .

# Install Diffusers from source.
pip install diffusers==0.33.1
```

### Decord Setup

[X86 Installation]

```bash
pip install decord==0.6.0
```

[ARM Installation]

For installation via `apt`, please [refer to the link](https://github.com/dmlc/decord).

For installation via `yum`, please [refer to the script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

## Weight Download and Offline Conversion

### Diffusers Weight Download

|   Model   |   Hugging Face Download Link   |
| ---- | ---- |
|   T2V-1.3B   |   <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>   |
|  T2V-14B    |  <https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers>    |
|  I2V-14B-480P  |   <https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers>   |
|  I2V-14B-720P  |   <https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers>   |
|  FLF2V-14B-720P |   <https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers>   |

### Weight Conversion

The `transformer` part of the downloaded Wan2.1 model weights needs to be converted. Run the weight conversion script:

```shell
mm-convert WanConverter hf_to_mm \
 --cfg.source_path <./weights/Wan-AI/Wan2.1-{T2V/I2V/FLF2v}-{1.3/14}B-Diffusers/transformer/> \
 --cfg.target_path <./weights/Wan-AI/Wan2.1-{T2V/I2V/FLF2v}-{1.3/14}B-Diffusers/transformer/> \
 --cfg.target_parallel_config.pp_layers <pp_layers>
```

The parameters of the weight conversion script are described as follows:

| Parameter         | Meaning                                      | Default Value                                                |
| :---------------- | :------------------------------------------- | :----------------------------------------------------------- |
| --cfg.source_path | Original weight path                         | /                                                            |
| --cfg.target_path | Save path for converted or sharded weights     | /                                                            |
| --pp_layers       | Number of PP/VPP layers                      | When PP is enabled, using PP and VPP requires specifying the number of layers at each stage and conversion. The default is `[]`, meaning not used. |

If you need to convert back to the Hugging Face format, run the weight conversion script:

**Note**: If training with LayerZeRO is performed, you must first perform [training weight post-processing](#jump1), then proceed with the following operations:

```shell
mm-convert WanConverter mm_to_hf \
 --cfg.source_path <path for your saved weight/> \
 --cfg.target_path <./converted_weights/Wan-AI/Wan2.1-{T2V/I2V/FLF2v}-{1.3/14}B-Diffusers/transformer/> \
 --cfg.hf_dir <weights/Wan-AI/Wan2.1-{T2V/I2V/FLF2v}-{1.3/14}B-Diffusers/transformer/>
```

The parameter descriptions for the weight conversion script are as follows:

| Parameter | Meaning | Default Value |
|:------------|:----|:----|
| --cfg.source_path | Path to MindSpeed MM saved weights | / |
| --cfg.target_path | Path to converted Hugging Face weights | / |
| --cfg.hf_dir | Path to the original Hugging Face weights, from which the original Hugging Face configuration files are obtained | / |

## Weight Download and Online Loading

### Diffusers Weight Download

| Model (Verified) | Hugging Face Download Link |
|----------| ---- |
| T2V-1.3B | <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers> |
| T2V-14B | <https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers> |

### Online Loading

If you need to use online weight loading mode for model training, simply assign the downloaded Hugging Face original weights to the `LOAD_PATH` parameter in `examples/wan2.1/14b/t2v/pretrain_fsdp2.sh`:

```shell
LOAD_PATH="./weights/Wan-AI/Wan2.1-T2V-14B-Diffusers/transformer/"
```

Also, set `bridge_patch` in `examples/wan2.1/14b/t2v/pretrain_fsdp2.sh` to `true`.

```shell
    "patch": {
        "bridge_patch": true
    }
```

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

Here, `videos/` stores the videos, and `data.json` contains all video-text pair information in the dataset. A specific example is as follows:

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

Modify the `examples/wan2.1/feature_extract/data.txt` file, where each line represents a dataset. The first parameter indicates the path to the data folder, and the second parameter indicates the path to the `data.json` file, separated by `,`.

### Feature Extraction

#### Preparation

Before starting, please ensure that environment preparation, model weights, and dataset preprocessing have been completed.

#### Parameter Configuration

Check whether configurations such as the model weight path, dataset path, and save path for extracted features are complete.

| Configuration File   |   Field to be Modified  | Modification Notes  |
| --- | :---: | :--- |
| examples/wan2.1/feature_extract/data.json              |      num_frames       | Maximum number of frames; if exceeded, randomly selects num_frames frames        |
| examples/wan2.1/feature_extract/data.json              | max_height, max_width | Maximum height and width; if exceeded, center crops to the maximum resolution            |
| examples/wan2.1/feature_extract/data.json              |    from_pretrained    | Modify to the path corresponding to the downloaded tokenizer weight            |
| examples/wan2.1/feature_extract/feature_extraction.sh  |     NPUS_PER_NODE     | Number of NPUs                                                |
| examples/wan2.1/feature_extract/feature_extraction.sh  |     MM_MODEL          | Modify to the model file path for the target task, e.g., model_t2v.json    |
| examples/wan2.1/feature_extract/model_{task}.json      |    from_pretrained    | Modify to the path corresponding to the downloaded weights (including vae, text_encoder) |
| mindspeed_mm/tools/tools.json                          |       save_path       | Save path for extracted features                                |

#### Start Feature Extraction

```bash
bash examples/wan2.1/feature_extract/feature_extraction.sh
```

### Training

#### Preparation

Before starting, please confirm that environment preparation, model weight download, and feature extraction have been completed.

#### Parameter Configuration

Check whether the model weight path, parallel parameter configuration, etc. are complete

| Configuration File | Modify Field | Modification Description |
| --- | :---: | :--- |
| examples/wan2.1/{model_size}/{task}/feature_data.json | basic_parameters |  Dataset path. Set `data_path` and `data_folder` to the file path and directory of the extracted features, respectively. |
| examples/wan2.1/{model_size}/{task}/pretrain.sh | NPUS_PER_NODE | Number of NPUs per node |
| examples/wan2.1/{model_size}/{task}/pretrain.sh | NNODES | Number of nodes |
| examples/wan2.1/{model_size}/{task}/pretrain.sh | LOAD_PATH | Path to the pre-trained weights after weight conversion |
| examples/wan2.1/{model_size}/{task}/pretrain.sh | SAVE_PATH | Path to weights saved during training |
| examples/wan2.1/{model_size}/{task}/pretrain.sh | CP | CP size during training (recommended to adjust it based on the resolution set during training) |

[Parallel Parameter Configuration]

When adjusting model parameters or video sequence length, the following parallel strategies need to be enabled based on actual conditions, and the optimal parallel strategy should be determined through debugging.

- CP: Sequence Parallelism

  - Usage Scenario: When the video sequence (resolution × number of frames) is large, it can be enabled to reduce memory usage.

  - Enablement Method: Set CP to a value greater than 1 in the script, e.g., `CP=2`;

  - Constraints: The number of heads must be divisible by CP (see `num_heads` configured in `examples/mindsporewan2.1/{model_size}/{task}/pretrain_model.json`).

  - The default mode is Ulysses.

  - DiT-RingAttention: See [DiT-RingAttention](../../docs/en/features/dit_ring_attention.md).

  - DiT-USP: Ulysses + RingAttention. For details, see [DiT-USP](../../docs/en/features/dit_usp.md).

  - FPDT (Fully Pipelined Distributed Transformer): Ulysses Offload. For details, see [FPDT](../../docs/en/features/fpdt.md).

  - Note: wan2.1 uses full attention, corresponding to `general`, i.e., `--attention-mask-type general`.

- layer_zero

  - Usage Scenario: When the model parameter scale is large and a single card cannot accommodate the complete model, you can enable layerZeRO to reduce static memory.

  - Enablement Method: Add `--layerzero` and `--layerzero-config ${layerzero_config}` to `GPT_ARGS` in `examples/wan2.1/{model_size}/{task}/pretrain.sh`.

  <a id="jump1"></a>
  - Training Weight Post-processing: When training with this feature, the saved weights need to be post-processed using the following conversion script before they can be used for inference:

    ```bash
    # Modify the ascend-toolkit path according to the actual situation
    source /usr/local/Ascend/cann/set_env.sh
    mm-convert WanConverter layerzero_to_mm \
     --cfg.source_path <./save_ckpt/wan2.1/> \
     --cfg.target_path <./save_ckpt/wan2.1_megatron_ckpt/>
    ```

- PP: Pipeline Parallelism

  Currently, the predictor model can be partitioned into pipelines.

  - Usage Scenario: When the model parameters are large, they can be partitioned and parallelized through pipelining to reduce training memory usage.

  - Enablement Method:
    - Modify the `pipeline_num_layers` field in the `pretrain_model.json` file, which is of type list. The length of this list is the number of pipeline ranks, and each value represents the number of layers in `rank_i`. For example, `[7, 8, 8, 7]` means there are 4 pipeline stages, each accommodating 7/8 DiT layers. Note that the sum of all values in the list should equal the total `num_layers` field. In addition, the stage with `pp_rank==0` accommodated `text_encoder` and `ae` in addition to the DiT layers, so the number of DiT layers in the 0th stage can be reduced as appropriate. Ensure that the PP parameter configuration is consistent with the parameter configuration during model conversion.
    - Additionally, when using PP, enable the following parameters in `GPT_ARGS`:

    ```shell
    PP = 4 # PP > 1 Enabled
    GPT_ARGS="
    --optimization-level 2 \
    --use-multiparameter-pipeline-model-parallel \  # PP or VPP functionality must be enabled for use.
    --variable-seq-lengths \  # Enable as needed. This configuration is required for dynamic shape training; do not add it for static shape training.
    "
    ```

- VPP: Virtual Pipeline Parallelism

    Currently, the predictor model can be partitioned into virtual pipelines.

  - Usage Scenario: Further split pipelines to reduce pipeline bubbles through virtualization.
  - Enablement Method:
    - To enable VPP, change the one-dimensional `pipeline_num_layers` array in the `pretrain_model.json` file into a two-dimensional array, where the first dimension represents the number of virtual pipelines, and the second dimension represents the number of pipeline stages. For example, `[[3, 4, 4, 4], [3, 4, 4, 4]]` indicates that the first dimension has two arrays, meaning `VP=2`, and the second dimension has 4 stages, meaning `pp=3` or `pp=4`.
    - The following variables need to be modified in `pretrain.sh`. Note that VPP only takes effect when `PP` is greater than 1:

    ```shell
    PP=4
    VP=2

    GPT_ARGS="
      --pipeline-model-parallel-size ${PP} \
      --virtual-pipeline-model-parallel-size ${VP} \
      --optimization-level 2 \
      --use-multiparameter-pipeline-model-parallel \  # Must be enabled to use PP or VPP.
      --variable-seq-lengths \  # Enable as needed. This configuration is required for dynamic shape training but should not be added for static shape training.
    "
    ```

- Selective recomputation + FA activation offload

  - If the GPU memory is relatively sufficient, you can enable selective recomputation (self-attention does not perform recomputation) to improve throughput. It is recommended to simultaneously enable FA activation offloading to asynchronously offload FA activations to the CPU.

  - Selective recomputation

    - In `examples/wan2.1/{model_size}/{task}/pretrain.sh`, add the parameters `--recompute-skip-core-attention` and `--recompute-num-layers-skip-core-attention x` to enable selective recomputation. The number after `--recompute-num-layers-skip-core-attention` indicates the number of layers that skip self-attention computation, while the number after `--recompute-num-layers` indicates the number of layers with full recomputation. It is recommended to decrease `recompute-num-layers` while increasing `recompute-num-layers-skip-core-attention` until the GPU memory is fully utilized.

      ```bash
      GPT_ARGS="
          --recompute-granularity full \
          --recompute-method block \
          --recompute-num-layers 0 \
          --recompute-skip-core-attention \
          --recompute-num-layers-skip-core-attention 40 \
      "
      ```

  - Asynchronous offload of self-attention activations without recomputation
    - In `examples/wan2.1/{model_size}/{task}/pretrain_model.json`, asynchronous offload can be enabled via the `attention_async_offload` field. It is recommended to enable this feature to save more GPU memory.

- FSDP2

  - Usage Scenario: When the model parameter scale is large, static memory can be reduced by enabling FSDP2.

  - Enablement Method: Add `--use-torch-fsdp2`, `--fsdp2-config-path ${fsdp2_config}`, `--untie-embeddings-and-output-weights`, and `--ckpt-format torch_dist` to the `GPT_ARGS` in `examples/wan2.1/{model_size}/{task}/pretrain_fsdp2.sh`. For the `fsdp2_config` configuration, please refer to [FSDP2 Feature Guide](../../docs/en/features/fsdp2.md).

#### Start Training

```bash
bash examples/wan2.1/{model_size}/{task}/pretrain.sh
```

Or

```shell
bash examples/wan2.1/{model_size}/{task}/pretrain_fsdp2.sh
```

## LoRA Fine-tuning

### Preparation

The procedure of data processing, feature extraction, weight download and conversion is the same as that in the "pre-training" section.

### Parameter Configuration

The parameter configuration is the same as that in the "training" section, but the following parameters specific to LoRA fine-tuning are additionally required.

| Configuration File                                             |  Field to be Modified         | Modification Notes                        |
|--------------------------------------------------|:-------------------:|:-----------------------------|
| examples/wan2.1/{model_size}/{task}/finetune_lora.sh |       lora-r        | Dimension of the LoRA update matrix                  |
| examples/wan2.1/{model_size}/{task}/finetune_lora.sh |     lora-alpha      | Controls the degree to which the decomposed matrices influence the original matrix. |
| examples/wan2.1/{model_size}/{task}/finetune_lora.sh | lora-target-modules | List of modules to which LoRA is applied                  |

### Start Fine-tuning

```bash
bash examples/wan2.1/{model_size}/{task}/finetune_lora.sh
```

After fine-tuning is complete, you can use the weight conversion tool to merge the trained LoRA weights with the original weights.

```bash
mm-convert WanConverter merge_lora_to_base \
 --cfg.source_path <./converted_weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer/> \
 --cfg.target_path <./converted_weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer_merge/> \
 --cfg.lora_path <lora_save_path> \
 --lora_alpha 64 \
 --lora_rank 64
```

## DPO Training

Currently, only basic DPO training for the I2V task is supported, with more features to be refined later.

### Environment Preparation

1. Install VBench and its dependencies by referring to [VBench Evaluation](../../docs/zh/features/vbench-evaluate.md).
2. Download the [VBench T2V json](https://github.com/Vchitect/VBench/blob/master/vbench/VBench_full_info.json) to the MM code root path `"./vbench/VBench_full_info.json"`.

### Video Sample Generation

1. Modify the inference configuration file.

    | Parameter Configuration File                                                 |                Field to be Modified              | Modification Notes                          |
    |------------------------------------------------------------|:--------------------------------:|:----------------------------------|
    | examples/wan2.1/14b/i2v/inference_model.json      |         from_pretrained          | Modify to the path corresponding to the downloaded weights (including vae, tokenizer, and text_encoder). |
    | examples/wan2.1/14b/i2v/inference_model.json      |  num_inference_videos_per_sample | Number of video samples generated per prompt. It is recommended that the value be at least greater than 2.          |
    | examples/wan2.1/14b/i2v/inference_model.json        |  save_path | Save to generated videos                         |
    | examples/wan2.1/14b/i2v/inference.sh              |   LOAD_PATH | Path to the converted transform weights               |

    | I2V Prompt Configuration File                                   |               Field to be Modified              |       Modification Description       |
    |--------------------------------------------|:--------------------------------:|:----------------:|
    | examples/wan2.1/samples_i2v_images.txt  |               File content               |       Image path       |
    | examples/wan2.1/samples_i2v_prompts.txt |               File content               |    Custom prompt     |

2. Start the inference process to generate video samples.

    ```shell
    bash examples/wan2.1/14b/i2v/inference.sh
    ```

3. Delete `video_grid.mp4` in the video sample save path. The final number of video samples is `number of prompts * $num_inference_videos_per_sample`.

### Preference Dataset Generation

Execute the following command to score the generated video samples and generate the preference data file.

```bash
python examples/stepvideo/histogram_generator.py --prompt_file <prompt file path> --videos_path <video sample path> --num_inference_videos_per_sample <number of video samples generated per prompt>
```

The parameters of the preference dataset script are described as follows:

| Parameter | Meaning | How to Configure |
|:------------|:----|:----|
| --prompt_file | Prompt file path | Keep it consistent with `prompt` in the inference configuration file when generating video samples. |
| --videos_path | Video sample path | Keep it consistent with `save_path` in the inference configuration file when generating video samples. |
| --num_inference_videos_per_sample | Number of video samples generated per prompt | Keep it consistent with `num_inference_videos_per_sample` in the inference configuration file when generating video samples. |

After executing the script, the preference dataset file (`data.jsonl`) and the scoring probability histogram file "(`video_score_histogram.json`) will be generated, defaulting to the same directory level as the video sample directory.

`data.jsonl` contains pairs of video preference data and text information. A specific example is as follows:

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

re starting, please confirm that the environment preparation, model weight preparation, and preference data preparation have been completed.

1. Weight Configuration

    You need to add the actual path of the converted weights to the `LOAD_PATH="your_converted_dit_ckpt_dir"` variable in the startup script file `posttrain.sh` based on the actual task situation, for example, `LOAD_PATH="./weights/Wan-AI/Wan2.1-I2V-14B-Diffusers/transformer/"`, where `./weights/Wan-AI/Wan2.1-I2V-14B-Diffusers/transformer/` is the actual path of the converted weights. The complete path filled in the `LOAD_PATH` variable must be correct. An incorrect path will cause the weights to fail to load, but the run will not prompt an error.
    Fill in the path in the `SAVE_PATH` variable as needed to save the trained weights.

2. Preference Dataset Path Configuration

    Modify the preference dataset path in `feature_data.json` according to the actual situation: replace `"data_path": "./sora_features/data.jsonl"` with the actual path where `data.jsonl` is located, and replace `"data_folder": "./sora_features/"` with the actual path where the video samples are located.

3. VAE, text_encoder, and Tokenizer Path Configuration

    Modify the `from_pretrained` field in the `inference_model.json` file to configure the paths for VAE, text_encoder, and tokenizer according to the actual situation.

4. DPO Parameter Configuration

Modify the histogram file path in `posttrain_model.json` according to the actual situation, i.e., configure the value of `histogram_path` to the path of the `video_score_histogram.json` file generated after executing the generation preference dataset script.

### Start DPO Training

```bash
bash examples/wan2.1/14b/i2v/posttrain.sh
```

## Inference

### Preparation

Before starting, please confirm that the environment preparation and model weight download have been completed.

### Parameter Configuration

Check whether configurations such as the model weight path and parallel parameters are complete.

| Configuration File                                                     |  Field to be Modified  | Modification Notes |
|----------------------------------------------------------|:------:|:-----|
| examples/wan2.1/{model_size}/{task}/inference_model.json | from_pretrained |  Modify to the path corresponding to the downloaded weights (including VAE, tokenizer, and text_encoder).   |
| examples/wan2.1/samples_t2v_prompts.txt                  |    File content |  Prompts for T2V inference tasks; customizable; one prompt per line   |
| examples/wan2.1/samples_i2v_prompts.txt                  |    File content |  Prompts for I2V inference tasks; customizable; one prompt per line   |
| examples/wan2.1/samples_i2v_images.txt                   |    File content |  First frame image path for I2V inference tasks; customizable; one image path per line   |
| examples/wan2.1/samples_flf2v_prompts.txt                |    File content |  Prompts for FLF2V inference tasks; customizable; one prompt per line   |
| examples/wan2.1/samples_flf2v_images.txt                 |    File content |  First and last frame image path for FLF2V inference tasks; customizable; two image paths (first and last frames) per line, separated by ", "   |
| examples/wan2.1/samples_v2v_prompts.txt                  |    File content |  Prompts for V2V inference tasks; customizable; one prompt per line   |
| examples/wan2.1/samples_v2v_videos.txt                   |    File content |  First video path for V2V inference tasks; customizable; one video path per line   |
| examples/wan2.1/{model_size}/{task}/inference_model.json |  save_path |  Save path to generated videos |
| examples/wan2.1/{model_size}/{task}/inference_model.json |  dual_image |  Dual-frame inference input; `true` only in FLF2V tasks; not required for other tasks |
| examples/wan2.1/{model_size}/{task}/inference_model.json |  input_size |  Resolution of the generated video in the of format [t, h, w] |
| examples/wan2.1/{model_size}/{task}/inference_model.json |  flow_shift |  Scheduler parameter. `shift=3.0` recommended for 480P, `shift=5.0` for 720P, and `shift=16.0` for FLF2V tasks. |
| examples/wan2.1/{model_size}/{task}/inference.sh         |   LOAD_PATH | Path to the converted transformer weights |

### Start Inference

```shell
bash examples/wan2.1/{model_size}/{task}/inference.sh
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
