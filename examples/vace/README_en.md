# VACE User Guide

- [VACE User Guide](#vace-user-guide)
  - [Version Notes](#version-notes)
    - [Reference Implementation](#reference-implementation)
  - [Supported Task List](#supported-task-list)
  - [Environment Setup](#environment-setup)
    - [Repository Cloning](#repository-cloning)
    - [Environment Setup](#environment-setup-1)
    - [Decord Setup](#decord-setup)
  - [Weight Download and Conversion](#weight-download-and-conversion)
    - [Weight Download](#weight-download)
    - [Weight Conversion](#weight-conversion)
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
  - [Inference](#inference)
    - [Parameter Configuration](#parameter-configuration-2)
    - [Preparation](#preparation-2)
    - [Start Inference](#start-inference)
  - [Environment Variable Declaration](#environment-variable-declaration)

## Version Notes

### Reference Implementation

[Data Processing]

```shell
uel=https://github.com/ali-vilab/VACE
commit_id=0897c6d
```

[Training]

```shell
url=https://github.com/modelscope/DiffSynth-Studio.git
commit_id=8332ece
```

Modified some hyperparameters in DiffSynth-Studio:

| Parameter      | DiffSynth-Studio | MindSpeed-MM |
|--------------|:-----------------|:-------------|
| lr           | 1e-4             | 5e-5         |
| weight_decay | 0.01             | 0.1          |
| adam-beta2   | 0.999            | 0.95         |
| adam-eps     | 1e-8             | 1e-5         |

Experimental validation shows that the original parameter configuration of DiffSynth-Studio causes loss spikes during long runs on both GPU and NPU. The modified parameter settings resolve this issue: `lr` and `weight_decay` adopt the settings from the VACE paper, while `adam-beta2` and `adam-eps` are empirically adjusted based on experimental results. This configuration scheme can serve as a reference.

## Supported Task List

| Model Size             | Pre-training | Inference |
|------------------|:----|:----|
| Wan2.1-VACE-1.3B | ✔ | ✔ |
| Wan2.1-VACE-14B  | ✔ | ✔ |
| Wan2.2-VACE-A14B | ✔ | ✔ |

## Environment Setup

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md).

### Repository Cloning

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

# Install torch and torch_npu. Note that you need to select the torch, torch_npu, and apex packages corresponding to your Python version and x86 or arm architecture.
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

# Install other dependencies.
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

## Weight Download and Conversion

### Weight Download

| Model               | Hugging Face Download Link                                            |
|------------------|------------------------------------------------------------|
| Wan2.1-VACE-1.3B | <https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B> |
| Wan2.1-VACE-14B  | <https://huggingface.co/Wan-AI/Wan2.1-VACE-14B>  |
| Wan2.2-VACE-A14B | <https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B>  |

Currently, the MindSpeed MM repository only supports the VAE and TextEncoder in Diffusers format, so you need to additionally download the VAE and TextEncoder from [Wan2.1-VACE-1.3B-diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers).

### Weight Conversion

The VACE model requires weight conversion on the downloaded weights. Run the weight conversion script:

```shell
# Wan2.1-VACE
mm-convert VACEConverter hf_to_mm \
 --cfg.source_path <./weights/Wan-AI/Wan2.1-VACE-{model_type}/> \
 --cfg.target_path <./weights/Wan-AI-mm/Wan2.1-VACE-{model_type}/> \
 --cfg.target_parallel_config.pp_layers <pp_layers>
```

```shell
# Wan2.2-VACE
mm-convert VACEConverter hf_to_mm \
 --cfg.source_path <./weights/alibaba-pai/Wan2.2-VACE-Fun-A14B/{{high/low}_noise_model}/> \
 --cfg.target_path <./weights/alibaba-pai/Wan2.2-VACE-Fun-A14B/{{high/low}_noise_model}/> \
 --cfg.target_parallel_config.pp_layers <pp_layers>
```

The parameters of the weight conversion script are described as follows:

| Parameter | Meaning | Default Value |
| :---------------- | :----------------------- | :----------------------------------------------------------- |
| --cfg.source_path | Original weight path | / |
| --cfg.target_path | Path to save converted or partitioned weights | / |
| --pp_layers | Number of PP/VPP layers | When PP is enabled, using PP and VPP requires specifying the number of layers for each stage and converting. The value defaults to `[]`, meaning not enabled. |

To convert the VACE model back to Hugging Face format, run the weight conversion script:

```shell
# Wan2.1-VACE
mm-convert VACEConverter mm_to_hf \
 --cfg.source_path <path for your saved weight/> \
 --cfg.target_path <./converted_weights/Wan-AI/Wan2.1-VACE-{model_type}/post_train.pt/>
```

```shell
# Wan2.2-VACE
mm-convert VACEConverter mm_to_hf \
 --cfg.source_path <path for your saved weight/> \
 --cfg.target_path <./converted_weights/alibaba-pai/Wan2.2-VACE-Fun-A14B/{{high/low}_noise_model}/post_train.pt/>
```

The parameters of the weight conversion script are described as follows:

| Parameter | Meaning | Default Value |
| :------------ | :---- | :---- |
| --cfg.source_path | Weight path saved by MindSpeed MM | / |
| --cfg.target_path | Converted Hugging Face weight path | / |

## Pre-training

### Data Preprocessing

Process the data into the following format:

```bash
</dataset>
  ├──data.json
  ├──videos
  │  ├──video0001.mp4
  │  ├──video0002.mp4
  ├──src_videos
  │  ├──src_video0001.mp4
  │  ├──src_video0002.mp4
  ├──src_ref_images
  │  ├──src_ref_images0001_1.jpg
  │  ├──src_ref_images0001_2.jpg
  │  ├──src_ref_images0002_1.jpg
  ├──src_video_mask
  │  ├──src_video_mask0001.mp4
  │  ├──src_video_mask0002.mp4
```

`videos/` stores videos, `src_videos/` and `src_video_mask/` store the edited videos and masks, and `src_ref_images/` stores images. `data.json` contains all image-video-text pair information in the dataset, where `src_video`, `src_ref_images`, and `src_video_mask` can be null. A specific example is as follows:

```json
[
    {
        "video": "videos/video0001.mp4",
        "cap": "Video discrimination1.",
        "src_video": "src_videos/src_video0001.mp4",
        "src_ref_images": ["src_ref_images/src_ref_images0001_1.jpg","src_ref_images/src_ref_images0001_2.jpg"],
        "src_video_mask": "src_video_mask/src_video_mask0001.mp4"
    },
    {
        "video": "videos/video0002.mp4",
        "cap": "Video discrimination2.",
        "src_video": "src_videos/src_video0002.mp4",
        "src_ref_images": ["src_ref_images/src_ref_images0002_1.jpg"],
        "src_video_mask": "src_video_mask/src_video_mask0002.mp4"
    },
    ......
]
```

Modify the `examples/vace/feature_extract/data.txt` file, where each line represents a dataset. The first parameter indicates the path to the data folder, and the second parameter indicates the path to the `data.json` file, separated by `,`.

### Feature Extraction

#### Preparation

Before starting, please ensure that the environment setup, model weights, and dataset preprocessing are complete.

#### Parameter Configuration

Check whether the configurations for the model weight path, dataset path, and save path for extracted features are complete.

| Configuration File                                                |    Field to be Modified  | Modification Notes                                |
|-----------------------------------------------------| :---: |:------------------------------------|
| examples/vace/feature_extract/data.json             |      num_frames       | Maximum number of frames; if exceeded, selects `num_frames` frames.           |
| examples/vace/feature_extract/data.json             | max_hxw | Maximum resolution; if exceeded, the video will be cropped and compressed to this resolution.              |
| examples/vace/feature_extract/data.json             |    from_pretrained    | Modify it to the path corresponding to the downloaded tokenizer weights.            |
| examples/vace/feature_extract/feature_extraction.sh |     NPUS_PER_NODE     | Number of NPUs.                                  |
| examples/vace/feature_extract/model.json     |    from_pretrained    | Modify it to the path corresponding to the downloaded weights (including VAE and text_encoder). |
| mindspeed_mm/tools/tools.json                       |       save_path       | Save path for the extracted features.                          |

#### Start Feature Extraction

```bash
bash examples/vace/feature_extract/feature_extraction.sh
```

### Training

#### Preparation

Before starting, please confirm that the environment setup, model weight download, and feature extraction have been completed.

#### Parameter Configuration

Check if the model weight path, parallel parameter configuration, etc., are complete.

| Configuration File                                           |            Field to be Modified           | Modification Notes                                              |
|------------------------------------------------|:-------------------------:|:--------------------------------------------------|
| examples/vace/{model_type}/feature_data.json   |     basic_parameters      | Dataset path. Set `data_path` and `data_folder` to the file path and directory of the extracted features, respectively. |
| examples/vace/{model_type}/pretrain_fsdp.sh    |       NPUS_PER_NODE       | Number of NPUs per node                                           |
| examples/vace/{model_type}/pretrain_fsdp.sh    |          NNODES           | Number of nodes                                              |
| examples/vace/{model_type}/pretrain_fsdp.sh    |         SAVE_PATH         | Path to weights saved during training                                      |
| examples/vace/{model_type}/pretrain_model.json | predictor.from_pretrained | Pre-training/post-training weight path after weight conversion                                 |

[Parallel Parameter Configuration]

- FSDP2

  - Usage Scenario: When the model parameter scale is large, enabling FSDP2 can reduce static memory.

  - How to Enable: Add `--use-torch-fsdp2`, `--fsdp2-config-path ${fsdp2_config}`, `--untie-embeddings-and-output-weights`, and `--ckpt-format torch_dcp` to `GPT_ARGS` in `examples/vace/{model_type}/pretrain_fsdp.sh`. For the `fsdp2_config` configuration, please refer to [FSDP2 Feature Guide](../../docs/en/features/fsdp2.md).
  <a id="jump1"></a>
  - Training Weight Post-processing: When training with this feature is used, the saved weights require post-processing using the following conversion script before they can be used for further training or inference:

    ```bash
    # Weight path for saved weights after training
    save_path="./vace_weight_save"
    iter_dir="$save_path/iter_$(printf "%07d" $(cat $save_path/latest_checkpointed_iteration.txt))"
    # Target path for weight conversion
    convert_dir="./dcp_to_torch"
    mkdir -p $convert_dir/release/mp_rank_00
    cp $save_path/latest_checkpointed_iteration.txt $convert_dir/
    echo "release" > $convert_dir/latest_checkpointed_iteration.txt
    python -m torch.distributed.checkpoint.format_utils dcp_to_torch "$iter_dir" "$convert_dir/release/mp_rank_00/model_optim_rng.pt"
    ```

#### Start Training

```bash
bash examples/vace/{model_type}/pretrain_fsdp.sh
```

## Inference

### Parameter Configuration

| Configuration File                                                             |     Field to be Modified    |  Modification Notes |
|------------------------------------------------------------------|:-----------:|:-----|
| examples/vace/inference/inference_wan{2.1/2.2}_{model_type}.json |    model    |  Modify it to the path corresponding to the weights (including VAE, tokenizer, text encoder, and transformer).   |
| examples/vace/inference/inference_wan{2.1/2.2}_{model_type}.json | video_path  |  Video path required for the inference task   |
| examples/vace/inference/inference_wan{2.1/2.2}_{model_type}.json | image_path  |  Image path required for the inference task   |
| examples/vace/inference/inference_wan{2.1/2.2}_{model_type}.json | output_path |  Save path for the generated video  |

 1. The VAE and text encoder in `model_paths` need to use the [non-diffusers version](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) of the Hugging Face weights.
 2. If you want to use the weights saved during training for the transformer in `model_paths`, run the weight conversion script `mm-convert VACEConverter mm_to_hf` in advance to convert the MM format weights to Hugging Face format.

### Preparation

1. [Download DiffSynth-Studio]

    ```shell
    cd examples/vace
    git clone https://github.com/modelscope/DiffSynth-Studio.git
    cd DiffSynth-Studio
    git checkout 8332ece
    cp ../inference/Wan-VACE-Inference.py examples/wanvideo/model_inference
    ```

2. [NPU Adaptation]

    ```shell
    vim diffsynth/utils/__init__.py
      ```

    Change `torch.cuda.mem_get_info(self.device)[1] / (1024 ** 3)` at line 131
    to `torch.npu.mem_get_info()[1] / (1024 ** 3)`.

    ```shell
    vim diffsynth/vram_management/layers.py
      ```

    Change `torch.cuda.mem_get_info(self.computation_device)` at line 16
    to `torch.npu.mem_get_info()`.

    ```shell
    vim diffsynth/models/wan_video_dit.py
      ```

    Change `freqs` at line 96
    to `freqs.to(torch.complex64)`.

### Start Inference

```shell
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
python examples/wanvideo/model_inference/Wan-VACE-Inference.py ../inference/inference_wan{2.1/2.2}_{model_type}.json
```

## Environment Variable Declaration

| Environment Variable                          | Description                                                       | Value Description                                                                                                              |
|-------------------------------|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
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
