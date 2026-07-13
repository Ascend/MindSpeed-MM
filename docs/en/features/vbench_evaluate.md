# VBench Evaluate

VBench is a relatively comprehensive evaluation framework in the current video generation field. It supports task evaluation for text-to-video (t2v), image-to-video (i2v), and long video scenarios, and can evaluate generated video quality from multiple dimensions. For details, please refer to [VBench Introduction](https://github.com/Vchitect/VBench/blob/master/README.md).

## Environment Installation

After completing the basic environment setup for the model to be evaluated according to the README, the following additional operations are required:

Use `pip install` to install the following packages:

```shell
vbench==0.1.5
transformers==4.45.0
scenedetect==0.6.5.2
av==13.1.0
ffmpeg
moviepy==1.0.3
dreamsim==0.2.1
cloudpickle==3.1.1
imageio_ffmpeg==0.5.1
portalocker==2.8.2
timm==1.0.8
```

Install detectron2:

```shell
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
```

After installing VBench, you need to copy files from the source code to the installation directory, where "..." with the actual installation directory of VBench. The specific operations are as follows:

```shell
git clone https://github.com/Vchitect/VBench.git

cd VBench
cp -r vbench2_beta_i2v/third_party .../envs/test/lib/python3.10/site-packages/vbench2_beta_i2v/third_party
cp -r vbench2_beta_long/configs  .../envs/test/lib/python3.10/site-packages/vbench2_beta_long/configs

```

## Dataset Preparation

Currently, video generation evaluation using VBench is supported, covering t2v, i2v, long-video and other scenarios. Relevant data needs to be downloaded for video generation.

For t2v/long-video scenarios, prompts and json files need to be prepared.

[t2v/long-video prompt dataset](https://github.com/Vchitect/VBench/tree/master/prompts)

[t2v json download path](https://github.com/Vchitect/VBench/blob/master/vbench/VBench_full_info.json): `$VBench_full_info.json`

[long-video json download path](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_long/VBench_full_info.json)

### t2v/long-video Configuration

    ```shell
    $vbench_prompts
    ├── augmented_prompts
    │   ├── gpt_enhanced_prompts
    │   │   ├── prompts_per_category_longer
    │   │   ├── prompts_per_dimension_longer
    │   │   │   ├── appearance_style_longer.txt
    │   │   │   └── ...
    │   │   ├── all_category_longer.txt
    │   │   └── ...
    │   └── hunyuan_all_dimension.txt
    ├── prompts_per_category
    │   └── ...
    ├── metadata
    │   └── ...
    ├── prompts_per_dimension_chinese
    │   └── ...
    ├── prompts_per_dimension
    │   ├── appearance_style.txt
    │   └── ...
    ├── all_dimension.txt
    ├── all_dimension_cn.txt
    └── ...
    ```

Configuration Instructions:

```json5
{
  "eval_config": {
    "dataset": {
      "type": "vbench_eval", // Indicates t2v/long-video scenario
      "basic_param": {
        "data_path": "$VBench_full_info.json",
        "data_folder": "$vbench_prompts",    // $vbench_prompts path
        "return_type": "list",
        "data_storage_mode": "standard"
      },
      "extra_param": {
        "augment": false,                 // Data augmentation switch. The enhanced prompt will be used when enabled
        "prompt_file": "all_dimension.txt",   // When the dimensions list is empty, this file is used as the video file prefix. Defaults to the English version of all dimensions
        "augmented_prompt_file": "augmented_prompts/gpt_enhanced_prompts/all_dimension_longer.txt"   // If dimensions list is empty and augment=true, this file is used for video generation prompts; otherwise, the prompt_file is used. Defaults to the GPT-enhanced English version of all dimensions
      }
    },
    "dataloader_param": {                 // Dataloader parameter, with the same functionality as the training configuration
      "dataloader_mode": "sampler",
      "sampler_type": "SequentialSampler",
      "shuffle": true,
      "drop_last": false,                 // Disable drop_last to ensure all prompts generate videos
      "pin_memory": true,
      "group_frame": false,
      "group_resolution": false,
      "collate_param": {},
      "prefetch_factor": 4
    },
    "evaluation_model": "cogvideox-1.5",  // Model being evaluated
    "evaluation_impl": "vbench_eval",     // Use VBench for evaluation
    "eval_type": "t2v",                   // t2v or long-video, with a 5-second boundary
    "load_ckpt_from_local": true,
    "long_eval_config": "path_to_long_eval_configs",  // Required when eval_type is long. Configure to the path of vbench2_beta_long after VBench installation is complete
    "dimensions": [                       // Evaluation dimension configuration
      "subject_consistency",
      "background_consistency",
      "aesthetic_quality",
      "imaging_quality",
      "temporal_style",
      "overall_consistency",
      "human_action",
      "temporal_flickering",
      "motion_smoothness",
      "dynamic_degree",
      "appearance_style"
    ]
  }
}
```

### i2v Configuration

Download the [i2v json](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/vbench2_i2v_full_info.json) and save it to `$vbench2_i2v_full_info.json`.

Download the [image dataset](https://drive.google.com/drive/folders/1fdOZKQ7HWZtgutCKKA7CMzOhMFUGv4Zx?usp=sharing), extract it, and configure the directory as follows:

    ```shell
    $vbench_i2v
    ├── data
    │   ├── [crop]($vbench_i2v_crop)
    │   │   ├── 1-1
    │   │   ├── 7-4
    │   │   │   ├── a bald eagle flying over a tree filled forest.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   └── origin
    │       └── ...
    └── [vbench2_i2v_full_info.json]($vbench2_i2v_full_info.json)
    ```

Configuration Instructions:

```json5
{
  "eval_config": {
    "dataset": {
      "type": "vbench_i2v",   // Dataset type, applicable to i2v
      "basic_param": {
        "data_path": "$vbench2_i2v_full_info.json",   // i2v_full_info file path
        "data_folder": "$vbench_i2v",     // Configure the root path
        "return_type": "list",
        "data_storage_mode": "standard"
      },
      "extra_param": {
        "ratio": "16-9"                   // Image aspect ratio. Supports 1-1, 8-5, 7-4, 16-9
      }
    },
    "dataloader_param": {
      "dataloader_mode": "sampler",
      "sampler_type": "SequentialSampler",
      "shuffle": true,
      "drop_last": false,                 // Disable drop_last to ensure all prompts generate videos
      "pin_memory": true,
      "group_frame": false,
      "group_resolution": false,
      "collate_param": {},
      "prefetch_factor": 4
    },
    "evaluation_model": "cogvideox-1.5",  // Model being evaluated
    "evaluation_impl": "vbench_eval",     // Using VBench for evaluation
    "eval_type": "i2v",                   // Evaluation scenario: i2v
    "load_ckpt_from_local": true,
    "dimensions": [                       // Evaluation dimension configuration
      "subject_consistency"
    ],
    "image_path": "$vbench_i2v_crop"      //  Path to original images
  }
}
```

## Parameter Configuration

1. Weight and Model File Configuration

    Please refer to the "Inference - Configuration Parameters" section to configure the model and weight paths in the evaluation script (e.g., `eval_cogvideox_i2v_1.5.sh`) and evaluation configuration file (e.g., `eval_model_i2v_1.5.json`).

2. Evaluation Dimension Configuration
    Currently, VBench evaluation supports three evaluation types: i2v, t2v, and long-video (for videos >= 5s). You need to configure the `dimensions` in the `eval_model_i2v_1.5.json`/`eval_model_t2v_1.5.json` files, and multiple dimensions are supported.

    Supported dimensions for t2v:

    ```shell
   ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", "overall_consistency", "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]
    ```

    Supported dimensions for i2v:

    ```shell
   ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "temporal_flickering", "motion_smoothness", "dynamic_degree", "i2v_subject", "i2v_background", "camera_motion"]

    ```

    Supported dimensions for long-video:

    ```shell
   ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", "overall_consistency", "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]
    ```

3. Dataset Parameter Configuration

    See [Dataset Preparation](#dataset-preparation).

## Launching Evaluation

Start i2v evaluation:

```bash
bash examples/cogvideox/i2v_1.5/eval_cogvideox_i2v_1.5.sh
```

Start t2v evaluation:

```bash
bash examples/cogvideox/t2v_1.5/eval_cogvideox_t2v_1.5.sh
```

For long-video evaluation launching, modify `eval_type` in `examples/cogvideox/t2v_1.5/eval_model_t2v_1.5.json` to `long` and execute the following command:

```bash
bash examples/cogvideox/t2v_1.5/eval_cogvideox_t2v_1.5.sh
```
