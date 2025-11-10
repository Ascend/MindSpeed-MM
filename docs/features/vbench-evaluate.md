# VBench Evaluate
VBench是当前视频生成领域相对全面的评测框架，支持t2v、i2v、long场景任务评测，支持从多种维度评测生成的视频质量，详细可查看：
[VBench介绍](https://github.com/Vchitect/VBench/blob/master/README.md)

## 环境安装

待评测模型按照README完成基础环境安装后，还需完成以下操作：

使用pip install 安装以下包：
```
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
detectron2安装：
```shell
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
```

vbench完成安装后，需要从源码复制文件到安装目录，其中"..."需要改为vbench实际安装的目录，具体操作如下：
```shell
git clone https://github.com/Vchitect/VBench.git

cd VBench
cp -r vbench2_beta_i2v/third_party .../envs/test/lib/python3.10/site-packages/vbench2_beta_i2v/third_party
cp -r vbench2_beta_long/configs  .../envs/test/lib/python3.10/site-packages/vbench2_beta_long/configs

```

## 数据集准备

当前支持vbench视频生成评测，支持t2v/long/i2v等场景。需要下载相关数据用于生成视频。

t2v/long场景需要准备prompts、json文件。

[t2v/long prompt 数据集](https://github.com/Vchitect/VBench/tree/master/prompts)

[t2v json下载路径](https://github.com/Vchitect/VBench/blob/master/vbench/VBench_full_info.json)下载到$VBench_full_info.json

[long json下载路径](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_long/VBench_full_info.json)

### t2v/long配置说明

    ```
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

配置说明：

```json5
{
  "eval_config": {
    "dataset": {
      "type": "vbench_eval", // 表示t2v/long场景
      "basic_param": {
        "data_path": "$VBench_full_info.json",
        "data_folder": "$vbench_prompts",    // $vbench_prompts路径
        "return_type": "list",
        "data_storage_mode": "standard"
      },
      "extra_param": {
        "augment": false,                 // 数据增强开关，开启后将使用强化后的prompt
        "prompt_file": "all_dimension.txt",   // 在dimension配置为空列表时，使用此文件作为视频文件前缀，默认为英文版本全维度
        "augmented_prompt_file": "augmented_prompts/gpt_enhanced_prompts/all_dimension_longer.txt"   // 如dimension配置为空列表，在augment=true时，使用此文件作为视频生成提示词，否则使用prompt file，默认为GPT强化的英文版本全维度
      }
    },
    "dataloader_param": {                 // dataloader参数，功能与训练配置相同
      "dataloader_mode": "sampler",
      "sampler_type": "SequentialSampler",
      "shuffle": true,
      "drop_last": false,                 // 关闭drop last保证所有prompt都生成视频
      "pin_memory": true,
      "group_frame": false,
      "group_resolution": false,
      "collate_param": {},
      "prefetch_factor": 4
    },
    "evaluation_model": "cogvideox-1.5",  // 被评测模型
    "evaluation_impl": "vbench_eval",     // 使用vbench评测
    "eval_type": "t2v",                   // t2v或者long，以5秒为界
    "load_ckpt_from_local": true, 
    "long_eval_config": "path_to_long_eval_configs",  // eval_type 为long时需要配置，配置为vbench安装完成后vbench2_beta_long的路径
    "dimensions": [                       // 评测维度配置
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
### i2v配置说明

[i2v json下载路径](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/vbench2_i2v_full_info.json)保存到$vbench2_i2v_full_info.json

[图片数据集下载链接](https://drive.google.com/drive/folders/1fdOZKQ7HWZtgutCKKA7CMzOhMFUGv4Zx?usp=sharing)，解压后将目录配置如下

    ```
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

配置说明

```json5
{
  "eval_config": {
    "dataset": {
      "type": "vbench_i2v",   // 数据集合类型，适用于i2v
      "basic_param": {
        "data_path": "$vbench2_i2v_full_info.json",   // i2v_full_info文件路径
        "data_folder": "$vbench_i2v",     // 配置根路径
        "return_type": "list",
        "data_storage_mode": "standard"
      },
      "extra_param": {
        "ratio": "16-9"                   // 图片比例，支持1-1、8-5、7-4、16-9四种比例
      }
    },
    "dataloader_param": {
      "dataloader_mode": "sampler",
      "sampler_type": "SequentialSampler",
      "shuffle": true,
      "drop_last": false,                 // 关闭drop last保证所有prompt都生成视频
      "pin_memory": true,
      "group_frame": false,
      "group_resolution": false,
      "collate_param": {},
      "prefetch_factor": 4
    },
    "evaluation_model": "cogvideox-1.5",  // 被评测模型
    "evaluation_impl": "vbench_eval",     // 使用vbench评测
    "eval_type": "i2v",                   // 评测场景i2v
    "load_ckpt_from_local": true,
    "dimensions": [                       // 评测维度配置
      "subject_consistency"
    ],
    "image_path": "$vbench_i2v_crop"      // 原始图片路径
  }
}
```

## 参数配置
1. 权重及模型文件配置

    请参考"推理-配置参数"章节，配置评测脚本（如`eval_cogvideox_i2v_1.5.sh`）、评测配置文件（如`eval_model_i2v_1.5.json`）中的模型和权重路径。


2. 评测维度配置
    当前vbench评测支持三种评测类型：i2v、t2v、long（>=5s视频评测），需要在`eval_model_i2v_1.5.json`/`eval_model_t2v_1.5.json`文件中配置`dimensions`的维度，支持配置多个。
    
    t2v支持维度：
    ```
   ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", "overall_consistency", "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]
    ```

    i2v支持维度：
    ```
   ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "temporal_flickering", "motion_smoothness", "dynamic_degree", "i2v_subject", "i2v_background", "camera_motion"]

    ```
    long支持维度：
    ```
   ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", "overall_consistency", "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]
    ```
    
3. 数据集参数配置
    
    见[数据集准备](#数据集准备)


## 启动评测

i2v启动评测：
```bash
bash examples/cogvideox/i2v_1.5/eval_cogvideox_i2v_1.5.sh
```
t2v启动评测：
```bash
bash examples/cogvideox/t2v_1.5/eval_cogvideox_t2v_1.5.sh
```
long启动评测，修改examples/cogvideox/t2v_1.5/eval_model_t2v_1.5.json中的eval_type为long，执行如下命令即可：
```bash
bash examples/cogvideox/t2v_1.5/eval_cogvideox_t2v_1.5.sh
```