# VACE 使用指南

- [VACE 使用指南](#VACE-使用指南)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
  - [任务支持列表](#任务支持列表)
  - [环境安装](#环境安装)
    - [仓库拉取](#仓库拉取)
    - [环境搭建](#环境搭建)
    - [Decord搭建](#decord搭建)
  - [权重下载及转换](#权重下载及转换)
    - [权重下载](#权重下载)
    - [权重转换](#权重转换)
  - [预训练](#预训练)
    - [数据预处理](#数据预处理)
    - [特征提取](#特征提取)
      - [准备工作](#准备工作)
      - [参数配置](#参数配置)
      - [启动特征提取](#启动特征提取)
    - [训练](#训练)
      - [准备工作](#准备工作-1)
      - [参数配置](#参数配置-1)
      - [启动训练](#启动训练)
  - [推理](#推理)
    - [参数配置](#参数配置-2)
    - [准备工作](#准备工作-2)
    - [启动推理](#启动推理)
  - [环境变量声明](#环境变量声明)

## 版本说明

#### 参考实现
【数据处理】
```
uel=https://github.com/ali-vilab/VACE
commit_id=0897c6d
```
【训练】
```
url=https://github.com/modelscope/DiffSynth-Studio.git
commit_id=8332ece
```

修改了DiffSynth-Studio中的部分超参数：

| 参数           | DiffSynth-Studio | MindSpeed-MM |
|--------------|:-----------------|:-------------|
| lr           | 1e-4             | 5e-5         |
| weight_decay | 0.01             | 0.1          |
| adam-beta2   | 0.999            | 0.95         |
| adam-eps     | 1e-8             | 1e-5         |

实验验证表明，DiffSynth-Studio的原始参数配置在GPU/NPU长跑时均会引发Loss尖刺。修改后的参数设置解决了此问题：其中`lr`和`weight_decay`采用VACE论文的设置，`adam-beta2`和`adam-eps`根据实验效果进行经验性调整。该配置方案可供参考。

## 任务支持列表

| 模型大小             | 预训练 |推理|
|------------------|:----|:----|
| Wan2.1-VACE-1.3B | ✔ | ✔ |
| Wan2.1-VACE-14B  | ✔ | ✔ |
| Wan2.2-VACE-A14B | ✔ | ✔ |

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

### 仓库拉取

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ../MindSpeed-MM
```

### 环境搭建

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl
# apex for Ascend 参考 https://gitcode.com/Ascend/apex
# 建议从原仓编译安装 

# 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
source /usr/local/Ascend/cann/set_env.sh 

# 安装加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout 93c45456c7044bacddebc5072316c01006c938f9
pip install -r requirements.txt 
pip install -e .
cd ..

# 安装其余依赖库
pip install -e .

# 源码安装Diffusers
pip install diffusers==0.33.1
```

### Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

---

## 权重下载及转换

### 权重下载

| 模型               | Hugging Face下载链接                                            |
|------------------|------------------------------------------------------------|
| Wan2.1-VACE-1.3B | <https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B> |
| Wan2.1-VACE-14B  | <https://huggingface.co/Wan-AI/Wan2.1-VACE-14B>  |
| Wan2.2-VACE-A14B | <https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B>  |

目前MM仓只支持Diffusers格式的VAE和TextEncoder，因此需要额外下载[Wan2.1-VACE-1.3B-diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers)的VAE和TextEncoder来使用
### 权重转换

VACE模型需要对下载后的权重进行权重转换，运行权重转换脚本：
```shell
# Wan2.1-VACE
mm-convert VACEConverter hf_to_mm \
 --cfg.source_path <./weights/Wan-AI/Wan2.1-VACE-{model_type}/> \
 --cfg.target_path <./weights/Wam-AI/Wan2.1-VACE-{model_type}/> \
 --cfg.target_parallel_config.pp_layers <pp_layers>
```
```shell
# Wan2.2-VACE
mm-convert VACEConverter hf_to_mm \
 --cfg.source_path <./weights/alibaba-pai/Wan2.2-VACE-Fun-A14B/{{high/low}_noise_model}/> \
 --cfg.target_path <./weights/alibaba-pai/Wan2.2-VACE-Fun-A14B/{{high/low}_noise_model}/> \
 --cfg.target_parallel_config.pp_layers <pp_layers>
```
权重转换脚本的参数说明如下：

| 参数              | 含义                     | 默认值                                                       |
| :---------------- | :----------------------- | :----------------------------------------------------------- |
| --cfg.source_path | 原始权重路径             | /                                                            |
| --cfg.target_path | 转换或切分后权重保存路径 | /                                                            |
| --pp_layers   | PP/VPP层数               | 开启PP时, 使用PP和VPP需要指定各stage的层数并转换, 默认为`[]`，即不使用 |


VACE模型如需转回Hugging Face格式，需运行权重转换脚本：
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
权重转换脚本的参数说明如下：

|参数| 含义 | 默认值 |
|:------------|:----|:----|
| --cfg.source_path | MindSpeed MM保存的权重路径                                   | /      |
| --cfg.target_path | 转换后的Hugging Face权重路径                                 | /      |
---

## 预训练

### 数据预处理

将数据处理成如下格式

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

其中，`videos/`下存放视频，`src_videos/`和`src_video_mask/`下存放被编辑的视频和掩码，`src_ref_images/`下存放图片。data.json中包含该数据集中所有的图像-视频-文本对信息，其中`src_video，src_ref_images，src_video_mask`可以为null，具体示例如下：

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

修改`examples/vace/feature_extract/data.txt`文件，其中每一行表示一个数据集，第一个参数表示数据文件夹的路径，第二个参数表示`data.json`文件的路径，用`,`分隔

### 特征提取

#### 准备工作

在开始之前，请确认环境准备、模型权重和数据集预处理已经完成

#### 参数配置

检查模型权重路径、数据集路径、提取后的特征保存路径等配置是否完成

| 配置文件                                                |   修改字段  | 修改说明                                |
|-----------------------------------------------------| :---: |:------------------------------------|
| examples/vace/feature_extract/data.json             |      num_frames       | 最大的帧数，超过则选取其中的num_frames帧           |
| examples/vace/feature_extract/data.json             | max_hxw | 最大的分辨率，超过会将视频裁剪压缩到该分辨率              |
| examples/vace/feature_extract/data.json             |    from_pretrained    | 修改为下载的tokenizer的权重所对应的路径            |
| examples/vace/feature_extract/feature_extraction.sh |     NPUS_PER_NODE     | 卡数                                  |
| examples/vace/feature_extract/model.json     |    from_pretrained    | 修改为下载的权重所对应路径（包括vae,  text_encoder） |
| mindspeed_mm/tools/tools.json                       |       save_path       | 提取后的特征保存路径                          |

#### 启动特征提取

```bash
bash examples/vace/feature_extract/feature_extraction.sh
```

### 训练

#### 准备工作

在开始之前，请确认环境准备、模型权重下载、特征提取已完成。

#### 参数配置

检查模型权重路径、并行参数配置等是否完成

| 配置文件                                           |           修改字段            | 修改说明                                              |
|------------------------------------------------|:-------------------------:|:--------------------------------------------------|
| examples/vace/{model_type}/feature_data.json   |     basic_parameters      | 数据集路径，`data_path`和`data_folder`分别配置提取后的特征的文件路径和目录 |
| examples/vace/{model_type}/pretrain_fsdp.sh    |       NPUS_PER_NODE       | 每个节点的卡数                                           |
| examples/vace/{model_type}/pretrain_fsdp.sh    |          NNODES           | 节点数量                                              |
| examples/vace/{model_type}/pretrain_fsdp.sh    |         SAVE_PATH         | 训练过程中保存的权重路径                                      |
| examples/vace/{model_type}/pretrain_model.json | predictor.from_pretrained | 权重转换后的预训练/训练后权重路径                                 |

【并行化配置参数说明】：

- fsdp2

  - 使用场景：在模型参数规模较大时，可以通过开启fsdp2降低静态内存。
  
  - 使能方式：`examples/vace/{model_type}/pretrain_fsdp.sh`的`GPT_ARGS`中加入`--use-torch-fsdp2`，`--fsdp2-config-path ${fsdp2_config}`，`--untie-embeddings-and-output-weights`以及`--ckpt-format torch_dcp`，其中fsdp2_config配置请参考：[FSDP2说明](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/fsdp2.md)
  <a id="jump1"></a>
  - 训练权重后处理：使用该特性训练时，保存的权重需要使用下面的转换脚本进行后处理才能再次训练或者用于推理：

    ```bash
    # 训练结束后保存的权重路径
    save_path="./vace_weight_save"
    iter_dir="$save_path/iter_$(printf "%07d" $(cat $save_path/latest_checkpointed_iteration.txt))"
    # 权重转换的目标路径
    convert_dir="./dcp_to_torch"
    mkdir -p $convert_dir/release/mp_rank_00
    cp $save_path/latest_checkpointed_iteration.txt $convert_dir/
    echo "release" > $convert_dir/latest_checkpointed_iteration.txt
    python -m torch.distributed.checkpoint.format_utils dcp_to_torch "$iter_dir" "$convert_dir/release/mp_rank_00/model_optim_rng.pt"
    ```
#### 启动训练

```bash
bash examples/vace/{model_type}/pretrain_fsdp.sh
```

## 推理

### 参数配置

| 配置文件                                                             |    修改字段     |  修改说明 |
|------------------------------------------------------------------|:-----------:|:-----|
| examples/vace/inference/inference_wan{2.1/2.2}_{model_type}.json |    model    |  修改为权重所对应路径（包括vae、tokenizer、text_encoder、transformer）   |
| examples/vace/inference/inference_wan{2.1/2.2}_{model_type}.json | video_path  |  推理任务需要使用的视频路径   |
| examples/vace/inference/inference_wan{2.1/2.2}_{model_type}.json | image_path  |  推理任务需要使用的图像路径   |
| examples/vace/inference/inference_wan{2.1/2.2}_{model_type}.json | output_path |  生成视频的保存路径  |

 1. model_paths中的vae和text_encoder需要使用[非diffusers版](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B)的huggingface权重
 2. model_paths中的transformer如果想使用训练中保存的权重，需要提前运行权重转换脚本`mm-convert VACEConverter mm_to_hf`将MM格式权重转换为Hugging Face格式



### 准备工作
1. 【下载DiffSynth-Studio】
```shell
cd examples/vace
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
git checkout 8332ece
cp ../inference/Wan-VACE-Inference.py examples/wanvideo/model_inference
```
2. 【npu适配】
 ```shell
vim diffsynth/utils/__init__.py
  ```
将131行的`torch.cuda.mem_get_info(self.device)[1] / (1024 ** 3)`
改为`torch.npu.mem_get_info()[1] / (1024 ** 3)`
```shell
vim diffsynth/vram_management/layers.py
  ```
将16行的`torch.cuda.mem_get_info(self.computation_device)`
改为`torch.npu.mem_get_info()`

```shell
vim diffsynth/models/wan_video_dit.py
  ```
将96行的`freqs`
改为`freqs.to(torch.complex64)`

### 启动推理

```shell
source /usr/local/Ascend/cann/set_env.sh
python examples/wanvideo/model_inference/Wan-VACE-Inference.py ../inference/inference_wan{2.1/2.2}_{model_type}.json
```

## 环境变量声明

| 环境变量                          | 描述                                                       | 取值说明                                                                                                              |
|-------------------------------|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 是否开启日志打印                                                 | `0`: 关闭日志打屏<br>`1`: 开启日志打屏                                                                                        |
| `ASCEND_GLOBAL_LOG_LEVEL`     | 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志                             | `0`: 对应DEBUG级别<br>`1`: 对应INFO级别<br>`2`: 对应WARNING级别<br>`3`: 对应ERROR级别<br>`4`: 对应NULL级别，不输出日志                      |
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                              | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                                                   |
| `COMBINED_ENABLE`             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景       | `0`: 关闭<br>`1`: 开启                                                                                                |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                                                          |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                              | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                                          |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征                                                                 |
| `HCCL_EXEC_TIMEOUT`           | 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步                 | 需要配置为整数，取值范围`[68,17340]`，默认值为`1800`，单位`s`                                                                         |
| `ACLNN_CACHE_LIMIT`           | 配置单算子执行API在Host侧缓存的算子信息条目个数                              | 需要配置为整数，取值范围`[1, 10,000,000]`，默认值为`10000`                                                                         |
| `TOKENIZERS_PARALLELISM`      | 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为 | `False`: 禁用并行分词<br>`True`: 开启并行分词                                                                                 |
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启                                             | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用                                                                                    |
| `NPU_ASD_ENABLE`              | 控制是否开启Ascend Extension for PyTorch的特征值检测功能               | 设置`0`或未设置: 关闭特征值检测<br>`1`: 表示开启特征值检测，只打印异常日志，不告警<br>`2`:开启特征值检测，并告警<br>`3`:开启特征值检测，并告警，同时会在device侧info级别日志中记录过程数据 |
| `ASCEND_LAUNCH_BLOCKING`      | 控制算子执行时是否启动同步模式                                          | `0`: 采用异步方式执行<br>`1`: 强制算子采用同步模式运行                                                                                |
| `NPUS_PER_NODE`               | 配置一个计算节点上使用的NPU数量                                        | 整数值（如 `1`, `8` 等）                                                                                                 |