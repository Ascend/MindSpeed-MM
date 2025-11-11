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
    - [Diffusers权重下载](#diffusers权重下载)
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
    - [准备工作](#准备工作-3)
    - [参数配置](#参数配置-3)
    - [启动推理](#启动推理)
  - [环境变量声明](#环境变量声明)

## 版本说明

#### 参考实现

```
url=https://github.com/modelscope/DiffSynth-Studio.git
commit_id=f0ea049
```

## 任务支持列表

| 模型大小 | 预训练 |
|------|:----|
| 1.3B | ✔ |
| 14B | ✔ |

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
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

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

### Diffusers权重下载

| 模型             | Huggingface下载链接                                            |
|----------------|------------------------------------------------------------|
| VACE-1.3B      | <https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers> |
| VACE-14B       | <https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers>  |

### 权重转换

需要对下载后的VACE模型权重`transformer`部分进行权重转换，运行权重转换脚本：

```shell
mm-convert VACEConverter hf_to_mm \
 --cfg.source_path <./weights/Wan-AI/Wan2.1-VACE-{1.3/14}B-Diffusers/transformer/> \
 --cfg.target_path <./weights/Wan-AI/Wan2.1-VACE-{1.3/14}B-Diffusers/transformer/>
 --cfg.target_parallel_config.pp_layers <pp_layers>
```

权重转换脚本的参数说明如下：

| 参数              | 含义                     | 默认值                                                       |
| :---------------- | :----------------------- | :----------------------------------------------------------- |
| --cfg.source_path | 原始权重路径             | /                                                            |
| --cfg.target_path | 转换或切分后权重保存路径 | /                                                            |
| --pp_layers   | PP/VPP层数               | 开启PP时, 使用PP和VPP需要指定各stage的层数并转换, 默认为`[]`，即不使用 |

如需转回Hugging Face格式，需运行权重转换脚本：
f
```shell
mm-convert VACEConverter mm_to_hf \
 --cfg.source_path <path for your saved weight/> \
 --cfg.target_path <./converted_weights/Wan-AI/Wan2.1-VACE-{1.3/14}B-Diffusers/transformer/>
 --cfg.hf_path <weights/Wan-AI/Wan2.1-VACE-{1.3/14}B-Diffusers/transformer/>
```

权重转换脚本的参数说明如下：

|参数| 含义 | 默认值 |
|:------------|:----|:----|
| --cfg.source_path | MindSpeed MM保存的权重路径                                   | /      |
| --cfg.target_path | 转换后的Hugging Face权重路径                                 | /      |
| --cfg.hf_path     | 原始Hugging Face权重路径，需要从该目录下获取原始huggingface配置文件 |    /   |

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

| 配置文件                                         |      修改字段       | 修改说明      |
|----------------------------------------------| :---: | :--- |
| examples/vace/{model_size}/feature_data.json |  basic_parameters   | 数据集路径，`data_path`和`data_folder`分别配置提取后的特征的文件路径和目录 |
| examples/vace/{model_size}/pretrain_fsdp.sh  |    NPUS_PER_NODE    | 每个节点的卡数                                      |
| examples/vace/{model_size}/pretrain_fsdp.sh  |       NNODES        | 节点数量                                            |
| examples/vace/{model_size}/pretrain_fsdp.sh  |      LOAD_PATH      | 权重转换后的预训练权重路径                          |
| examples/vace/{model_size}/pretrain_fsdp.sh  |      SAVE_PATH      | 训练过程中保存的权重路径                            |

【并行化配置参数说明】：

- fsdp2

  - 使用场景：在模型参数规模较大时，可以通过开启fsdp2降低静态内存。
  
  - 使能方式：`examples/vace/{model_size}/pretrain_fsdp.sh`的`GPT_ARGS`中加入`--use-torch-fsdp2`，`--fsdp2-config-path ${fsdp2_config}`，`--untie-embeddings-and-output-weights`以及`--ckpt-format torch_dcp`，其中fsdp2_config配置请参考：[FSDP2说明](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/fsdp2.md)
  <a id="jump1"></a>
  - 训练权重后处理：使用该特性训练时，保存的权重需要使用下面的转换脚本进行后处理才能用于推理：

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
bash examples/vace/{model_size}/pretrain_fsdp.sh
```
