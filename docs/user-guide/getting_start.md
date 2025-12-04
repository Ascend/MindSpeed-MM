# 快速上手
MindSpeed MM同时支持多模态生成和多模态理解模型，因此下面分别介绍wan2.1和Qwen2.5VL两个典型模型使用方法，引导开发者快速上手预置模型在昇腾NPU上的高效运行。

## Qwen2.5-VL-3B 快速上手指南
更多细节请[参考](https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/qwen2.5vl)
### 1. 环境安装
#### 1.1 昇腾软件安装
昇腾环境安装请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/tree/master/docs/user-guide/installation.md)
(基于Python3.10)

#### 1.2 仓库拉取以及MindSpeed MM依赖安装
仓库拉取：
```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
mkdir logs
mkdir data
mkdir ckpt
```

激活环境：
```bash
# 激活上面构建的python3.10版本的环境
conda activate test
```

安装其它依赖：
```bash
# 安装加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_v0.12.1
git checkout 93c45456c7044bacddebc5072316c01006c938f9
# 安装mindspeed及依赖
pip install -e .
cd ..
# 安装mindspeed mm及依赖
pip install -e .
```

### 2. 权重下载及转换
#### 2.1 权重下载

从Huggingface库下载对应的模型权重:

模型地址: [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main)；

 将下载的模型权重保存到本地的`ckpt/hf_path/Qwen2.5-VL-3B-Instruct`目录下。

#### 2.2 权重转换
MindSpeed MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的互相转换以及PP（Pipeline Parallel）权重的重切分，详情参考[权重转换工具](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)。

以下是hf2mm的转换示例：
```bash
# 3b
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[36]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 1

# 其中：
# mm_dir: 转换后保存目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
```

MindSpeed-MM修改了部分原始网络的结构名称，在微调后，如果需要将权重转回huggingface格式，可使用`mm-convert`权重转换工具对微调后的权重进行转换，将权重名称修改为与原始网络一致。

以下是mm2hf的转换示例：
```bash
mm-convert  Qwen2_5_VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-3B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [36] \
  --cfg.parallel_config.vit_pp_layers [32] \
  --cfg.parallel_config.tp_size 1
# 其中：
# save_hf_dir: mm微调后转换回hf模型格式的目录
# mm_dir: 微调后保存的权重目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
```

如果需要用转换后模型训练的话，同步修改`examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重 `ckpt/hf_path/Qwen2.5-VL-3B-Instruct`进行区分。

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-3B-Instruct"
```

### 3. 数据集准备及处理
#### 3.1 数据集下载（以coco2017数据集为例）
(1)用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中。

(2)获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下。

#### 3.2 数据集处理
运行数据转换脚本python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py，转换后参考数据目录结构如下：

   ```
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

---

### 4. 启动微调
#### 4.1 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

#### 4.2 配置参数

【数据目录配置】

根据实际情况修改`examples/qwen2.5vl/data_3b.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

以Qwen2.5VL-3B为例，`data_3b.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径。

**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

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

【模型保存加载及日志信息配置】

根据实际情况配置`examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 加载路径
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-3B-Instruct"
# 保存路径
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # 不加载优化器状态，若需加载请移除
    --no-load-rng \  # 不加载随机数状态，若需加载请移除
    --no-save-optim \  # 不保存优化器状态，若需保存请移除
    --no-save-rng \  # 不保存随机数状态，若需保存请移除
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # 日志间隔
    --save-interval 5000 \  # 保存间隔
    ...
    --log-tps \  # 增加此参数可使能在训练中打印每步语言模块的平均序列长度，并在训练结束后计算每秒吞吐tokens量。
"
```


【单机运行配置】

配置`examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh`参数如下所示。

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

【多机运行配置】

配置`examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPUS_PER_NODE=8
MASTER_ADDR=xx.xx.xx.xx  # 主节点IP
MASTER_PORT=29501
NNODES=N  # N为实际节点数
NODE_RANK=i  # i为节点顺序编号，取值[0:N-1]
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

#### 4.3 启动微调

以Qwen2.5VL-3B为例，启动微调训练任务。

```shell
bash examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh
```

## Wan2.1（T2V 1.3B） 快速上手指南
更多细节请[参考](https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/wan2.1)
### 1. 环境安装
#### 1.1 昇腾软件安装
昇腾环境安装请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/tree/master/docs/user-guide/installation.md)
(基于python3.10)

#### 1.2 仓库拉取以及MindSpeed MM依赖安装
仓库拉取：
```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ../MindSpeed-MM
```

激活环境：
```bash
# 激活上面构建的python3.10版本的环境
conda activate test
```

安装其它依赖：
```bash
# 安装加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_v0.12.1
git checkout 93c45456c7044bacddebc5072316c01006c938f9
pip install -r requirements.txt 
pip install -e .
cd ..

# 安装其余依赖库
pip install -e .

# 源码安装Diffusers
pip install diffusers==0.33.1

```

#### 1.3 Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

### 2. 权重下载及转换

#### 2.1 Diffusers权重下载

|   模型   |   Huggingface下载链接   |
| ---- | ---- |
|   T2V-1.3B   |   <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>   |


#### 2.2 权重转换

需要对下载后的Wan2.1模型权重`transformer`部分进行权重转换，运行权重转换脚本：

* 启动脚本中修改LOAD_PATH为权重转换后的路径 (./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/)，SAVE_PATH为保存路径
```shell
python examples/wan2.1/convert_ckpt.py --source_path <./weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer/> --target_path <./weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer/> --mode convert_to_mm
```

权重转换脚本的参数说明如下：

| 参数              | 含义                      | 默认值                                                                |
|:----------------|:------------------------|:-------------------------------------------------------------------|
| --source_path   | 原始下载权重transformer文件夹的路径 | ./weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer/ |
| --target_path   | 转换后的权重保存路径              | ./weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer/ |
| --mode          | 转换模式                    | 需选择convert_to_mm                                                   |
| --pp_vpp_layers | PP/VPP层数                | 在convert_to_mm时, 使用PP和VPP需要指定各stage的层数并转换, 默认不使用                   |

如需转回Hugging Face格式，需运行权重转换脚本：

**注**： 如进行layer zero训练，则需首先进行其[训练权重后处理](#jump1)，再进行如下操作：

```shell
python examples/wan2.1/convert_ckpt.py --source_path <path for your saved weight/> --ckpt_path <./weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer/> --target_path <path for your saved weight/> --mode convert_to_hf
```

权重转换脚本的参数说明如下：

|参数| 含义 | 默认值 |
|:------------|:----|:----|
| --source_path | 训练权重/layer zero训练后处理权重 | path for your saved weight |
| --ckpt_path | 原始下载权重transformer文件夹的路径 | ./weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer/ |
| --target_path | 转换后的权重保存路径 | ./weights/Wan-AI/Wan2.1-{T2V/I2V}-{1.3/14}B-Diffusers/transformer/ |
| --mode | 转换模式 | 需选择convert_to_hf |

### 3. 数据预处理

将数据处理成如下格式

```bash
</dataset>
  ├──data.json
  ├──videos
  │  ├──video0001.mp4
  │  ├──video0002.mp4
```

其中，`videos/`下存放视频，data.json中包含该数据集中所有的视频-文本对信息，具体示例如下：

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

修改`examples/wan2.1/feature_extract/data.txt`文件，其中每一行表示一个数据集，第一个参数表示数据文件夹的路径，第二个参数表示`data.json`文件的路径，用`,`分隔

### 4.1 特征提取

#### 4.2 准备工作

在开始之前，请确认环境准备、模型权重和数据集预处理已经完成

#### 4.3 参数配置

检查模型权重路径、数据集路径、提取后的特征保存路径等配置是否完成

| 配置文件   |   修改字段  | 修改说明  |
| --- | :---: | :--- |
| examples/wan2.1/feature_extract/data.json              |      num_frames       | 最大的帧数，超过则随机选取其中的num_frames帧        |
| examples/wan2.1/feature_extract/data.json              | max_height, max_width | 最大的长宽，超过则centercrop到最大分辨率            |
| examples/wan2.1/feature_extract/data.json |    from_pretrained    | 修改为下载的tokenizer权重所对应路径 |
| examples/wan2.1/feature_extract/feature_extraction.sh  |     NPUS_PER_NODE     | 卡数                                                |
| examples/wan2.1/feature_extract/model.json |    from_pretrained    | 修改为下载的权重所对应路径（包括vae, text_encoder） |
| mindspeed_mm/tools/tools.json                 |    save_path          | 提取后的特征保存路径                               |

#### 4.4 启动特征提取
* 修改`examples/wan2.1/feature_extract/data.txt`文件，其中每一行表示一个数据集，第一个参数表示数据文件夹的路径，第二个参数表示`data.json`文件的路径，用`,`分隔。

* 修改`examples/wan2.1/feature_extract/model.json`修改为下载的权重所对应路径（包括vae, tokenizer, text_encoder）

```bash
bash examples/wan2.1/feature_extract/feature_extraction.sh
```

### 5. 启动训练

#### 5.1 准备工作

在开始之前，请确认环境准备、模型权重下载、特征提取已完成。

#### 5.2 参数配置

检查模型权重路径、并行参数配置等是否完成。

| 配置文件   |      修改字段       | 修改说明      |
| --- | :---: | :--- |
| examples/wan2.1/{model_size}/{task}/data.txt    | 文件内容  | 提取后的特征保存路径 |
| examples/wan2.1/{model_size}/{task}/feature_data.json   |   from_pretrained   | 修改为下载的权重所对应路径 |
| examples/wan2.1/feature_extract/tools.json | task | 修改为当前的任务类型t2v(可选的任务类型为t2v和i2v) |
| examples/wan2.1/{model_size}/{task}/pretrain.sh |    NPUS_PER_NODE    | 每个节点的卡数                                      |
| examples/wan2.1/{model_size}/{task}/pretrain.sh |       NNODES        | 节点数量                                            |
| examples/wan2.1/{model_size}/{task}/pretrain.sh |      LOAD_PATH      | 权重转换后的预训练权重路径                          |
| examples/wan2.1/{model_size}/{task}/pretrain.sh |      SAVE_PATH      | 训练过程中保存的权重路径                            |
| examples/wan2.1/{model_size}/{task}/pretrain.sh |         CP          | 训练时的CP size（建议根据训练时设定的分辨率调整）   |


#### 5.3 启动训练
feature_data.json中修改tokenizer权重路径

```bash
bash examples/wan2.1/1.3b/t2v/pretrain.sh
```
