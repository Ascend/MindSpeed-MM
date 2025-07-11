# InternVL2.5 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
- [预训练](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动预训练](#jump4.3)
- [推理](#jump5)
  - [准备工作](#jump5.1)
  - [配置参数](#jump5.2)
  - [启动推理](#jump5.3)
- [环境变量声明](#jump6)
- [注意事项](#jump7)

## 版本说明
#### 参考实现
```
url=https://github.com/OpenGVLab/InternVL.git
commit_id=2d57e21
```

#### 变更记录

2025.02.20: 首次发布InternVL2.5模型

---
<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

<a id="jump1.1"></a>

#### 1. 仓库拉取

```shell
git clone --branch 2.1.0 https://gitee.com/ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
mkdir logs
mkdir dataset
mkdir ckpt
```

<a id="jump1.2"></a>

#### 2. 环境搭建

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
pip install torch-2.1.0-cp310-cp310m-manylinux2014_aarch64.whl
pip install torch_npu-2.1.0*-cp310-cp310m-linux_aarch64.whl

# apex for Ascend 参考 https://gitee.com/ascend/apex
# 建议从原仓编译安装

# 安装加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.8.0
git checkout 8db5013167274913e106ce6f31aca0a3b085effb
pip install -r requirements.txt
pip3 install -e .
cd ..

# 安装其余依赖库
pip install -e .
```

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

- [
InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B)；

- [
InternVL2_5-78B](https://huggingface.co/OpenGVLab/InternVL2_5-78B)；

将模型权重保存在`raw_ckpt`目录下，例如`raw_ckpt/InternVL2_5-78B`。

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的转换以及PP（Pipeline Parallel）的权重切分。

`mm-convert`工具详细用法参考[权重转换工具](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)。


```bash
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 4B
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2_5-4B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2_5-4B" \
  --cfg.parallel_config.llm_pp_layers [[36]] \
  --cfg.parallel_config.vit_pp_layers [[24]] \
  --cfg.trust_remote_code True

# 78B
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2_5-78B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2_5-78B" \
  --cfg.parallel_config.llm_pp_layers [[0,3,6,6,6,6,6,6,6,6,6,6,5,5,5,2]] \
  --cfg.parallel_config.vit_pp_layers [[45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] \
  --cfg.trust_remote_code True

# 其中：
# mm_dir: 转换后保存目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# trust_remote_code: 为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性
```

同步修改`examples/internvl2.5/finetune_internvl2.5_*b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重`raw_ckpt/InternVL2_5-*B`进行区分。

以`InternVL2_5-78B`为例
```shell
LOAD_PATH="pretrained/InternVL2_5-78B"
```

---

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

【图片数据】

用户需自行获取并解压[InternVL-Finetune](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data)数据集到`dataset/playground`目录下，以数据集ai2d为例，解压后的数据结构如下：

   ```
   $playground
   ├── data
       ├── ai2d
           ├── abc_images
           ├── images
   ├── opensource
       ├── ai2d_train_12k.jsonl
   ```


<a id="jump4"></a>

## 微调

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`from_pretrained`、`data_path`、`data_folder`等字段。

以InternVL2_5-78B为例，`data_78B.json`进行以下修改，注意`tokenizer_config`的权重路径为转换前的权重路径。

```json
{
  "dataset_param": {
      ...
      "basic_parameters": {
          "data_path": "dataset/playground/opensource/ai2d_train_12k.jsonl",
          "data_folder": "dataset/playground/data/ai2d"
      },
      ...
      "tokenizer_config": {
          ...
          "from_pretrained": "raw_ckpt/InternVL2_5-78B",
          ...
      },
      ...
  },
  ...
}
```

【模型保存加载及日志信息配置】

根据实际情况配置`examples/internvl2.5/finetune_internvl2.5_xx.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）, 以InternVL2.5-78B为例：

```shell
...
# 加载路径
LOAD_PATH="ckpt/InternVL2_5-78B"
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

若需要加载指定迭代次数的权重、优化器等状态，需将加载路径`LOAD_PATH`设置为保存文件夹路径`LOAD_PATH="save_dir"`，并修改`latest_checkpointed_iteration.txt`文件内容为指定迭代次数

```
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

【单机运行配置】

配置`examples/internvl2.5/finetune_internvl2.5_xx.sh`参数如下

```shell
  # 根据实际情况修改 ascend-toolkit 路径
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  NPUS_PER_NODE=8
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动微调

以InternVL2_5-78B为例，启动微调训练任务。

```shell
bash examples/internvl2.5/finetune_internvl2.5_78B.sh
```
<a id="jump5"></a>

## 推理

<a id="jump5.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：环境安装、权重下载及转换，详情可查看对应章节。（当前仅支持4B单卡推理）

推理权重转换命令如下：

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 4B
mm-convert InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2_5-4B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2_5-4B" \
  --cfg.parallel_config.llm_pp_layers [[36]] \
  --cfg.parallel_config.vit_pp_layers [[24]] \
  --cfg.trust_remote_code True
```
<a id="jump5.2"></a>

#### 2. 配置参数

【参数配置】

修改inference_*B.json文件，包括`infer_data_type`、`file_path`、`prompts`、`from_pretrained`以及tokenizer的`from_pretrained`等字段。

【单图推理】

以InternVL2_5-4B为例，按实际情况修改inference_4B.json对应参数，注意tokenizer_config的权重路径为转换前的权重路径。

```json
{
    "infer_data_type": "image",
    "file_path": "./examples/internvl2.5/view.jpg",    # 按实际情况输入图片路径
    "prompts": "Please describe the image shortly.", # 按实际情况输入提示词（支持中英文）
    "model_id": "InternVLPipeline",
    "from_pretrained": "./pretrained/InternVL2_5-4B/release/mp_rank_00/model_optim_rng.pt", # 注意路径要到.pt文件
    ...
    "tokenizer":{
        ...
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "raw_ckpt/InternVL2_5-4B",
        ...
    },
    ...
}
```

【视频推理】

以InternVL2_5-4B为例，按实际情况修改inference_4B.json对应参数，注意tokenizer_config的权重路径为转换前的权重路径。

推理demo视频下载[red-panda](https://huggingface.co/OpenGVLab/InternVL2-8B/blob/main/examples/red-panda.mp4)

```json
{
    "infer_data_type": "video",
    "file_path": "examples/internvl2.5/red-panda.mp4",    # 按实际情况输入视频路径
    "prompts": "Please describe the video shortly.", # 按实际情况输入提示词（支持中英文）
    "model_id": "InternVLPipeline",
    "from_pretrained": "./pretrained/InternVL2_5-4B/release/mp_rank_00/model_optim_rng.pt", # 注意路径要到.pt文件
    ...
    "tokenizer":{
        ...
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "raw_ckpt/InternVL2_5-4B",
        ...
    },
    ...
}
```

【启动脚本配置】
按实际情况修改inference_internvl.sh脚本，

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
...
MM_MODEL="./examples/internvl2.5/inference_4B.json"
```
<a id="jump5.3"></a>

#### 3. 启动推理

```shell
bash examples/internvl2.5/inference_internvl.sh
```

<a id="jump6"></a>

## 环境变量声明
ASCEND_SLOG_PRINT_TO_STDOUT： 是否开启日志打印， 0：关闭日志打屏，1：开启日志打屏  
ASCEND_GLOBAL_LOG_LEVEL： 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。0：对应DEBUG级别，1：对应INFO级别，2：对应WARNING级别，3：对应ERROR级别，4：对应NULL级别，不输出日志  
TASK_QUEUE_ENABLE： 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化  
COMBINED_ENABLE： 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景  
CPU_AFFINITY_CONF： 控制CPU端算子任务的处理器亲和性，即设定任务绑核，设置0或未设置：表示不启用绑核功能， 1：表示开启粗粒度绑核， 2：表示开启细粒度绑核  
HCCL_CONNECT_TIMEOUT:  用于限制不同设备之间socket建链过程的超时等待时间，需要配置为整数，取值范围[120,7200]，默认值为120，单位s  
PYTORCH_NPU_ALLOC_CONF： 控制缓存分配器行为  
ACLNN_CACHE_LIMIT： 配置单算子执行API在Host侧缓存的算子信息条目个数  
NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量

<a id="jump7"></a>

## 注意事项
1. 在使用流水线并行策略进行多机训练可能会出现卡住现象，可参考[此处](https://gitee.com/ascend/MindSpeed/pulls/1627/files)修改。