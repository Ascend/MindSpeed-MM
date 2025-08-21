# FLUX DanceGRPO 使用指南

<p align="left">
</p>

## 目录

- [简介](#jump0)
- [环境安装](#jump1)
    - [仓库拉取](#jump1.1)
    - [环境搭建](#jump1.2)
- [权重下载](#jump2)
- [数据集准备及处理](#jump3)
- [训练](#jump4)
    - [准备工作](#jump4.1)
    - [启动训练](#jump4.2)

<a id="jump0"></a>

## 简介

以 MindSpeed MM 仓库复现 [DanceGRPO](https://arxiv.org/abs/2505.07818)
后训练方法来帮助用户快速入门，前期需要完成代码仓、环境、数据集以及权重等准备工作，再按照说明中的启动方式启动训练，以下为具体的操作说明。

### 参考实现

DanceGRPO开源代码仓以及对应commit id如下：

```
url=https://github.com/XueZeyue/DanceGRPO
commit_id=2149f36f22db601f9dbf70472fea11576f62a0f6
```

<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](../../docs/user-guide/installation.md)

> DanceGRPO场景下，Python版本推荐3.10


<a id="jump1.1"></a>

### 1. 仓库拉取

```shell
git clone https://gitee.com/ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..

cd MindSpeed-MM
mkdir -p logs data ckpt
cd ..
```

<a id="jump1.2"></a>

### 2. 环境搭建

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# for torch-npu dev version or x86 machine [Optional]
# pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
# 安装torch和torch_npu
pip install torch-2.7.1+cpu-cp310-cp310-*.whl
pip install torch_npu-2.7.1*.whl

# 安装加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 6d63944cb2470a0bebc38dfb65299b91329b8d92
cp -r mindspeed ../MindSpeed-MM/
cd ..

# 安装dance grpo依赖库
cd MindSpeed-MM
pip install -r ./examples/dancegrpo/requirements-lint.txt
cd ..

git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
git checkout 866735ecaae999fa714bd9edfa05aa2672669ee3
pip install -e . 
cd ..
```

### 3.Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)


<a id="jump2"></a>

## 权重下载
创建保存权重的目录：
```bash
cd MindSpeed-MM
mkdir ckpt/flux
mkdir ckpt/hps_ckpt
cd ..
```

下载FLUX预训练权重[FLUX预训练权重](https://huggingface.co/black-forest-labs/FLUX.1-dev)
，下载至MindSpeed MM工程根目录下的ckpt/flux目录中。

下载HPS-v2.1预训练权重[HPS-v2.1预训练权重](https://huggingface.co/xswu/HPSv2/tree/main)
，下载至MindSpeed MM工程根目录下的ckpt/hps_ckpt目录中。

<a id="jump3"></a>

## 数据集准备及处理

下载FLUX DanceGRPO使用的[提示词数据集](https://github.com/XueZeyue/DanceGRPO/blob/main/prompts.txt)。在文件页面点击download
raw file下载文件至MindSpeed MM工程根目录的data目录下。

数据集下载完成后要对数据进行预处理，在启动预处理之前，可以根据自身训练配置需要修改[ 数据预处理脚本 ](./preprocess_flux_rl_embeddings.sh)的配置，以FLUX模型为例：
1. vae模型权重所在路径为`LOAD_PATH`，默认为data/flux；
2. 预处理后的数据集存放路径为`OUTPUT_DIR`，默认为data/rl_embeddings；
3. 提示词文件路径为`PROMPT_DIR`，默认为data/prompts.txt。

上述注意点修改完毕后，可启动脚本进行数据预处理：
```bash
cd MindSpeed-MM
bash examples/dancegrpo/preprocess_flux_rl_embeddings.sh
```

处理后的数据默认会存储在MindSpeed MM根目录下的data/rl_embeddings目录中。

<a id="jump4"></a>

## 训练

<a id="jump4.1"></a>

### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

### 2. 三方库修改
找到使用的Python环境的根目录，对于使用conda安装的环境，可以使用如下指令找到：
```bash
echo $(conda info --envs | grep test) | awk '{print $NF}'
```

1. 将文件`lib/python3.10/site-packages/diffusers/models/embeddings.py`中`FluxPosEmbed`类的`forward`函数的如下代码：
```python
is_mps = ids.device.type == "mps"
freqs_dtype = torch.float32 if is_mps else torch.float64
```
修改为：
```python
is_mps = ids.device.type == "mps"
is_npu = ids.device.type == "npu"
freqs_dtype = torch.float32 if is_mps or is_npu else torch.float64
```

2. 将文件`lib/python3.10/site-packages/diffusers/models/embeddings.py`中的`get_1d_rotary_pos_embed`函数的如下代码：
```python
freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
```
修改为：
```python
freqs_cos = freqs.cos().T.repeat_interleave(2, dim=0).T.contiguous().float()
freqs_sin = freqs.sin().T.repeat_interleave(2, dim=0).T.contiguous().float()
```

3. 将文件`lib/python3.10/site-packages/diffusers/models/attention_processor.py`中`Attention`类的`__init__`函数的如下代码：
```python
elif qk_norm == "rms_norm":
    self.norm_q = RMSNorm(dim_head, eps=eps)
    self.norm_k = RMSNorm(dim_head, eps=eps)
```
修改为：
```python
elif qk_norm == "rms_norm":
    self.norm_q = NpuFusedRMSNorm(dim_head, eps=eps)
    self.norm_k = NpuFusedRMSNorm(dim_head, eps=eps)
```

增加如下类：
```python
class NpuFusedRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return torch_npu.npu_rms_norm(x.to(self.weight.dtype), self.weight, epsilon=self.eps)[0]
```

### 3. 启动训练

以 FLUX 模型为例，在启动训练之前，可根据自身训练配置需要修改[ 启动脚本 ](./posttrain_flux_dancegrpo.sh)的配置：

1. 根据使用机器的情况，修改`NNODES`、`NPUS_PER_NODE`配置， 例如单机8卡 可设置`NNODES`为 1 、`NPUS_PER_NODE`为8；
2. 如果为多机训练，需要保证各个节点的`MASTER_ADDR`一致，且为其中一台节点的IP；各节点的`MASTER_PORT`
   配置为相同端口号；从IP为MASTER_ADDR的节点开始，将各节点的`NODE_RANK`配置为从0开始依次递增的整数；
3. 数据集配置信息路径为`MM_DATA`，默认路径为./examples/dancegrpo/data_dancegrpo.json；
4. 模型配置信息路径为`MM_MODEL`，默认路径为./examples/dancegrpo/model_dancegrpo.json；
5. DiT模型预训练权重加载路径为`LOAD_PATH`，默认路径为data/flux，用户也可以根据自身权重存放位置进行调整；
6. 训练权重的保存路径为`SAVE_PATH`，默认为save_dir；
7. 模型训练过程的reward值保存文件的路径为`HPS_REWARD_SAVE_PATH`，默认为./hps_reward.txt。

在启动训练前，可根据自身训练配置需要修改数据集配置[data_dancegrpo.json](./data_dancegrpo.json)：
1. dataset_param.basic_parameters.data_path表示预处理数据中的元数据文件videos2caption.json的路径。

在启动训练前，可根据自身训练配置需要修改模型配置[model_dancegrpo.json](./model_dancegrpo.json)：
1. reward.ckpt_dir表示奖励模型预训练权重的路径。


上述注意点修改完毕后，可启动脚本开启训练：
```bash
bash examples/dancegrpo/posttrain_flux_dancegrpo.sh
```

> *注意：所有节点的代码、权重、数据等路径的层级要保持一致，且启动训练脚本的时候都位于MindSpeed MM目录下*

训练完成后，会在logs目录中生成运行日志文件，生成训练reward记录文件。
