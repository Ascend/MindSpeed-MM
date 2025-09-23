# OpenSoraPlan1.3.1 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
  - [Decord安装](#jump1.3)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
  - [权重转换](#jump2.2)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
  - [数据集处理](#jump3.2)
- [预训练](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动预训练](#jump4.3)
- [推理](#jump5)
  - [准备工作](#jump5.1)
  - [配置参数](#jump5.2)
  - [启动推理](#jump5.3)
- [特性使用介绍](#jump6)
  - [DistTrain](#jump6.1)
- [环境变量声明](#jump7)

## 版本说明
#### 参考实现
```
url=https://github.com/PKU-YuanGroup/Open-Sora-Plan.git
commit_id=4b14d58
```

#### 变更记录

2024.10.30: 首次发布OpenSoraPlan1.3

---
<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

<a id="jump1.1"></a>

#### 1. 仓库拉取

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
```

<a id="jump1.2"></a>

#### 2. 环境搭建

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
git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
pip install -r requirements.txt
pip3 install -e .
cd ..

# 安装其余依赖库
pip install -e .
```

<a id="jump1.3"></a>

#### 3. Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

- [LanguageBind/Open-Sora-Plan-v1.3.1](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main)：WFVAE模型和SparseVideoDiT模型；

- [DeepFloyd/mt5-xxl](https://huggingface.co/google/mt5-xxl/)： MT5模型；

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，因此需要使用权重转换工具进行转换，该转换工具实现了从hugging face下载的预训练权重到到MindSpeed-MM权重的转换以及TP（Tensor Parallel）和PP（Pipeline Parallel）权重的切分。

转换vae部分的权重

```bash
mm-convert OpenSoraPlanConverter --version v1.3 vae_convert \
	--cfg.source_path <"raw_ckpt/open-sora-plan/any93x640x640/">
	--cfg.target_path <"mm_ckpt/open-sora-plan/pretrained-checkpoint-wfvae">
```

转换dit部分的权重

```bash
mm-convert OpenSoraPlanConverter --version v1.3 hf_to_mm \
	--cfg.source_path <"./raw_ckpt/open-sora-plan/93x480p/">
	--cfg.target_path <"mm_ckpt/open-sora-plan/pretrained-checkpoint-dit">
	--cfg.target_parallel_config.tp_size <tp_size>
	--cfg.target_parallel_config.pp_layers <pp_layers>
```

权重转换工具的参数说明与默认值如下：

| 参数                                   | 含义                                                         | 默认值           |
| :------------------------------------- | :----------------------------------------------------------- | :--------------- |
| --version                              | opensoraplan系列不同版本                                     | 需要设置为`v1.3` |
| --cfg.source_path                      | 原始权重路径                                                 | /                |
| --cfg.target_path                      | 转换或切分后权重保存路径                                     | /                |
| --cfg.target_parallel_config.tp_size   | dit部分切分时的tp size                                       | 1                |
| --cfg.target_parallel_config.pp_layers | dit部分切分时的pp_layer，`[]`表示不开PP，`[8,8,8,8]`表示PP size为4，每个stage 8层，[[4,4,4,4],[4,4,4,4]]表示PP size为4， vpp size为2 | []      
---

同步修改examples/opensoraplan1.3/t2v/pretrain_t2v.sh中的--load参数，该路径为转换后或者切分后的权重，注意--load配置的是转换到MindSpeed-MM后的dit权重路径，vae权重路径在pretrain_t2v_model.json中配置

    LOAD_PATH="mm_ckpt/open-sora-plan/pretrained-checkpoint-dit"

---

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

用户需自行获取并解压[pixabay_v2](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/pixabay_v2_tar)数据集和对应[标注文件](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/anno_json)，获取数据结构如下：

   ```
   $pixabay_v2
   ├── v1.1.0_HQ_part3.json
   ├── folder_01
   ├── ├── video0.mp4
   ├── ├── video1.mp4
   ├── ├── ...
   ├── folder_02
   ├── folder_03
   └── ...
   ```

---
<a id="jump3.2"></a>

#### 2. 数据集处理

根据实际下载的数据，过滤标注文件，删去标注的json文件中未下载的部分；
修改data.txt中的路径，示例如下:

   ```
/data/open-sora-plan/dataset,/data/open-sora-plan/annotation/v1.1.0_HQ_part3.json
   ```

---
其中，第一个路径为数据集的根目录，第二个路径为标注文件的路径。

<a id="jump4"></a>

## 预训练

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

需根据实际情况修改`pretrain_t2v_model.json`和`data.json`中的权重和数据集路径，包括`from_pretrained`、`data_path`、`data_folder`字段。

【并行化配置参数】：

默认场景无需调整，当增大模型参数规模或者视频序列长度时，需要根据实际情况启用以下并行策略，并通过调试确定最优并行策略。

+ CP: 序列并行，当前支持Ulysses序列并行。

  - 使用场景：在视频序列（分辨率×帧数）较大时，可以开启来降低内存占用。
  
  - 使能方式：在启动脚本中设置 CP > 1，如：CP=2。
  
  - 限制条件：head数量需要能够被TP*CP整除。


+ TP: 张量模型并行

  - 使用场景：模型参数规模较大时，单卡上无法承载完整的模型，通过开启TP可以降低静态内存和运行时内存。

  - 使能方式：在启动脚本中设置 TP > 1，如：TP=8。

  - 限制条件：head 数量需要能够被TP*CP整除。


+ SP: Megatron序列并行
  
  - 使用场景：在张量模型并行的基础上，进一步对 LayerNorm 和 Dropout 模块的序列维度进行切分，以降低动态内存。 

  - 使能方式：在 GPT_ARGS 设置 --sequence-parallel。
  
  - 限制条件：前置必要条件为开启TP。


+ PP：流水线并行

  目前支持将predictor模型切分流水线。在pretrain_xx_model.json文件修改字段"pipeline_num_layers", 类型为list。该list的长度即为 pipeline rank的数量，每一个数值代表rank_i中的层数。例如，[8, 8, 8, 8]代表有4个pipeline stage， 每个容纳8个dit layers。注意list中 所有的数值的和应该和num_layers字段相等。此外，pp_rank==0的stage中除了包含dit层数以外，还会容纳text_encoder和ae，因此可以酌情减少第0个 stage的dit层数。注意保证PP模型参数配置和模型转换时的参数配置一致。

  - 使用场景：模型参数较大时候，通过流水线方式切分并行，降低内存。

  - 使能方式：使用pp时需要在运行脚本GPT_ARGS中打开以下几个参数。
  
  ```shell
    PP = 4 # PP > 1 开启 
  
    --optimization-level 2 \
    --use-multiparameter-pipeline-model-parallel \
    --variable-seq-lengths \
  
    # 同时pretrain_xx_model.json中修改相应配置 
    "pipeline_num_layers": [8, 8, 8, 8],
  ```


+ VP: 虚拟流水线并行

  目前支持将predictor模型切分虚拟流水线并行。将pretrain_xxx_model.json文件中的"pipeline_num_layers"一维数组改造为两维，其中第一维表示虚拟并行的数量，二维表示流水线并行的数量，例如[[4, 4, 4, 4], [4, 4, 4, 4]]其中第一维两个数组表示vp为2, 第二维的stage个数为4表示流水线数量pp为4。

  - 使用场景：对流水线并行进行进一步切分，通过虚拟化流水线，降低空泡。

  - 使能方式:如果想要使用虚拟流水线并行，需要在pretrain.t2v.sh或者prerain_i2v.sh当中修改如下变量，需要注意的是，VP仅在PP大于1的情况下生效:

  ```shell
  PP=4
  VP=4

  GPT_ARGS="
    --pipeline-model-parallel-size ${PP} \
    --virtual-pipeline-model-parallel-size ${VP} \
  ...
  ```

+ VAE-CP：VAE序列并行
  - 使用场景：视频分辨率/帧数设置的很大时，训练过程中，单卡无法完成vae的encode计算，需要开启VAE-CP。
  - 使能方式：在xxx_model.json中设置vae_cp_size, vae_cp_size为大于1的整数时生效, 建议设置等于Dit部分cp_size。
  - 限制条件：暂不兼容PP。

+ Encoder-DP：Encoder数据并行
  - 使用场景：在开启TP、CP时，DP较小，存在vae和text_encoder的冗余encode计算。开启以减小冗余计算，但会增加通信，需要按场景开启。T2V、I2V任务均支持。
  - 使能方式：在xxx_model.json中设置"enable_encoder_dp": true。
  - 限制条件：暂不兼容PP、VAE-CP。支持与Encoder Interleaved Offload功能同时开启。

+ Encoder Interleaved Offload: Encoder 交替卸载
  - 使用场景：在NPU内存瓶颈的训练场景中，可以一次性编码多步训练输入数据然后卸载编码器至cpu上，使得文本编码器无需常驻内存，减少内存占用。
    故可在不增加内存消耗的前提下实现在线训练，避免手动离线提取特征。T2V、I2V任务均支持。
  - 使能方式：在xxx_model.json中，设置 encoder_offload_interval > 1. 建议设置根据实际场景设置大于10，可以极小化卸载带来的性能损耗
  - 限制条件：启用时建议调大num_worker以达最佳性能; 支持与Encoder-DP同时开启。


+ DiT-RingAttention：DiT RingAttention序列并行

  - 使用场景：视频分辨率/帧数设置的很大时，训练过程中，单卡无法完成DiT的计算，需要开启DiT-RingAttention。
  - 使能方式：在启动脚本 pretrain_xxx.sh 中修改如下变量。

  ```shell
  CP=8
  
  GPT_ARGS="
    --context-parallel-size ${CP} \
    --context-parallel-algo megatron_cp_algo \
    --attention-mask-type general \
    --use-cp-send-recv-overlap \
    --cp-window-size 1
  ...
  ```

  - ```--use-cp-send-recv-overlap```为可选参数，建议开启，开启后支持send receive overlap功能。
  - ```--cp-window-size [int]```为可选参数，设置算法中双层Ring Attention的内层窗口大小，需要确保cp_size能被该参数整除。
    - 缺省值为1，即使用原始的Ring Attention算法。
    - 大于1时，即使用Double Ring Attention算法，优化原始Ring Attention性能。


+ DiT-USP: DiT USP混合序列并行（Ulysses + RingAttention）
  - 使用场景：视频分辨率/帧数设置的很大时，训练过程中，单卡无法完成DiT的计算，需要开启DiT-RingAttention。
  - 使能方式：在启动脚本pretrain_xxx.sh中修改如下变量。

  ```shell
  CP=8
  
  GPT_ARGS="
    --context-parallel-size ${CP} \
    --context-parallel-algo hybrid_cp_algo \
    --attention-mask-type general \
    --use-cp-send-recv-overlap \
    --ulysses-degree-in-cp [int]
  ...
  ```

  - 需要确保```--context-parallel-size```可以被```--ulysses-degree-in-cp```整除且大于1。
    - 例如当设置```--context-parallel-size```为8时，可以设置```--ulysses-degree-in-cp```为2或```--ulysses-degree-in-cp```为4。
    - 同时需要确保```--ulysses-degree-in-cp```可以被num-attention-heads数整除。
  - RingAttention相关参数解析见DiT-RingAttention部分。

【动态/固定分辨率】
- 支持使用动态分辨率或固定分辨率进行训练，默认为动态分辨率训练，如切换需修改启动脚本pretrain_xxx.sh
```shell
    # 以t2v实例，使用动态分辨率训练
    MM_DATA="./examples/opensoraplan1.3/t2v/data_dynamic_resolution.json"
    
    # 以t2v实例，使用固定分辨率训练
    MM_DATA="./examples/opensoraplan1.3/t2v/data_static_resolution.json"
```

【单机运行】

```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=localhost
    MASTER_PORT=29501
    NNODES=1  
    NODE_RANK=0  
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

【多机运行】

```shell
    # 根据分布式集群实际情况配置分布式参数
    GPUS_PER_NODE=8  #每个节点的卡数
    MASTER_ADDR="your master node IP"  #都需要修改为主节点的IP地址（不能为localhost）
    MASTER_PORT=29501
    NNODES=2  #集群里的节点数，以实际情况填写,
    NODE_RANK="current node id"  #当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动预训练

t2v(文生视频):

```shell
    bash examples/opensoraplan1.3/t2v/pretrain_t2v.sh
```

i2v(图生视频):

```shell
    bash examples/opensoraplan1.3/i2v/pretrain_i2v.sh
```

**注意**：

- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，需要在每个节点准备训练数据和模型权重

---

<a id="jump5"></a>

## 推理

<a id="jump5.1"></a>

#### 1. 准备工作

参考上述的权重下载及转换章节，推理所需的预训练权重需要到huggingface中下载，以及参考上面的权重转换步骤进行转换。

如果在训练时对dit部分进行了TP切分，推理时可以运行下面的脚本将该部分权重进行合并

```bash
mm-convert OpenSoraPlanConverter --version v1.3 resplit \
	--cfg.source_path <"./raw_ckpt/open-sora-plan/93x480p/">
	--cfg.target_path <"mm_ckpt/open-sora-plan/pretrained-checkpoint-dit">
```


<a id="jump5.2"></a>

#### 2. 配置参数

将准备好的权重传入到inference_t2v_model.json中，更改其中的路径，包括from_pretrained，自定义的prompt可以传入到prompt字段中

对于i2v任务，除了上述prompt，还需要将自定义的图片数据传入inference_i2v_model.json中的conditional_pixel_values_path字段中

<a id="jump5.3"></a>

#### 3. 启动推理

t2v 启动推理脚本

```shell
bash examples/opensoraplan1.3/t2v/inference_t2v.sh
```

i2v 启动推理脚本

```shell
bash examples/opensoraplan1.3/i2v/inference_i2v.sh
```
---
<a id="jump6"></a>
## 特性使用介绍

<a id="jump6.1"></a>
### DistTrain(分离部署)

#### 1. 特性介绍
DistTrain特性详细介绍参考文档[分离部署特性](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/dist-train.md)

#### 2. 模型分离部署权重转换

提供了MM CKPT与DistTrain CKPT之间的权重转换工具。

MM CKPT转DistTrain CKPT：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python examples/opensoraplan1.3/opensoraplan1_3_mm_convert_to_dt_ckpt.py \
  --load-dir mm_ckpt/open-sora-plan/pretrained-checkpoint-dit \
  --save-dir mm_ckpt/open-sora-plan/pretrained-checkpoint-dit-dist-train \
  --target-vae-tp-size 1 \
  --target-vae-pp-size 1 \
  --target-vae-cp-size 1 \
  --target-dit-tp-size 1 \
  --target-dit-pp-size 3 \
  --target-dit-cp-size 1 \
  --target-dit-pp-layers '[10,11,11]'
```

DistTrain CKPT转MM CKPT：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python examples/opensoraplan1.3/opensoraplan1_3_dt_convert_to_mm_ckpt.py \
  --load-dir mm_ckpt/open-sora-plan/pretrained-checkpoint-dit-dist-train \
  --save-dir mm_ckpt/open-sora-plan/pretrained-checkpoint-dit-dist-train-to-mm \
  --target-tp-size 1 \
  --target-pp-size 4 \
  --target-cp-size 1 \
  --target-dit-pp-layers '[8,8,8,8]'
```

同步修改`examples/opensoraplan1.3/t2v/pretrain_t2v.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重进行区分。

#### 3. 使用方法
以`opensoraplan1.3-t2v`为例

在启动脚本中添加参数`--dist-train`。
```shell
GPT_ARGS="
    ...
    --dist-train \
"
```

需要在MindSpeed-MM仓库中，对应模型目录下的`pretrain_t2v_model.json`中添加`dist_config`字段，具体配置示例如下：
```json
{
  "dist_config": {
    "model_name": "opensoraplan1.3",  // 多模态模型名称
    "use_multiparam_send_recv": true,  // 模型间是否传递tensor列表
    "model_config": [
      {
        "name": "vae",  // 内部模型名称
        "model_index": 0,  // 模型位于流水线中的序号
        "world_size": 1,  // 模型使用卡数
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "forward_only": true // 是否不做反向计算
      },
      {
        "name": "dit",
        "model_index": 1,
        "world_size": 8,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 4,
        "forward_only": false
      }
    ]
  }
}
```

当前的测试配置中最少需要9张卡，对于单机8卡设备需要配置双机运行，以单机8卡设备双机为例：
需要在`examples/opensoraplan1.3/t2v/pretrain_t2v.sh`中做如下类似修改：
```shell
IPs=('xxx' 'xxx')
LOCAL_HOST=$(ifconfig enp189s0f0 | grep 'inet ' | awk '{print $2}')
if [ "$LOCAL_HOST" = "${IPs[0]}" ]; then
    export ASCEND_RT_VISIBLE_DEVICES="0"
elif [ "$LOCAL_HOST" = "${IPs[1]}" ]; then
   export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
else
   echo "[Error] check IPs and LOCAL_HOST"
   exit 1
fi
MASTER_ADDR=${IPs[0]}
MASTER_PORT=6000
NNODES=${#IPs[@]}
NODE_RANK=""
for i in "${!IPs[@]}";
do 
  echo "${IPs[$i]}"
  echo "$LOCAL_HOST"
  if [ "$LOCAL_HOST" == "${IPs[$i]}" ];
  then 
    NODE_RANK=$i
    break
  fi
done
if [[ $NODE_RANK == "" ]]; then
  echo "[Error] para \"NODE_RANK\" must be error"
  exit 1
fi
GPUS_PER_NODE=$(echo $ASCEND_RT_VISIBLE_DEVICES | tr ',' '\n' | grep -c '^[0-9]')  

```

由于该模型vae是冻结不训练的，使能分离部署时，还需要打开dummy_optimizer，使能方式参考[dummy_optimizer](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/dummy_optimizer.md)
   ```shell
   GPT_ARGS="
       ...
       --enable-dummy-optimizer \
   "
   ```

---
<a id="jump7"></a>
## 环境变量声明

| 环境变量                      | 描述                                                                 | 取值说明                                                                                         |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 是否开启日志打印                                                           | `0`: 关闭日志打屏<br>`1`: 开启日志打屏                                                                   |
| `ASCEND_GLOBAL_LOG_LEVEL`     | 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志                             | `0`: 对应DEBUG级别<br>`1`: 对应INFO级别<br>`2`: 对应WARNING级别<br>`3`: 对应ERROR级别<br>`4`: 对应NULL级别，不输出日志 |
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                                    | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                              |
| `COMBINED_ENABLE`             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景 | `0`: 关闭<br>`1`: 开启                                                                           |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                    | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                                     |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                                  | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                          | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征                                            |
| `HCCL_EXEC_TIMEOUT`           | 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步         | 需要配置为整数，取值范围`[68,17340]`，默认值为`1800`，单位`s`                                                    |
| `ACLNN_CACHE_LIMIT`           | 配置单算子执行API在Host侧缓存的算子信息条目个数                                  | 需要配置为整数，取值范围`[1, 10,000,000]`，默认值为`10000`                                                    |
| `TOKENIZERS_PARALLELISM`      | 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为    | `False`: 禁用并行分词<br>`True`: 开启并行分词                                                            |
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用                                                               |
| `NPU_ASD_ENABLE`   | 控制是否开启Ascend Extension for PyTorch的特征值检测功能 | 设置`0`或未设置: 关闭特征值检测<br>`1`: 表示开启特征值检测，只打印异常日志，不告警<br>`2`:开启特征值检测，并告警<br>`3`:开启特征值检测，并告警，同时会在device侧info级别日志中记录过程数据 |
| `ASCEND_LAUNCH_BLOCKING`   | 控制算子执行时是否启动同步模式 | `0`: 采用异步方式执行<br>`1`: 强制算子采用同步模式运行                                                               |
| `NPUS_PER_NODE`               | 配置一个计算节点上使用的NPU数量                                                  | 整数值（如 `1`, `8` 等）                                                                            |
