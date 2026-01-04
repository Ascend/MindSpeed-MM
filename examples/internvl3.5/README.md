# InternVL3.5 使用指南

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
  - [权重转换](#jump2.2)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
- [微调](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动微调](#jump4.3)
  - [启动推理](#jump4.4)
- [环境变量声明](#jump6)

## 版本说明
#### 参考实现
```
url=https://github.com/OpenGVLab/InternVL.git
commit_id=2410d1db
```

#### 变更记录

2025.10.21: 首次支持InternVL3_5--30B-A3B-Instruct模型


---
<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

<a id="jump1.1"></a>
#### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)，完成昇腾软件安装。
> Python版本推荐3.10，torch和torch_npu版本推荐2.7.1版本

<a id="jump1.2"></a>

#### 2. 环境搭建
```bash
git clone --branch 2.3.0 https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
mkdir logs dataset ckpt

# 安装加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout 93c45456c7044bacddebc5072316c01006c938f9

# 安装mindspeed及依赖
pip install -e .
cd ..
# 安装mindspeed mm及依赖
pip install -e .

# 安装新版transformers（支持internvl3.5模型）
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout c0dbe09
pip install -e .

```

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

### 权重下载

从Hugging Face等网站下载开源模型权重

- [InternVL3_5-30B-A3B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-Instruct)

将模型权重保存在`ckpt/hf_path/`目录下，例如`ckpt/hf_path/InternVL3_5-30B-A3B-Instruct`。

#### 特别说明

权重下载后，需修改权重路径下的`ckpt/hf_path/modeling_internvl_chat.py`代码文件，96行将None修改为151671，112行为forward函数增加**kwargs参数

```python
    # 96行示例
    self.img_context_token_id = 151671

    # 112行示例
    def forward(
        ...
        return_dict: Optional[bool] = None,
        **kwargs
    )
```

<a id="jump2.2"></a>

### 权重转换
MindSpeed-MM修改了moe部分原始网络的结构名称，需对原始预训练权重进行转换：

```shell
# 对修改了网络结构的部分进行权重转换
mm-convert ExpertMergeDcpConverter hf_to_dcp --hf_dir "ckpt/hf_path/InternVL3_5-30B-A3B-Instruct" --save_dir "ckpt/convert_path/InternVL3_5-30B-A3B-Instruct"
```

---

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

【图片数据】

用户需自行获取并解压[InternVL-Finetune](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data)数据集到`dataset/playground`目录下，解压后的数据结构如下：

   ```
   $playground
   ├── data
       ├── ai2d
           ├── abc_images
           ├── images
       ├── coco
           ├── train2017
       ├── docvqa
           ├── train
           ├── test
           ├── val
       ├──...
   ├── opensource
       ├── ai2d_train_12k.jsonl
       ├── sharegpt4v_instruct_gpt4-vision_cap100k.jsonl
       ├── chartqa_train_18k.jsonl
       ├── ...
   ```

【视频数据】

使用视频进行训练，可参考[视频数据集构造](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data)自行构造视频数据集。

同时依赖Decord库读取视频，Decord安装方法如下：

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

<a id="jump4"></a>

## 微调

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载** 、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`from_pretrained`、`data_path`、`data_folder`等字段。

以InternVL3_5-30B-A3B-Instruct为例，`data.json`进行以下修改，注意`tokenizer_config`的权重路径为原始权重路径。

```json
{
  "dataset_param": {
      ...
      "basic_parameters": {
          "data_path": "dataset/playground/opensource/sharegpt4v_instruct_gpt4-vision_cap100k.jsonl",
          "data_folder": "dataset/playground/data"
      },
      ...
      "tokenizer_config": {
          ...
          "from_pretrained": "./ckpt/hf_path/InternVL3_5-30B-A3B-Instruct",
          ...
      },
      ...
  },
  ...
}
```

【模型路径配置】

根据实际情况修改`model.json`中的权重路径，包括`from_pretrained`、`data_path`、`data_folder`等字段。

以InternVL3_5-30B-A3B-Instruct为例，`model.json`进行以下修改，`init_from_hf_path`的权重路径为原始权重路径。

```json
{
    ...
    "init_from_hf_path": "./ckpt/hf_path/InternVL3_5-30B-A3B-Instruct",
    ...
}
```

【模型保存加载及日志信息配置】

根据实际情况配置`examples/internvl3.5/finetune_internvl3_5.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）, 以InternVL3_5-30B-A3B-Instruct为例：

```shell
...
# 加载路径：原始权重路径
HF_PATH="ckpt/hf_path/InternVL3_5-30B-A3B-Instruct"
# 加载路径：权重转换后路径
LOAD_PATH="ckpt/convert_path/InternVL3_5-30B-A3B-Instruct"
# 保存路径
SAVE_PATH="internvl35_finetune_result"
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

```
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

【单机运行配置】

配置`examples/internvl3.5/finetune_internvl3_5.sh`参数如下

```shell
  # 根据实际情况修改 ascend-toolkit 路径
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  NPUS_PER_NODE=16 # A3单机可跑
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

【多机运行配置】

配置`examples/internvl3.5/finetune_internvl3_5.sh`参数如下

```shell
  # 根据实际情况修改 ascend-toolkit 路径
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  # 根据分布式集群实际情况配置分布式参数
  GPUS_PER_NODE=8  # 每个节点的卡数，以实际情况填写
  MASTER_ADDR="your master node IP"  # 都需要修改为主节点的IP地址（不能为localhost）
  MASTER_PORT=6000
  NNODES=2  # 集群里的节点数，以实际情况填写
  NODE_RANK="current node id"  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
  WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动微调

以InternVL3_5-30B-A3B-Instruct为例，启动微调训练任务。

```shell
bash examples/internvl3.5/finetune_internvl3_5.sh
```

<a id="jump4.4"></a>

#### 4. 启动推理
训练完成之后，将保存在`SAVE_PATH`目录下的权重转换成huggingface格式
```shell
mm-convert ExpertMergeDcpConverter dcp_to_hf --hf_dir "ckpt/hf_path/InternVL3_5-30B-A3B-Instruct" --dcp_dir "internvl35_finetune_result/iter_000xx" --save_dir "ckpt/dcp_to_hf/InternVL3_5-30B-A3B-Instruct"
```

其中，`--hf_dir`表示原始huggingface权重的路径，`--dcp_dir`表示微调后的权重保存路径，`iter_000xx`表示保存的第xx步的权重，`--save_dir`表示转换后的权重保存路径。

完成权重转换之后，即可使用transformers库进行推理。


<a id="jump6"></a>

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
