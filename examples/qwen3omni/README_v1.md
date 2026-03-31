# Qwen3-Omni 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#jump1)
  - [环境准备](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
  - [权重转换](#jump2.2)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
  - [混合数据集处理](#jump3.2)
- [微调](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动微调](#jump4.3)
- [环境变量声明](#jump10)
- [注意事项](#jump11)

## 版本说明

### 参考实现

```bash
url=https://github.com/huggingface/transformers.git
commit_id=7a833d1
```

### 变更记录

2026.03.30: 首次支持纯fsdp2后端的Qwen3-Omni模型

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md)，完成昇腾软件安装。
> Python版本推荐3.10，torch和torch_npu版本推荐2.7.1版本

<a id="jump1.2"></a>

### 2. 环境搭建

拉取MindSpeed MM代码仓：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
# 安装mindspeed及依赖
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
cp -r mindspeed ../MindSpeed-MM/

# 安装mindspeed-mm及依赖
cd ../MindSpeed-MM
pip install -e .

# 安装transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 7a833d1
pip install -e .

# 安装其它依赖
pip install accelerate==1.11.0 librosa==0.11.0 datasets==4.0.0

```

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

### 1. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/collections/Qwen/qwen3-omni)；

将下载的模型权重保存到本地的``ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct`目录下。

<a id="jump2.2"></a>

### 2. 权重转换

当前用多卡微调时，会遇到梯度通信问题，根据是否使能use_grouped_expert_matmul，做不同的权重转换

1.如果使能use_grouped_expert_matmul，需对原始预训练权重按如下方式进行转换：

```shell
mm-convert ExpertMergeDcpConverter hf_to_dcp \
  --hf_dir "ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct" \
  --save_dir "ckpt/convert_path/Qwen3-Omni-30B-A3B-Instruct" \
  --dcp_prefix ""

# 转换后的目录结构为：
# ———— Qwen3-Omni-30B-A3B-Instruct
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```

并在`xxx_config.yaml`中将`init_model_with_meta_device`参数配置为`true`，同时将`load`参数修改为转换后的dcp权重路径（写到`release`文件夹的上一级目录），`model_name_or_path`仍为转换前的权重路径。

训练完成之后，支持将保存在`save`目录下的权重转换成huggingface格式：

```shell
mm-convert ExpertMergeDcpConverter dcp_to_hf \
  --hf_dir "ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct" \
  --dcp_dir "save_path/iter_00000xx" \
  --save_dir "ckpt/dcp_to_hf_test/Qwen3-Omni-30B-A3B-Instruct" \
  --dcp_prefix ""
```

其中，`--hf_dir`表示原始huggingface权重的路径，`--dcp_dir`表示微调后的权重保存路径，路径中的`iter_00000xx`表示保存的第xx步权重，`--save_dir`表示转换后的huggingface格式权重保存路径。

2.如果关闭use_grouped_expert_matmul，需对原始预训练权重按如下方式进行转换：

```shell
mm-convert GenericDCPConverter hf_to_dcp \
  --hf_dir "ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct" \
  --dcp_dir "ckpt/convert_path/Qwen3-Omni-30B-A3B-Instruct" \
  --hf_prefix "thinker."
```

并在`xxx_config.yaml`中将`init_model_with_meta_device`参数配置为`true`，同时将`load`参数修改为转换后的dcp权重路径（写到`release`文件夹的上一级目录），`model_name_or_path`仍为转换前的权重路径。

训练完成之后，支持将保存在`save`目录下的权重转换成huggingface格式：

```shell
mm-convert GenericDCPConverter dcp_to_hf \
  --model_assets_dir "ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct" \
  --load_dir "save_path/iter_00000xx" \
  --save_dir "ckpt/dcp_to_hf/Qwen3-Omni-30B-A3B-Instruct" \
  --hf_prefix "thinker."
```

其中，`--model_assets_dir`表示原始huggingface权重的路径，`--load_dir`表示微调后的权重保存路径，路径中的`iter_00000xx`表示保存的第xx步权重，`--save_dir`表示转换后的huggingface格式权重保存路径。

---
<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

### 1. 数据集下载(以coco2017数据集为例)

(1)用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中。

(2)获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下。

(3)运行数据转换脚本python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py，转换后参考数据目录结构如下：

   ```shell
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

---
当前支持读取多个以`,`（注意不要加空格）分隔的数据集，配置方式为相应xxx_config_v1.yaml中
data->dataset_param->basic_parameters->dataset
从"./data/mllm_format_llava_instruct_data.json"修改为"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"

同时注意`data->dataset_param->basic_parameters->max_samples`的配置，会限制数据只读`max_samples`条，这样可以快速验证功能。如果正式训练时，可以把该参数去掉则读取全部的数据。

<a id="jump3.2"></a>

### 2.纯文本或有图无图混合训练数据(以LLaVA-Instruct-150K为例)

现在本框架已经支持纯文本/混合数据（有图像和无图像数据混合训练）。

在数据构造时，对于包含图片的数据，需要保留`image`这个键值。

```python
{
  "id": your_id,
  "image": your_image_path,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

在数据构造时，对于纯文本数据，可以去除`image`这个键值。

```python
{
  "id": your_id,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

<a id="jump4"></a>

## 微调

<a id="jump4.1"></a>

### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

### 2. 配置参数

【数据目录配置】

根据实际情况修改`xxx_config_v1.yaml`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

示例：如果数据及其对应的json都在/home/user/data/目录下，其中json目录为/home/user/data/video_data_path.json，此时配置如下：
`dataset_dir`配置为/home/user/data/;
`dataset`配置为./data/video_data_path.json
注意此时`dataset`需要配置为相对路径
**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

【音频训练】

如果需要进行音频数据训练，需要对`attr`进行修改，删除`images`字段，并设置`audios`字段。输入音频采样率可以通过`audio_sampling_rate`字段进行配置，训练时会自动重采样到16kHz，以适配Qwen3-Omni音频特征提取。

【音视频训练】

如果需要支持语音、视频数据，并进行跨模态融合，需要对`attr`进行修改，删除`images`字段，并设置`videos`和`audios`字段，可以将`use_audio_in_video`设置为true.

【大量数据训练】

如果加载大量数据遇到通信TIMEOUT，可以在`basic_parameters`中添加`preprocess_on_fly`字段并置为true。

【activation_offload配置】

使用activation_offload可以将重计算过程中产生的checkpoint点的激活值移动到host，反向异步从host传输到device，降低device激活显存占用，配置方式为在`xxx_config_v1.yaml`中的`model`模型配置增加`activation_offload_plan`字段。

```yaml
  activation_offload_plan:
    apply_modules:
      - model.layers.{*}
```

【chunkloss 配置】

参考[chunk loss文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/features/chunkloss.md)
将`xxx_config_v1.yaml`中`enable_chunk_loss`字段设置为true，chunk_size表示每个子序列的最大长度（即每个 chunk 所包含的 token 数量）

```yaml
  enable_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    chunk_size: 1024
```

也可以设置动态chunk，约束动态分块的总计算量，根据batch_size自适应调整分块大小

```yaml
  enable_dynamic_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    total_chunk_size: 4096
```

【模块冻结配置】

当前支持自定义冻结模块，在`xxx_config_v1.yaml`中model->freeze字段中配置需要冻结的模块即可实现相应模块冻结。

【模型保存加载及日志信息配置】

根据实际情况配置`xxx_config_v1.yaml`的`training`参数，包括保存路径以及保存间隔`save`、`save_interval`

【单机运行配置】

以qwen3omni模型为例：
配置`examples/qwen3omni/finetune_qwen3omni_v1.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
```

【多机运行配置】

如需拉起多机训练，修改启动脚本下 MASTER_ADDR、MASTER_PORT、NNODES以及NODE_RANK变量

``` shell
export GLOO_SOCKET_IFNAME="Your SOCKET IFNAME" # 通过ifconfig获取
MASTER_ADDR=<master_ip_address> # 都要修改为主节点的IP地址（不能为localhost)
MASTER_PORT=6000 # 各个节点保持一致
NNODES=2 # 集群里的节点数，根据实际情况填写
NODE_RANK=0 # 当前节点的RANK，多个节点不能重复，主节点为0，其他节点可以是1,2..
```

---

<a id="jump4.3"></a>

### 3. 启动微调

```shell
cd MindSpeed-MM/
bash examples/qwen3omni/finetune_qwen3omni_v1.sh
```

<a id="jump10"></a>

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

---
<a id="jump11"></a>

## 注意事项

‼️当前用多卡微调时，会遇到梯度通信问题，需要使能use_grouped_expert_matmul，在transformers中对MOE实现方式改写，性能更好；如果关闭use_grouped_expert_matmul，让所有专家参与前向运算，性能较差
