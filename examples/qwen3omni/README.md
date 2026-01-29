# Qwen3_Omni 使用指南

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
- [支持工具调用数据的微调](#jump4.5)
- [环境变量声明](#jump10)
- [注意事项](#jump11)

## 版本说明
#### 参考实现
```
url=https://github.com/huggingface/transformers.git
commit_id=7a833d1
```

#### 变更记录

2025.11.13: 首次支持Qwen3-Omni模型

---
<a id="jump1"></a>
## 环境安装

<a id="jump1.1"></a>
#### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)，完成昇腾软件安装。
> Python版本推荐3.10，torch和torch_npu版本推荐2.7.1版本

推荐使用以下版本

- [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)
- [torch_npu](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html)

<a id="jump1.2"></a>
#### 2. 环境搭建
```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
mkdir logs data ckpt

# 安装加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout d76dbddd

# 安装mindspeed及依赖
pip install -e .
cd ..
# 安装mindspeed mm及依赖
pip install -e .

# 安装新版transformers（支持qwen3omni模型）
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 7a833d1
pip install -e .
pip install accelerate==1.11.0 librosa==0.11.0 datasets==4.0.0
```

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Hugging Face库下载对应的模型权重:

- 模型地址: [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/collections/Qwen/qwen3-omni)；

将下载的模型权重保存到本地的`ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct`目录下。

#### 特别说明

权重下载后，需修改权重路径下的`ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct/config.json`代码文件，将`enable_audio_output`的true修改为false

<a id="jump2.2"></a>

#### 2. 权重转换

当前用多卡微调时，会遇到梯度通信问题，MindSpeed-MM修改了transformers中MOE实现方式，需对原始预训练权重进行转换：

```shell
mm-convert ExpertMergeDcpConverter hf_to_dcp \
  --hf_dir "ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct" \
  --save_dir "ckpt/convert_path/Qwen3-Omni-30B-A3B-Instruct"
```
并在examples/qwen3omni/finetune_qwen3omni.sh的`GPT_ARGS`中加入`--init-model-with-meta-device`参数。

训练完成之后，支持将保存在`SAVE_PATH`目录下的权重转换成huggingface格式：
```shell
mm-convert ExpertMergeDcpConverter dcp_to_hf \
  --hf_dir "ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct" \
  --dcp_dir "save_dir/iter_000xx" \
  --save_dir "ckpt/dcp_to_hf/Qwen3-Omni-30B-A3B-Instruct"
```

其中，`--hf_dir`表示原始huggingface权重的路径，`--dcp_dir`表示微调后的权重保存路径，路径中的`iter_000xx`表示保存的第xx步权重，`--save_dir`表示转换后的huggingface格式权重保存路径。

 ---
<a id="jump3"></a>
## 数据集准备及处理

<a id="jump3.1"></a>
#### 1. 数据集下载（以COCO2017数据集为例）

(1)用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中。

(2)获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下。

(3)运行数据转换脚本python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py，转换后参考数据目录结构如下：

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
当前支持读取多个以`,`（注意不要加空格）分隔的数据集，配置方式为`data.json`中
dataset_param->basic_parameters->dataset
从"./data/mllm_format_llava_instruct_data.json"修改为"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"

同时注意`data.json`中`dataset_param->basic_parameters->max_samples`的配置，会限制数据只读`max_samples`条，这样可以快速验证功能。如果正式训练时，可以把该参数去掉则读取全部的数据。

<a id="jump3.2"></a>
#### 2.混合数据集处理(以LLaVA-Instruct-150K为例)

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
#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>
#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

示例：如果数据及其对应的json都在/home/user/data/目录下，其中json目录为/home/user/data/video_data_path.json，此时配置如下：
`dataset_dir`配置为/home/user/data/;
`dataset`配置为./data/video_data_path.json
注意此时`dataset`需要配置为相对路径

以Qwen3Omni为例，`data.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径,即原始hf权重路径。

**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct",
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

如果需要加载大批量数据，可使用流式加载，修改`data.json`中的`sampler_type`字段，增加`streaming`字段。（注意：使用流式加载后当前仅支持`num_workers=0`，单进程处理数据，会有性能波动，并且不支持断点续训功能。）
```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            ...
            "streaming": true
            ...
        },
        ...
    },
    "dataloader_param": {
        ...
        "sampler_type": "stateful_distributed_sampler",
        ...
    }
}

```

如果需要进行音频数据训练，需要对`attr`进行修改，`images`字段设为null，并设置`audios`字段。输入音频采样率可以通过`audio_sampling_rate`字段进行配置，训练时会自动重采样到16kHz，以适配Qwen3-Omni音频特征提取。

```json
{
    "dataset_param": {
        ...
        "preprocess_parameters": {
            ...
            "audio_sampling_rate": 16000
            ...
        },
        ...
    },
    ...
    "attr": {
        ...
        "system": null,
        "images": null,
        "videos": null,
        "audios": "audios",
        ...
    }
}

```

如果需要支持语音、视频数据，并进行跨模态融合，可以将`use_audio_in_video`设置为true.
```json
{
    "dataset_param": {
        ...
        "preprocess_parameters": {
            ...
            "use_audio_in_video": true,
            ...
        },
        "attr": {
            ...
            "images": null,
            "videos": "videos",
            "audios": "audios",
            ...
        },
        ...
    },
    ...
}
```

如果加载大量数据遇到通信TIMEOUT，可以在`data.json`中添加`dataset_param.basic_parameters.preprocess_on_fly`字段并置为true。

【模型保存加载及日志信息配置】

根据实际情况配置`examples/qwen3omni/finetune_qwen3omni.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 权重加载路径：转换后的权重
LOAD_PATH="./ckpt/convert_path/Qwen3-Omni-30B-A3B-Instruct"
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
    --save $SAVE_PATH \ # 保存路径
"
```

根据实际情况配置`examples/qwen3omni/model.json`中的`init_from_hf_path`参数，该参数表示初始权重的加载路径。
根据实际情况配置`examples/qwen3omni/model.json`中的`image_encoder.vision_encoder.freeze`、`image_encoder.vision_projector.freeze`、`audio_encoder.audio_encoder.freeze`、`text_decoder.freeze`参数，该参数分别代表是否冻结vision model模块、multi model projector模块、audio model模块、及language model模块。
注：当前`examples/qwen3omni/model.json`中的各网络层数均为未过校验的无效配置，如需减层请修改原始hf路径下相关配置文件config.json。

【单机运行配置】

配置`examples/qwen3omni/finetune_qwen3omni.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 和 nnal 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 单机16卡可以跑满层
NPUS_PER_NODE=16
# 如果想要指定单卡0，则增加export ASCEND_RT_VISIBLE_DEVICES=0
# 并修改NPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
# 可以修改步数为5000步
--train-iters 5000
```

【多机运行配置】

配置`examples/qwen3omni/finetune_qwen3omni.sh`参数如下（性能场景默认双机运行配置）

```shell
# 根据实际情况修改 ascend-toolkit 和 nnal 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 根据分布式集群实际情况配置分布式参数
export GLOO_SOCKET_IFNAME="Your SOCKET IFNAME" # 通过ifconfig获取
# 如果节点的卡数大于8，需要指定设备，如果指定前8卡，则可以按如下设置
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPUS_PER_NODE=8 # 每个节点的卡数，根据实际情况填写，性能场景每个节点8卡
MASTER_ADDR=<master_ip_address> # 都要修改为主节点的IP地址（不能为localhost)
MASTER_PORT=6000 # 各个节点保持一致
NNODES=2 # 集群里的节点数，根据实际情况填写
NODE_RANK=0 # 当前节点的RANK，多个节点不能重复，主节点为0，其他节点可以是1,2..
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
# 可以修改步数为200步
--train-iters 200
```

<a id="jump4.3"></a>
#### 3. 启动微调

以Qwen3-Omni为例，启动微调训练任务。  
loss计算方式差异会对训练效果造成不同的影响，在启动训练任务之前，请查看关于loss计算的文档，选择合适的loss计算方式[vlm_model_loss_calculate_type.md](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/vlm_model_loss_calculate_type.md)
```shell
cd MindSpeed-MM/
bash examples/qwen3omni/finetune_qwen3omni.sh
```
<a id="jump4.5"></a>
#### 4.支持工具调用数据的微调
- 数据组织格式：
```json
{
'messages': [
{'role': 'system', 'content': '你是一名专业、高效的AI智能客服。当前对话的时间为: 2024年06月15日 14时30分00秒 (今天是星期六)。'}, 
{'role': 'user', 'content': '您好，在不在'}, 
{'role': 'assistant', 'content': '"您好，请问是杜女士吗？'}, 
{'role': 'user', 'content': '您好，是的，我是杜女士。'}, 
{'role': 'assistant', 'content': '杜女士，感谢您的确认。我是疾控中心的工作人员。我们注意到您还没有接种新型流感疫苗。为了您的健康和公共卫生安全，我们建议您尽快接种。'},
{'role': 'user', 'content': '我现在身体状况挺好的，暂时不想接种疫苗。'}, 
{'role': 'assistant', 'content': '理解您的顾虑，但接种疫苗不仅能有效预防感染，还能保护您和周围人的健康。特别是在当前疫情环境下，接种疫苗是非常重要的。'},
{'role': 'user', 'content': '我担心接种后会有不良反应，真的安全吗？'}, 
{'role': 'assistant', 'content': '您的担忧是可以理解的。新型流感疫苗经过严格的安全性测试，具有较高的安全性。接种后可能会有轻微的不适，但大多数人都能顺利度过。此外，接种后我们会要求您留观30分钟，以确保及时处理任何不良反应。'},
{'role': 'user', 'content': '好吧，听您这么说，我同意接种疫苗。'}, 
{'role': 'assistant', 'content': '非常感谢您的理解和支持。我们提供的接种时间是周一到周六，每天早上九点到十二点，下午两点到晚上九点。您可以选择一个方便的时间来接种。'},
{'role': 'user', 'content': '那我选择周三下午三点可以吗？'}, 
{'role': 'tool_call', 'content': '<tool_call>\n{"name": "register_vaccine_appointment", "arguments": {"appointment_time": "周三下午三点"}}\n</tool_call>'}, 
{'role': 'tool_response', 'content': '{"status": "success", "message": "预约成功"}'}, 
{'role': 'assistant', 'content': '好的，已经为您登记了周三下午三点的接种时间。请您携带有效身份证件按时前往指定接种点。感谢您对疫情防控工作的支持。'},
{'role': 'user', 'content': '谢谢您，辛苦了。'}, 
{'role': 'assistant', 'content': '不客气，这是我们的职责。祝您健康，再见。", "id": "客服.礼貌结束"}'}, 
], 

'audios': '/speeches/7_Katerina.wav',
'tools': ['{"type": "function", "function": {"name": "register_vaccine_appointment", "description": "登记用户的疫苗接种预约", "parameters": {"type": "object", "properties": {"appointment_time": {"type": "string", "description": "用户选择的接种时间"}}, "required": ["appointment_time"]}}}']}
```

 <font color='red'>请注意 tools的数据类型是list[str]</font>

- 修改data.json
```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            "template": "qwen3_omni_nothink",
        },
        "attr": {
          ...
            "system_tag": "system",
            "formatting": "multimodal_tool"
        }
    },
    ...
}
```

---

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
‼️当前用多卡微调时，会遇到梯度通信问题，需要在transformers中对MOE实现方式改写，需要转换权重的改写方式可以有更好的性能，其他改写方式（比如，让所有专家参与前向运算）的性能较差
