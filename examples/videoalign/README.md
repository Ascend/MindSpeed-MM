# VideoAlign 使用指南


## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
  - [权重转换hf2mm](#jump2.2)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
- [微调](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动微调](#jump4.3)
- [推理](#jump5)
  - [准备工作](#jump5.1)
  - [配置参数](#jump5.2)
  - [启动推理](#jump5.3)
- [评测](#jump6)
  - [准备工作](#jump6.1)
  - [配置参数](#jump6.2)
  - [启动评测](#jump6.3)
- [特性使用介绍](#jump7)
  - [lora微调](#jump7.1)
- [环境变量声明](#jump8)
- [注意事项](#jump9)

## 版本说明
#### 参考实现
```
url=https://github.com/KwaiVGI/VideoAlign.git
commit_id=0150859
```

#### 变更记录

2025.08.26: 首次支持VideoAlign模型

---
<a id="jump1"></a>
## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md)

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
mkdir logs
mkdir data
mkdir ckpt
```

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
git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
# 安装mindspeed及依赖
pip install -e .
cd ..
# 安装mindspeed mm及依赖
pip install -e .

# 指定版本库安装
pip install peft==0.10.0
```

---
<a id="jump2"></a>
## 权重下载及转换

<a id="jump2.1"></a>
#### 1. 权重下载

从Hugging Face库下载对应的模型权重:

- 模型地址: [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/main)；

- 模型地址: [VideoAlign](https://huggingface.co/KwaiVGI/VideoReward/tree/main)；

 将下载的模型权重保存到对应目录下：
 Qwen2-VL-2B：`ckpt/hf_path/Qwen2-VL-2B-Instruct` 
 VideoAlign：`ckpt/hf_path/VideoReward` 

<a id="jump2.2"></a>
#### 2. 权重转换(hf2mm)

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重到MindSpeed-MM权重的转换。参考[权重转换工具](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/features/mm_convert.md)

| 参数 | 含义及用法                                                                                                                                                                                                                                     |
| --- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|mm_dir| 转换后保存目录                                                                                                                                                                                                                                   |
|hf_dir| huggingface权重目录                                                                                                                                                                                                                           |
|pt_path| videoalign模型 .pth格式路径                                                                                                                                                                                                                     |
|llm_pp_layers| llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致                                                                                                                                                                                     |
|vit_pp_layers| vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致                                                                                                                                                                                     |
|tp_size| tp并行数量，注意要和微调启动脚本中的配置一致                                                                                                                                                                                                                   |
|resize_vocab_size| 采用评测指标tokens后的vocab_size<br>会根据vocab_size变化对Qwen2-VL-2B模型embed_tokens.weight层权重进行resize<br>['VQ', 'MQ', 'TA']打分指标对应vocab_size为151660                                                                                                       |
|model_prefix| 消除huggingface中VideoAlign权重里因peft包裹产生的前缀（"base_model.model."）                                                                                                                                                                              |
|new_transformers_weight_key| 是否使用新Qwen2VL权重名的huggingface权重<br>若huggingface的权重名为transformers新权重名：model.language_model.layers.xx, model.visual.blocks.xx（原来权重名为：model.layers.xx, visual.blocks.xx）, 设置如下命令：<br>--cfg.common_model_config.new_transformers_weight_key true \ |
|enable_canonical_hf_struct| MM权重是否使用标准huggingface模型结构。true: 使用huggingface的transformers模型结构；false: 使用megatron原生模型结构。默认为false，启用lora微调时建议开启                                                                                                                             |


```bash
# Qwen2VL权重转mm格式用于训练
mm-convert  VideoAlignConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2-VL-2B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-2B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[28]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 1 \
  --cfg.common_model_config.resize_vocab_size 151660 \
  --cfg.common_model_config.enable_canonical_hf_struct true

# VideoAlign权重转mm用于微调/推理/评测
mm-convert  VideoAlignConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/VideoReward" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-2B-Instruct" \
  --cfg.pt_path "ckpt/hf_path/VideoReward/checkpoint-xxx/model.pth" \
  --cfg.parallel_config.llm_pp_layers [[28]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 1 \
  --cfg.common_model_config.model_prefix "base_model.model." \
  --cfg.common_model_config.enable_canonical_hf_struct true

```

如果需要用转换后模型训练的话，同步修改`examples/videoalign/finetune_lora.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重进行区分。

```shell
LOAD_PATH="ckpt/mm_path/Qwen2-VL-2B-Instruct"
```
或
```shell
LOAD_PATH="ckpt/mm_path/VideoReward"
```


---
<a id="jump3"></a>
## 数据集准备及处理

<a id="jump3.1"></a>
#### 数据集下载(以RewardBench数据集为例)

用户需要自行下载RewardBench数据集[VideoGen-RewardBench](https://huggingface.co/datasets/KwaiVGI/VideoGen-RewardBench/tree/main)，并解压到项目目录下的./datafolder/VideoGen-RewardBench文件夹中


---
<a id="jump4"></a>
## 微调

<a id="jump4.1"></a>
#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

微调中数据集csv文件应包含应包含如下数据：

   | path_A | path_B | prompt | VQ | MQ | TA | fps_A | num_frames_A | fps_B | num_frames_B |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |



<a id="jump4.2"></a>
#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`model_name_or_path`、`data_folder`、`data_path`, `cache_dir`等字段。注意`model_name_or_path`的权重路径为转换前的权重路径。

**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

```json
{
    "dataset_param": {
        "dataset_type": "rewardvideo",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2-VL-2B-Instruct/",
            ...
        },
        "basic_parameters": {
            ...
            "data_folder": "./datafolder/VideoGen-RewardBench/",
            "data_path": "./datafolder/VideoGen-RewardBench/videogen-rewardbench.csv",
            "cache_dir": "./cache_dir",
            ...
        },
        ...
    },
    ...
}
```

【模型目录配置】
根据实际情况修改`model.json`中的训练配置，包括:

| 参数 | 含义及用法 |
| ---  |  ---|
|`output_dim`| 模型输出维度 |
|`reward_token`| reward_token的计算方式 |
|`loss_type`| reward loss的计算类型 |
|`lora_mixed_training`| 是否开启lora参数和非lora参数混合训练（例如ViT采用全参微调，text-decoder采用lora微调）|
|`lora_apply_modules`| 混合训练时采用lora微调的模块 |
|`lora_save_full_weight`| 混合训练中同时保持lora权重和非lora权重 |
|`freeze`| 冻结该模块参数，不参与训练 |


```json
{
    "model_id": "qwen2vl",
    "img_context_token_id": 151656,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "output_dim": 1,
    "reward_token": "special",
    "loss_type": "btt",
    "lora_apply_modules": ["text_decoder"],
    "lora_mixed_training": true,
    "lora_save_full_weight": true,
    ...
```
```json
"image_encoder": {
        "vision_encoder": {
            ...
            "freeze": true,
            ...
```

注：默认配置image-encoder全参训练，text-decoder采用lora微调的混合训练方式，否则容易出现显存不足

【vit模块重计算配置（可选）】

当放开vit训练时（默认配置中冻结vit，若要放开请将model.json文件中`vision_encoder`部分配置为`"vision_encoder": {"freeze": false}`。），可以启用重计算以降低显存（注意，此举会对性能产生影响）

若要开启vit重计算，需在model.json中的vision_encoder部分添加重计算相关参数。
通过`recompute_granularity`参数可以配置重计算模块为`full`或`selective`。

1. full模式

TransformerLayer中的所有组件（layernorm、attention、mlp）都进行重计算，此时可以配置重计算的层数。
- `recompute_method`: 控制重计算层数计算的方法，可选值为`uniform`（均匀重计算）或`block`（按块重计算）。
- `recompute_num_layers`: 控制重计算的层数，指定需要重计算的层数量。

示例配置如下：

```json
{
  "model_id": "videoalign",
  "img_context_token_id": 151656,
  "video_token_id": 151656,
  "vision_start_token_id": 151652,
  ...
  "image_encoder": {
    "vision_encoder": {
      ...
      "recompute_granularity": "full",
      "recompute_method": "uniform",
      "recompute_num_layers": 1
    }
  },
  ...
}
```
2. selective模式

仅对TransformerLayer中attention的core_attention组件进行重计算。注意：lora场景无法使用。

示例配置如下：

```json
{
  "model_id": "videoalign",
  "img_context_token_id": 151656,
  "video_token_id": 151656,
  "vision_start_token_id": 151652,
  ...
  "image_encoder": {
    "vision_encoder": {
      ...
      "recompute_granularity": "selective"
    }
  },
  ...
}
```

【模型保存加载及日志信息配置】

根据实际情况配置`examples/videoalign/finetune_lora.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 加载路径
LOAD_PATH="ckpt/mm_path/Qwen2-VL-2B-Instruct"
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
(此功能coming soon)


```
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

若开启lora混合训练，保存权重包含lora权重和非lora权重，需要通过转换脚本进行拆分后分别加载。(此功能coming soon)

【单机运行配置】

配置`examples/videoalign/finetune_lora.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/cann/set_env.sh
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

注：训练默认开启FA，否则可能会显存不足报错，通过开关--use-flash-attn 控制。

当开启PP时，model.json中配置的vision_encoder和text_decoder的pipeline_num_layer参数控制了各自的PP切分策略。对于流水线并行，要先处理vision_encoder再处理text_decoder。比如[32,0,0,0]、[1,10,10,7]，其含义为PP域内第一张卡先放32层vision_encoder再放1层text_decoder、第二张卡放text_decoder接着的10层、第三张卡放text_decoder接着的10层、第四张卡放text_decoder接着的7层，vision_encoder没有放完时不能先放text_decoder（比如[30,2,0,0]、[1,10,10,7]的配置是错的）。

如果某张卡上的参数全部冻结时会导致没有梯度（比如vision_encoder冻结时PP配置[30,2,0,0]、[0,11,10,7]），需要在finetune_qwen2vl_7b.sh中GPT_ARGS参数中增加--enable-dummy-optimizer，参考[dummy_optimizer特性文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/features/dummy_optimizer.md)。


<a id="jump4.3"></a>
#### 3. 启动微调

通过以下命令启动微调训练任务。

```shell
bash examples/videoalign/finetune_lora.sh
```

---
<a id="jump5"></a>
## 推理

<a id="jump5.1"></a>
#### 1、准备工作（以微调环境为基础，包括环境安装、权重下载及转换）

推理数据csv应包含如下数据：

   | path_A | path_B | prompt | fps_A | num_frames_A | fps_B | num_frames_B |
   | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
   | ... | ... | ... | ... | ... | ... | ... |



<a id="jump5.2"></a>
#### 2、配置参数
【数据目录配置】  
根据实际情况修改examples/videoalign/data.json中的数据路径等参数：

|   参数   |   配置方法   |
| ---- | ---- |
| `model_name_or_path` | 配置与微调保持中一致            
|`data_folder`| 推理数据集所在文件夹路径 |  
|`data_path`| 推理数据集csv路径 |
|`save_path`| 推理结果xlsx保存路径 |
|`task`| 任务类型，选择`inference`|


```json
"inference_param": {
        "data_folder": "./datafolder/VideoGen-RewardBench/",
        "data_path": "./datafolder/VideoGen-RewardBench/videogen-rewardbench.csv",
        "save_path": "./inference_result/",
        "task": "inference",
        ...
```

<a id="jump5.3"></a>
#### 3、启动推理

```shell
bash examples/videoalign/inference.sh
```
推理结果会输出到`inference_result`路径中，会输出结果文件：
- reward_out_single.xlsx文件，这个文件会输出每条视频的评测分数。


<a id="jump6"></a>
### 评测

<a id="jump6.1"></a>
#### 1、准备工作（以微调环境为基础，包括环境安装、权重下载及转换）

评测数据csv应包含如下数据：

| path_A | path_B | prompt | VQ | MQ | TA | Overall | fps_A | num_frames_A | fps_B | num_frames_B |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | --- |



<a id="jump6.2"></a>
#### 2、配置参数
【数据目录配置】
根据实际情况修改examples/videoalign/data.json中的数据路径等参数：

|   参数   |   配置方法   |
| ---- | ---- |
| `model_name_or_path` | 配置与微调保持中一致            
|`data_folder`| 评测数据集所在文件夹路径 |
|`data_path`| 评测数据集csv路径 |
|`save_path`| 评测结果xlsx保存路径 |
|`task`| 任务类型，选择`evaluate` |
|`reward_attributes`| 评测指标 |
|`use_norm`| 评测分数是否归一化 |
|`norm_param`| 归一化参数 |


```json
"inference_param": {
        "data_folder": "./datafolder/VideoGen-RewardBench/",
        "data_path": "./datafolder/VideoGen-RewardBench/videogen-rewardbench.csv",
        "save_path": "./evaluate_result/",
        "task": "evaluate",
        "reward_attributes": ["VQ", "MQ", "TA", "Overall"],
        "use_norm": false,
        "norm_param": {
            "VQ_mean": 3.6757,
            "VQ_std": 2.2476,
            "MQ_mean": 1.1646,
            "MQ_std": 1.3811,
            "TA_mean": 2.8105,
            "TA_std": 2.5121
        }
        ...
```

<a id="jump6.3"></a>
#### 3、启动评测

```shell
bash examples/videoalign/inference.sh
```

评测结果会输出到`evaluate_result`路径中，会输出结果文件：

- reward_out_single.xlsx文件，这个文件会输出每条视频的评测分数。
- reward_out_pair.xlsx文件，这个文件会输出每对样本的评测分数及胜负情况。
- eval_accuracy.json文件，这个文件会输出各评测指标统计准确率等数据。

---
<a id="jump7"></a>
## 特性使用介绍

<a id="jump7.1"></a>
### lora微调
LoRA为框架通用能力，当前功能已支持，可参考[LoRA特性文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/features/lora_finetune.md)。

<a id="jump8"></a>
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
<a id="jump9"></a>
## 注意事项

1. 在 `finetune_lora.sh`里，与模型结构相关的参数并不生效，以`examples/videoalign/model.json`里同名参数配置为准，非模型结构的训练相关参数在 `finetune_lora.sh`修改。

