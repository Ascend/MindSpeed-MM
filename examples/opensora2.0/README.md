# OpenSora2.0 使用指南

<p align="left">
</p>

## 目录

- [OpenSora2.0 使用指南](#opensora20-使用指南)
  - [目录](#目录)
  - [版本说明](#版本说明)
      - [参考实现](#参考实现)
      - [变更记录](#变更记录)
  - [环境安装](#环境安装)
      - [1. 仓库拉取](#1-仓库拉取)
      - [2. 环境搭建](#2-环境搭建)
  - [权重下载及转换](#权重下载及转换)
      - [1. 权重下载](#1-权重下载)
      - [2. 权重转换](#2-权重转换)
  - [数据集准备及处理](#数据集准备及处理)
  - [预训练](#预训练)
      - [1. 准备工作](#1-准备工作)
      - [2. 配置参数](#2-配置参数)
      - [3. 启动预训练](#3-启动预训练)
  - [环境变量声明](#环境变量声明)

## 版本说明
#### 参考实现
```
url=https://github.com/hpcaitech/Open-sora.git
commit_id=d0cd5ac
```

#### 变更记录

2025.06.25: 首次支持Open-sora 2.0 T2V

---

<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/install_guide.md)

<a id="jump1.1"></a>

#### 1. 仓库拉取

```shell
git clone --branch 2.2.0 https://gitcode.com/Ascend/MindSpeed-MM.git 
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

# 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch及torch_npu包
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl



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

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface网站下载开源模型权重

- [OpenSoraV2模型](https://huggingface.co/hpcai-tech/Open-Sora-v2/blob/main/Open_Sora_v2.safetensors)
- [vae模型](https://huggingface.co/hpcai-tech/Open-Sora-v2/blob/main/hunyuan_vae.safetensors)
- [T5模型](https://huggingface.co/hpcai-tech/Open-Sora-v2/tree/main/google)
- [Clip模型](https://huggingface.co/hpcai-tech/Open-Sora-v2/tree/main/openai)

<a id="jump2.2"></a>

#### 2. 权重转换

需要对[OpenSoraV2模型]模型进行权重转换，运行权重转换脚本：
```shell
mm-convert OpenSoraConverter hf_to_mm \
  --cfg.source_path <OpenSoraV2模型> \
  --cfg.target_path <OpenSoraV2模型转化后路径>
```


---

<a id="jump3"></a>

## 数据集准备及处理

用户需自行准备训练数据集，需要提供对应的切片视频集合datasets和csv文件，csv文件命名为train_data.csv，作为模型输入的data_path。

数据集数据结构如下：

   ```
   train_data.csv
   datasets
   ├── video1990_scene-4.mp4
   ├── video1990_scene-5.mp4
   ├── video1991_scene-1.mp4
   ...
   ```

csv文件内容格式如下：

   ```
   path,text,num_frames,height,width,aspect_ratio,resolution,fps
   ./datasets/pexels_45k/popular_3/853857_scene-0_cut-border.mp4,"an aerial view of a large...",330.0,1036.0,1102.0,0.94010889292196,1141672.0,30.0
   ```
   注意: csv文件的path字段需要填充切片视频的相对路径或绝对路径，如果是相对路径需要在data.json文件中的`data_folder`字段补充父路径

---

<a id="jump4"></a>

## 预训练

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节

<a id="jump4.2"></a>

#### 2. 配置参数

默认的配置已经经过测试，用户可按照自身环境修改如下内容：

| 配置文件                                                   |      修改字段       | 修改说明                                            |
| -------------------------------------------------------- | :-----------------: | :-------------------------------------------------- |
| examples/opensora2.0/data.json                           |  basic_parameters   | `data_path`提供数据集csv文件路径，`data_folder`为数据集切片视频路径前缀(非必填) |
| examples/opensora2.0/pretrain_model.json           |  text_encoder  | 配置两种text encoder路径`"from_pretrained": "Open-Sora-v2/google/t5-v1_1-xxl"`及`"from_pretrained": "Open-Sora-v2/openai/clip-vit-large-patch14"` |
| examples/opensora2.0/pretrain_model.json           |       ae       | 配置VAE模型路径`"from_pretrained": "Open-Sora-v2/hunyuan_vae.safetensors"`       |
| examples/opensora2.0/pretrain_opensora2_0.sh       |    NPUS_PER_NODE    | 每个节点的卡数                                      |
| examples/opensora2.0/pretrain_opensora2_0.sh       |       NNODES        | 节点数量                                            |
| examples/opensora2.0/pretrain_opensora2_0.sh       |      LOAD_PATH      | 权重转换后的预训练权重路径                          |
| examples/opensora2.0/pretrain_opensora2_0.sh       |      SAVE_PATH      | 训练过程中保存的权重路径                            |

【数据集桶配置参数说明】：

bucket_config（dict）：一个包含bucket配置的字典。

词典应采用以下格式：
```json
"bucket_config": {
    "256px": {"1": [1.0, 3], "125": [1.0, 2], "129": [1.0, 1]},
    "720p": {"100": [0.5, 1]}
}
```

案例解释:

`256px`表示256*256像素的视频

`720p`表示宽高比为16:9且其中高度为720像素的视频

`{"100": [0.5, 1]}` 其中100为视频帧数, `0.5`为视频采用概率(介于0和1之间的浮点数), `1`为当前视频规格的batch_size

【并行化配置参数说明】：

由于OpenSora2.0模型参数规模较大，单机无法跑下完整模型，故默认配置已整合`layer_zero`优化

+ layer_zero使用介绍

  - 使用场景：在模型参数规模较大时，单卡上无法承载完整的模型，可以通过开启layerzero降低静态内存。
  
  - 使能方式：`examples/opensora2.0/pretrain_opensora2_0.sh`的`GPT_ARGS`中加入`--layerzero`和`--layerzero-config $LAYERZERO_CONFIG`

  - 使用建议: 配置文件`examples/opensora2.0/zero_config.yaml`中的`zero3_size`推荐设置为单机的卡数
  
  - 训练权重后处理：使用该特性训练时，保存的权重需要使用下面的转换脚本进行后处理才能用于推理：
  
  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  # your_mindspeed_path和your_megatron_path分别替换为之前下载的mindspeed和megatron的路径
  export PYTHONPATH=$PYTHONPATH:<your_mindspeed_path>
  export PYTHONPATH=$PYTHONPATH:<your_megatron_path>
  # input_folder为layerzero训练保存权重的路径，output_folder为输出的megatron格式权重的路径
  mm-convert OpenSoraConverter layerzero_to_mm \
      --cfg.source_path <./save_ckpt/opensora2/> \
      --cfg.target_path <./save_ckpt/opensora2_megatron_ckpt/>
  ```


<a id="jump4.3"></a>

#### 3. 启动预训练

```shell
bash examples/opensora2.0/pretrain_opensora2_0.sh
```

---
<a id="jump5"></a>
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
