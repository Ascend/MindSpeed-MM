# Wan2.2 使用指南

- [Wan2.2 使用指南](#wan22-使用指南)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
    - [变更记录](#变更记录)
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
    - [训练](#训练)
      - [准备工作](#准备工作)
      - [参数配置](#参数配置)
      - [启动训练](#启动训练)
  - [推理](#推理)
    - [准备工作](#准备工作-1)
    - [参数配置](#参数配置-1)
    - [启动推理](#启动推理)
  - [环境变量声明](#环境变量声明)

## 版本说明

#### 参考实现

【预训练任务】

5B:

```
url=https://github.com/modelscope/DiffSynth-Studio.git
commit_id=f0ea049
```

A14B:

```
url=https://github.com/modelscope/DiffSynth-Studio.git
commit_id=833ba1e
```

【推理】

```
url=https://github.com/huggingface/diffusers/tree/v0.35.1
```

#### 变更记录

2025.10.11: 首次支持Wan2.2模型

## 任务支持列表

| 模型大小 | 任务类型 | 预训练 | 在线T2V推理 | 在线I2V推理 |
|------|:----:|:----|:-----|:-----|
| 5B | t2v  | ✔ | ✔ |  |
| 5B | ti2v  | ✔ |  | ✔ |
| A14B  | t2v  | ✔ | ✔ |  |
| A14B  | i2v  | ✔ |  | ✔ |

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

### 仓库拉取

```shell
git clone --branch 2.3.0 https://gitcode.com/Ascend/MindSpeed-MM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ../MindSpeed-MM
```

### 环境搭建

```bash
# 安装对应python版本、x86或arm的torch、torch_npu
# 参考 安装指南 -> PTA安装

# 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 安装加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout 93c45456c7044bacddebc5072316c01006c938f9
pip install -e .
cd ..

# 安装其余依赖库
pip install -e .

# 更新diffusers、peft
pip install diffusers==0.35.1 peft==0.17.1
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

|   模型   |   Huggingface下载链接   |
| ---- | ---- |
|   5B   |   <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers>   |
|  T2V-14B    |  <https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers>    |
|  I2V-14B  |   <https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers>   |

### 权重转换

需要对下载后的Wan2.2模型权重`transformer`部分进行权重转换，运行权重转换脚本：

```shell
mm-convert WanConverter hf_to_mm \
 --cfg.source_path ./weights/Wan-AI/Wan2.2-{TI2V/T2V/I2V}-{5/A14}B-Diffusers/transformer* \
 --cfg.target_path ./weights/Wan-AI/Wan2.2-{TI2V/T2V/I2V}-{5/A14}B-Diffusers/transformer*
```

权重转换脚本的参数说明如下：

| 参数              | 含义                     |
| :---------------- | :----------------------- |
| --cfg.source_path | 原始权重路径             |
| --cfg.target_path | 转换或切分后权重保存路径 |

如需转回Hugging Face格式，需运行权重转换脚本：

**注**： wan2.2使用fsdp2进行训练，需首先进行其[训练权重后处理](#jump1)，再进行如下操作：

```shell
mm-convert WanConverter mm_to_hf \
 --cfg.source_path path_for_your_saved_weight \
 --cfg.target_path ./converted_weights/Wan-AI/Wan2.2-{TI2V/T2V/I2V}-{5/A14}B-Diffusers/transformer* \
 --cfg.hf_path weights/Wan-AI/Wan2.2-{TI2V/T2V/I2V}-{5/A14}B-Diffusers/transformer*
```

权重转换脚本的参数说明如下：

| 参数                | 含义                                            |
|:------------------|:----------------------------------------------|
| --cfg.source_path | MindSpeed MM保存的权重路径                           |
| --cfg.target_path | 转换后的Hugging Face权重路径                          |
| --cfg.hf_path     | 原始Hugging Face权重路径，需要从该目录下获取原始huggingface配置文件 |

**注**： 对A14B模型，hugging face diffusers权重中包含两个transformer权重，
后缀中transformer对应高噪声（high）模型，transformer_2对应低噪声（low）模型。

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

修改`examples/wan2.2/data.txt`文件，其中每一行表示一个数据集，第一个参数表示数据文件夹的路径，第二个参数表示`data.json`文件的路径，用`,`分隔

### 训练

#### 准备工作

在开始之前，请确认环境准备、模型权重下载、数据预处理已完成。

#### 参数配置

检查数据集路径、模型权重路径、并行参数配置等是否完成

| 配置文件   |      修改字段       | 修改说明      |
| --- | :---: | :--- |
| examples/wan2.2/{model_size}/{task}/data.json            |  from_pretrained  | 修改为下载的tokenizer的权重所对应的路径 |
| examples/wan2.2/{model_size}/{task}/pretrain_model*.json |  from_pretrained  | 修改为下载的权重所对应路径（包括vae,  text_encoder） |
| examples/wan2.2/{model_size}/{task}/pretrain*.sh         |    NPUS_PER_NODE  | 每个节点的卡数                                     |
| examples/wan2.2/{model_size}/{task}/pretrain*.sh         |       NNODES      | 节点数量                                          |
| examples/wan2.2/{model_size}/{task}/pretrain*.sh         |      LOAD_PATH    | 权重转换后的预训练权重路径                          |
| examples/wan2.2/{model_size}/{task}/pretrain*.sh         |      SAVE_PATH    | 训练过程中保存的权重路径                            |
| examples/wan2.2/{model_size}/{task}/pretrain*.sh         |        CP         | 训练时的CP size（建议根据训练时设定的分辨率调整）   |

**注**： 当前LOAD_PATH路径无效时，MindSpeed会对模型随机初始化从头训练。为防止加载失败，请留意日志中的warning信息，或者自行确认路径合法。

【并行化配置参数说明】：

- CP: 序列并行。

  - 使用场景：在视频序列（分辨率X帧数）较大时，可以开启来降低内存占用。
  
  - 使能方式：在启动脚本中设置 CP > 1，如：CP=2；
  
  - 限制条件：head 数量需要能够被CP整除（在`examples/wan2.2/{model_size}/{task}/pretrain_model*.json`中配置，参数为`num_heads`）

  - 默认使能方式为Ulysses序列并行。

  - DiT-RingAttention：DiT RingAttention序列并行请[参考文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/dit_ring_attention.md)

  - DiT-USP: DiT USP混合序列并行（Ulysses + RingAttention）请[参考文档](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/dit_usp.md)

- fsdp2

  - 使用场景：在模型参数规模较大时，可以通过开启fsdp2降低静态内存。
  
  - 使能方式：`examples/wan2.2/{model_size}/{task}/pretrain.sh`的`GPT_ARGS`中加入`--use-torch-fsdp2`，`--fsdp2-config-path ${fsdp2_config}`，`--untie-embeddings-and-output-weights`以及`--ckpt-format torch_dcp`，其中fsdp2_config配置请参考：[FSDP2说明](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/fsdp2.md)
  <a id="jump1"></a>
  - 训练权重后处理：使用该特性训练时，保存的权重需要使用下面的转换脚本进行后处理才能用于推理：

    ```bash
    # 训练结束后保存的权重路径
    save_path="./wandit_weight_save"
    iter_dir="$save_path/iter_$(printf "%07d" $(cat $save_path/latest_checkpointed_iteration.txt))"
    # 权重转换的目标路径
    convert_dir="./dcp_to_torch"
    mkdir -p $convert_dir/release/mp_rank_00
    cp $save_path/latest_checkpointed_iteration.txt $convert_dir/
    echo "release" > $convert_dir/latest_checkpointed_iteration.txt
    python -m torch.distributed.checkpoint.format_utils dcp_to_torch "$iter_dir" "$convert_dir/release/mp_rank_00/model_optim_rng.pt"
    ```

#### 启动训练

【5B】

```bash
bash examples/wan2.2/{model_size}/{task}/pretrain.sh
```

【A14B】

```bash
bash examples/wan2.2/{model_size}/{task}/pretrain_{type}.sh
```


## 推理

### 准备工作

在开始之前，请确认环境准备、模型权重下载已完成

### 参数配置

检查模型权重路径、并行参数等配置是否完成

| 配置文件                                                     |        修改字段         | 修改说明                                        |
|----------------------------------------------------------|:-------------------:|:--------------------------------------------|
| examples/wan2.2/{model_size}/{task}/inference_model.json |   from_pretrained   | 修改为下载的权重所对应路径（包括vae、tokenizer、text_encoder） |
| examples/wan2.2/samples_t2v_prompts.txt                  |        文件内容         | T2V推理任务的prompt，可自定义，一行为一个prompt             |
| examples/wan2.2/samples_i2v_prompts.txt                  |        文件内容         | I2V推理任务的prompt，可自定义，一行为一个prompt             |
| examples/wan2.2/samples_i2v_images.txt                   |        文件内容         | I2V推理任务的首帧图片路径，可自定义，一行为一个图片路径               |
| examples/wan2.2/{model_size}/{task}/inference_model.json |      save_path      | 生成视频的保存路径                                   |
| examples/wan2.2/{model_size}/{task}/inference_model.json |     input_size      | 生成视频的分辨率，格式为 [t, h, w]                      |
| examples/wan2.2/{model_size}/{task}/inference_model.json | low_noise_predictor | 转换之后的transformer_2（低噪声）部分权重路径，仅A14B模型涉及     |
| examples/wan2.2/{model_size}/{task}/inference.sh         |      LOAD_PATH      | 转换之后的transformer部分权重路径，对A14B模型对应高噪声模型       |

### 启动推理

```shell
bash examples/wan2.2/{model_size}/{task}/inference.sh
```

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

