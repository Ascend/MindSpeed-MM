# LLaVA1.5 使用指南

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
- [预训练](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动预训练](#jump4.3)
- [推理](#jump5)
  - [准备工作](#jump5.1)
  - [配置参数](#jump5.2)
  - [启动推理](#jump5.3)
- [评测](#jump6)
  - [数据集准备](#jump6.1)
  - [配置参数](#jump6.2)
  - [启动评测](#jump6.3)
- [环境变量声明](#jump7)

## 版本说明

#### 参考实现

```
url=https://github.com/haotian-liu/LLaVA.git
commit_id=3e337ad
```

#### 变更记录

2024.10.15: 首次支持LLaVA1.5模型

---
<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

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
    mkdir logs
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

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

- [ViT-L-14-336px](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt)：CLIPViT模型；

- [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/)： GPT模型；

<a id="jump2.2"></a>

#### 2. 权重转换（当前依赖openai-clip库，正在规划重构）

MindSpeed-MM修改了部分原始网络的结构名称，因此需要使用如下脚本代码对下载的预训练权重进行转换。 当前训练只使用了ViT-L-14-336px和lmsys/vicuna-7b-v1.5两个模型，以下介绍这两个模型从开源仓转换成MindSpeed-MM所需权重的方法：

- ViT-L-14-336px权重转换

  脚本参考 NVIDIA/Megatron-LM中[Vision model](https://github.com/NVIDIA/Megatron-LM/blob/core_r0.8.0/examples/multimodal/README.md#vision-model) ,将[ViT-L-14-336px](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt)权重下载到本地后，
  执行如下命令：

  ```bash
    # 安装依赖（加载原始权重需要依赖openai-clip库）
    pip install git+https://github.com/openai/CLIP.git

    # llava模型构建时会将clipvit的最后一层剔除，如在其他场景需要转换完整vit权重，请增加--no-drop-last-layer传参
    python examples/llava1.5/clip_converter.py \
      --download-root {dir_to_model}/ViT-L-14-336px.pt \
      --output {target_dir} \
  ```

  其中{dir_to_model}为下载模型权重所在的路径，转换后权重将保存在{target_dir}/converted_clip.pt。

- lmsys/vicuna-7b-v1.5权重转换

  下载权重后执行如下命令：

  ```shell
  python examples/llava1.5/vicuna_converter.py \
    --load-dir {dir_to_model}/vicuna-7b-v1.5 \
    --save-dir {target_dir} \
    --trust-remote-code True # 为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性
  ```

  其中{dir_to_model}为下载模型权重所在的路径，转换后权重将保存在{target_dir}/converted_vicuna.pt。
  
<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

用户需自行获取并解压image.zip得到[LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)数据集，获取数据结构如下：

   ```
   $LLaVA-Pretrain
   ├── blip_laion_cc_sub_558k.json
   ├── blip_laion_cc_sub_558k_meta.json
   ├── images
   ├── ├── 00000
   ├── ├── 00001
   ├── ├── 00002
   └── └── ...
   ```

---

<a id="jump4"></a>

## 预训练

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

【数据目录配置】

需根据实际情况修改`data.json`中的权重和数据集路径，包括`from_pretrained`、`data_path`、`data_folder`字段。

注意`tokenizer_config`的权重路径为转换前的权重路径。

```json
{
  "dataset_param": {
      ...
      "basic_parameters": {
          "data_path": "LLaVA-Pretrain/ai2d_train_12k.jsonl",
          "data_folder": "LLaVA-Pretrain/images"
      },
      ...
      "tokenizer_config": {
          ...
          "from_pretrained": "{dir_to_raw_model}/vicuna-7b-v1.5",
          ...
      },
      ...
  },
  ...
}
```

根据实际情况修改`model.json`中的权重路径为转换后权重，无需预训练权重则传入null。

```json
{
    ...
    "text_decoder": {
      ...
      "ckpt_path": "/<your_vicuna_weights_path>/converted_vicuna.pt"
    },
    "image_encoder": {
      "vision_encoder":{
        ...
        "ckpt_path": "/<your_clip_weights_path>/converted_clip.pt"
      },
      "vision_projector":{
        ...
        "ckpt_path": null
      }
    }
}
```

【模型保存加载配置】

根据实际情况配置`examples/llava1.5/pretrain_llava1_5.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
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
"
```

若需要加载指定迭代次数的权重、优化器等状态，需修改`examples/llava1.5/pretrain_llava1_5.sh`，并将加载路径`LOAD_PATH`设置为保存文件夹路径`LOAD_PATH="save_dir"`

```shell
...
LOAD_PATH="save_dir"
...
OUTPUT_ARGS="
    ...
    --load $LOAD_PATH
"
```

并修改`latest_checkpointed_iteration.txt`文件内容为指定迭代次数

```
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

【单机运行】

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

<a id="jump4.3"></a>

#### 3. 启动预训练

```shell
    bash examples/llava1.5/pretrain_llava1_5.sh
```

<a id="jump5"></a>

## 推理

<a id="jump5.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：环境安装、权重下载及转换，详情可查看对应章节。

推理任务除了需要上述提到的converted_vicuna.pt权重、converted_clip.pt权重，以及原始的vicuna-7b-v1.5权重外，还需要[clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)权重和vision_projector的权重，vision_projector权重需要从此[链接](https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main/mm_projector.bin?download=true)下载。
vision_projector下载后需要做权重转换。

```python
import torch
def convert_mlp(ckpt_path):
    # ckpt_path 为原始权重
    target_mlp = {}
    mlp = torch.load(ckpt_path)
    target_mlp["encoder.linear_fc1.weight"] = mlp["model.mm_projector.0.weight"]
    target_mlp["encoder.linear_fc1.bias"] = mlp["model.mm_projector.0.bias"]
    target_mlp["encoder.linear_fc2.weight"] = mlp["model.mm_projector.2.weight"]
    target_mlp["encoder.linear_fc2.bias"] = mlp["model.mm_projector.2.bias"]
    torch.save(target_mlp,"./mlp.pt")

```

<a id="jump5.2"></a>

#### 2. 配置参数

将准备好的权重传入到inference_llava.json中，根据json中路径的提示更改其中的路径，包括from_pretrained、ckpt_path等，自定义的prompt可以传入到prompt字段中。

<a id="jump5.3"></a>

#### 3. 启动推理

启动推理脚本

```shell
bash examples/llava1.5/inference_llava1_5.sh
```

---

<a id="jump6"></a>

## 评测

<a id="jump6.1"></a>

### 数据集准备

当前模型支持AI2D(test)、ChartQA(test)、Docvqa(val)、MMMU(val)四种数据集的评测。
数据集参考下载链接：

- [MMMU_DEV_VAL](https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv)
- [DocVQA_VAL](https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv)
- [AI2D_TEST](https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv)
- [ChartQA_TEST](https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv)
<a id="jump6.2"></a>

### 参数配置

如果要进行评测需要将要评测的数据集名称和路径传到examples/llava1.5/evaluate_llava1_5.json
需要更改的字段有

- `text_decoder`中的`ckpt_path`为前面的权重转换章节中lmsys/vicuna-7b-v1.5权重转换脚本后的权重
- `vision_encoder`中的`ckpt_path`为前面权重转换章节中ViT-L-14-336px权重转换后的权重
- `vision_projector`中的`ckpt_path`为推理章节中vision_projector权重转换后的权重
- `tokenizer`中的`from_pretrained`为huggingface的[llava权重路径](https://huggingface.co/liuhaotian/llava-v1.5-7b)，自行下载传入
- `dataset_path`为上述评测数据集的本地路径
- `evaluation_dataset`为评测数据集的名称可选的名称有(`ai2d_test`、`mmmu_dev_val`、`docvqa_val`、`chartqa_test`)， **注意**：需要与上面的数据集路径相对应。
- `result_output_path`为评测结果的输出路径，**注意**：每次评测前需要将之前保存在该路径下评测文件删除。
- `image_processer_path`下载链接为[clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)，自行下载传入

```json
  "text_decoder": {
    "ckpt_path": "/<your_vicuna_weights_path>/converted_vicuna.pt"
  },
  "vision_encoder": {
    "ckpt_path": "/<your_clip_weights_path>/converted_clip.pt"
  }
  "vision_projector": {
    "ckpt_path": "/<your_clip_weights_path>/converted_mlp.pt"
  }
   "tokenizer": {
    "from_pretrained": "./llava_7b",
  }, 
           
  "dataset_path": "./AI2D_TEST.tsv",
  "evaluation_dataset": "ai2d_test",
  "evaluation_model": "llava_v1.5_7b",
  "result_output_path": "./evaluation_outputs/",
  "image_processer_path": "./llava_weights_mm/clip-vit-large-patch14-336"

```

examples/llava1.5/evaluate_llava1_5.json改完后，需要将json文件的路径传入到examples/llava1.5/evaluate_llava1_5.sh MM_MODEL字段中

```shell
MM_MODEL=examples/llava1.5/evaluate_llava1_5.json
```

评测支持多卡DP评测需要更改的配置,为NPU卡数量

```shell
NPUS_PER_NODE=1
```

<a id="jump6.3"></a>

### 启动评测

评测额外依赖一些python包，使用下面命令进行安装

```shell
pip install -e ".[evaluate]"
```

启动shell开始评测

```shell
bash examples/llava1.5/evaluate_llava1_5.sh
```

评测结果会输出到`result_output_path`路径中，会输出结果文件：

- *.xlsx文件，这个文件会输出每道题的预测结果和答案等详细信息。
- *.csv文件，这个文件会输出统计准确率等数据。
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

