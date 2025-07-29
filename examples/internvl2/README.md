# InternVL2 使用指南

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
- [评测](#jump6)
  - [数据集准备](#jump6.1)
  - [配置参数](#jump6.2)
  - [启动评测](#jump6.3)
- [特性使用介绍](#jump7)
  - [DistTrain](#jump7.1)
- [环境变量声明](#jump8)
- [注意事项](#jump9)


## 版本说明
#### 参考实现
```
url=https://github.com/OpenGVLab/InternVL.git
commit_id=c62fa4f
```

#### 变更记录

2024.10.21: 首次发布InternVL2模型

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
pip install torch-2.6.0-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.6.0*-cp310-cp310-manylinux_2_28_aarch64.whl

# apex for Ascend 参考 https://gitee.com/ascend/apex
# 建议从原仓编译安装

# 安装加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.8.0
git checkout 077c41370332460fbf749ea94fe1a994f367f4bf
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

- [InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B/tree/main)；
- [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main)；
- [InternVL2-26B](https://huggingface.co/OpenGVLab/InternVL2-26B/tree/main)；
- [InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B/tree/main)；

将模型权重保存在`raw_ckpt`目录下，例如`raw_ckpt/InternVL2-8B`。

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的转换以及PP（Pipeline Parallel）和VPP（Virtual Pipeline Parallel）的权重切分(详细VPP配置参考[vpp特性说明](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/virtual_pipeline_parallel.md))。

`mm-convert`工具详细用法参考[权重转换工具](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)

以InternVL2-8B为例，使用命令如下


```bash
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 2B
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-2B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-2B" \
  --cfg.parallel_config.llm_pp_layers [[24]] \
  --cfg.parallel_config.vit_pp_layers [[24]] \
  --cfg.trust_remote_code True

# 8B
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-8B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-8B" \
  --cfg.parallel_config.llm_pp_layers [[6,9,9,8]] \
  --cfg.parallel_config.vit_pp_layers [[24,0,0,0]] \
  --cfg.trust_remote_code True

# 8B VPP
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-8B-vpp" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-8B" \
  --cfg.parallel_config.llm_pp_layers [[0,0,0,1],[4,4,4,4],[4,4,4,3]] \
  --cfg.parallel_config.vit_pp_layers [[6,7,7,4],[0,0,0,0],[0,0,0,0]] \
  --cfg.trust_remote_code True

# 76B
mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "pretrained/InternVL2-Llama3-76B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-Llama3-76B" \
  --cfg.parallel_config.llm_pp_layers [[0,0,0,1,5,6,7,7,7,7,7,7,7,7,6,6]] \
  --cfg.parallel_config.vit_pp_layers [[11,12,12,10,0,0,0,0,0,0,0,0,0,0,0,0]] \
  --cfg.trust_remote_code True

# 其中：
# mm_dir: 转换后保存目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# trust_remote_code: 为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性
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

【视频数据】

若要使用视频进行训练，用户可参考[视频数据集构造](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data)自行构造视频数据集。

---

<a id="jump4"></a>

## 微调

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`from_pretrained`、`data_path`、`data_folder`等字段。

以InternVL2-8B为例，`data_8B.json`进行以下修改，注意`tokenizer_config`的权重路径为转换前的权重路径。

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
          "from_pretrained": "raw_ckpt/InternVL2-8B",
          ...
      },
      ...
  },
  ...
}
```

【模型保存加载及日志信息配置】

根据实际情况配置`examples/internvl2/finetune_internvl2_xx.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）, 以InternVL2-8B为例：

```shell
...
# 加载路径
LOAD_PATH="ckpt/InternVL2-8B"
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

配置`examples/internvl2/finetune_internvl2_xx.sh`参数如下

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

以InternVL2-8B为例，启动微调训练任务。

```shell
bash examples/internvl2/finetune_internvl2_8B.sh
```

<a id="jump5"></a>

## 推理

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：环境安装、权重下载及转换，详情可查看对应章节。（当前仅支持2B和8B单卡推理）

推理权重转换命令如下：

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

mm-convert  InternVLConverter hf_to_mm \
  --cfg.mm_dir "raw_ckpt/InternVL2-8B" \
  --cfg.hf_config.hf_dir "pretrained/InternVL2-8B" \
  --cfg.parallel_config.llm_pp_layers [[32]] \
  --cfg.parallel_config.vit_pp_layers [[24]] \
  --cfg.trust_remote_code True
```

#### 2. 配置参数

【参数配置】

修改inference_xx.json文件，包括`infer_data_type`、`file_path`、`prompts`、`from_pretrained`以及tokenizer的`from_pretrained`等字段。

【单图推理】

以InternVL2-8B为例，按实际情况修改inference_8B.json对应参数，注意tokenizer_config的权重路径为转换前的权重路径。

```json
{
    "infer_data_type": "image",
    "file_path": "./examples/internvl2/view.jpg",    # 按实际情况输入图片路径
    "prompts": "Please describe the image shortly.", # 按实际情况输入提示词
    "model_id": "InternVLPipeline",
    "from_pretrained": "./pretrained/InternVL2-8B/release/mp_rank_00/model_optim_rng.pt", # 注意路径要到.pt文件
    ...
    "tokenizer":{
        ...
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "raw_ckpt/InternVL2-8B",
        ...
    },
    ...
}
```

【视频推理】

以InternVL2-8B为例，按实际情况修改inference_8B.json对应参数，注意tokenizer_config的权重路径为转换前的权重路径。

推理demo视频下载[red-panda](https://huggingface.co/OpenGVLab/InternVL2-8B/blob/main/examples/red-panda.mp4)

```json
{
    "infer_data_type": "video",
    "file_path": "examples/internvl2/red-panda.mp4",    # 按实际情况输入视频路径
    "prompts": "Please describe the video shortly.", # 按实际情况输入提示词
    "model_id": "InternVLPipeline",
    "from_pretrained": "./pretrained/InternVL2-8B/release/mp_rank_00/model_optim_rng.pt", # 注意路径要到.pt文件
    ...
    "tokenizer":{
        ...
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "raw_ckpt/InternVL2-8B",
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
MM_MODEL="./examples/internvl2/inference_8B.json"
```

#### 3. 启动推理

```shell
bash examples/internvl2/inference_internvl.sh
```

<a id="jump6"></a>

## 训练后权重转回huggingface格式
MindSpeed-MM修改了部分原始网络的结构名称，在微调后，如果需要将权重转回huggingface格式，可使用`mm-convert`权重转换工具对微调后的权重进行转换，将权重名称修改为与原始网络一致。

```bash
mm-convert  InternVLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/InternVL2-8B" \
  --cfg.mm_dir "ckpt/mm_path/InternVL2-8B" \
  --cfg.hf_config.hf_dir "raw_ckpt/InternVL2-8B" \
  --cfg.parallel_config.llm_pp_layers [[6,9,9,8]] \
  --cfg.parallel_config.vit_pp_layers [[24,0,0,0]] \
  --cfg.trust_remote_code True

# 其中：
# save_hf_dir: mm微调后转换回hf模型格式的目录
# mm_dir: 微调后保存的权重目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致
# trust_remote_code: 为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性
```

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

如果要进行评测需要将要评测的数据集名称和路径传到examples/internvl2/evaluate_internvl2_8B.json
需要更改的字段有

- `from_pretrained` 需要改为模型的权重文件的路径，如果使用的是huggingface的权重则需要进行权重转换(参考前面的权重转换的章节)，如果使用MindSpeed-MM训练出的则不需要进行权重转换。
- `dataset_path` 需要填入上面下载的数据集文件路径。
- `evaluation_dataset` 为评测数据集的名称可选的名称有(`ai2d_test`、`mmmu_dev_val`、`docvqa_val`、`chartqa_test`)， **注意**：需要与上面的数据集路径相对应。
- `result_output_path` 为评测结果的输出路径，**注意**：每次评测前需要将之前保存在该路径下评测文件删除。
- `tokenizer`下面的`from_pretrained`为huggingface下载的InternVL2-8B权重路径。

```json
    "model_id": "InternVLPipeline",
    "from_pretrained": "./internvl8b_mm/release/mp_rank_00/model_optim_rng.pt",
    "dataset_path": "./AI2D_TEST.tsv",
    "evaluation_dataset":"ai2d_test",
    "evaluation_model":"internvl2_8b",
    "result_output_path":"./evaluation_outputs/",

    "tokenizer":{
        "hub_backend": "hf",
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "./InternVL2-8B",
        "model_max_length": 4096,
        "add_eos_token": false,
        "trust_remote_code": true,
        "use_fast": false
    }

```

examples/internvl2/evaluate_internvl2_8B.json改完后，需要将json文件的路径传入到examples/internvl2/evaluate_internvl2_8B.sh MM_MODEL字段中

```shell
MM_MODEL=examples/internvl2/evaluate_internvl2_8B.json
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
bash examples/internvl2/evaluate_internvl2_8B.sh
```

评测结果会输出到`result_output_path`路径中，会输出结果文件：

- *.xlsx文件，这个文件会输出每道题的预测结果和答案等详细信息。
- *.csv文件，这个文件会输出统计准确率等数据。

<a id="jump7"></a>
## 特性使用介绍

<a id="jump7.1"></a>
### DistTrain(分离部署)

#### 1. 特性介绍
DistTrain特性详细介绍参考文档[分离部署特性](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/dist-train.md)

#### 2. 模型分离部署权重转换

提供了MM CKPT与DistTrain CKPT之间的权重转换工具。

MM CKPT转DistTrain CKPT：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python examples/internvl2/internvl2_mm_convert_to_dt_ckpt.py \
  --load-dir pretrained/InternVL2-8B \
  --save-dir pretrained/InternVL2-8B-DistTrain \
  --target-vit-tp-size 1 \
  --target-vit-pp-size 1 \
  --target-vit-cp-size 1 \
  --target-vit-pp-layers '[24]' \
  --target-gpt-tp-size 1 \
  --target-gpt-pp-size 3 \
  --target-gpt-cp-size 1 \
  --target-gpt-pp-layers '[10,12,10]'
```

DistTrain CKPT转MM CKPT：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python examples/internvl2/internvl2_dt_convert_to_mm_ckpt.py \
  --load-dir pretrained/InternVL2-8B-DistTrain \
  --save-dir pretrained/InternVL2-8B-DistTrain-to-MM \
  --target-tp-size 1 \
  --target-pp-size 4 \
  --target-cp-size 1 \
  --target-vit-pp-layers '[24,0,0,0]' \
  --target-gpt-pp-layers '[6,9,9,8]'
```
同步修改`examples/internvl2/finetune_internvl2_*b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重`raw_ckpt/InternVL2-*B`进行区分。

以`InternVL2-8B`为例
```shell
LOAD_PATH="pretrained/InternVL2-8B"
```

#### 3. 使用方法
以`InternVL2-8B`为例

在启动脚本中添加参数`--dist-train`。
```shell
GPT_ARGS="
    ...
    --dist-train \
"
```
需要在MindSpeed-MM仓库中，对应模型目录下的`model.json`中添加`dist_config`字段，具体配置示例如下：
```json
{
  "dist_config": {
    "model_name": "internvl2",  // 多模态模型名称
    "use_multiparam_send_recv": false,  // 模型间是否传递tensor列表
    "model_config": [
      {
        "name": "vit",  // 内部模型名称
        "model_index": 0,  // 模型位于流水线中的序号
        "world_size": 1,  // 模型使用卡数
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "forward_only": false // 是否不做反向计算
      },
      {
        "name": "gpt",
        "model_index": 1,
        "world_size": 3,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 3,
        "context_parallel_size": 1,
        "forward_only": false
      }
    ]
  }
}
```

修改model.json中text_decoder中的pipeline_num_layers参数（跟上述dist_config中的pipeline_model_parallel_size参数对应）

```json
    "text_decoder": {
        "model_id": "internllm",
        "num_layers": 32,
        "pipeline_num_layers": [10, 12, 10],
        "hidden_size": 4096,
        ......
    }
```

修改启动脚本中的NPUS_PER_NODE参数，总的world_size为上述model.json中两个world_size之和，以单机上述配置为例：
```shell
NPUS_PER_NODE=4
```

---

<a id="jump8"></a>
## 环境变量声明
ASCEND_SLOG_PRINT_TO_STDOUT： 是否开启日志打印， 0：关闭日志打屏，1：开启日志打屏
ASCEND_GLOBAL_LOG_LEVEL： 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。0：对应DEBUG级别，1：对应INFO级别，2：对应WARNING级别，3：对应ERROR级别，4：对应NULL级别，不输出日志
TASK_QUEUE_ENABLE： 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化
COMBINED_ENABLE： 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景
CPU_AFFINITY_CONF： 控制CPU端算子任务的处理器亲和性，即设定任务绑核，设置0或未设置：表示不启用绑核功能， 1：表示开启粗粒度绑核， 2：表示开启细粒度绑核
HCCL_CONNECT_TIMEOUT:  用于限制不同设备之间socket建链过程的超时等待时间，需要配置为整数，取值范围[120,7200]，默认值为120，单位s
HCCL_EXEC_TIMEOUT： 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步
PYTORCH_NPU_ALLOC_CONF： 控制缓存分配器行为
ACLNN_CACHE_LIMIT： 配置单算子执行API在Host侧缓存的算子信息条目个数
TOKENIZERS_PARALLELISM： 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为
NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量
<a id="jump9"></a>