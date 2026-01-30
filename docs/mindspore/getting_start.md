# 快速上手（MindSpore后端）
以MindSpore AI套件为后端的MindSpeed MM同时支持了部分多模态生成和多模态理解模型。下面介绍典型模型Qwen2.5VL在MindSpore后端下的使用方法，引导开发者快速上手预置模型在MindSpore + 昇腾NPU上的高效运行。

## Qwen2.5-VL-7B MindSpore后端快速上手指南
更多细节请参考[qwen2.5vl](../../examples/mindspore/qwen2.5vl/README.md)。
### 1. 环境安装
#### 1.1 昇腾软件安装
基于Python3.10版本，昇腾环境安装请参考[MindSpore后端安装指南](./install_guide.md)。


针对MindSpeed MindSpore后端，昇腾社区提供了一键转换工具MindSpeed-Core-MS，旨在帮助用户自动拉取相关代码仓并对torch代码进行一键适配，进而使用户无需再额外手动开发适配即可在华为MindSpore+CANN环境下一键拉起模型训练。在进行一键转换前，用户需要拉取相关的代码仓以及进行环境搭建：

```shell
# 创建conda环境
conda create -n test python=3.10
conda activate test

# 使用环境变量
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0

# 安装MindSpeed-Core-MS转换工具
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.4.0

# 使用MindSpeed-Core-MS内部脚本自动拉取相关代码仓并一键适配、提供配置环境
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm

# 替换MindSpeed中的文件
cd MindSpeed-MM
cp examples/mindspore/qwen2vl/dot_product_attention.py ../MindSpeed/mindspeed/core/transformer/dot_product_attention.py
mkdir ckpt
mkdir data
mkdir logs
```

### 2. 权重下载及转换
#### 2.1 权重下载

从Hugging Face库下载[Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)的模型权重, 并将下载的模型权重保存到本地的`ckpt/hf_path/Qwen2.5-VL-7B-Instruct`目录下。

#### 2.2 权重转换
MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了Hugging Face权重和MindSpeed-MM权重的互相转换以及PP（Pipeline Parallel）权重的重切分。更多细节请参考[权重转换工具](https://gitcode.com/Ascend/MindSpeed-MM/blob/2.2.0/docs/features/权重转换工具.md)。

以下是将Hugging Face权重转为MindSpeed MM权重的转换示例：
```bash
# 7b
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
  --cfg.parallel_config.tp_size 1
```

其中参数含义如下：
- mm_dir: 转换后保存目录
- hf_dir: Hugging Face权重目录
- llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
- vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
- tp_size: tp并行数量，注意要和微调启动脚本中的配置一致

MindSpeed-MM修改了部分原始网络的结构名称，在微调后，如果需要将权重转回Hugging Face格式，可使用`mm-convert`权重转换工具对微调后的权重进行转换，将权重名称修改为与原始网络一致。

以下是mm2hf的转换示例：
```bash
mm-convert  Qwen2_5_VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-7B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 1
```

其中参数含义如下：
- save_hf_dir: mm微调后转换回hf模型格式的目录
- mm_dir: 微调后保存的权重目录
- hf_dir: Hugging Face权重目录
- llm_pp_layers: llm在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致
- vit_pp_layers: vit在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致
- tp_size: tp并行数量，注意要和微调启动脚本中的配置一致

如果需要使用转换的模型进行训练，同步修改`examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重目录，注意与原始权重 `ckpt/hf_path/Qwen2.5-VL-7B-Instruct`进行区分。

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
```

### 3. 数据集准备及处理
#### 3.1 数据集下载（以COCO2017数据集为例）
(1)用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中。

(2)获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下。

#### 3.2 数据集处理
将数据集按以下目录结构整理后，直接运行数据转换脚本 `python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py`。

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

### 4. 启动微调
#### 4.1 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

#### 4.2 配置参数

##### 4.2.1 数据目录配置

根据实际情况修改`examples/mindspore/qwen2.5vl/data_7b.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

以Qwen2.5VL-7B为例，`data_7b.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径。

**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2.5-VL-7B-Instruct",
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

##### 4.2.2 模型保存加载及日志信息配置

根据实际情况配置`examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 加载路径
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
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

若需要加载指定迭代次数的权重、优化器等状态，需将加载路径`LOAD_PATH`设置为保存文件夹路径`LOAD_PATH="save_dir"`，并修改`latest_checkpointed_iteration.txt`文件内容为指定迭代次数。

```
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

##### 4.2.3 单机运行配置

配置`examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`参数如下。

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/cann/set_env.sh
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

##### 4.2.4 多机运行配置

创建文件`examples/mindspore/qwen2.5vl/hostfile.txt`，在该文件中，每行填写一台训练机器的 IP 地址，例如：

```
xxx.xxx.xxx.xxx
xxx.xxx.xxx.xxx
xxx.xxx.xxx.xxx
```

在所有机器上配置`examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`参数如下。

```shell
NPUS_PER_NODE=8
HOSTFILE="examples/mindspore/qwen2.5vl/hostfile.txt"
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')  # The first line is master address.
MASTER_PORT=29501
NODE_ADDR=`hostname -I | awk '{for(i=1;i<=NF;i++)print $i}' | grep ${MASTER_ADDR%.*}. | awk -F " " '{print$1}'`
NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
NNODES=$(cat $HOSTFILE | wc -l)
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
export LOCAL_WORLD_SIZE=$NPUS_PER_NODE
```

#### 4.3 启动微调

以Qwen2.5VL-7B为例，启动微调训练任务。

```shell
bash examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
```
> 注：启动多机任务时，需在所有参与的机器上分别执行该命令。
