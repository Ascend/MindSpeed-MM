Qwen3VL-30B-MoE模型微调实践
===========================

Last updated: 12/08/2025. Author: cxiaolong

背景介绍
------------

Qwen3-VL是2025年发布的多模态 (vision + language) MoE 模型，是通义千问（Qwen）系列里的旗舰级视觉-语言大模型，它既能处理文本，也能图像/视频输入，同时保持了强大的语言理解与生成能力。

本次的实践的目标是使用COCO2017数据集在NPU上对Qwen3VL-30B模型进行微调。


数据集介绍
------------

COCO2017（Common Objects in Context 2017）是一个大规模图像理解数据集，其训练集包含超过118,000张图像。图像内容覆盖80类常见物体，包含真实场景中的检测目标、物体实例分割、关键点标注和全景分割等信息，是计算机视觉中物体识别、图像理解、图文任务和多模态大模型训练中最常用的基础数据集之一。

LLaVA-Instruct-150K 是一个图像-文本对话监督微调数据集，包含约 150,000 条人工构建的“图像 + 多轮对话（指令/回答）”示例。数据主要由 GPT-4 生成，涵盖图像描述、视觉问答、推理、定位、细节理解、指令性问答等任务，用于训练视觉语言模型（如 LLaVA/Qwen-VL）理解视觉内容并进行自然语言互动，是目前多模态大模型常用的高质量对话式视觉指令微调数据。


实践流程
------------

该流程主要包含环境搭建、权重下载及转换、数据集下载及预处理、参数配置、启动微调等步骤。



Step 1: 搭建开发环境
::::::::::::::::::::

1. 硬件准备

昇腾A3或A2加速卡，建议8卡或以上

2. 参考 `环境搭建 <https://mindspeed-mm.readthedocs.io/zh-cn/latest/quick_start/%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.html>`_ 章节，完成HDK、CANN、PyTorch和TorchNPU基础环境搭建；

3. 创建并激活Python虚拟环境；

.. code:: bash

    conda create -n [env_name] python=3.10
    conda activate [env_name]
    # 安装CANN latest下的te包，这会自动安装sympy,decorator等必装包
    pip install [CANN_HOME_PATH]/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl

4. 按照下面的步骤安装MindSpeed-MM及其依赖包：

拉取MindSpeed MM代码仓，并进入代码仓根目录：

.. code:: bash

    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    cd MindSpeed-MM

对于X86架构机器，执行如下指令：

.. code:: bash

    bash scripts/install.sh --arch x86 --msid d76dbddd4517d48a2fc1cd494de8b9a6cfdbfbab&& pip install -r examples/qwen3vl/requirements.txt

对于ARM架构机器，执行如下指令：

.. code:: bash

    bash scripts/install.sh --arch arm --msid d76dbddd4517d48a2fc1cd494de8b9a6cfdbfbab&& pip install -r examples/qwen3vl/requirements.txt


Step 2: 权重下载及转换
::::::::::::::::::::::::

1. **权重下载**

从Huggingface库下载模型权重:
- `Qwen3-VL-30B-A3B-Instruct <https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct>`_  
保存在MindSpeed-MM/ckpt/hf_path/Qwen3-VL-30B-A3B-Instruct

2. **权重转换**

使用MindSpeed-MM提供的转换脚本将HuggingFace格式的Qwen3-VL模型转换为MindSpeed/DCP(Pytorch分布式检查点)格式：

.. code:: bash

    mm-convert Qwen3VLConverter hf_to_dcp \
                --hf_dir ./ckpt/hf_path/Qwen3-VL-30B-A3B-Instruct \
                --dcp_dir ./ckpt/dcp_path/Qwen3-VL-30B-A3B-Instruct-dcp \

转换完成后，dcp_dir下面会有一个release文件夹和一个latest_checkpointed_iteration.txt文件，release文件夹下面包含了转换后的DCP权重文件。


Step 3: 数据集下载及预处理
:::::::::::::::::::::::::::

1. **数据集下载**

下载 `COCO2017数据集 <https://cocodataset.org/#download>`_ [118K/18GB]，并解压到项目目录下的./data/COCO2017文件夹中。
下载图片数据集的描述文件 `LLaVA-Instruct-150K <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K>`_  至./data/路径下。

2. **数据集预处理**
运行数据转换脚本`python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py`，转换后参考数据目录结构如下：

.. code:: text

    ├── data
        ├── COCO2017
            ├── train2017

        ├── llava_instruct_150k.json
        ├── mllm_format_llava_instruct_data.json
    ...


Step 4: 参数配置
::::::::::::::::::::::::::::::

启动微调之前，需要分别对数据`data.json`，模型`model.json`，训练脚本`finetune.sh`进行配置：

1. **examples/qwen3vl/data_30B.json**

需要对`model_name_or_path`、`dataset_dir`、`dataset`等字段进行修改：

- `model_name_or_path`：hf权重存放路径
- `dataset_dir`：数据集存放路径
- `dataset`：转换后的数据集配置文件（mllm_format_llava_instruct_data.json）路径。

例如：

.. code:: text

    {
        "dataset_param": {
            "dataset_type": "huggingface",
            "preprocess_parameters": {
                "model_name_or_path": "./ckpt/hf_path/Qwen3-VL-30B-A3B-Instruct",
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

2. **examples/qwen3vl/model_30B.json**

需要对`init_from_hf_path`字段进行修改，改为实际的hf权重存放路径。
例如：

.. code:: text

    {
        ...
        "init_from_hf_path": "./ckpt/hf_path/Qwen3-VL-30B-A3B-Instruct",
        ...
    }

3. **MindSpeed-MM/examples/qwen3vl/finetune_qwen3vl_30B.sh**

一个单机8卡的微调脚本示例如下，注意需要修改`LOAD_PATH`为DCP格式权重存放路径

.. code:: bash

    #!/bin/bash
    # 改为实际的环境变量路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # 该变量只用于规避megatron对其校验，对npu无效
    export CUDA_DEVICE_MAX_CONNECTIONS=2 # 开启FSDP2时，不能置为1
    export ASCEND_SLOG_PRINT_TO_STDOUT=0
    export ASCEND_GLOBAL_LOG_LEVEL=3
    export TASK_QUEUE_ENABLE=2
    export COMBINED_ENABLE=1
    export CPU_AFFINITY_CONF=1
    export HCCL_CONNECT_TIMEOUT=1200
    export NPU_ASD_ENABLE=0
    export ASCEND_LAUNCH_BLOCKING=0
    export ACLNN_CACHE_LIMIT=100000
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export MULTI_STREAM_MEMORY_REUSE=1

    NPUS_PER_NODE=8
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

    MM_DATA="./examples/qwen3vl/data_30B.json"
    MM_MODEL="./examples/qwen3vl/model_30B.json"
    MM_TOOL="./mindspeed_mm/tools/tools.json"
    LOAD_PATH="ckpt/dcp_path/Qwen3-VL-30B-A3B-Instruct-dcp" 
    SAVE_PATH="save_dir"
    FSDP2_PATH="./examples/qwen3vl/fsdp2_config.yaml"

    TP=1
    PP=1
    CP=1
    MBS=1
    GRAD_ACC_STEP=1
    SEQ_LEN=1024
    DP=$(($WORLD_SIZE/$TP/$PP/$CP))
    GBS=$(($MBS*$GRAD_ACC_STEP*$DP))


    DISTRIBUTED_ARGS="
        --nproc_per_node $NPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "

    # GPT_ARGS中模型相关参数具体配置在example/qwen2vl/model_xb.json中，训练相关参数配置在这里
    GPT_ARGS="
        --use-mcore-models \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --context-parallel-algo ulysses_cp_algo \
        --micro-batch-size ${MBS} \
        --global-batch-size ${GBS} \
        --tokenizer-type NullTokenizer \
        --vocab-size 152064 \
        --seq-length ${SEQ_LEN} \
        --make-vocab-size-divisible-by 1 \
        --normalization RMSNorm \
        --use-fused-rmsnorm \
        --swiglu \
        --use-fused-swiglu \
        --no-masked-softmax-fusion \
        --lr 1.0e-5 \
        --lr-decay-style cosine \
        --weight-decay 0 \
        --train-iters 10000 \
        --lr-warmup-fraction 0.1 \
        --clip-grad 0.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.999 \
        --no-gradient-accumulation-fusion \
        --seed 42 \
        --load $LOAD_PATH \
        --use-flash-attn \
        --no-load-optim \
        --no-load-rng \
        --no-save-optim \
        --no-save-rng \
        --num-workers 8 \
        --use-torch-fsdp2 \
        --untie-embeddings-and-output-weights \
        --ckpt-format torch_dcp \
        --fsdp2-config-path $FSDP2_PATH \
        --optimizer-selection fused_torch_adamw \
        --use-cpu-initialization \
        --calculate-per-token-loss \
        --init-model-with-meta-device \
        --log-tps
    "

    MM_ARGS="
        --mm-data $MM_DATA \
        --mm-model $MM_MODEL \
        --mm-tool $MM_TOOL
    "

    OUTPUT_ARGS="
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 10000 \
        --eval-iters 5000 \
        --save $SAVE_PATH \
    "
    logfile=$(date +%Y%m%d)_$(date +%H%M%S)
    mkdir -p logs
    torchrun $DISTRIBUTED_ARGS pretrain_transformers.py \
        $GPT_ARGS \
        $MM_ARGS \
        $OUTPUT_ARGS \
        --distributed-backend nccl \
        2>&1 | tee logs/train_${logfile}.log


Step 5: 启动微调
::::::::::::::::::::::

.. code:: bash

    bash examples/qwen3vl/finetune_qwen3vl_xxB.sh
