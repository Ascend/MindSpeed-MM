Wan2.2模型微调实践
===========================

Last updated: 12/08/2025. Author: cxiaolong

本文档介绍了如何使用 MindSpeed-MM 对 Wan2.2-T2V-A14B 模型进行微调的实践步骤。

背景介绍
------------

Wan2.2 是由阿里通义万相团队开发的一款视频生成基础模型，引入了扩散MoE架构，能够根据文本提示生成电影级别的视频内容。
同时支持T2V、I2V、TI2V等多模态生成任务。在AI驱动的视觉内容创作领域实现了显著的飞跃。

数据集介绍
------------

此次实践采用 Open-Sora-Dataset pixabay_v2 数据集进行微调。pixabay_v2 数据集是Open-Sora-Plan项目用于训练其视频生成大模型的核心数据之一，主要包含了Pixabay来源的主体数据，Pixabay视频数量为31,616个，占总数据量的近78.5%，时长约为219小时，占总体时长的约80%。


实践流程
------------

该流程主要包含环境搭建、权重下载及转换、数据集下载及预处理、参数配置、启动微调、启动推理等步骤。


Step 1: 搭建开发环境
::::::::::::::::::::::::::

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
    
.. code:: bash

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
    git checkout 93c45
    # 安装mindspeed及依赖
    pip install -e .
    cd ..

    # 安装mindspeed mm及依赖
    pip install -e .

    # 更新diffusers、peft
    pip install diffusers==0.35.1 peft==0.17.1

5. Decord下载

Decord是一款为深度学习设计的开源视频处理库，提供硬件加速解码和高效视频帧采样能力。

【X86系统安装】

.. code:: bash

    pip install decord

【Arm系统安装】

- apt方式安装请参考 `链接 <https://github.com/dmlc/decord>`_ 
- yum方式安装请参考 `链接 <https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh>`_ 


Step 2: 权重下载及转换
::::::::::::::::::::::::::::::

1. **权重下载**

从Huggingface库下载模型权重: `Wan2.2-T2V-A14B-Diffusers <https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct>`_  
保存在 ``MindSpeed-MM/ckpt/hf_path/Wan2.2-T2V-A14B-Diffusers`` 文件夹中。


2. **权重转换**

使用MindSpeed-MM提供的转换脚本将HuggingFace格式的Wan2.2-T2V-A14B-Diffusers模型的DiT部分权重（transformer）转换为MindSpeed-MM/Megatron格式：

.. code:: bash

    mm-convert WanConverter hf_to_mm \
               --cfg.source_path ./ckpt/hf_path/Wan2.2-T2V-A14B-Diffusers/transformer/ \
               --cfg.target_path ./ckpt/mm_path/Wan2.2-T2V-A14B-mm/transformer/ \
    
    mm-convert Wan2.2T2VConverter hf_to_dcp \
               --cfg.source_path ./ckpt/hf_path/Wan2.2-T2V-A14B-Diffusers/transformer_2/ \
               --cfg.target_path ./ckpt/mm_path/Wan2.2-T2V-A14B-mm/transformer_2/ \

.. note:: 
    huggingface Diffusers权重中包含两个transformer权重， 其中transformer文件夹对应高噪声（high）模型权重，transformer_2文件夹对应低噪声（low）模型权重

转换完成后，mm_path下面会有一个release文件夹和一个latest_checkpointed_iteration.txt文件，release文件夹下面包含了转换后的权重文件。

.. code:: text

    Wan2.2-T2V-A14B-mm/
    ├── transformer/                    # 高噪模型
    │   ├── latest_checkpointed_iteration.txt
    │   └── release/
    │       └── mp_rank_00/
    │           └── model_optim_rng.pt
    │
    └── transformer_2/                  # 低噪模型
        ├── latest_checkpointed_iteration.txt
        └── release/
            └── mp_rank_00/
                └── model_optim_rng.pt

Step 3: 数据集下载及预处理
::::::::::::::::::::::::::::::::::

1. **数据集下载**

下载 `Open-Sora-Dataset pixabay_v2 数据集 <https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/pixabay_v2_tar>`_ ，并解压到项目目录下的 ``./data/pixabay_v2`` 文件夹中。
下载数据集的标注文件 `video_pixabay_65f_601513.json <https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/blob/main/anno_jsons/video_pixabay_65f_601513.json>`_  至 ``./data/`` 路径下。

.. note:: 
    完整数据集较大[1.22TB]，可以选择下载部分folder的数据进行测试，例如只下载folder_01到folder_04的数据，约100GB。但注意需要处理 ``video_pixabay_65f_601513.json`` 文件，删除对应未下载视频的数据项。

2. **数据预处理**

将数据组织成如下格式

.. code:: text

    ├── data
        ├── data.json
        ├── videos
            ├── video000001.mp4
            ├── video000002.mp4
            ├── ...

videos/下存放视频文件，data.json中包含该数据集中所有的视频-文本对信息，具体示例如下：

.. code:: json

    [
        {
            "path": "./videos/video000001.mp4",
            "cap": "A scenic view of mountains during sunrise.",
            "num_frames": 81,
            "fps": 24,
            "resolution": {480, 832}
        },
        {
            "path": "./videos/video000002.mp4",
            "cap": "A bustling city street with people walking and cars passing by.",
            "num_frames": 81,
            "fps": 24,
            "resolution": {480, 832}
        },
        ...
    ]



Step 4: 参数配置
::::::::::::::::::::::::::::::

启动微调之前，需要分别对数据 ``data.txt``、 ``data.json``，模型 ``model.json``，训练脚本 ``finetune.sh`` 进行配置：

1. **examples/wan2.2/data.txt**

其中每一行表示一个数据集，第一个参数表示数据文件夹的路径，第二个参数表示data.json文件的路径，用 ``,`` 分隔。例如：

.. code:: text

    ./data/pixabay_v2,./data/pixabay_v2/data.json

2. **examples/wan2.2/A14B/t2v/data.json**

需要对tokenizer_config的 ``from_pretrained`` 字段进行修改，改为下载的huggingface tokenizer路径：

.. code:: json

    {
        "dataset_param": {
            ...
            "tokenizer_config": {
                "autotokenizer_name": "AutoTokenizer",
                "hub_backend": "hf",
                "from_pretrained": "./ckpt/hf_path/Wan2.2-T2V-A14B-Diffusers/tokenizer/",
                "model_max_length": 512
            }
        },
        ...
    }

3. **examples/wan2.2/A14B/t2v/pretrain_model_high.json**、**examples/wan2.2/A14B/t2v/finetune_model_low.json**

需要对 ``ae`` 和 ``text_encoder`` 的 ``from_pretrained`` 等字段进行修改，修改为下载的huggingface权重路径：

.. code:: json

    {
        ...
        "ae": {
            ...
            "from_pretrained": "./ckpt/Wan2.2-T2V-A14B-Diffusers/vae/",
            ...
        },
        "text_encoder": {
            ...
            "from_pretrained": "./ckpt/Wan2.2-T2V-A14B-Diffusers/text_encoder/",
            ...
        },
        ...
    }

4. **examples/wan2.2/A14B/t2v/pretrain_high.sh**、**examples/wan2.2/A14B/t2v/pretrain_low.sh**

一个单机8卡的高噪模型微调脚本示例如下(pretrain_high.sh)，注意需要修改 ``LOAD_PATH`` 为MM格式权重实际存放路径， ``SAVE_PATH`` 修改为微调后权重保存路径。

.. code:: bash

    #!/bin/bash
    source /usr/local/Ascend/cann/set_env.sh
    # 该变量只用于规避megatron对其校验，对npu无效
    export CUDA_DEVICE_MAX_CONNECTIONS=2 # 开启FSDP2时，不能置为1
    export ASCEND_SLOG_PRINT_TO_STDOUT=0
    export ASCEND_GLOBAL_LOG_LEVEL=3
    export TASK_QUEUE_ENABLE=1
    export COMBINED_ENABLE=1
    export CPU_AFFINITY_CONF=1
    export HCCL_CONNECT_TIMEOUT=1200
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    NPUS_PER_NODE=8
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

    TP=1
    PP=1
    VP=1
    CP=1
    MBS=1
    GRAD_ACC_STEP=1
    DP=$(($WORLD_SIZE/$TP/$PP/$CP))
    GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

    MM_DATA="./examples/wan2.2/A14B/t2v/data.json"
    MM_MODEL="./examples/wan2.2/A14B/t2v/pretrain_model_high.json"
    MM_TOOL="./mindspeed_mm/tools/tools.json"
    LOAD_PATH="./ckpt/mm_path/Wan2.2-T2V-A14B-mm/transformer/"
    SAVE_PATH="path to save your high noise expert wandit weight"
    FSDP_CONFIG="./examples/wan2.2/A14B/fsdp2_config.yaml"

    DISTRIBUTED_ARGS="
        --nproc_per_node $NPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "

    GPT_ARGS="
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --virtual-pipeline-model-parallel-size ${VP} \
        --context-parallel-size ${CP} \
        --context-parallel-algo ulysses_cp_algo \
        --micro-batch-size ${MBS} \
        --global-batch-size ${GBS} \
        --num-workers 8 \
        --lr 1e-5 \
        --min-lr 1e-5 \
        --adam-beta1 0.9 \
        --adam-beta2 0.999 \
        --adam-eps 1e-8 \
        --lr-decay-style constant \
        --weight-decay 1e-2 \
        --lr-warmup-init 0 \
        --lr-warmup-iters 0 \
        --clip-grad 1.0 \
        --train-iters 5000 \
        --no-gradient-accumulation-fusion \
        --no-load-optim \
        --no-load-rng \
        --no-save-optim \
        --no-save-rng \
        --bf16 \
        --distributed-timeout-minutes 20 \
        --use-fused-rmsnorm \
        --use-torch-fsdp2 \
        --untie-embeddings-and-output-weights \
        --fsdp2-config-path ${FSDP_CONFIG} \
        --optimizer-selection fused_torch_adamw \
        --use-cpu-initialization \
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
        --eval-iters 10 \
        --load $LOAD_PATH \
        --save $SAVE_PATH \
        --ckpt-format torch_dcp \
    "

    logfile=wan_high_$(date +%Y%m%d)_$(date +%H%M%S)
    mkdir -p logs
    torchrun $DISTRIBUTED_ARGS pretrain_sora.py \
        $GPT_ARGS \
        $MM_ARGS \
        $OUTPUT_ARGS \
        --distributed-backend nccl \
        2>&1 | tee logs/train_${logfile}.log

Step 5: 启动微调
::::::::::::::::::::::::::::::

按照上一步配置好的脚本，运行以下命令启动微调：

.. code:: bash

    bash examples/wan2.2/A14B/t2v/pretrain_high.sh


Step6 启动推理
::::::::::::::::::::::::::::::

1. 推理参数配置

- **examples/wan2.2/A14B/t2v/inference_model.json**：将 ``ae``、 ``tokenizer``、 ``text_encoder`` 的 ``from_pretrained`` 字段修改为下载的huggingface权重路径，将和 ``low_noise_predictor`` 字段值修改为MM低噪权重保存路径。例如：

.. code:: json

    {
        ...
        "ae": {
            ...
            "from_pretrained": "./ckpt/Wan2.2-T2V-A14B-Diffusers/vae/",
            ...
        },
        "tokenizer": {
            ...
            "from_pretrained": "./ckpt/hf_path/Wan2.2-T2V-A14B-Diffusers/tokenizer/",
            ...
        },
        "text_encoder": {
            ...
            "from_pretrained": "./ckpt/Wan2.2-T2V-A14B-Diffusers/text_encoder/",
            ...
        },
        ...
        "low_noise_predictor": "./ckpt/mm_path/Wan2.2-T2V-A14B-mm/transformer_2/",
        ...
    }

- **examples/wan2.2/samples_t2v_prompts.txt**：修改为你想生成的视频文本提示词文件，每行一个文本提示词。

- **examples/wan2.2/A14B/t2v/inference.sh**：修改 ``LOAD_PATH`` 为高噪模型MM权重路径。示例脚本如下：

.. code:: bash

    source /usr/local/Ascend/cann/set_env.sh

    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export ASCEND_SLOG_PRINT_TO_STDOUT=0
    export ASCEND_GLOBAL_LOG_LEVEL=3
    export TASK_QUEUE_ENABLE=1
    export COMBINED_ENABLE=1
    export CPU_AFFINITY_CONF=1
    export HCCL_CONNECT_TIMEOUT=1200
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
    NPUS_PER_NODE=1
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

    TP=1
    PP=1
    CP=1
    MBS=1
    GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

    MM_MODEL="examples/wan2.2/A14B/t2v/inference_model.json"
    LOAD_PATH="./ckpt/mm_path/Wan2.2-T2V-A14B-mm/transformer/"

    DISTRIBUTED_ARGS="
        --nproc_per_node $NPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "
    MM_ARGS="
    --mm-model $MM_MODEL
    "

    GPT_ARGS="
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --context-parallel-algo ulysses_cp_algo \
        --micro-batch-size ${MBS} \
        --global-batch-size ${GBS} \
        --lr 5e-6 \
        --min-lr 5e-6 \
        --train-iters 5010 \
        --weight-decay 0 \
        --clip-grad 0.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.999 \
        --no-gradient-accumulation-fusion \
        --no-load-optim \
        --no-load-rng \
        --no-save-optim \
        --no-save-rng \
        --bf16 \
        --load $LOAD_PATH \
    "

    torchrun $DISTRIBUTED_ARGS inference_sora.py $MM_ARGS $GPT_ARGS


2. 启动推理

运行以下命令启动推理：

.. code:: bash

    bash examples/wan2.2/A14B/t2v/inference.sh

最后推理生成的视频保存在 ``examples/wan2.2/output_videos/`` 文件夹下。