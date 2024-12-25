#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=2
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
# 根据机器实际情况填写
NPUS_PER_NODE=16
# 注意，当前为多机运行，需要根据实际的机器ip创建examples/qwen2vl/hostfile.txt文件，其中每行为一台机器的ip地址
HOSTFILE="examples/qwen2vl/hostfile.txt"
MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')  # 获取hostfile第一行为masteraddr
MASTER_PORT=6000
NODE_ADDR=`hostname -I | awk '{for(i=1;i<=NF;i++)print $i}' | grep ${MASTER_ADDR%.*}. | awk -F " " '{print$1}'`  # 获取本机IP
NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
NNODES=$(cat $HOSTFILE | wc -l)
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
echo $MASTER_ADDR
echo $NODE_ADDR
echo $NODE_RANK
echo $NNODES


MM_DATA="./examples/qwen2vl/data_72b.json"
MM_MODEL="./examples/qwen2vl/model_72b.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
# 需要先根据readme把huggingface格式模型转换为mm格式
LOAD_PATH="ckpt/Qwen2-VL-72B-Instruct"
SAVE_PATH="save_dir"

TP=1
PP=16
CP=1
SEQ_LEN=1024
MBS=1
GRAD_ACC_STEP=96
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# GPT_ARGS中模型相关参数具体配置在example/qwen2vl/model_72b.json中，训练相关参数配置在这里
GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-layers 1 \
    --hidden-size 1 \
    --ffn-hidden-size 1 \
    --num-attention-heads 1 \
    --tokenizer-type NullTokenizer \
    --vocab-size 1 \
    --seq-length 1 \
    --max-position-embeddings 1 \
    --make-vocab-size-divisible-by 1 \
    --init-method-std 0.01 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 0 \
    --train-iters 10000 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 42 \
    --bf16 \
    --load $LOAD_PATH \
    --variable-seq-lengths \
    --enable-one-logger \
    --use-distributed-optimizer \
    --reuse-fp32-param \
    --use-flash-attn \
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
torchrun $DISTRIBUTED_ARGS pretrain_qwen2vl.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl > logs/train_${logfile}.log 2>&1
chmod 440 logs/train_${logfile}.log