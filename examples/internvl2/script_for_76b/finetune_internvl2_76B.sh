#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=16
MASTER_PORT=60111

HOSTFILE='./examples/internvl2/script_for_76b/hostfile'
NODEADDR=$(hostname -I | awk -F " " '{print$1}')
NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODEADDR'"];}' $HOSTFILE)
NNODES=$(cat $HOSTFILE | wc -l)
MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo $MASTER_ADDR
echo $NODE_RANK
echo $NNODES
echo $WORLD_SIZE
echo $NETNAME

MBS=1
GRAD_ACC_STEP=128
TP=1
PP=16
CP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

MM_DATA="./examples/internvl2/script_for_76b/data_76B.json"
MM_MODEL="./examples/internvl2/script_for_76b/model_76B.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="InternVL2-76B_pp16"
SAVE_PATH="save_dir"

MM_ARGS="
    --mm-data ${MM_DATA} \
    --mm-model ${MM_MODEL} \
    --mm-tool ${MM_TOOL}
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
echo $DISTRIBUTED_ARGS
GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 8192 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --vocab-size 128258 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --lr 1e-5 \
    --min-lr 0.0 \
    --train-iters 50 \
    --lr-decay-style cosine \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --bf16 \
    --variable-seq-lengths \
    --use-flash-attn \
    --load $LOAD_PATH \
    --use-distributed-optimizer \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --eval-iters 5000 \
    --save $SAVE_PATH \
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

torchrun $DISTRIBUTED_ARGS \
    pretrain_internvl.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl >> logs/train_${logfile}.log 2>&1

set +x