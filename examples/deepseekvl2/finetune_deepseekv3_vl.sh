#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACLNN_CACHE_LIMIT=100000

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=64
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

MBS=1
GRAD_ACC_STEP=64
TP=8
PP=8
CP=1
EP=8
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

MM_DATA="./examples/deepseekvl2/data.json"
MM_MODEL="./examples/deepseekvl2/model_deepseekv3_vl.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="pretrained/DeepSeekVL"
SAVE_PATH="save_dir"

MM_ARGS="
    --mm-data ${MM_DATA} \
    --mm-model ${MM_MODEL} \
    --mm-tool ${MM_TOOL}
"

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
    --context-parallel-size ${CP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --seq-length 4096 \
    --tokenizer-type NullTokenizer \
    --vocab-size 129280 \
    --position-embedding-type rope \
    --rotary-base 10000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --lr 4e-5 \
    --min-lr 0.0 \
    --train-iters 5000 \
    --lr-decay-style cosine \
    --weight-decay 0.05 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer \
    --bf16 \
    --load $LOAD_PATH \
    --variable-seq-lengths \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers 4 \
    --use-flash-attn \
"
# To ensure code security, configure trust_remote_code to default to False.
# Users need to add the following parameter and ensure the security of the models and data they download.
# --trust-remote-code \
MLA_ARGS="
    --multi-head-latent-attention \
    --q-lora-rank 1536 \
    --qk-rope-head-dim 64 \
    --qk-nope-head-dim 128 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --use-fused-moe-token-permute-and-unpermute \
"

ROPE_ARGS="
    --rope-scaling-beta-fast 32 \
    --rope-scaling-beta-slow 1 \
    --rope-scaling-factor 40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim  1.0 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
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
    pretrain_deepseekvl.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log
chmod 440 logs/train_${logfile}.log
chmod -R 640 $SAVE_PATH
STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
SAMPLES_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration: $STEP_TIME"
echo "Average Samples per Second: $SAMPLES_PER_SECOND"
LOG_TOKENS_PER_SECOND=`grep "tokens per sample" logs/train_${logfile}.log`
if [ "$LOG_TOKENS_PER_SECOND" ]; then
    AVERAGE_TOKENS=`grep "tokens per sample" logs/train_${logfile}.log | awk -F 'tokens per sample:' '{print$2}' | awk -F '|' '{print$1}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
    TOKENS_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${SAMPLES_PER_SECOND}'*'${AVERAGE_TOKENS}'}'`
    echo "Consumed Tokens per Second: $TOKENS_PER_SECOND"
fi
