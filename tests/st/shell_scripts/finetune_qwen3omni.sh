#!/bin/bash

set -e
INITIAL_DIR=$(pwd)
trap 'cd $INITIAL_DIR; echo "force back to ${INITIAL_DIR}"' EXIT

BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)
cd "$BASEPATH"

TMP_FILE=$(mktemp)
pip freeze | grep -E "transformers|accelerate|datasets" > "$TMP_FILE"
cat "$TMP_FILE"

cp -r /home/ci_resource/code/transformers-7a833d1/transformers .
cd transformers
pip install -e .
pip install accelerate==1.11.0 librosa==0.11.0 datasets==4.0.0

cd "$INITIAL_DIR"
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

# 单机8卡，需要减层
NPUS_PER_NODE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

MM_DATA="$BASEPATH/tests/st/run_configs/finetune_qwen3omni/data.json"
MM_MODEL="$BASEPATH/tests/st/run_configs/finetune_qwen3omni/model.json"
MM_TOOL="$BASEPATH/mindspeed_mm/tools/tools.json"
FSDP2_PATH="$BASEPATH/examples/qwen3omni/fsdp2_config.yaml"

TP=1
PP=1
CP=1
MBS=1
GRAD_ACC_STEP=1
SEQ_LEN=2048
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))


DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# GPT_ARGS中模型相关参数具体配置在example/qwen3omni/model.json中，训练相关参数配置在这里
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
    --train-iters 4 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --seed 42 \
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
    --distributed-timeout-minutes 60 \
    --init-model-with-meta-device \
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
    --log-tps \
"
logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS $BASEPATH/pretrain_transformers.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log

pip uninstall -y librosa
pip install -r "$TMP_FILE"
cd "$BASEPATH"
rm -f "$TMP_FILE"
rm -rf transformers
cd "$INITIAL_DIR"