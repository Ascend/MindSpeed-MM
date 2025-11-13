source /usr/local/Ascend/ascend-toolkit/set_env.sh

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
NPUS_PER_NODE=2
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

TP=1
PP=1
CP=2
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)

MM_MODEL="$BASEPATH/tests/st/run_configs/inference_wan2.2_t2v/inference_model.json"
LOAD_PATH="./weights/Wan-AI/Wan2.2-T2V-A14B-Diffusers/transformer/"  # ensure the wandit weight be converted

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
    --use-fused-rmsnorm \
    --bf16 \
    --load $LOAD_PATH \
    --context-parallel-algo ulysses_cp_algo \
"

pip install diffusers==0.35.1 peft==0.17.1

torchrun $DISTRIBUTED_ARGS $BASEPATH/inference_sora.py $MM_ARGS $GPT_ARGS

pip install diffusers==0.30.3 peft==0.7.1