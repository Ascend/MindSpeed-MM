source /usr/local/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="0"
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
MASTER_ADDR=localhost
MASTER_PORT=12875
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))


TP=1
PP=1
CP=1
MBS=1
GBS=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
MM_ARGS="
 --mm-model "examples/qihoo_t2x/inference_model_image.json"
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 2e-5 \
    --min-lr 2e-5 \
    --train-iters 5010 \
    --weight-decay 0 \
    --clip-grad 1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --fp16 \
"

torchrun $DISTRIBUTED_ARGS  inference_qihoo.py  $MM_ARGS $GPT_ARGS