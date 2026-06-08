#!/bin/bash

set -e
INITIAL_DIR=$(pwd)
trap 'cd $INITIAL_DIR; echo "force back to ${INITIAL_DIR}"' EXIT

BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)
cd "$BASEPATH"

TMP_FILE=$(mktemp)
pip freeze | grep -E "transformers" > "$TMP_FILE"
cat "$TMP_FILE"

pip install transformers==4.57.3 torchaudio==2.7.1 librosa soundfile sox onnxruntime gradio einops

cd "$INITIAL_DIR"

fetch_and_copy_mindspeed() {
    DATE_SUFFIX=$(date +%Y%m%d)
    TARGET_PATH="/home/ci_resource/code/MindSpeed-date/MindSpeed-${DATE_SUFFIX}"

    if [ -d "$TARGET_PATH" ]; then
        echo "Path already exists: $TARGET_PATH"
        cp -r "$TARGET_PATH/mindspeed" "$BASEPATH/"
        echo "Copied mindspeed from $TARGET_PATH to $BASEPATH"
    else
        echo "Path does not exist: $TARGET_PATH"
        if git clone --depth 1 https://gitcode.com/Ascend/MindSpeed "$TARGET_PATH"; then
            echo "Clone successful: $TARGET_PATH"
            cp -r "$TARGET_PATH/mindspeed" "$BASEPATH/"
            echo "Copied mindspeed from $TARGET_PATH to $BASEPATH"
        else
            echo "Clone failed, will use cache file"
            CACHE_PATH="/home/ci_resource/code/MindSpeed-26.0.0"
            if [ -d "$CACHE_PATH/mindspeed" ]; then
                cp -r "$CACHE_PATH/mindspeed" "$BASEPATH/"
                echo "Copied mindspeed from $CACHE_PATH to $BASEPATH"
            else
                echo "Cache path does not contain mindspeed folder: $CACHE_PATH"
                exit 1
            fi
        fi
    fi
}

need_restore=false
if [ -e "$BASEPATH/mindspeed" ]; then
    echo "mindspeed exists, moving to mindspeed-bak"
    rm -rf "$BASEPATH/mindspeed-bak"
    mv "$BASEPATH/mindspeed" "$BASEPATH/mindspeed-bak"
    need_restore=true
fi

if fetch_and_copy_mindspeed; then
    echo "Successfully fetched and copied mindspeed"
else
    echo "Failed to fetch and copy mindspeed"
    if [ "$need_restore" = true ]; then
        echo "Restoring backup mindspeed-bak to mindspeed"
        rm -rf "$BASEPATH/mindspeed" 2>/dev/null
        mv "$BASEPATH/mindspeed-bak" "$BASEPATH/mindspeed"
    fi
    exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export NON_MEGATRON=true
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS $BASEPATH/mindspeed_mm/fsdp/train/trainer.py \
    $BASEPATH/tests/st/run_configs/finetune_qwen3tts/qwen3tts_config.yaml \
    2>&1 | tee logs/train_${logfile}.log

STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F 'elapsed time per iteration [(]ms[)]:' '{print$2}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
GBS=`grep "global batch size" logs/train_${logfile}.log | awk -F 'global batch size:' '{print$2}' | awk -F '|' '{print$1}' | head -n 1 | awk '{print $1}'`
SAMPLES_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration (ms): $STEP_TIME" | tee -a logs/train_${logfile}.log
echo "Average Samples per Second: $SAMPLES_PER_SECOND" | tee -a logs/train_${logfile}.log

rm -rf "$BASEPATH/mindspeed"
echo "Remove temporarily fetched mindspeed file"
if [ "$need_restore" = true ]; then
    echo "Restoring original mindspeed from backup"
    mv "$BASEPATH/mindspeed-bak" "$BASEPATH/mindspeed"
fi

pip install -r "$TMP_FILE"
rm -f "$TMP_FILE"
cd "$INITIAL_DIR"
