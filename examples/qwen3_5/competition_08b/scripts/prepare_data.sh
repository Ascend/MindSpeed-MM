#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/qwen35_08b
MM=/opt/MindSpeed-MM
LLAVA=/workspace/shared_assets/datasets/LLaVA-Instruct-150K/llava_instruct_150k.json

test -f "$LLAVA"
mkdir -p "$ROOT/data/coco"

if [ ! -d "$ROOT/data/coco/train2017" ]; then
  cd "$ROOT/data/coco"
  modelscope download --dataset PAI/COCO2017 train2017.zip --local_dir .
  unzip -q -o train2017.zip
fi

if [ ! -f "$ROOT/data/output_llava_coco_data.json" ]; then
  cd "$MM"
  /usr/local/python3.12.13/bin/python3 \
    mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py \
    --llava_json_path "$LLAVA" \
    --coco_path "$ROOT/data/coco" \
    --output_json_path "$ROOT/data/output_llava_coco_data.json"
fi

test -s "$ROOT/data/output_llava_coco_data.json"
echo "Training data ready: $ROOT/data/output_llava_coco_data.json"
