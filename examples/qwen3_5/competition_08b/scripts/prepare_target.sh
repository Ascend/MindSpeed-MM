#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/qwen35_08b
MM=/opt/MindSpeed-MM
HF=/workspace/shared_assets/HC2026/helloworld/model/Qwen3.5-0.8B

test -f "$HF/config.json"
test -f "$HF/model.safetensors-00001-of-00001.safetensors"
test -d "$MM/.git"
mkdir -p "$ROOT"/{data/coco,cache,logs,save,profiling,memory_snapshot}

source /usr/local/Ascend/cann/set_env.sh
cd "$MM"

if ! command -v mm-convert >/dev/null 2>&1; then
  bash scripts/install.sh --msbranch 26.1.0_core_r0.12.1 --yes
fi
bash examples/qwen3_5/install_extensions.sh

if ! /usr/local/python3.12.13/bin/python3 -c 'import modelscope' >/dev/null 2>&1; then
  /usr/local/python3.12.13/bin/python3 -m pip install modelscope==1.38.1
fi

if [ ! -f "$ROOT/dcp/latest_checkpointed_iteration.txt" ]; then
  mm-convert Qwen35Converter hf_to_dcp \
    --hf_dir "$HF" --dcp_dir "$ROOT/dcp" --num_workers 0
fi

echo "Environment and DCP weights are ready under $ROOT"
