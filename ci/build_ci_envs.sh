#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../docker"

# These are default settings. Feel free to tweak them based on your actual situation.
BASE_IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.12"
TORCH_VERSION="2.7.1"
TORCH_NPU_VERSION="2.7.1.post8"
MINDSPEED_MM_VERSION="26.1.0"
PYTHON_VERSION="3.11"

bash "build.sh" \
    --base-image "$BASE_IMAGE" \
    --torch-version "$TORCH_VERSION" \
    --torch-npu-version "$TORCH_NPU_VERSION" \
    -v "$MINDSPEED_MM_VERSION" \
    --python-version "$PYTHON_VERSION" \
    --cleanup-on-fail \
    --build-ci
