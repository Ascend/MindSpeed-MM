#!/bin/bash
set -e

source /tmp/common_functions.sh

MINDSPEED_MM_BRANCH="${1:-master}"

ENV_NAME="verl_qwen3vl"
WORK_DIR="/workspace/${ENV_NAME}"

echo "=========================================="
echo "Installing ${ENV_NAME} environment"
echo "=========================================="

source /opt/conda/etc/profile.d/conda.sh

# Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create --clone base -n ${ENV_NAME} -y
conda activate ${ENV_NAME}

mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

echo ">>> Configuring pip for architecture..."
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
elif [ "$ARCH" = "aarch64" ]; then
    pip config set global.extra-index-url "https://mirrors.huaweicloud.com/ascend/repos/pypi"
fi

echo ">>> Installing cmake..."
conda install -c conda-forge cmake=3.26.4 -y

echo ">>> Installing pybind11..."
pip_install_retry "pybind11==3.0.1" 3

echo ">>> Sourcing CANN environment..."
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/cann/set_env.sh || true
source /usr/local/Ascend/nnal/atb/set_env.sh || true

echo ">>> Cloning and installing vllm..."
[ ! -d "vllm" ] && git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout v0.11.0 && cd ${WORK_DIR}
cd vllm
pip_install_requirements_retry "requirements/build.txt" 3 || true
VLLM_TARGET_DEVICE=empty pip install --no-cache-dir -v -e .
cd ${WORK_DIR}

echo ">>> Cloning and installing vllm-ascend..."
[ ! -d "vllm-ascend" ] && git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && git checkout fed8145 && cd ${WORK_DIR}
cd vllm-ascend
pip_install_requirements_retry "requirements.txt" 3 || true
pip install --no-cache-dir -v -e .
cd ${WORK_DIR}

echo ">>> Cloning and installing verl..."
[ ! -d "verl" ] && git clone https://github.com/volcengine/verl.git
cd verl && git checkout 7df2afb && cd ${WORK_DIR}
cd verl
pip_install_requirements_retry "requirements.txt" 3 || true
pip install --no-cache-dir -v -e .
cd ${WORK_DIR}

echo ">>> Cloning and installing transformers..."
[ ! -d "transformers" ] && git clone https://github.com/huggingface/transformers.git
cd transformers && git checkout 7a833d1ccd41673030c85107f65f454c0c3222f5 && cd ${WORK_DIR}
cd transformers
pip_install_retry ".[torch]" 3
cd ${WORK_DIR}

echo ">>> Installing additional dependencies..."
pip_install_retry "qwen-vl-utils==0.0.11" 3
pip_install_retry "mathruler" 3
pip_install_retry "viztracer" 3
pip_install_retry "uvloop==0.21.0" 3
pip_install_retry "setuptools==80.9.0" 3
pip_install_retry "cloudpickle==3.1.2" 3

echo ">>> Reinstalling torch and torch_npu..."
reinstall_torch_and_npu

echo ">>> Cloning MindSpeed MM for verl_plugin..."
[ ! -d "MindSpeed-MM" ] && git clone --branch ${MINDSPEED_MM_BRANCH} https://gitcode.com/Ascend/MindSpeed-MM.git

echo ">>> Installing verl_plugin..."
cd ${WORK_DIR}/MindSpeed-MM/verl_plugin
export MODEL_SELECT="Qwen3vl"
export VERL_PATH="${WORK_DIR}/verl"
pip install --no-cache-dir -v -e .
cp -r ../examples/verl_examples/qwen3vl/* ${WORK_DIR}/verl/examples/grpo_trainer/
cd ${WORK_DIR}

conda clean -ya && rm -rf /root/.cache/pip

echo "=========================================="
echo "${ENV_NAME} environment installed successfully!"
echo "=========================================="
