#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker"
COMMON_DIR="${DOCKER_DIR}/common"
MODEL_INSTALL_DIR="${DOCKER_DIR}/scripts/model_install"

# 可通过环境变量覆盖默认值
OS="${OS:-openeuler24.03}"

# 根据 OS 推导 OS_FAMILY 与软件源配置脚本
case "$OS" in
    openeuler*) OS_FAMILY="openeuler"; REPO_SCRIPT="configure_yum_repo.sh" ;;
    ubuntu*)    OS_FAMILY="ubuntu";    REPO_SCRIPT="configure_apt_repo.sh" ;;
    *) echo "ERROR: 不支持的操作系统: $OS（可选: openeuler24.03 / ubuntu22.04）"; exit 1 ;;
esac

# 检测 CPU 架构
ARCH="$(uname -m)"
if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
    echo "ERROR: 不支持的 CPU 架构: $ARCH（仅支持 x86_64 / aarch64）"
    exit 1
fi

echo "=========================================="
echo "准备 devcontainer 构建上下文"
echo "  OS:          ${OS}"
echo "  OS_FAMILY:   ${OS_FAMILY}"
echo "  架构:        ${ARCH}"
echo "  上下文目录:  ${DOCKER_DIR}"
echo "=========================================="

cd "${DOCKER_DIR}"

# 1. 下载并暂存 Miniconda 安装器（统一命名为 miniconda.sh，便于构建参数固定）
MINICONDA_FILE="Miniconda3-py311_26.1.1-1-Linux-${ARCH}.sh"
if [ ! -f "miniconda.sh" ]; then
    if [ -f "downloads/${MINICONDA_FILE}" ]; then
        echo ">>> 复用已下载的 Miniconda: downloads/${MINICONDA_FILE}"
    else
        echo ">>> 下载 Miniconda (${ARCH})..."
        bash "${COMMON_DIR}/download_miniconda.sh" "${DOCKER_DIR}/downloads" "${ARCH}"
    fi
    cp "downloads/${MINICONDA_FILE}" "miniconda.sh"
else
    echo ">>> Miniconda 已存在: miniconda.sh"
fi

# 2. 暂存 common_functions.sh
cp "${COMMON_DIR}/common_functions.sh" "common_functions.sh"

# 3. 暂存软件源配置脚本为 configure_repo.sh（Dockerfile 固定 COPY 该名称）
cp "${COMMON_DIR}/${REPO_SCRIPT}" "configure_repo.sh"

# 4. 暂存模型环境安装脚本到 install_scripts/
mkdir -p "install_scripts"
for script in "${MODEL_INSTALL_DIR}"/install_*.sh; do
    [ -f "$script" ] && cp "$script" "install_scripts/"
done

# 5. 暂存 decord 安装脚本（ARM 构建需要；x86 构建会被跳过，但文件必须存在）
cp "${COMMON_DIR}/install_decord_on_arm.sh" "install_decord_on_arm.sh"

# 6. 创建 torch_wheels 占位目录（空目录 = 在线安装 PyTorch）
mkdir -p "torch_wheels"
touch "torch_wheels/.placeholder"

# 7. 创建 decord_deps 目录
#    - x86_64: decord 已在阶段一通过 pip 安装，仅创建占位目录满足 COPY 路径
#    - aarch64: 下载 decord 源码编译依赖
if [ "$ARCH" = "aarch64" ]; then
    if [ ! -d "decord_deps" ] || [ -z "$(ls -A decord_deps 2>/dev/null)" ]; then
        echo ">>> 下载 decord 编译依赖(ARM)..."
        rm -rf "decord_deps"
        bash "${COMMON_DIR}/download_decord_deps.sh" "${DOCKER_DIR}/decord_deps"
    else
        echo ">>> decord 依赖已存在: decord_deps"
    fi
else
    mkdir -p "decord_deps"
    touch "decord_deps/.placeholder"
fi

echo "=========================================="
echo "构建上下文准备完成"
echo "=========================================="
