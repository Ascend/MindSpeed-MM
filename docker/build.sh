#!/bin/bash
# ============================================
# MindSpeed MM Docker Image Build Script
# ============================================

cleanup_dangling() {
    echo ">>> Cleaning up <none> tagged images and corresponding containers..."

    local dangling_images=$(docker images -f "dangling=true" -q 2>/dev/null)
    if [ -n "$dangling_images" ]; then
        for img_id in $dangling_images; do
            local containers=$(docker ps -a -q --filter "ancestor=$img_id" 2>/dev/null)
            if [ -n "$containers" ]; then
                echo ">>> Removing containers from dangling image: $img_id"
                docker rm -f $containers 2>/dev/null || true
            fi
        done
        echo ">>> Removing dangling images..."
        docker rmi $dangling_images 2>/dev/null || true
    else
        echo ">>> No dangling images found"
    fi

    echo ">>> Cleanup complete"
}
# Dockerfile naming: Dockerfile (unified, supports all NPU types and OS)
# Image tag naming: {version}-{chip}-{os}-py{python_version}-{architecture}
#
# Usage:
#   bash build.sh -t A3 [-m /path/to/miniconda.sh] [-o ubuntu22.04]
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="${SCRIPT_DIR}/common"
SCRIPTS_DIR="${SCRIPT_DIR}/scripts"

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build MindSpeed MM Docker Image

Required:
    -t, --npu-type TYPE      NPU type: A3 or 910B (auto-detected from --base-image if not specified)

Optional:
    -m, --miniconda PATH     Miniconda installer path (auto-download if not specified)
    -d, --decord-deps PATH   decord dependencies directory path (auto-download for ARM)
    -s, --decord-script PATH decord install script path (default: common/install_decord_on_arm.sh)
    -i, --image-name NAME    Image name (default: mindspeed-mm:{version}-{chip}-{os}-py{py_ver}-{arch})
    -n, --no-cache           Build without cache
    -o, --os OS              OS: openeuler24.03 or ubuntu22.04 (auto-detected from --base-image if not specified)
    -v, --version VERSION    MindSpeed MM version (default: master, determines model install scripts)
    --torch-version VER      PyTorch version (default: 2.7.1, for online install)
    --torch-npu-version VER  torch-npu version (default: 2.7.1, for online install)
    --torch-whl PATH         torch .whl file path (offline install)
    --torch-npu-whl PATH     torch-npu .whl file path (offline install)
    --torchvision-whl PATH   torchvision .whl file path (optional, offline install)
    --torchaudio-whl PATH    torchaudio .whl file path (optional, offline install)
    --base-image-version VER Base image CANN version (default: 9.0.0-beta.2)
    --base-image IMAGE       Full base image name (higher priority than --base-image-version)
    --cleanup-on-fail        Clean up dangling images/containers if build fails
    -h, --help               Show help

Dockerfile naming convention: Dockerfile (unified, supports all NPU types and OS)
    NPU type and OS are passed as build arguments

Image tag naming convention: {version}-{chip}-{os}-py{python_version}-{architecture}
    e.g. master-a3-openeuler24.03-py3.11-x86_64
         master-910b-openeuler24.03-py3.11-aarch64

Examples:
    bash $0 -t A3
    bash $0 -t A3 -o ubuntu22.04
    bash $0 --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.2-910b-openeuler24.03-py3.11
    bash $0 -t 910B --torch-version 2.7.1 --torch-npu-version 2.7.1
    bash $0 -t A3 --base-image-version 9.0.0
    bash $0 -t A3 -i myproject/mindspeed-mm:v1.0
    bash $0 -t A3 --torch-whl /path/to/torch.whl --torch-npu-whl /path/to/torch_npu.whl
EOF
}

NPU_TYPE=""
MINICONDA_PATH=""
DECORD_DEPS_PATH=""
DECORD_SCRIPT_PATH=""
IMAGE_NAME=""
NO_CACHE=""
OS="openeuler24.03"
TORCH_VERSION="2.7.1"
TORCH_NPU_VERSION="2.7.1"
TORCH_WHL_PATH=""
TORCH_NPU_WHL_PATH=""
TORCHVISION_WHL_PATH=""
TORCHAUDIO_WHL_PATH=""
BASE_IMAGE_VERSION="9.0.0-beta.2"
MINDSPEED_MM_VERSION="master"
PYTHON_VERSION="3.11"
NPU_TYPE_EXPLICIT=false
OS_EXPLICIT=false
CLEANUP_ON_FAIL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--npu-type)      NPU_TYPE="$2"; NPU_TYPE_EXPLICIT=true; shift 2 ;;
        -m|--miniconda)     MINICONDA_PATH="$2"; shift 2 ;;
        -d|--decord-deps)   DECORD_DEPS_PATH="$2"; shift 2 ;;
        -s|--decord-script) DECORD_SCRIPT_PATH="$2"; shift 2 ;;
        -i|--image-name)    IMAGE_NAME="$2"; shift 2 ;;
        -n|--no-cache)      NO_CACHE="--no-cache"; shift ;;
        -o|--os)            OS="$2"; OS_EXPLICIT=true; shift 2 ;;
        -v|--version)       MINDSPEED_MM_VERSION="$2"; shift 2 ;;
        --torch-version)    TORCH_VERSION="$2"; shift 2 ;;
        --torch-npu-version) TORCH_NPU_VERSION="$2"; shift 2 ;;
        --torch-whl)        TORCH_WHL_PATH="$2"; shift 2 ;;
        --torch-npu-whl)    TORCH_NPU_WHL_PATH="$2"; shift 2 ;;
        --torchvision-whl)  TORCHVISION_WHL_PATH="$2"; shift 2 ;;
        --torchaudio-whl)   TORCHAUDIO_WHL_PATH="$2"; shift 2 ;;
        --base-image-version) BASE_IMAGE_VERSION="$2"; shift 2 ;;
        --base-image)       BASE_IMAGE="$2"; shift 2 ;;
        --cleanup-on-fail)  CLEANUP_ON_FAIL=true; shift ;;
        -h|--help)          show_help; exit 0 ;;
        *)                  echo "Unknown argument: $1"; show_help; exit 1 ;;
    esac
done

parse_base_image_tag() {
    local image="$1"
    local tag=""

    if [[ "$image" == *":"* ]]; then
        tag="${image##*:}"
    else
        echo "Warning: No tag found in base image name"
        return 1
    fi

    echo ">>> Parsing base image tag: $tag"

    local tag_lower=$(echo "$tag" | tr '[:upper:]' '[:lower:]')

    local detected_npu=""
    if [[ "$tag_lower" == *"910b"* ]]; then
        detected_npu="910B"
    elif [[ "$tag_lower" == *"-a3-"* ]] || [[ "$tag_lower" == *"-a3-py"* ]]; then
        detected_npu="A3"
    fi

    local detected_os=""
    if [[ "$tag_lower" == *"openeuler24.03"* ]]; then
        detected_os="openeuler24.03"
    elif [[ "$tag_lower" == *"ubuntu22.04"* ]]; then
        detected_os="ubuntu22.04"
    fi

    if [ -n "$detected_npu" ]; then
        DETECTED_NPU_TYPE="$detected_npu"
        echo ">>> Auto-detected NPU type from base image: $detected_npu"
    fi

    if [ -n "$detected_os" ]; then
        DETECTED_OS="$detected_os"
        echo ">>> Auto-detected OS from base image: $detected_os"
    fi

    return 0
}

DETECTED_NPU_TYPE=""
DETECTED_OS=""

if [ -n "$BASE_IMAGE" ]; then
    echo ">>> Auto-detecting NPU type and OS from base image..."
    parse_base_image_tag "$BASE_IMAGE"

    if [ "$NPU_TYPE_EXPLICIT" = false ] && [ -n "$DETECTED_NPU_TYPE" ]; then
        NPU_TYPE="$DETECTED_NPU_TYPE"
        echo ">>> Using auto-detected NPU type: $NPU_TYPE"
    fi

    if [ "$OS_EXPLICIT" = false ] && [ -n "$DETECTED_OS" ]; then
        OS="$DETECTED_OS"
        echo ">>> Using auto-detected OS: $OS"
    fi
fi

if [ -z "$NPU_TYPE" ]; then
    echo "Error: NPU type is required (-t or --npu-type) or provide --base-image for auto-detection"
    show_help
    exit 1
fi

NPU_TYPE=$(echo "$NPU_TYPE" | tr '[:lower:]' '[:upper:]')
NPU_TYPE_LOWER=$(echo "$NPU_TYPE" | tr '[:upper:]' '[:lower:]')
OS=$(echo "$OS" | tr '[:upper:]' '[:lower:]')

if [ "$NPU_TYPE" != "A3" ] && [ "$NPU_TYPE" != "910B" ]; then
    echo "Error: NPU type must be A3 or 910B"
    exit 1
fi

if [ "$OS" != "openeuler24.03" ] && [ "$OS" != "ubuntu22.04" ]; then
    echo "Error: OS must be openeuler24.03 or ubuntu22.04"
    exit 1
fi

OS_FAMILY=""
case "$OS" in
    openeuler*) OS_FAMILY="openeuler" ;;
    ubuntu*)    OS_FAMILY="ubuntu" ;;
esac

OS_NAME=""
case "$OS_FAMILY" in
    openeuler) OS_NAME="openEuler" ;;
    ubuntu)    OS_NAME="ubuntu" ;;
esac

REPO_SCRIPT=""
case "$OS_FAMILY" in
    openeuler) REPO_SCRIPT="configure_yum_repo.sh" ;;
    ubuntu)    REPO_SCRIPT="configure_apt_repo.sh" ;;
esac


if [ -n "$TORCH_WHL_PATH" ] && [ ! -f "$TORCH_WHL_PATH" ]; then
    echo "Error: torch .whl file not found: $TORCH_WHL_PATH"
    exit 1
fi
if [ -n "$TORCH_NPU_WHL_PATH" ] && [ ! -f "$TORCH_NPU_WHL_PATH" ]; then
    echo "Error: torch-npu .whl file not found: $TORCH_NPU_WHL_PATH"
    exit 1
fi
if [ -n "$TORCHVISION_WHL_PATH" ] && [ ! -f "$TORCHVISION_WHL_PATH" ]; then
    echo "Error: torchvision .whl file not found: $TORCHVISION_WHL_PATH"
    exit 1
fi
if [ -n "$TORCHAUDIO_WHL_PATH" ] && [ ! -f "$TORCHAUDIO_WHL_PATH" ]; then
    echo "Error: torchaudio .whl file not found: $TORCHAUDIO_WHL_PATH"
    exit 1
fi

DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found: $DOCKERFILE"
    exit 1
fi

MODEL_INSTALL_DIR="${SCRIPTS_DIR}/model_install"

if [ ! -d "$MODEL_INSTALL_DIR" ]; then
    echo "Error: Model install scripts directory not found: $MODEL_INSTALL_DIR"
    exit 1
fi

if [ -z "$MINICONDA_PATH" ]; then
    echo ">>> Miniconda path not specified, will auto-download..."
    DOWNLOAD_ARCH=$(uname -m)
    if [ "$DOWNLOAD_ARCH" = "x86_64" ]; then
        DOWNLOAD_ARCH="x86_64"
    elif [ "$DOWNLOAD_ARCH" = "aarch64" ]; then
        DOWNLOAD_ARCH="aarch64"
    else
        echo "Error: Unsupported architecture: $DOWNLOAD_ARCH"
        exit 1
    fi
    DOWNLOAD_SCRIPT="${COMMON_DIR}/download_miniconda.sh"

    if [ -f "$DOWNLOAD_SCRIPT" ]; then
        DOWNLOAD_DIR="${SCRIPT_DIR}/downloads"
        echo ">>> Auto-downloading Miniconda (${DOWNLOAD_ARCH})..."
        bash "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$DOWNLOAD_ARCH"

        MINICONDA_FILE="Miniconda3-py311_26.1.1-1-Linux-${DOWNLOAD_ARCH}.sh"
        MINICONDA_PATH="${DOWNLOAD_DIR}/${MINICONDA_FILE}"

        if [ ! -f "$MINICONDA_PATH" ]; then
            echo "Error: Miniconda installer not found after auto-download"
            exit 1
        fi
        echo ">>> Miniconda download complete: $MINICONDA_PATH"
    else
        echo "Error: Download script not found: $DOWNLOAD_SCRIPT"
        exit 1
    fi
fi

MINICONDA_NAME=$(basename "$MINICONDA_PATH")
IS_ARM=false
ARCH_NAME="x86_64"
if [[ "$MINICONDA_NAME" == *"aarch64"* ]]; then
    IS_ARM=true
    ARCH_NAME="aarch64"
fi

if [ -z "$IMAGE_NAME" ]; then
    IMAGE_NAME="mindspeed-mm:${MINDSPEED_MM_VERSION}-${NPU_TYPE_LOWER}-${OS}-py${PYTHON_VERSION}-${ARCH_NAME}"
fi

if [ ! -f "$MINICONDA_PATH" ]; then
    echo "Warning: Miniconda installer not found: $MINICONDA_PATH"
    DOWNLOAD_ARCH=$(uname -m)
    if [ "$IS_ARM" = true ]; then
        DOWNLOAD_ARCH="aarch64"
    fi
    DOWNLOAD_SCRIPT="${COMMON_DIR}/download_miniconda.sh"

    if [ -f "$DOWNLOAD_SCRIPT" ]; then
        echo ">>> Auto-downloading Miniconda (${DOWNLOAD_ARCH})..."
        DOWNLOAD_DIR=$(dirname "$MINICONDA_PATH")
        bash "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$DOWNLOAD_ARCH"
        MINICONDA_NAME=$(basename "$MINICONDA_PATH")
        if [ ! -f "$MINICONDA_PATH" ]; then
            echo "Error: Miniconda installer not found after auto-download"
            exit 1
        fi
        echo ">>> Miniconda download complete: $MINICONDA_PATH"
    else
        echo "Error: Download script not found: $DOWNLOAD_SCRIPT"
        exit 1
    fi
fi

if [ "$IS_ARM" = true ]; then
    if [ -z "$DECORD_DEPS_PATH" ]; then
        echo "Warning: decord dependencies directory required for ARM architecture"
        DOWNLOAD_SCRIPT="${COMMON_DIR}/download_decord_deps.sh"

        if [ -f "$DOWNLOAD_SCRIPT" ]; then
            echo ">>> Auto-downloading decord dependencies..."
            DECORD_DEPS_PATH="${SCRIPT_DIR}/decord_deps"
            bash "$DOWNLOAD_SCRIPT" "$DECORD_DEPS_PATH"
            if [ ! -d "$DECORD_DEPS_PATH" ]; then
                echo "Error: decord dependencies directory not found after auto-download"
                exit 1
            fi
            echo ">>> decord dependencies download complete: $DECORD_DEPS_PATH"
        else
            echo "Error: Download script not found: $DOWNLOAD_SCRIPT"
            exit 1
        fi
    fi
    if [ ! -d "$DECORD_DEPS_PATH" ]; then
        echo "Error: decord dependencies directory not found: $DECORD_DEPS_PATH"
        exit 1
    fi
fi

if [ -z "$DECORD_SCRIPT_PATH" ]; then
    DECORD_SCRIPT_PATH="${COMMON_DIR}/install_decord_on_arm.sh"
fi
if [ ! -f "$DECORD_SCRIPT_PATH" ]; then
    echo "Error: decord install script not found: $DECORD_SCRIPT_PATH"
    exit 1
fi

echo "=========================================="
echo "Build Configuration"
echo "=========================================="
echo "NPU Type:           ${NPU_TYPE}"
echo "OS:                 ${OS}"
echo "CPU Architecture:   ${ARCH_NAME}"
echo "Dockerfile:         ${DOCKERFILE}"
echo "Image Name:         ${IMAGE_NAME}"
echo "Base Image Version: ${BASE_IMAGE_VERSION}"
echo "PyTorch Version:    ${TORCH_VERSION}"
echo "torch-npu Version:  ${TORCH_NPU_VERSION}"
echo "MindSpeed MM Ver:   ${MINDSPEED_MM_VERSION}"
echo "Model Scripts Dir:  ${MODEL_INSTALL_DIR}"
if [ -n "$TORCH_WHL_PATH" ] && [ -n "$TORCH_NPU_WHL_PATH" ]; then
    echo "Install Mode:       Offline (.whl)"
elif [ -n "$TORCH_WHL_PATH" ]; then
    echo "Install Mode:       Mixed (torch offline, torch-npu online)"
elif [ -n "$TORCH_NPU_WHL_PATH" ]; then
    echo "Install Mode:       Mixed (torch online, torch-npu offline)"
else
    echo "Install Mode:       Online (pip)"
fi
echo "Miniconda:          ${MINICONDA_PATH}"
echo "Decord Script:      ${DECORD_SCRIPT_PATH}"
if [ -n "$DECORD_DEPS_PATH" ]; then
    echo "Decord Deps:        ${DECORD_DEPS_PATH}"
fi
echo "No Cache:           ${NO_CACHE:-No}"
echo "=========================================="

if [ -n "$BASE_IMAGE" ]; then
    echo ""
    echo ">>> Checking if base image exists..."
    if ! docker image inspect "$BASE_IMAGE" > /dev/null 2>&1; then
        echo ">>> Base image not found, pulling: ${BASE_IMAGE}"
        docker pull "$BASE_IMAGE"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to pull base image"
            exit 1
        fi
    else
        echo ">>> Base image already exists: ${BASE_IMAGE}"
    fi
    echo ""
fi

cd "$SCRIPT_DIR"

cp "$MINICONDA_PATH" .

DECORD_SCRIPT_NAME=$(basename "$DECORD_SCRIPT_PATH")
cp "$DECORD_SCRIPT_PATH" .

cp "${COMMON_DIR}/common_functions.sh" .

cp "${COMMON_DIR}/${REPO_SCRIPT}" configure_repo.sh

mkdir -p install_scripts
for script in "${MODEL_INSTALL_DIR}"/install_*.sh; do
    cp "$script" install_scripts/
done

DECORD_DEPS_NAME=""
DECORD_DEPS_COPIED=false
if [ -n "$DECORD_DEPS_PATH" ]; then
    DECORD_DEPS_NAME=$(basename "$DECORD_DEPS_PATH")
    DECORD_DEPS_REAL=$(realpath "$DECORD_DEPS_PATH")
    CURRENT_DIR_REAL=$(realpath .)
    if [ "$DECORD_DEPS_REAL" != "${CURRENT_DIR_REAL}/${DECORD_DEPS_NAME}" ]; then
        cp -r "$DECORD_DEPS_PATH" .
        DECORD_DEPS_COPIED=true
    fi
fi

mkdir -p torch_wheels
touch torch_wheels/.placeholder
if [ -n "$TORCH_WHL_PATH" ]; then
    cp "$TORCH_WHL_PATH" torch_wheels/
fi
if [ -n "$TORCH_NPU_WHL_PATH" ]; then
    cp "$TORCH_NPU_WHL_PATH" torch_wheels/
fi
if [ -n "$TORCHVISION_WHL_PATH" ]; then
    cp "$TORCHVISION_WHL_PATH" torch_wheels/
fi
if [ -n "$TORCHAUDIO_WHL_PATH" ]; then
    cp "$TORCHAUDIO_WHL_PATH" torch_wheels/
fi

BUILD_ARGS="--build-arg MINICONDA_SH=${MINICONDA_NAME}"
BUILD_ARGS="$BUILD_ARGS --build-arg DECORD_SCRIPT=${DECORD_SCRIPT_NAME}"
BUILD_ARGS="$BUILD_ARGS --build-arg OS=${OS}"
BUILD_ARGS="$BUILD_ARGS --build-arg OS_FAMILY=${OS_FAMILY}"
BUILD_ARGS="$BUILD_ARGS --build-arg NPU_TYPE=${NPU_TYPE_LOWER}"
BUILD_ARGS="$BUILD_ARGS --build-arg TORCH_VERSION=${TORCH_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg TORCH_NPU_VERSION=${TORCH_NPU_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg TORCH_WHL_DIR=torch_wheels"

if [ -n "$BASE_IMAGE" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg BASE_IMAGE=${BASE_IMAGE}"
else
    BUILD_ARGS="$BUILD_ARGS --build-arg BASE_IMAGE_VERSION=${BASE_IMAGE_VERSION}"
fi

if [ -n "$DECORD_DEPS_NAME" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg DECORD_DEPS_DIR=${DECORD_DEPS_NAME}"
fi

echo ""
echo "Starting image build..."
echo ""

# Temporarily disable set -e to handle build failure gracefully
set +e

docker build \
    -t "$IMAGE_NAME" \
    -f "$DOCKERFILE" \
    $BUILD_ARGS \
    $NO_CACHE \
    --network=host \
    .

BUILD_RESULT=$?

# Restore set -e
set -e

# Clean up temporary files regardless of build result
rm -f "${MINICONDA_NAME}"
rm -f "${DECORD_SCRIPT_NAME}"
rm -f "common_functions.sh"
rm -f "configure_repo.sh"
rm -rf "install_scripts"
rm -rf "torch_wheels"
if [ -n "$DECORD_DEPS_NAME" ] && [ "$DECORD_DEPS_COPIED" = true ]; then
    rm -rf "${DECORD_DEPS_NAME}"
fi

# Check build result and handle accordingly
if [ $BUILD_RESULT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Build Complete!"
    echo "Image: ${IMAGE_NAME}"
    echo "=========================================="
    echo ""
    echo "Usage:"
    echo "  docker run -it --rm ${IMAGE_NAME} bash"
    echo ""
    exit 0
else
    echo ""
    echo "=========================================="
    echo "Build Failed!"
    echo "=========================================="
    if [ "$CLEANUP_ON_FAIL" = true ]; then
        echo ""
        echo ">>> Cleaning up dangling images and containers..."
        cleanup_dangling
    fi
    exit $BUILD_RESULT
fi
