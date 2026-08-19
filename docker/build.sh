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

# Dockerfile naming: Dockerfile (dev image), Dockerfile.ci (CI image)
# Dev image tag:  {version}-cann{cann}-torch_npu{torch_npu}-{chip}-{os}-py{python}-{arch}
# CI image tag:   {dev_tag}-ci  (built on top of the dev image)
#
# CANN version, NPU type (910b/a3/950), OS and architecture are auto-detected
# from the base image tag. The Python version defaults to the Python detected
# from the base image tag, but can be overridden with --python-version, which
# controls the Miniconda variant and hence the conda base environment.
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="${SCRIPT_DIR}/common"

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build MindSpeed MM Docker Image

Required:
    --base-image IMAGE       Full base image name (e.g. swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.11)
                             CANN version, NPU type (910b/a3/950), OS and architecture are auto-detected from the image tag.

Optional:
    --python-version VER     Python version for the conda base environment (e.g. 3.11, 3.10, 3.12).
                             Selects the matching Miniconda installer. Defaults to the Python version
                             detected from the base image tag.
    --tag TAG                Custom image tag (default: auto-generated from build info; CI appends '-ci')
    -v, --version VERSION    MindSpeed MM version (default: 26.1.0; branch used to clone the repo)
    --torch-version VER      PyTorch version (default: 2.7.1, online install)
    --torch-npu-version VER  torch-npu version (default: 2.7.1.post8, online install)
    -n, --no-cache           Build without cache
    --build-ci               Build CI image on top of the dev image (multi-version conda environments)
    --cleanup-on-fail        Clean up dangling images/containers if build fails
    -h, --help               Show help

Examples:
    bash $0 --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-a3-openeuler24.03-py3.11
    bash $0 --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.11 --torch-version 2.7.1 --torch-npu-version 2.7.1.post8
    bash $0 --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-openeuler24.03-py3.12 --python-version 3.12
    bash $0 --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.11 --build-ci
EOF
}

IMAGE_TAG=""
NO_CACHE=""
TORCH_VERSION="2.7.1"
TORCH_NPU_VERSION="2.7.1.post8"
BASE_IMAGE=""
MINDSPEED_MM_VERSION="26.1.0"
CLEANUP_ON_FAIL=false
BUILD_CI=false
PYTHON_VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)               IMAGE_TAG="$2"; shift 2 ;;
        -n|--no-cache)       NO_CACHE="--no-cache"; shift ;;
        -v|--version)        MINDSPEED_MM_VERSION="$2"; shift 2 ;;
        --torch-version)     TORCH_VERSION="$2"; shift 2 ;;
        --torch-npu-version) TORCH_NPU_VERSION="$2"; shift 2 ;;
        --base-image)        BASE_IMAGE="$2"; shift 2 ;;
        --python-version)    PYTHON_VERSION="$2"; shift 2 ;;
        --build-ci)          BUILD_CI=true; shift ;;
        --cleanup-on-fail)   CLEANUP_ON_FAIL=true; shift ;;
        -h|--help)           show_help; exit 0 ;;
        *)                   echo "Unknown argument: $1"; show_help; exit 1 ;;
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
    elif [[ "$tag_lower" == *"950"* ]]; then
        detected_npu="950"
    elif [[ "$tag_lower" == *"-a3-"* ]] || [[ "$tag_lower" == *"-a3-py"* ]]; then
        detected_npu="A3"
    fi

    local detected_os=""
    if [[ "$tag_lower" == *"openeuler24.03"* ]]; then
        detected_os="openeuler24.03"
    elif [[ "$tag_lower" == *"ubuntu22.04"* ]]; then
        detected_os="ubuntu22.04"
    fi

    local detected_cann=""
    if [ -n "$detected_npu" ]; then
        local npu_lower=$(echo "$detected_npu" | tr '[:upper:]' '[:lower:]')
        detected_cann="${tag_lower%%-${npu_lower}-*}"
    else
        detected_cann="${tag_lower%%-*}"
    fi

    # Python version: match 'py<x.y>' field in the tag (e.g. py3.11 -> 3.11)
    local detected_python=""
    if [[ "$tag_lower" =~ py([0-9]+\.[0-9]+) ]]; then
        detected_python="${BASH_REMATCH[1]}"
    fi

    if [ -n "$detected_npu" ]; then
        DETECTED_NPU_TYPE="$detected_npu"
        echo ">>> Auto-detected NPU type from base image: $detected_npu"
    fi

    if [ -n "$detected_cann" ]; then
        DETECTED_CANN_VERSION="$detected_cann"
        echo ">>> Auto-detected CANN version from base image: $detected_cann"
    fi

    if [ -n "$detected_os" ]; then
        DETECTED_OS="$detected_os"
        echo ">>> Auto-detected OS from base image: $detected_os"
    fi

    if [ -n "$detected_python" ]; then
        DETECTED_PYTHON_VERSION="$detected_python"
        echo ">>> Auto-detected Python version from base image: $detected_python"
    fi

    return 0
}

DETECTED_NPU_TYPE=""
DETECTED_OS=""
DETECTED_CANN_VERSION=""
DETECTED_PYTHON_VERSION=""

if [ -z "$BASE_IMAGE" ]; then
    echo "Error: --base-image is required."
    show_help
    exit 1
fi

echo ">>> Auto-detecting NPU type, OS, CANN version and Python version from base image..."
parse_base_image_tag "$BASE_IMAGE"

if [ -z "$DETECTED_NPU_TYPE" ]; then
    echo "Error: Failed to detect NPU type from base image tag (expected '910b', '950' or 'a3')."
    exit 1
fi
NPU_TYPE="$DETECTED_NPU_TYPE"

if [ -z "$DETECTED_OS" ]; then
    echo "Error: Failed to detect OS from base image tag (expected 'openeuler24.03' or 'ubuntu22.04')."
    exit 1
fi
OS="$DETECTED_OS"

if [ -z "$DETECTED_CANN_VERSION" ]; then
    echo "Error: Failed to detect CANN version from base image tag."
    exit 1
fi
CANN_VERSION="$DETECTED_CANN_VERSION"

# Resolve Python version: user-specified takes precedence, else auto-detected.
if [ -z "$PYTHON_VERSION" ]; then
    if [ -z "$DETECTED_PYTHON_VERSION" ]; then
        echo "Error: Failed to detect Python version from base image tag (expected 'py<x.y>', e.g. py3.11)."
        echo "       Use --python-version to specify it explicitly."
        exit 1
    fi
    PYTHON_VERSION="$DETECTED_PYTHON_VERSION"
fi

NPU_TYPE=$(echo "$NPU_TYPE" | tr '[:lower:]' '[:upper:]')
NPU_TYPE_LOWER=$(echo "$NPU_TYPE" | tr '[:upper:]' '[:lower:]')
OS=$(echo "$OS" | tr '[:upper:]' '[:lower:]')

if [ "$NPU_TYPE" != "A3" ] && [ "$NPU_TYPE" != "910B" ] && [ "$NPU_TYPE" != "950" ]; then
    echo "Error: NPU type must be A3, 910B or 950"
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

REPO_SCRIPT=""
case "$OS_FAMILY" in
    openeuler) REPO_SCRIPT="configure_yum_repo.sh" ;;
    ubuntu)    REPO_SCRIPT="configure_apt_repo.sh" ;;
esac

# CPU architecture: used for Miniconda/decord selection and the image tag.
DOWNLOAD_ARCH=$(uname -m)
if [ "$DOWNLOAD_ARCH" = "aarch64" ]; then
    IS_ARM=true
    ARCH_NAME="aarch64"
else
    IS_ARM=false
    ARCH_NAME="x86_64"
fi
# aarch64 builds decord from source in the decord-builder stage.
DECORD_BUILD=$([ "$IS_ARM" = true ] && echo "true" || echo "false")

DOCKERFILE="${SCRIPT_DIR}/Dockerfile"
CI_DOCKERFILE="${SCRIPT_DIR}/Dockerfile.ci"

if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found: $DOCKERFILE"
    exit 1
fi
if [ "$BUILD_CI" = true ] && [ ! -f "$CI_DOCKERFILE" ]; then
    echo "Error: Dockerfile.ci not found: $CI_DOCKERFILE"
    exit 1
fi

# Auto-download Miniconda (variant selected by the requested Python version).
echo ">>> Auto-downloading Miniconda (${DOWNLOAD_ARCH}, Python ${PYTHON_VERSION})..."
DOWNLOAD_SCRIPT="${COMMON_DIR}/download_miniconda.sh"
DOWNLOAD_DIR="${SCRIPT_DIR}/downloads"
mkdir -p "$DOWNLOAD_DIR"
MINICONDA_FILE=$(bash "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$DOWNLOAD_ARCH" "$PYTHON_VERSION" | tail -n 1)
MINICONDA_PATH="${DOWNLOAD_DIR}/${MINICONDA_FILE}"
if [ ! -f "$MINICONDA_PATH" ]; then
    echo "Error: Miniconda installer not found after auto-download: $MINICONDA_PATH"
    exit 1
fi
echo ">>> Miniconda download complete: $MINICONDA_PATH"

MINICONDA_NAME=$(basename "$MINICONDA_PATH")

# ARM: auto-download decord source dependencies.
if [ "$IS_ARM" = true ]; then
    echo ">>> Auto-downloading decord dependencies..."
    DECORD_DEPS_PATH="${SCRIPT_DIR}/decord_deps"
    bash "${COMMON_DIR}/download_decord_deps.sh" "$DECORD_DEPS_PATH"
    if [ ! -d "$DECORD_DEPS_PATH" ]; then
        echo "Error: decord dependencies directory not found after auto-download"
        exit 1
    fi
else
    # x86_64: decord is installed via pip in the base stage; still need a valid
    # build-context path for the Dockerfile COPY. Created under SCRIPT_DIR with
    # an absolute path so it lands in the build context regardless of the
    # caller's CWD (cd "$SCRIPT_DIR" runs later, at line ~340).
    mkdir -p "${SCRIPT_DIR}/decord_deps"
    touch "${SCRIPT_DIR}/decord_deps/.placeholder"
fi

DECORD_SCRIPT_PATH="${COMMON_DIR}/install_decord_on_arm.sh"
if [ ! -f "$DECORD_SCRIPT_PATH" ]; then
    echo "Error: decord install script not found: $DECORD_SCRIPT_PATH"
    exit 1
fi

# Image naming.
DEFAULT_TAG="v${MINDSPEED_MM_VERSION}-cann${CANN_VERSION}-torch_npu${TORCH_NPU_VERSION}-${NPU_TYPE_LOWER}-${OS}-py${PYTHON_VERSION}-${ARCH_NAME}"
if [ -n "$IMAGE_TAG" ]; then
    DEV_TAG="$IMAGE_TAG"
else
    DEV_TAG="$DEFAULT_TAG"
fi
DEV_IMAGE_NAME="mindspeed-mm:${DEV_TAG}"
if [ "$BUILD_CI" = true ]; then
    IMAGE_NAME="mindspeed-mm:${DEV_TAG}-ci"
else
    IMAGE_NAME="$DEV_IMAGE_NAME"
fi

echo "=========================================="
echo "Build Configuration"
echo "=========================================="
echo "Base Image:         ${BASE_IMAGE}"
echo "NPU Type:           ${NPU_TYPE}"
echo "OS:                 ${OS}"
echo "CANN Version:       ${CANN_VERSION}"
echo "Python Version:     ${PYTHON_VERSION}"
echo "CPU Architecture:   ${ARCH_NAME}"
echo "PyTorch Version:    ${TORCH_VERSION}"
echo "torch-npu Version:  ${TORCH_NPU_VERSION}"
echo "MindSpeed MM Ver:   ${MINDSPEED_MM_VERSION}"
echo "Build CI:           ${BUILD_CI}"
echo "No Cache:           ${NO_CACHE:-No}"
echo "=========================================="

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

cd "$SCRIPT_DIR"

cp "$MINICONDA_PATH" .

DECORD_SCRIPT_NAME=$(basename "$DECORD_SCRIPT_PATH")
cp "$DECORD_SCRIPT_PATH" .

cp "${COMMON_DIR}/${REPO_SCRIPT}" configure_repo.sh

BUILD_ARGS="--build-arg MINICONDA_SH=${MINICONDA_NAME}"
BUILD_ARGS="$BUILD_ARGS --build-arg DECORD_SCRIPT=${DECORD_SCRIPT_NAME}"
BUILD_ARGS="$BUILD_ARGS --build-arg TORCH_VERSION=${TORCH_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg TORCH_NPU_VERSION=${TORCH_NPU_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg MINDSPEED_MM_BRANCH=${MINDSPEED_MM_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg BASE_IMAGE=${BASE_IMAGE}"
BUILD_ARGS="$BUILD_ARGS --build-arg PYTHON_VERSION=${PYTHON_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg DECORD_DEPS_DIR=decord_deps"
BUILD_ARGS="$BUILD_ARGS --build-arg DECORD_BUILD=${DECORD_BUILD}"

echo ""
echo ">>> Building dev image: ${DEV_IMAGE_NAME}"
echo ""

set +e
docker build \
    -t "$DEV_IMAGE_NAME" \
    -f "$DOCKERFILE" \
    $BUILD_ARGS \
    $NO_CACHE \
    --network=host \
    .
BUILD_RESULT=$?
set -e

if [ $BUILD_RESULT -eq 0 ] && [ "$BUILD_CI" = true ]; then
    echo ""
    echo ">>> Building CI image on top of dev image: ${IMAGE_NAME}"
    echo ""
    set +e
    docker build \
        -t "$IMAGE_NAME" \
        -f "$CI_DOCKERFILE" \
        --build-arg DEV_IMAGE="$DEV_IMAGE_NAME" \
        $NO_CACHE \
        --network=host \
        .
    BUILD_RESULT=$?
    set -e
fi

# Clean up temporary build-context files regardless of build result.
rm -f "${MINICONDA_NAME}"
rm -f "${DECORD_SCRIPT_NAME}"
rm -f "configure_repo.sh"
rm -rf "decord_deps"

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
