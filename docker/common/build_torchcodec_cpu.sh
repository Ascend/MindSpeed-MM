#!/bin/bash
# Prepare a CPU-only TorchCodec wheel for the active Python environment.
# x86_64: official CPU wheel; aarch64: source build against system FFmpeg.
# Output: /torchcodec_wheels (consumed by the final Docker stage).
# ARM build reference: https://gitcode.com/MoCuishle-M/MindSpeed-MM_docker/issues/10
# Build dependencies are installed separately by install_dependencies.sh.
# Use a wheel rather than an editable install so the final image needs no source tree.
set -euo pipefail

TORCHCODEC_WHEEL_VERSION="${TORCHCODEC_WHEEL_VERSION:-0.10.0}"
TORCHCODEC_REF="${TORCHCODEC_REF:-v0.10.0-rc1}"
TORCHCODEC_VERSION="${TORCHCODEC_VERSION:-0.10.0rc1}"

case "$(uname -m)" in
    x86_64)
        python -m pip download --no-cache-dir --retries 3 --no-deps \
            --only-binary=:all: "torchcodec==${TORCHCODEC_WHEEL_VERSION}" \
            --index-url https://download.pytorch.org/whl/cpu \
            --dest /torchcodec_wheels
        exit 0
        ;;
    aarch64) ;;
    *) echo "ERROR: Unsupported architecture: $(uname -m)"; exit 1 ;;
esac

# The torchcodec-builder stage provides the compiler, Python build tools,
# pkg-config and FFmpeg development libraries before invoking this script.
git clone --depth 1 --branch "${TORCHCODEC_REF}" \
    https://github.com/pytorch/torchcodec.git /tmp/torchcodec
export ENABLE_CUDA=0 TORCHCODEC_BUILD_WEBP=0 TORCHCODEC_BUILD_HEIC=0
export BUILD_VERSION="${TORCHCODEC_VERSION}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
# Match the installation guide without importing torch during shell setup.
CONDA_PREFIX="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())'):${CMAKE_PREFIX_PATH:-}"
pkg-config --modversion libavcodec libavformat libavutil libswresample
# Upstream requires this opt-in for wheels dynamically linked to system FFmpeg.
# FFmpeg is a runtime dependency and is not bundled in the generated wheel.
# setup.py imports torch for its C++ extension build. Do not auto-load
# torch_npu here: Docker build has no host Ascend driver libraries.
TORCH_DEVICE_BACKEND_AUTOLOAD=0 I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
    python -m pip wheel /tmp/torchcodec --no-build-isolation --no-deps \
        --wheel-dir /torchcodec_wheels
python -m zipfile -e /torchcodec_wheels/*.whl /tmp/torchcodec-wheel-check
find /tmp/torchcodec-wheel-check -name 'libtorchcodec*.so' -exec readelf -d {} + > /tmp/torchcodec-needed.txt
test -s /tmp/torchcodec-needed.txt
if grep -Ei 'NEEDED.*(cuda|nvrtc)' /tmp/torchcodec-needed.txt; then
    echo "ERROR: TorchCodec unexpectedly links CUDA"
    exit 1
fi
