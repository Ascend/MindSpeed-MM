#!/bin/bash
set -e

OUTPUT_DIR="${1:-./decord_deps}"

# Function to verify file integrity using SHA256 checksum
verify_checksum() {
    local file_path="$1"
    local expected_checksum="$2"
    local file_name="$(basename "$file_path")"

    if [ ! -f "$file_path" ]; then
        echo "Error: File $file_name not found!"
        return 1
    fi

    echo ">>> Verifying $file_name integrity..."
    local actual_checksum=$(sha256sum "$file_path" | awk '{print $1}')

    if [ "$actual_checksum" != "$expected_checksum" ]; then
        echo "Error: Checksum verification failed for $file_name!"
        echo "  Expected: $expected_checksum"
        echo "  Actual:   $actual_checksum"
        return 1
    fi

    echo ">>> Checksum verification passed for $file_name"
    return 0
}

echo "=========================================="
echo "Downloading decord dependencies for ARM"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

CMAKE_VERSION="3.19.3"
CMAKE_CHECKSUM="a398b284396db530a3d584480a303c061b5e5e57d531d042e321dbb7106be223"

echo ">>> Downloading cmake ${CMAKE_VERSION}..."
if [ ! -f "cmake-${CMAKE_VERSION}-linux-aarch64.sh" ] || ! verify_checksum "cmake-${CMAKE_VERSION}-linux-aarch64.sh" "$CMAKE_CHECKSUM"; then
    wget -c "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-aarch64.sh"
    verify_checksum "cmake-${CMAKE_VERSION}-linux-aarch64.sh" "$CMAKE_CHECKSUM"
fi

NASM_CHECKSUM="5bc940dd8a4245686976a8f7e96ba9340a0915f2d5b88356874890e207bdb581"
echo ">>> Downloading nasm 2.16.03..."
if [ ! -f "nasm-2.16.03.tar.gz" ] || ! verify_checksum "nasm-2.16.03.tar.gz" "$NASM_CHECKSUM"; then
    wget -c https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/nasm-2.16.03.tar.gz
    verify_checksum "nasm-2.16.03.tar.gz" "$NASM_CHECKSUM"
fi

YASM_CHECKSUM="3dce6601b495f5b3d45b59f7d2492a340ee7e84b5beca17e48f862502bd5603f" # Example checksum
echo ">>> Downloading yasm 1.3.0..."
if [ ! -f "yasm-1.3.0.tar.gz" ] || ! verify_checksum "yasm-1.3.0.tar.gz" "$YASM_CHECKSUM"; then
    wget -c https://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
    verify_checksum "yasm-1.3.0.tar.gz" "$YASM_CHECKSUM"
fi

echo ">>> Cloning x264..."
if [ ! -d "x264" ]; then
    git clone --depth 1 https://github.com/mirror/x264.git
fi

echo ">>> Cloning libvpx..."
if [ ! -d "libvpx" ]; then
    git clone --depth 1 https://chromium.googlesource.com/webm/libvpx.git
fi

FFMPEG_CHECKSUM="f98a482520c47507521a907914daa9efbc1384e0591b5afc3da18aa897de2948"
echo ">>> Downloading ffmpeg 4.4.2..."
if [ ! -f "ffmpeg-4.4.2.tar.bz2" ] || ! verify_checksum "ffmpeg-4.4.2.tar.bz2" "$FFMPEG_CHECKSUM"; then
    wget -c https://ffmpeg.org/releases/ffmpeg-4.4.2.tar.bz2
    verify_checksum "ffmpeg-4.4.2.tar.bz2" "$FFMPEG_CHECKSUM"
fi

echo ">>> Cloning decord..."
if [ ! -d "decord" ]; then
    git clone --recursive https://github.com/dmlc/decord.git
fi

echo "=========================================="
echo "All dependencies downloaded successfully!"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""
echo "Directory contents:"
ls -lh "${OUTPUT_DIR}"
