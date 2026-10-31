#!/bin/bash
# Central dependency provisioning for dev/CI Docker build stages.
# runtime: final-image system packages (works before Miniconda is installed).
# build: ARM compilers, FFmpeg headers and Python build tools (requires Python).
set -euo pipefail
MODE="${1:-runtime}"
case "$MODE" in runtime|build) ;; *) echo "Invalid dependency mode: $MODE"; exit 1 ;; esac
. /etc/os-release
case "$(echo "$ID" | tr '[:upper:]' '[:lower:]')" in
    *openeuler*)
        if [ "$MODE" = runtime ]; then
            packages=(git iproute wget curl vim gawk ffmpeg)
        else
            packages=(gcc gcc-c++ make binutils pkgconf-pkg-config ffmpeg-devel
                autoconf automake bzip2 bzip2-devel freetype-devel libtool
                mercurial zlib-devel unzip patch)
        fi
        yum install -y "${packages[@]}"
        yum clean all
        ;;
    *ubuntu*)
        if [ "$MODE" = runtime ]; then
            packages=(ca-certificates git iproute2 wget curl vim gawk ffmpeg)
        else
            packages=(build-essential pkg-config binutils libavcodec-dev
                libavformat-dev libavutil-dev libavdevice-dev libavfilter-dev
                libswresample-dev libswscale-dev autoconf automake bzip2
                libbz2-dev libfreetype-dev libtool mercurial zlib1g-dev unzip patch)
        fi
        apt-get update
        apt-get install -y --no-install-recommends "${packages[@]}"
        apt-get clean
        ;;
    *) echo "ERROR: Unsupported OS ID: $ID"; exit 1 ;;
esac
if [ "$MODE" = build ]; then
    python -m pip install --no-cache-dir --retries 3 cmake ninja pybind11 setuptools wheel
fi
