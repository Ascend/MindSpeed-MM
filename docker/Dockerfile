# ============================================
# MindSpeed-MM Docker Image Build File
#
# Supported NPU types (configurable): 910b, a3, etc.
# Supported CPU architectures:        x86_64 and aarch64 (ARM)
# Supported operating systems:        openEuler and Ubuntu
#
# Build targets:
#   - Default dev image (BUILD_CI=false): Installs PyTorch/NPU, verl_qwen3vl,
#     and clones the MindSpeed-MM code.
#   - CI image (BUILD_CI=true): Extends the dev image by creating multiple conda
#     environments, one per MindSpeed-MM release branch.
#
# Multi-stage build:
#   Stage 1 (base): Install system deps, Miniconda, PyTorch, torch_npu, etc.
#   Stage 2 (final): Install decord, CI multi-version envs or verl_qwen3vl,
#                    then produce the final image.
# ============================================

# --- Base image arguments ---
# OS: Operating system type, e.g. openeuler24.03 or ubuntu22.04
ARG OS=openeuler24.03
# BASE_IMAGE_VERSION: CANN base image version tag
ARG BASE_IMAGE_VERSION=9.0.0-beta.2
# NPU_TYPE: Ascend NPU model (910b / a3 / etc.)
ARG NPU_TYPE=910b
# BASE_IMAGE: Full base image URL; the default splicing rule is used when empty
ARG BASE_IMAGE=""

# Stage 1 (base): Build the base runtime environment.
# Pulls the CANN base image from the Huawei Cloud Ascend registry by default;
# comes with CANN and Python 3.11 pre-installed.
FROM ${BASE_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${BASE_IMAGE_VERSION}-${NPU_TYPE}-${OS}-py3.11} AS base

# Switch to root (the CANN base image defaults to a non-root user)
USER root

# --- Build arguments (used in the base stage only) ---
# MINICONDA_SH: Path to the Miniconda installer script (passed via --build-arg)
ARG MINICONDA_SH
# TORCH_VERSION / TORCH_NPU_VERSION: PyTorch and torch_npu versions
ARG TORCH_VERSION=2.7.1
ARG TORCH_NPU_VERSION=2.7.1
# TORCH_WHL_DIR: Directory containing local PyTorch .whl files (optional, for offline install)
ARG TORCH_WHL_DIR=torch_wheels
# OS_FAMILY: Operating system family (openeuler / ubuntu), used to select the package manager
ARG OS_FAMILY

# Detect CPU architecture and print info (x86_64 / aarch64)
RUN ARCH=$(uname -m) && \
    echo "==========================================" && \
    echo "Detected CPU architecture: ${ARCH}" && \
    if [ "$ARCH" = "x86_64" ]; then \
        echo "Architecture type: x86"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        echo "Architecture type: ARM"; \
    else \
        echo "ERROR: Unsupported architecture: $ARCH"; \
        exit 1; \
    fi && \
    echo "=========================================="

# Configure DNS resolution (ensures the container can access the public internet).
# Uses: domestic DNS (114) -> Google DNS (8.8.8.8) -> Alibaba DNS (223.5.5.5)
RUN if [ -f /etc/resolv.conf ]; then \
        echo "Existing resolv.conf:" && cat /etc/resolv.conf; \
    else \
        echo "No resolv.conf found, creating one..."; \
    fi && \
    echo "nameserver 114.114.114.114" > /etc/resolv.conf && \
    echo "nameserver 8.8.8.8" >> /etc/resolv.conf && \
    echo "nameserver 223.5.5.5" >> /etc/resolv.conf && \
    echo "Updated resolv.conf:" && cat /etc/resolv.conf

# Configure system package repositories (via external script for yum/apt sources)
COPY configure_repo.sh /tmp/configure_repo.sh
RUN chmod +x /tmp/configure_repo.sh && \
    bash /tmp/configure_repo.sh && \
    rm /tmp/configure_repo.sh

# Install system-level dependencies.
# - Build toolchain: gcc/g++, make, cmake (for compiling C++ extensions like decord)
# - Network tools:   git, wget, curl
# - Media processing: ffmpeg and related dev libraries (for video decoding)
RUN echo "Installing system dependencies..." && \
    if [ "$OS_FAMILY" = "openeuler" ]; then \
        yum install -y \
            git \
            iproute \
            wget \
            curl \
            vim \
            gcc \
            gcc-c++ \
            make \
            cmake \
            ffmpeg-devel \
        && yum clean all; \
    elif [ "$OS_FAMILY" = "ubuntu" ]; then \
        apt-get install -y \
            git \
            iproute2 \
            wget \
            curl \
            vim \
            gcc \
            g++ \
            make \
            cmake \
            ffmpeg \
            libavcodec-dev \
            libavformat-dev \
            libavutil-dev \
            libswscale-dev \
        && apt-get clean; \
    else \
        echo "ERROR: Unsupported OS: $OS_FAMILY"; \
        exit 1; \
    fi

# Install Miniconda (for creating and managing Python virtual environments).
# -b: silent install (no user prompts)
# -p /opt/conda: target installation path
COPY ${MINICONDA_SH} /tmp/miniconda.sh
RUN echo "Installing Miniconda..." && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# Add conda to PATH and disable auto-update (avoids errors in non-interactive sessions)
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Configure pip to use Huawei Cloud PyPI mirror (faster downloads in China) and init conda
RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && \
    pip config set global.trusted-host "repo.huaweicloud.com" && \
    # Initialize conda for bash shell (auto-activates the base env on container start)
    /opt/conda/bin/conda init bash

# Copy local PyTorch wheel files (preferred when provided for offline installation)
COPY ${TORCH_WHL_DIR} /tmp/torch_wheels/

# Install PyTorch and torch_npu.
# Prefers local .whl files (offline mode); falls back to online PyPI installation.
# Online install retries up to 3 times to tolerate network instability.
# - x86_64: Installs from the official PyTorch CPU index (better stability).
# - aarch64: Installs from the default PyPI index.
RUN if ls /tmp/torch_wheels/*.whl 1>/dev/null 2>&1; then \
        echo "Installing PyTorch and torch_npu from local .whl files..." && \
        ls /tmp/torch_wheels/*.whl && \
        pip install --no-cache-dir /tmp/torch_wheels/*.whl; \
    else \
        echo "Installing PyTorch and torch_npu from PyPI..." && \
        ARCH=$(uname -m) && \
        MAX_RETRIES=3 && \
        for retry in $(seq 1 $MAX_RETRIES); do \
            echo ">>> Attempt $retry of $MAX_RETRIES"; \
            pip cache purge 2>/dev/null || true; \
            rm -rf /root/.cache/pip; \
            if [ "$ARCH" = "x86_64" ]; then \
                echo "Installing PyTorch for x86_64..." && \
                pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && break; \
            elif [ "$ARCH" = "aarch64" ]; then \
                echo "Installing PyTorch for aarch64..." && \
                pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio && break; \
            fi; \
        done && \
        # Install torch_npu
        for retry in $(seq 1 $MAX_RETRIES); do \
            echo ">>> Installing torch-npu, attempt $retry of $MAX_RETRIES"; \
            pip cache purge 2>/dev/null || true; \
            rm -rf /root/.cache/pip; \
            pip install --no-cache-dir torch-npu==${TORCH_NPU_VERSION} && break; \
        done; \
    fi && \
    rm -rf /tmp/torch_wheels

# Install decord (video decoding library) via pip on x86_64.
# On aarch64, decord must be built from source — handled in Stage 2 (final).
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        echo "Installing decord for x86_64..." && \
        MAX_RETRIES=3 && \
        for retry in $(seq 1 $MAX_RETRIES); do \
            echo ">>> Attempt $retry of $MAX_RETRIES"; \
            pip cache purge 2>/dev/null || true; \
            rm -rf /root/.cache/pip; \
            pip install --no-cache-dir decord==0.6.0 && break; \
        done; \
    fi

# Clean conda cache to reduce image size
RUN conda clean -ya

# ========================================
# Stage 2 (final): Produce the final image
# ========================================
FROM base AS final

# MINDSPEED_MM_BRANCH: MindSpeed-MM branch to clone (used in non-CI mode)
ARG MINDSPEED_MM_BRANCH=26.1.0
# BUILD_CI: Whether to build a CI multi-version image (true = CI / false = dev)
ARG BUILD_CI=false
# FINAL_WORKDIR: working directory of the final image
#   (/workspace for CI, /workspace/MindSpeed-MM for dev; overridden by build.sh for CI)
ARG FINAL_WORKDIR=/workspace/MindSpeed-MM
ENV FINAL_WORKDIR=${FINAL_WORKDIR}

WORKDIR /workspace

# Parameters for building decord from source (aarch64 only)
ARG DECORD_SCRIPT
ARG DECORD_DEPS_DIR=decord_deps

# Copy the decord build script and pre-downloaded dependencies
COPY ${DECORD_SCRIPT} /tmp/install_decord.sh
RUN chmod +x /tmp/install_decord.sh

COPY ${DECORD_DEPS_DIR} /tmp/decord_deps/

# On aarch64: build and install decord from source using pre-downloaded deps.
# On x86_64: decord was already installed via pip in Stage 1 — skip.
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        echo "Installing decord for ARM architecture using pre-downloaded dependencies..." && \
        bash /tmp/install_decord.sh /tmp/decord_deps; \
    else \
        echo "Skipping decord build for x86_64 (already installed via pip)"; \
    fi && \
    rm -rf /tmp/decord_deps && \
    rm -f /tmp/install_decord.sh

# ========================================
# CI environment build (conditional, only active when BUILD_CI=true)
# ========================================
# Install tomli to parse pyproject.toml from each branch
RUN pip install tomli

# Copy CI build scripts and config files into the image.
# Note: the docker build context is the docker/ directory, so relative paths start from docker/.
COPY scripts/ci/ /tmp/ci_scripts/

# When BUILD_CI=true: run build_ci_envs.py to create multi-version conda environments.
# When BUILD_CI=false (default): skip this step.
RUN if [ "$BUILD_CI" = "true" ]; then \
      echo ">>> Building CI multi-version environments..." && \
      python /tmp/ci_scripts/build_ci_envs.py && \
      rm -rf /tmp/ci_scripts; \
    fi

# ========================================
# Pre-download CI test dependencies (transformers, Megatron-LM)
# Cloned during image build to avoid network instability at test time.
# ========================================
RUN if [ "$BUILD_CI" = "true" ]; then \
      echo ">>> Pre-downloading transformers (commit 7a833d1)..." && \
      git clone https://github.com/huggingface/transformers.git /workspace/transformers && \
      cd /workspace/transformers && \
      git checkout 7a833d1 && \
      cd /workspace && \
      mv transformers transformers-7a833d1 && \
      echo ">>> Pre-downloading Megatron-LM (branch core_v0.12.1)..." && \
      git clone https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM && \
      cd /workspace/Megatron-LM && \
      git checkout core_v0.12.1 && \
      cd /workspace && \
      mv Megatron-LM Megatron-LM_core_v0.12.1; \
    fi

# ========================================
# Non-CI scenario (BUILD_CI=false, the default dev image)
# ========================================
# NOTE: COPY is a Docker build-time instruction and cannot be placed inside a
#       RUN if block, so we copy unconditionally and conditionally execute.
COPY common_functions.sh /tmp/common_functions.sh
COPY install_scripts/install_verl_qwen3vl.sh /tmp/install_verl_qwen3vl.sh

# Install verl_qwen3vl (the verl RL training framework for Qwen3VL models)
RUN if [ "$BUILD_CI" != "true" ]; then \
      chmod +x /tmp/install_verl_qwen3vl.sh && \
      bash /tmp/install_verl_qwen3vl.sh "$MINDSPEED_MM_BRANCH"; \
    fi && \
    rm -f /tmp/common_functions.sh /tmp/install_verl_qwen3vl.sh

# Clone the MindSpeed-MM repository (non-CI mode)
RUN if [ "$BUILD_CI" != "true" ]; then \
      echo ">>> Cloning MindSpeed MM..." && \
      git clone --branch ${MINDSPEED_MM_BRANCH} https://gitcode.com/Ascend/MindSpeed-MM.git; \
    fi

# Final cleanup: remove conda cache, pip cache, and temp files to shrink the image
RUN conda clean -ya && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/*

# Set the default working directory (CI: /workspace, dev: /workspace/MindSpeed-MM)
WORKDIR ${FINAL_WORKDIR}

# Configure bash startup: cd into the working directory and activate the conda base env
RUN echo "cd ${FINAL_WORKDIR}" >> /root/.bashrc && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /root/.bashrc && \
    echo 'conda activate base' >> /root/.bashrc

# Ensure conda is in PATH (overwrites any previously reset env var)
ENV PATH=/opt/conda/bin:$PATH
# CANN Ascend Toolkit installation path (adjust for your actual CANN version)
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit
