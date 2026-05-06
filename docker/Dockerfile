# ============================================
# MindSpeed MM Docker Image
# NPU Type: Configurable (910b, a3, etc.)
# Supports: x86_64 and aarch64 (ARM)
# Supports: openEuler and Ubuntu
# ============================================

ARG OS=openeuler24.03
ARG BASE_IMAGE_VERSION=9.0.0-beta.2
ARG NPU_TYPE=910b
ARG BASE_IMAGE=""

FROM ${BASE_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${BASE_IMAGE_VERSION}-${NPU_TYPE}-${OS}-py3.11} AS base

USER root

ARG MINICONDA_SH
ARG TORCH_VERSION=2.7.1
ARG TORCH_NPU_VERSION=2.7.1
ARG TORCH_WHL_DIR=torch_wheels
ARG OS_FAMILY

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

RUN if [ -f /etc/resolv.conf ]; then \
        echo "Existing resolv.conf:" && cat /etc/resolv.conf; \
    else \
        echo "No resolv.conf found, creating one..."; \
    fi && \
    echo "nameserver 114.114.114.114" > /etc/resolv.conf && \
    echo "nameserver 8.8.8.8" >> /etc/resolv.conf && \
    echo "nameserver 223.5.5.5" >> /etc/resolv.conf && \
    echo "Updated resolv.conf:" && cat /etc/resolv.conf

COPY configure_repo.sh /tmp/configure_repo.sh
RUN chmod +x /tmp/configure_repo.sh && \
    bash /tmp/configure_repo.sh && \
    rm /tmp/configure_repo.sh

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

COPY ${MINICONDA_SH} /tmp/miniconda.sh
RUN echo "Installing Miniconda..." && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && \
    pip config set global.trusted-host "repo.huaweicloud.com" && \
    # Initialize conda for bash shell (default in containers)
    /opt/conda/bin/conda init bash

COPY ${TORCH_WHL_DIR} /tmp/torch_wheels/

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
        for retry in $(seq 1 $MAX_RETRIES); do \
            echo ">>> Installing torch-npu, attempt $retry of $MAX_RETRIES"; \
            pip cache purge 2>/dev/null || true; \
            rm -rf /root/.cache/pip; \
            pip install --no-cache-dir torch-npu==${TORCH_NPU_VERSION} && break; \
        done; \
    fi && \
    rm -rf /tmp/torch_wheels

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

RUN conda clean -ya

FROM base AS final

ARG MINDSPEED_MM_BRANCH=master

WORKDIR /workspace

ARG DECORD_SCRIPT
ARG DECORD_DEPS_DIR

COPY ${DECORD_SCRIPT} /tmp/install_decord.sh
RUN chmod +x /tmp/install_decord.sh

COPY ${DECORD_DEPS_DIR} /tmp/decord_deps/

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        echo "Installing decord for ARM architecture using pre-downloaded dependencies..." && \
        bash /tmp/install_decord.sh /tmp/decord_deps; \
    else \
        echo "Skipping decord build for x86_64 (already installed via pip)"; \
    fi && \
    rm -rf /tmp/decord_deps && \
    rm -f /tmp/install_decord.sh

RUN echo ">>> Cloning MindSpeed MM..." && \
    git clone --branch ${MINDSPEED_MM_BRANCH} https://gitcode.com/Ascend/MindSpeed-MM.git

COPY common_functions.sh /tmp/common_functions.sh
COPY install_scripts/install_verl_qwen3vl.sh /tmp/install_verl_qwen3vl.sh
RUN chmod +x /tmp/install_verl_qwen3vl.sh && \
    bash /tmp/install_verl_qwen3vl.sh "$MINDSPEED_MM_BRANCH" && \
    rm -f /tmp/common_functions.sh /tmp/install_verl_qwen3vl.sh

RUN conda clean -ya && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/*

WORKDIR /workspace/MindSpeed-MM

RUN echo 'cd /workspace/MindSpeed-MM' >> /root/.bashrc && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /root/.bashrc && \
    echo 'conda activate base' >> /root/.bashrc

ENV PATH=/opt/conda/bin:$PATH
# 根据实际情况修改 ascend-toolkit 路径
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit
