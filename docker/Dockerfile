# ============================================
# MindSpeed MM Docker Image Build File (dev image)
#
# Supported NPU types:        910b, a3, 950
# Supported CPU architectures: x86_64 and aarch64 (ARM)
# Supported operating systems: openEuler and Ubuntu
#
# Multi-stage build:
#   Stage 1 (base):            system runtime deps, Miniconda, PyTorch, torch_npu,
#                              decord (x86_64 via pip)
#   Stage 2 (decord-builder):  build decord + ffmpeg from source (aarch64 only)
#   Stage 3 (final):           assemble runtime artifacts, clone MindSpeed-MM
#
# The CI image is built separately on top of this dev image (see Dockerfile.ci).
# ============================================

ARG BASE_IMAGE=""
FROM ${BASE_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-a3-openeuler24.03-py3.11} AS base

# The CANN base image defaults to a non-root user
USER root

# --- Build arguments (base stage) ---
ARG MINICONDA_SH
ARG TORCH_VERSION=2.7.1
ARG TORCH_NPU_VERSION=2.7.1.post8
ARG PYTHON_VERSION=3.11

# Print CPU architecture (x86_64 / aarch64) for early feedback.
RUN ARCH=$(uname -m) && \
    echo "==========================================" && \
    echo "Detected CPU architecture: ${ARCH}" && \
    case "$ARCH" in \
        x86_64|aarch64) echo "Architecture supported" ;; \
        *) echo "ERROR: Unsupported architecture: $ARCH"; exit 1 ;; \
    esac && \
    echo "=========================================="

# Configure system package repositories (external script handles yum/apt sources).
# Kept early in the build: this layer only changes when the script changes, so
# the layers below it are reused from cache across builds.
COPY configure_repo.sh /tmp/configure_repo.sh
RUN chmod +x /tmp/configure_repo.sh && \
    bash /tmp/configure_repo.sh && \
    rm /tmp/configure_repo.sh

# Install runtime system dependencies only.
# Build-only toolchain (gcc/g++/make/cmake/autoconf/...) is intentionally NOT
# installed here: it is needed only by the decord-builder stage for aarch64.
# This keeps this stage and the final image smaller. The OS family is detected
# from /etc/os-release, so no build arg is required.
RUN . /etc/os-release && \
    ID_LOWER=$(echo "$ID" | tr '[:upper:]' '[:lower:]') && \
    case "$ID_LOWER" in \
        *openeuler*) \
            yum install -y git iproute wget curl vim gawk ffmpeg && \
            yum clean all ;; \
        *ubuntu*) \
            apt-get install -y git iproute2 wget curl vim gawk ffmpeg \
                libavcodec-dev libavformat-dev libavutil-dev libswscale-dev && \
            apt-get clean ;; \
        *) echo "ERROR: Unsupported OS ID: $ID" && exit 1 ;; \
    esac

# Install Miniconda. The installer variant (py3xx) is selected by
# --python-version, which sets the Python version of the conda base environment.
# -b: silent install (no user prompts); -p /opt/conda: target install path.
COPY ${MINICONDA_SH} /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    export PATH=/opt/conda/bin:$PATH && \
    # conda 25.x requires accepting the Anaconda channel ToS before installing.
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    # Pin the conda base environment to the requested Python version
    conda install -y -n base python=${PYTHON_VERSION} && \
    conda clean -ya

ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Configure pip to use the Huawei Cloud PyPI mirror and init conda for bash
# (auto-activates the base env on container start).
RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && \
    pip config set global.trusted-host "repo.huaweicloud.com" && \
    /opt/conda/bin/conda init bash

# Install PyTorch and torch_npu (online). Retries tolerate network instability.
# - x86_64:  official PyTorch CPU index (more stable).
# - aarch64: default PyPI index.
RUN ARCH=$(uname -m) && \
    MAX_RETRIES=3 && \
    for retry in $(seq 1 $MAX_RETRIES); do \
        pip cache purge 2>/dev/null || true; \
        rm -rf /root/.cache/pip; \
        if [ "$ARCH" = "x86_64" ]; then \
            if pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then break; fi; \
        elif [ "$ARCH" = "aarch64" ]; then \
            if pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio; then break; fi; \
        fi; \
    done && \
    for retry in $(seq 1 $MAX_RETRIES); do \
        pip cache purge 2>/dev/null || true; \
        rm -rf /root/.cache/pip; \
        if pip install --no-cache-dir torch-npu==${TORCH_NPU_VERSION} pyyaml; then break; fi; \
    done

# Install decord via pip on x86_64 (on aarch64 it is built from source in the
# decord-builder stage).
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        for retry in $(seq 1 3); do \
            pip cache purge 2>/dev/null || true; \
            rm -rf /root/.cache/pip; \
            if pip install --no-cache-dir decord==0.6.0; then break; fi; \
        done; \
    fi

# ========================================
# Stage 2 (decord-builder): aarch64 only.
# Builds ffmpeg + decord from source and collects the runtime artifacts into
# /decord_artifacts, so the final image does not retain build tools or sources.
# ========================================
FROM base AS decord-builder

ARG DECORD_SCRIPT
ARG DECORD_DEPS_DIR=decord_deps
ARG DECORD_BUILD=false

COPY ${DECORD_SCRIPT} /tmp/install_decord.sh
COPY ${DECORD_DEPS_DIR} /tmp/decord_deps/

RUN chmod +x /tmp/install_decord.sh && \
    ARCH=$(uname -m) && \
    mkdir -p /decord_artifacts/lib /decord_artifacts/site && \
    if [ "$ARCH" = "aarch64" ] && [ "$DECORD_BUILD" = "true" ]; then \
        echo "Building decord for aarch64..." && \
        bash /tmp/install_decord.sh /tmp/decord_deps && \
        echo "Collecting decord runtime artifacts..." && \
        cp -L /usr/local/lib/*.so* /decord_artifacts/lib/ && \
        SITE=$(python -c 'import site; print(site.getsitepackages()[0])') && \
        cp -r "${SITE}"/decord* /decord_artifacts/site/; \
    else \
        echo "Skipping decord source build (x86_64 or DECORD_BUILD=false)"; \
    fi && \
    rm -rf /tmp/decord_deps /tmp/install_decord.sh ~/ffmpeg_build ~/bin ~/ffmpeg_sources

# ========================================
# Stage 3 (final): assemble the dev image
# ========================================
FROM base AS final

ARG MINDSPEED_MM_BRANCH=26.1.0
ARG DECORD_BUILD=false
# FINAL_WORKDIR: working directory of the dev image.
ARG FINAL_WORKDIR=/workspace/MindSpeed-MM
ENV FINAL_WORKDIR=${FINAL_WORKDIR}

WORKDIR /workspace

# Copy decord runtime artifacts built in the decord-builder stage (aarch64).
COPY --from=decord-builder /decord_artifacts/ /decord_artifacts/
RUN if [ "$DECORD_BUILD" = "true" ]; then \
        echo "Installing decord artifacts..." && \
        cp -L /decord_artifacts/lib/*.so* /usr/local/lib/ && \
        SITE=$(python -c 'import site; print(site.getsitepackages()[0])') && \
        cp -r /decord_artifacts/site/* "${SITE}"/ && \
        echo '/usr/local/lib' > /etc/ld.so.conf.d/mindspeed-libs.conf && \
        ldconfig && \
        python -c "import decord; print(f'decord version: {decord.__version__}')"; \
    fi && \
    rm -rf /decord_artifacts

# Clone the MindSpeed-MM repository (dev image).
RUN echo ">>> Cloning MindSpeed MM..." && \
    git clone --branch ${MINDSPEED_MM_BRANCH} https://gitcode.com/Ascend/MindSpeed-MM.git

# Final cleanup: remove conda/pip caches and temp files to shrink the image.
RUN conda clean -ya && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/*

WORKDIR ${FINAL_WORKDIR}

# Configure bash startup: cd into the working directory and activate conda base.
RUN echo "cd ${FINAL_WORKDIR}" >> /root/.bashrc && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /root/.bashrc && \
    echo 'conda activate base' >> /root/.bashrc

ENV PATH=/opt/conda/bin:$PATH
# CANN Ascend Toolkit installation path (adjust for your actual CANN version)
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit
