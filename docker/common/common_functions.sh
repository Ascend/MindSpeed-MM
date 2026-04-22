#!/bin/bash

pip_install_retry() {
    local pkg="$1"
    local max_retries="${2:-3}"
    local retry=0

    while [ $retry -lt $max_retries ]; do
        echo ">>> pip install attempt $((retry+1)) of $max_retries: $pkg"

        pip cache purge 2>/dev/null || true
        rm -rf /root/.cache/pip

        if pip install --no-cache-dir $pkg; then
            echo ">>> Success: $pkg installed"
            return 0
        else
            echo ">>> Failed, retrying..."
            retry=$((retry+1))
            sleep 2
        fi
    done

    echo ">>> ERROR: Failed to install $pkg after $max_retries attempts"
    return 1
}

pip_install_editable_retry() {
    local dir="$1"
    local max_retries="${2:-3}"
    local retry=0

    while [ $retry -lt $max_retries ]; do
        echo ">>> pip install -e attempt $((retry+1)) of $max_retries: $dir"

        pip cache purge 2>/dev/null || true
        rm -rf /root/.cache/pip

        if pip install --no-cache-dir -e "$dir"; then
            echo ">>> Success: $dir installed"
            return 0
        else
            echo ">>> Failed, retrying..."
            retry=$((retry+1))
            sleep 2
        fi
    done

    echo ">>> ERROR: Failed to install $dir after $max_retries attempts"
    return 1
}

pip_install_requirements_retry() {
    local req_file="$1"
    local max_retries="${2:-3}"
    local retry=0

    while [ $retry -lt $max_retries ]; do
        echo ">>> pip install -r attempt $((retry+1)) of $max_retries: $req_file"

        pip cache purge 2>/dev/null || true
        rm -rf /root/.cache/pip

        if pip install --no-cache-dir -r "$req_file"; then
            echo ">>> Success: $req_file installed"
            return 0
        else
            echo ">>> Failed, retrying..."
            retry=$((retry+1))
            sleep 2
        fi
    done

    echo ">>> ERROR: Failed to install $req_file after $max_retries attempts"
    return 1
}

reinstall_torch_and_npu() {
    local torch_version="${1:-2.7.1}"
    local torch_npu_version="${2:-2.7.1}"
    local max_retries="${3:-3}"

    local torch_whl=""
    local npu_whl=""
    local torchvision_whl=""
    local torchaudio_whl=""

    if [ "$1" = "--torch-whl" ]; then
        while [ $# -gt 0 ]; do
            case "$1" in
                --torch-whl)
                    torch_whl="$2"
                    shift 2
                    ;;
                --npu-whl)
                    npu_whl="$2"
                    shift 2
                    ;;
                --torchvision-whl)
                    torchvision_whl="$2"
                    shift 2
                    ;;
                --torchaudio-whl)
                    torchaudio_whl="$2"
                    shift 2
                    ;;
                --max-retries)
                    max_retries="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        if [ -z "$torch_whl" ] || [ -z "$npu_whl" ]; then
            echo ">>> ERROR: --torch-whl and --npu-whl are both required in offline mode"
            return 1
        fi

        echo ">>> Reinstalling torch and torch_npu from local .whl files..."
        echo ">>>   torch:         ${torch_whl}"
        echo ">>>   torch_npu:     ${npu_whl}"
        [ -n "$torchvision_whl" ] && echo ">>>   torchvision:   ${torchvision_whl}"
        [ -n "$torchaudio_whl" ] && echo ">>>   torchaudio:    ${torchaudio_whl}"

        local retry=0
        while [ $retry -lt $max_retries ]; do
            echo ">>> torch .whl install attempt $((retry+1)) of $max_retries"

            pip cache purge 2>/dev/null || true
            rm -rf /root/.cache/pip

            local torch_pkgs="${torch_whl}"
            [ -n "$torchvision_whl" ] && torch_pkgs="${torch_pkgs} ${torchvision_whl}"
            [ -n "$torchaudio_whl" ] && torch_pkgs="${torch_pkgs} ${torchaudio_whl}"

            if pip install --no-cache-dir --force-reinstall $torch_pkgs; then
                break
            fi

            echo ">>> Failed, retrying..."
            retry=$((retry+1))
            sleep 2
        done

        if [ $retry -ge $max_retries ]; then
            echo ">>> ERROR: Failed to install torch from ${torch_whl} after $max_retries attempts"
            return 1
        fi

        retry=0
        while [ $retry -lt $max_retries ]; do
            echo ">>> torch_npu .whl install attempt $((retry+1)) of $max_retries"

            pip cache purge 2>/dev/null || true
            rm -rf /root/.cache/pip

            if pip install --no-cache-dir --force-reinstall "${npu_whl}"; then
                echo ">>> Success: torch and torch_npu installed from .whl files"
                return 0
            fi

            echo ">>> Failed, retrying..."
            retry=$((retry+1))
            sleep 2
        done

        echo ">>> ERROR: Failed to install torch_npu from ${npu_whl} after $max_retries attempts"
        return 1
    fi

    echo ">>> Reinstalling torch==${torch_version} and torch_npu==${torch_npu_version}..."

    local ARCH=$(uname -m)
    local retry=0

    while [ $retry -lt $max_retries ]; do
        echo ">>> torch install attempt $((retry+1)) of $max_retries"

        pip cache purge 2>/dev/null || true
        rm -rf /root/.cache/pip

        if [ "$ARCH" = "x86_64" ]; then
            echo ">>> Installing PyTorch for x86_64..."
            if pip install --no-cache-dir torch==${torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
                break
            fi
        elif [ "$ARCH" = "aarch64" ]; then
            echo ">>> Installing PyTorch for aarch64..."
            if pip install --no-cache-dir torch==${torch_version} torchvision torchaudio; then
                break
            fi
        else
            echo ">>> WARNING: Unsupported architecture: $ARCH, skipping torch reinstall"
            return 0
        fi

        echo ">>> Failed, retrying..."
        retry=$((retry+1))
        sleep 2
    done

    if [ $retry -ge $max_retries ]; then
        echo ">>> ERROR: Failed to install torch==${torch_version} after $max_retries attempts"
        return 1
    fi

    retry=0
    while [ $retry -lt $max_retries ]; do
        echo ">>> torch_npu install attempt $((retry+1)) of $max_retries"

        pip cache purge 2>/dev/null || true
        rm -rf /root/.cache/pip

        if pip install --no-cache-dir torch-npu==${torch_npu_version}; then
            echo ">>> Success: torch==${torch_version} and torch_npu==${torch_npu_version} installed"
            return 0
        fi

        echo ">>> Failed, retrying..."
        retry=$((retry+1))
        sleep 2
    done

    echo ">>> ERROR: Failed to install torch_npu==${torch_npu_version} after $max_retries attempts"
    return 1
}
