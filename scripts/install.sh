#!/bin/bash

# show help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -a, --arch ARCH         Target architecture (x86|arm) [required]
    -t, --torchversion VERSION   PyTorch version to install (default: 2.7.1)
    -m, --msid COMMIT_ID    MindSpeed commit ID [required]
    -y, --yes               Auto confirm all reinstallations
    -n, --no                Auto skip all reinstallations
    -mt, --megatron         Install Megatron-LM
    -h, --help              Display this help message and exit

Examples:
    # Auto confirm all reinstallations
    bash $0 --arch x86 --torchversion 2.6.0 --msid 93c45456c7044bacddebc5072316c01006c938f9 --yes

    # Auto skip all reinstallations
    bash $0 --arch arm --msid abcdef1234567890 --no

    # Interactive mode (default)
    bash $0 --arch x86 --torchversion 2.7.1 --msid abcdef1234567890
EOF
}

# Default values
ARCH=""
TORCH_VERSION="2.7.1"
MINDSPEED_COMMIT_ID=""
AUTO_CONFIRM=""  # 自动确认模式: "", "yes", "no"
INSTALL_MEGATRON=false  # 是否安装Megatron-LM

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--arch) ARCH="$2"; shift 2 ;;
        -t|--torchversion) TORCH_VERSION="$2"; shift 2 ;;
        -m|--msid) MINDSPEED_COMMIT_ID="$2"; shift 2 ;;
        -y|--yes) AUTO_CONFIRM="yes"; shift ;;
        -n|--no) AUTO_CONFIRM="no"; shift ;;
        -mt|--megatron) INSTALL_MEGATRON=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; show_help; exit 1 ;;
    esac
done

# Check required parameters
if [ -z "$ARCH" ]; then
    echo "Error: Architecture parameter is required"
    show_help
    exit 1
fi

if [ -z "$MINDSPEED_COMMIT_ID" ]; then
    echo "Error: MindSpeed commit ID parameter is required"
    show_help
    exit 1
fi

# Function to check if package is installed
is_package_installed() {
    local package_name=$1
    pip3 show "$package_name" &>/dev/null
    return $?
}

# Function to install with retry
install_with_retry() {
    local package_name=$1
    local install_cmd=$2
    local max_retries=3
    local retry_count=0

    echo "Installing $package_name..."

    while [ $retry_count -lt $max_retries ]; do
        echo "Attempt $((retry_count + 1)) of $max_retries..."

        if $install_cmd; then
            # Check if installation was successful
            if is_package_installed "$package_name"; then
                echo "$package_name installed successfully!"
                return 0
            else
                echo "Installation command succeeded but package not found, retrying..."
            fi
        else
            echo "Installation command failed, retrying..."
        fi

        retry_count=$((retry_count + 1))

        # Wait 3 seconds before retry (except on last attempt)
        if [ $retry_count -lt $max_retries ]; then
            echo "Waiting 3 seconds before retry..."
            sleep 3
        fi
    done

    echo "Error: Failed to install $package_name after $max_retries attempts"
    return 1
}

# Function to get torch version
get_torch_version() {
    if is_package_installed "torch"; then
        pip3 show torch | grep "^Version:" | awk '{print $2}'
    else
        echo ""
    fi
}

# Version validation function
check_existing_versions() {
    local install_torch=true
    local install_torch_npu=true
    local message=""

    echo "Auto confirm mode: ${AUTO_CONFIRM:-"interactive (no auto confirm)"}"

    # Check if torch is already installed
    if is_package_installed "torch"; then
        local current_torch_version
        current_torch_version=$(get_torch_version)

        if [ "$current_torch_version" != "$TORCH_VERSION" ]; then
            message+="\n=== PyTorch Version Mismatch ===\n"
            message+="Currently installed torch version: $current_torch_version\n"
            message+="Target version: $TORCH_VERSION\n"
            message+="y: Reinstall PyTorch to target version\n"
            message+="n: Skip PyTorch installation, continue with other components\n"

            echo "Version check results:"
            echo -e "$message"

            # 处理自动确认逻辑
            if [ "$AUTO_CONFIRM" = "yes" ]; then
                echo "Auto confirming: Will reinstall PyTorch (--yes flag detected)"
                install_torch=true
            elif [ "$AUTO_CONFIRM" = "no" ]; then
                echo "Auto skipping: Will skip PyTorch installation (--no flag detected)"
                install_torch=false
            else
                # 交互式模式
                while true; do
                    echo "torch version mismatch detected. Reinstall PyTorch? (y/n)"
                    read -r user_input

                    case $user_input in
                        [Yy]* )
                            echo "Will reinstall PyTorch..."
                            install_torch=true
                            break
                            ;;
                        [Nn]* )
                            echo "Will skip PyTorch installation..."
                            install_torch=false
                            break
                            ;;
                        * )
                            echo "Invalid input. Please enter y or n"
                            ;;
                    esac
                done
            fi
        else
            echo "Current torch version matches target version: $TORCH_VERSION"
            install_torch=false
        fi
    else
        echo "torch not detected, will install new version: $TORCH_VERSION"
    fi

    # Check if torch_npu is already installed
    if is_package_installed "torch_npu"; then
        local current_torch_npu_version
        current_torch_npu_version=$(pip3 show torch_npu | grep "^Version:" | awk '{print $2}')

        if [ "$current_torch_npu_version" != "$TORCH_VERSION" ]; then
            echo -e "\n=== torch_npu Version Mismatch ==="
            echo "Currently installed torch_npu version: $current_torch_npu_version"
            echo "Target version: $TORCH_VERSION"

            # 处理自动确认逻辑
            if [ "$AUTO_CONFIRM" = "yes" ]; then
                echo "Auto confirming: Will reinstall torch_npu (--yes flag detected)"
                install_torch_npu=true
            elif [ "$AUTO_CONFIRM" = "no" ]; then
                echo "Auto skipping: Will skip torch_npu installation (--no flag detected)"
                install_torch_npu=false
            else
                # 交互式模式
                while true; do
                    echo "Reinstall torch_npu to match PyTorch version? (y/n)"
                    read -r user_input

                    case $user_input in
                        [Yy]* )
                            echo "Will reinstall torch_npu..."
                            install_torch_npu=true
                            break
                            ;;
                        [Nn]* )
                            echo "Will skip torch_npu installation..."
                            install_torch_npu=false
                            break
                            ;;
                        * )
                            echo "Invalid input. Please enter y or n"
                            ;;
                    esac
                done
            fi
        else
            echo "Current torch_npu version matches target version: $TORCH_VERSION"
            install_torch_npu=false
        fi
    else
        echo "torch_npu not detected, will install new version: $TORCH_VERSION"
    fi

    # Return values
    echo "install_torch=$install_torch" > /tmp/install_flags
    echo "install_torch_npu=$install_torch_npu" >> /tmp/install_flags
    return 0
}

# Execute version check
if ! check_existing_versions; then
    exit 1
fi

# Read installation flags
source /tmp/install_flags
rm -f /tmp/install_flags

echo ""
echo "Installation plan:"
echo "- Install torch: $install_torch"
echo "- Install torch_npu: $install_torch_npu"
echo ""

# Install PyTorch components
echo "Starting PyTorch components installation..."

# Install PyTorch if needed
if [ "$install_torch" = true ]; then
    echo "Installing PyTorch $TORCH_VERSION..."

    if [ "$ARCH" = "x86" ]; then
        echo "Installing x86 version of PyTorch $TORCH_VERSION..."
        pip3 install "torch==$TORCH_VERSION+cpu" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
    elif [ "$ARCH" = "arm" ]; then
        echo "Installing ARM version of PyTorch $TORCH_VERSION..."
        pip3 install "torch==$TORCH_VERSION" "torchvision" "torchaudio"
    else
        echo "Error: Unsupported architecture '$ARCH'"
        show_help
        exit 1
    fi

    if is_package_installed "torch"; then
        installed_torch=$(get_torch_version)
        echo "torch version: $installed_torch successfully installed！"
    else
        echo "[ERROR] Installation failed! Reason: torch installation error."
    fi
else
    # 用户选择不重新安装torch，安装与当前torch版本兼容的torchvision和torchaudio
    if is_package_installed "torch"; then
        current_torch_version=$(get_torch_version)
        echo "Using existing torch version: $current_torch_version"
        echo "Installing torchvision and torchaudio compatible with torch $current_torch_version..."

        if [ "$ARCH" = "x86" ]; then
            echo "Installing x86 compatible packages..."
            pip3 install "torch==$current_torch_version+cpu" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
        elif [ "$ARCH" = "arm" ]; then
            echo "Installing ARM compatible packages..."
            pip3 install "torch==$current_torch_version" "torchvision" "torchaudio"
        fi
    else
        echo "torch not installed, installing default versions..."
        if [ "$ARCH" = "x86" ]; then
            pip3 install "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
        elif [ "$ARCH" = "arm" ]; then
            pip3 install "torch" "torchvision" "torchaudio"
        fi
    fi
fi

# Install torch_npu if needed
if [ "$install_torch_npu" = true ]; then
    echo "Installing torch_npu $TORCH_VERSION..."
    pip3 install torch-npu=="$TORCH_VERSION"

    if is_package_installed "torch_npu"; then
        installed_npu=$(pip3 show torch_npu | grep "^Version:" | awk '{print $2}')
        echo "torch_npu version: $installed_npu successfully installed！"
    else
        echo "[ERROR] Installation failed! Reason: torch_npu installation error."
    fi
else
    if is_package_installed "torch_npu"; then
        current_torch_npu_version=$(pip3 show torch_npu | grep "^Version:" | awk '{print $2}')
        echo "Using existing torch_npu version: $current_torch_npu_version"
    fi
fi

# Install megatron (仅当指定了-mt或--megatron参数时)
if [ "$INSTALL_MEGATRON" = true ]; then
    echo "[INFO] Installing Megatron-LM..."
    cd ..
    if [ ! -d "Megatron-LM" ]; then
        git clone https://github.com/NVIDIA/Megatron-LM.git
    fi
    cd Megatron-LM
    git checkout core_v0.12.1
    if [ ! -d "megatron" ]; then
        echo "[ERROR] Installation failed! Reason: Megatron-LM installation error."
        exit 1
    fi

    cp -r megatron ../MindSpeed-MM/
    cd ../MindSpeed-MM/

    echo "Megatron-LM successfully installed!"
else
    echo "[INFO] Skipping Megatron-LM installation (not specified with -mt or --megatron)"
fi

# Install mindspeed with retry mechanism
echo "[INFO] Installing MindSpeed with commit ID: $MINDSPEED_COMMIT_ID"
if [ ! -d "MindSpeed" ]; then
    git clone https://gitcode.com/Ascend/MindSpeed.git
fi
cd MindSpeed
git checkout "$MINDSPEED_COMMIT_ID"

# Install MindSpeed with retry mechanism
if ! install_with_retry "mindspeed" "pip3 install -e ."; then
    echo "[ERROR] Installation failed! Reason: MindSpeed installation failed after multiple attempts."
    exit 1
fi

echo "[INFO] MindSpeed with commit ID: $MINDSPEED_COMMIT_ID successfully installed!"

cd ..

# Create directories
echo "[INFO] Creating necessary directories..."
mkdir -p logs data ckpt

# Install mindspeed-mm dependency library with retry mechanism
echo "[INFO] Installing mindspeed-mm dependency library..."
if ! install_with_retry "mindspeed-mm" "pip3 install -e ."; then
    echo "[ERROR] Installation failed! Reason: mindspeed-mm installation failed after multiple attempts."
    exit 1
fi

packages=("mindspeed-mm" "mindspeed")
all_found=true

for pkg in "${packages[@]}"; do
    if ! pip3 list 2>/dev/null | grep -q "^${pkg} "; then
        all_found=false
        break
    fi
done

if $all_found; then
    echo "[INFO] mindspeed mm successfully installed!"
else
    echo "[ERROR] Installation failed! Reason: mindspeed-mm or mindspeed install failed."
fi