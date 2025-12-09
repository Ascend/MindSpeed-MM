#!/bin/bash

# show help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -a, --arch ARCH         Target architecture (x86|arm) [required]
    -t, --torchversion VERSION   PyTorch version to install (default: 2.7.1)
    -m, --msid COMMIT_ID    MindSpeed commit ID [required]
    -h, --help              Display this help message and exit

Examples:
    bash $0 --arch x86 --torchversion 2.6.0 --msid 93c45456c7044bacddebc5072316c01006c938f9
    bash $0 --arch arm --msid abcdef1234567890
EOF
}

# Default values
ARCH=""
TORCH_VERSION="2.7.1"
MINDSPEED_COMMIT_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--arch) ARCH="$2"; shift 2 ;;
        -t|--torchversion) TORCH_VERSION="$2"; shift 2 ;;
        -m|--msid) MINDSPEED_COMMIT_ID="$2"; shift 2 ;;
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

# Version validation function
check_existing_versions() {
    local need_confirmation=false
    local message=""

    # Check if torch is already installed
    if is_package_installed "torch"; then
        local current_torch_version
        current_torch_version=$(pip3 show torch | grep "^Version:" | awk '{print $2}')

        if [ "$current_torch_version" != "$TORCH_VERSION" ]; then
            need_confirmation=true
            message+="Currently installed torch version: $current_torch_version, target version: $TORCH_VERSION\n"
        else
            message+="Current torch version matches target version: $TORCH_VERSION\n"
        fi
    else
        message+="torch not detected, will install new version: $TORCH_VERSION\n"
    fi

    # Check if torch_npu is already installed
    if is_package_installed "torch_npu"; then
        local current_torch_npu_version
        current_torch_npu_version=$(pip3 show torch_npu | grep "^Version:" | awk '{print $2}')

        if [ "$current_torch_npu_version" != "$TORCH_VERSION" ]; then
            need_confirmation=true
            message+="Currently installed torch_npu version: $current_torch_npu_version, target version: $TORCH_VERSION\n"
        else
            message+="Current torch_npu version matches target version: $TORCH_VERSION\n"
        fi
    else
        message+="torch_npu not detected, will install new version: $TORCH_VERSION\n"
    fi

    # Display version information
    echo "Version check results:"
    echo -e "$message"

    # If confirmation is needed, prompt user
    if [ "$need_confirmation" = true ]; then
        echo "Version mismatch detected. Continue installation? (y/n)"
        read -r user_input

        case $user_input in
            [Yy]* )
                echo "Continuing installation..."
                return 0
                ;;
            [Nn]* )
                echo "Installation cancelled"
                exit 0
                ;;
            ▪ )

                echo "Please enter y or n"
                return 1
                ;;
        esac
    else
        return 0
    fi
}

# Execute version check
if ! check_existing_versions; then
    exit 1
fi

# Determine architecture and install corresponding libraries
if [ "$ARCH" = "x86" ]; then
    echo "Installing x86 version of PyTorch $TORCH_VERSION..."
    pip3 install torch=="$TORCH_VERSION+cpu" torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

elif [ "$ARCH" = "arm" ]; then
    echo "Installing ARM version of PyTorch $TORCH_VERSION..."
    pip3 install torch=="$TORCH_VERSION" torchvision torchaudio

else
    echo "Error: Unsupported architecture '$ARCH'"
    show_help
    exit 1
fi

# Install torch_npu
echo "Installing torch_npu $TORCH_VERSION..."
pip3 install torch-npu=="$TORCH_VERSION"

# Post-installation verification
echo "Installation completed, verifying versions:"
if is_package_installed "torch"; then
    installed_torch=$(pip3 show torch | grep "^Version:" | awk '{print $2}')
    echo "torch version: $installed_torch"
else
    echo "torch installation failed"
fi

if is_package_installed "torch_npu"; then
    installed_npu=$(pip3 show torch_npu | grep "^Version:" | awk '{print $2}')
    echo "torch_npu version: $installed_npu"
else
    echo "torch_npu installation failed"
fi

# Install megatron
echo "Installing Megatron-LM..."
cd ..
if [ ! -d "Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git
fi
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ../MindSpeed-MM/

# Install mindspeed with retry mechanism
echo "Installing MindSpeed with commit ID: $MINDSPEED_COMMIT_ID"
if [ ! -d "MindSpeed" ]; then
    git clone https://gitcode.com/Ascend/MindSpeed.git
fi
cd MindSpeed
git checkout "$MINDSPEED_COMMIT_ID"

# Install MindSpeed with retry mechanism
if ! install_with_retry "mindspeed" "pip3 install -e ."; then
    echo "Critical error: MindSpeed installation failed after multiple attempts"
    exit 1
fi

cd ..

# Create directories
echo "Creating necessary directories..."
mkdir -p logs data ckpt

# Install mindspeed-mm dependency library with retry mechanism
echo "Installing mindspeed-mm dependency library..."
if ! install_with_retry "mindspeed-mm" "pip3 install -e ."; then
    echo "Critical error: mindspeed-mm installation failed after multiple attempts"
    exit 1
fi