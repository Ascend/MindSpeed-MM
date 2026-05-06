# MindSpeed MM Docker Image Overview

## Quick Reference

| Item | Description |
| ------ | ------ |
| **Image Name** | mindspeed-mm |
| **Maintainer** | MindSpeed MM Team |
| **Source Repository** | [https://gitcode.com/Ascend/MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) |
| **Dockerfile Path** | `docker/` |
| **License** | Apache-2.0 |

## Key Field Description of Image Tag

Image tag naming follows the template: `{version}-{chip_info}-{os}-py{python_version}-{architecture}`

| Field | Description | Example |
| ------ | ------ | -------- |
| Version | MindSpeed MM version identifier, also the Git branch name | master |
| Chip Info | NPU chip type (lowercase) | a3, 910b |
| Operating System | Operating system | openeuler24.03, ubuntu22.04 |
| Python Version | Python version | py3.11 |
| Architecture | CPU architecture | x86_64, aarch64 |

### Example Tags

| Tag | NPU | Operating System | Python | Architecture |
| ----- | ----- | --------- | -------- | ------ |
| `master-a3-openeuler24.03-py3.11-x86_64` | A3 | openEuler 24.03 | 3.11 | x86_64 |
| `master-a3-openeuler24.03-py3.11-aarch64` | A3 | openEuler 24.03 | 3.11 | aarch64 |
| `master-910b-openeuler24.03-py3.11-x86_64` | 910B | openEuler 24.03 | 3.11 | x86_64 |
| `master-910b-openeuler24.03-py3.11-aarch64` | 910B | openEuler 24.03 | 3.11 | aarch64 |

## Dockerfile Archive Path

| NPU | Operating System | Dockerfile Path |
| ----- | --------- | ---------------- |
| A3 | openEuler | `docker/Dockerfile` |
| A3 | Ubuntu | `docker/Dockerfile` |
| 910B | openEuler | `docker/Dockerfile` |
| 910B | Ubuntu | `docker/Dockerfile` |

Dockerfile naming follows the template: `Dockerfile[.{chip_info}.{os}.{other_fields}]`

- Unified `Dockerfile` supports all NPU types and OS versions through build arguments
- Fields are connected with `.`
- Hyphens `-` are used within fields
- Chip info uses lowercase (a3, 910b)
- OS uses PascalCase (openEuler, ubuntu)

## Project Directory Structure Specification

The Docker project directory follows a clear hierarchical structure for easy maintenance and expansion:

### Core Directory Structure

```text
docker/
├── Dockerfile                 # Unified Dockerfile supporting multiple NPU types and OS
├── build.sh                   # Image build script with flexible parameter configuration
├── OVERVIEW.md                # English documentation
├── OVERVIEW.zh.md             # Chinese documentation
└── scripts/                   # Script directory
    └── model_install/         # Model environment installation scripts
        └── install_*.sh       # Specific model installation scripts
```

### Directory Description

1. **Dockerfile**: Unified build file supporting all NPU types and OS versions through build arguments
2. **build.sh**: Image build script providing flexible parameter configuration and auto-detection functionality
3. **scripts/**: Organized by script functionality
    - **model_install/**: Stores model environment installation scripts, named in the format `install_{environment_name}.sh`

### Script Usage Mechanism

The `docker/build.sh` script will during the build process:

1. Locate the corresponding script directory based on the version number specified by the `-v` parameter (default: master)
2. Copy all `install_*.sh` scripts from the `docker/scripts/model_install/` directory to the `install_scripts/` temporary directory

**Important Note**: The current version of Dockerfile only executes predefined specific installation scripts (such as `install_verl_qwen3vl.sh`). If you need to add new model environment installation scripts, you must also update the Dockerfile to include the copy and execution logic for the new scripts.

## 1. Image Usage Guide

**Important Note:** Due to differences in dependencies between models, the image only pre-installs basic dependencies including torch, torch_npu, and decord. After pulling the image and starting a container, users need to manually install the dependencies required for their target model in the base environment according to the model's README file.

### Running the Image

```bash
# Basic run
docker run -it --rm \
    mindspeed-mm:master-a3-openeuler24.03-py3.11-x86_64 bash

# Run with NPU device (example: device /dev/davinci1)
# Change the ascend-toolkit path to the actual installation path.
# Assuming your NPU device is installed at /dev/davinci1 and NPU driver is installed at /usr/local/Ascend:
docker run -it --rm \
    --device=/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    mindspeed-mm:master-a3-openeuler24.03-py3.11-x86_64 bash

# Run with data directory mounting (example: device /dev/davinci1)
docker run -it --rm \
    --device=/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /path/to/data:/data \
    -v /path/to/weights:/weights \
    mindspeed-mm:master-a3-openeuler24.03-py3.11-x86_64 bash
```

### Built-in Environments

The image includes the following pre-configured environments:

| Environment | Description | Working Directory |
| ------ | ------ | --------- |
| base | Basic environment containing PyTorch, torch_npu, decord, MindSpeed MM | /workspace/MindSpeed-MM |
| verl_qwen3vl | VERL Qwen3VL model environment (vllm, vllm-ascend, verl) | /workspace/verl_qwen3vl |

**Environment Notes:**

- Since the verl environment requires source code compilation, which is time-consuming, it has been pre-installed and configured in the image.
- Considering differences in dependencies between models, the image only pre-installs basic dependencies including torch, torch_npu, and decord.
- After pulling the image and starting a container, users need to manually install the dependencies required for their target model in the base environment according to the model's README file.

## 2. Local Custom Installation Guide

### Build Script Parameter Description

The build script `build.sh` supports multiple parameter configurations. The following default values are examples; please adjust according to actual needs:

| Parameter | Description | Default (Example) |
| ------ | ------ | ------------ |
| `-t, --npu-type` | NPU type: A3 or 910B | None (required) |
| `-o, --os` | Operating system: openeuler24.03 or ubuntu22.04 | openeuler24.03 |
| `-v, --version` | MindSpeed MM version identifier, used as Git branch name and script directory selection basis | master |
| `--torch-version` | PyTorch version | 2.7.1 |
| `--torch-npu-version` | torch-npu version | 2.7.1 |
| `--base-image-version` | Base image CANN version | 9.0.0-beta.2 |
| `--base-image` | Complete base image name | None |
| `--torch-whl` | torch .whl file path (offline installation) | None |
| `--torch-npu-whl` | torch-npu .whl file path (offline installation) | None |
| `--cleanup-on-fail` | Clean up dangling images/containers on build failure | None |

### Basic Build Examples

```bash
cd docker

# Build A3 + openEuler image (default)
bash build.sh -t A3

# Build 910B + openEuler image
bash build.sh -t 910B

# Build A3 + Ubuntu image
bash build.sh -t A3 -o ubuntu22.04

# Build with specified PyTorch version
bash build.sh -t A3 --torch-version 2.7.1 --torch-npu-version 2.7.1

# Build with offline .whl files
bash build.sh -t A3 \
    --torch-whl /path/to/torch-2.7.1+cpu-cp311-cp311-linux_x86_64.whl \
    --torch-npu-whl /path/to/torch_npu-2.7.1-cp311-cp311-linux_x86_64.whl

# Build with specified base image version
bash build.sh -t A3 --base-image-version 9.0.0

# Build with specified version
bash build.sh -t A3 -v master
```

### Automatic Download Function Description

The build script supports automatic download of the following resources. Please ensure network connectivity:

1. **Miniconda installer**: Automatically downloaded when the `--miniconda` parameter is not specified
2. **decord dependency package**: Automatically downloaded for ARM architecture
3. **Base image**: Automatically pulled when `--base-image` is specified and the image doesn't exist locally

## 3. Custom Image Building/Usage Guide

### Automatic Base Image Recognition

The build script automatically recognizes key information from the base image name:

1. **NPU type recognition**: Recognizes `910b` or `a3` patterns from the image tag
2. **Operating system recognition**: Recognizes `openeuler24.03` or `ubuntu22.04` from the image tag
3. **Automatic image tag generation**: Automatically generates image tags that conform to naming rules based on recognized information

### Best Practice Example

The following example shows how to build a MindSpeed MM image using a custom base image:

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Custom configuration
BASE_IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.2-910b-openeuler24.03-py3.11"
TORCH_VERSION="2.7.1"
TORCH_NPU_VERSION="2.7.1.post2"
MINDSPEED_MM_VERSION="master"

# Execute build
bash "${SCRIPT_DIR}/build.sh" \
    --base-image "$BASE_IMAGE" \
    --torch-version "$TORCH_VERSION" \
    --torch-npu-version "$TORCH_NPU_VERSION" \
    -v "$MINDSPEED_MM_VERSION" \
    --cleanup-on-fail
```

**Key Feature Description:**

1. **Automatic recognition**: The script automatically recognizes NPU type (910B) and operating system (openeuler24.03) from `BASE_IMAGE`. If `BASE_IMAGE` doesn't exist in the system, it will be automatically pulled.
2. **Automatic tag generation**: Automatically generates image tags based on recognition results, such as `mindspeed-mm:master-910b-openeuler24.03-py3.11-x86_64`
3. **Automatic download**: If the Miniconda installer or decord dependencies are not available locally, the script will automatically download them
4. **Failure cleanup**: The `--cleanup-on-fail` parameter ensures cleanup of dangling resources if the build fails

### Adding Installation Guide for Other Models

If you need to add environment installation support for other models, you can follow these steps:

#### 1. View Model Examples

First, check the README files for related models in the `examples/` directory to understand the model's environment dependencies and installation requirements.

#### 2. Create Installation Scripts Following Directory Structure Specification

Create new installation scripts according to the hierarchical structure of the `docker/scripts` directory:

1. **Create script**: Create a new installation script in the `docker/scripts/model_install/` directory
2. **Naming convention**: Script files must be named in the format `install_{environment_name}.sh`

### Example: Adding Qwen3.5 Model Environment Installation Script

- **Script path**: `docker/scripts/model_install/install_qwen3.5.sh`

Create a new installation script by referring to the format of `docker/scripts/master/model_install/install_verl_qwen3vl.sh`:

```bash
#!/bin/bash
set -e

source /tmp/common_functions.sh

MINDSPEED_MM_BRANCH="${1:-master}"

ENV_NAME="your_model_name"  # Modify to your model environment name
WORK_DIR="/workspace/${ENV_NAME}"

echo "=========================================="
echo "Installing ${ENV_NAME} environment"
echo "=========================================="

source /opt/conda/etc/profile.d/conda.sh

# Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create --clone base -n ${ENV_NAME} -y
conda activate ${ENV_NAME}

mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

# Install dependencies according to model requirements
# Example: pip_install_retry "package_name==version" 3
# Example: git clone repository and install

# Reinstall torch and torch_npu if needed
reinstall_torch_and_npu

conda clean -ya && rm -rf /root/.cache/pip

echo "=========================================="
echo "${ENV_NAME} environment installation completed"
echo "=========================================="
```

#### 3. Update Dockerfile to Include New Script

Since the current Dockerfile only executes predefined specific installation scripts, you need to manually update the Dockerfile to include the copy and execution logic for the new script:

1. **Add copy command for new script in Dockerfile**:

   ```dockerfile
   # Copy new installation script
   COPY install_scripts/install_your_model.sh /tmp/install_your_model.sh
   ```

2. **Add execution command for new script in Dockerfile**:

   ```dockerfile
   # Execute new installation script
   RUN chmod +x /tmp/install_your_model.sh && \
       bash /tmp/install_your_model.sh "$MINDSPEED_MM_BRANCH" && \
       rm -f /tmp/install_your_model.sh
   ```

3. **Automatic copy mechanism of build script**: The `docker/build.sh` script automatically copies all `install_*.sh` scripts from `docker/scripts/model_install/` to the `install_scripts/` temporary directory, so the new script will be automatically copied to the build context.

4. **Version correspondence**: By specifying the version number with the `-v` parameter, this version number is used to specify the Git branch to clone for MindSpeed-MM.

#### 4. Build Image for Testing

Build an image containing the new model environment using the following command:

```bash
# Use the -v parameter to specify the version number, and the build script will automatically find the corresponding installation script
bash build.sh -t A3 -v master
```

#### 5. Update Documentation

Add information about the new environment in the "Built-in Environments" section:

| Environment | Description | Working Directory |
| ------ | ------ | --------- |
| your_model_name | Description of your model environment | /workspace/your_model_name |

#### Notes

1. **Directory structure specification**: Must follow the `docker/scripts/model_install/` directory structure
2. **Script naming convention**: Installation scripts must be named in the format `install_{environment_name}.sh`
3. **Dockerfile update**: Need to manually update the Dockerfile to include copy and execution logic for new scripts
4. **Executable permissions**: Ensure installation scripts have executable permissions
5. **Dependency installation**: Use the `pip_install_retry` function for retry installation
6. **Cleanup work**: Clean temporary files and cache at the end of the script to reduce image size
7. **Test verification**: Test whether the new environment works properly after building the image

### Secondary Development

Create a custom Dockerfile based on this image:

```dockerfile
FROM mindspeed-mm:master-a3-openeuler24.03-py3.11-x86_64

RUN pip install your-package==1.0.0

COPY . /workspace/your-project

WORKDIR /workspace/your-project
```

Build and run (example: device /dev/davinci1):

```bash
docker build -t my-mindspeed-app:latest .
docker run -it --rm \
    --device=/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    my-mindspeed-app:latest bash
```

### Software Stack

| Component | Version |
| ------ | ------ |
| CANN | 9.0.0-beta.2 |
| Python | 3.11 |
| Miniconda | 26.1.1-1 |
| PyTorch | 2.7.1 |
| torch_npu | 2.7.1 |
| decord | 0.6.0 |
| MindSpeed MM | master |

### Compatibility Change Notes

#### Date: 2026-04-20

- Initial release version
- Based on CANN 9.0.0-beta.2
- PyTorch 2.7.1 + torch_npu 2.7.1
- Python 3.11 (Miniconda 26.1.1-1)
- Includes verl_qwen3vl conda environment
- Supports openEuler 24.03 and Ubuntu 22.04

## License

MindSpeed MM is released under the Apache License 2.0. See the [LICENSE](../LICENSE) file for details.

Like all Docker images, these images may also contain other software under other licenses (such as Bash from the base distribution, and any direct or indirect dependencies of the included main software).

For any use of pre-built images, it is the responsibility of the image user to ensure that any use of this image complies with the relevant licenses of all software contained therein.
