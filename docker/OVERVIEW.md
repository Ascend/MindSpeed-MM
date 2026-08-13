# MindSpeed MM Docker Image Overview

## Quick Reference

| Item | Description |
| ------ | ------ |
| **Image Name** | mindspeed-mm |
| **Maintainer** | MindSpeed MM Team |
| **Source Repository** | [https://gitcode.com/Ascend/MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) |
| **Dockerfile Path** | `docker/` |
| **License** | Apache-2.0 |

## MindSpeed MM

MindSpeed MM: An Atlas multimodal large model suite for large-scale distributed training, supporting mainstream multimodal large model training in the industry. It aims to provide an end-to-end multimodal training solution for Huawei [Atlas chips](https://www.hiAscend.com/), including features such as pre-built mainstream models, data engineering, distributed training and acceleration, pre-training, fine-tuning, post-training, and online inference tasks.

The MindSpeed MM image is based on both Ubuntu 22.04 and openEuler 24.03 operating systems, supporting x86_64 and aarch64 (ARM64) CPU architectures. The image comes with the following pre-installed software:

- **PyTorch** + **TorchNPU**: Deep learning framework
- **decord 0.6.0**: High-performance video decoding library
- **CANN**: Huawei Atlas AI processor base software stack

Due to differences in dependencies between models, only the above basic dependencies are pre-installed in the image. After pulling the image and starting a container, users need to manually install the additional dependencies required by the target model in the base environment according to the target model's README file.

Image download: Please visit the [Image Center](https://www.hiascend.com/developer/ascendhub) and search for mindspeed-mm to get the corresponding `docker pull` command.
The current image supports two operating systems, `openEuler 24.03` and `ubuntu22.04`, and provides both `x86_64` and `aarch64` (ARM64) CPU architectures (x86 and aarch64 combined).

## Image Tag Key Field Description

The image tag naming follows the template:

`{version}-{CANN version}-{TorchNPU version}-{product info}-{OS}-{Python version}`

All fields are mandatory, the field order cannot be changed, and all separators use `-`.

| Field | Mandatory | Description | Example Values |
| ------ | ------ | ------ | -------- |
| version | Yes | MindSpeed MM version identifier (v indicates version; the number represents the branch) | v26.0.0, v26.1.0 |
| CANN version | Yes | `cann` + version number | cann9.1.0 |
| TorchNPU version | Yes | `torch_npu` + version number | torch_npu2.7.1.post8 |
| product info | Yes | NPU chip type (lowercase) | 910b, a3, 950 |
| OS | Yes | Operating system | openeuler24.03, ubuntu22.04 |
| Python version | Yes | `py` + version number | py3.11 |

> The tags in the image repository are multi-architecture images combining x86 and aarch64, and **do not** include the `x86_64`/`aarch64` architecture suffix. Only when building locally via the Dockerfile will tags with the architecture suffix (`-x86_64` / `-aarch64`) be generated.

### Example Tags

| Tag | version | CANN | TorchNPU | NPU | OS | Python |
| ----- | ----- | ----- | ----- | ----- | --------- | -------- |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11` | v26.1.0 | 9.1.0 | 2.7.1.post8 | A3 | openEuler 24.03 | 3.11 |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11` | v26.1.0 | 9.1.0 | 2.7.1.post8 | 910B | openEuler 24.03 | 3.11 |

### Tag Meaning Explanation

Taking `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11` as an example:

| Field | Value | Meaning |
| ------ | ------ | ------ |
| version | `v26.1.0` | MindSpeed MM Git tag (branch: 26.1.0) |
| CANN version | `cann9.1.0` | Based on CANN 9.1.0 |
| TorchNPU version | `torch_npu2.7.1.post8` | TorchNPU 2.7.1.post8 |
| product info | `a3` | For Atlas A3 servers |
| OS | `openeuler24.03` | Based on openEuler 24.03 |
| Python version | `py3.11` | Python 3.11 |

> Building locally via the Dockerfile generates tags with the architecture suffix, such as `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11-aarch64` (aarch64) and `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11-x86_64` (x86_64).
> The CANN version, NPU type (910b/a3/950), OS, and Python version are derived from the base image tag by default; the Python version can be overridden with `--python-version`. The TorchNPU version comes from the `--torch-npu-version` parameter.

Dockerfile naming:

- `Dockerfile`: unified dev image build file, supporting all NPU types (910b/a3/950) and OS versions through build arguments
- `Dockerfile.ci`: CI image build file that stacks multi-version conda environments on top of the dev image (used via `--build-ci`)

## Supported Tags and Dockerfile Links

### Latest Version v26.1.0

Below are all image tags of the current latest version v26.1.0 (`cann9.1.0` + `torch_npu2.7.1.post8`). The latest tags are images combining x86 and aarch64, built using the [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docker/Dockerfile). You can obtain the corresponding `docker pull` command from the ascendhub ([mindspeed-mm Image Center](https://www.hiascend.com/developer/ascendhub/detail/6857f6fc2cfa4a678710a7075426ee5e)).

| Tag | Dockerfile | Description |
| --- | --- | --- |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docker/Dockerfile) | 910B + openEuler 24.03, x86_64/aarch64 combined |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docker/Dockerfile) | 910B + Ubuntu 22.04, x86_64/aarch64 combined |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docker/Dockerfile) | A3 + openEuler 24.03, x86_64/aarch64 combined |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu22.04-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docker/Dockerfile) | A3 + Ubuntu 22.04, x86_64/aarch64 combined |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docker/Dockerfile) | 950 + openEuler 24.03, x86_64/aarch64 combined |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-ubuntu22.04-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docker/Dockerfile) | 950 + Ubuntu 22.04, x86_64/aarch64 combined |

> The latest tags in the table above are multi-architecture images (x86 and aarch64 combined). Once actually built, this Dockerfile generates tags with the architecture suffix, such as `-aarch64` / `-x86_64` (e.g. `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11-aarch64`). For all tags of historical versions, please refer to [Supported Tags](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docker/supported_tags.md).

## Project Directory Structure Specification

The Docker project directory follows a clear hierarchical structure for easy maintenance and expansion:

### Core Directory Structure

```text
docker/
├── Dockerfile                 # Unified dev image Dockerfile, supporting multiple NPU types and OS versions
├── Dockerfile.ci              # CI image Dockerfile (stacked on top of the dev image)
├── build.sh                   # Image build script, supporting various parameter configurations
├── OVERVIEW.md                # English documentation
├── OVERVIEW.zh.md             # Chinese documentation
├── supported_tags.md          # Historical tag list
└── scripts/                   # Script directory
    └── ci/                    # CI build scripts and version configuration
```

### Directory Description

1. **Dockerfile**: Unified dev image build file supporting all NPU types and OS versions through build arguments
2. **Dockerfile.ci**: CI image build file that stacks multi-version conda environments on top of the dev image (used via `--build-ci`)
3. **build.sh**: Image build script providing flexible parameter configuration and auto-detection functionality
4. **scripts/**: Organized by script functionality (e.g. `ci/` contains CI build scripts)

### Script Usage Mechanism

The `docker/build.sh` script uses the version number specified by the `-v` parameter (default: 26.1.0) as the branch for git cloning MindSpeed-MM during the build process.

## 1. Image Usage Guide

**Important Notes:**

1. Due to differences in dependencies between models, the image only pre-installs basic dependencies including PyTorch, TorchNPU, and decord. After pulling the image and starting a container, users need to manually install the dependencies required for the target model in the base environment according to the target model's README file.
2. If the NPU driver is not installed in the default path (/usr/local/Ascend/driver), you need to add the path information in the command when running the following docker commands. Taking the path "/usr/local/npu/driver" as an example:

    ```bash
    # Basic run
    docker run -it --rm \
        -e LD_LIBRARY_PATH="/usr/local/npu/driver/lib64/driver:/usr/local/npu/driver/lib64/common:$LD_LIBRARY_PATH" \
        mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11 bash
    ```

### Prerequisites (Optional)

#### Installing the Driver

A compatible Atlas NPU driver must be installed on the host matching the CANN version inside the container. Please visit the [CANN Version Compatibility Website](https://www.hiascend.com/developer/download/compatibility) to understand the correspondence between driver and CANN versions.

### Running the Image

```bash

# Run with NPU device & mount data directory (example: device /dev/davinci1)
# Modify the ascend-toolkit path according to actual situation
# Assuming your NPU device is installed at /dev/davinci1 and the NPU driver is installed at /usr/local/Ascend:
docker run -it \
    --name mm_container \
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
    mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11 bash
```

### Running the CANN Container on 950 Series aarch64 Architecture Products

On 950 series aarch64 architecture products, the CANN container can be run with the following command:

```bash

docker run \
    --name mm_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    --device /dev/ummu \
    --device /dev/uburma \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11 bash
```

### Built-in Environments

The image includes the following pre-configured environments:

| Environment | Description | Working Directory |
| ------ | ------ | --------- |
| base | Basic environment, including PyTorch, TorchNPU, decord, MindSpeed MM | /workspace/MindSpeed-MM |

**Environment Notes:**

- Considering differences in dependencies between models, the image only pre-installs basic dependencies including PyTorch, TorchNPU, and decord.
- After pulling the image and starting a container, users need to manually install the dependencies required for the target model in the base environment according to the target model's README file.

## 2. Local Custom Installation Guide

### Build Script Parameter Description

The build script `build.sh` supports multiple parameter configurations. The CANN version, NPU type (910b/a3/950), and OS are **only** obtained by parsing the `--base-image` tag — they cannot be specified manually. The Python version is auto-detected by default, but can be specified manually with `--python-version`.

| Parameter | Description | Default |
| ------ | ------ | ------------ |
| `--base-image` | **Required.** Full base image name. CANN version, NPU type (910b/a3/950), OS, and Python version are auto-detected from the image tag | None (required) |
| `--python-version` | Python version of the conda base environment (e.g. 3.11/3.10/3.12). Selects the matching Miniconda installer, which finally determines the conda base environment's Python version | Auto-detected from the base image tag |
| `-v, --version` | MindSpeed MM version identifier, also used as the Git branch name | 26.1.0 |
| `--tag` | Custom image tag (overrides the default tag; CI appends `-ci`) | Auto-generated |
| `-n, --no-cache` | Build without cache | None |
| `--torch-version` | PyTorch version (online installation) | 2.7.1 |
| `--torch-npu-version` | TorchNPU version (online installation) | 2.7.1.post8 |
| `--build-ci` | Build CI image on top of the dev image (multi-version conda environments); output tag appends `-ci` | None |
| `--cleanup-on-fail` | Clean up dangling images/containers on build failure | None |

### Basic Build Examples

```bash
cd docker

# Build A3 + openEuler image
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11

# Build 910B + Ubuntu image
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11

# Build 950 + openEuler image with Python 3.12 as the conda base environment
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-openeuler24.03-py3.12 --python-version 3.12

# Build with specified PyTorch version
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11 --torch-version 2.7.1 --torch-npu-version 2.7.1.post4

# Build with specified version
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11 -v 26.1.0

# Build the CI image (on top of the dev image; tag appends -ci)
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11 --build-ci
```

### Automatic Download Function Description

The build script supports automatic download of the following resources. Please ensure network connectivity:

1. **Miniconda installer**: Automatically downloaded, with the variant selected by `--python-version`
2. **decord dependency package**: Automatically downloaded for ARM architecture
3. **Base image**: Automatically pulled when `--base-image` is specified and doesn't exist locally

## 3. Custom Image Building/Usage Guide

### Automatic Base Image Recognition

The build script automatically recognizes key information from the base image tag (the `--base-image` parameter is required):

1. **CANN version recognition**: Extracts the CANN version number (e.g., `9.0.0`) from the image tag
2. **NPU type recognition**: Recognizes the NPU type (e.g., `a3`) from the image tag
3. **Operating system recognition**: Recognizes `openeuler24.03` or `ubuntu22.04` from the image tag
4. **Python version recognition**: Extracts the Python version (e.g., `3.11`) from the `py<x.y>` field in the image tag
5. **Automatic image tag generation**: Automatically generates image tags conforming to naming rules based on recognized information

### Best Practice Example

The following example shows how to build a MindSpeed MM image using a custom base image:

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Custom configuration
BASE_IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.12"
TORCH_VERSION="2.7.1"
TORCH_NPU_VERSION="2.7.1.post8"
MINDSPEED_MM_VERSION="26.1.0"
PYTHON_VERSION="3.11"

# Execute build
bash "${SCRIPT_DIR}/build.sh" \
    --base-image "$BASE_IMAGE" \
    --torch-version "$TORCH_VERSION" \
    --torch-npu-version "$TORCH_NPU_VERSION" \
    -v "$MINDSPEED_MM_VERSION" \
    --python-version "$PYTHON_VERSION" \
    --cleanup-on-fail
```

**Key Feature Description:**

1. **Automatic recognition**: The script automatically recognizes the CANN version (9.1.0), NPU type (910b), and operating system (openeuler24.03) from `BASE_IMAGE`. If `BASE_IMAGE` doesn't exist in the system, it will be automatically pulled.
2. **Specify Python version**: `--python-version` is used to specify the Python version for running the model.
3. **Automatic tag generation**: Automatically generates image tags conforming to the naming rules based on recognition results (e.g. `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11-aarch64`).
4. **Automatic download**: If the Miniconda installer or decord dependencies are not available locally, the script will automatically download them
5. **Failure cleanup**: The `--cleanup-on-fail` parameter ensures cleanup of dangling resources if the build fails

### Secondary Development

Create a custom Dockerfile based on this image:

```dockerfile
FROM mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11

RUN pip install your-package==1.0.0

COPY . /workspace/your-project

WORKDIR /workspace/your-project
```

Build and run (example: device /dev/davinci1):

```bash
# Modify the ascend-toolkit path according to actual situation
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

## License

MindSpeed MM is released under the Apache License 2.0. See the [LICENSE](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/LICENSE) file for details.

This image is built on top of the CANN base image. For license information about the CANN software series contained in these images, please refer to the [CANN Software License](https://www.hiascend.com/legal/cannua-download?isNewCon=true).

Like all Docker images, these images may also contain other software under other licenses (such as Bash from the base distribution, and any direct or indirect dependencies of the included main software).

For any use of pre-built images, it is the responsibility of the image user to ensure that any use of this image complies with the relevant licenses of all software contained therein.

## Disclaimer

The released Atlas software images are community versions and are not intended for commercial accountability. They are provided solely as references for production practices.
