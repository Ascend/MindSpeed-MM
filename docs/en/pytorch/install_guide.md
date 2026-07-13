# MindSpeed MM Installation Guide

This document describes how to quickly install MindSpeed MM, a multimodal model suite, in the PyTorch framework.

## Supported Hardware and OSs

**Table 1** Product hardware support

| Product | Support for Training |
|--|:-:|
| <term>Atlas A3 training products</term> | √ |
| <term>Atlas A3 inference products</term> | x |
| <term>Atlas A2 training products</term> | √ |
| <term>Atlas A2 inference products</term> | x |
| <term>Atlas 200I/500 A2 inference products</term> | x |
| <term>Atlas inference products</term> | x |
| <term>Atlas training products</term> | x |

> [!NOTE]
>
> In the table, √ indicates that the product is supported, and x indicates that the product is not supported.

- For the OSs supported by each hardware product in bare-metal deployment scenarios, see the [Compatibility Query Assistant](https://www.hiascend.com/hardware/compatibility).

- For the OSs supported by each hardware product in VM and container deployment scenarios, see [OS Compatibility](https://www.hiascend.com/document/detail/en/canncommercial/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum) in *CANN Software Installation* (commercial edition) or [OS Compatibility](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum) in *CANN Software Installation* (community edition).

## Installation Preparations

See [Related Product Versions](../release_notes_mm.md#related-product-version-compatibility) in *Release Notes* to download and install the corresponding software versions.

> [!NOTICE]
>
> You are advised to use a non-root user to install and run the program and properly control permissions on the installer directories and files. Set the directory permissions to 750 and the file permissions to 640. You can control the permissions of installed files by setting `umask` to a value such as `0027`. For more security information, see "File Permission Control" for each component in [Security Statement](../SECURITYNOTE.md).

Download the [firmware and driver](https://hiascend.com/hardware/firmware-drivers/community). Select the matching community or commercial edition based on the system and hardware product model.
Run the following commands to install them:

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

## Installing MindSpeed MM

### Method 1: Installation Using an Image

> [!NOTE]
>
> - Before using an image, confirm the machine model. The latest images support only the AArch64 architecture. Run the `uname -a` command to check whether the current environment meets the requirements.
> - The matching images contain CANN 9.0.0 and Ascend Extension for PyTorch 26.0.0. Select an image as required.
> - If the current environment is incompatible with the provided images, use [Method 2: Installation from Source](#method-2-installation-from-source).

1. Pull an image.

   The latest images correspond to the [MindSpeed MM 26.0.0 branch](https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0). [Pull an image](https://www.hiascend.com/developer/ascendhub/detail/6857f6fc2cfa4a678710a7075426ee5e) as required.

   - <term>Atlas A2 training products</term>: 26.0.0-910b-openeuler24.03-py3.11-aarch64

   - <term>Atlas A3 training products</term>: 26.0.0-a3-openeuler24.03-py3.11-aarch64

   ```bash
   # Check whether the image is pulled successfully
   docker image list
   ```

2. Create a container.

   ```bash
   # Mount the image
   docker run -dit --ipc=host --network host --name 'container_name' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ -v /data/:/data image_name:tag /bin/bash
   ```

   By default, the driver and firmware are installed in `/usr/local/Ascend`. If the actual paths are different, modify the paths in the command.

   By default, the container initializes the NPU driver and CANN environment. To install new versions, replace them or manually run the `source` command. For details, see `~/.bashrc` in the container.

   - Example 1: Basic run

     ```bash
     docker run -it --rm \
         mindspeed-mm:26.0.0-a3-openeuler24.03-py3.11-aarch64 bash
     ```

   - Example 2: Run with an NPU device (device `/dev/davinci1` is used as an example)

     ```bash
     # Modify the ascend-toolkit path based on the actual environment
     # Assume that the NPU device is installed at /dev/davinci1 and the NPU driver is installed in /usr/local/Ascend
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
         mindspeed-mm:26.0.0-a3-openeuler24.03-py3.11-aarch64 bash
     ```

   - Example 3: Run with a mounted data directory (device `/dev/davinci1` is used as an example)

     ```bash
     # Modify the ascend-toolkit path based on the actual environment
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
         mindspeed-mm:26.0.0-a3-openeuler24.03-py3.11-aarch64 bash
     ```

3. Access the container and check the environment status.

   ```bash
   # Access the container
   docker exec -it container_name bash
   # Check whether the NPU is available
   npu-smi info
   ```

### Method 2: Installation from Source

You can install MindSpeed MM in either of the following ways:

- Manual installation: Flexibly specify the required third-party dependencies and MindSpeed MM.
- One-click installation: Quickly install the latest matching third-party dependencies and MindSpeed MM. Currently, only the Qwen3 and Qwen3.5 models support this method. Select a method based on your requirements.

#### One-Click Installation

Currently, the [Qwen3-VL](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.0.0/examples/qwen3vl/README.md) and [Qwen3.5](https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/qwen3_5) models support one-click installation.

The one-click command installs PyTorch, `torch_npu`, Megatron-LM, MindSpeed, and MindSpeed MM in sequence. Because Megatron-LM does not yet fully support installation through `pip install`, the command copies its source code for use.

The installation of the Qwen3.5 model is used as an example:

1. Obtain the MindSpeed MM repository and go to its root directory.

   ```bash
   git clone https://gitcode.com/Ascend/MindSpeed-MM.git
   cd MindSpeed-MM
   git checkout 26.0.0
   ```

2. Run the following command to perform one-click installation:

   ```bash
   bash scripts/install.sh --msid eb10b92 && bash examples/qwen3_5/install_extensions.sh
   ```

   **Table 2** Options of the scripts/install.sh file

   | Parameter | Description | Mandatory | Value Range |
   |--|--|--|:-:|
   | -t, --torchversion | Specifies the current torch version. | No | 2.6.0 or 2.7.1 |
   | -m, --msid | Specifies the commit ID of the MindSpeed acceleration library installed from source. | Yes | Commit ID of the latest MindSpeed commercial branch |
   | -y, --yes | Confirms reinstallation of all software. | No | - |
   | -n, --no | Automatically skips installation of third-party dependencies. | No | - |
   | -mt, --megatron | Installs Megatron-LM. | No | Megatron-LM 0.12.0 is installed by default. |
   | -ic, --install-cann | Installs CANN. | No | CANN 8.5.0 is installed by default. |
   | -h, --help | Displays installation help. | No | - |

3. If PyTorch or `torch_npu` is installed, respond to the installation prompt as follows. Otherwise, skip this step.

   The following information in the console indicates that PyTorch 2.6.0 and `torch_npu` 2.6.0 are already installed in the environment. To install new versions of PyTorch and `torch_npu`, enter `y`. To retain the installed versions, enter `n`.

   ```text
   Version check results:
   Currently installed torch version: 2.6.0, target version: 2.7.1
   Currently installed torch_npu version: 2.6.0, target version: 2.7.1
   Version mismatch detected. Continue installation? (y/n)
   ```

4. Check whether the installation succeeded. The following information in the console indicates that the installation succeeded:

   ```text
   mindspeed mm successfully installed!
   ```

#### Manual Installation

This method applies to users who install PyTorch and other third-party libraries separately for development and debugging.

1. Activate the environment.

   ```bash
   # Activate the Python 3.10 environment created previously
   conda create -n test python=3.10
   conda activate test
   ```

2. Install CANN.

   Install matching versions of the NPU driver and firmware and CANN software (Toolkit, ops, and NNAL), and configure the CANN environment variables. For details, see [CANN Software Installation](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html) (commercial edition) or [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html) (community edition).

   CANN provides a process-level environment variable configuration script. Before you run service code on an NPU in training or inference scenarios, run this script. Otherwise, the service code cannot run.

   ```shell
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

   The preceding commands use the default paths after installation by the `root` user as examples. Replace them with the actual paths of the `set_env.sh` scripts.

3. Install PyTorch and `torch_npu`.

   See [Installing PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/en/installation_guide/installation_via_binary_package.md) in *Ascend Extension for PyTorch Installation Guide* to obtain matching PyTorch and `torch_npu` software packages.
   You can use the following commands as a reference:

   ```shell
   # For instructions on building torch and torch_npu, visit https://gitcode.com/ascend/pytorch/releases
   pip3 install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
   pip3 install torch_npu-2.7.1post4-cp310-cp310-manylinux_2_28_aarch64.whl
   ```

4. Obtain the MindSpeed MM and Megatron-LM source code.

   ```shell
   git clone https://gitcode.com/Ascend/MindSpeed-MM.git
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout core_v0.12.1
   cp -r megatron ../MindSpeed-MM/
   cd ..
   cd MindSpeed-MM
   ```

5. Obtain and install the source code of the MindSpeed acceleration library.

   ```shell
   # Obtain the source code
   git clone https://gitcode.com/Ascend/MindSpeed.git
   # Switch to a specific branch or commit ID as required
   cd MindSpeed
   git checkout 26.0.0_core_r0.12.1
   # Install the acceleration library
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```

6. Install MindSpeed MM and its dependencies. You can configure the list of third-party dependencies in [pyproject.toml](../../../pyproject.toml).

   ```shell
   pip install -e .
   ```
