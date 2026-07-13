# Installation Guide

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-05-26T07:09:21.513Z pushedAt=2026-05-26T07:32:09.373Z -->
## Version Compatibility Table

MindSpeed MM supports Ascend training hardware forms such as the Atlas 800T A2. For the software version compatibility table, please refer to the [Related Product Version Compatibility Notes](../release_notes_mm.md#version-compatibility-notes) section in the *Release Notes*.

## Ascend Software Installation

It is recommended to use the matching environment version during model development.

<table>
  <tr>
    <th>Dependency/Software</th>
    <th>Version</th>
  </tr>
  <tr>
    <td>Ascend NPU driver</td>
    <td rowspan="2">In-development</td>
  </tr>
  <tr>
    <td>Ascend NPU firmware</td>
  </tr>
  <tr>
    <td>Toolkit</td>
      <td rowspan="3">In-development</td>
  </tr>
  <tr>
    <td>Ops (Operator package)</td>
  </tr>
  <tr>
    <td>NNAL (Ascend Transformer Boost Acceleration Library)</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td> 3.10 </td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>2.6.0, 2.7.1</td>
  </tr>
  <tr>
    <td>torch_npu plugin</td>
    <td>In-development</td>
  </tr>
</table>

### Driver and Firmware Installation

Download the driver and firmware. Select the corresponding versions of the driver and firmware based on your system and hardware product model. Refer to [Installing NPU Driver and Firmware](https://www.hiascend.com/document/detail/en/canncommercial/850/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit) or execute the following commands to install:

```shell
bash Ascend-hdk-*-npu-driver_*.run --full --force
bash Ascend-hdk-*-npu-firmware_*.run --full
```

### CANN Installation

Download CANN. Select the corresponding versions of `cann-toolkit`, `cann-kernel`, and `cann-nnal` for `aarch64` or `x86_64` based on your system architecture (`aarch64` or `x86_64`). Refer to [CANN Installation Guide](https://www.hiascend.com/document/detail/en/canncommercial/850/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit) or execute the following commands to install:

```shell
# Because of version iteration, the package name may vary. Modify it according to the actual situation.
bash Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
bash Ascend-cann-*-ops_8.5.0_linux-aarch64.run --install
source /usr/local/Ascend/cann/set_env.sh # Installing the nnal package requires sourcing the environment variables.
bash Ascend-cann-nnal_8.5.0_linux-aarch64.run --install
# Set environment variables
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### Python Installation

There are two methods for Python installation: One-click installation and manual installation. One-click Installation can quickly install all required third-party libraries, while manual installation allows you to flexibly specify the versions of each third-party library to use, facilitating debugging. Please choose the appropriate installation method based on your needs.

#### One-click Installation

The one-click installation command will sequentially install the pyTorch, torch_npu, Megatron-LM, MindSpeed, and MindSpeed MM libraries. Since Megatron-LM's compatibility with the `pip install` method needs improvement, it is installed by copying the source code.

Take the installation command for the qwen3vl model as an example:

```bash
bash scripts/install.sh --arch x86 --msid 93c45456c7044bacddebc5072316c01006c938f9 && pip install -r examples/qwen3vl/requirements.txt
```

`scripts/install.sh` provides the following options for use:

```text
Options:
    -a, --arch ARCH         Target architecture (x86|arm) [required]
    -t, --torchversion VERSION   PyTorch version to install (default: 2.7.1)
    -m, --msid COMMIT_ID    MindSpeed commit ID [required]
    -h, --help              Display this help message and exit
```

`-a, --arch`: CPU architecture of the current installation environment, currently supporting x86 or Arm. This option is mandatory and affects the torch and torch_npu versions.
`-t, --torchversion`: Optional. It indicates the torch version currently in use, with a default value of 2.7.1.
`-m, --msid`: Mandatory. It indicates the commit id of the MindSpeed acceleration library installed from source code.
`-h, --help`: Optional. It displays installation help information.

If PyTorch and torch_npu have been already installed in the current environment, the following information will be printed on the console during installation. A sample is shown below:

```text
Version check results:
Currently installed torch version: 2.6.0, target version: 2.7.1
Currently installed torch_npu version: 2.6.0, target version: 2.7.1
Version mismatch detected. Continue installation? (y/n)
```

This indicates that PyTorch and torch_npu version 2.6.0 have been detected in the environment. If you wish to install a new version, please enter `y`; if you wish to keep the installed PyTorch and torch_npu, please enter `n`.

After the installation is complete, if the console prints the following information:

```text
mindspeed mm successfully installed！
```

It indicates that the installation was successful.

**Supported Model List**

Currently, models such as qwen3vl and wan2.2 support one-click installation. For specific details, refer to the README of each model.

#### Manual Installation

This method is suitable for users who need to install PTA and other third-party libraries separately for development and debugging.

1. Install torch and torch_npu.

  Download [torch_npu](https://www.hiascend.com/developer/download/community/result?module=pt), and refer to [Ascend Extension for PyTorch Configuration and Installation](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md) or execute the following command to install:

  The following example uses python 3.10 + torch 2.7.1.

    ```shell
    conda create -n test python=3.10
    conda activate test
    # Note: If you need to install torch 2.6.0, you must modify the corresponding whl package and change the torch version in MindSpeed-MM/pyproject.toml to 2.6.0.
    pip install torch-2.7.1-cp310-cp310*.whl
    pip install torch_npu-2.7.1*-cp310-cp310*.whl
    ```

2. Clone the MindSpeed MM repository and install Megatron.

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_v0.12.1
    cp -r megatron ../MindSpeed-MM/
    cd ..
    cd MindSpeed-MM
    ```

3. Clone the MindSpeed repository and install it.

    ```shell
    # Install Acceleration Library
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed
    # Switch to a specific branch or commit ID as needed
    git checkout 93c45456c7044bacddebc5072316c01006c938f9
    pip install -r requirements.txt
    pip install -e .
    cd ..
    ```

4. Install other dependencies.

    ```shell
    # Install other dependencies for MindSpeed MM
    pip install -e .
    ```
