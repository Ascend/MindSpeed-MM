# Installation Guide

This document introduces how to quickly install MindSpeed MM based on the MindSpore framework.

## Supported Hardware and Operating Systems

**Table 1** Product hardware support list

|Product|Whether Supported (Training Scenario)|
|--|:-:|
|<term>Atlas A3 training products</term>|√|
|<term>Atlas A3 inference products</term>|x|
|<term>Atlas A2 training products</term>|√|
|<term>Atlas A2 inference products</term>|x|
|<term>Atlas 200I/500 A2 inference product</term>|x|
|<term>Atlas inference products</term>|x|
|<term>Atlas training products</term>|x|

> [!NOTE]
> In the table, "√" indicates support, and "x" indicates no support.

- For the operating systems supported by each hardware product in physical machine scenarios, please refer to the [Compatibility Checker](https://www.hiascend.com/hardware/compatibility).

- For the operating systems supported by each hardware product in virtual machine and container scenarios, please refer to the "[Operating System Compatibility Notes (Commercial Edition)](https://www.hiascend.com/document/detail/en/canncommercial/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)" section or the "[Operating System Compatibility Notes (Community Edition)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)" section in *CANN Software Installation*.

## Pre-installation Preparation

Please refer to the "[Related Product Version Compatibility Notes](../release_notes_mm.md#related-product-version-compatibility)" section in *Release Notes* to download and install the corresponding software versions.

### Installing the Driver and Firmware

Download the [driver and firmware](https://www.hiascend.com/hardware/firmware-drivers/community). Select the community or commercial version of the driver and firmware that matches your operating system and hardware product model. Run the following commands to install:

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

### Installing CANN

See *[CANN Quick Installation](https://www.hiascend.com/cann/download)* to install CANN (including Toolkit, ops, and NNAL packages) and configure the environment variables.

```shell
# Set environment variables
source /usr/local/Ascend/cann/set_env.sh # Modify to the actual installed Toolkit package path /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0 # Modify to the actual installed NNAL package path
```

> [!NOTICE]
> It is recommended to install and run torch_npu using a non-root user, and to properly manage permissions for the installation program's directory and files: set folder permissions to `750` and file permissions to `640`. You can control the permissions of installed files by setting `umask`, for example, setting `umask` to `0027`.
> For more security-related information, please refer to the "File Permission Control" instructions for each component in *[Security Statement](../SECURITYNOTE.md)*.

### Installing MindSpore

Refer to the [official MindSpore installation guide](https://www.mindspore.cn/install/en) to obtain the corresponding installation command for MindSpore 2.9.0 based on your system type, CANN version, and Python version. Ensure network connectivity before installation.

## One-click Adaptation for MindSpeed MM

For the MindSpore framework, we provide a one-click conversion tool, MindSpeed-Core-MS, designed to help you automatically pull relevant code repositories and perform one-click adaptation of torch code. This allows you to launch model training with a single click in the MindSpore + CANN environment without additional manual adaptation development.

```shell
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b master
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm
cd MindSpeed-MM
```
