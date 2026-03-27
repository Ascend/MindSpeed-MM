# 安装说明

  本文主要向用户介绍如何快速基于PyTorch框架以及MindSpore框架完成MindSpeed MM（多模态模型套件）的安装。

## 硬件配套和支持的操作系统

**表 1**  产品硬件支持列表

|产品|是否支持（训练场景）|
|--|:-:|
|<term>Atlas A3 训练系列产品</term>|√|
|<term>Atlas A3 推理系列产品</term>|x|
|<term>Atlas A2 训练系列产品</term>|√|
|<term>Atlas A2 推理系列产品</term>|x|
|<term>Atlas 200I/500 A2 推理产品</term>|x|
|<term>Atlas 推理系列产品</term>|x|
|<term>Atlas 训练系列产品</term>|√|

> [!NOTE]  
> 本节表格中“√”代表支持，“x”代表不支持。

## 安装前准备

根据[版本说明](./release_notes.md)获取配套的版本软件。

### 驱动固件安装

下载[驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Debian&Software=cannToolKit)，请根据系统和硬件产品型号选择对应版本的`driver`和`firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Debian&Software=cannToolKit)或执行以下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

### CANN安装

获取[CANN](https://www.hiascend.com/cann/download)，安装配套版本的Toolkit、ops和NNAL并配置CANN环境变量。具体请参考《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html)》（商用版）或《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)》（社区版）。

```shell
#基于PyTorch框架设置环境变量
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
source /usr/local/Ascend/nnal/atb/set_env.sh # 修改为实际安装的nnal包路径
```

```shell
#基于MindSpore框架设置环境变量
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0 # 修改为实际安装的nnal包路径
```

> [!NOTICE]  
> 建议使用非root用户安装运行torch\_npu，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。
> 更多安全相关内容请参见《[安全声明](SECURITYNOTE.md)》中各组件关于“文件权限控制”的说明。

## 基于PyTorch框架

### 安装PyTorch以及torch_npu

请参考《Ascend Extension for PyTorch 软件安装指南》中的“[安装PyTorch框架](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)”章节，获取配套版本的PyTorch以及torch_npu软件包。
可参考如下安装命令：

```shell
# 安装torch和torch_npu 构建参考 https://gitcode.com/ascend/pytorch/releases
pip3 install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl 
pip3 install torch_npu-2.7.1rc1-cp310-cp310-manylinux_2_28_aarch64.whl
```

### 安装MindSpeed MM

安装MindSpeed MM有如下两种方式：

  - 一键安装：快速安装最新配套的第三方依赖及MindSpeed MM
  - 手动安装：灵活指定需要使用的第三方依赖及MindSpeed MM

 **一键安装**

  目前Qwen3-VL、Wan2.2等模型已支持一键安装，具体支持情况可以查看具体模型的[README](https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples)。

  一键式命令会依次安装`PyTorch`、`torch_npu`、`Megatron-LM`、`MindSpeed`、`MindSpeed MM`。由于Megatron-LM对于`pip install`安装方式适配性待提升，采用源码拷贝方式进行使用。

  以Qwen3.5模型安装为例：

  1. 获取取MindSpeed MM代码仓，并进入代码仓根目录：

      ```bash
        git clone https://gitcode.com/Ascend/MindSpeed-MM.git
        cd MindSpeed-MM
        git checkout master
      ```

  2. 执行如下指令一键安装：

      ```bash
        bash scripts/install.sh --msid eb10b92 && bash examples/fsdp2/ qwen3_5/install_extensions.sh
      ```

   scripts/install.sh提供了如下选项供使用：

   ```text
        Options:
          -t, --torchversion VERSION   PyTorch version to install (default: 2.7.1)
          -m, --msid COMMIT_ID    MindSpeed commit ID [required]
          -y, --yes               Auto confirm all reinstallations
          -n, --no                Auto skip all reinstallations
          -mt, --megatron         Install Megatron-LM
          -ic, --install-cann     Install CANN (Compute Architecture for Neural Networks)
          -h, --help              Display this help message and exit
   ```

  |参数名称|说明|是否必选|取值范围|
  |--|--|--|:-:|
  |-t, --torchversion|表示当前使用的torch版本|否|2.6.0或2.7.1|
  |-m, --msid|表示当前基于源码安装的MindSpeed加速库的commit id|是|MindSpeed最新商发分支commit id|
  |-y, --yes|确认所有软件重新安装|否|-|
  |-n, --no|自动跳过第三依赖安装|否|-|
  |-mt, --megatron|安装Megatron-LM|否|默认安装版本Megatron-LM 0.12.0|
  |-ic, --install-cann |安装CANN|否|默认安装版本CANN 8.5.0|
  |-h, --help|显示安装帮助|否|-|

  若当前环境中已经安装了PyTorch或torch_npu，安装时会在控制台打印如下信息，样例如下：

  ```text
      Version check results:
      Currently installed torch version: 2.6.0, target version: 2.7.1
      Currently installed torch_npu version: 2.6.0, target version: 2.7.1
    Version mismatch detected. Continue installation? (y/n)
  ```

  表示检测到环境中已经安装了2.6.0版本的PyTorch和torch_npu。如果您希望安装新版本的torch和torch_npu，请输入`y`；如果希望保持已安装的PyTorch和torch_npu，请输入`n`。

  在安装完成之后，若控制台打印如下信息，说明安装成功：

  ```text
    mindspeed mm successfully installed！
  ```

 **手动安装**

  该方法适用于单独安装PyTorch和其他三方库进行开发调试的用户使用。

  1. 获取MindSpeed MM和Megatron-LM源码。

      ```shell
        git clone https://gitcode.com/Ascend/MindSpeed-MM.git 
        git clone https://github.com/NVIDIA/Megatron-LM.git
        cd Megatron-LM
        git checkout core_v0.12.1
        cp -r megatron ../MindSpeed-MM/
        cd ..
        cd MindSpeed-MM
      ```
        
  2. 获取MindSpeed加速库源码并安装。
      

      ```shell
          # 获取源码
          git clone https://gitcode.com/Ascend/MindSpeed.git
          # 根据需要切换到特定的分支或commitid
          cd MindSpeed
          git checkout 93c45456c7044bacddebc5072316c01006c938f9
          # 安装加速库
          pip install -r requirements.txt 
          pip install -e .
          cd ..
      ```

  3. 安装Mindpseed MM及其相关依赖，可通过[pyptoject.toml](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/pyproject.toml)配置第三方依赖清单。
  
      ```shell
        pip install -e .
      ```

## 基于MindSpore框架

### 安装MindSpore

参考[MindSpore官方安装指导](https://www.mindspore.cn/install)，根据系统类型、CANN版本及Python版本选择相应的安装命令进行安装，安装前请确保网络畅通。

### 一键式适配MindSpeed MM

针对MindSpore框架，我们提供了一键转换工具MindSpeed-Core-MS，旨在帮助用户自动拉取相关代码仓并对torch代码进行一键适配，进而使用户无需再额外手动开发适配即可在MindSpore+CANN环境下一键拉起模型训练。

```shell
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.5.0 
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm
cd MindSpeed-MM
```
