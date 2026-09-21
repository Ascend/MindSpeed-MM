# MindSpeed MM安装指导

  本文主要向用户介绍如何快速基于PyTorch框架完成MindSpeed MM（多模态模型套件）的安装。

## 硬件配套和支持的操作系统

**表 1**  产品硬件支持列表

|产品|是否支持（训练场景）|
|--|:-:|
|<term>Ascend 950PR&950DT系列产品</term>|√|
|<term>Atlas A3训练系列产品</term>|√|
|<term>Atlas A3推理系列产品</term>|x|
|<term>Atlas A2训练系列产品</term>|√|
|<term>Atlas A2推理系列产品</term>|x|
|<term>Atlas 200I/500 A2推理产品</term>|x|
|<term>Atlas推理系列产品</term>|x|
|<term>Atlas训练系列产品</term>|x|

> [!NOTE]
>
> 本节表格中“√”代表支持，“x”代表不支持。

- 各硬件产品对应物理机部署场景支持的操作系统请参考[兼容性查询助手](https://www.hiascend.com/hardware/compatibility)。

- 各硬件产品对应虚拟机及容器部署场景支持的操作系统请参考《CANN 软件安装》“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)”章节。

## 安装前准备

请参见《版本说明》中的“[相关产品版本配套说明](../../release_notes_mm.md#相关产品版本配套说明)”章节，下载安装对应的软件版本。

> [!NOTE]
>
> 安装运行程序建议使用非root用户，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。更多安全相关内容请参见《[安全声明](../../../../SECURITYNOTE.md)》中各组件关于“文件权限控制”的说明。

下载[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers)，请根据系统和硬件产品型号选择对应版本的社区版本或商用版本的固件与驱动。
参考如下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

## 安装MindSpeed MM

### 方式一：镜像安装

> [!NOTE]
>
> - 使用镜像前，请先确认机器型号。最新镜像仅支持aarch64架构，可通过uname -a命令确认当前环境是否符合要求。
> - 配套镜像已预装配套的CANN 9.1.0软件及TorchNPU 26.1.0插件，您可根据需要选用。
> - 若您当前环境与提供的镜像不兼容，请选择[方式二：源码安装](#方式二源码安装)。
> - master分支后续会更新新的镜像，如果需要自定义构建镜像，请参见[镜像概述](../../../../docker/OVERVIEW.zh.md)。

1. 拉取镜像

   当前可使用MindSpeed MM 26.1.0分支对应镜像，请按需[拉取镜像](https://www.hiascend.com/developer/ascendhub/detail/6857f6fc2cfa4a678710a7075426ee5e)。

   <!-- npu="950" id1 -->
   - <term>Ascend 950PR&950DT系列产品</term>：v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-openeuler24.03-py3.11

   - <term>Ascend 950PR&950DT系列产品</term>：v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-ubuntu22.04-py3.11
   <!-- end id1 -->

   <!-- npu="A3" id2 -->
   - <term>Atlas A3训练系列产品</term>：v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11

   - <term>Atlas A3训练系列产品</term>：v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu22.04-py3.11
   <!-- end id2 -->
   
   <!-- npu="910b" id3 -->
   - <term>Atlas A2训练系列产品</term>：v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11

   - <term>Atlas A2训练系列产品</term>：v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.11
   <!-- end id3 -->

   ```bash
      # 确认是否成功拉取镜像
      docker image list
   ```

2. 创建容器

   ```bash
    # 挂载镜像
    docker run -dit --ipc=host --network host --name '容器名' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ -v /data/:/data 镜像名:标签 /bin/bash
   ```

   当前默认配置驱动和固件安装在/usr/local/Ascend，如有差异请修改指令路径。

   当前容器默认初始化NPU驱动和CANN环境信息，如需要安装新的，请自行替换或手动source，详见容器的~/.bashrc。

    - 示例一：基本运行

      ```bash
      docker run -it --rm \
          mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11 bash
      ```

    - 示例二：使用 NPU 设备运行（示例：设备 /dev/davinci1）

      ```bash
      # 根据实际情况修改 ascend-toolkit 路径
      # 假设 NPU 设备安装在 /dev/davinci1 上，并且 NPU 驱动程序安装在 /usr/local/Ascend 上
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
          mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11 bash
      ```

    - 示例三：挂载数据目录运行（示例：设备 /dev/davinci1）

      ```bash
      # 根据实际情况修改 ascend-toolkit 路径
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
          mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11 bash
      ```

    具体参数配置说明可查看MindSpeed MM Docker镜像概述的[构建脚本参数说明](../../../../docker/OVERVIEW.zh.md#构建脚本参数说明)

3. 加载容器并确认环境状态

   ```bash
    # 查询本地运行中的容器ID/名称
    docker ps -a
    # 加载容器
    docker exec -it 容器名 bash
    # 确认NPU是否可以正常使用
    npu-smi info
   ```

### 方式二：源码安装

安装MindSpeed MM有如下两种方式：

  - 手动安装：灵活指定需要使用的第三方依赖及MindSpeed MM。
  - 一键安装：快速安装最新配套的第三方依赖及MindSpeed MM，当前只有qwen3，qwen3.5模型支持，请按照实际需求选择。

#### 一键安装

  目前[Qwen3-VL](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.0.0/examples/qwen3vl/README.md)、[Qwen3.5](https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/qwen3_5)模型已支持一键安装。

  一键式命令会依次安装`PyTorch`、`TorchNPU`、`Megatron-LM`、`MindSpeed`、`MindSpeed MM`。由于Megatron-LM对于`pip install`安装方式适配性待提升，采用源码拷贝方式进行使用。

  以Qwen3.5模型安装为例：

  1. 获取MindSpeed MM代码仓，并进入代码仓根目录：

      ```bash
        git clone https://gitcode.com/Ascend/MindSpeed-MM.git
        cd MindSpeed-MM
        git checkout master
      ```

  2. 执行如下指令一键安装：

      ```bash
        bash scripts/install.sh --msid eb10b92 && bash examples/qwen3_5/install_extensions.sh
      ```

      **表 2** scripts/install.sh文件选项参数表

        |参数名称|说明|是否必选|取值范围|
        |--|--|--|:-:|
        |-t, --torchversion|表示当前使用的torch版本|否|2.7.1或2.10.0|
        |-m, --msid|表示当前基于源码安装的MindSpeed加速库的commit id|是|MindSpeed最新release版本commit id|
        |-y, --yes|确认所有软件重新安装|否|-|
        |-n, --no|自动跳过第三方依赖库安装|否|-|
        |-mt, --megatron|安装Megatron-LM|否|默认安装版本Megatron-LM 0.12.0|
        |-ic, --install-cann |安装CANN|否|默认安装版本CANN 9.1.0|
        |-h, --help|显示安装帮助|否|-|

  3. 如已安装了PyTorch或TorchNPU，请按以下步骤操作；未安装可跳过本步骤：

      控制台打印了如下信息，表示检测到环境中已经安装了2.6.0版本的PyTorch和TorchNPU。如果您希望安装新版本的PyTorch和TorchNPU，请输入`y`；如果希望保持已安装的PyTorch和TorchNPU，请输入`n`。

        ```text
        Version check results:
        Currently installed torch version: 2.6.0, target version: 2.10.0
        Currently installed torch_npu version: 2.6.0, target version: 2.10.0
        Version mismatch detected. Continue installation? (y/n)
        ```

  4. 检查安装是否成功，若控制台打印如下信息，说明安装成功：

      ```text
      mindspeed mm successfully installed!
      ```

#### 手动安装

  该方法适用于单独安装PyTorch和其他第三方库进行开发调试的用户使用。

  1. 激活环境：

      ```bash
      # 激活上面构建的Python3.12版本的环境
      conda create -n test python=3.12
      conda activate test
      ```

  2. 安装CANN

      安装配套版本的NPU驱动固件、CANN软件（Toolkit、ops和NNAL）并配置CANN环境变量，具体请参考《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html)》。

      CANN软件提供进程级环境变量设置脚本，训练或推理场景下使用NPU执行业务代码前需要调用该脚本，否则业务代码将无法执行。

        ```shell
        source /usr/local/Ascend/cann/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh
        ```

       以上命令以root用户安装后的默认路径为例，请用户根据set_env.sh的实际路径进行替换。

  3. 安装PyTorch以及TorchNPU

      根据引导安装配套版本的PyTorch以及TorchNPU，具体请参考《[TorchNPU 快速安装](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=175&ids=89dda9ba9de741349efa03687a487678%2C98%2C107%2C1%2C6%2C177%2C)》。

      > [!NOTE]
      >
      > - 更多TorchNPU插件版本请单击[Link](https://gitcode.com/ascend/pytorch/releases)。
      > - TorchNPU相关文档请参见《[TorchNPU使用导读](https://www.hiascend.com/document/detail/zh/Pytorch/latest/index/index.html)》。

  4. 获取MindSpeed MM和Megatron-LM源码。

      ```shell
      git clone https://gitcode.com/Ascend/MindSpeed-MM.git
      git clone https://github.com/NVIDIA/Megatron-LM.git
      cd Megatron-LM
      git checkout core_v0.12.1
      cp -r megatron ../MindSpeed-MM/
      cd ..
      cd MindSpeed-MM
      ```

  5. 获取MindSpeed加速库源码并安装。

      ```shell
      # 获取源码
      git clone https://gitcode.com/Ascend/MindSpeed.git
      # 根据需要切换到特定的分支或commitid
      cd MindSpeed
      git checkout master
      # 安装加速库
      pip install -r requirements.txt
      pip install -e .
      cd ..
      ```

  6. 安装MindSpeed MM及其相关依赖，可通过[pyproject.toml](../../../../pyproject.toml)配置第三方依赖清单。

      ```shell
      pip install -e .
      ```

      > [!NOTE]
      >
      > 安装过程中若看到pip's dependency resolver ... dependency conflicts关于transformers版本的提示，属于已知现象，不影响MindSpeed MM的实际安装结果，可通过pip show mindspeed-mm验证。

  7. 安装Triton-Ascend（按需）

      安装配套版本的Triton-Ascend，请参考《[Triton-Ascend安装指南](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#piptriton-ascend)》，获取配套版本的Triton-Ascend安装指令。

      可参考如下安装命令：

      ```shell
      # 注意：triton-ascend 3.2.0 及以下 Triton-Ascend和Triton 不能同时存在。需要先卸载社区 Triton，再安装 Triton-Ascend。
      pip install triton-ascend==3.2.2 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
      ```
