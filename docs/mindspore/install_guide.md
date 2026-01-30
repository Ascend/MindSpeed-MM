# MindSpeed-MM MindSpore后端安装指导

## 版本配套

<table border="0">
  <tr>
    <th>软件</th>
    <th>版本</th>
    <th>安装指南</th>
  </tr>
  <tr>
    <td> Python </td>
    <td> >= 3.10 </td>
    <td>  </td>
  </tr>
  <tr>
    <td> Driver </td>
    <td> AscendHDK 25.5.0 </td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> Firmware </td>
    <td> AscendHDK 25.5.0 </td>
  </tr>
  <tr>
    <td> CANN </td>
    <td> CANN 8.5.0 </td>
    <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0008.html">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> MindSpore </td>
    <td> 2.7.2 </td>
    <td> 《<a href="https://www.mindspore.cn/install/">MindSpore安装</a>》</td>
  </tr>
</table>

## 驱动固件安装

下载[驱动固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.5.0&driver=Ascend+HDK+25.5.0)，请根据系统和硬件产品型号选择对应版本的 `driver`和 `firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)或执行以下命令安装：

```shell
bash Ascend-hdk-*-npu-firmware_*.run --full
bash Ascend-hdk-*-npu-driver_*.run --full --force
```

## CANN安装

下载[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)，请根据系统选择 `aarch64`或 `x86_64`对应版本的 `cann-toolkit`、`cann-kernel`和 `cann-nnal`。参考[CANN安装](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0008.html)或执行以下命令安装：

```shell
bash Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
bash Ascend-cann-*-ops_8.5.0_linux-aarch64.run --install
source /usr/local/Ascend/cann/set_env.sh # 安装nnal包需要source环境变量
bash Ascend-cann-nnal_8.5.0_linux-aarch64.run --install
# 设置环境变量
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
```

## MindSpore安装

参考[MindSpore官方安装指导](https://www.mindspore.cn/install)，根据系统类型、CANN版本及Python版本选择相应的安装命令进行安装，安装前请确保网络畅通。或执行以下命令安装：

```shell
pip install mindspore==2.7.2
```

## 代码一键适配

MindSpeed-Core-MS提供了代码、环境的一键适配功能，执行以下命令完成一键适配后，用户即可开启基于MindSpore AI套件的多模态模型之旅。

```shell
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.5.0
cd MindSpeed-Core-MS
source auto_convert.sh mm
cd MindSpeed-MM
```
