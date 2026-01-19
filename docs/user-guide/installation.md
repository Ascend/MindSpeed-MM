# 安装指南

## 版本配套表

MindSpeed MM支持Atlas 800T A2等昇腾训练硬件形态。软件版本配套表如下：

|MindSpeed MM版本 | MindSpeed版本             | Megatron版本      | PyTorch版本   | torch_npu版本 | CANN版本  | Python版本                               |
|--------------|-------------------------|-----------------|------------- |-------------|---------|----------------------------------------|
|master（主线） | master       | Core 0.12.1     |   2.6.0, 2.7.1     | 在研版本        | 在研版本    | Python3.10|
|2.2.0（商用） | 2.2.0_core_r0.12.1        | Core 0.12.1     |   2.6.0, 2.7.1     | 7.2.0       | 8.3.RC1    | Python3.10 |
|2.1.0（商用） | 2.1.0_core_r0.8.0         | Core 0.8.0      |   2.1.0, 2.6.0     | 7.1.0       | 8.2.RC1    | Python3.8, Python3.10 |
|2.0.0（商用） | 2.0.0_core_r0.8.0         | Core 0.8.0      |   2.1.0     | 7.0.0       | 8.1.RC1    | Python3.8, Python3.10 |
|1.0.0（商用） | 1.0.0_core_r0.6.0         | Core 0.6.0      |   2.1.0     | 6.0.0       | 8.0.0    | Python3.8, Python3.10 |

## 昇腾软件安装

### 1. 模型开发时推荐使用配套的环境版本

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">在研版本</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
      <td rowspan="3">在研版本</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
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
    <td>torch_npu插件</td>
    <td>在研版本</td>
  </tr>
</table>

### 2. 驱动固件安装

下载[驱动固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.3.RC1&driver=Ascend+HDK+25.3.0)，请根据系统和硬件产品型号选择对应版本的`driver`和`firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)或执行以下命令安装：

```shell
bash Ascend-hdk-*-npu-driver_*.run --full --force
bash Ascend-hdk-*-npu-firmware_*.run --full
```

### 3. CANN安装

下载[CANN](https://www.hiascend.com/developer/download/commercial/result?module=cann)，请根据系统选择`aarch64`或`x86_64`对应版本的`cann-toolkit`、`cann-kernel`和`cann-nnal`。参考[CANN安装](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)或执行以下命令安装：

```shell
# 因为版本迭代，包名存在出入，根据实际修改
bash Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run --install
bash Ascend-cann-kernels-*_8.3.RC1_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh # 安装nnal包需要source环境变量
bash Ascend-cann-nnal_8.3.RC1_linux-aarch64.run --install
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 4. 一键安装
#### 4.1. 使用说明
一键安装指令会依次安装pytorch、torch_npu、Megatron-LM、MindSpeed、MindSpeed-MM库。由于Megatron-LM对于`pip install`安装方式适配性待提升，采用源码拷贝方式进行使用。

以qwen3vl模型安装指令为例：
```bash
bash scripts/install.sh --arch x86 --msid 93c45456c7044bacddebc5072316c01006c938f9 && pip install -r examples/qwen3vl/requirements.txt
```

scripts/install.sh提供了如下选项供使用：
```text
Options:
    -a, --arch ARCH         Target architecture (x86|arm) [required]
    -t, --torchversion VERSION   PyTorch version to install (default: 2.7.1)
    -m, --msid COMMIT_ID    MindSpeed commit ID [required]
    -h, --help              Display this help message and exit
```
-a, --arch：表示当前安装环境的机器的CPU架构，当前支持x86或arm。此选项为必选选项，会影响安装torch、torch_npu库时使用的版本。
-t, --torchversion：非必选项。表示当前使用的torch版本，默认值为2.7.1。
-m, --msid：必选选项。表示当前基于源码安装的MindSpeed加速库的commit id。
-h, --help：非必选选项，显示安装帮助。

若当前环境中已经安装了pytorch或torch_npu，安装时会在控制台打印如下信息，样例如下：
```text
Version check results:
Currently installed torch version: 2.6.0, target version: 2.7.1
Currently installed torch_npu version: 2.6.0, target version: 2.7.1
Version mismatch detected. Continue installation? (y/n)
```
表示检测到环境中已经安装了2.6.0版本的pytorch和torch_npu。如果您希望安装新版本的torch和torch_npu，请输入`y`；如果希望保持已安装的pytorch和torch_npu，请输入`n`。

在安装完成之后，若控制台打印如下信息：
```text
mindspeed mm successfully installed！
```
说明安装成功。


#### 4.2. 支持模型列表
目前qwen3vl、wan2.2模型已支持一键安装。

### 5. PTA安装
此章节适用于单独安装PTA进行开发调试的用户使用。

准备[torch_npu](https://www.hiascend.com/developer/download/community/result?module=pt)，参考[Ascend Extension for PyTorch 配置与安装](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html)或执行以下命令安装：

安装torch和torch_npu，以下以python 3.10 + torch 2.7.1为例：

```shell
conda create -n test python=3.10
conda activate test
# 注：若需安装torch2.6.0版本需要修改列对应whl包，并且修改 MindSpeed-MM/pyproject.toml中的torch版本为2.6.0
pip install torch-2.7.1-cp310-cp310*.whl 
pip install torch_npu-2.7.1*-cp310-cp310*.whl
```
