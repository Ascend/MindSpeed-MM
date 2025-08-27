# 安装指南

## 版本配套表

MindSpeed MM支持Atlas 800T A2等昇腾训练硬件形态。软件版本配套表如下：

|MindSpeed MM版本 | MindSpeed版本             | Megatron版本      | PyTorch版本   | torch_npu版本 | CANN版本  | Python版本                               |
|--------------|-------------------------|-----------------|------------- |-------------|---------|----------------------------------------|
|master（主线） | 2.1.0_core_r0.12.1       | Core 0.12.1     |   2.6.0, 2.7.1     | 在研版本        | 在研版本    | Python3.10|
|2.1.0（商用） | 2.1.0_core_r0.8.0         | Core 0.8.0      |   2.1.0, 2.6.0     | 7.1.0       | 8.2.RC1    | Python3.8, Python3.10 |
|2.0.0（商用） | 2.0.0_core_r0.8.0         | Core 0.8.0      |   2.1.0     | 7.0.0       | 8.1.RC1    | Python3.10|
|1.0.0（商用） | 1.0.0_core_r0.6.0         | Core 0.6.0      |   2.1.0     | 6.0.0       | 8.0.0    | Python3.10 |

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
    <td><a href="https://gitee.com/ascend/pytorch#pytorch%E4%B8%8Epython%E7%89%88%E6%9C%AC%E9%85%8D%E5%A5%97%E8%A1%A8">PT配套版本</a></td>
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

下载[驱动固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.2.RC1&driver=Ascend+HDK+25.2.0)，请根据系统和硬件产品型号选择对应版本的`driver`和`firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0003.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)或执行以下命令安装：

```shell
bash Ascend-hdk-*-npu-driver_*.run --full --force
bash Ascend-hdk-*-npu-firmware_*.run --full
```

### 3. CANN安装

下载[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)，请根据根据系统选择`aarch64`或`x86_64`对应版本的`cann-toolkit`、`cann-kernel`和`cann-nnal`。参考[CANN安装](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0003.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)或执行以下命令安装：

```shell
# 因为版本迭代，包名存在出入，根据实际修改
bash Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run --install
bash Ascend-cann-kernels-*_8.2.RC1_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh # 安装nnal包需要source环境变量
bash Ascend-cann-nnal_8.2.RC1_linux-aarch64.run --install
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 4. PTA安装

准备[torch_npu](https://www.hiascend.com/developer/download/community/result?module=pt)和[apex](https://gitee.com/ascend/apex)，参考[Ascend Extension for PyTorch 配置与安装](https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html)或执行以下命令安装：

安装torch和torch_npu，以下以python 3.10 + torch 2.7.1为例：

```shell
conda create -n test python=3.10
conda activate test
# 注：若需安装torch2.6.0版本需要修改列对应whl包，并且修改 MindSpeed-MM/pyproject.toml中的torch版本为2.6.0
pip install torch-2.7.1-cp310-cp310*.whl 
pip install torch_npu-2.7.1*-cp310-cp310*.whl
```

安装apex

```shell
# apex for Ascend 构建参考 https://gitee.com/ascend/apex
pip3 install --upgrade apex-0.1+ascend-{version}.whl # version为python版本和cpu架构
```