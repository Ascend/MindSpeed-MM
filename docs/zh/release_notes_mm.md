# 版本说明

## 版本配套说明

### 产品版本信息

<table><tbody><tr><th class="firstcol" valign="top" width="26.25%"><p>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p><span>MindSpeed MM</span></p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" ><p>26.1.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" ><p>正式版本</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p>发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p>2026年7月</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>6个月</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
>
> 有关MindSpeed MM的版本维护策略，具体请参见[版本维护](https://gitcode.com/Ascend/MindSpeed-MM#%E7%89%88%E6%9C%AC%E7%BB%B4%E6%8A%A4)。

### 相关产品版本配套说明

**表 1**  MindSpeed MM配套表

| MindSpeed MM版本 | MindSpeed Core代码分支名称 | Megatron版本 | PyTorch版本  | TorchNPU版本 | CANN版本 | Triton-Ascend版本 | Python版本     |
| ---------------- | ------------------ | ------------ | -----------  | ------------- |--------------------- |-----------------| ------------------- |
| 26.1.0           | 26.1.0_core_r0.12.1 | core_v0.12.1  | 2.7.1       | 26.1.0        | 9.1.0  | 3.2.1           | Python3.10      |
| 26.0.0           | 26.0.0_core_r0.12.1 | core_v0.12.1  | 2.7.1       | 26.0.0        | 9.0.0  | 3.2.1           | Python3.10      |

>[!NOTE]
>
>- 用户可根据需要选择MindSpeed MM代码分支下载源码并进行安装。
>- Triton-Ascend版本与CANN版本强绑定，Triton-Ascend的使用应该与CANN版本一一对应，详见[Triton-Ascend兼容性](https://gitcode.com/Ascend/triton-ascend#%E5%85%BC%E5%AE%B9%E6%80%A7)。

## 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套，“Y”表示可配套。

**表 2**  MindSpeed MM与Ascend Extention for PyTorch版本兼容

<table style="table-layout: fixed; width: 531px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">MindSpeed MM</th>
    <th colspan="4">Ascend Extention for PyTorch版本</th>
  </tr>
  <tr>
    <th>7.2.0</th>
    <th>7.3.0</th>
    <th>26.0.0</th>
    <th>26.1.0</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>26.1.0</td>
    <td>/</td>
    <td>Y</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
  <tr>
    <td>26.0.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>Y</td>
    <td>/</td>
  </tr>
</tbody>
</table>

**表 3**  MindSpeed MM与CANN版本兼容

<table style="table-layout: fixed; width: 547px"><colgroup>
<col style="width: 162px">
<col style="width: 91px">
<col style="width: 94px">
<col style="width: 100px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">MindSpeed MM</th>
    <th colspan="4">CANN版本</th>
  </tr>
  <tr>
    <th>8.3.RC1</th>
    <th>8.5.0</th>
    <th>9.0.0</th>
    <th>9.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.1.0</td>
    <td>/</td>
    <td>Y</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
  <tr>
    <td>26.0.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>Y</td>
    <td>/</td>
  </tr>
</tbody>
</table>

## 版本使用注意事项

无

## 更新说明

### 新增特性

|组件|描述|目的|
|--|--|--|
|MindSpeed MM|新模型|支持Qwen3.5, Kimi-K2.5|
|MindSpeed MM|新特性|Qwen3.5、Kimi-K2.5支持激活值异步卸载、chunk loss、chunk mbs|
|MindSpeed MM|新增硬件支持|支持Ascend 950系列产品|

### 删除特性

无

### 接口变更说明

无

### 已解决问题

无

### 遗留问题

无

## 升级影响

### 升级过程中对现行系统的影响

- 对业务的影响

    软件版本升级过程中会导致业务中断。

- 对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无

## 配套文档

|文档名称|内容简介|更新说明|
|《[MindSpeed MM安装指导](../zh/pytorch/install_guide.md)》|指导用户如何在NPU上基于PyTorch完成MindSpeed MM的安装，内容涵盖硬件与操作系统兼容性说明、驱动固件及CANN基础软件安装，以及基于PyTorch框架下的完整安装流程，帮助用户快速搭建多模态模型训练环境。|-|
|《[MindSpeed MM快速入门（基于Megatron训练后端）](../zh/pytorch/quickstart.md)》|以Wan2.1和Qwen2.5-VL为例，指导开发者基于Megatron训练后端完成微调任务，帮助用户快速上手多模态模型训练。|-|
|《[MindSpeed MM快速入门（基于FSDP2训练后端）](../zh/pytorch/quickstart_fsdp2.md)》|以Qwen3-VL-30B，指导开发者基于FSDP2训练后端完成微调任务，帮助用户快速上手多模态模型训练。|-|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-07-05 08:00:00.0|2026-07-06|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-07-06 10:03:00|2026-07-06|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.101158|2026-07-06|无病毒，无恶意|

### 漏洞修补列表

无
