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
<td class="cellrowborder" valign="top" width="73.75%" ><p>26.0.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" ><p>正式版本</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p>发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p>2026年4月</p>
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
> 有关MindSpeed的版本维护，具体请参见[分支维护策略](https://gitcode.com/Ascend/MindSpeed/tree/26.0.0_core_r0.12.1#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5)。

### 相关产品版本配套说明

**表 1**  MindSpeed MM配套表

|MindSpeed MM代码分支名称|CANN版本|Ascend Extension for PyTorch版本|Python版本|PyTorch版本|MindSpeed-Core-MS版本|
|--|--|--|--|--|--|
|master(在研版本)|在研版本|在研版本|Python3.10|2.7.1|在研版本|
|2.3.0|8.5.0|7.3.0|Python3.10|2.7.1|r0.5.0|
|2.2.0|8.3.RC1|7.2.0|Python3.10|2.7.1|r0.4.0|

>[!NOTE]
>用户可根据需要选择MindSpeedMM代码分支下载源码并进行安装。

## 版本兼容性说明

|MindSpeed MM版本|CANN版本|Ascend Extension for PyTorch版本|MindSpeed-Core-MS版本|
|--|--|--|--|
|2.3.0|CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>|7.3.0|r0.5.0|
|2.2.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|7.2.0|r0.4.0|

## 版本使用注意事项

无

## 更新说明

### 新增特性

无

|组件|描述|目的|
|--|--|--|
|MindSpeed MM|新模型|支持HunyuanVideo-1.5 I2V/T2V，CosyVoice3.0，FLUX2.0|
|MindSpeed MM|新特性|Qwen3VL 30B支持EP，Qwen3VL 30B和Wan2.2 T2V 14B支持LoRA|
|MindSpeed MM|安全加固|支持多模态理解大模型微调PMCC保护|

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
|--|--|--|
|《[MindSpeed MM安装指导](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.0.0/docs/zh/install_guide.md)》|指导如何在NPU上基于PyTorch和MindSpore框架完成MindSpeed MM的安装，内容涵盖硬件与操作系统兼容性说明、驱动固件及CANN基础软件安装，以及两种框架下的完整安装流程，帮助用户快速搭建大语言模型分布式训练环境。|-|
|《[MindSpeed MM迁移调优指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.0.0/docs/zh/pytorch/model-migration.md)》|以Qwen2-VL-7B为例，基于华为昇腾芯片产品（NPU）开发，将原本运行在GPU或其他硬件平台的深度学习模型迁移至NPU， 并保障模型在合理精度误差范围内高性能运行，并对昇腾芯片进行极致优化以发挥芯片最大性能。|-|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

无

### 漏洞修补列表

无
