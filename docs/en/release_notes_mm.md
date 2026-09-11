# Release Notes

## Version Compatibility Notes

### Product Version Information

<table>
  <tbody>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Product Name</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Product Version</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>26.1.0</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Version Type</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>Official Release</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Component Name</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed MM</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Release Date</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>July, 2026</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>Maintenance Period</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>6 months</p></td>
    </tr>
  </tbody>
</table>

> [!NOTE]
>
> For the version maintenance policy of MindSpeed MM, see [Version Maintenance](https://gitcode.com/Ascend/MindSpeed-MM#%E7%89%88%E6%9C%AC%E7%BB%B4%E6%8A%A4).

### Related Product Version Compatibility

**Table 1**  MindSpeed MM compatibility table

| MindSpeed MM Version | MindSpeed Core Code Branch | Megatron Version | PyTorch Version | TorchNPU Version | CANN Version | Triton-Ascend Version | Python Version |
| ---------------- | ------------------ | ------------ | -----------  | ------------- |--------------------- |-----------------| ------------------- |
| 26.1.0           | 26.1.0_core_r0.12.1 | core_v0.12.1  | 2.7.1       | 26.1.0        | 9.1.0  | 3.2.1           | Python3.10      |
| 26.0.0           | 26.0.0_core_r0.12.1 | core_v0.12.1  | 2.7.1       | 26.0.0        | 9.0.0  | 3.2.1           | Python3.10      |

>[!NOTE]
>
>- Users can select the MindSpeed MM code branch as needed to download the source code and perform installation.
>- The Triton-Ascend version is strongly bound to the CANN version. The use of Triton-Ascend must correspond one-to-one with the CANN version. For details, see [Triton-Ascend Compatibility](https://gitcode.com/Ascend/triton-ascend#%E5%85%BC%E5%AE%B9%E6%80%A7).

## Version Compatibility Notes

> [!NOTE]
>
> In the tables in this section, "/" indicates incompatibility, and "Y" indicates compatibility.

**Table 2**  MindSpeed MM and TorchNPU version compatibility

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
    <th colspan="4">TorchNPU</th>
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

**Table 3**  MindSpeed MM and CANN version compatibility

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
    <th colspan="4">CANN</th>
  </tr>
  <tr>
    <th>8.3.RCX</th>
    <th>8.5.X</th>
    <th>9.0.X</th>
    <th>9.1.X</th>
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

## Version Usage Notes

None

## Release Notes

### New Features

|Component|Description|Purpose|
|--|--|--|
|MindSpeed MM|New model|Supports Qwen3.5 and Kimi-K2.5|
|MindSpeed MM|New feature|Qwen3.5 and Kimi-K2.5 support asynchronous activation offloading, ChunkLoss, and ChunkMBS|
|MindSpeed MM|New hardware support|Supports Ascend 950 products|

### Removed Features

None

### Interface Change Description

None

### Resolved Issues

None

### Known Issues

None

## Upgrade Impact

### Impact on the Current System During the Upgrade

- Impact on services

    Service interruption occurs during the software version upgrade.

- Impact on network communication

    No impact on communication.

### Impact on the Current System After the Upgrade

None

## Related Documents

|Document Name|Introduction|Release Notes|
|--|--|--|
|*[MindSpeed MM Software Installation](../en/pytorch/install_guide.md)*|Guides users to complete the installation of MindSpeed MM based on PyTorch on NPUs. It covers hardware and operating system compatibility, driver firmware and CANN base software installation, as well as the complete installation process under the PyTorch framework, helping users quickly set up a multimodal model training environment.|-|
|*[MindSpeed MM Quick Start (Based on Megatron Training Backend)](../en/pytorch/quickstart.md)*|Using Wan2.1 and Qwen2.5-VL as examples, guides developers to complete fine-tuning tasks based on the Megatron training backend, helping users quickly get started with multimodal model training.|-|
|*[MindSpeed MM Quick Start (Based on FSDP2 Training Backend)](../en/pytorch/quickstart_fsdp2.md)*|Using Qwen3-VL-30B as an example, guides developers to complete fine-tuning tasks based on the FSDP2 training backend, helping users quickly get started with multimodal model training.|-|

## Virus Scan and Vulnerability Patching List

### Virus Scan Result

|Antivirus Software Name|Antivirus Software Version|Virus Database Version|Scan Time|Scan Result|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-07-05 08:00:00.0|2026-07-06|Virus-free and Malware-free|
|Kaspersky|12.0.0.6672|2026-07-06 10:03:00|2026-07-06|Virus-free and Malware-free|
|Bitdefender|7.5.1.200224|7.101158|2026-07-06|Virus-free and Malware-free|

### Vulnerability Patch List

None
