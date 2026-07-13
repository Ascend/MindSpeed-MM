# Release Notes

## Version Compatibility

### Product Version Information

<table><tbody><tr><th class="firstcol" valign="top" width="26.25%"><p>Product</p></th>
<td class="cellrowborder" valign="top" width="73.75%"><p><span>MindSpeed MM</span></p></td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Version</p></th>
<td class="cellrowborder" valign="top" width="73.75%" ><p>26.0.0</p></td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Version Type</p></th>
<td class="cellrowborder" valign="top" width="73.75%" ><p>Official release</p></td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p>Release Date</p></th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p>April 2026</p></td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Maintenance Period</p></th>
<td class="cellrowborder" valign="top" width="73.75%"><p>6 months</p></td>
</tr>
</tbody>
</table>

> [!NOTE]
> For version maintenance of MindSpeed, see [Branch Maintenance Policy](https://gitcode.com/Ascend/MindSpeed/tree/26.0.0_core_r0.12.1#branch-maintenance-policy) for details.

### Related Product Version Compatibility

**Table 1** MindSpeed MM compatibility

|MindSpeed MM Code Branch|Megatron Version|CANN Version|Ascend Extension for PyTorch Version|Python Version|PyTorch Version|
|--|--|--|--|--|--|
|26.0.0|core_v0.12.1|9.0.0|26.0.0|Python3.10|2.7.1|
|2.3.0|core_v0.12.1|8.5.0|7.3.0|Python3.10|2.7.1|
|2.2.0|core_v0.12.1|8.3.RC1|7.2.0|Python3.10|2.7.1|

>[!NOTE]
>Select the MindSpeed MM code branch as needed to download the source code and perform installation.

## Version Compatibility Notes

|MindSpeed MM Version|CANN Version|Ascend Extension for PyTorch Version|
|--|--|--|
|26.0.0|CANN 9.0.0<br>CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>|26.0.0|
|2.3.0|CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>|7.3.0|
|2.2.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|7.2.0|

## Version Usage Notes

None

## Feature Change Description

### New Features

|Component|Description|Purpose|
|--|--|--|
|MindSpeed MM|New models|Supports HunyuanVideo-1.5 I2V/T2V, CosyVoice3.0, FLUX2.0.|
|MindSpeed MM|New features|Qwen3VL 30B supports EP, Qwen3VL 30B and Wan2.2 T2V 14B support LoRA.|
|MindSpeed MM|Security hardening|Supports PMCC protection for multimodal understanding model fine-tuning.|

### Removed Features

None

### API Change Description

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

|Document|Description|Release Notes|
|--|--|--|
|*[MindSpeed MM Installation Guide](pytorch/install_guide.md)*|Provides instructions on how to install MindSpeed MM on NPUs using PyTorch. It covers hardware and OS compatibility, installation of drivers, firmware, and the CANN software stack, as well as the complete PyTorch-based installation workflow. It is designed to help users quickly set up a multimodal model training environment.|-|
|*[MindSpeed MM Quick Start](pytorch/quickstart.md)*|Using Wan2.1 and Qwen2.5-VL as examples, this guide instructs developers on fine-tuning the Wan2.1 and Qwen2.5-VL models under the PyTorch framework, helping them quickly get started with multimodal model training.|-|

## Virus Scan and Vulnerability Patch List

### Virus Scan Result

|Antivirus Software|Antivirus Software Version|Virus Database Version|Scan Time|Scan Result|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-04-01 08:00:00.0|2026-04-02|No viruses, no malware|
|Kaspersky|12.0.0.6672|2026-04-02 10:05:00|2026-04-02|No viruses, no malware|
|Bitdefender|7.5.1.200224|7.100588|2026-04-02|No viruses, no malware|

### Vulnerability Patch List

None
