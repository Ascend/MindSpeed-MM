# Security Statement

## System Security Hardening

1. Enable address space layout randomization (ASLR) at level 2 on the system, also known as **full address space layout randomization**. Run the following command to enable it:

    ```bash
    echo 2 > /proc/sys/kernel/randomize_va_space
    ```

## User Account Recommendations

For security and to minimize permissions, you are advised not to use `root` or other privileged accounts to run MindSpeed MM.

## File Permission Control

1. Set the system `umask` to `0027` or stricter on the host and in containers. This setting ensures that newly created directories have maximum default permissions of `750` and newly created files have maximum default permissions of `640`.
2. Apply appropriate permission controls to sensitive content such as personal data, business assets, source files, and files saved during training. Applicable scenarios include permission control for the MindSpeed MM installation directory and shared datasets in multi-user environments. For recommended permission settings, see Table 1.
3. MindSpeed MM generates training data during data preprocessing and weight files during training. The default file permissions are `640`. You can apply stricter permission controls to generated files as needed.

**Table 1 Recommended maximum permissions for files and directories in different scenarios**

| Type | Maximum Linux permissions |
| ---- | ------------------------- |
| User home directory | 750 (`rwxr-x---`) |
| Program files (including scripts and library files) | 550 (`r-xr-x---`) |
| Program file directory | 550 (`r-xr-x---`) |
| Configuration file | 640 (`rw-r-----`) |
| Configuration file directory | 750 (`rwxr-x---`) |
| Log files, after they complete or are archived | 440 (`r--r-----`) |
| Log files, while they are being written | 640 (`rw-r-----`) |
| Log file directory | 750 (`rwxr-x---`) |
| Debug files | 640 (`rw-r-----`) |
| Debug file directory | 750 (`rwxr-x---`) |
| Temporary file directory | 750 (`rwxr-x---`) |
| Maintenance and upgrade file directory | 770 (`rwxrwx---`) |
| Service data files | 640 (`rw-r-----`) |
| Service data file directory | 750 (`rwxr-x---`) |
| Key components, private keys, certificates, and ciphertext file directory | 700 (`rwx------`) |
| Key components, private keys, certificates, and encrypted ciphertext | 600 (`rw-------`) |
| Encryption and decryption interfaces and scripts | 500 (`r-x------`) |

## Data Security Statement

1. Risk overview:
   The MindSpeed MM model framework loads and saves models. Note that its underlying implementation may use the [Python `pickle`](https://docs.python.org/3/library/pickle.html) module to serialize and deserialize some files. This module poses inherent security risks.

2. Key risk scenarios:
   A key security risk when the `torch.load()` method provided by PyTorch loads model files is the `weights_only=False` setting. With this setting:

   Framework-specific implementations: The native Megatron-LM framework code and the weight conversion scripts provided by MindSpeed MM, which convert the Megatron format to the Hugging Face format, explicitly set `weights_only=False`. Therefore, these loading operations inherit the potential risks of the `pickle` module and allow arbitrary code execution.
   Attack surface: An attacker may craft a malicious model file and exploit a `pickle` deserialization vulnerability to achieve remote code execution (RCE).

3. Critical vulnerability warning (CVE-2025-32434)

   Even when `weights_only` is set to `True`, users still face serious risks, especially with PyTorch versions less than or equal to 2.5.1:

   An attacker can craft a malicious model file in the legacy `.tar` format. The crafted file can bypass the security checks of `weights_only=True`, and successful exploitation can trigger RCE. For details, see CVE-2025-32434.

4. Key security protection measures

   Given these high risks, you are strongly advised to take the following measures:

   Trusted sources: Load model files only from official release channels or highly trusted repositories.
   Integrity verification: After download, use a cryptographic hash, such as SHA-256, to verify the integrity and authenticity of the model file.
   Environment isolation: Run model-loading code in an isolated environment, such as a container or sandbox, and strictly limit system access permissions for that environment. Sandbox escape is a separate security concern.
   Security tools: Use dedicated security tools, such as scanners for `pickle`, to inspect model files and identify potentially malicious serialized objects.
   PyTorch upgrade: Do not use PyTorch versions less than or equal to 2.5.1. Upgrade immediately to a later version that fixes CVE-2025-32434.

5. References

   [`torch.load()` documentation](https://pytorch.org/docs/main/generated/torch.load.html#torch.load), including the description and risks of the `weights_only` parameter.

   [PyTorch distributed communication documentation](https://pytorch.org/docs/main/distributed.html#collective-functions).

## Runtime Security Statement

1. You are advised to write training scripts based on the available runtime resources. If a training script does not match the available resources, errors may occur and the process may exit unexpectedly. For example, the memory required to load a dataset may exceed the available memory, or the data generated locally by a training script may exceed the available drive space.
2. MindSpeed MM uses PyTorch and `torch_npu` internally. Version incompatibility may cause runtime errors. For details, see the [security statement](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/SECURITYNOTE.md) for PyTorch and `torch_npu`.
3. The MindSpeed MM dependencies `transformers` and `datasets` may set `trust_remote_code=True` when they use the `from_pretrained` method. This setting directly executes code downloaded from a remote repository. The code may contain malicious logic or backdoors, which expose the system to security threats such as code injection attacks. Ensure that downloaded models and data are secure.

## Public API Statement

MindSpeed MM has not yet released a wheel package. Therefore, it does not provide any formal public interface. All functionality is invoked through shell scripts. The entry scripts are:

- [evaluate_gen](../../evaluate_gen.py)
- [evaluate_vlm](../../evaluate_vlm.py)
- [inference_qihoo](../../inference_qihoo.py)
- [inference_sora](../../inference_sora.py)
- [inference_videoalign](../../inference_videoalign.py)
- [inference_vlm](../../inference_vlm.py)
- [posttrain_flux_dancegrpo](../../posttrain_flux_dancegrpo.py)
- [posttrain_qwen2vl_dpo](../../posttrain_qwen2vl_dpo.py)
- [posttrain_sora_dpo](../../posttrain_sora_dpo.py)
- [pretrain_ae](../../pretrain_ae.py)
- [pretrain_deepseekvl](../../pretrain_deepseekvl.py)
- [pretrain_internvl](../../pretrain_internvl.py)
- [pretrain_lumina](../../pretrain_lumina.py)
- [pretrain_qwen2vl](../../pretrain_qwen2vl.py)
- [pretrain_sora](../../pretrain_sora.py)
- [pretrain_transformers](../../pretrain_transformers.py)
- [pretrain_videoalign](../../pretrain_videoalign.py)
- [pretrain_vlm](../../pretrain_vlm.py)
- [pretrain_whisper](../../pretrain_whisper.py)

## Communication Security Hardening

See [Communication Security Hardening](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA).

## Communication Matrix

See [Communication Matrix](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5).

## Special Scenarios

| Scenario | Usage Method | Port | Possible Risk |
| -------- | ------------ | ---- | ------------- |
| When MindSpeed MM performs a training task with the Megatron backend, it adds `3 × number of NPUs` random ports by default each time it initializes a model parallel group. If multiple distributed optimizers are enabled, it also adds `number of distributed optimizers × number of NPUs` random ports and configures one port through `master-port`. This port is the same as the port configured through `master-port` for `torch_npu`. | MindSpeed MM calls the native Megatron function `mpu.initialize_model_parallel` to initialize model parallel groups and uses PyTorch distributed training APIs to start tasks. | [1024, 65520] | Incorrect network configuration may cause port conflicts or connection problems and reduce training efficiency. |

### Public Network Address Statement

For public network addresses used in the code, see [public_address_statement.md](./public_address_statement.md).
