# Security Declaration

## System Security Hardening

1. It is recommended that users enable level-2 Address Space Layout Randomization (ASLR) in the system. The following method can be used for reference:

    ```bash
    echo 2 > /proc/sys/kernel/randomize_va_space
    ```

## Running User Recommendations

For security and least-privilege considerations, it is not recommended to use administrator accounts such as `root` to run MindSpeed MM.

## File Permission Control

1. It is recommended that users set the system umask value to `0027` or higher on the host (including the physical machine) and in containers, ensuring that the default maximum permission for newly created folders is `750` and the default maximum permission for newly created files is `640`.
2. It is recommended that users properly control permissions for sensitive content such as personal data, commercial assets, source files, and various files saved during training. Scenarios involved include permission control for the MindSpeed MM installation directory and permission control for shared datasets used by multiple users. The permissions can be set by referring to Table 1.
3. MindSpeed MM generates training data during data preprocessing and generates weight files during training. The default file permission is `640`, and users can perform advanced control over the permissions of generated files according to actual requirements.

**Table 1 Recommended maximum permissions for files (folders) in various scenarios**

| Type          | Maximum Linux Permission |
| --------------- | --------------------|
| User home directory                          |    750 (rwxr-x---)                |
| Program files (including script files, library files, etc.)      |    550 (r-xr-x---)                |
| Program file directory                        |    550 (r-xr-x---)                |
| Configuration file                            |    640 (rw-r-----)                |
| Configuration file directory                        |    750 (rwxr-x---)                |
| Log file (recording completed or archived)      |    440 (r--r-----)                |
| Log file (being recorded)                 |    640 (rw-r-----)                |
| Log file record                        |    750 (rwxr-x---)                |
| Debug file                          |    640 (rw-r-----)                |
| Debug file directory                      |    750 (rwxr-x---)                 |
| Temporary file directory                       |     750 (rwxr-x---)                |
| Maintenance and upgrade file directory                    |    770 (rwxrwx---)                |
| Business data file                       |     640 (rw-r-----)                |
| Business data file directory                   |     750 (rwxr-x---)                |
| Key component, private key, certificate, and ciphertext file directory   |     700 (rwx------)                |
| Key component, private key, certificate, and encrypted ciphertext      |     600 (rw-------)                |
| Encryption/decryption interface and encryption/decryption script             |     500 (r-x------)                |

## Data Security Declaration

1. Risk overview
   The MindSpeed MM framework performs model loading and saving operations. It is important to note that its underlying implementation may use the [Python pickle](https://docs.python.org/3/library/pickle.html) module for serialization/deserialization of certain files, and this module carries inherent security risks.

2. Core risk scenarios
   When model files are loaded via the `torch.load()` method provided by PyTorch, a critical security risk lies in setting `weights_only=False`. Under this setting:

   Specific framework implementations: In the native code calls of the Megatron-LM framework and the weight conversion scripts provided by MindSpeed MM (which convert the Megatron format to the Hugging Face format), `weights_only=False` is explicitly set. This means that these loading operations inherit the potential dangers of the pickle module, allowing arbitrary code execution.
   Attack surface: An attacker may craft a malicious model file to exploit the pickle deserialization vulnerability and achieve remote code execution (RCE).

3. Critical vulnerability warning (CVE-2025-32434)

   Even when `weights_only` is set to `True,` users still face serious risks, especially when using PyTorch version <= 2.5.1:

   An attacker can exploit legacy `.tar` model files to craft a malicious model. Such crafting can bypass the security check mechanism of `weights_only=True`. Successful exploitation can trigger RCE. Users must refer to CVE-2025-32434 for details.

4. Key security protection measures

   Given the high risks described above, it is strongly recommended that:

   Trusted sources: Load model files only from official release channels or highly trusted repositories.
   Integrity verification: After downloading, always verify the integrity and source authenticity of model files using cryptographic hashes (such as SHA-256).
   Environment isolation: Run model loading code in an isolated environment (such as a container or sandbox), and strictly restrict the system access permissions of that environment (sandbox escape is a separate security concern).
   Security tools: Use dedicated security tools (such as scanners targeting Pickle) to inspect model files and identify potentially malicious serialized objects.
   PyTorch version upgrade: Avoid using PyTorch <= 2.5.1. Upgrade immediately to a higher version that has fixed the CVE-2025-32434 vulnerability.

5. Reference

   [torch.load() documentation](<https://pytorch.org/docs/main/generated/torch.load.html#torch.load>) (including `weights_only` description and its risks)

   [PyTorch distributed communication documentation](<https://pytorch.org/docs/main/distributed.html#collective-functions>)

## Runtime Security Declaration

1. It is recommended that users write corresponding training scripts based on the actual resource conditions. If the training script does not match the resource conditions, such as the dataset loading memory size exceeding the memory capacity limit, or the training script generating local data exceeding the disk space size, errors may occur and cause the process to exit unexpectedly.
2. MindSpeed MM internally uses PyTorch and TorchNPU, which may cause runtime errors due to version mismatch. For details, refer to the PyTorch and TorchNPU [security declaration](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md).
3. When the dependency libraries Transformers and datasets of MindSpeed MM use the `from_pretrained` method, there are cases where `trust_remote_code=True` is configured. This setting directly executes code downloaded from remote repositories, which may contain malicious logic or backdoor programs, exposing the system to security threats such as code injection attacks. Users must ensure the security of the models and data they download.

## Public API Declaration

MindSpeed MM has not yet released a wheel package and has no formal public APIs. All functions are invoked through shell scripts. The 19 entry scripts are:

- [evaluate_gen](./evaluate_gen.py)
- [evaluate_vlm](./evaluate_vlm.py)
- [inference_sora](./inference_sora.py)
- [inference_vlm](./inference_vlm.py)
- [posttrain_flux_dancegrpo](./posttrain_flux_dancegrpo.py)
- [posttrain_sora_dpo](./posttrain_sora_dpo.py)
- [pretrain_sora](./pretrain_sora.py)
- [pretrain_transformers](./pretrain_transformers.py)
- [pretrain_vlm](./pretrain_vlm.py)

## Communication Security Hardening

[Communication Security Hardening Notes](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA)

## Communication Matrix

[Communication Matrix Description](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md#communication-matrix)

## Special Scenarios

| Scenario                                                                             | Usage                                                                                                        | Port           | Potential Risk                                                   |
|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------| -------------- | ------------------------------------------------------------ |
| When using MindSpeed MM for training tasks, in the Megatron backend scenario, each time the model parallel group is initialized, (3 \* number of NPUs) random ports are added by default. When multiple distributed optimizers are enabled, an additional (number of distributed optimizers \* number of NPUs) random ports are added, and one master-port is configured (this port is the same as the TorchNPU master-port). | MindSpeed MM calls the Megatron native function `mpu.initialize_model_parallel` to initialize the model parallel group, and starts any task by using PyTorch distributed training related APIs. | Within [1024,65520] | Incorrect network configuration may cause port conflicts or connection issues, affecting training efficiency.       |

## Public Address Declaration

For details, see [public_address_statement.md](./docs/en/public_address_statement.md).

## Vulnerability Response

We attach great importance to the security of the community version. The Mind open-source community receives, investigates, and discloses security vulnerabilities related to this community. For details, see [Ascend Vulnerability Response](https://gitcode.com/Ascend/community/blob/master/docs/security.md).
