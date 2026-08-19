# MindSpeed MM Docker 镜像概述

## 快速参考

| 项目 | 说明 |
| ------ | ------ |
| **镜像名称** | mindspeed-mm |
| **维护者** | MindSpeed MM 团队 |
| **源码仓库** | [https://gitcode.com/Ascend/MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) |
| **Dockerfile 路径** | `docker/` |
| **许可证** | Apache-2.0 |

## MindSpeed MM

MindSpeed MM：面向大规模分布式训练的昇腾多模态大模型套件，支持业界主流多模态大模型训练，旨在为华为 [昇腾芯片](https://www.hiAscend.com/) 提供端到端的多模态训练解决方案, 包含预置业界主流模型，数据工程，分布式训练及加速，预训练、微调、后训练、在线推理任务等特性。

MindSpeed MM 镜像基于 Ubuntu 22.04 和 openEuler 24.03 两种操作系统，支持 x86_64 和 aarch64（ARM64）两种 CPU 架构。镜像中预安装了以下软件：

- **PyTorch** + **TorchNPU**：深度学习框架
- **decord 0.6.0**：高效视频解码库
- **CANN**：华为昇腾 AI 处理器基础软件栈

由于不同模型的依赖环境存在差异，镜像中仅预安装了上述基础依赖包。用户在拉取镜像并启动容器后，需根据目标模型的 README 文件，在 base 环境中手动安装该模型所需的额外依赖。

镜像下载：请访问 [镜像中心](https://www.hiascend.com/developer/ascendhub) 搜索 mindspeed-mm，获取对应的 `docker pull` 命令。
当前镜像支持 `openEuler 24.03` 与 `ubuntu22.04` 两种操作系统，并同时提供 `x86_64` 与 `aarch64`（ARM64）两种 CPU 架构（x86 与 aarch64 二合一）。

## 镜像 Tag 关键字段描述

镜像 Tag 命名遵循模板：

`{版本号}-{CANN版本}-{TorchNPU版本}-{适用产品信息}-{操作系统}-{Python版本}`

各字段均为必选，字段顺序不可调整，连接符均使用 `-`。

| 字段 | 必选 | 说明 | 示例值 |
| ------ | ------ | ------ | -------- |
| 版本号 | 是 | MindSpeed MM 版本标识(v表示version，数字代表分支) | v26.0.0, v26.1.0 |
| CANN版本 | 是 | `cann` + 版本号 | cann9.1.0 |
| TorchNPU版本 | 是 | `torch_npu` + 版本号 | torch_npu2.7.1.post8 |
| 适用产品信息 | 是 | NPU 芯片类型（小写） | 910b, a3, 950 |
| 操作系统 | 是 | 操作系统 | openeuler24.03, ubuntu22.04 |
| Python版本 | 是 | `py` + 版本号 | py3.11 |

> 镜像仓库中的 Tag 为 x86 与 aarch64 二合一的多架构镜像，**不包含** `x86_64`/`aarch64` 架构后缀。仅当通过 Dockerfile 在本地构建时，才会生成带架构后缀（`-x86_64` / `-aarch64`）的 Tag。

### 示例 Tag

| Tag | 版本号 | CANN | TorchNPU | NPU | 操作系统 | Python |
| ----- | ----- | ----- | ----- | ----- | --------- | -------- |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11` | v26.1.0 | 9.1.0 | 2.7.1.post8 | A3 | openEuler 24.03 | 3.11 |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11` | v26.1.0 | 9.1.0 | 2.7.1.post8 | 910B | openEuler 24.03 | 3.11 |

### Tag 含义说明

以 `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11` 为例：

| 字段 | 值 | 含义 |
| ------ | ------ | ------ |
| 版本号 | `v26.1.0` | MindSpeed MM Git tag（分支名：26.1.0） |
| CANN版本 | `cann9.1.0` | 基于 CANN 9.1.0 |
| TorchNPU版本 | `torch_npu2.7.1.post8` | TorchNPU 2.7.1.post8 |
| 适用产品信息 | `a3` | 适用于昇腾 A3 服务器 |
| 操作系统 | `openeuler24.03` | 基于 openEuler 24.03 |
| Python版本 | `py3.11` | Python 3.11 |

> 通过 Dockerfile 本地构建后，会生成带架构后缀的 Tag，如 `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11-aarch64`（aarch64）与 `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11-x86_64`（x86_64）。
> CANN 版本、NPU 类型（910b/a3/950）、操作系统和 Python 版本默认来源于构建时基础镜像的 tag；Python 版本可通过 `--python-version` 覆盖。TorchNPU 版本来源于 `--torch-npu-version` 参数。

Dockerfile 命名：

- `Dockerfile`：统一的 dev 镜像构建文件，通过构建参数支持所有 NPU 类型（910b/a3/950）和操作系统版本
- `Dockerfile.ci`：CI 镜像构建文件，在 dev 镜像基础上叠加多版本 conda 环境（通过 `--build-ci` 使用）

## 支持的 Tags 及 Dockerfile 链接

### 最新版本 v26.1.0

如下是当前最新版本 v26.1.0 的所有镜像 Tag（`cann9.1.0` + `torch_npu2.7.1.post8`）。最新 Tag 为 x86 与 aarch64 二合一的镜像，使用 [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.1.0/docker/Dockerfile) 构建完成，可通过 ascendhub 获取对应的 `docker pull` 命令（[mindspeed-mm 镜像中心](https://www.hiascend.com/developer/ascendhub/detail/6857f6fc2cfa4a678710a7075426ee5e)）。

| Tag | Dockerfile | 说明 |
| --- | --- | --- |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.1.0/docker/Dockerfile) | 910B + openEuler 24.03，x86_64/aarch64 二合一 |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.1.0/docker/Dockerfile) | 910B + Ubuntu 22.04，x86_64/aarch64 二合一 |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.1.0/docker/Dockerfile) | A3 + openEuler 24.03，x86_64/aarch64 二合一 |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu22.04-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.1.0/docker/Dockerfile) | A3 + Ubuntu 22.04，x86_64/aarch64 二合一 |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.1.0/docker/Dockerfile) | 950 + openEuler 24.03，x86_64/aarch64 二合一 |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-ubuntu22.04-py3.11` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.1.0/docker/Dockerfile) | 950 + Ubuntu 22.04，x86_64/aarch64 二合一 |

> 上表中的最新 Tag 为多架构镜像（x86 与 aarch64 二合一）。该 Dockerfile 实际构建完成后会生成带架构后缀的 Tag，例如 `-aarch64` / `-x86_64` 后缀（如 `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11-aarch64`）。历史版本所有 Tag 请参考 [Supported Tags](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docker/supported_tags.md)。

## 项目目录结构规范

Docker 项目目录遵循清晰的分层结构，便于维护和扩展：

### 核心目录结构

```text
docker/
├── Dockerfile                 # 统一 dev 镜像 Dockerfile，支持多 NPU 类型和操作系统
├── Dockerfile.ci              # CI 镜像 Dockerfile（基于 dev 镜像叠加）
├── build.sh                   # 镜像构建脚本，支持多种参数配置
├── OVERVIEW.md                # 英文版说明文档
├── OVERVIEW.zh.md             # 中文版说明文档
├── supported_tags.md          # 历史 Tag 列表
└── scripts/                   # 脚本目录
    └── ci/                    # CI 构建脚本与版本配置
```

### 目录说明

1. **Dockerfile**：统一的 dev 镜像构建文件，通过构建参数支持所有 NPU 类型和操作系统版本
2. **Dockerfile.ci**：CI 镜像构建文件，在 dev 镜像基础上叠加多版本 conda 环境（`--build-ci` 使用）
3. **build.sh**：镜像构建脚本，提供灵活的参数配置和自动识别功能
4. **scripts/**：按脚本功能进行目录组织（如 `ci/` 存放 CI 构建相关脚本）

### 脚本使用机制

`docker/build.sh` 脚本在构建过程中根据 `-v` 参数指定的版本号（默认为 26.1.0）作为 git clone MindSpeed-MM 的分支。

## 1. 镜像使用指导

**重要提示：**

1. 由于不同模型的依赖环境存在差异，镜像中仅预安装了 PyTorch、TorchNPU 和 decord 基础依赖包。用户在拉取镜像并启动容器后，需根据目标模型的 README 文件，在 base 环境中手动安装该模型所需的依赖环境。
2. 如果环境NPU驱动程序未安装在默认路径(/usr/local/Ascend/driver), 在执行下列docker运行命令时，需在命令中补充路径信息。以路径"/usr/local/npu/driver"为例:

    ```bash
    # 基本运行
    docker run -it --rm \
        -e LD_LIBRARY_PATH="/usr/local/npu/driver/lib64/driver:/usr/local/npu/driver/lib64/common:$LD_LIBRARY_PATH" \
        mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11 bash
    ```

### 前置要求（可选）

#### 安装驱动

主机上必须安装与容器内 CANN 版本兼容的昇腾 NPU 驱动。请访问 [CANN 版本配套网站](https://www.hiascend.com/developer/download/compatibility) 了解驱动与 CANN 版本的对应关系。

### 运行镜像

```bash

# 使用 NPU 设备 & 挂载数据目录运行 （示例：设备 /dev/davinci1）
# 根据实际情况修改 ascend-toolkit 路径
# 假设您的 NPU 设备安装在 /dev/davinci1 上，并且 NPU 驱动程序安装在 /usr/local/Ascend 上：
docker run -it \
    --name mm_container \
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

### 在 950 系列 aarch64 架构产品上运行 CANN 容器

在 950 系列 aarch64 架构产品上，可通过以下命令运行 CANN 容器：

```bash

docker run \
    --name mm_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    --device /dev/ummu \
    --device /dev/uburma \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11 bash
```

### 内置环境

镜像包含以下预配置环境：

| 环境 | 说明 | 工作目录 |
| ------ | ------ | --------- |
| base | 基础环境，包含 PyTorch、TorchNPU、decord、MindSpeed MM | /workspace/MindSpeed-MM |

**环境说明：**

- 考虑到不同模型的依赖环境存在差异，镜像中仅预安装了 PyTorch、TorchNPU 和 decord 基础依赖包。
- 用户在拉取镜像并启动容器后，需根据目标模型的 README 文件，在 base 环境中手动安装该模型所需的依赖环境。

## 2. 本地自定义安装指导

### 构建脚本参数说明

构建脚本 `build.sh` 支持多种参数配置。其中 CANN 版本、NPU 类型（910b/a3/950）和操作系统**只能**通过解析 `--base-image` 的 tag 获取，不支持手动指定。Python 版本默认自动识别，但可通过 `--python-version` 手动指定。

| 参数 | 说明 | 默认值 |
| ------ | ------ | ------------ |
| `--base-image` | **必选。** 完整基础镜像名称，CANN 版本、NPU 类型（910b/a3/950）、操作系统和 Python 版本均从镜像 tag 自动识别 | 无（必需） |
| `--python-version` | conda base 环境的 Python 版本（如 3.11/3.10/3.12）。用于选择对应的 Miniconda 安装器，最终决定 conda base 环境的 Python 版本 | 从 base 镜像 tag 自动识别 |
| `-v, --version` | MindSpeed MM 版本标识，同时作为 Git 分支名称 | 26.1.0 |
| `--tag` | 自定义镜像 tag（覆盖默认 tag；CI 构建自动追加 `-ci`） | 自动生成 |
| `-n, --no-cache` | 构建时不使用缓存 | 无 |
| `--torch-version` | PyTorch 版本（在线安装） | 2.7.1 |
| `--torch-npu-version` | TorchNPU 版本（在线安装） | 2.7.1.post8 |
| `--build-ci` | 在 dev 镜像基础上构建 CI 镜像（多版本 conda 环境），输出 tag 追加 `-ci` | 无 |
| `--cleanup-on-fail` | 构建失败时清理悬空镜像/容器 | 无 |

### 基础构建示例

```bash
cd docker

# 构建 A3 + openEuler 镜像
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11

# 构建 910B + Ubuntu 镜像
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11

# 构建 950 + openEuler 镜像，并指定 conda base 环境为 Python 3.12
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-openeuler24.03-py3.12 --python-version 3.12

# 指定 PyTorch 版本构建
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11 --torch-version 2.7.1 --torch-npu-version 2.7.1.post4

# 指定版本构建
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11 -v 26.1.0

# 构建 CI 镜像（在 dev 镜像基础上，tag 追加 -ci）
bash build.sh --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11 --build-ci
```

### 自动下载功能说明

构建脚本支持自动下载以下资源，请确保网络通畅：

1. **Miniconda 安装器**：按 `--python-version` 选择对应变体自动下载
2. **decord 依赖包**：ARM 架构下自动下载
3. **基础镜像**：当指定 `--base-image` 且本地不存在时自动拉取

## 3. 自定义镜像构建/使用指导

### 自动识别基础镜像

构建脚本会自动从基础镜像 tag 中识别关键信息（`--base-image` 参数为必选）：

1. **CANN 版本识别**：从镜像 tag 中提取 CANN 版本号（如 `9.0.0`）
2. **NPU 类型识别**：从镜像 tag 中识别 NPU 类型（如 `a3`）
3. **操作系统识别**：从镜像 tag 中识别 `openeuler24.03` 或 `ubuntu22.04`
4. **Python 版本识别**：从镜像 tag 中的 `py<x.y>` 字段提取 Python 版本（如 `3.11`）
5. **自动生成镜像 tag**：基于识别到的信息自动生成符合命名规则的镜像 tag

### 最佳实践示例

以下示例展示了如何使用自定义基础镜像构建 MindSpeed MM 镜像：

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 自定义配置
BASE_IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.12"
TORCH_VERSION="2.7.1"
TORCH_NPU_VERSION="2.7.1.post8"
MINDSPEED_MM_VERSION="26.1.0"
PYTHON_VERSION="3.11"

# 执行构建
bash "${SCRIPT_DIR}/build.sh" \
    --base-image "$BASE_IMAGE" \
    --torch-version "$TORCH_VERSION" \
    --torch-npu-version "$TORCH_NPU_VERSION" \
    -v "$MINDSPEED_MM_VERSION" \
    --python-version "$PYTHON_VERSION" \
    --cleanup-on-fail
```

**关键特性说明：**

1. **自动识别**：脚本会自动从 `BASE_IMAGE` 中识别 CANN 版本（9.1.0）、NPU 类型（910b）和操作系统（openeuler24.03）。如果`BASE_IMAGE`在系统中不存在，会自动拉取。
2. **指定 Python 版本**：`--python-version` 用于指定模型运行 Python 版本。
3. **自动生成 tag**：基于识别结果自动生成符合命名规则的镜像 tag（如 `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.11-aarch64`）。
4. **自动下载**：如果本地没有 Miniconda 安装器或 decord 依赖，脚本会自动下载
5. **失败清理**：`--cleanup-on-fail` 参数确保构建失败时清理悬空资源

### 二次开发

基于此镜像创建自定义 Dockerfile：

```dockerfile
FROM mindspeed-mm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.11

RUN pip install your-package==1.0.0

COPY . /workspace/your-project

WORKDIR /workspace/your-project
```

构建并运行（示例：设备 /dev/davinci1）：

```bash
# 根据实际情况修改 ascend-toolkit 路径
docker build -t my-mindspeed-app:latest .
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
    my-mindspeed-app:latest bash
```

## 许可证

MindSpeed MM 基于 Apache License 2.0 许可证发布。详见 [LICENSE](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/LICENSE) 文件。

本镜像基于 CANN 基础镜像构建，其中包含的 CANN 系列软件的许可证信息请参见 [CANN 软件许可证](https://www.hiascend.com/legal/cannua-download?isNewCon=true)。

与所有 Docker 镜像一样，这些镜像可能还包含受其他许可证约束的其他软件（例如基础发行版中的 Bash，以及所包含主要软件的任何直接或间接依赖项）。

对于预构建镜像的任何使用，镜像用户有责任确保对此镜像的任何使用符合其中包含的所有软件的相关许可证。

## 免责声明

发布的昇腾软件镜像均是社区版本，不对商业负责、仅作为生产实践的参考。
