
# <p align="center"> <img src="sources/images/mm_logo.png" height="103px" width="700px"> </p>

<p align="center">
    <a href="./LICENSE">
        <img alt="Badge" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
    <a href="https://gitcode.com/Ascend/MindSpeed-MM">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>

# 简介

---

MindSpeed MM：面向大规模分布式训练的昇腾多模态大模型套件，支持业界主流多模态大模型训练，旨在为华为 [昇腾芯片](https://www.hiAscend.com/) 提供端到端的多模态训练解决方案, 包含预置业界主流模型，数据工程，分布式训练及加速，预训练、微调、后训练、在线推理任务等特性。

# 未来规划

---

📅未来规划会动态刷新在[MindSpeed MM RoadMap](https://gitcode.com/Ascend/MindSpeed-MM/issues/176)中，欢迎大家通过此链接进行互动并提出诉求

# 社区会议

---

- MindSpeed系列TC及SIG会议安排请查看[Ascend会议中心](https://meeting.ascend.osinfra.cn/)

# 加入我们

---

为了交流开发经验、分享使用心得、及时获取项目更新，我们创建了MindSpeed MM官方微信群。

无论你是正在使用这个项目，还是有奇思妙想，都欢迎加入👋

加入方式：

1. 直接扫码加入微信交流群（二维码7天有效，定期更新，当前1群已达到扫码加入人数上限，可加入2群）
2. 添加昇腾开源小助手，获取群链接，进入MindSpeed MM社区交流群

<div style="display: flex; justify-content: flex-start; gap: 30px; align-items: flex-start; padding-left: 60px;">
  <div style="text-align: center;">
    <div>MindSpeed MM社区交流群</div>
    <img src="./sources/images/MM_wechat_qrcode.jpg" width="150" alt="MindSpeed MM社区交流2群">
  </div>
  <div style="text-align: center;">
    <div>昇腾开源小助手</div>
    <img src="./sources/images/wechat_ascend_assistant.jpg" width="150" alt="昇腾小助手 微信">
  </div>
</div>

# 目录结构

关键目录如下，详细目录介绍参见[目录介绍](docs/zh/dir_structure.md)

```bash
├─bridge          # mbridge在线权重转换
├─checkpoint      # 离线权重转换工具
├─ci              # Continuous Integration
├─docs            # 项目文档目录
│  └─zh           # 中文文档目录
├─examples        # 预置模型，包括模型配置、数据集配置、训练脚本、推理脚本等文件
├─mindspeed_mm    # 核心代码目录
├─scripts         # 脚本目录
├─sources         # 图片视频目录
├─tests           # 测试代码目录
│  ├─st           # 系统测试用例
│  └─ut           # 单元测试用例
├─UserGuide       # 用户指南目录
└─verl_plugin     # verl插件模块
```

# 最新消息

---

- [Apr. 17, 2026]: 🚀 MindSpeed MM基于FSDP2支持[Qwen3.6](./examples/qwen3_6)模型【Prototype】
- [Mar. 24, 2026]: 🚀 MindSpeed MM基于FSDP2支持[LTX2](./examples/ltx2)模型【Prototype】
- [Mar. 09, 2026]: 🚀 MindSpeed MM基于FSDP2支持[FunASR](./examples/funasr)模型
- [Feb. 16, 2026]: 🚀 MindSpeed MM基于FSDP2支持[Qwen3.5](./examples/qwen3_5)模型【Prototype】
- [Feb. 14, 2026]: 🚀 MindSpeed MM基于FSDP2支持[CosyVoice3](./examples/cosyvoice3)模型训练
- [Feb. 13, 2026]: 🚀 MindSpeed MM基于FSDP2支持[Kimi-K2.5](./examples/kimik2_5)模型【Prototype】
- [Feb. 12, 2026]: 🚀 MindSpeed MM基于FSDP2支持[HunyuanVideo1.5](./examples/hunyuanvideo_1.5)模型训练demo【Prototype】
- [Feb. 03, 2026]: 🚀 MindSpeed MM基于FSDP2支持[DeepseekOCR2](./examples/deepseekocr2/README.md)模型训练demo【Prototype】
- [Jan. 29, 2026]: 🎉 昇腾镜像仓库上线[MindSpeed MM镜像](https://www.hiascend.com/developer/ascendhub/detail/6857f6fc2cfa4a678710a7075426ee5e)
- [Jan. 29, 2026]: 🚀 MindSpeed MM基于FSDP2支持[Qwen3-TTS](./examples/qwen3tts)模型【Prototype】
- [Jan. 28, 2026]: 🚀 MindSpeed MM基于FSDP2支持Magistral-Small-2509模型【Prototype】
- [Jan. 08, 2026]: 🚀 MindSpeed MM支持FLUX.2模型【Prototype】
- [Dec. 25, 2025]: 🎉 用户手册上线！体验链接：<https://mindspeed-mm.readthedocs.io/zh-cn/latest/>
- [Dec. 03, 2025]: 🚀 MindSpeed MM基于FSDP2支持Glm4.5v模型训练demo【Prototype】
- [Dec. 02, 2025]: 🚀 MindSpeed MM支持Self-Forcing基于Wan2.1-1.3B的DMD蒸馏 【Prototype】
- [Nov. 27, 2025]: 🚀 MindSpeed MM基于fully shard支持Qwen3VL-235B模型
- [Nov. 20, 2025]: 🚀 MindSpeed MM基于FSDP2支持Qwen3-Omni模型
- [Nov. 19, 2025]: 🚀 MindSpeed MM支持Qwen Image、Qwen Image Edit模型 【Prototype】
- [Nov. 13, 2025]: 🚀 MindSpeed MM基于FSDP2支持InternVL3.5-30B模型
- [Nov. 06, 2025]: 🚀 MindSpeed MM基于FSDP2支持DeepseekOCR模型训练demo【Prototype】
- [Oct. 31, 2025]: 🚀 MindSpeed MM基于fully shard支持Qwen3VL-8B/30B模型
- [Oct. 22, 2025]: 🚀 MindSpeed MM基于fully shard支持Wan2.2系列模型
- [Sep. 08, 2025]: 🚀 MindSpeed MM支持FLUX.1-Kontext模型
- [Sep. 8, 2025]: 🚀 MindSpeed MM支持FLUX **强化学习** DanceGRPO训练
- **[Sep. 03, 2025]: 🎉 强化学习上线！ MindSpeed MM支持Qwen2.5VL 7B/32B [GRPO训练](./examples/verl_examples/qwen2.5vl/README.md)**
- [Aug. 15, 2025]: 🤝 MindSpeed MM**原生支持**Lumina-mGPT 2.0模型
- [Jul. 29, 2025]: 🌴 MindSpeed MM支持core 0.12.1版本
- [Jul. 10, 2025]: 🚀 MindSpeed MM支持InternVL3-8B/78B模型
- [Jul. 02, 2025]: ⚡ MindSpeed MM **0Day**支持GLM-4.1V模型
- [Jun. 30, 2025]: 🌴 MindSpeed MM版本2.1.0发布
- [Jun. 25, 2025]: 🚀 MindSpeed MM支持HiDream-I1模型
- [Jun. 05, 2025]: 🚀 MindSpeed MM支持Qwen2.5Omni-7B模型
- [Jun. 05, 2025]: 🤝 MindSpeed MM**原生支持**OpenSoraPlan 1.5模型
- [Apr. 03, 2025]: 🚀 MindSpeed MM支持Qwen2.5VL-32B模型
- [Mar. 27, 2025]: 🚀 MindSpeed MM支持Wan2.1-1.3B/14B模型
- [Mar. 26, 2025]: 🚀 MindSpeed MM支持Qwen2.5VL-3B/7B/72B模型
- [Feb. 20, 2025]: 🚀 MindSpeed MM支持InternVL2.5-78B模型
- [Feb. 18, 2025]: 🚀 MindSpeed MM支持HunyuanVideo模型
- [Feb. 17, 2025]: 🔥 MindSpeed MM支持Mindspeed-Core & Megatron 0.8.0版本
- [Feb. 15, 2025]: 🚀 MindSpeed MM支持Sana模型
- [Jan. 24, 2025]: 🚀 MindSpeed MM支持CogVideoX 1.5模型
- [Dec. 30, 2024]: 🌴 MindSpeed MM版本1.0.0发布
- [Dec. 16, 2024]: 🤝 MindSpeed MM**原生支持**Qihoo-T2X模型
- [Dec. 03, 2024]: 🚀 MindSpeed MM支持SD3.5模型
- [Nov. 30, 2024]: 🎉 MindSpeed MM支持多模态理解测评
- [Nov. 22, 2024]: 🚀 MindSpeed MM支持CogVideoX模型
- [Nov. 06, 2024]: 🚀 MindSpeed MM支持FLUX模型
- [Oct. 30, 2024]: 🤝 MindSpeed MM**原生支持**OpenSoraPlan 1.3模型
- [Oct. 21, 2024]: 🚀 MindSpeed MM支持InternVL2、以及Qwen2VL模型
- [Oct. 16, 2024]: 🌱 MindSpeed MM首版本1.0.RC3发布

> 注意： **Prototype**特性未经过充分验证，可能存在不稳定和bug问题，**beta**表示非商用特性。

# 效果展示

---

## 文生视频： Wan 2.2 T2V

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="sources/videos/video_wan_T2V.gif" width="80%" controls autoplay loop>
          <p>Prompt: Ultra HD, 4K, cinematic composition, low contrast ratio, low saturation, cool tone; The queen wears an iron crown and rides on the dragon over the city. She holds a big flag that shows:" MindSpeed MM".</p>
      </td>
  </tr>
</table>

## 文生视频： OpensoraPlan 1.5 T2V

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="sources/videos/video_osp15mini_1.gif" width="100%" controls autoplay loop>
          <p>Prompt: A fluffy white rabbit with soft, velvety fur and twitching pink nose sits curiously near a rustic wooden fence, surrounded by a lush garden of vibrant wildflowers and tall grasses swaying gently in the breeze. The rabbit's large, expressive eyes scan the environment, reflecting the golden hues of the setting sun. As it nibbles on a patch of clover, its ears perk up at the distant sound of chirping birds. The fence, weathered and covered in patches of moss, adds a charming, pastoral backdrop to this serene scene, capturing the essence of a peaceful countryside moment.</p>
      </td>
      <td>
          <img src="sources/videos/video_osp15mini_2.gif" width="100%" controls autoplay loop>
          <p>Prompt: A majestic Berlin tower stands tall against the night sky, its structure bathed in a mesmerizing array of vibrant lights, casting a kaleidoscope of colors across the cityscape. The tower's intricate architectural details are highlighted by the illumination, creating a stunning contrast against the deep indigo sky. As the camera pans upward, the lights shift, revealing a dynamic play of shadows and hues that dance across the tower's surface. The surrounding city lights twinkle in harmony, enhancing the tower's grandeur and creating a breathtaking visual symphony that captures the essence of Berlin's vibrant nightlife.</p>
      </td>
  </tr>
</table>

## 文生图：Qwen-Image -> 图片编辑 Flux.1-Kontext

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="sources/images/qwen_image.png" width="800">
          <p>Prompt for generation: A coffee shop entrance features a chalkboard sign reading "MindSpeed Coffee 😊 $2 per cup," with a neon light displaying "MindSpeed MM". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "Welcome to use MindSpeed MM". Ultra HD, 4K, cinematic composition. (Qwen-Image)</p>
      </td>
      <td>
          <img src="sources/images/flux_kontext.png" width="1500">
          <p>Prompt for edition: Change the decoration of the coffee shop to a modern style with white painting. (Flux.1-Kontext)</p>
      </td>
  </tr>
</table>

## 理解模型：Qwen2VL

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <p>Input image for both models:</p>
          <img src="sources/images/view.jpg" width="1000" height="700">
          <p>Input text for both models: Please describe the image shortly</p>
          <p>Qwen2VL推理结果: The image depicts a serene lakeside scene with a wooden dock extending into the calm waters. The dock is made of weathered wooden planks and leads to a small platform with a ladder, suggesting it is used for swimming or diving. The lake is surrounded by lush green forests and mountains in the background, creating a picturesque and tranquil setting. The sky is overcast, adding to the calm and peaceful atmosphere of the scene.</p>
          <p>Input text for Qwen2VL: 请用中文简短描述这张照片</p>
          <p>Qwen2VL推理结果: 这张图片展示了一座木制码头延伸到平静的湖面上，背景是连绵的山脉和茂密的森林。天空多云，整体色调偏冷，给人一种宁静和自然的感觉。</p>
      </td>
  </tr>
</table>

# 版本说明

---

MindSpeed MM支持Atlas 800T A2等昇腾训练硬件形态，软件版本配套表如下：

| MindSpeed MM版本 | MindSpeed版本      | Megatron版本 | PyTorch版本  | TorchNPU版本 | CANN版本 | Python版本            |
| ---------------- | ------------------ | ------------ | ------------ | ------------- | -------- | --------------------- |
| master（在研版本） | master（在研版本）       | Core 0.12.1  | 2.7.1 | 在研版本       | 在研版本  | Python3.10            |
| 26.0.0（商用）   | 26.0.0_core_r0.12.1 | Core 0.12.1  | 2.7.1       | 26.0.0         | 9.0.0    | Python3.10            |
| 2.3.0（商用）    | 2.3.0_core_r0.12.1 | Core 0.12.1  | 2.6.0, 2.7.1 | 7.3.0         | 8.5.0    | Python3.10            |
| 2.2.0（商用）    | 2.2.0_core_r0.12.1 | Core 0.12.1  | 2.6.0, 2.7.1 | 7.2.0         | 8.3.RC1  | Python3.10            |
| 2.1.0（商用）    | 2.1.0_core_r0.8.0  | Core 0.8.0   | 2.1.0, 2.6.0 | 7.1.0         | 8.2.RC1  | Python3.8, Python3.10 |
| 2.0.0（商用）    | 2.0.0_core_r0.8.0  | Core 0.8.0   | 2.1.0        | 7.0.0         | 8.1.RC1  | Python3.8, Python3.10 |
| 1.0.0（商用）    | 1.0.0_core_r0.6.0  | Core 0.6.0   | 2.1.0        | 6.0.0         | 8.0.0    | Python3.8, Python3.10 |

>[!Note]
>
> “在研版本”指当前正处于开发迭代中的版本，由于该版本的功能仍处于持续迭代与优化阶段，其配套依赖项即使采用已发布的商用版本，仍可能存在兼容性风险或运行不稳定性，如需稳定使用，建议优先使用已正式发布的商用版本。

更多详情请参考[版本配套表](docs/zh/release_notes_mm.md#版本配套说明)。

# 安装

---

MindSpeed MM具体的安装请参考[安装指南](docs/zh/pytorch/install_guide.md)。
当前qwen3vl、wan2.2模型已支持一键安装，一键安装使用说明详见[一键安装使用说明](docs/zh/pytorch/install_guide.md)。

# 快速上手

---

MindSpeed MM将以Qwen2.5-VL-3B和Wan2.1-T2V-1.3B模型为例，引导开发者快速上手预置模型在昇腾NPU上的高效运行。具体的操作请参考[快速上手](./docs/zh/pytorch/quickstart.md)。

# 特性/模型介绍

---

## 已支持特性概览

|       模型 \ 特性        | [TP](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/tensor-parallel.md) | [TP-SP](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/sequence-parallel.md) | [VPP](docs/zh/features/virtual_pipeline_parallel.md) | [PP](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/pipeline-parallel.md) | CP | [Distributed Optimizer](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/distributed-optimizer.md) | [Recomputation](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/recomputation.md) | [LoRA](./docs/zh/features/lora_finetune.md) | RL | [FSDP2](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/fsdp2.md) |
|:--------------------:|:------:|:------:|:------:|:---------------------------------------------------------------------------------------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Magistral-Small-2509 |  |  |  |  |  |  | ✔ | ✔ |  | ✔ |
|   InternVL3.5-30B    |  |  |  |  |  |  | ✔ |  |  | ✔ |
|     Qwen3-VL-8B      |  |  |  |  |  |  | ✔ |  |  | ✔ |
|     Qwen3-VL-30B     |  |  |  |  |  |  | ✔ |  |  | ✔ |
|        Wan2.2        |  |  |  |  | CP (Ulysses) |  | ✔ |  |  | ✔ |
| OpenSoraPlan1.5-T2V  | ✔ | ✔ |  |  |  |  | ✔ |  |  |  |
|        Wan2.1        |  |  |  |  | CP (Ulysses) | ✔ | ✔ | ✔ |  | ✔ |
|     HunyuanVideo     | ✔ | ✔ |  |  | CP (Ulysses) | ✔ | ✔ | ✔ |  |  |
|   HunyuanVideo1.5    |  |  |  |  |  | ✔ | ✔ |  |  | ✔ |
|   CogVideoX系列-T2V    | ✔ | ✔ |  |  | CP (Ulysses) | ✔ | ✔ | ✔ |  |  |
|   CogVideoX系列-I2V    | ✔ | ✔ |  |  | CP (Ulysses) | ✔ | ✔ | ✔ |  |  |
| OpensoraPlan1.3-T2V  | ✔ | ✔ | ✔ | ✔ | CP (Ulysses) | ✔ | ✔ |  |  |  |
| OpensoraPlan1.3-I2V  | ✔ | ✔ | ✔ | ✔ | CP (Ulysses) | ✔ | ✔ |  |  |  |
|       GLM-4.1V       |  |  |  | ✔ |  | ✔ | ✔ |  |  |  |
|      Qwen2VL-2B      | ✔ | ✔ |  | ✔ | CP (Ulysses) | ✔ | ✔ | ✔ |  |  |
|      Qwen2VL-7B      | ✔ | ✔ |  | ✔ | CP (Ulysses) | ✔ | ✔ | ✔ |  |  |
|     Qwen2VL-72B      | ✔ | ✔ |  | ✔ | CP (Ulysses) | ✔ | ✔ | ✔ | DPO |  |
|     Qwen2.5VL-3B     | ✔ | ✔ |  | ✔ |  | ✔ | ✔ |  | GRPO |  |
|     Qwen2.5VL-7B     | ✔ | ✔ |  | ✔ |  | ✔ | ✔ |  | GRPO |  |
|    Qwen2.5VL-32B     | ✔ | ✔ |  | ✔ |  | ✔ | ✔ |  | GRPO |  |
|    Qwen2.5VL-72B     | ✔ | ✔ |  | ✔ |  | ✔ | ✔ | ✔ |  |  |
|    Qwen2.5Omni-7B    | ✔ |  |  | ✔ |  | ✔ |  | ✔ |  |  |
|      Qwen3-Omni      |  |  |  |  |  |  | ✔ |  |  | ✔ |
|     InternVL3-8B     | ✔ | ✔ | ✔ | ✔ | CP (Ring) | ✔ | ✔ |  |  |  |
|    InternVL3-78B     | ✔ | ✔ | ✔ | ✔ | CP (Ring) | ✔ | ✔ |  |  |  |

备注：

- TP: [Tensor Parallel](https://arxiv.org/abs/1909.08053)
- TP-SP: [Tensor Parallel with Sequence Parallel](https://arxiv.org/abs/2205.05198)
- VPP: [Virtual Pipeline Parallel](https://arxiv.org/abs/2104.04473)
- PP: [Pipeline Parallel](https://arxiv.org/abs/2104.04473)
- DSP: [Dynamic Sequence Parallel](https://arxiv.org/abs/2403.10266)
- CP (Ulysses): [Context Parallel](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) by leveraging [Deepspeed Ulysses](https://arxiv.org/abs/2309.14509) with Sequence Parallel
- CP (Ring Attention): Context Parallel with [Ring Attention](https://arxiv.org/abs/2310.01889)
- Distributed Optimizer: [Zero Redundancy Optimizer](https://arxiv.org/abs/1910.02054) (ZeRO)
- Recomputation: Reducing Activation [Recomputation](https://arxiv.org/abs/2205.05198)
- LoRA: [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- RL: Reinforcement Learning
- FSDP2: [Fully Sharded Data Parallelism](https://arxiv.org/abs/2304.11277)

---

## 配套版本与支持模型

MindSpeed MM 预置了覆盖多模态生成、多模态理解等任务的丰富模型；各模型的参数规模、训练任务、推荐集群、实测性能与认证状态，详见 **[MindSpeed MM 支持模型列表](docs/zh/pytorch/supported_models.md)**。

大语言模型（稠密模型、稀疏模型和状态空间模型）由 MindSpeed-LLM 专项维护，如需进行大语言模型训练，请访问 [MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/models/supported_models.md) 获取详细使用说明。

# 常用参数解释说明

针对MindSpeed MM套件中运行所使用的参数做解释说明，具体见[README](./docs/zh/pytorch/args_readme.md)

# 特性规划

---

- 【模型特性】 CogVideoX: PP
- 【模型特性】 OpensoraPlan1.3: CP (Ring Attention)
- 【模型特性】 Qwen2VL: VPP, CP (Ulysses & Ring Attention)
- 【模型特性】 InternVL2: TP, CP (Ulysses & Ring Attention)
- 【基础特性】 Hetero-parallel

<a id="jump2"></a>

# 工具使用

---

<a id="jump2.1"></a>

## 昇腾Profiling采集工具

MindSpeed MM集成了昇腾profiling采集工具，以提供对模型运行情况的分析。该工具能够依照配置采集模型的算子、显存等关键信息，同时支持动静态两种采集方式，协助开发者分析模型瓶颈，并可根据实际场景需求选择使用。

  具体方法见 [README](./docs/zh/tools.md) 的profiling章节

## MindStudio Insight性能分析工具

针对大模型集群场景的性能调优，这里推荐一款优秀的可视化调优工具MindStudio Insight。
MindStudio Insight提供了包括Timeline视图、通信分析、计算耗时等的可视化呈现，以便用户分析潜在的性能瓶颈，并指导如何采取措施消除或减少这些瓶颈。

  具体安装及使用方法见[《MindStudio Insight操作指南》](https://msinsight.readthedocs.io/zh-cn/latest/zh/user_guide/basic_operations.html)

## Sora类模型特征提取

MindSpeed MM支持提取视频和文本特征并保存

  具体方法见 [README](./docs/zh/tools.md) 的Sora类模型特征提取章节

## 内存快照提取

MindSpeed MM集成了昇腾内存快照采集工具，以提供对模型运行情况的分析。

  具体方法见 [README](./docs/zh/tools.md) 的内存快照提取章节

## Tensorboard使用

MindSpeed MM支持Tensorboard的使用

  具体方法见 [README](./docs/zh/tools.md) 的Tensorboard使用章节

# 版本维护

---

MindSpeed MM版本有以下五个维护阶段：

| **状态**            | **时间** | **说明**                                                               |
| ------------------- | -------- |----------------------------------------------------------------------|
| 计划                | 1—3 个月 | 计划特性                                                                 |
| 开发                | 3 个月   | 开发特性                                                                 |
| 维护                | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的MindSpeed MM版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布                                             |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                                                           |

MindSpeed MM已发布版本维护策略：

| **MindSpeed MM版本** | **维护策略** | **当前状态** | **发布时间**   | **后续状态**         | **EOL日期** |
|--------------------|-----------|-------|------------|------------------|-----------|
| 26.0.0             |  常规版本  | 维护   | 2026/03/30 | 预计2026/09/30起无维护 |           |
| 2.3.0              |  常规版本  | 无维护  | 2025/12/30 | 预计2026/06/30起无维护 |           |
| 2.2.0              |  常规版本  | 无维护  | 2025/09/30 | 预计2026/03/30起无维护 |           |
| 2.1.0              |  常规版本  | 无维护  | 2025/06/30 | 预计2025/12/30起无维护 |           |
| 2.0.0              |  常规版本  | 无维护  | 2025/03/30 | 预计2025/09/30起无维护 |           |
| 1.0.0              |  常规版本  | 无维护  | 2024/12/30 | 预计2025/06/30起无维护 |           |
| 1.0.RC3            |  常规版本  | 无维护  | 2024/09/30 | 预计2025/03/30起无维护 |           |

# 常见问题

---

相关FAQ请参考链接：[FAQ](./docs/zh/FAQ.md)

# 相关资源

---

1. [面向大规模分布式训练的多模态套件](https://mp.weixin.qq.com/s/Qiw_qThKA72T0lLOSpjkKw)
2. [凭借昇腾澎湃算力，Open-Sora Plan实现电影级视频生成](https://mp.weixin.qq.com/s/KY2tLthhre-SRbuWka3c2w)
3. [MindSpeed MM支持主流多模态理解大模型，性能实现大幅提升！](https://mp.weixin.qq.com/s/3pZRy24ITyKl3nGc33Sq7w)
4. [基于昇腾原生训练！中大和360联合打造多模态任务新范式Qihoo-T2X](https://mp.weixin.qq.com/s/zQAy_hbL9cR3c8-NO6lKnA)
5. [基于昇腾MindSpeed MM玩转Wan2.1视频生成SOTA模型](https://mp.weixin.qq.com/s/g2ShV2F6YpoVAniw6CBN_w)
6. [多模态理解SOTA模型开箱即用，MindSpeed MM支持Qwen2.5-VL最佳实践](https://mp.weixin.qq.com/s/ac7RUWw79stunwQIyC-ykQ)
7. [联创首发-基于昇腾MindSpeed MM玩转Open-Sora Plan V1.5模型](https://mp.weixin.qq.com/s/3cgO8yqrOIEHYqW69VQQcQ)
8. [开源即支持！基于昇腾MindSpeed MM玩转GLM-4.1V-Thinking多模态理解最新模型](https://mp.weixin.qq.com/s/FLgCfBVG7pOzNHji2uwcDg)

# 安全声明

---

[MindSpeed MM 安全声明](./SECURITYNOTE.md)

# 免责声明

---

## 致MindSpeed MM使用者

1. MindSpeed MM提供的模型仅供您用于非商业目的。
2. 对于各模型，MindSpeed MM平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用MindSpeed MM模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitcode提交issue，我们将及时审视并解决。
4. MindSpeed MM功能依赖的Megatron等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，MindSpeed MM仓库不保证第三方开源软件本身的问题进行修复，也不保证会测试，纠正所有第三方开源软件的漏洞和错误。

## 致数据集所有者

如果您不希望您的数据集在MindSpeed MM中的模型被提及，或希望更新MindSpeed MM中的模型关于您的数据集的描述，请在Gitcode提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对MindSpeed MM的理解和贡献。

# License声明

Ascend MindSpeed MM提供的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅Ascend MindSpeed MM根目录[LICENSE](./LICENSE)文件，`docs`目录下的文档适用CC-BY 4.0许可证，具体参见文档[LICENSE](./docs/LICENSE)。

# 贡献声明

---

## 1. 报告问题

- 如果您发现任何问题，请先查看仓库的[issues列表](https://gitcode.com/Ascend/MindSpeed-MM/issues)，尝试寻找类似问题或解决方案。

- 如果现有[issues列表](https://gitcode.com/Ascend/MindSpeed-MM/issues)中没有您遇到的问题，可以[提交一个新的issue](https://gitcode.com/Ascend/MindSpeed-MM/issues/create/choose)，并尽量提供清晰的问题描述、复现步骤与环境信息。

## 2. 贡献代码流程

若您希望提交代码改动，请遵循以下简要步骤：

- 在您的个人分支上开发并提交，然后向本项目仓库发起Pull Request（PR）；

- 在我们的[SIG例会PR评审申请登记](https://gitcode.com/Ascend/MindSpeed-MM/issues/256)中，参照既定格式申请PR评审，并按时参加对应的评审会议；

- 根据评审意见进行修改，并更新PR；

- PR通过评审后，在评论区输入`compile`以触发门禁流水线（CI）；

- 当PR的CI通过且获得足够的标签后，仓库Committer将进行最终审核，并合入在研分支。

感谢您的参与与贡献！我们期待与您共同推动项目发展。

# 致谢

---

MindSpeed MM 由华为公司的下列部门及昇腾生态合作伙伴联合贡献：

华为公司：

- 计算产品线
- 公共开发部
- 2012实验室
- 华为云

生态合作伙伴：

- 360 AI Research
- 北大OpenSoraPlan团队
- 微信技术架构部基础架构中心
- 京东零售九数研发技术部

感谢来自社区的每一个PR，欢迎贡献 MindSpeed MM。
