  <p align="center"> <img src="sources/images/mm_logo.png" height="103px" width="700px"> </p>

<p align="center">
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
        <img alt="Badge" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
    <a href="https://gitee.com/ascend/MindSpeed">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>

MindSpeed MM是面向大规模分布式训练的昇腾多模态大模型套件，同时支持多模态生成及多模态理解，旨在为华为 [昇腾芯片](https://www.hiascend.com/) 提供端到端的多模态训练解决方案, 包含预置业界主流模型，数据工程，分布式训练及加速，预训练、微调、在线推理任务等特性。

# 通知: 本项目已经正式迁移至 [Gitcode](https://gitcode.com/Ascend) 平台\n
---

# 🔥🔥🔥Latest News

* [Sep. 08, 2025]: 🚀 MindSpeed MM支持FLUX.1-Kontext模型 【Prototype】
* [Aug. 15, 2025]: 🤝 MindSpeed MM**原生支持**Lumina-mGPT 2.0模型
* [Jul. 29, 2025]: 🌴 MindSpeed MM支持core 0.12.1版本
* [Jul. 10, 2025]: 🚀 MindSpeed MM支持InternVL3-8B/78B模型
* **[Jul. 03, 2025]: 🎉 强化学习上线！ MindSpeed MM支持Qwen2.5VL 3B/7B/32B [GRPO训练](https://gitee.com/ascend/MindSpeed-MM/tree/2.1.0/examples/rl)**
* [Jul. 02, 2025]: ⚡ MindSpeed MM **0Day**支持GLM-4.1V模型
* [Jun. 30, 2025]: 🌴 MindSpeed MM版本2.1.0发布
* [Jun. 25, 2025]: 🚀 MindSpeed MM支持HiDream-I1模型
* [Jun. 05, 2025]: 🚀 MindSpeed MM支持Qwen2.5Omni-7B模型
* [Jun. 05, 2025]: 🤝 MindSpeed MM**原生支持**OpenSoraPlan 1.5模型
* [Apr. 03, 2025]: 🚀 MindSpeed MM支持Qwen2.5VL-32B模型
* [Mar. 27, 2025]: 🚀 MindSpeed MM支持Wan2.1-1.3B/14B模型
* [Mar. 26, 2025]: 🚀 MindSpeed MM支持Qwen2.5VL-3B/7B/72B模型
* [Feb. 20, 2025]: 🚀 MindSpeed MM支持InternVL2.5-78B模型
* [Feb. 18, 2025]: 🚀 MindSpeed MM支持HunyuanVideo模型
* [Feb. 17, 2025]: 🔥 MindSpeed MM支持Mindspeed-Core & Megatron 0.8.0版本
* [Feb. 15, 2025]: 🚀 MindSpeed MM支持Sana模型
* [Jan. 24, 2025]: 🚀 MindSpeed MM支持CogVideoX 1.5模型
* [Dec. 30, 2024]: 🌴 MindSpeed MM版本1.0.0发布
* [Dec. 16, 2024]: 🤝 MindSpeed MM**原生支持**Qihoo-T2X模型
* [Dec. 03, 2024]: 🚀 MindSpeed MM支持SD3.5模型
* [Nov. 30, 2024]: 🎉 MindSpeed MM支持多模态理解测评
* [Nov. 22, 2024]: 🚀 MindSpeed MM支持CogVideoX模型
* [Nov. 06, 2024]: 🚀 MindSpeed MM支持FLUX模型
* [Oct. 30, 2024]: 🤝 MindSpeed MM**原生支持**OpenSoraPlan 1.3模型
* [Oct. 21, 2024]: 🚀 MindSpeed MM支持InternVL2、以及Qwen2VL模型
* [Oct. 16, 2024]: 🌱 MindSpeed MM首版本1.0.RC3发布

> 注： **Prototype**特性未经过充分验证，可能存在不稳定和bug问题，**beta**表示非商用特性

---

# 效果展示

## 文生视频： OpensoraPlan 1.5 T2V

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="sources/videos/video_osp15mini_1.gif" width="100%" controls autoplay loop></video>
          <p>Prompt: A fluffy white rabbit with soft, velvety fur and twitching pink nose sits curiously near a rustic wooden fence, surrounded by a lush garden of vibrant wildflowers and tall grasses swaying gently in the breeze. The rabbit's large, expressive eyes scan the environment, reflecting the golden hues of the setting sun. As it nibbles on a patch of clover, its ears perk up at the distant sound of chirping birds. The fence, weathered and covered in patches of moss, adds a charming, pastoral backdrop to this serene scene, capturing the essence of a peaceful countryside moment.</p>
      </td>
      <td>
          <img src="sources/videos/video_osp15mini_2.gif" width="100%" controls autoplay loop></video>
          <p>Prompt: A majestic Berlin tower stands tall against the night sky, its structure bathed in a mesmerizing array of vibrant lights, casting a kaleidoscope of colors across the cityscape. The tower's intricate architectural details are highlighted by the illumination, creating a stunning contrast against the deep indigo sky. As the camera pans upward, the lights shift, revealing a dynamic play of shadows and hues that dance across the tower's surface. The surrounding city lights twinkle in harmony, enhancing the tower's grandeur and creating a breathtaking visual symphony that captures the essence of Berlin's vibrant nightlife.</p>
      </td>
  </tr>
</table>

## 图生视频： Wan 2.1 I2V

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <p>输入图片：</p>
          <img src="examples/wan2.1/i2v_input.jpg" width="800" height="100%"></img>
      </td>
      <td>
          <img src="sources/videos/video_wan_I2V.gif" width="100%" controls autoplay loop></video>
          <p>Prompt: An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.</p>
      </td>
  </tr>
</table>

## 文生图：Flux T2I

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="sources/images/flux_cat.png" width="800"></img>
          <p>Prompt: A cat holding a sign that says hello world</p>
      </td>
      <td>
          <img src="sources/images/flux_cat_MM.png" width="800"></img>
          <p>Prompt: A cat holding a sign that says MindSpeed</p>
      </td>
  </tr>
</table>

## 理解模型：InternVL2 & Qwen2VL

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <p>Input image for both models:</p>
          <img src="sources/images/view.jpg" width="1000" height="700"></img>
          <p>Input text for both models: Please describe the image shortly</p>
          <p>InternVL2推理结果: The image depicts a serene lakeside scene with a wooden dock extending into the calm water. The water reflects the surrounding landscape, which includes dense forests and a mountain range in the background. The sky is partly cloudy, adding to the tranquil atmosphere of the scene</p>
          <p>Qwen2VL推理结果: The image depicts a serene lakeside scene with a wooden dock extending into the calm waters. The dock is made of weathered wooden planks and leads to a small platform with a ladder, suggesting it is used for swimming or diving. The lake is surrounded by lush green forests and mountains in the background, creating a picturesque and tranquil setting. The sky is overcast, adding to the calm and peaceful atmosphere of the scene.</p>
          <p>Input text for InternVL2: 请简短描述这张照片</p>
          <p>InternVL2推理结果: 这张图片展示了一个宁静的湖泊，湖面平静，反射着天空和周围景物的影像。湖的中央有一个木制码头，延伸到湖中，码头上有几根柱子支撑。 湖的远端是一片茂密的森林，树木高大，覆盖着茂密的绿色植被。森林的尽头是一座高耸的山峰，山峰上覆盖着积雪，显得格外壮丽。 天空中有一些云朵，但整体上是晴朗的，阳光从云层中透出，照亮了整个湖面和周围的景色。 这张图片整体给人一种宁静、祥和的感觉，仿佛是一个远离尘嚣的世外桃源</p>
          <p>Input text for Qwen2VL: 请用中文简短描述这张照片</p>
          <p>Qwen2VL推理结果: 这张图片展示了一座木制码头延伸到平静的湖面上，背景是连绵的山脉和茂密的森林。天空多云，整体色调偏冷，给人一种宁静和自然的感觉。</p>
      </td>
  </tr>
</table>

---

# 环境部署

具体部署步骤请查看[部署文档](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

---

# 快速上手

快速上手操作请查看[快速上手文档](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/user-guide/getting_start.md)

---

# 特性/模型介绍

## 已支持特性概览

|       模型 \ 特性       | [TP](https://gitee.com/ascend/MindSpeed/blob/master/docs/features/tensor-parallel.md) | [TP-SP](https://gitee.com/ascend/MindSpeed/blob/master/docs/features/sequence-parallel.md) | [VPP](docs/features/virtual_pipeline_parallel.md) | [PP](https://gitee.com/ascend/MindSpeed/blob/master/docs/features/pipeline-parallel.md) | CP | [Distributed Optimizer](https://gitee.com/ascend/MindSpeed/blob/master/docs/features/distributed-optimizer.md) | [Recomputation](https://gitee.com/ascend/MindSpeed/blob/master/docs/features/recomputation.md) | [LoRA](./docs/features/lora_finetune.md) | RL |
|:-------------------:|:------:|:------:|:------:|:---------------------------------------------------------------------------------------:|:------:|:------:|:------:|:------:|:------:|
| OpenSoraPlan1.5-T2V | ✔ | ✔ |  |  |  |  | ✔ |  |  |
|       Wan2.1        |  |  |  |  | CP (Ulysses) | ✔ | ✔ | ✔ |  |
|    HunyuanVideo     | ✔ | ✔ |  |  | CP (Ulysses) | ✔ | ✔ | ✔ |  |
|   CogVideoX系列-T2V   | ✔ | ✔ |  |  | CP (Ulysses) | ✔ | ✔ | ✔ |  |
|   CogVideoX系列-I2V   | ✔ | ✔ |  |  | CP (Ulysses) | ✔ | ✔ | ✔ |  |
|     Opensora1.2     |  |  |  |  | DSP | ✔ | ✔ |  |  |
| OpensoraPlan1.3-T2V | ✔ | ✔ | ✔ | ✔ | CP (Ulysses) | ✔ | ✔ |  |  |
| OpensoraPlan1.3-I2V | ✔ | ✔ | ✔ | ✔ | CP (Ulysses) | ✔ | ✔ |  |  |
|      GLM-4.1V       |  |  |  | ✔ |  | ✔ | ✔ |  |  |
|    InternVL2-2B     |  |  | ✔ | ✔ |  | ✔ | ✔ |  |  |
|    InternVL2-8B     |  |  | ✔ | ✔ |  | ✔ | ✔ |  |  |
|    InternVL2-26B    |  |  | ✔ | ✔ |  | ✔ | ✔ |  |  |
|    InternVL2-76B    |  |  | ✔ | ✔ |  | ✔ | ✔ |  |  |
|     Qwen2VL-2B      | ✔ | ✔ |  | ✔ | CP (Ulysses) | ✔ | ✔ | ✔ |  |
|     Qwen2VL-7B      | ✔ | ✔ |  | ✔ | CP (Ulysses) | ✔ | ✔ | ✔ |  |
|     Qwen2VL-72B     | ✔ | ✔ |  | ✔ | CP (Ulysses) | ✔ | ✔ | ✔ | DPO |
|    Qwen2.5VL-3B     | ✔ | ✔ |  | ✔ |  | ✔ | ✔ |  | GRPO |
|    Qwen2.5VL-7B     | ✔ | ✔ |  | ✔ |  | ✔ | ✔ |  | GRPO |
|    Qwen2.5VL-32B    | ✔ | ✔ |  | ✔ |  | ✔ | ✔ |  | GRPO |
|    Qwen2.5VL-72B    | ✔ | ✔ |  | ✔ |  | ✔ | ✔ | ✔ |  |
|    Qwen2.5Omni-7B   | ✔ |  |  | ✔ |  | ✔ |  | ✔ |  |
|    InternVL3-8B    | ✔ | ✔ | ✔ | ✔ | CP (Ring) | ✔ | ✔ |  |
|    InternVL3-78B    | ✔ | ✔ | ✔ | ✔ | CP (Ring) | ✔ | ✔ |  |

备注：

* TP: [Tensor Parallel](https://arxiv.org/abs/1909.08053)
* TP-SP: [Tensor Parallel with Sequence Parallel](https://arxiv.org/abs/2205.05198)
* VPP: [Virtual Pipeline Parallel](https://arxiv.org/abs/2104.04473)
* PP: [Pipeline Parallel](https://arxiv.org/abs/2104.04473)
* DSP: [Dynamic Sequence Parallel](https://arxiv.org/abs/2403.10266)
* CP (Ulysses): [Context Parallel](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html) by leveraging [Deepspeed Ulysses](https://arxiv.org/abs/2309.14509) with Sequence Parallel
* CP (Ring Attention): Context Parallel with [Ring Attention](https://arxiv.org/abs/2310.01889)
* Distributed Optimizer: [Zero Redundancy Optimizer](https://arxiv.org/abs/1910.02054) (ZeRO)
* Recomputation: Reducing Activation [Recomputation](https://arxiv.org/abs/2205.05198)
* LoRA: [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
* RL: Reinforcement Learning

---

## 配套版本与支持模型

【现版本实测性能（硬件信息：Atlas 900 A2 PODc）】

下述列表中支持的模型，我们在各模型的**README**文件中提供了相应的使用说明，里面有详细的模型训练、推理、微调等流程

**模型**列中的超链接指向各模型的文件夹地址， **参数量**列中的超链接指向模型的社区资源地址

**认证**【Pass】表示已经通过测试的模型，【Test】表示测试中的模型

Samples per Second 为 (SPS); Frames per Second 为 (FPS); Tokens per Second 为 (TPS)

(注：此处SPS、FPS展示集群吞吐；TPS展示单卡吞吐)

**平均序列长度**是指在性能测试过程中所使用数据集的平均序列长度，通过统计各个序列长度的出现频率进行加权平均计算得出

**亲和场景**为调整少量结构或参数，使得模型更加亲和昇腾，性能更优

**A3** 为硬件 Atlas A3 训练系列产品

<table>
  <a id="jump1"></a>
  <caption>MindSpeed MM模型列表</caption>
  <thead>
    <tr>
      <th>模型任务</th>
      <th>模型</th>
      <th>参数量</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
      <th>NPU性能</th>
      <th>参考性能</th>
      <th>平均序列长度</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="36"> 多模态生成 </td>
      </tr>
      <tr>
      <td><a href="https://github.com/Alpha-VLLM/Lumina-mGPT-2.0">Lumina-mGPT 2.0</a></td>
      <td><a href="https://huggingface.co/Alpha-VLLM/Lumina-mGPT-2.0">7B</a></td>
      <td> 微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 8.24 (SPS)</td>
      <td> 8.79 (SPS)</td>
      <td> 1024 </td>
      <td>【Pass】</td>
    </tr>
      <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/blob/master/examples/opensoraplan1.5/">OpenSoraPlan1.5</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.5.0">8.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.83 (SPS) </td>
      <td> / </td>
      <td> / </td>
      <td>【北大贡献】</td>
    </tr>
      <td rowspan="4"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/wan2.1">Wan2.1-T2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">1.3B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.918 (SPS) </td>
      <td> 1.04 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">1.3B</a></td>
      <td> Lora微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.954 (SPS) </td>
      <td> 1.042 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers">14B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.160 (SPS) </td>
      <td> 0.160 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers">14B</a></td>
      <td> Lora微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.179 (SPS) </td>
      <td> 0.174 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    </tr>
      <td rowspan="3"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/wan2.1">Wan2.1-I2V</a></td>
      <td>1.3B</td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.76 (SPS) </td>
      <td>  / </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers">14B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.130 (SPS) </td>
      <td> / </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers">14B</a></td>
      <td> Lora微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.179 (SPS) </td>
      <td> 0.173 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    </tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/hunyuanvideo">HunyuanVideo-T2V</a></td>
      <td><a href="https://huggingface.co/tencent/HunyuanVideo">13B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.171 (SPS) </td>
      <td> 0.181 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
 </tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/hunyuanvideo">HunyuanVideo-I2V</a></td>
      <td><a href="https://huggingface.co/tencent/HunyuanVideo-I2V">13B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.164 (SPS) </td>
      <td> 0.202 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensora1.0">OpenSora 1.0</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/Open-Sora/tree/main">5.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 3.18 (SPS)</td>
      <td> 2.04 (SPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensora1.2">OpenSora 1.2</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3">5.2B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 7.31 (SPS) </td>
      <td> 8.15 (SPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensora2.0">OpenSora 2.0-T2V</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/Open-Sora-v2">11B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1.33 (SPS) </td>
      <td> 1.46 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensoraplan1.2">OpenSoraPlan 1.2</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0">8.7B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.42 (SPS) </td>
      <td> 0.37 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensoraplan1.3">OpenSoraPlan 1.3-T2V</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0"> 8.6B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.29 (SPS) </td>
      <td> 1.27 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensoraplan1.3">OpenSoraPlan 1.3-I2V</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0"> 8.6B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.17 (SPS) </td>
      <td> 1.15 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/vae">WFVAE</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/vae"> 0.18B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 23.860 (SPS) </td>
      <td> 26.091 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/cogvideox">CogVideoX-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 0.37 (SPS) </td>
      <td> 0.46 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/cogvideox">CogVideoX-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 0.37 (SPS) </td>
      <td> 0.46 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
  <tr>
      <td rowspan="2"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/cogvideox">CogVideoX 1.5-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.88 (SPS) </td>
      <td> 2.09 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2.76 (SPS) </td>
      <td> 2.64 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/cogvideox">CogVideoX 1.5-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.81 (SPS) </td>
      <td> 2.01 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2.33 (SPS) </td>
      <td> 2.04 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/qihoo_t2x">Qihoo-T2X</a></td>
      <td><a href="https://huggingface.co/qihoo360/Qihoo-T2X">1.1B</a></td>
      <td> 推理 </td>
      <td> 1x1 </td>
      <td> BF16 </td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td>【奇虎360贡献】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/sdxl">SDXL</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/5956b68a6927126daffc2c5a6d1a9a189defe288">3.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 29.92  (FPS)</td>
      <td> 30.65 (FPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/5956b68a6927126daffc2c5a6d1a9a189defe288">3.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 28.51 (FPS)</td>
      <td> 30.23 (FPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/sd3">SD3</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/5f724735437d91ed05304da478f3b2022fe3f6fb">2B</a></td>
      <td> 全参微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 16.09 (FPS)</td>
      <td> 16.01 (FPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/sd3">SD3.5</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/5f724735437d91ed05304da478f3b2022fe3f6fb"> 8.1B </a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 26.20 (FPS)</td>
      <td> 28.33 (FPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/94643fac8a27345f695500085d78cc8fa01f5fa9"> 8.1B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> FP16 </td>
      <td> 47.93 (FPS)</td>
      <td> 47.95 (FPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/flux">Flux</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">12B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 55.23 (FPS) </td>
      <td> 53.65 (FPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/flux-kontext">Flux-Kontext</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">12B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.97 (FPS) </td>
      <td> 2.00 (FPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/sana">Sana</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">1.6B</a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 28.7 (FPS) </td>
      <td> 32.8 (FPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/hidream">HiDream</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">17B</a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 18.37 (FPS) </td>
      <td> 19.61 (FPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/diffusers/kolors">Kolors</a></td>
      <td><a href="https://github.com/Kwai-Kolors/Kolors">2.6B</a></td>
      <td> 推理 </td>
      <td> 1x1 </td>
      <td> FP16 </td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="17"> 多模态理解 </td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/glm4.1v">GLM-4.1V</a></td>
      <td><a href="https://github.com/THUDM/GLM-4.1V-Thinking">9B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 12.16(SPS) </td>
      <td> 10.28(SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/llava1.5">LLaVA 1.5</a></td>
      <td><a href="https://github.com/haotian-liu/LLaVA">7B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 48.27 (SPS) </td>
      <td> 49.94 (SPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
   <tr>
      <td rowspan="4"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/internvl2">InternVL 2.0</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-2B">2B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 33.77 (SPS) </td>
      <td> 22.46 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-8B">8B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 12.86 (SPS) </td>
      <td> 11.00 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-26B">26B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 3.31 (SPS) </td>
      <td> 3.26 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B">76B</a></td>
      <td> 全参微调 </td>
      <td> 8x16 </td>
      <td> BF16 </td>
      <td> 214 (TPS) </td>
      <td> 191 (TPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan=""><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/internvl2.5">InternVL 2.5</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-78B">78B</a></td>
      <td> 微调 </td>
      <td> 8x8 </td>
      <td> BF16 </td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/internvl3">InternVL 3.0</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL3-8B">8B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 7.07 (SPS) </td>
      <td> 6.67 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL3-78B">78B</a></td>
      <td> 微调 </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
      <td> 3.79 (SPS) </td>
      <td> 4.69 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/qwen2vl">Qwen2-VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct">2B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 34.15 (SPS) </td>
      <td> 34.88 (SPS) </td>
      <td> 563 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct">7B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 13.28 (SPS) </td>
      <td> 11.66 (SPS) </td>
      <td> 563 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct">72B</a></td>
      <td> 微调 </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
      <td> 261.25 (TPS) </td>
      <td> 257.63 (TPS) </td>
      <td> 563 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/qwen2.5vl">Qwen2.5-VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct">3B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 23.77 (SPS) </td>
      <td> 21.79 (SPS) </td>
      <td> 563 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">7B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 18.82 (SPS) </td>
      <td> 12.67 (SPS) </td>
      <td> 563 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct">32B</a></td>
      <td> 微调 </td>
      <td> 2x8 </td>
      <td> BF16 </td>
      <td> 257.50 (TPS) </td>
      <td> / </td>
      <td> 563 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">72B</a></td>
      <td> 微调 </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
      <td> 322.96 (TPS) </td>
      <td> 256.28 (TPS) </td>
      <td> 563 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitee.com/zhaomingyu13/MindSpeed-MM/tree/master/examples/qwen2.5omni">Qwen2.5-Omni</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Omni-7B">7B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 12.99 (SPS) </td>
      <td> 14.44 (SPS) </td>
      <td> 296 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 语音识别 </td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/whisper">Whisper</a></td>
      <td><a href="https://github.com/openai/whisper">1.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 93.38 (SPS) </td>
      <td> 109.23 (SPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    </tbody>
</table>

---

<table>
  <caption><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mm">其他已适配昇腾的多模态大模型</a></caption>
  <thead>
    <tr>
      <th>模型</th>
      <th>参数量</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
      <th>NPU性能</th>
      <th>参考性能</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mm/CogVLM2">CogVLM-2</a></td>
      <td><a href="https://github.com/THUDM/CogVLM2">8B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 3.9 (s/it) </td>
      <td> 3.3 (s/it) </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mm/PLLaVA">PLLaVA</a></td>
      <td><a href="https://github.com/magic-research/PLLaVA">7B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.841 (s/step) </td>
      <td> 0.935 (s/step) </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/magic-research/PLLaVA">7B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> FP32 </td>
      <td> 0.935 (s/step) </td>
      <td> 1.08 (s/step) </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mm/MiniCPM-V">miniCPM-V 2.5</a></td>
      <td><a href="https://github.com/OpenBMB/MiniCPM-V">8B</a></td>
      <td> 全参微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1046 (s)/50-200steps </td>
      <td> 847 (s)/50-200steps </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/OpenBMB/MiniCPM-V">8B</a></td>
      <td> Lora微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 603 (s)/50-200steps </td>
      <td> 490 (s)/50-200steps </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mm/HunyuanDiT">HunYuanDiT</a></td>
      <td><a href="https://github.com/Tencent/HunyuanDiT">1.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1099.5 (ms/step) </td>
      <td> 1059.3 (ms/step) </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mm/InternVL1.5">InternVL 1.5</a></td>
      <td><a href="https://github.com/OpenGVLab/InternVL/tree/v1.5.0">26B</a></td>
      <td> 微调训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 4.952 (FPS) </td>
      <td> 5.151 (FPS) </td>
      <td>【Pass】</td>
    </tr>
  </tbody>
</table>

---

# 特性规划

* 【新模型】  JanusPro
* 【模型特性】 CogVideoX: PP
* 【模型特性】 OpensoraPlan1.3: CP (Ring Attention)
* 【模型特性】 Qwen2VL: VPP, CP (Ulysses & Ring Attention)
* 【模型特性】 InternVL2: TP, CP (Ulysses & Ring Attention)
* 【基础特性】 Distrain

---

<a id="jump2"></a>

# 工具使用

<a id="jump2.1"></a>

## 昇腾Profiling采集工具

MindSpeed MM集成了昇腾profiling采集工具，以提供对模型运行情况的分析。该工具能够依照配置采集模型的算子、显存等关键信息，同时支持动静态两种采集方式，协助开发者分析模型瓶颈，并可根据实际场景需求选择使用。

  具体方法见 [README](./mindspeed_mm/tools/README.md) 的profiling章节

## MindStudio Insight性能分析工具

针对大模型集群场景的性能调优，这里推荐一款优秀的可视化调优工具MindStudio Insight。
MindStudio Insight提供了包括Timeline视图、通信分析、计算耗时等的可视化呈现，以便用户分析潜在的性能瓶颈，并指导如何采取措施消除或减少这些瓶颈。

  具体使用方法见[《MindStudio Insight操作指南》](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html)，下载地址[《MindStudio Insight》](https://support.huawei.com/enterprise/zh/ascend-computing/mindstudio-pid-251913966/software/264465341?idAbsPath=fixnode01|23710424|251366513|254884019|261408772|251913966)

## Sora类模型特征提取

MindSpeed MM支持提取视频和文本特征并保存

  具体方法见 [README](./mindspeed_mm/tools/README.md) 的Sora类模型特征提取章节

## 内存快照提取

MindSpeed MM集成了昇腾内存快照采集工具，以提供对模型运行情况的分析。

  具体方法见 [README](./mindspeed_mm/tools/README.md) 的内存快照提取章节

## Tensorboard使用

MindSpeed MM支持Tensorboard的使用

  具体方法见 [README](./mindspeed_mm/tools/README.md) 的Tensorboard使用章节

---

# 版本维护

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
|-----------------|-----------|--------|------------|-----------------------|-----------|
| 2.1.0             |  常规版本  | 维护   | 2025/06/30 | 预计2025/12/30起无维护  |
| 2.0.0             |  常规版本  | 维护   | 2025/03/30 | 预计2025/09/30起无维护  |
| 1.0.0             |  常规版本  | 维护   | 2024/12/30 | 预计2025/06/30起无维护  |           |
| 1.0.RC3             |  常规版本  | 维护   | 2024/09/30 | 预计2025/03/30起无维护  |           |

---

# 常见问题

相关FAQ请参考链接：[FAQ](./docs/FAQ.md)

---

# 相关资源

1. [面向大规模分布式训练的多模态套件](https://mp.weixin.qq.com/s/Qiw_qThKA72T0lLOSpjkKw)
2. [凭借昇腾澎湃算力，Open-Sora Plan实现电影级视频生成](https://mp.weixin.qq.com/s/KY2tLthhre-SRbuWka3c2w)
3. [MindSpeed MM支持主流多模态理解大模型，性能实现大幅提升！](https://mp.weixin.qq.com/s/3pZRy24ITyKl3nGc33Sq7w)
4. [基于昇腾原生训练！中大和360联合打造多模态任务新范式Qihoo-T2X](https://mp.weixin.qq.com/s/zQAy_hbL9cR3c8-NO6lKnA)
5. [基于昇腾MindSpeed MM玩转Wan2.1视频生成SOTA模型](https://mp.weixin.qq.com/s/g2ShV2F6YpoVAniw6CBN_w)
6. [多模态理解SOTA模型开箱即用，MindSpeed MM支持Qwen2.5-VL最佳实践](https://mp.weixin.qq.com/s/ac7RUWw79stunwQIyC-ykQ)
7. [联创首发-基于昇腾MindSpeed MM玩转Open-Sora Plan V1.5模型](https://mp.weixin.qq.com/s/3cgO8yqrOIEHYqW69VQQcQ)
8. [开源即支持！基于昇腾MindSpeed MM玩转GLM-4.1V-Thinking多模态理解最新模型](https://mp.weixin.qq.com/s/FLgCfBVG7pOzNHji2uwcDg)

---

# 安全申明

[MindSpeed MM 安全申明](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/SECURITYNOTE.md)

---

# 免责声明

## 致MindSpeed MM使用者

1. MindSpeed MM提供的模型仅供您用于非商业目的。
2. 对于各模型，MindSpeed MM平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用MindSpeed MM模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。
4. MindSpeed MM功能依赖的Megatron等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，MindSpeed MM仓库不保证第三方开源软件本身的问题进行修复，也不保证会测试，纠正所有第三方开源软件的漏洞和错误。

## 致数据集所有者

如果您不希望您的数据集在MindSpeed MM中的模型被提及，或希望更新MindSpeed MM中的模型关于您的数据集的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对MindSpeed MM的理解和贡献。

## License声明

Ascend MindSpeed MM提供的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅Ascend MindSpeed MM根目录。

---

# 致谢

MindSpeed MM 由华为公司的下列部门及昇腾生态合作伙伴联合贡献：

华为公司：

* 计算产品线
* 公共开发部
* 2012实验室
* 华为云

生态合作伙伴：

* 360 AI Research
* 北大OpenSoraPlan团队
* 微信技术架构部基础架构中心

感谢来自社区的每一个PR，欢迎贡献 MindSpeed MM

---
