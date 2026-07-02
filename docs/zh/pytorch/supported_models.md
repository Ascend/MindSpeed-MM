# MindSpeed MM 支持模型列表

本表展示了MindSpeed MM支持的模型。
>[!NOTE]
>
>下述列表中支持的模型，我们在各模型的**README**文件中提供了相应的使用说明，里面有详细的模型训练、推理、微调等流程。
> 如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

**表1** MindSpeed MM模型列表

<table>
  <thead>
    <tr>
      <th>模型任务</th>
      <th>模型</th>
      <th>参数量</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="42"> 多模态生成 </td>
      </tr>
      <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/lumina">Lumina-mGPT 2.0</a></td>
      <td><a href="https://huggingface.co/Alpha-VLLM/Lumina-mGPT-2.0">7B</a></td>
      <td> 微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
      <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/opensoraplan1.5">OpenSoraPlan1.5</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.5.0">8.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
      <tr>
      <td rowspan="2"><a href="../../../examples/wan2.2">Wan2.2-T2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers">5B</a></td>
      <td> 预训练 </td>
      <td> 1x4 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers">A14B</a></td>
      <td> 预训练 </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
    </tr>
      <tr>
      <td rowspan="1"><a href="../../../examples/wan2.2">Wan2.2-TI2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers">5B</a></td>
      <td> 预训练 </td>
      <td> 1x4 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="../../../examples/wan2.2">Wan2.2-I2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers">A14B</a></td>
      <td> 预训练 </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="../../../examples/wan2.1">Wan2.1-T2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">1.3B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">1.3B</a></td>
      <td> Lora微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers">14B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers">14B</a></td>
      <td> Lora微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="../../../examples/wan2.1">Wan2.1-I2V</a></td>
      <td>1.3B</td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers">14B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers">14B</a></td>
      <td> Lora微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/self_forcing">Self-Forcing</a></td>
      <td><a href="https://huggingface.co/gdhe17/Self-Forcing">1.3B</a></td>
      <td> DMD蒸馏 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/hunyuanvideo">HunyuanVideo-T2V</a></td>
      <td><a href="https://huggingface.co/tencent/HunyuanVideo">13B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/hunyuanvideo">HunyuanVideo-I2V</a></td>
      <td><a href="https://huggingface.co/tencent/HunyuanVideo-I2V">13B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/hunyuanvideo_1.5">HunyuanVideo1.5-T2V</a></td>
      <td><a href="https://huggingface.co/tencent/HunyuanVideo1.5-T2V">8B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/opensora1.0">OpenSora 1.0</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/Open-Sora/tree/main">5.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/opensora1.2">OpenSora 1.2</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3">5.2B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/opensora2.0">OpenSora 2.0-T2V</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/Open-Sora-v2">11B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/opensoraplan1.2">OpenSoraPlan 1.2</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0">8.7B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/opensoraplan1.3">OpenSoraPlan 1.3-T2V</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0"> 8.6B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/opensoraplan1.3">OpenSoraPlan 1.3-I2V</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0"> 8.6B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/vae">WFVAE</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/vae"> 0.18B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/cogvideox">CogVideoX-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/cogvideox">CogVideoX-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
  <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/cogvideox">CogVideoX 1.5-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/cogvideox">CogVideoX 1.5-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/qihoo_t2x">Qihoo-T2X</a></td>
      <td><a href="https://huggingface.co/qihoo360/Qihoo-T2X">1.1B</a></td>
      <td> 推理 </td>
      <td> 1x1 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="../../../examples/diffusers/sdxl">SDXL</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/5956b68a6927126daffc2c5a6d1a9a189defe288">3.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/5956b68a6927126daffc2c5a6d1a9a189defe288">3.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> FP16 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="../../../examples/diffusers/sd3">SD3.5</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/5f724735437d91ed05304da478f3b2022fe3f6fb"> 8.1B </a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/94643fac8a27345f695500085d78cc8fa01f5fa9"> 8.1B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> FP16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffusers/flux">Flux</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">12B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffusers/flux2">Flux2-T2I</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">32B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffusers/flux2">Flux2-I2I</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">32B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffusers/flux-kontext">Flux-Kontext</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">12B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffusers/qwen_image">Qwen-Image</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">27B</a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffsynth/qwen_image_edit">Qwen-Image-Edit</a></td>
      <td><a href="https://github.com/modelscope/Diffsynth-Studio/tree/main/examples/qwen_image">27B</a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="28"> 多模态理解 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/glm4.1v">GLM-4.1V</a></td>
      <td><a href="https://github.com/THUDM/GLM-4.1V-Thinking">9B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/deepseekocr">DeepSeek-OCR</a></td>
      <td><a href="https://github.com/deepseek-ai/DeepSeek-OCR">3B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/llava1.5">LLaVA 1.5</a></td>
      <td><a href="https://github.com/haotian-liu/LLaVA">7B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/internvl2">InternVL 2.0</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-2B">2B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-8B">8B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-26B">26B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B">76B</a></td>
      <td> 全参微调 </td>
      <td> 8x16 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/internvl2.5">InternVL 2.5</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-78B">78B</a></td>
      <td> 微调 </td>
      <td> 8x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/internvl3">InternVL 3.0</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL3-8B">8B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL3-78B">78B</a></td>
      <td> 微调 </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/internvl3.5">InternVL 3.5</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-Instruct">30B</a></td>
      <td> 微调 </td>
      <td> 1x8 (A3)  </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/qwen2vl">Qwen2-VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct">2B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct">7B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct">72B</a></td>
      <td> 微调 </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="../../../examples/qwen2.5vl">Qwen2.5-VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct">3B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">7B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct">32B</a></td>
      <td> 微调 </td>
      <td> 2x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">72B</a></td>
      <td> 微调 </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="../../../examples/qwen3vl">Qwen3-VL</a></td>
      <td><a href="https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe"> 8B </a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe"> 30B </a></td>
      <td> 微调 </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe"> 235B </a></td>
      <td> 微调 </td>
      <td> 16x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="../../../examples/qwen3_5">Qwen3.5</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3.5-27B">27B</a></td>
      <td> 微调 </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3.5-35B-A3B">35B</a></td>
      <td> 微调 </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3.5-397B-A17B">397B</a></td>
      <td> 微调 </td>
      <td> 16x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/qwen2.5omni">Qwen2.5-Omni</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Omni-7B">7B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/qwen3omni">Qwen3-Omni</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct">30B</a></td>
      <td> 微调 </td>
      <td> 2x4 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/magistral-2509">Magistral-Small-2509</a></td>
      <td><a href="https://huggingface.co/mistralai/Magistral-Small-2509">24B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td> 语音识别 </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/whisper">Whisper</a></td>
      <td><a href="https://github.com/openai/whisper">1.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td> 语音生成 </td>
      <td><a href="../../../examples/cosyvoice3">CosyVoice3</a></td>
      <td><a href="https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512">0.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    </tbody>
</table>
