# MindSpeed MM Supported Model List

<!-- md-trans-meta sourceCommit=eef34b0ea66a376f4eba2ea15f710237338e5e15 translatedAt=2026-08-18T07:55:06.655Z pushedAt=2026-08-18T08:00:25.390Z -->

This table shows the models supported by MindSpeed MM.
>[!NOTE]
>
> For the supported models listed below, we provide corresponding usage instructions in each model's **README** file, which contains detailed procedures for model training, inference, fine-tuning, and more.
> If you cannot access the HuggingFace community to download resources, we recommend downloading from ModelScope, and you should pay attention to the correctness and security of the files to be downloaded.

**Table 1** MindSpeed MM model list

<table>
  <thead>
    <tr>
      <th>Model Task</th>
      <th>Model</th>
      <th>Parameter Count</th>
      <th>Task</th>
      <th>Cluster</th>
      <th>Precision Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="42">Multimodal generation</td>
      </tr>
      <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/lumina">Lumina-mGPT 2.0</a></td>
      <td><a href="https://huggingface.co/Alpha-VLLM/Lumina-mGPT-2.0">7B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
      <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/opensoraplan1.5">OpenSoraPlan1.5</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.5.0">8.5B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
      <tr>
      <td rowspan="2"><a href="../../../examples/wan2.2">Wan2.2-T2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers">5B</a></td>
      <td> Pre-training </td>
      <td> 1x4 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers">A14B</a></td>
      <td> Pre-training </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
    </tr>
      <tr>
      <td rowspan="1"><a href="../../../examples/wan2.2">Wan2.2-TI2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers">5B</a></td>
      <td> Pre-training </td>
      <td> 1x4 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="../../../examples/wan2.2">Wan2.2-I2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers">A14B</a></td>
      <td> Pre-training </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="../../../examples/wan2.1">Wan2.1-T2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">1.3B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">1.3B</a></td>
      <td> LoRA fine-tuning </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers">14B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers">14B</a></td>
      <td> LoRA fine-tuning </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="../../../examples/wan2.1">Wan2.1-I2V</a></td>
      <td>1.3B</td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers">14B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers">14B</a></td>
      <td> LoRA fine-tuning </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/self_forcing">Self-Forcing</a></td>
      <td><a href="https://huggingface.co/gdhe17/Self-Forcing">1.3B</a></td>
      <td> DMD distillation </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/hunyuanvideo">HunyuanVideo-T2V</a></td>
      <td><a href="https://huggingface.co/tencent/HunyuanVideo">13B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/hunyuanvideo">HunyuanVideo-I2V</a></td>
      <td><a href="https://huggingface.co/tencent/HunyuanVideo-I2V">13B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/hunyuanvideo_1.5">HunyuanVideo1.5-T2V</a></td>
      <td><a href="https://huggingface.co/tencent/HunyuanVideo1.5-T2V">8B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/opensora1.0">OpenSora 1.0</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/Open-Sora/tree/main">5.5B</a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/opensora1.2">OpenSora 1.2</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3">5.2B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/opensora2.0">OpenSora 2.0-T2V</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/Open-Sora-v2">11B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/opensoraplan1.2">OpenSoraPlan 1.2</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0">8.7B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/opensoraplan1.3">OpenSoraPlan 1.3-T2V</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0"> 8.6B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/opensoraplan1.3">OpenSoraPlan 1.3-I2V</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0"> 8.6B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/vae">WFVAE</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/vae"> 0.18B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/cogvideox">CogVideoX-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/cogvideox">CogVideoX-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
  <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/cogvideox">CogVideoX 1.5-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> LoRA fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/cogvideox">CogVideoX 1.5-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> LoRA fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/qihoo_t2x">Qihoo-T2X</a></td>
      <td><a href="https://huggingface.co/qihoo360/Qihoo-T2X">1.1B</a></td>
      <td> Inference </td>
      <td> 1x1 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="../../../examples/diffusers/sdxl">SDXL</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/5956b68a6927126daffc2c5a6d1a9a189defe288">3.5B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/5956b68a6927126daffc2c5a6d1a9a189defe288">3.5B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> FP16 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="../../../examples/diffusers/sd3">SD3.5</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/5f724735437d91ed05304da478f3b2022fe3f6fb"> 8.1B </a></td>
      <td> Full-parameter fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/94643fac8a27345f695500085d78cc8fa01f5fa9"> 8.1B </a></td>
      <td> LoRA fine-tuning </td>
      <td> 1x8 </td>
      <td> FP16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffusers/flux">Flux</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">12B</a></td>
      <td> Full-parameter fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffusers/flux2">Flux2-T2I</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">32B</a></td>
      <td> Full-parameter fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffusers/flux2">Flux2-I2I</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">32B</a></td>
      <td> Full-parameter fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffusers/flux-kontext">Flux-Kontext</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">12B</a></td>
      <td> Full-parameter fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffusers/qwen_image">Qwen-Image</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">27B</a></td>
      <td> LoRA fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/diffsynth/qwen_image_edit">Qwen-Image-Edit</a></td>
      <td><a href="https://github.com/modelscope/Diffsynth-Studio/tree/main/examples/qwen_image">27B</a></td>
      <td> LoRA fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="28"> Multimodal understanding </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/glm4.1v">GLM-4.1V</a></td>
      <td><a href="https://github.com/THUDM/GLM-4.1V-Thinking">9B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/deepseekocr">DeepSeek-OCR</a></td>
      <td><a href="https://github.com/deepseek-ai/DeepSeek-OCR">3B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/llava1.5">LLaVA 1.5</a></td>
      <td><a href="https://github.com/haotian-liu/LLaVA">7B</a></td>
      <td> Full-parameter fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/internvl2">InternVL 2.0</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-2B">2B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-8B">8B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-26B">26B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B">76B</a></td>
      <td> Full-parameter fine-tuning </td>
      <td> 8x16 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/internvl2.5">InternVL 2.5</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-78B">78B</a></td>
      <td> Fine-tuning </td>
      <td> 8x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/internvl3">InternVL 3.0</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL3-8B">8B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL3-78B">78B</a></td>
      <td> Fine-tuning </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/internvl3.5">InternVL 3.5</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-Instruct">30B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 (A3)  </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/qwen2vl">Qwen2-VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct">2B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct">7B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct">72B</a></td>
      <td> Fine-tuning </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="../../../examples/qwen2.5vl">Qwen2.5-VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct">3B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">7B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct">32B</a></td>
      <td> Fine-tuning </td>
      <td> 2x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">72B</a></td>
      <td> Fine-tuning </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="../../../examples/qwen3vl">Qwen3-VL</a></td>
      <td><a href="https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe"> 8B </a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe"> 30B </a></td>
      <td> Fine-tuning </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe"> 235B </a></td>
      <td> Fine-tuning </td>
      <td> 16x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="../../../examples/qwen3_5">Qwen3.5</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3.5-27B">27B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3.5-35B-A3B">35B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3.5-397B-A17B">397B</a></td>
      <td> Fine-tuning </td>
      <td> 16x8 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/qwen2.5omni">Qwen2.5-Omni</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Omni-7B">7B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/qwen3omni">Qwen3-Omni</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct">30B</a></td>
      <td> Fine-tuning </td>
      <td> 2x4 (A3) </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td><a href="../../../examples/magistral-2509">Magistral-Small-2509</a></td>
      <td><a href="https://huggingface.co/mistralai/Magistral-Small-2509">24B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td> Speech recognition </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/whisper">Whisper</a></td>
      <td><a href="https://github.com/openai/whisper">1.5B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td> speech generation </td>
      <td><a href="../../../examples/cosyvoice3">CosyVoice3</a></td>
      <td><a href="https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512">0.5B</a></td>
      <td> Pre-training </td>
      <td> 1x8</td>
      <td> BF16 </td>
    </tr>
    </tbody>
</table>
