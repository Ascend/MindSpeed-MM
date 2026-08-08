import math
import random
import argparse

from PIL import Image
from transformers import AutoConfig


def random_pil_image(w, h):
    """
    生成指定宽高的纯色 PIL Image 图像

    Args:
        w: 图像宽度（像素）
        h: 图像高度（像素）

    Returns:
        PIL.Image.Image: 随机纯色的 RGB 图像
    """
    color = tuple(random.randint(0, 255) for _ in range(3))
    return Image.new('RGB', (w, h), color)


def _preprocess_image(image, image_max_pixels, image_min_pixels):
    """
    按指定像素范围缩放 + 格式转换 + 尺寸约束 + 宽高比限制

    Args:
        image (PIL.Image): 输入的 PIL 图像
        image_max_pixels (int): 图像允许的最大像素总数
        image_min_pixels (int): 图像允许的最小像素总数

    Returns:
        PIL.Image: 预处理后的标准 RGB 图像
    """
    if (image.width * image.height) > image_max_pixels:
        resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < image_min_pixels:
        resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = max(image.width, 28), max(image.height, 28)
    image = image.resize((width, height), resample=Image.NEAREST)

    if image.width / image.height > 200:
        width, height = image.height * 180, image.height
        image = image.resize((width, height), resample=Image.NEAREST)

    if image.height / image.width > 200:
        width, height = image.width, image.width * 180
        image = image.resize((width, height), resample=Image.NEAREST)

    return image


def regularize_images(images, **kwargs):
    """
    图像标准化预处理

    Args:
        images (list): 原始图像列表, 支持PIL对象/图片路径两种输入
        **kwargs: 透传给 _preprocess_image 的关键字参数, 如 image_max_pixels、image_min_pixels

    Returns:
        list[Image.Image]: 经过统一规整、尺寸限制、格式转换后的PIL图像列表
    """
    results = []

    for image in images:
        if not isinstance(image, Image.Image):
            with Image.open(image) as img:
                processed_img = _preprocess_image(img, **kwargs)
            results.append(processed_img)
        else:
            results.append(_preprocess_image(image, **kwargs))

    return results


def get_mm_inputs(images, hf_ckpt_path, image_max_pixels=512 * 512, image_min_pixels=1024):
    """
    加载图像处理器 + 图像正则化 + 图像张量转换

    Args:
        images: 输入图像
        hf_ckpt_path: HuggingFace模型权重路径
        image_max_pixels: 图像最大像素值上限，需与模型配置中的 image_max_pixels 参数对齐
        image_min_pixels: 图像最小像素值下限，需与模型配置中的 image_min_pixels 参数对齐

    Returns:
        经过 image_processor 处理后的 PyTorch 张量格式图像输入
    """
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(hf_ckpt_path, trust_remote_code=True)
    image_processor = processor.image_processor

    if not isinstance(images, list):
        images = [images]
    images = regularize_images(images, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels)
    image_inputs = [{"type": "image", "image": img} for img in images]
    return image_processor(image_inputs, return_tensors="pt")


def estimate_vit_flops(hf_cfg, seq_length=None, freeze=False):
    """
    计算 ViT 部分的flops

    Args:
        hf_cfg: huggingface配置
        seq_length: 序列长度
        freeze: 是否冻结 ViT 部分

    Returns:
        ViT 部分的flops
    """
    vit_cfg = hf_cfg.vision_config

    in_dim = 3 # 此处 kimi 无传参, 设为默认值 3
    patch_size = vit_cfg.patch_size
    hidden_size = vit_cfg.mm_hidden_size

    flops_scale = 1 if freeze else 3

    # patch_embedding，kimi的卷积核大小为(in_dim * hidden_size * (patch_size ** 2))
    patch_embedding_flops = 2 * seq_length * in_dim * hidden_size * (patch_size ** 2) * flops_scale

    num_layers = vit_cfg.vt_num_hidden_layers

    # attention flops
    qkv_proj_flops = 2 * seq_length * (hidden_size ** 2) * 3 * flops_scale
    output_proj_flops = 2 * seq_length * (hidden_size ** 2) * flops_scale
    # Q @ KT：前向包含一次矩阵乘法, 反向包含两次矩阵乘法, 且在反向重计算中需额外进行一次前向计算
    # score @ V：前向包含一次矩阵乘法, 反向包含两次矩阵乘法
    attn_flops_scale = flops_scale + flops_scale + (0 if freeze else 1)
    full_attention_flops = 2 * attn_flops_scale * (seq_length ** 2) * hidden_size
    attention_flops = (qkv_proj_flops + full_attention_flops + output_proj_flops) * num_layers

    # mlp flops
    mlp_flops = 2 * 2 * seq_length * hidden_size * vit_cfg.vt_intermediate_size * num_layers * flops_scale

    # vit flops
    vit_flops = patch_embedding_flops + attention_flops + mlp_flops

    # projector flops
    spatial_merge_size = vit_cfg.merge_kernel_size[0]
    seq_length_projector = seq_length // (spatial_merge_size ** 2)
    hidden_size_projector = hidden_size * (spatial_merge_size ** 2)
    hidden_size_llm = vit_cfg.text_hidden_size

    projector_flops = (2 * seq_length_projector * (hidden_size_projector ** 2) \
        + 2 * seq_length_projector * hidden_size_projector * hidden_size_llm) * flops_scale

    return vit_flops + projector_flops


def compute_llm_mlp_flops(text_cfg, tokens_sum):
    hidden_size = text_cfg.hidden_size
    intermediate_size = text_cfg.intermediate_size
    num_hidden_layers = text_cfg.num_hidden_layers
    routed_expert_hidden_size = text_cfg.routed_expert_hidden_size
    first_k_dense_replace = text_cfg.first_k_dense_replace
    num_experts = text_cfg.num_experts
    num_experts_per_token = text_cfg.num_experts_per_token
    num_shared_experts = text_cfg.num_shared_experts
    moe_intermediate_size = text_cfg.moe_intermediate_size

    # dense flops
    per_dense_flops = 2 * tokens_sum * hidden_size * intermediate_size * 3
    dense_flops = per_dense_flops * first_k_dense_replace * 3

    # moe flops
    down_proj_flops = 2 * tokens_sum * hidden_size * routed_expert_hidden_size
    gate_flops = 2 * tokens_sum * hidden_size * num_experts

    shared_flops = 2 * tokens_sum * hidden_size * moe_intermediate_size * 3 * num_shared_experts
    up_proj_flops = 2 * tokens_sum * hidden_size * routed_expert_hidden_size
    router_flops = 2 * (num_experts_per_token * tokens_sum) * routed_expert_hidden_size * moe_intermediate_size * 3

    per_moe_flops = down_proj_flops + gate_flops + shared_flops + up_proj_flops + router_flops
    moe_flops = per_moe_flops * (num_hidden_layers - first_k_dense_replace) * 3
    return dense_flops + moe_flops

def compute_llm_attn_flops(text_cfg, tokens_sum, batch_seqlens):
    hidden_size = text_cfg.hidden_size
    num_hidden_layers = text_cfg.num_hidden_layers
    mla_num_layers = len(text_cfg.linear_attn_config["full_attn_layers"])
    kda_num_layers = len(text_cfg.linear_attn_config["kda_layers"])
    assert mla_num_layers + kda_num_layers == num_hidden_layers, f"MLA layers num {mla_num_layers} and KDA layers num {kda_num_layers} do not sum up to total layers {num_hidden_layers}"

    # mla flops
    q_lora_rank = text_cfg.q_lora_rank
    qk_rope_head_dim = text_cfg.qk_rope_head_dim
    kv_lora_rank = text_cfg.kv_lora_rank
    v_head_dim = text_cfg.v_head_dim
    qk_nope_head_dim = text_cfg.qk_nope_head_dim
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    num_heads = text_cfg.num_attention_heads

    mla_attention_flops = 0
    # q_a_proj
    mla_attention_flops += 2 * tokens_sum * hidden_size * q_lora_rank
    # q_b_proj
    mla_attention_flops += 2 * tokens_sum * q_lora_rank * (num_heads * q_head_dim)
    # kv_a_proj_with_mqa
    mla_attention_flops += 2 * tokens_sum * hidden_size * (kv_lora_rank + qk_rope_head_dim)
    #  kv_b_proj
    mla_attention_flops += 2 * tokens_sum * kv_lora_rank * (num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim))
    # g_proj
    mla_attention_flops += 2 * tokens_sum * hidden_size * (num_heads * v_head_dim)
    # o_proj
    mla_attention_flops += 2 * tokens_sum * (num_heads * v_head_dim) * hidden_size

    # attention
    attn_qkv_flops = 0
    for seqlen in batch_seqlens:
        attn_qkv_flops += (2 * seqlen * seqlen * q_head_dim * num_heads * (3 + 1) + \
                           2 * seqlen * seqlen * v_head_dim * num_heads * 3) / 2

    total_mla_flops = (mla_attention_flops * 3 + attn_qkv_flops) * mla_num_layers

    # kda
    linear_attn_head_dim = text_cfg.linear_attn_config['head_dim']
    linear_attn_num_heads = text_cfg.linear_attn_config['num_heads']

    projection_k_size = linear_attn_head_dim * linear_attn_num_heads
    projection_size = linear_attn_head_dim * linear_attn_num_heads
    kimi_delta_attention_flops = 0
    # q_proj
    kimi_delta_attention_flops += 2 * tokens_sum * hidden_size * projection_k_size
    # k_proj
    kimi_delta_attention_flops += 2 * tokens_sum * hidden_size * projection_size
    # v_proj
    kimi_delta_attention_flops += 2 * tokens_sum * hidden_size * projection_size
    # f_a_proj
    kimi_delta_attention_flops += 2 * tokens_sum * hidden_size * linear_attn_head_dim
    # f_b_proj
    kimi_delta_attention_flops += 2 * tokens_sum * linear_attn_head_dim * projection_size
    # b_proj
    kimi_delta_attention_flops += 2 * tokens_sum * hidden_size * linear_attn_num_heads
    # o_proj
    kimi_delta_attention_flops += 2 * tokens_sum * projection_size * hidden_size
    # g_proj
    kimi_delta_attention_flops += 2 * tokens_sum * projection_size * hidden_size

    BT = 64 # chunksize 默认是 64
    HV = linear_attn_num_heads
    K = linear_attn_head_dim
    V = linear_attn_head_dim

    """
    A_i: B HV NT 2 K BT BT
    w: B HV NT 2 K BT BT
    u: B HV NT 2 V BT BT
    Aqk: B HV NT 2 K BT BT
    v_i: B HV NT 2 K V BT
    o1: B HV NT 2 K V BT
    o2: B HV NT 2 V BT BT
    S: B HV NT 2 K V BT
    total:B HV NT 2 BT (3K BT + 2V BT + 3KV) = B HV T 2 (3K BT + 2V BT + 3KV)
    如果 K=V=D, total = B HV T D (10 BT + 6D)
    """
    chunk_kda_flops =  tokens_sum * HV * 2 * (3 * K * BT + 2 * V * BT + 3 * K * V)

    total_kda_flops = (chunk_kda_flops + kimi_delta_attention_flops) * 3 * kda_num_layers
    return total_mla_flops + total_kda_flops


def compute_attn_res(text_cfg, tokens_sum):
    hidden_size = text_cfg.hidden_size
    num_hidden_layers = text_cfg.num_hidden_layers
    attn_res_block_size = getattr(text_cfg, "attn_res_block_size", None)
    attn_res_flops = 0
    if attn_res_block_size:
        for layer_idx in range(num_hidden_layers):
            num_blocks = layer_idx // attn_res_block_size
            single_attn_res = 2 * tokens_sum * (num_blocks + 1) * hidden_size
            attn_res_flops += single_attn_res * 2 * 3
    return attn_res_flops


def estimate_llm_flops(text_cfg, tokens_sum, batch_seqlens):
    """
    计算 LLM (Kimi K3) 部分的flops

    Args:
        text_cfg: 文本部分的配置
        tokens_sum: 总token数量
        batch_seqlens: 列表, batch中每条样本的序列长度 [seq_len1, seq_len2, ..., seq_lenB]

    Returns:
        LLM (Kimi K3) 部分的flops
    """
    mlp_flops = compute_llm_mlp_flops(text_cfg, tokens_sum)
    attn_flops = compute_llm_attn_flops(text_cfg, tokens_sum, batch_seqlens)
    attn_res_flops = compute_attn_res(text_cfg, tokens_sum)

    # lm_head
    hidden_size = text_cfg.hidden_size
    vocab_size = text_cfg.vocab_size
    lm_head_flops = 2 * tokens_sum * hidden_size * vocab_size * 3
    return mlp_flops + attn_flops + attn_res_flops + lm_head_flops


def main(args):
    # 生成伪图片
    fake_image = random_pil_image(args.width, args.height)

    # 图像预处理
    mm_inputs = get_mm_inputs(fake_image, args.hf_ckpt_path)
    pixel_values, grid_thw = mm_inputs["pixel_values"], mm_inputs["grid_thws"]

    # 读取模型配置
    hf_cfg = AutoConfig.from_pretrained(args.hf_ckpt_path, trust_remote_code=True)

    # 图像编码器 FLOPs
    image_encoder_flops = 0
    if args.image_num > 0:
        vit_flops_per_image = estimate_vit_flops(hf_cfg, pixel_values.shape[0], freeze=args.freeze_vit)
        image_encoder_flops = vit_flops_per_image * args.gbs * args.image_num
        print(f"Image encoder flops is: {image_encoder_flops / 1e12:.3f} TFLOPS")

    # 文本解码器 FLOPs
    text_decoder_flops = estimate_llm_flops(
        hf_cfg.text_config,
        tokens_sum=args.gbs * args.seq_length,
        batch_seqlens=[args.seq_length] * args.gbs
    )
    print(f"Text decoder flops is: {text_decoder_flops / 1e12} TFLOPS")

    # 总 FLOPs
    total_flops = image_encoder_flops + text_decoder_flops
    print(f"Total FLOPs is: {total_flops / 1e12:.3f} TFLOPS")


    # 单个 rank 的平均 FLOPs
    if args.n_npu <= 0 or args.latency <= 0:
        raise ValueError(f"n_npu {args.n_npu} and latency {args.latency} must be positive values")
    average_flops = total_flops / (args.n_npu * args.latency)
    print(f"Average FLOPs per second per npu is: {average_flops / 1e12:.3f} TFLOPS")

    if args.hardware_flops is not None:
        if args.hardware_flops > 0:
            mfu = average_flops / args.hardware_flops
            print(f"MFU is: {mfu*100:.2f}%")
        else:
            raise ValueError(f"Hardware FLOPs {args.hardware_flops} must be a positive value.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kimi-K3 FLOPs Calculation Tool")
    parser.add_argument("--gbs", type=int, default=1, help="global batch size")
    parser.add_argument('--image_num', type=int, default=1, help='Number of images')
    parser.add_argument('--width', type=int, help='Image width')
    parser.add_argument('--height', type=int, help='Image height')
    parser.add_argument('--seq_length', type=int, help='Text sequence length')
    parser.add_argument('--hf_ckpt_path', type=str, help='HuggingFace config path')
    parser.add_argument('--n_npu', type=int, help='Number of NPU')
    parser.add_argument('--latency', type=float, help='Elapsed time per iteration (s)')
    parser.add_argument('--hardware_flops', type=float, default=None, help='Hardware FLOPs')
    parser.add_argument('--freeze_vit', default=False, action='store_true', help='Freeze VIT')

    args = parser.parse_args()
    main(args)


"""
示例:
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python mindspeed_mm/fsdp/tools/flops_tool/flops_kimik3.py \
    --gbs 16 \
    --image_num 10 \
    --width 1024 \
    --height 1024 \
    --seq_length 8192 \
    --hf_ckpt_path "./mindspeed_mm/fsdp/models/kimik3" \
    --n_npu 16 \
    --latency 34.340
"""
