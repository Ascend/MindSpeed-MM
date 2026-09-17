# MagiHuman (single-stream audio-video diffusion DiT) FLOPs / MFU 计算工具。
#
# 口径与 flops_qwen3_5.py / flops_kimik2_5.py 完全一致:
#   - 线性投影: 6 (fwd+bwd) * 有效参数 * token 数
#   - 二次注意力(Q@K + attn@V): 7 * sum(seqlen^2) * head_dim * num_heads_q * num_layers
#   - MoE 一律用「有效(激活)参数」而非总参数。MagiHuman 的 mm_layers 是**模态-MoE**
#     (num_modality=3, 每个 token 只走自己模态那一套权重), 故 mm_layers 内所有线性层
#     的有效参数 = 总参数 / num_modality。
#
# 有效参数从 base 权重的分片 safetensors 形状直接读取(逐层 FFN 维不同, 不宜硬编码)。

import argparse
import glob
import json
import re
import struct


# MagiHuman 15B 架构常量(与 inference/common/config.py::ModelConfig 一致, 实测核对)。
DEFAULT_NUM_LAYERS = 40
DEFAULT_NUM_HEADS_Q = 40          # hidden_size // head_dim = 5120 // 128
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_MODALITY = 3          # VIDEO / AUDIO / TEXT，各持一套 mm_layer 权重
DEFAULT_MM_LAYERS = (0, 1, 2, 3, 36, 37, 38, 39)

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def load_shapes(ckpt_path):
    """读分片 safetensors 头部(仅元数据, 不加载张量), 返回 {name: shape}。"""
    shapes = {}
    shard_files = sorted(glob.glob(f"{ckpt_path}/*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"no *.safetensors under {ckpt_path}")
    for shard in shard_files:
        with open(shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for name, meta in header.items():
            if name != "__metadata__":
                shapes[name] = meta["shape"]
    return shapes


class MagiHumanFlopsCounter:
    def __init__(
        self,
        shapes,
        num_layers=DEFAULT_NUM_LAYERS,
        num_heads_q=DEFAULT_NUM_HEADS_Q,
        head_dim=DEFAULT_HEAD_DIM,
        num_modality=DEFAULT_NUM_MODALITY,
        mm_layers=DEFAULT_MM_LAYERS,
    ):
        self.num_layers = num_layers
        self.num_heads_q = num_heads_q
        self.head_dim = head_dim
        # 有效(每 token 激活)线性参数量: mm_layer 内的权重按模态数折算
        self.active_linear_n = self._active_linear_params(shapes, num_modality, set(mm_layers))

    @staticmethod
    def _active_linear_params(shapes, num_modality, mm_layers):
        """逐权重累加每 token 有效参数: 仅 2D 线性权重计入; mm_layer 的权重 / num_modality。"""
        total = 0.0
        for name, shape in shapes.items():
            if len(shape) != 2:          # 跳过 norm / bias / rope 等 1D 及标量
                continue
            numel = shape[0] * shape[1]
            m = _LAYER_RE.search(name)
            if m is not None and int(m.group(1)) in mm_layers:
                numel /= num_modality    # 模态-MoE: 每 token 只激活 1/num_modality
            total += numel
        return total

    def _model_flops(self, seqlens):
        """一组样本(seqlens 为各样本 token 数)的 fwd+bwd 模型 FLOPs。"""
        tokens_sum = sum(seqlens)
        # 线性投影(注意力 qkv/proj + MLP + adapter/final): 6 * 有效参数 * token
        linear_flops = 6 * self.active_linear_n * tokens_sum
        # 二次注意力(Q@K + attn@V), 全部层为 full attention;
        # 系数 7 = fwd 2 + bwd 5(FA 反向重算 kv), 与 flops_qwen3_5 一致。
        seqlen_square_sum = sum(s * s for s in seqlens)
        attn_flops = 7 * seqlen_square_sum * self.head_dim * self.num_heads_q * self.num_layers
        return linear_flops + attn_flops

    def estimate_flops(self, seqlens, step_time):
        """每样本组的 achieved FLOPs/s（尚未按 gbs / device_num 归一, 与 flops_qwen3_5 对齐）。"""
        return self._model_flops(seqlens) / step_time


def get_args():
    parser = argparse.ArgumentParser(description="MagiHuman FLOPs / MFU Calculation Tool")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="base 分片 safetensors 目录(读形状算有效参数)")
    parser.add_argument("--seqlens", type=int, nargs="+", default=[2112],
                        help="每个样本的打包序列 token 数(video+audio+text); 通常传一个样本, 由 --gbs 放大")
    parser.add_argument("--gbs", type=int, default=1, help="global batch size (mbs * dp_size * grad_accum)")
    parser.add_argument("--device_num", type=int, default=1, help="参与计算的 NPU 数")
    parser.add_argument("--step_time", type=float, required=True, help="每步耗时(秒)")
    parser.add_argument("--hardware_flops", type=float, default=None,
                        help="单卡峰值算力(FLOP/s); 910B3 取 2.8e14(280 TFLOPS)")
    parser.add_argument("--num_modality", type=int, default=DEFAULT_NUM_MODALITY)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    counter = MagiHumanFlopsCounter(load_shapes(args.ckpt), num_modality=args.num_modality)
    print(f"active (per-token) linear params: {counter.active_linear_n / 1e9:.3f}B")

    # 每卡 achieved FLOP/s = 单样本组 FLOPs/step_time * gbs / device_num（与 flops_qwen3_5 主流程一致）
    flops = counter.estimate_flops(args.seqlens, args.step_time) * args.gbs / args.device_num
    print(f"flops: {flops:.4e}  ({flops / 1e12:.1f} TFLOP/s per device)")

    if args.hardware_flops is not None:
        if args.hardware_flops > 0:
            mfu = flops / args.hardware_flops
            print(f"MFU is: {mfu * 100:.2f}%")
        else:
            raise ValueError("Hardware FLOPs must be a positive value.")


"""
e.g. (8 卡 FSDP2 全参, mbs=1, 打包 4606-token, 实测步耗时 4.6348s, 910B3 峰值 2.8e14=280 TFLOPS):
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python mindspeed_mm/fsdp/tools/flops_tool/flops_magihuman.py \
    --ckpt /path/to/daVinci-MagiHuman/base \
    --seqlens 4606 \
    --gbs 8 \
    --device_num 8 \
    --step_time 4.6348 \
    --hardware_flops 2.8e14
"""
