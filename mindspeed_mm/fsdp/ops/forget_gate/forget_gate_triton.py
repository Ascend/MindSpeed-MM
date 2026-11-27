# -*- coding: utf-8 -*-
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
"""GLM5.3 Glm5NextTextForgetGate safe-gate 融合 Triton 算子。

来源：
- 草案版本：triton-ascend-kernels attention/fla/kda/safe_gate.py
  （单算子 verify 1/1 通过, fp32 三项判定全过; benchmark 1.840ms -> 0.357ms, 5.16x, 锁频 50 rep）

对外接口：
- fused_safe_gate(g_raw, dt_bias, A_log, lower_bound, head_dim=128)：函数式接口,
  可直接替换 Glm5NextTextForgetGate.forward 的 safe 路径 python 链。

融合内容 (原 8 个 aclnn vector kernel, ~1.84ms/层):
    Cast(bf16->f32) + Cast(dt_bias) + Add(dt_bias) + Cast(A_log)
    + Exp(A_log) + Mul(decay) + Sigmoid + Muls(lower_bound)
    => 单 kernel: Y = lower_bound * sigmoid( exp(A_log[h]) * (raw.f32 + dt_bias[h,d]) )

架构: row-streaming-full-width
    - 参数 dt_bias[HD]/decay[HD] 循环外一次加载, 常驻 UB (64KB f32)
    - 行方向 1D grid-stride, 每迭代整行连续 16KB 读(bf16) / 32KB 写(f32)
    - 全程 fp32 计算, 输出 fp32 与参考链一致

约束:
    - g_raw 总元素 R*HD 须 < 2^31 (int32 索引; 本任务 67M)
    - HD 须为 D 的倍数 (pow2 constexpr)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _safe_gate_fwd_kernel(
    g_raw_ptr,          # [R, HD] bf16, 裸 gate (f_b_proj 输出)
    dt_bias_ptr,        # [HD]    bf16
    a_log_ptr,          # [H]     bf16
    y_ptr,              # [R, HD] f32, 逻辑视图 [B, T, H, D]
    R,                  # 行数 = B*T
    n_progs,            # 必须与实际 grid size 严格相等
    lower_bound,        # 标量, 如 -5.0
    HD: tl.constexpr,   # = H*D = 8192
    D: tl.constexpr,    # = 128
):
    pid = tl.program_id(0)

    # ---- 参数一次加载, 常驻 UB (循环外加载; 64KB f32) ----
    offs = tl.arange(0, HD)                                # int32, constexpr 长度
    dt_b = tl.load(dt_bias_ptr + offs).to(tl.float32)      # [HD] f32, 32KB
    h_idx = offs // D                                      # int32 整除
    decay = tl.exp(tl.load(a_log_ptr + h_idx).to(tl.float32))  # [HD] f32, 32KB; 64项表 gather

    # ---- 行方向 grid-stride: 每迭代整行连续流式 ----
    # base = r * HD: R*HD < 2^31 (本任务 67M), int32 安全
    for r in range(pid, R, n_progs):                       # 偏移每轮从 r 独立派生
        base = r * HD
        x = tl.load(g_raw_ptr + base + offs).to(tl.float32)   # [HD] f32
        # y = lower_bound * sigmoid(exp(A_log[h]) * (raw + dt_bias[h,d])), 全 fp32
        # sigmoid 手写 1/(1+exp(-x)): 与 torch.sigmoid 同公式, 规避 tl.sigmoid 可用性风险
        s = 1.0 / (1.0 + tl.exp(-(decay * (x + dt_b))))
        y = lower_bound * s
        tl.store(y_ptr + base + offs, y)                   # 保持 fp32


# ---- 核数查询, 模块级缓存 (进程内一次, 平台接口获取, 不硬编码) ----
_VEC_CORES_CACHE = None


def _get_vec_cores() -> int:
    global _VEC_CORES_CACHE
    if _VEC_CORES_CACHE is None:
        import torch_npu
        import triton.runtime.driver as driver
        props = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
        _VEC_CORES_CACHE = props["num_vectorcore"]
    return _VEC_CORES_CACHE


def fused_safe_gate(g_raw: torch.Tensor, dt_bias: torch.Tensor, A_log: torch.Tensor,
                    lower_bound: float, head_dim: int = 128) -> torch.Tensor:
    """host wrapper: 只做 shape/contiguous/launch.

    g_raw: [B, T, H*D] bf16 (f_b_proj 输出, 裸 gate)
    返回:  [B, T, H, D] f32
    """
    B, T, HD = g_raw.shape
    H = HD // head_dim
    R = B * T
    g = g_raw if g_raw.is_contiguous() else g_raw.contiguous()
    dt_b = dt_bias if dt_bias.is_contiguous() else dt_bias.contiguous()
    a_lg = A_log if A_log.is_contiguous() else A_log.contiguous()

    y = torch.empty((R, HD), dtype=torch.float32, device=g_raw.device)
    grid = (min(R, _get_vec_cores()),)
    _safe_gate_fwd_kernel[grid](
        g, dt_b, a_lg, y,
        R, grid[0], float(lower_bound),
        HD=HD, D=head_dim,
    )
    return y.view(B, T, H, head_dim)
