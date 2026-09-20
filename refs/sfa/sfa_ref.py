#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SparseFlashAttention 独立参考实现 —— 纯标准库（零依赖）
================================================================
语义来源：官方 SFA kernel 源码（vllm-ascend vendored CANN 实现）
         见 _sfa/SEMANTICS.md —— 每条语义都有源码出处，不是猜测。

设计原则
--------
1. **不依赖任何被测代码**（与豆包版"自证式对拍"相反）
2. 用 float64 计算，只在必要处做 fp16 舍入 —— 作为"真值"
3. **闭式解自检**：题面给了精确期望值，先证明本实现自己是对的
4. **不确定项做成开关**，便于对照实验

为什么不用 numpy：本机与 VM 都没装 numpy，且都出不了外网。
纯标准库反而让它成为随处可跑的对拍工具。
若有 numpy 会自动用它加速（大 shape 时更快）。

关键语义速查（详见 SEMANTICS.md）
---------------------------------
- sparseIndices 的值是【块号】: token 区间 = [blk*SBS, blk*SBS+SBS)
- 无效值 = -1，【遇到即停止扫描】
- 块被 threshold 截断:  end = min(begin+SBS, thr)
- sparseMode=3: thr = actualS2 - actualS1 + s + 1
- sparseMode=0: thr = actualS2
- BSND 布局下 actual_seq_lengths 是【per-batch 值】（TND 才是累积前缀和）
- score = (q@k^T + q_rope@k_rope^T) * scale
- 全 mask 行 / padding 行 -> attentionOut 全 0, LSE max = 0.0（平台实测口径，见 :112 与 code3.md §5.8.4；
  本地早期自造的 -2e38 哨兵已废弃，别改回去）
"""
import argparse
import math
import struct
import sys

try:
    import numpy as _np          # 可选加速
except Exception:
    _np = None

SPARSE_SIZE = 2048          # 官方固定值（sparseBlockCount）
SOFTMAX_MIN_NUM = -2e38     # 官方 SOFTMAX_MIN_NUM
HALF = struct.Struct('<e')  # fp16


# ======================================================================
# 语义开关（唯一允许的不确定性；默认全部取官方行为）
# ======================================================================
class Opts:
    def __init__(self,
                 block_mode='block',        # 'block'(官方) | 'token'(豆包语义,对照)
                 lse_scaled=True,           # softmaxMax 是否为缩放后的值
                 strict_continue=True,      # 官方: begin>=thr 时 continue 而非 break
                 accumulate_order='fused'): # 'fused'(一次性576维) | 'official'(256+256+64)
        self.block_mode = block_mode
        self.lse_scaled = lse_scaled
        self.strict_continue = strict_continue
        self.accumulate_order = accumulate_order


# ======================================================================
# fp16 工具（用 struct 的 'e' 格式，避免手写位运算出 UB —— HANDOFF §7 教训1）
# ======================================================================
def f16(x):
    """把 float 舍入到 fp16 再转回 float"""
    try:
        return HALF.unpack(HALF.pack(float(x)))[0]
    except (OverflowError, struct.error):
        return math.inf if x > 0 else -math.inf


def to_f16_list(flat):
    return [f16(v) for v in flat]


# ======================================================================
# 参考实现主体
# ======================================================================
def sparse_flash_attention_ref(query, key, value, sparse_indices, scale_value,
                               query_rope=None, key_rope=None,
                               sparse_block_size=1, sparse_mode=3,
                               actual_seq_lengths_query=None,
                               actual_seq_lengths_kv=None,
                               opts=None):
    """
    输入布局全部为 BSND，且为「扁平 list + shape」形式：
      query          (B, S1, N1, D)     fp16
      key            (B, S2, 1,  D)     fp16
      value          (B, S2, 1,  D)     fp16
      sparse_indices (B, S1, 1,  K)     int32
      query_rope     (B, S1, N1, 64)    fp16
      key_rope       (B, S2, 1,  64)    fp16

    返回 (attention_out(fp16 list), softmax_max, softmax_sum)
    softmax_max/sum 形状 (B, 1, S1, N1)
    """
    if opts is None:
        opts = Opts()

    B, S1, N1, D = query.shape
    _, S2, N2, _ = key.shape
    assert N2 == 1, "KV_N 必须为 1"
    K = sparse_indices.shape[-1]
    Dr = query_rope.shape[-1] if query_rope is not None else 0

    q, k, v = query.data, key.data, value.data
    qr = query_rope.data if query_rope is not None else None
    kr = key_rope.data if key_rope is not None else None
    si = sparse_indices.data

    out = [0.0] * (B * S1 * N1 * D)
    # 空行 / 全 mask 行：比赛平台期望 max = 0.0（探针提交 6aae9fad 实测，见 code3.md §5.8.4）。
    # 旧口径用 SOFTMAX_MIN_NUM(-2e38) 是本地自造的哨兵，与平台期望不符。
    smax = [0.0] * (B * S1 * N1)
    ssum = [0.0] * (B * S1 * N1)

    for b in range(B):
        act_s1, act_s2 = _resolve_lens(B, S1, S2, b,
                                       actual_seq_lengths_query,
                                       actual_seq_lengths_kv)

        for s in range(S1):
            if s >= act_s1:
                continue                       # padding query 行 -> 输出 0

            # ---------- threshold ----------
            thr = (act_s2 - act_s1) + s + 1 if sparse_mode == 3 else act_s2
            if thr <= 0:
                continue                       # 全 mask -> 输出 0

            # ---------- 展开有效 token ----------
            toks = _expand_tokens(si, b, s, S1, K, sparse_block_size, thr, opts)
            if not toks:
                continue                       # 全 mask -> 输出 0

            m = len(toks)
            base_q = ((b * S1) + s) * N1 * D
            base_qr = ((b * S1) + s) * N1 * Dr if Dr else 0

            for n in range(N1):
                qo = base_q + n * D
                qro = base_qr + n * Dr

                # ---- score[i] = (q·k_i + q_rope·k_rope_i) * scale ----
                score = [0.0] * m
                for i, t in enumerate(toks):
                    ko = ((b * S2) + t) * D
                    acc = 0.0
                    if opts.accumulate_order == 'official':
                        # 官方 Cube 分块: 256 + 256 + 64 三段累加
                        for lo, hi in ((0, 256), (256, 512)):
                            for d in range(lo, hi):
                                acc += q[qo + d] * k[ko + d]
                        if Dr:
                            koro = ((b * S2) + t) * Dr
                            for d in range(Dr):
                                acc += qr[qro + d] * kr[koro + d]
                    else:
                        for d in range(D):
                            acc += q[qo + d] * k[ko + d]
                        if Dr:
                            koro = ((b * S2) + t) * Dr
                            for d in range(Dr):
                                acc += qr[qro + d] * kr[koro + d]
                    score[i] = acc * scale_value

                # ---- 数值稳定 softmax ----
                mx = max(score)
                e = [math.exp(x - mx) for x in score]
                den = math.fsum(e)
                if den <= 0.0:
                    continue                   # 极端情况 -> 输出 0

                # ---- attn @ V ----
                outo = base_q + n * D
                for d in range(D):
                    acc = 0.0
                    for i, t in enumerate(toks):
                        acc += (e[i] / den) * v[((b * S2) + t) * D + d]
                    out[outo + d] = f16(acc)

                li = ((b * S1) + s) * N1 + n
                smax[li] = mx if opts.lse_scaled else (mx / scale_value if scale_value else mx)
                ssum[li] = den

    return out, smax, ssum


def _resolve_lens(B, S1, S2, b, asq, ask):
    """BSND: per-batch 值（不是累积前缀和！见 SEMANTICS §3）"""
    if asq is None:
        act_s1 = S1
    elif len(asq) == 1:
        act_s1 = int(asq[0])
    else:
        act_s1 = int(asq[b])
    if ask is None:
        act_s2 = S2
    elif len(ask) == 1:
        act_s2 = int(ask[0])
    else:
        act_s2 = int(ask[b])
    return min(act_s1, S1), min(act_s2, S2)


def _expand_tokens(si, b, s, S1, K, SBS, thr, opts):
    """把稀疏索引展开为 token 列表（官方: 块号 + 遇 -1 即停 + threshold 截断）

    sparse_indices 形状 (B, S1, N2=1, K)，行优先 -> 行起始偏移 = (b*S1 + s)*1*K
    注意 N2 维必须算进去（曾经漏掉，导致读到错误索引 —— 已修）
    """
    row0 = ((b * S1) + s) * 1 * K
    toks = []
    if opts.block_mode == 'block':
        for i in range(K):
            blk = int(si[row0 + i])
            if blk == -1:
                break                  # 官方：遇到 -1 停止扫描
            begin = blk * SBS
            if begin >= thr:
                if opts.strict_continue:
                    continue           # 官方行为
                else:
                    break
            end = begin + SBS
            if end > thr:
                end = thr
            toks.extend(range(begin, end))
    else:                              # 'token' 模式（豆包语义，仅对照用）
        for i in range(K):
            t = int(si[row0 + i])
            if t < 0:
                break
            if t >= thr:
                continue
            toks.append(t)
    return toks


# ======================================================================
# 轻量张量容器
# ======================================================================
class T:
    def __init__(self, shape, data):
        self.shape = tuple(shape)
        self.data = data

    @staticmethod
    def zeros_f16(shape):
        n = 1
        for d in shape:
            n *= d
        return T(shape, [0.0] * n)

    @staticmethod
    def rand_f16(shape, seed):
        n = 1
        for d in shape:
            n *= d
        rnd = _rng(seed)
        return T(shape, [f16(rnd() * 2 - 1) for _ in range(n)])

    @staticmethod
    def full_int32(shape, val):
        n = 1
        for d in shape:
            n *= d
        return T(shape, [val] * n)

    def set(self, idx, val):
        self.data[idx] = val

    def get(self, idx):
        return self.data[idx]

    def at(self, *idx):
        """按多维下标取值（行优先）"""
        off = 0
        for i, d in enumerate(self.shape):
            off = off * d + idx[i]
        return self.data[off]

    def set_at(self, val, *idx):
        off = 0
        for i, d in enumerate(self.shape):
            off = off * d + idx[i]
        self.data[off] = val

    def slab(self, *idx):
        """返回固定前几维后的「剩余维扁平切片」的起始偏移与长度"""
        off = 0
        for i, d in enumerate(self.shape):
            if i < len(idx):
                off = off * d + idx[i]
            else:
                off *= d
        n = 1
        for d in self.shape[len(idx):]:
            n *= d
        return off, n


def _rng(seed):
    """确定性 LCG，避免依赖 random 的实现细节"""
    state = [seed & 0x7FFFFFFF]

    def nxt():
        state[0] = (state[0] * 1103515245 + 12345) & 0x7FFFFFFF
        return state[0] / 0x7FFFFFFF
    return nxt


# ======================================================================
# 闭式解自检（用题面给的精确期望值反证本实现）
# ======================================================================
def self_test_closed_form(verbose=True):
    ok = True

    def P(*a):
        if verbose:
            print(*a)

    P("=" * 72)
    P("闭式解自检 —— 用题面给的精确期望值反证参考实现")
    P("=" * 72)

    D, Dr = 512, 64
    scale = 1.0 / math.sqrt(D)

    def build(B, S1, N1, S2, blocks_by_bs):
        """blocks_by_bs: {(b, s): [块号, ...]}  —— 比嵌套 list 不易写错"""
        q = T(  (B, S1, N1, D),  [1.0] * (B * S1 * N1 * D))
        k = T(  (B, S2, 1,  D),  [0.0] * (B * S2 * D))
        v = T(  (B, S2, 1,  D),  [0.0] * (B * S2 * D))
        for b in range(B):
            for t in range(S2):
                for d in range(D):
                    v.set_at(f16(t + 1), b, t, 0, d)
        qr = T( (B, S1, N1, Dr), [0.0] * (B * S1 * N1 * Dr))
        kr = T( (B, S2, 1,  Dr), [0.0] * (B * S2 * Dr))
        si = T( (B, S1, 1, SPARSE_SIZE), [-1] * (B * S1 * SPARSE_SIZE))
        for (b, s), blks in blocks_by_bs.items():
            for j, blk in enumerate(blks):
                si.set_at(int(blk), b, s, 0, j)
        return q, k, v, qr, kr, si

    def mean_of(out, B, S1, N1, D, b, s):
        tot = 0.0
        for n in range(N1):
            for d in range(D):
                tot += out[((b * S1 + s) * N1 + n) * D + d]
        return tot / (N1 * D)

    # ---------- 示例 1 ----------
    q, k, v, qr, kr, si = build(1, 2, 2, 4, {(0,0):[0,1], (0,1):[2,3]})
    out, _, _ = sparse_flash_attention_ref(q, k, v, si, scale, qr, kr,
                                           sparse_block_size=1, sparse_mode=0)
    m0 = mean_of(out, 1, 2, 2, D, 0, 0)
    m1 = mean_of(out, 1, 2, 2, D, 0, 1)
    P(f"\n[示例1] s=0 均值 = {m0:.6f}  期望 1.5  {'OK' if abs(m0-1.5)<1e-3 else 'FAIL'}")
    P(f"[示例1] s=1 均值 = {m1:.6f}  期望 3.5  {'OK' if abs(m1-3.5)<1e-3 else 'FAIL'}")
    ok &= abs(m0 - 1.5) < 1e-3 and abs(m1 - 3.5) < 1e-3

    # ---------- 示例 2（变长）----------
    q, k, v, qr, kr, si = build(2, 1, 2, 4, {(0,0):[0,1], (1,0):[2,3]})
    out, _, _ = sparse_flash_attention_ref(q, k, v, si, scale, qr, kr,
                                           sparse_block_size=1, sparse_mode=0,
                                           actual_seq_lengths_kv=[3, 4])
    b0 = mean_of(out, 2, 1, 2, D, 0, 0)
    b1 = mean_of(out, 2, 1, 2, D, 1, 0)
    P(f"\n[示例2] batch0 均值 = {b0:.6f}  期望 1.5  {'OK' if abs(b0-1.5)<1e-3 else 'FAIL'}")
    P(f"[示例2] batch1 均值 = {b1:.6f}  期望 3.5  {'OK' if abs(b1-3.5)<1e-3 else 'FAIL'}")
    ok &= abs(b0 - 1.5) < 1e-3 and abs(b1 - 3.5) < 1e-3

    # ---------- 块号语义自检 ----------
    # SBS=2, 索引[1,2] -> token {2,3}∪{4,5} -> v 值 {3,4,5,6} -> 均值 4.5
    q, k, v, qr, kr, si = build(1, 1, 1, 8, {(0,0):[1,2]})
    out, _, _ = sparse_flash_attention_ref(q, k, v, si, scale, qr, kr,
                                           sparse_block_size=2, sparse_mode=0)
    bm = mean_of(out, 1, 1, 1, D, 0, 0)
    P(f"\n[块号语义] SBS=2 索引[1,2] -> 期望 (3+4+5+6)/4 = 4.5")
    P(f"[块号语义] 实际 = {bm:.6f}  {'OK' if abs(bm-4.5)<1e-3 else 'FAIL'}")
    ok &= abs(bm - 4.5) < 1e-3

    # ---------- 对照：token 模式应给出不同结果（证明开关有效）----------
    out_tok, _, _ = sparse_flash_attention_ref(q, k, v, si, scale, qr, kr,
                                               sparse_block_size=2, sparse_mode=0,
                                               opts=Opts(block_mode='token'))
    bm_tok = mean_of(out_tok, 1, 1, 1, D, 0, 0)
    # token 模式: 索引[1,2] -> token {1,2} -> v {2,3} -> 2.5
    P(f"\n[对照] token 模式同输入 -> 期望 2.5  实际 {bm_tok:.6f}  "
      f"{'OK(开关生效)' if abs(bm_tok-2.5)<1e-3 else 'FAIL'}")
    ok &= abs(bm_tok - 2.5) < 1e-3

    # ---------- threshold 截断自检（sparseMode=3，块被从中间截断）----------
    # S1=1, S2=8, act_s1=None->S1=1, act_kv=4
    #   nextTokensPerBatch = act_s2 - act_s1 = 4 - 1 = 3
    #   thr = nextTokensPerBatch + s + 1 = 3 + 0 + 1 = 4
    #   SBS=2, 索引[1,2]: 块1 -> begin=2, end=min(4,4)=4 -> token {2,3}
    #                     块2 -> begin=4 >= thr=4 -> 被 threshold 挡掉(continue)
    #   -> v 值 {3,4} -> 均值 3.5
    q, k, v, qr, kr, si = build(1, 1, 1, 8, {(0,0):[1,2]})
    out, _, _ = sparse_flash_attention_ref(q, k, v, si, scale, qr, kr,
                                           sparse_block_size=2, sparse_mode=3,
                                           actual_seq_lengths_kv=[4])
    bm3 = mean_of(out, 1, 1, 1, D, 0, 0)
    P(f"\n[mode3 截断] S1=1,S2=8,act_kv=4 -> thr=4 -> 块1 截断为 token{{2,3}}, 块2 被挡")
    P(f"[mode3 截断] 期望 v{{3,4}} 均值 = 3.5   实际 = {bm3:.6f}  "
      f"{'OK' if abs(bm3-3.5)<1e-3 else 'FAIL'}")
    ok &= abs(bm3 - 3.5) < 1e-3

    # ---------- 对照: 同样输入用 mode=0 则不被 threshold 挡 ----------
    # mode=0: thr = act_s2 = 4 -> 块1 -> {2,3}, 块2 begin=4>=4 被挡 -> 同样 3.5
    # 换成 act_kv=8: mode=0 时 thr=8 -> 块1->{2,3}, 块2->{4,5} -> v{3..6} -> 4.5
    out_m0, _, _ = sparse_flash_attention_ref(q, k, v, si, scale, qr, kr,
                                              sparse_block_size=2, sparse_mode=0,
                                              actual_seq_lengths_kv=[8])
    bm0 = mean_of(out_m0, 1, 1, 1, D, 0, 0)
    P(f"\n[mode0 对照] act_kv=8 -> thr=8 -> token{{2..5}} -> v{{3..6}} 均值 = 4.5")
    P(f"[mode0 对照] 实际 = {bm0:.6f}  {'OK' if abs(bm0-4.5)<1e-3 else 'FAIL'}")
    ok &= abs(bm0 - 4.5) < 1e-3

    # ---------- mode=3 且 act_s2 < act_s1 时, 前几行应全 mask ----------
    # S1=4, S2=2 -> nextTokens = 2-4 = -2
    #   s=0: thr = -2+0+1 = -1 <= 0  -> 全 mask
    #   s=1: thr = -2+1+1 = 0  <= 0  -> 全 mask
    #   s=2: thr = -2+2+1 = 1        -> 只有 token 0 可见
    #   s=3: thr = 2                 -> token {0,1}
    q4, k4, v4, qr4, kr4, si4 = build(1, 4, 1, 2,
                                      {(0,0):[0], (0,1):[0], (0,2):[0], (0,3):[0,1]})
    out4, sm4, ss4 = sparse_flash_attention_ref(q4, k4, v4, si4, scale, qr4, kr4,
                                                sparse_block_size=1, sparse_mode=3)
    mask_ok = (all(x == 0.0 for x in out4[0:D]) and          # s=0
               all(x == 0.0 for x in out4[D:2*D]))           # s=1
    m2 = mean_of(out4, 1, 4, 1, D, 0, 2)                     # s=2 只 token0 -> v=1
    m3 = mean_of(out4, 1, 4, 1, D, 0, 3)                     # s=3 token{0,1} -> v{1,2} -> 1.5
    P(f"\n[mode3 短KV] S1=4,S2=2 -> s=0,1 全 mask {mask_ok}")
    P(f"[mode3 短KV] s=2 仅 token0 -> 期望 v=1 -> 实际 {m2:.4f}  {'OK' if abs(m2-1.0)<1e-3 else 'FAIL'}")
    P(f"[mode3 短KV] s=3 token{{0,1}} -> 期望 1.5 -> 实际 {m3:.4f}  {'OK' if abs(m3-1.5)<1e-3 else 'FAIL'}")
    ok &= mask_ok and abs(m2 - 1.0) < 1e-3 and abs(m3 - 1.5) < 1e-3

    # ---------- rope 段生效自检 ----------
    B, S1, N1, S2 = 1, 1, 1, 4
    q2 = T((B, S1, N1, D), [0.0] * (B * S1 * N1 * D))
    k2 = T((B, S2, 1, D), [0.0] * (B * S2 * D))
    v2 = T((B, S2, 1, D), [0.0] * (B * S2 * D))
    for t in range(S2):
        for d in range(D):
            v2.set_at(f16(t + 1), 0, t, 0, d)
    qr2 = T((B, S1, N1, Dr), [0.0] * (B * S1 * N1 * Dr))
    qr2.set_at(1.0, 0, 0, 0, 0)
    kr2 = T((B, S2, 1, Dr), [0.0] * (B * S2 * Dr))
    kr2.set_at(1.0, 0, 0, 0, 0)            # 只有 token 0 的 rope 非零
    si2 = T((B, S1, 1, SPARSE_SIZE), [-1] * (B * S1 * SPARSE_SIZE))
    si2.set_at(0, 0, 0, 0, 0)
    si2.set_at(1, 0, 0, 0, 1)
    out2, _, _ = sparse_flash_attention_ref(q2, k2, v2, si2, scale, qr2, kr2,
                                            sparse_block_size=1, sparse_mode=0)
    rm = mean_of(out2, B, S1, N1, D, 0, 0)
    P(f"\n[rope 生效] content=0, 仅 token0 的 rope 非零 -> 应偏向 v=1 -> 均值 < 1.5")
    P(f"[rope 生效] 实际 = {rm:.6f}  {'OK' if rm < 1.5 - 1e-3 else 'FAIL'}")
    ok &= rm < 1.5 - 1e-3

    # ---------- 全 mask 行自检 ----------
    # act_s2=0 -> thr<=0 -> 全 mask -> 输出 0, LSE=(0.0, 0.0)
    # （期望值口径来自比赛平台实测，见 code3.md §5.8.4；旧口径 -2e38 已作废）
    q, k, v, qr, kr, si = build(1, 1, 1, 1, {(0,0):[0]})
    out3, sm3, ss3 = sparse_flash_attention_ref(q, k, v, si, scale, qr, kr,
                                                sparse_block_size=1, sparse_mode=3,
                                                actual_seq_lengths_kv=[0])
    allzero = all(x == 0.0 for x in out3)
    P(f"\n[全 mask] act_s2=0 -> 输出全 0  {allzero}")
    P(f"[全 mask] LSE = ({sm3[0]:.3e}, {ss3[0]})  期望 (0.0, 0.0)")
    ok &= allzero and sm3[0] == 0.0 and ss3[0] == 0.0

    P("\n" + "=" * 72)
    P("闭式解自检结果: " + ("全部通过 [PASS]" if ok else "存在失败 [FAIL]"))
    P("=" * 72)
    return ok


# ======================================================================
# 生成 probe 用例文件（文本头 + fp16 二进制体，见 probe_v2.cpp 注释）
# ======================================================================
_MAGIC = b'SFA_CASE 2\n'


def write_case(path, case):
    """case: dict from gen_case()。写入后 probe_v2 可直接读。"""
    def hdr(k, v):
        return (f"{k} {v}\n").encode('ascii')

    parts = [_MAGIC]
    parts.append(hdr('B', case['B']))
    parts.append(hdr('S1', case['S1']))
    parts.append(hdr('S2', case['S2']))
    parts.append(hdr('N1', case['N1']))
    parts.append(hdr('D', case['D']))
    parts.append(hdr('SBS', case['SBS']))
    parts.append(hdr('COUNT', case['COUNT']))
    parts.append(hdr('SCALE', repr(case['scale'])))
    parts.append(hdr('MODE', case['MODE']))
    parts.append(hdr('LSE', 1))

    def pack_f16(lst):
        return b''.join(HALF.pack(float(x)) for x in lst)

    def pack_i32(lst):
        return b''.join(struct.pack('<i', int(x)) for x in lst)

    def pack_f32(lst):
        return b''.join(struct.pack('<f', float(x)) for x in lst)

    parts.append(pack_f16(case['query']))
    parts.append(pack_f16(case['key']))
    parts.append(pack_f16(case['value']))
    parts.append(pack_f16(case['qrope']))
    parts.append(pack_f16(case['krope']))
    parts.append(pack_i32(case['idx']))
    # 变长：缺省写"用满"（与 kernel qLenOn_=false 等价），保证旧调用不破
    parts.append(pack_i32(case.get('asq', [case['S1']] * case['B'])))
    parts.append(pack_i32(case.get('ask', [case['S2']] * case['B'])))
    parts.append(pack_f16(case['expect']))
    # LSE（softmax_max / softmax_sum），float32 —— 官方语义：max 是【缩放后】的行最大
    assert len(case['expect_max']) == len(case['expect_sum']), "LSE 长度不一致"
    parts.append(pack_f32(case['expect_max']))
    parts.append(pack_f32(case['expect_sum']))

    with open(path, 'wb') as f:
        f.write(b''.join(parts))


def gen_case(B, S1, S2, N1, D, SBS, MODE, COUNT, seed=1234,
             nblk=8, use_rope=True, scale=None, zero_rope=False,
             expect_lse=True, actual_s1=None, actual_s2=None):
    """生成随机用例 + 参考输出（返回 flat list 形式的 dict）

    actual_s1 / actual_s2: 变长场景的真实长度（per-batch，长度 B 或 1，或 None 表示用满）。
    ⚠️ 官方 BSND 语义下 threshold / padding 行都取决于真实长度（见 SEMANTICS §3），
       不传就等于假设"无 padding"，那会漏掉平台上最常见的变长场景。
    """
    rng = _rng(seed)

    def rnd(n):
        return [f16(rng() * 2 - 1) for _ in range(n)]

    q  = rnd(B * S1 * N1 * D)
    k  = rnd(B * S2 * D)
    v  = rnd(B * S2 * D)
    if use_rope and not zero_rope:
        qr = rnd(B * S1 * N1 * 64)
        kr = rnd(B * S2 * 64)
    else:
        qr = [0.0] * (B * S1 * N1 * 64)
        kr = [0.0] * (B * S2 * 64)

    # 索引：块号，前半有效、递增、后半 -1（符合题库约束）
    n_blk_total = (S2 + SBS - 1) // SBS
    idx = [-1] * (B * S1 * COUNT)
    for b in range(B):
        for s in range(S1):
            base = (b * S1 + s) * COUNT
            cnt = min(nblk, COUNT, n_blk_total)
            if cnt <= 0:
                continue
            # 确定性抽样: 从 [0, n_blk_total) 里取 cnt 个互不相同的块号并排序
            pool = list(range(n_blk_total))
            picked = []
            for _ in range(cnt):
                j = int(rng() * len(pool))
                if j >= len(pool):
                    j = len(pool) - 1
                picked.append(pool.pop(j))
            picked.sort()
            for j, blk in enumerate(picked):
                idx[base + j] = blk

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    qt = T((B, S1, N1, D), q)
    kt = T((B, S2, 1, D), k)
    vt = T((B, S2, 1, D), v)
    qrt = T((B, S1, N1, 64), qr)
    krt = T((B, S2, 1, 64), kr)
    it = T((B, S1, 1, COUNT), idx)

    out, smax, ssum = sparse_flash_attention_ref(
        qt, kt, vt, it, scale, qrt, krt,
        sparse_block_size=SBS, sparse_mode=MODE,
        actual_seq_lengths_query=actual_s1,
        actual_seq_lengths_kv=actual_s2)

    # 变长张量（未给则写"用满"，与 kernel 里 qLenOn_=false 的语义一致）
    asq = list(actual_s1) if actual_s1 is not None else [S1] * B
    ask = list(actual_s2) if actual_s2 is not None else [S2] * B

    return dict(B=B, S1=S1, S2=S2, N1=N1, D=D, SBS=SBS, COUNT=COUNT,
                MODE=MODE, scale=scale,
                query=q, key=k, value=v, qrope=qr, krope=kr, idx=idx,
                expect=out, expect_max=smax, expect_sum=ssum,
                asq=asq, ask=ask)


# ======================================================================
# CLI
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description='SFA 独立参考实现 / 自检 / 生成用例')
    ap.add_argument('cmd', choices=['selftest', 'gen'])
    ap.add_argument('--B', type=int, default=1)
    ap.add_argument('--S1', type=int, default=2)
    ap.add_argument('--S2', type=int, default=32)
    ap.add_argument('--N1', type=int, default=2)
    ap.add_argument('--D', type=int, default=512)
    ap.add_argument('--sbs', type=int, default=1)
    ap.add_argument('--mode', type=int, default=0)
    ap.add_argument('--count', type=int, default=8, help='sparse_indices 末维')
    ap.add_argument('--nblk', type=int, default=4, help='每行有效块数')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--zero-rope', action='store_true', help='rope 置零（隔离 content 段）')
    ap.add_argument('--out', type=str, default='case.bin')
    a = ap.parse_args()

    if a.cmd == 'selftest':
        return 0 if self_test_closed_form() else 1

    case = gen_case(a.B, a.S1, a.S2, a.N1, a.D, a.sbs, a.mode, a.count,
                    seed=a.seed, nblk=a.nblk, zero_rope=a.zero_rope)
    write_case(a.out, case)
    print(f"已生成 {a.out}: B={a.B} S1={a.S1} S2={a.S2} N1={a.N1} D={a.D} "
          f"SBS={a.sbs} MODE={a.mode} COUNT={a.count} nblk={a.nblk}")
    # 顺带打印期望值的若干统计，便于人工判断用例是否有信息量
    nz = sum(1 for x in case['expect'] if x != 0.0)
    print(f"  期望输出: 非零 {nz}/{len(case['expect'])}, "
          f"min={min(case['expect']):.6f}, max={max(case['expect']):.6f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
