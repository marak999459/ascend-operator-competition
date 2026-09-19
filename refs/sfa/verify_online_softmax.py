#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立验证「分块在线 softmax」的数学 —— 不依赖 sfa_ref.py
用一个 1 头、4 token、2 chunk 的手算例子，对比：
  (A) 一次性 softmax（gold）
  (B) 官方 online softmax 更新式
  (C) 我在 kernel 里写的更新式（逐行复刻 C++ 逻辑）

目的：定位 c2_chunk/c4_shortkv/c5_blocks 的失败到底在数学还是别的环节。
"""
import math

# ---- 构造一个可手算的极小例子：1 头，4 个 token，2 个 chunk ----
v = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]   # 4 token × D=2
s = [0.5, -0.2, 1.3, 0.7]                                # score（已含 scale）
D = 2
NEG = -2e38


def softmax(xs):
    mx = max(xs)
    e = [math.exp(x - mx) for x in xs]
    d = sum(e)
    return e, d, mx


def out_from(e, d):
    return [sum(e[j] * v[j][k] for j in range(len(e))) / d for k in range(D)]


# ---------- (A) 一次性（gold） ----------
eA, dA, mxA = softmax(s)
outA = out_from(eA, dA)
print("(A) 一次性 softmax (gold)")
print(f"    max={mxA:.6f} sum={dA:.6f} out={['%.6f' % x for x in outA]}")

# ---------- (B) 官方 online softmax ----------
def online(chunks):
    m = NEG
    l = 0.0
    O = [0.0] * D
    for ch in chunks:
        mx = max(s[j] for j in ch)
        mNew = max(m, mx)
        alpha = math.exp(m - mNew) if m > NEG else 0.0
        # 官方做法：用 SoftmaxFlashV2 的 "update" 语义
        for k in range(D):
            O[k] = O[k] * alpha
        lNew = l * alpha
        for j in ch:
            e = math.exp(s[j] - mNew)
            lNew += e
            for k in range(D):
                O[k] += e * v[j][k]
        m, l = mNew, lNew
    return m, l, [O[k] / l for k in range(D)]


mB, lB, outB = online([[0, 1], [2, 3]])
print("\n(B) 官方 online softmax（2 chunk）")
print(f"    max={mB:.6f} sum={lB:.6f} out={['%.6f' % x for x in outB]}")
print(f"    与 (A) 一致? {all(abs(outB[k]-outA[k]) < 1e-9 for k in range(D))}")

# ---------- (C) 复刻我 kernel 的写法 ----------
def my_kernel_sim(chunks, nblk, trace=False):
    """逐行复刻 sfa_kernel_v2.cpp 的 FlushChunk + ProcessToken 状态机"""
    m = NEG
    l = 0.0
    O = [0.0] * D
    # 状态机
    flat = []
    for ch in chunks:
        flat.extend(ch)
    curIdx = 0            # 位置指针
    hasBlock = False
    filled = 0
    sc = [0.0] * nblk     # score 缓冲
    vb = [0.0] * (nblk * D)
    step = 0
    while True:
        if filled == nblk:
            # ---- FlushChunk ----
            if trace:
                print(f"    [chunk {step}] m_in={m:.6f} l_in={l:.6f} O_in={O}")
            mx = sc[0]
            for j in range(1, filled):
                if sc[j] > mx:
                    mx = sc[j]
            mNew = mx if mx > m else m
            alpha = math.exp(m - mNew)
            for k in range(D):
                O[k] *= alpha
            lNew = l * alpha
            for j in range(filled):
                e = math.exp(sc[j] - mNew)
                lNew += e
                for k in range(D):
                    O[k] += e * vb[j * D + k]
            m, l = mNew, lNew
            if trace:
                print(f"    [chunk {step}] m_out={m:.6f} l_out={l:.6f} O_out={O}")
            filled = 0
            step += 1
        if curIdx >= len(flat):
            break
        # 装一个 token
        j = flat[curIdx]
        sc[filled] = s[j]
        for k in range(D):
            vb[filled * D + k] = v[j][k]
        filled += 1
        curIdx += 1
    if filled > 0:
        mx = sc[0]
        for j in range(1, filled):
            if sc[j] > mx:
                mx = sc[j]
        mNew = mx if mx > m else m
        alpha = math.exp(m - mNew)
        for k in range(D):
            O[k] *= alpha
        lNew = l * alpha
        for j in range(filled):
            e = math.exp(sc[j] - mNew)
            lNew += e
            for k in range(D):
                O[k] += e * vb[j * D + k]
        m, l = mNew, lNew
        if trace:
            print(f"    [chunk {step}] m_out={m:.6f} l_out={l:.6f} O_out={O}")
    return m, l, [(O[k] / l) if l > 0 else 0.0 for k in range(D)]


mC, lC, outC = my_kernel_sim([[0, 1], [2, 3]], nblk=2, trace=True)
print("\n(C) 复刻我 kernel 的写法（N_BLK=2, 2 chunk）")
print(f"    max={mC:.6f} sum={lC:.6f} out={['%.6f' % x for x in outC]}")
print(f"    与 (A) 一致? {all(abs(outC[k]-outA[k]) < 1e-9 for k in range(D))}")

# ---------- (C') 单 chunk 对照 ----------
mC1, lC1, outC1 = my_kernel_sim([[0, 1, 2, 3]], nblk=4)
print("\n(C') 同写法但 N_BLK=4（单 chunk）")
print(f"    max={mC1:.6f} sum={lC1:.6f} out={['%.6f' % x for x in outC1]}")
print(f"    与 (A) 一致? {all(abs(outC1[k]-outA[k]) < 1e-9 for k in range(D))}")
