#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P100：**decode 形复现案**（极少 query 行 × 每行几万~十万 KV token）。

为什么要它：§5.8.3 用"破坏输出→掉多少分"量出平台六点的**行数**是 4/8/4/16/4/32，
P99 又量出六点 **Q_N ≤ 8**，而 P38 六点时间是 7.8~15 ms。三件事只有在一个形状下同真：
`S2` 十万级、每行 gather 数万~十万 token、行数个位到几十。这个形状下我们
`units = B·Q_S·n_blk·ks` 只有 4~64 ⇒ **每核 104 ns/单元 的单价被"核数"卡住**，
而不是被"机制"卡住 —— 这两件事的优化方向完全相反，所以必须先分清楚。

判据（同一族里只差"行数"）：
  · 并行度饥饿模型 ⇒ 墙上 ≈ 每行 token 数 × 104 ns，**与行数无关**（行数 ≤40 时平），
    4 行与 32 行同速 ⇒ 该做的是"把 KV 轴切给更多核"（官方 SFA 的 splitKVNum）。
  · 带宽模型 ⇒ 墙上 ∝ 读进 HBM 的字节数，**与行数成正比**（每行各读一份），
    32 行 ≈ 4 行的 8 倍 ⇒ 该做的是"一行多列复用同一份 KV"。
  · 两个模型的比值 R32/R4 一个 =1、一个 =8 ⇒ 一发本地测量就能裁。

expect 全零 ⇒ 只读时间（正确性另用 `p*` 族的小 decode 案兜）。
"""
import math
import os
import sys

import numpy as np

B, S2, D, SBS, MODE = 1, 131072, 512, 1, 3
NBLK = (S2 + SBS - 1) // SBS          # 块数（SBS=1 ⇒ 就是 token 数）
MAGIC = b'SFA_CASE 2\n'
outdir = os.environ.get('SFA_CASES', os.path.expanduser('~/sfa_real/cases'))

# 名称, S1, N1, COUNT
CASES = [
    ('r4c65',   4,  1, 65536),
    ('r32c65',  32, 1, 65536),
    ('r64c65',  64, 1, 65536),
    ('r4c131',  4,  1, NBLK),
    ('r4n2c65', 4,  2, 65536),
    ('r2c65',   2,  1, 65536),
    ('r8c65',   8,  1, 65536),
    ('r8n2c65', 8,  2, 65536),
    ('r4c16',   4,  1, 16384),
    ('r4n4c65', 4,  4, 65536),
]

want = sys.argv[1:] or [c[0] for c in CASES]
rng = np.random.default_rng(20260924)


def f16(n):
    return (rng.random(n, dtype=np.float32) * 2 - 1).astype('<f2')


# 大张量只造一次，多个案共用同样的 K/V/kr（各案只差行数与每行的选择列表）
key = f16(B * S2 * D)
value = f16(B * S2 * D)
krope = f16(B * S2 * 64)
print('K/V/kr = %.0f MB 复用' % ((key.nbytes + value.nbytes + krope.nbytes) / 1e6))


def idx_rows(s1, n1, count):
    """每行取 count 个**递增**块号（题面语义：有效在前、-1 补尾）。
    格式是 (B, S1, KV_N=1, COUNT) ⇒ N1 个头共用同一份列表，不复制。"""
    out = np.full((s1, count), -1, dtype='<i4')
    for s in range(s1):
        if count >= NBLK:
            out[s] = np.arange(NBLK, dtype='<i4')
        else:
            out[s] = np.sort(rng.choice(NBLK, size=count, replace=False)).astype('<i4')
    return out.reshape(1, s1, 1, count)


def write(path, s1, n1, count):
    q = f16(B * s1 * n1 * D)
    qr = f16(B * s1 * n1 * 64)
    idx = idx_rows(s1, n1, count)
    scale = 1.0 / math.sqrt(D)

    def hdr(k, v):
        return ('%s %s\n' % (k, v)).encode('ascii')

    body = [MAGIC]
    for k, v in (('B', B), ('S1', s1), ('S2', S2), ('N1', n1), ('D', D),
                 ('SBS', SBS), ('COUNT', count), ('SCALE', repr(scale)),
                 ('MODE', MODE), ('LSE', 1)):
        body.append(hdr(k, v))
    body += [q.tobytes(), key.tobytes(), value.tobytes(), qr.tobytes(), krope.tobytes(),
             idx.tobytes(),
             np.full(B, s1, dtype='<i4').tobytes(), np.full(B, S2, dtype='<i4').tobytes(),
             np.zeros(B * s1 * n1 * D, dtype='<f2').tobytes(),
             np.zeros(B * s1 * n1, dtype='<f4').tobytes(),
             np.zeros(B * s1 * n1, dtype='<f4').tobytes()]
    with open(path, 'wb') as f:
        f.write(b''.join(body))


os.makedirs(outdir, exist_ok=True)
for name, s1, n1, count in [c for c in CASES if c[0] in want]:
    p = os.path.join(outdir, '%s.bin' % name)
    write(p, s1, n1, count)
    units = B * s1 * n1 * count
    print('%-8s 行=%-3d N1=%d tok/行=%-6d 单元=%-4d ht·tok=%.2e  %5.0f MB'
          % (name, s1 * B, n1, count, B * s1 * n1, units, os.path.getsize(p) / 1e6))
    print('         饥饿模型预测 %.2f ms   带宽模型预测 %.2f ms'
          % (count * 104e-6, units * 2048 / 8e11 * 1e3))
