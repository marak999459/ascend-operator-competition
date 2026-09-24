#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P99b：**恒定工作量、只换稀疏粒度**的一族靶案（SBS × COUNT，SBS·COUNT 恒 = 4096 token/行）。

为什么现在要它：P99 把平台六点钉成 `fp16 ∧ D=512 ∧ dr=64 ∧ Q_N≤8`，而 §1.2 那句
"平台六点 ≈ 本地 w3/w4"从来只是**毫秒量级对上**，没有一个参数是被量过的。我们整套本地
语料（p1~p6 / w1~w6 / k_* / n1x~n8x）清一色 `SBS=2, COUNT=2048` —— 也就是"每行从 4096
个块里挑 2048 个、每块只有 2 个 token"，这是**最碎**的一档。而 `sparseBlockSize` 题面允许
到 128。P55 在 AIC 侧量到"碎拷贝 ≈50 ns/条、与 nValue 无关"⇒ 若平台的点其实很粗
（SBS=64/128），我们这整套单价表连**税在哪**都指错了方向，榜首那 4× 也可能根本不是算法差，
是我们把 DMA 调用条数当成了天花板去撞。

口径：`B=1, S1=512, N1=1, S2=8192, mode=3` 固定，`SBS·COUNT=4096` ⇒ 每行有效 KV token 数
恒定 ⇒ 计算量与读出的字节数都恒定，**唯一变的是每次搬运的连续性**（COUNT 条 → 1 条）。
expect 全填哑零 ⇒ 只读时间。
"""
import math
import os
import sys
import time

sys.path.insert(0, os.environ.get('SFAREF', os.path.expanduser('~/sfa_real')))
import sfa_ref as S  # noqa: E402

# (名称, SBS, COUNT)   —— SBS*COUNT = 4096 = 每行 token 数；S2/SBS 必须 ≥ COUNT
GRID = [
    ('g002', 2, 2048),     # = n1x，本地最碎档（对照）
    ('g004', 4, 1024),
    ('g008', 8, 512),
    ('g016', 16, 256),
    ('g032', 32, 128),
    ('g064', 64, 64),
    ('g128', 128, 32),     # 题面上限档
    ('g127', 128, 16),     # 粗且少挑 ⇒ 每行只读 2048 token（工作量减半，用来看斜率）
]
B, S1, S2, N1, D, MODE = 1, 512, 8192, 1, 512, 3

outdir = os.environ.get('SFA_CASES', os.path.expanduser('~/sfa_real/cases'))
want = sys.argv[1:] or [g[0] for g in GRID]
os.makedirs(outdir, exist_ok=True)
for n, SBS, CN in [g for g in GRID if g[0] in want]:
    t0 = time.time()
    rng = S._rng(hash(n) % 10000)

    def rnd(cnt):
        return [S.f16(rng() * 2 - 1) for _ in range(cnt)]

    q = rnd(B * S1 * N1 * D)
    k = rnd(B * S2 * D)
    v = rnd(B * S2 * D)
    qr = rnd(B * S1 * N1 * 64)
    kr = rnd(B * S2 * 64)

    n_blk_total = (S2 + SBS - 1) // SBS
    idx = [-1] * (B * S1 * CN)
    for b in range(B):
        for s in range(S1):
            base = (b * S1 + s) * CN
            cnt = min(CN, n_blk_total)
            picked = sorted(int(rng() * n_blk_total) for _ in range(cnt))
            for j, blk in enumerate(picked):
                idx[base + j] = blk

    case = dict(B=B, S1=S1, S2=S2, N1=N1, D=D, SBS=SBS, COUNT=CN, MODE=MODE,
                scale=1.0 / math.sqrt(D),
                query=q, key=k, value=v, qrope=qr, krope=kr, idx=idx,
                expect=[0.0] * (B * S1 * N1 * D),
                expect_max=[0.0] * (B * S1 * N1), expect_sum=[0.0] * (B * S1 * N1),
                asq=[S1] * B, ask=[S2] * B)
    path = os.path.join(outdir, '%s.bin' % n)
    S.write_case(path, case)
    print('%-5s SBS=%-4d COUNT=%-5d 可选块=%-5d token/行=%-5d ht=%-5d  %7.1f MB  %.1fs'
          % (n, SBS, CN, n_blk_total, SBS * min(CN, n_blk_total), B * S1 * N1,
             os.path.getsize(path) / 1e6, time.time() - t0))
