#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P34 前置：**大形状**计时用例（code3.md §15.54(4) 假设①/② 的判据实验）。

为什么要有它：本地七个靶案全在 0.12~0.74 ms 的**发射受限**区，而平台六点 7.8~14.7 ms
⇒ "每个 nb 取最大可行 k"这条贪心只在小区间被验证过（§15.51(2) 的 24 格全在那一区）。
这一族把 `rows×heads` 与 KV 工作集两轴各自放大，用来问：**k 的最优点会不会随形状变大往左走。**

口径与 `gen_sbs.py` 完全一致：**只造输入、expect 填哑值**，判据不看它（看 `dev.sh matrix`）。
两轴分开放大是为了归因：
  * `w1/w2/w3/w4` 只放大 `S1` ⇒ 工作量线性涨、KV 张量不变（每行的 token 数由 nblk 封顶，与 S1 无关）；
  * `w5/w6` 只放大 `S2` ⇒ 工作量不变（token/行仍是 `SBS×nblk`）、**KV 张量footprint** 涨 ⇒ 越过 L2 后是不是 MTE 先撑不住。
用法（真机 ~/sfa_real 下）：python3 gen_bigshape.py w1 w3 w5
"""
import math
import os
import sys
import time

sys.path.insert(0, os.environ.get('SFAREF', os.path.expanduser('~/sfa_real')))
import sfa_ref as S  # noqa: E402

# (名称, B, S1, S2, N1, D, SBS, MODE, COUNT, nblk)
SHAPES = {
    'w1': (1, 4,   8192, 4, 512, 2, 3, 2048, 2048),   # = p1s2，已知 0.245 ms 的基准
    'w2': (1, 32,  8192, 4, 512, 2, 3, 2048, 2048),   # ×8  行×头
    'w3': (1, 128, 8192, 4, 512, 2, 3, 2048, 2048),   # ×32 ⇒ 应落在 8 ms 量级（平台区间）
    'w4': (1, 128, 8192, 8, 512, 2, 3, 2048, 2048),   # ×64 ⇒ 平台最大点之上
    'w5': (1, 32,  32768, 4, 512, 2, 3, 2048, 2048),  # 工作量同 w2，KV footprint ×4（64 MB/份）
    'w6': (1, 4,   131072, 4, 512, 2, 3, 2048, 2048), # 工作量同 w1，KV footprint ×16
}

outdir = os.environ.get('SFA_CASES', os.path.expanduser('~/sfa_real/cases'))
names = sys.argv[1:] or list(SHAPES)
os.makedirs(outdir, exist_ok=True)
for n in names:
    B, S1, S2, N1, D, SBS, MODE, CN, nblk = SHAPES[n]
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
            cnt = min(nblk, CN, n_blk_total)
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

    case = dict(B=B, S1=S1, S2=S2, N1=N1, D=D, SBS=SBS, COUNT=CN, MODE=MODE,
                scale=1.0 / math.sqrt(D),
                query=q, key=k, value=v, qrope=qr, krope=kr, idx=idx,
                expect=[0.0] * (B * S1 * N1 * D),
                expect_max=[0.0] * (B * S1 * N1), expect_sum=[0.0] * (B * S1 * N1),
                asq=[S1] * B, ask=[S2] * B)
    path = os.path.join(outdir, '%s.bin' % n)
    S.write_case(path, case)
    print('%-4s B=%d S1=%-4d S2=%-6d N1=%d SBS=%d COUNT=%-5d 有效块=%-5d '
          'token/行=%-5d 行×头=%-5d KV=%.0f MB  %7.1f MB  %.1fs'
          % (n, B, S1, S2, N1, SBS, CN, nblk, SBS * min(nblk, n_blk_total),
             B * S1 * N1, 2 * B * S2 * (D + 64) * 2 / 1e6,
             os.path.getsize(path) / 1e6, time.time() - t0))
