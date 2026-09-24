#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P99：把 §1.3 那张单价表缺的那两列补上 —— **大形状 × 小头数**（N1=1/2）。

动机（P98-F 的读数）：平台六点用"只给 fp32 夹 blockDim"的判别器打出去，六点全部落在
±7 % 噪声带里 ⇒ **平台喂的是 fp16**。fp16 + 形态合法（题面钉死 Q_D=512、Dr=64）下，
A′ 的形态门只剩一条能在六点上一律关死的拒因：**`CubeBlock(1)==0` ⇒ `Q_N==1` 从来进不了
cube**。而"MLA-absorb"这个题名本身就把头数往 1 上逼。

于是这里问一个纯本地、不花提交的问题：**我们的 AIV 路径在 N1=1/2 的大形状上值多少**。
口径：**固定 head·token = 512**（与 w3 同量级的总工作量），只动 N1 与 S1：
  n1x : N1=1, S1=512   n2x : N1=2, S1=256
  n4x : N1=4, S1=128（= w3，对照）   n8x : N1=8, S1=64
其余参数逐字同 w3（S2=8192 / SBS=2 / mode=3 / COUNT=2048 / nblk=2048 ⇒ 每行 4096 个 KV
token）。expect 填哑零 ⇒ 只读时间不读判据。
"""
import math
import os
import sys
import time

sys.path.insert(0, os.environ.get('SFAREF', os.path.expanduser('~/sfa_real')))
import sfa_ref as S  # noqa: E402

# (名称, B, S1, S2, N1, D, SBS, MODE, COUNT, nblk)   —— B*S1*N1 恒 = 512
SHAPES = {
    'n1x': (1, 512, 8192, 1, 512, 2, 3, 2048, 2048),
    'n2x': (1, 256, 8192, 2, 512, 2, 3, 2048, 2048),
    'n4x': (1, 128, 8192, 4, 512, 2, 3, 2048, 2048),
    'n8x': (1,  64, 8192, 8, 512, 2, 3, 2048, 2048),
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
    print('%-4s B=%d S1=%-4d S2=%-6d N1=%d SBS=%d COUNT=%-5d token/行=%-5d '
          '行×头=%-5d ht=%-5d KV=%.0f MB  %7.1f MB  %.1fs'
          % (n, B, S1, S2, N1, SBS, CN, SBS * min(nblk, n_blk_total), B * S1 * N1,
             B * S1 * N1, 2 * B * S2 * (D + 64) * 2 / 1e6,
             os.path.getsize(path) / 1e6, time.time() - t0))
