#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P103：**平台六点形状复现族** —— 把"本地 157~815 µs vs 平台 7.68~14.42 ms"的 48× 旧账
钉成一个可跑的靶。

已知（`LOG#5.8.3` + `LOG#15.78(g)`）：六点 `rows = 4/8/4/16/4/32`、**`N1 = 4`**（C2 是 2）、
dtype = fp16。未知：每行 token 数与 `sbs`（`LOG#15.49(b)` 猜过 `SBS=2`，但那是拿
"每行 2048 token"的本地标量版反推的，而 2048 token/行 在本机连 1 ms 都不到 ⇒ 六点的
**绝对时间**本身就排除了"2048 token/行"）。

这一族把三个自由度摊开各打一枪，全部 `B=1, N1=4, MODE=3, S2=131072`：
  · `rows` ∈ {4, 16, 32}（对应 C1/C4/C6）
  · `tok/行` ∈ {2048, 16384, 65536}（= count × sbs）
  · `sbs` ∈ {128, 1}（同 tok 下只差切成多少段 gather）
判据就一条：**哪一格落在平台那一列的 ms 数上，那一格就是靶形状**；配上 A′(cube) 与
P38(AIV) 两侧同场次读数，"cube 在平台形状上到底赚不赚"这一发就不用再靠猜。
expect 全零 ⇒ 只读时间。
"""
import math
import os
import sys

import numpy as np

B, S2, D, MODE, N1 = 1, 131072, 512, 3, 4
MAGIC = b'SFA_CASE 2\n'
outdir = os.environ.get('SFA_CASES', os.path.expanduser('~/sfa_real/cases'))

# 名称, S1, N1, SBS, COUNT
CASES = [
    ('p4s128c16',    4,  4, 128, 16),      # tok/行 = 2048（= 本地语料口径）
    ('p4s128c128',   4,  4, 128, 128),     # 16384
    ('p4s128c512',   4,  4, 128, 512),     # 65536
    ('p16s128c128',  16, 4, 128, 128),
    ('p32s128c128',  32, 4, 128, 128),
    ('p4s1c2048',    4,  4,   1, 2048),    # 同 tok 数、碎成 2048 段
    ('p4s1c16384',   4,  4,   1, 16384),
    ('p4n1s128c512', 4,  1, 128, 512),     # 头数摊销对照（同 tok）
    # 第二批：补 `sbs=1` × 多行 这一格 —— 第一格的读数把 cube 的胜负符号钉成
    # "sbs 大才赚"（p16s128c128 cube 快 2.4×）与"sbs=1 亏 1.5 倍"（p4s1c16384），
    # 而平台六点若真是"sbs 小 + tok 大"，cube 在 16/32 行上到底赚不赚就是空白。
    ('p16s1c16384', 16, 4,   1, 16384),    # tok/行 = 16384，sbs=1（碎 16384 段）
    ('p32s1c16384', 32, 4,   1, 16384),
    ('p16s1c4096',  16, 4,   1, 4096),
    ('p32s128c512', 32, 4, 128, 512),      # tok/行 = 65536，32 行（cube 门必开）
]

want = sys.argv[1:] or [c[0] for c in CASES]
rng = np.random.default_rng(20260924)


def f16(n):
    return (rng.random(n, dtype=np.float32) * 2 - 1).astype('<f2')


key = f16(B * S2 * D)
value = f16(B * S2 * D)
krope = f16(B * S2 * 64)
print('K/V/kr = %.0f MB 复用' % ((key.nbytes + value.nbytes + krope.nbytes) / 1e6))


def write(path, s1, n1, sbs, count):
    nblk = (S2 + sbs - 1) // sbs
    take = min(count, nblk)
    row = np.sort(rng.choice(nblk, size=take, replace=False)).astype('<i4')
    if take < count:
        row = np.concatenate([row, np.full(count - take, -1, dtype='<i4')])
    idx = np.tile(row, (B, s1, 1, 1))       # (B, S1, KV_N=1, COUNT)
    q = f16(B * s1 * n1 * D)
    qr = f16(B * s1 * n1 * 64)
    scale = 1.0 / math.sqrt(D)

    def hdr(k, v):
        return ('%s %s\n' % (k, v)).encode('ascii')

    body = [MAGIC]
    for k, v in (('B', B), ('S1', s1), ('S2', S2), ('N1', n1), ('D', D),
                 ('SBS', sbs), ('COUNT', count), ('SCALE', repr(scale)),
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
    runs = 1 + int((np.diff(row[row >= 0]) != 1).sum())
    print('%-13s 行=%-3d N1=%d sbs=%-4d count=%-6d tok/行=%-7d 连续段=%-5d ht=%.2e %4.0f MB'
          % (os.path.basename(path)[:-4], s1, n1, sbs, count, take * sbs, runs,
             s1 * n1 * take * sbs, os.path.getsize(path) / 1e6))


os.makedirs(outdir, exist_ok=True)
for name, s1, n1, sbs, count in [c for c in CASES if c[0] in want]:
    write(os.path.join(outdir, '%s.bin' % name), s1, n1, sbs, count)
