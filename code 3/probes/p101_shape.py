#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P101：decode 形的墙到底是谁 —— 两条正交斜率各打一枪。

P100 的复现案把墙钉成"每单元 token 数 × 141 ns，与头数、与索引密度都无关"。
代码里只有一个落点：`ProcessToken` 每登记**一段**就发 2 条 `CopyGm2Ub`（K、kr），
`FlushChunk` 里 V 再 1 条。而"相邻块号并成一段"这件事**根本没有**（`stageBeg_/
stageLen_` 只被 chunk 边界切开，从不因为 `idx[j+1]==idx[j]+1` 合并）⇒
每**块** 3 条碎拷贝；密度 0.5 与 0.125 同速正是因为两者块数相同（§4：50 ns/条、
与 `nValue` 无关）。于是两枪：

  A `sbs` 阶梯（`s1x*`）：块大小 1→128、**每行 token 数固定**。墙若是调用条数 ⇒
    时间按 ~1/sbs 掉；墙若是"每 token 一份向量功" ⇒ 平。这一枪判"条数 vs 计算"。
  B 相邻合并空间（`rr*` + `ct*`，全部 sbs=1、每行 65536 token，只差**连续率**）：
    `rr1p2` 段长 1（永不相邻）↔ `ctcontig` 一整段 ⇒ 这两发之差就是"实现合并"
    在平台形状上能吃的东西；`ct1/2/4/8` 是 50 % 密度下段长 1/2/4/8 的插值。

expect 全零 ⇒ 只读时间。
"""
import math
import os
import sys

import numpy as np

B, S2, D, MODE = 1, 131072, 512, 3
TOK_PER_ROW = 65536           # 每行扫多少 token（各案对齐，只差"怎么切成块 / 怎么选"）
MAGIC = b'SFA_CASE 2\n'
outdir = os.environ.get('SFA_CASES', os.path.expanduser('~/sfa_real/cases'))

# 名称, S1, N1, SBS, 取法
#   ('rand', p)     密度 p 的随机块号升序表
#   ('seg', g, p)   段长 g、周期 p 的块号表（g=1 永不相邻；g=p 即整段连续）
CASES = [
    ('s1x1',     4, 1,   1, ('rand', 0.5)),
    ('s1x2',     4, 1,   2, ('rand', 0.5)),
    ('s1x8',     4, 1,   8, ('rand', 0.5)),
    ('s1x32',    4, 1,  32, ('rand', 0.5)),
    ('s1x128',   4, 1, 128, ('rand', 0.5)),
    ('rr1p2',    4, 1,   1, ('seg', 1, 2)),
    ('rr2p4',    4, 1,   1, ('seg', 2, 4)),
    ('rr4p8',    4, 1,   1, ('seg', 4, 8)),
    ('rr8p16',   4, 1,   1, ('seg', 8, 16)),
    ('ctcontig', 4, 1,   1, ('seg', 65536, 65536)),
    # C 组：并行度**不饥饿**（32 行 ⇒ ks 门关）下扫 N1 阶梯 ⇒ 时间对 N1 平 = 每 token
    #    的墙（带宽/调用），随 N1 线性 = 每 head·token 的墙（向量功）。sbs=128 是为了
    #    先把调用条数摁到地板，只留"功"和"字节"两个候选。
    ('b32n1s128',  32, 1, 128, ('rand', 0.5)),
    ('b32n2s128',  32, 2, 128, ('rand', 0.5)),
    ('b32n4s128',  32, 4, 128, ('rand', 0.5)),
    ('b32n8s128',  32, 8, 128, ('rand', 0.5)),
    ('b32n8s1',    32, 8,   1, ('rand', 0.5)),
    ('b4n8s128',    4, 8, 128, ('rand', 0.5)),
]

want = sys.argv[1:] or [c[0] for c in CASES]
rng = np.random.default_rng(20260924)


def f16(n):
    return (rng.random(n, dtype=np.float32) * 2 - 1).astype('<f2')


key = f16(B * S2 * D)
value = f16(B * S2 * D)
krope = f16(B * S2 * 64)
print('K/V/kr = %.0f MB 复用' % ((key.nbytes + value.nbytes + krope.nbytes) / 1e6))


def idx_row(sbs, spec, count):
    """升序块号表（长度恒为 count，不足补 -1）；块号 < 总块数。"""
    nblk = (S2 + sbs - 1) // sbs
    kind = spec[0]
    if kind == 'rand':
        p = spec[1]
        take = max(1, min(nblk, int(round(nblk * p))))
        row = np.sort(rng.choice(nblk, size=take, replace=False)).astype('<i4')
    else:
        g, per = spec[1], spec[2]
        heads = np.arange(0, nblk, per, dtype='<i4')
        row = (heads[:, None] + np.arange(g, dtype='<i4')[None, :]).ravel()
        row = row[row < nblk]
    row = row[:count]
    if row.size < count:
        row = np.concatenate([row, np.full(count - row.size, -1, dtype='<i4')])
    return row.astype('<i4')


def write(path, s1, n1, sbs, spec):
    count = max(1, TOK_PER_ROW // sbs)
    row = idx_row(sbs, spec, count)
    q = f16(B * s1 * n1 * D)
    qr = f16(B * s1 * n1 * 64)
    idx = np.tile(row, (B, s1, 1, 1))          # (B, S1, KV_N=1, COUNT)：头共用同一份表
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
    real = int((row >= 0).sum())
    val = row[row >= 0]                        # ⚠️ 数"连续段"要用**块号值**的差分，不是数组下标
    runs = 1 + int((np.diff(val) != 1).sum()) if real else 0
    print('%-9s 行=%d N1=%d sbs=%-4d 条数=%-6d 有效块=%-6d tok/行=%-6d 连续段=%-6d %4.0f MB'
          % (os.path.basename(path)[:-4], s1, n1, sbs, count, real, real * sbs, runs,
             os.path.getsize(path) / 1e6))


os.makedirs(outdir, exist_ok=True)
for name, s1, n1, sbs, spec in [c for c in CASES if c[0] in want]:
    write(os.path.join(outdir, '%s.bin' % name), s1, n1, sbs, spec)
