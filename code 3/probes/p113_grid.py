#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P113b 靶族：**把"低算术强度"与"冷"这两条轴拆开**，再沿"复用度"扫一条阶梯。

为什么是这六个：P113a 读到 `r64c65`（N1=1、S2 131072）上两个惰性计量器变成 +24.0 / +22.6 %，
而 `w3/w4/big1/p6`（N1=4/8、S2 8192）是 +44~50 / +4~6 % ⇒ 平台比值 0.34 要的是"搬运吃掉墙"，
本地同时动了两件事（头数、冷热），**归不了因**。于是这一族每例只动一条：
  · `v*` = 池 17.8 MB（缓存兜得住）+ N1=1 ⇒ 只测"低算术强度"能不能单独把 gather 抬起来；
  · `g*` = 池 285 MB（装不进）+ 复用度 0.5×/2×/8× ⇒ 只测"冷"；
  · `n2` = 在冷档上把头数从 1 抬到 2 ⇒ 读"平台 q_N=1 还是 2"这个 P112 假设的**代价斜率**。
口径与 `p100_replica.py` 完全一致：只造输入、expect 全零 ⇒ **只读时间**（§7.3），
跨臂逐位比对用一次性目录 `golden_p113/`。形状参数由本文件写进 header，runner 再从 header 读回来。
"""
import math
import os
import sys

import numpy as np

MAGIC = b'SFA_CASE 2\n'
D, SBS, MODE, B = 512, 1, 3, 1
outdir = os.environ.get('SFA_CASES', os.path.expanduser('~/sfa_real/cases'))

# 名称, S2, S1, N1, COUNT
CASES = [
    ('v64c2',   8192,   64, 1, 2048),    # 暖池 + 低强度
    ('v64c8',   8192,   64, 1, 8192),    # 暖池 + 每行满表
    ('g64c1',   131072, 64, 1, 1024),    # 冷池，复用 0.5×
    ('g64c4',   131072, 64, 1, 4096),    # 冷池，复用 2×
    ('g64c16',  131072, 64, 1, 16384),   # 冷池，复用 8×
    ('g48c4n2', 131072, 48, 2, 4096),    # 冷池 + 两头
]
want = sys.argv[1:] or [c[0] for c in CASES]

_pools = {}


def pool(s2):
    """同一 S2 的 K/V/kr 造一次、多案共用（与 p100_replica 同思路，种子固定）。"""
    if s2 not in _pools:
        rng = np.random.default_rng(20260924 + s2)
        _pools[s2] = (rng.random(s2 * D, dtype=np.float32).astype('<f2'),
                      rng.random(s2 * D, dtype=np.float32).astype('<f2'),
                      rng.random(s2 * 64, dtype=np.float32).astype('<f2'),
                      rng)
    return _pools[s2]


def idx_rows(rng, s1, count, nblk):
    out = np.full((s1, count), -1, dtype='<i4')
    for s in range(s1):
        out[s] = (np.arange(nblk, dtype='<i4') if count >= nblk
                  else np.sort(rng.choice(nblk, size=count, replace=False)).astype('<i4'))
    return out.reshape(1, s1, 1, count)


def write(path, s2, s1, n1, count):
    key, value, krope, rng = pool(s2)
    q = rng.random(s1 * n1 * D, dtype=np.float32).astype('<f2')
    qr = rng.random(s1 * n1 * 64, dtype=np.float32).astype('<f2')
    idx = idx_rows(rng, s1, count, s2 // SBS)
    scale = 1.0 / math.sqrt(D)
    body = [MAGIC]
    for k, v in (('B', B), ('S1', s1), ('S2', s2), ('N1', n1), ('D', D),
                 ('SBS', SBS), ('COUNT', count), ('SCALE', repr(scale)),
                 ('MODE', MODE), ('LSE', 1)):
        body.append(('%s %s\n' % (k, v)).encode('ascii'))
    body += [q.tobytes(), key.tobytes(), value.tobytes(), qr.tobytes(), krope.tobytes(),
             idx.tobytes(),
             np.full(B, s1, dtype='<i4').tobytes(), np.full(B, s2, dtype='<i4').tobytes(),
             np.zeros(B * s1 * n1 * D, dtype='<f2').tobytes(),
             np.zeros(B * s1 * n1, dtype='<f4').tobytes(),
             np.zeros(B * s1 * n1, dtype='<f4').tobytes()]
    with open(path, 'wb') as f:
        f.write(b''.join(body))


os.makedirs(outdir, exist_ok=True)
for name, s2, s1, n1, count in [c for c in CASES if c[0] in want]:
    p = os.path.join(outdir, '%s.bin' % name)
    write(p, s2, s1, n1, count)
    req = B * s1 * ((n1 + 1) // 1) * count * (2 * D + 64) * 2.0
    kb_pool = B * s2 * (2 * D + 64) * 2.0
    print('%-9s S2=%-6d 行=%-3d N1=%d tok/行=%-6d 池=%5.1f MB 请求=%7.0f MB 复用=%4.1fx %6.0f B/案'
          % (name, s2, s1, n1, count, kb_pool / 1e6, req / 1e6, req / kb_pool, os.path.getsize(p) / 1e3))
