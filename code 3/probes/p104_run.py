#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P104：量"索引表连续率 ↦ gather 拷贝条数"的**换算斜率**，再决定 run-merge 值不值。

墙（`LOG#15.81`）：decode 形成本 ≈ 每 token 141 ns，与头数、与索引密度都不敏感 ⇒ 成本单位
是"**块**"不是"token"，而每块固定 3 条碎 `CopyGm2Ub`（K、kr 在 `ProcessToken`，V 在
`FlushChunk`）× ≈50 ns/条 = **54 ns/token 的税**（`sbs=1` 时 1 块 = 1 token）。
可"相邻块并成一条"这件事代码里从来没做过（段登记表只被 chunk 边界切开）。

⚠️ P101 那批 `rr*`/`ctcontig` 用例**读不出这件事**：kernel 一块一段，没有实现时两者同速，
平读只证明"成本 ∝ 块数"（那次"判死"就是这么错的）。要读斜率必须**先有实现**，
所以这一族的正确用法是：**改前跑一遍 = 基线，改后再跑一遍 = 上下界**。

口径：钉死 `B=1, S1=32, N1=4, S2=131072, sbs=1, COUNT=4096` ⇒ 每行 4096 token、
32 行 = 128 个核上单元（**多波**，避开并行度饥饿这个混淆项），只差**游程长度** `L`
⇒ 每行段数 = 4096/L：
  pm1     L=1     4096 段   ← 今天的 P38 行为（一条都并不了）= 下界
  pm2/4/8         2048/1024/512 段
  pm64              64 段
  pm1024             4 段
  pmfull           1 段     ← 满连续 = 上界
判读：`t(pm1)/t(pmfull)` ≈ 1 ⇒ 聚合无墙，P104 判死；≈ 2~3 ⇒ 每 token 省一半以上税。
expect 全零 ⇒ 只读时间不读判据。
"""
import math
import os

import numpy as np

B, S2, D, MODE, N1 = 1, 16384, 512, 3, 4
S1, SBS, COUNT = int(os.environ.get('ROWS', '32')), 1, 4096
MAGIC = b'SFA_CASE 2\n'
outdir = os.environ.get('SFA_CASES', os.path.expanduser('~/sfa_real/cases'))

# 名称 -> 游程长度 L（None = 一整段）
_PREF = 'pm' if S1 == 32 else 'pm%d' % S1        # rows=32 延用第一批的名字，别的行数控件不覆盖
CASES = [(f'{_PREF}1', 1), (f'{_PREF}2', 2), (f'{_PREF}4', 4), (f'{_PREF}8', 8),
         (f'{_PREF}64', 64), (f'{_PREF}256', 256), (f'{_PREF}1024', 1024),
         (f'{_PREF}full', None)]
# P105 只要阶梯的两端（每个行数控件 ~36 MB，别把 cases/ 塞满）：ONLY=1,full
_only = os.environ.get('ONLY', '').strip()
if _only:
    keep = set(_only.split(','))
    CASES = [c for c in CASES if c[0][len(_PREF):] in keep]

rng = np.random.default_rng(20260924)


def f16(n):
    return (rng.random(n, dtype=np.float32) * 2 - 1).astype('<f2')


key = f16(B * S2 * D)
value = f16(B * S2 * D)
krope = f16(B * S2 * 64)
print('K/V/kr = %.0f MB 复用' % ((key.nbytes + value.nbytes + krope.nbytes) / 1e6))


def row_of(runlen):
    """升序块号表，恰好 COUNT 个有效块；游程长度 runlen（None ⇒ 一整段）。"""
    if runlen is None:
        return np.arange(COUNT, dtype='<i4')
    nrun = COUNT // runlen
    base = np.arange(nrun, dtype='<i4') * (runlen + 1)      # run 之间留 1 个缺口
    return (base[:, None] + np.arange(runlen, dtype='<i4')[None, :]).ravel()


def write(path, runlen):
    row = row_of(runlen)[:COUNT]
    assert row.size == COUNT and int(row.max()) < S2
    idx = np.tile(row, (B, S1, 1, 1))          # (B, S1, KV_N=1, COUNT)：各行共用同一份表
    q = f16(B * S1 * N1 * D)
    qr = f16(B * S1 * N1 * 64)
    scale = 1.0 / math.sqrt(D)

    def hdr(k, v):
        return ('%s %s\n' % (k, v)).encode('ascii')

    body = [MAGIC]
    for k, v in (('B', B), ('S1', S1), ('S2', S2), ('N1', N1), ('D', D),
                 ('SBS', SBS), ('COUNT', COUNT), ('SCALE', repr(scale)),
                 ('MODE', MODE), ('LSE', 1)):
        body.append(hdr(k, v))
    body += [q.tobytes(), key.tobytes(), value.tobytes(), qr.tobytes(), krope.tobytes(),
             idx.tobytes(),
             np.full(B, S1, dtype='<i4').tobytes(), np.full(B, S2, dtype='<i4').tobytes(),
             np.zeros(B * S1 * N1 * D, dtype='<f2').tobytes(),
             np.zeros(B * S1 * N1, dtype='<f4').tobytes(),
             np.zeros(B * S1 * N1, dtype='<f4').tobytes()]
    with open(path, 'wb') as f:
        f.write(b''.join(body))
    runs = 1 + int((np.diff(row) != 1).sum())
    print('%-9s L=%-5s 段/行=%-5d tok/行=%d 单元=%d ht=%.2e %4.0f MB'
          % (os.path.basename(path)[:-4], runlen or 'full', runs, COUNT, S1 * N1,
             S1 * N1 * COUNT, os.path.getsize(path) / 1e6))


os.makedirs(outdir, exist_ok=True)
import sys                                    # noqa: E402
want = sys.argv[1:] or [c[0] for c in CASES]
for name, runlen in [c for c in CASES if c[0] in want]:
    write(os.path.join(outdir, '%s.bin' % name), runlen)
