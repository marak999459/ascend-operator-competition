#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P127 闸门补用例：**头数 ≥16 的形状**，因为这一发的钉档会把 `nb` 抬到 16、把 `n_blk` 压到 32，
而现有 26 例里最大 `Q_N` 只有 8 ⇒ 那两个组合在本地从没跑过（"没跑过"不能当成"是对的"）。

形状照抄 `gen_pshape.py` 的元组口径 `(B,S1,S2,N1,D,SBS,MODE,COUNT,nblk)`；
期望选档由 `p127_model.py` 的镜像算出，写在 `p127_gate.sh` 的命中数门里（fp16/fp32 两臂都要过）。
用法（在真机 ~/sfa_real 下）：python3 gen_n16.py
"""
import os
import sys
import time

sys.path.insert(0, os.environ.get('SFAREF', os.path.expanduser('~/sfa_real')))
import sfa_ref as S  # noqa: E402

# name: (B,S1,S2,N1,D,SBS,MODE,COUNT,nblk)  + 注释里给"auto -> pin"的镜像预测
SHAPES = {
    # rows=4、qN=16 ⇒ nbCap=16：auto[nb=8 k=40 ks=5] -> pin[nb=16 k=32 ks=10]（fp32: k 32->16）
    'pn16a': (1, 4, 8192, 16, 512, 1, 3, 2048, 2048),
    # rows=8、qN=32 ⇒ pin[nb=16 k=32 ks=2]，顺带逼出"nb=16 且头块数 >1"
    'pn32a': (1, 8, 8192, 32, 512, 1, 3, 2048, 2048),
}

outdir = os.environ.get('SFA_CASES', os.path.expanduser('~/sfa_real/cases'))
names = sys.argv[1:] or list(SHAPES)
os.makedirs(outdir, exist_ok=True)
for n in names:
    B, S1, S2, N1, D, SBS, MODE, CN, nblk = SHAPES[n]
    t0 = time.time()
    case = S.gen_case(B, S1, S2, N1, D, SBS, MODE, CN, seed=hash(n) % 10000, nblk=nblk)
    path = os.path.join(outdir, '%s.bin' % n)
    S.write_case(path, case)
    print('%-6s B=%d S1=%-3d S2=%d N1=%-2d nblk=%d  rows=%-3d  %7.1f MB  %.1fs'
          % (n, B, S1, S2, N1, nblk, B * S1, os.path.getsize(path) / 1e6, time.time() - t0))
