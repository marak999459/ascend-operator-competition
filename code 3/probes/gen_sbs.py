#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只造输入、不跑 python 参考的用例生成器（**纯计时用**，code3.md §15.35）。

为什么要有它：`sfa_ref.gen_case` 每次都要用纯 python 把整份 attention 算一遍当 expect，
p1 那种 16 单元 × 2048 token 的形状要跑十几分钟。而"换个 SBS / 换个列表长度看看时间"
这类实验只需要**输入**和正确的文件布局 —— 计时走 `act=none`，expect 填哑值即可。
判据不受影响：正确性闸门仍然全部走 `gen_pshape.py` 那批带真 expect 的用例。

输入部分与 `gen_case` 同种子同顺序（randn 流 + "块号抽样后排序、表尾 -1"），
所以以后要补真 expect，用同名参数跑 `gen_pshape.py` 就能对上。

用法（在真机 ~/sfa_real 下）：
    python3 gen_sbs.py                # 默认那一族
    python3 gen_sbs.py p1s2 p4s2      # 只造指定的
"""
import os
import sys
import time

sys.path.insert(0, os.environ.get('SFAREF', os.path.expanduser('~/sfa_real')))
import sfa_ref as S  # noqa: E402

# (名称, B, S1, S2, N1, D, SBS, MODE, COUNT, nblk) —— nblk = 表内有效块数
SHAPES = {
    # 假设 A：平台是 SBS=1、表满 2048（= 现有 p1..p6，§5.8.3 的反推）
    'p1s1': (1, 4,  8192, 4, 512, 1, 3, 2048, 2048),
    # 假设 B：平台是 SBS=2、表满 2048 ⇒ 每行 4096 token。
    #   依据：6/6 通过版（纯标量）在平台 C1 上是 57.22 ms，而本地"标量 1.6 ms/行
    #   @ 256 token/行"线性外推要 **2240 token/行**，SBS=1 时表长上限 2048 顶不住
    #   ⇒ 平台的 sparseBlockSize 极可能 ≥2（§15.35(a)）。
    'p1s2': (1, 4,  8192, 4, 512, 2, 3, 2048, 2048),
    'p2s2': (2, 4,  8192, 2, 512, 2, 3, 2048, 2048),
    'p4s2': (1, 16, 8192, 4, 512, 2, 3, 2048, 2048),
    'p6s2': (1, 32, 8192, 4, 512, 2, 3, 2048, 2048),
    # 另外两档 SBS，看"块内连续段变长"对 gather / 选档的影响有多大
    'p1s4': (1, 4,  8192, 4, 512, 4, 3, 2048, 2048),
    'p1s8': (1, 4,  8192, 4, 512, 8, 3, 1024, 1024),
    # SBS=1 但表只填一半：把"表长"与"token 数"这两个变量拆开
    'p1s1h': (1, 4,  8192, 4, 512, 1, 3, 2048, 1024),
    # 平台形状（SBS=2）+ 表只填一半 —— P18 的判据用例：P11v2 的连续切法在这里 1.9×→1.0×，
    # 奇偶交错切法应当把它拿回来（对照 p1s2 的 0.245 ms、p1s1 的 0.162 ms）
    'p1s2h': (1, 4,  8192, 4, 512, 2, 3, 2048, 1024),
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
    import math
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
    print('%-6s B=%d S1=%-3d S2=%d N1=%d SBS=%-3d COUNT=%-5d 有效块=%-5d '
          'token/行=%-6d 行×头=%-3d  %7.1f MB  %.1fs'
          % (n, B, S1, S2, N1, SBS, CN, nblk, SBS * min(nblk, n_blk_total),
             B * S1 * N1, os.path.getsize(path) / 1e6, time.time() - t0))
