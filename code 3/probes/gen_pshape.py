#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按比赛平台反推出的用例形状生成"少行多列"性能用例（code3.md §5.8.3）。

为什么需要它：§15 之前所有的性能账都记在 big1 上（S1=128 行 ⇒ 128 个单元摊到
40 个 AIV，每核 3.2 行，**机器是填满的**）。而 §5.8.3 由探针分数反推出的 6 个
真实评测点只有 **4 / 8 / 4 / 16 / 4 / 32 行**，每行 4 个头 —— 行维铺核的话只有
4~32 个核在干活。要衡量"少行"下的表现，必须有同形状的用例。

行数量级校核：平台 6 点耗时 57.22/33.7/71.24/251.98/109.2/251.5 ms，除掉行数
≈ 14 ms/行；而 big1 的标量版是 206.9 ms / 128 行 = 1.6 ms/行 @ nblk=256
⇒ 每行工作量约 9 倍 ⇒ 取 **nblk=2048**（= idx 数组长度 COUNT 的上限），
这样生成出来的形状与平台点数值同一量级，不是随手挑的。

用法（在真机 ~/sfa_real 下跑，和 gen_big.py 一样 import 远端 sfa_ref）：
    python3 gen_pshape.py            # 生成全部
    python3 gen_pshape.py p1 p6      # 只生成指定的
"""
import os
import sys
import time

sys.path.insert(0, os.environ.get('SFAREF', os.path.expanduser('~/sfa_real')))
import sfa_ref as S  # noqa: E402

# (名称, B, S1, S2, N1, D, SBS, MODE, COUNT, nblk) —— 对应 §5.8.3 反推的 C1..C6
SHAPES = {
    'p1': (1, 4,  8192, 4, 512, 1, 3, 2048, 2048),   # C1: 4 行 4 头
    'p2': (2, 4,  8192, 2, 512, 1, 3, 2048, 2048),   # C2: B=2 S1=4 N1=2 -> 8 行
    'p3': (1, 4,  8192, 4, 512, 1, 3, 2048, 2048),   # C3
    'p4': (1, 16, 8192, 4, 512, 1, 3, 2048, 2048),   # C4: 16 行
    'p5': (1, 4,  8192, 4, 512, 1, 3, 2048, 2048),   # C5
    'p6': (1, 32, 8192, 4, 512, 1, 3, 2048, 2048),   # C6: 32 行
    # 平台形状 + **表只填一半**（gen_case 的写法就是"前 nblk 项有效 + 尾部 -1"，
    # 且有效块号升序）—— P18 交错切分的**正确性**判据用例（带真 expect，不是哑零）：
    # 连续切法下分片 1 空转、交错切法下两半各拿 512 项，两者的输出都必须与参考一致。
    'q1h': (1, 4,  8192, 4, 512, 1, 3, 2048, 1024),
    'q2h': (1, 4,  8192, 4, 512, 2, 3, 2048, 1024),  # 同上，SBS=2（平台疑似档位）
    'q3h': (1, 8,  8192, 4, 512, 1, 3, 2048, 100),   # 极端：只有 100 项有效（V << count/2）
    # P18 交错切分的**边界**判据（同样带真 expect）。host 的三门只看 `sparse_count`（表长），
    # 不看"有多少项有效"，所以下面这些极小 V 的用例**照样走 ks=2**，正好把 MergeToken 里
    # "某一半为空"的两个分支（l1<=0 / l0<=0）逼出来：
    'e1empty': (1, 4, 8192, 4, 512, 1, 3, 2048, 0),      # 全 -1 ⇒ 两分片都空，LSE 必须 (0,0)
    'e2one':   (1, 4, 8192, 4, 512, 1, 3, 2048, 1),      # V=1 ⇒ 分片 0 拿 1 项、分片 1 空
    'e3two':   (1, 4, 8192, 4, 512, 1, 3, 2048, 2),      # V=2 ⇒ 各 1 项（奇偶各一个）
    'e4odd':   (1, 4, 8192, 4, 512, 1, 3, 2048, 3),      # V=3 ⇒ 分片 0 拿 2、分片 1 拿 1
    'e5s2one': (1, 4, 8192, 4, 512, 2, 3, 2048, 1),      # SBS=2 + V=1 ⇒ 一块 2 token 全在分片 0
    'e6many':  (1, 32, 8192, 4, 512, 1, 3, 2048, 1),     # rows=32（nb>1 档）+ V=1
}

# 变长场景：gen_case 的 actual_s1/actual_s2（per-batch 真实长度）。不传 = 用满，
# 而平台上最常见恰恰是带 padding 的形态（官方 BSND 语义下 threshold 也取决于真实长度）。
EXTRA = {
    'e7padq':  dict(actual_s1=[5]),    # S1=8 但只有 5 行真 ⇒ 3 个 padding 行走 MergeToken 的早退分支
    'e8padkv': dict(actual_s2=[1024]), # KV 短于 padded S2 ⇒ 块号上限被 threshold 夹住
}
SHAPES['e7padq'] = (1, 8, 8192, 4, 512, 1, 3, 2048, 64)
SHAPES['e8padkv'] = (1, 4, 8192, 4, 512, 1, 3, 2048, 64)

outdir = os.environ.get('SFA_CASES', os.path.expanduser('~/sfa_real/cases'))
names = sys.argv[1:] or list(SHAPES)
os.makedirs(outdir, exist_ok=True)
for n in names:
    B, S1, S2, N1, D, SBS, MODE, CN, nblk = SHAPES[n]
    t0 = time.time()
    case = S.gen_case(B, S1, S2, N1, D, SBS, MODE, CN, seed=hash(n) % 10000,
                      nblk=nblk, **EXTRA.get(n, {}))
    path = os.path.join(outdir, '%s.bin' % n)
    S.write_case(path, case)
    print('%-4s B=%d S1=%-3d S2=%d N1=%d nblk=%d  行×头=%-3d  %8.1f MB  %.1fs'
          % (n, B, S1, S2, N1, nblk, B * S1 * N1, os.path.getsize(path) / 1e6,
             time.time() - t0))
