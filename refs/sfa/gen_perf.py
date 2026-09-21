#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""性能用例生成器：shape 固定，只扫"每行有效 token 数"(nblk) 与 S2。

   gen_big.py 的 CFG 把 nblk 钉在 256 —— 于是 big1 实际每行只 gather 256 个 token
   （idx 里剩下的 2048-256 个位置全是 -1，扫描到第一个 -1 就停）。
   本脚本把 nblk 提成参数，用来测"耗时 vs 每行 token 数"的斜率：
   斜率 = 每个 (head,token) 的真实单价，截距 = 与 token 数无关的固定开销。

   用法: gen_perf.py <name> <nblk> [S2]
"""
import sys
import time

sys.path.insert(0, '/home/developer/sfa_real')
import sfa_ref as S  # noqa: E402

name = sys.argv[1]
nblk = int(sys.argv[2])
s2 = int(sys.argv[3]) if len(sys.argv) > 3 else 8192

t0 = time.time()
c = S.gen_case(1, 128, s2, 8, 512, 1, 3, 2048, seed=42, nblk=nblk)
S.write_case('/home/developer/sfa_real/cases/%s.bin' % name, c)
print('%s nblk=%d S2=%d -> 每行 %d token, 128 行 x 8 头 = %d 个点积  用时 %.1fs'
      % (name, nblk, s2, nblk, 128 * 8 * nblk, time.time() - t0))
