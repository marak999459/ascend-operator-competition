#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成大 shape 用例（真机性能测试用）"""
import sys, time
sys.path.insert(0, '/home/developer/sfa_real')
import sfa_ref as S

# 大 shape：贴近题面真实的"长序列推理"场景
#   S2=8192 长 KV；sparse 覆盖 256 个 token；N1=8 头；S1=128 个 query token
CFG = dict(B=1, S1=128, S2=8192, N1=8, D=512, SBS=1, MODE=3, COUNT=2048)

name = sys.argv[1] if len(sys.argv) > 1 else 'big1'
nblk = int(sys.argv[2]) if len(sys.argv) > 2 else 256

t0 = time.time()
c = S.gen_case(CFG['B'], CFG['S1'], CFG['S2'], CFG['N1'], CFG['D'],
               CFG['SBS'], CFG['MODE'], CFG['COUNT'],
               seed=42, nblk=nblk)
p = '/home/developer/sfa_real/cases/%s.bin' % name
S.write_case(p, c)
print('生成 %s: B=%d S1=%d S2=%d N1=%d SBS=%d MODE=%d nblk=%d  用时 %.1fs'
      % (name, CFG['B'], CFG['S1'], CFG['S2'], CFG['N1'],
         CFG['SBS'], CFG['MODE'], nblk, time.time() - t0))
