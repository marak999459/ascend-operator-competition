#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P92：把 CoreMemType 四块的**真实容量**打出来（只动远端副本，提交源里永远没有 fprintf）。
   动因：tiling.h:39-40 那句"L0C 只有 64 列 ⇒ Mmad n 最大 64"从没量过，而官方 arch22 SFA-MLA
   按 m=n=128 发 Mmad（=每乒乓槽 64 KB）⇒ 我们 cube/M2 两条线的额度全押在这句未测的话上。"""
import io
import sys

P = '/home/developer/sfa_real/code/op_host/sparse_flash_attention.cpp'
s = io.open(P, encoding='utf-8').read()
if '[MEM]' in s:
    sys.stderr.write('already patched\n')
    sys.exit(0)
anchor = '    const uint64_t ubSafe = (ubSize / 100ULL) * UB_SAFE_PCT;'
assert s.count(anchor) == 1, 'anchor miss %d' % s.count(anchor)
probe = ('    {\n'
         '        uint64_t q2 = 0;\n'
         '        const uint64_t ubQ = ubSize;\n'
         '        (void)platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, q2);\n'
         '        const uint64_t l1Q = q2; q2 = 0;\n'
         '        (void)platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, q2);\n'
         '        const uint64_t l0aQ = q2; q2 = 0;\n'
         '        (void)platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, q2);\n'
         '        const uint64_t l0bQ = q2; q2 = 0;\n'
         '        (void)platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, q2);\n'
         '        fprintf(stderr, "[MEM] UB=%llu L1=%llu L0A=%llu L0B=%llu L0C=%llu (bytes)\\n",\n'
         '                (unsigned long long)ubQ, (unsigned long long)l1Q, (unsigned long long)l0aQ,\n'
         '                (unsigned long long)l0bQ, (unsigned long long)q2);\n'
         '    }\n')
s = s.replace(anchor, probe + anchor, 1)
inc = '#include <cmath>'
assert s.count(inc) == 1, 'include anchor miss'
s = s.replace(inc, inc + '\n#include <cstdio>', 1)
io.open(P, 'w', encoding='utf-8').write(s)
sys.stderr.write('patched, [MEM] count=%d\n' % s.count('[MEM]'))
