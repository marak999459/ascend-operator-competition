#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[MIXGEO] 只改远端副本的 op_host：让块数读环境变量 SFA_BD ⇒ **一次构建扫多档 BD**。

动机：§15.22 只在 BD=40 一档量了 MIX 的 5~10×，分不清"可并发 AIV 块变少"还是"组内串行化"。
每档重编一次要 4 分钟，把 BD 做成运行期参数后一遍构建能扫 5~6 档。
⚠️ 这个补丁**绝不进提交源**（`code 3/code/` 只被读，写的是远端 ~/sfa_real/code/）。
"""
import io
import sys

p = 'code/op_host/sparse_flash_attention.cpp'
t = io.open(p, encoding='utf-8').read()

if 'MIXGEO' in t:
    print('host already patched: [MIXGEO]')
    sys.exit(0)

BR0 = t.count('{') - t.count('}')   # ⚠️ 花括号必须平：§15.24 的教训是"少一个 } 会让 g++ 失败，
                                     #    而构建脚本静默沿用旧 .so ⇒ 整批读数是假的"

A_INC = '#include <cstdint>'
assert t.count(A_INC) == 1, 'include anchor miss: %d' % t.count(A_INC)
t = t.replace(A_INC, A_INC + '\n#include <cstdlib>   // [MIXGEO] getenv/atoi 扫块数用', 1)

A_BD = '        context->SetBlockDim(blockDim);'
assert t.count(A_BD) == 1, 'blockdim anchor miss: %d' % t.count(A_BD)
t = t.replace(A_BD, (
    '        {   // [MIXGEO] 运行期改块数，免每档重编\n'
    '            const char *e = getenv("SFA_BD");\n'
    '            if (e != nullptr) { int v = atoi(e); if (v > 0) { blockDim = static_cast<uint32_t>(v); } }\n'
    '        }\n'
    + A_BD), 1)

BR1 = t.count('{') - t.count('}')
if BR1 != BR0:
    raise SystemExit('[MIXGEO] 花括号不平：%d -> %d，拒绝落盘' % (BR0, BR1))
io.open(p, 'w', encoding='utf-8').write(t)
print('host patched: [MIXGEO] (braces %+d, getenv=%d SetBlockDim=%d)' %
      (BR1, t.count('getenv("SFA_BD")'), t.count('context->SetBlockDim(blockDim);')))
