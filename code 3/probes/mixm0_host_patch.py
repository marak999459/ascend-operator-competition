#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[MIXM0] 只改远端副本的 op_host：MIX 的 blockDim 口径 + 运行期旋钮。

三个旋钮（全在 getenv，一个字节都不进提交源）：
  SFA_FORCE_KS  覆盖 CalcBlocking 选出的 kv_shard（1/2）⇒ 同一次构建里量"切/不切"
  SFA_BD        显式指定块数（覆盖下面的换算）
  SFA_MIXBD=1   把 AIV 口径的 blockDim 折成 **组数** `ceil(AIV块/2)`
               ⇒ MIX 下一次 launch 起 `2·BD` 个 AIV 块，与 AIV-only 的并行度对齐
另外无条件打一行 SFA_PICK（nb/nblk/ks/生效 blockDim）——§15.27 的纪律：
"形态对比必须同一屏打印生效的 SetBlockDim"，否则又拿旧块数读新档。

用法（远端 ~/sfa_real 下）: python3 /tmp/mixm0_host_patch.py
"""
import io
import sys

P = 'code/op_host/sparse_flash_attention.cpp'
t = io.open(P, encoding='utf-8').read()

if 'MIXM0' in t:
    print('host already patched: [MIXM0]')
    sys.exit(0)

BR0 = t.count('{') - t.count('}')   # 花括号必须平（§15.24：少一个 } 会让构建静默沿用旧 .so）

A_INC = '#include <cstdint>'
assert t.count(A_INC) == 1, 'include anchor miss: %d' % t.count(A_INC)
t = t.replace(A_INC, A_INC + '\n#include <cstdlib>   // [MIXM0] getenv\n#include <cstdio>    // [MIXM0] 打印生效档', 1)

# ---- 1) kv_shard 运行期覆盖（挂在 CalcBlocking / 降级块之后，tiling 赋值之前）----
A_KS = '        kvShard = 1U;  // 降级路径不赌并行度，退回与参考实现逐位一致的那条路\n    }\n'
assert t.count(A_KS) == 1, 'ks anchor miss'
t = t.replace(A_KS, A_KS + (
    '    { const char *fks = getenv("SFA_FORCE_KS");   // [MIXM0]\n'
    '      if (fks != nullptr && fks[0] != \'\\0\') { kvShard = static_cast<uint32_t>(atoi(fks)); } }\n'), 1)

# ---- 2) blockDim：可选折成组数 + 显式覆盖 + 打印 ----
A_BD = '        context->SetBlockDim(blockDim);'
assert t.count(A_BD) == 1, 'blockdim anchor miss'
t = t.replace(A_BD, (
    '        {   // [MIXM0] MIX 的 SetBlockDim 是【组数】口径：1 组 = 1 AIC + 2 AIV\n'
    '            const uint32_t aivBlk = blockDim;\n'
    '            const char *mb = getenv("SFA_MIXBD");\n'
    '            if (mb != nullptr && mb[0] == \'1\') {\n'
    '                blockDim = (aivBlk + 1u) / 2u;\n'
    '                if (blockDim == 0u) { blockDim = 1u; }\n'
    '            }\n'
    '            const char *bd = getenv("SFA_BD");\n'
    '            if (bd != nullptr && bd[0] != \'\\0\') { blockDim = static_cast<uint32_t>(atoi(bd)); }\n'
    '            static bool shown = false;\n'
    '            if (!shown) { shown = true;\n'
    '                std::printf("SFA_PICK nb=%u nblk=%u ks=%u sbs=%lld rows=%lld aivBlk=%u BD=%u\\n",\n'
    '                            nb, nBlk, kvShard, (long long)sbs,\n'
    '                            (long long)(B * Q_S), aivBlk, blockDim); }\n'
    '        }\n' + A_BD), 1)

BR1 = t.count('{') - t.count('}')
if BR1 != BR0:
    raise SystemExit('[MIXM0] 花括号不平：%d -> %d，拒绝落盘' % (BR0, BR1))
io.open(P, 'w', encoding='utf-8').write(t)
print('host patched: [MIXM0] (braces %+d->%+d getenv=%d)' %
      (BR0, BR1, t.count('getenv("SFA_')))
