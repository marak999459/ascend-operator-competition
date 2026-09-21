#!/usr/bin/env python3
# P17 前置差分扫描的补丁生成器：**只作用在远端副本** ~/sfa_real/code/op_host/…
# 加两个环境变量旋钮，一次构建就能扫 (nb, kv_shard) 组合：
#   SFA_FORCE_NB   —— 把 CalcBlocking 的候选表夹成单值
#   SFA_FORCE_KS   —— 覆盖 CalcBlocking 选出的 kvShard（1/2），用于量"切/不切"的纯倍率
# 本地提交源绝不落这个补丁（每次 npu.sh sync 会覆盖远端副本）。
#
# P16 之后 ks 的锚点变了：原来夹在 `units = units0 * kvShard` 之前，现在 kvShard 由
# CalcBlocking 直接出参，所以改挂在【调用方】—— CalcBlocking/降级块之后、SetBlockDim
# 那个作用域之前。这个位置覆盖的好处是 blockDim 会跟着重算，与 kernel 的
# `coreNum == 2*total0` 门自洽（超过核心数时 kernel 自己退回 ks=1）。
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "code/op_host/sparse_flash_attention.cpp"
src = open(PATH, "r", encoding="utf-8").read()
if "SFA_FORCE_NB" in src:
    print("ALREADY PATCHED", PATH)
    sys.exit(0)

# 1) <cstdlib>
if "#include <cstdlib>" not in src:
    src = src.replace("#include <cmath>", "#include <cmath>\n#include <cstdlib>", 1)

# 2) 夹 nb
anchor = "        const uint32_t nb = NB_CAND[i];\n"
assert src.count(anchor) == 1, "nb anchor"
src = src.replace(anchor, anchor +
    "        { const char *fnb = getenv(\"SFA_FORCE_NB\");\n"
    "          if (fnb != nullptr && fnb[0] != '\\0' &&\n"
    "              static_cast<uint32_t>(atoi(fnb)) != nb) { continue; } }\n", 1)

# 3) 覆盖 kvShard（P16 结构：挂在调用方）
anchor2 = "        kvShard = 1U;  // 降级路径不赌并行度，退回与参考实现逐位一致的那条路\n    }\n"
assert src.count(anchor2) == 1, "ks anchor"
src = src.replace(anchor2, anchor2 +
    "    { const char *fks = getenv(\"SFA_FORCE_KS\");\n"
    "      if (fks != nullptr && fks[0] != '\\0') { kvShard = static_cast<uint32_t>(atoi(fks)); } }\n", 1)

# 4) 一次性打印模型选出的档（只在第一次 Compute 时打，避免连发 20 次刷屏）
if "#include <cstdio>" not in src:
    src = src.replace("#include <cstdlib>", "#include <cstdlib>\n#include <cstdio>", 1)
anchor3 = "    SparseFlashAttentionTilingData *tiling = context->GetTilingData"
assert src.count(anchor3) == 1, "print anchor"
src = src.replace(anchor3,
    "    { static bool sfa_probe_shown = false;\n"
    "      if (!sfa_probe_shown) { sfa_probe_shown = true;\n"
    "        std::printf(\"SFA_PICK nb=%u nblk=%u ks=%u sbs=%lld rows=%lld count=%llu\\n\",\n"
    "                    nb, nBlk, kvShard, (long long)sbs, (long long)(B * Q_S),\n"
    "                    (unsigned long long)sparse_count); } }\n" + anchor3, 1)

open(PATH, "w", encoding="utf-8").write(src)
print("PATCHED", PATH)
