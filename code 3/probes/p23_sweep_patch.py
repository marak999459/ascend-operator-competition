#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P23 精度回归分诊的补丁生成器：**只作用在远端副本** ~/sfa_real/code/op_host/…

为什么要有它：P23 在 24 个本地用例（fp16+fp32）全 0 超差，却在比赛平台 3/6 用例上
precision_ratio 掉到 0.03~0.18 ⇒ 差异只可能来自【比赛平台挑中的档位组合】
(ubSize / coreNum 与本地不同 ⇒ CalcBlocking 出别的 nb/n_blk/ks)。
本地默认只测"模型选中的那一档"，所以要把整张 (nb, n_blk, ks) 表扫一遍才知道
哪一格是坏的。三个环境变量旋钮：
  SFA_FORCE_NB / SFA_FORCE_NBLK / SFA_FORCE_KS
外加一次性打印：nb / n_blk / ks / sbs / rows / count / need / ubSafe / ubSize / coreNum
（need > ubSafe 的组合是"本地扫到了但比赛平台不可能扫到"的越界格，判读时要单独看）。

本地提交源绝不落这个补丁（npu.sh sync 会用干净态覆盖远端副本）。
"""
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "code/op_host/sparse_flash_attention.cpp"
src = open(PATH, "r", encoding="utf-8").read()
if "SFA_FORCE_NB" in src:
    print("ALREADY PATCHED", PATH)
    sys.exit(0)

if "#include <cstdlib>" not in src:
    src = src.replace("#include <cmath>", "#include <cmath>\n#include <cstdlib>", 1)
if "#include <cstdio>" not in src:
    src = src.replace("#include <cstdlib>", "#include <cstdlib>\n#include <cstdio>", 1)

# 1) 打印平台真实 UB / 核数（拿到"比赛平台 vs 本地"的第一个可比量）
a0 = "    const uint64_t ubSafe = (ubSize / 100ULL) * UB_SAFE_PCT;"
assert src.count(a0) == 1, "ubSafe anchor"
src = src.replace(a0, a0 + "\n    { static bool sfa_env_shown = false;\n"
                  "      if (!sfa_env_shown) { sfa_env_shown = true;\n"
                  "        std::printf(\"SFA_ENV ub=%llu cores=%u\\n\",\n"
                  "                    (unsigned long long)ubSize, num_cores_aiv); } }", 1)

# 2) 夹 nb（FORCE 时连 UB 门一起绕过，越界格也要能被扫到）
anchor = "        const uint32_t nb = NB_CAND[i];\n"
assert src.count(anchor) == 1, "nb anchor"
src = src.replace(anchor, anchor +
    "        { const char *fnb = getenv(\"SFA_FORCE_NB\");\n"
    "          if (fnb != nullptr && fnb[0] != '\\0' &&\n"
    "              static_cast<uint32_t>(atoi(fnb)) != nb) { continue; } }\n", 1)

anchor_b = "                const uint32_t nBlk = NBLK_CAND[j];\n"
assert src.count(anchor_b) == 1, "nblk anchor"
src = src.replace(anchor_b, anchor_b +
    "                { const char *fb = getenv(\"SFA_FORCE_NBLK\");\n"
    "                  if (fb != nullptr && fb[0] != '\\0' &&\n"
    "                      static_cast<uint32_t>(atoi(fb)) != nBlk) { continue; } }\n", 1)

# 3) FORCE 时绕过 stage/UB/prefill 三道门（只为扫格，判读时看 need 列）
anchor_u = "                if (CalcUbNeed(nb, nBlk, qD, dr, elemSize) > ubSafe) { continue; }"
assert src.count(anchor_u) == 1, "ub anchor"
src = src.replace(anchor_u,
    "                { const char *ff = getenv(\"SFA_FORCE_NBLK\");\n"
    "                  const bool forced = (ff != nullptr && ff[0] != '\\0');\n"
    "                  if (!forced && CalcUbNeed(nb, nBlk, qD, dr, elemSize) > ubSafe) { continue; } }", 1)

anchor_s = "                if (nBlk > SFA_STAGE_MAX) { continue; }"
assert src.count(anchor_s) == 1, "stage anchor"
src = src.replace(anchor_s,
    "                { const char *ff = getenv(\"SFA_FORCE_NBLK\");\n"
    "                  if (!(ff != nullptr && ff[0] != '\\0') && nBlk > SFA_STAGE_MAX) { continue; } }", 1)

# 4) 覆盖 ks
anchor2 = "        kvShard = 1U;  // 降级路径不赌并行度，退回与参考实现逐位一致的那条路\n    }\n"
assert src.count(anchor2) == 1, "ks anchor"
src = src.replace(anchor2, anchor2 +
    "    { const char *fks = getenv(\"SFA_FORCE_KS\");\n"
    "      if (fks != nullptr && fks[0] != '\\0') { kvShard = static_cast<uint32_t>(atoi(fks)); } }\n", 1)

# 5) 一次性打印选档
anchor3 = "    SparseFlashAttentionTilingData *tiling = context->GetTilingData"
assert src.count(anchor3) == 1, "print anchor"
src = src.replace(anchor3,
    "    { static bool sfa_pick_shown = false;\n"
    "      if (!sfa_pick_shown) { sfa_pick_shown = true;\n"
    "        std::printf(\"SFA_PICK nb=%u nblk=%u ks=%u sbs=%lld rows=%lld count=%llu need=%llu safe=%llu e=%u\\n\",\n"
    "                    nb, nBlk, kvShard, (long long)sbs, (long long)(B * Q_S),\n"
    "                    (unsigned long long)sparse_count,\n"
    "                    (unsigned long long)CalcUbNeed(nb, nBlk, static_cast<uint32_t>(Q_D),\n"
    "                        static_cast<uint32_t>(Dr), elemSize),\n"
    "                    (unsigned long long)ubSafe, elemSize); } }\n" + anchor3, 1)

open(PATH, "w", encoding="utf-8").write(src)
print("PATCHED", PATH)
