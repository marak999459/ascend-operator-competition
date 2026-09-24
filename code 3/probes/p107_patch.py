#!/usr/bin/env python3
# P107 平台形状探针 —— **只作用在远端副本**（本地提交源不落这个补丁）。
#
# 要回答的问题（P106 的离线判别器给出的）：host 侧 P38→P105 的全部差异都过不了
# `units0*ks <= coreNum` 这道门 ⇒ 只在 B*Q_S <= 20 的形状上才存在。平台六点零响应，
# 于是要么六点全在 rows >= 21，要么 ks 这条路在平台上根本不运行（lseOn_ 拿不到 /
# blockDim 口径不同）。两者用"时间"这一条通道分不开，但用一个**与 ks/lseOn_ 无关**的
# 扰动就能分开：只按 rows 夹 n_blk。
#
#   SFA_PROBE_ROWS / SFA_PROBE_K —— rows<=ROWS 时把 n_blk 强制成 K（K>=16 且 8 的倍数）
# n_blk=16 正是 host 降级路径用的那一档 ⇒ 正确性天然成立，动的只有 chunk 宽度。
# 生效凭据 = nb_sweep_patch.py 那条 SFA_PICK 里的 nblk 字段。
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "code/op_host/sparse_flash_attention.cpp"
src = open(PATH, "r", encoding="utf-8").read()
if "SFA_PROBE_ROWS" in src:
    print("ALREADY PATCHED", PATH)
    sys.exit(0)
assert "SFA_PICK" in src, "先打 nb_sweep_patch.py（探针要用同一条 SFA_PICK 自证生效）"

if "#include <cstdlib>" not in src:
    src = src.replace("#include <cmath>", "#include <cmath>\n#include <cstdlib>", 1)

anchor = "    tiling->n_blk = nBlk;\n"
assert src.count(anchor) == 1, "n_blk anchor"
src = src.replace(anchor,
    "    { const char *pr = getenv(\"SFA_PROBE_ROWS\"), *pk = getenv(\"SFA_PROBE_K\");\n"
    "      if (pr != nullptr && pk != nullptr &&\n"
    "          static_cast<uint64_t>(B) * Q_S <= static_cast<uint64_t>(atoll(pr)) &&\n"
    "          atoi(pk) >= 8) { nBlk = static_cast<uint32_t>(atoi(pk)); } }\n"
    # ⚠️ 生效凭据必须打在**覆盖之后**：SFA_PICK 是 nb_sweep_patch 挂在 GetTilingData 之前的，
    #    它量不到这一行（第一次标定就是靠"pick 里 nblk 一直是 48 但时间动了 +25 %"抓到的）。
    "    { static bool sfa_eff_shown = false;\n"
    "      if (!sfa_eff_shown) { sfa_eff_shown = true;\n"
    "        std::printf(\"SFA_EFF nblk=%u rows=%llu\\n\", nBlk,\n"
    "                    (unsigned long long)(B * Q_S)); } }\n"
    + anchor, 1)

open(PATH, "w", encoding="utf-8").write(src)
print("PATCHED", PATH)
