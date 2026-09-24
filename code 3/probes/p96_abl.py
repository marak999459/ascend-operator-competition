#!/usr/bin/env python3
"""P96 分诊：把"环行距锁 n_blk"这一发拆成两个可分离的半边，各自在【远端副本】上还原。

  full     —— 不改（= 提交源当前状态：AIC Fixpipe nSize=n_blk + AIV 一条整块读回）
  fxTile   —— 只还原 AIC 那半边（Fixpipe 回到 nTile），AIV 仍一条整块读回
              ⇒ 尾片两侧行距不一致，数值必然错，只看"还挂不挂"
  aivLoop  —— 只还原 AIV 那半边（每头一条、行距仍是 n_blk），AIC 仍写 n_blk 行距
              ⇒ 数值应当正确，只看"还挂不挂"
  orig     —— 两半都还原 = P91 已验证态（对照组，必须三例全过）

用法：p96_abl.py <full|fxTile|aivLoop|orig> <输入文件> <输出文件>
"""
import sys

ARM, SRC, DST = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(SRC, encoding="utf-8").read()

BASE = "        const uint64_t base = static_cast<uint64_t>(ringBase) >> 1;\n"
CUR = BASE + (
    "        const uint32_t blkPerRow = nBlk_ * static_cast<uint32_t>(sizeof(float)) / sfa::UB_BLK;\n"
    "        const DataCopyParams dcp{1, static_cast<uint16_t>(nbCur * blkPerRow), 0, 0};")
LOOPBLK = BASE + (
    "        const uint32_t blkPerRow = nBlk_ * static_cast<uint32_t>(sizeof(float)) / sfa::UB_BLK;\n"
    "        const DataCopyParams dcp{1, static_cast<uint16_t>(blkPerRow), 0, 0};")
ORIGBLK = BASE + (
    "        const uint32_t nTile = (m + 15u) & ~15u;\n"
    "        const uint32_t blkPerRow = nTile * static_cast<uint32_t>(sizeof(float)) / sfa::UB_BLK;\n"
    "        const DataCopyParams dcp{1, static_cast<uint16_t>(blkPerRow), 0, 0};")
ONE = "        DataCopy(sc, ringOutGm_[base + static_cast<uint64_t>(sub_ * nb_) * nBlk_], dcp);"
LOOPD = """        for (uint32_t i = 0u; i < nbCur; ++i) {
            DataCopy(sc[i * nBlk_],
                     ringOutGm_[base + static_cast<uint64_t>((sub_ * nb_) + i) * nBlk_], dcp);
        }"""
ORIGD = """        for (uint32_t i = 0u; i < nbCur; ++i) {
            DataCopy(sc[i * nBlk_],
                     ringOutGm_[base + static_cast<uint64_t>((sub_ * nb_) + i) * nTile], dcp);
        }"""
FXW = ("        ctx.fx.nSize = static_cast<uint16_t>(nBlk_);\n"
       "        ctx.fx.dstStride = nBlk_;")
FXT = ("        ctx.fx.nSize = static_cast<uint16_t>(nTile);\n"
       "        ctx.fx.dstStride = nTile;")

REPL = {
    "full": [],
    "fxTile": [(FXW, FXT)],
    "aivLoop": [(CUR, LOOPBLK), (ONE, LOOPD)],
    "orig": [(CUR, ORIGBLK), (ONE, ORIGD), (FXW, FXT)],
}[ARM] if ARM in ("full", "fxTile", "aivLoop", "orig") else None

hits = 0
for old, new in REPL:
    hits += txt.count(old)
    txt = txt.replace(old, new)
if ARM != "full" and hits != len(REPL):
    raise SystemExit("PATCH-MISS: 臂 %s 锚点命中 %d/%d，作废" % (ARM, hits, len(REPL)))
open(DST, "w", encoding="utf-8").write(txt)
print("PATCH-OK %s hits=%d" % (ARM, hits))
