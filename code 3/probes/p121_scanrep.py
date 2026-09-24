#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P121 计量器：给"每 token 一条标量 GM 下标读"定价（§1.1 剩下的候选 (b)）。

为什么不能像 P114/P119 那样"原地重跑"：`NextTokenBlock` 推进 `tokIdx/curBegin/curEnd/hasBlock`
⇒ 不幂等，重跑会改输出。这里改成**只重读地址、不重建区间**：把本 chunk 刚扫过的那段下标
再读 N 遍，逐条 `& 0xFFFF` 累加进一个 uint64，最后喂给一个**可证明永不成立**的分支。

三道论证（都写进注释，闸门要能核）：
  ① 惰性：读的是同一张 `sparse_indices`、不写任何暂存 ⇒ 输出逐位不变（本地 8 档 out/max/sum 对拍）。
  ② 分支永不成立：每项 ≤ 0xFFFF、每 query 总项数 ≤ N·sparseCount ≤ 255·65536 ⇒ acc < 2^40
     而哨兵是 UINT64_MAX ⇒ store 不可达 ⇒ 连"kfBuf_ 是死区"那层论证都不需要（比 P118/P119 更硬）。
  ③ 不许被合并：每遍的地址带上 `acc >> 63`（运行时恒 0，但编译器无法证明）⇒ 各遍的地址
     符号上不相等 ⇒ 标量读不会被 CSE 成一遍。**没有这一条，N=3 与 N=1 会是同一份机器码。**

用法：p121_scanrep.py N 输出文件 [pristine 路径]  （N ∈ 0..255；N=0 ⇒ 原样输出 pristine）
      ⚠️ 第 3 个参数是给远端用的（那边 pristine 落在 ~/sfa_real/kernel_pristine.cpp）；
      本地不给就走仓库里的备份。
"""
import io
import re
import sys

N = int(sys.argv[1])
OUT = sys.argv[2]
PRISTINE = sys.argv[3] if len(sys.argv) > 3 else \
    "/home/fszqsn/ops_comp/ascend-operator-competition/code 3/probes/backup/p114_pre_probe/sparse_flash_attention.cpp"
src = io.open(PRISTINE, encoding="utf-8").read()

A_ACC = "        uint32_t tokIdx = tokBeg;\n"
A_LO = "            uint32_t nRun = 0u;\n            uint32_t cnt = 0u;\n"
A_RESCAN = ("                if (curBegin >= curEnd) { hasBlock = false; }\n"
            "            }\n"
            "            if (cnt == 0u) { break; }")
A_SINK = ("        if (pend > 0u) { FlushChunk(q, o, kb, kr, sc, ml, nbCur, pend, rowBase, pendRuns); }\n"
          "\n"
          "        // 归一化 + 写回")

for tag, a in (("ACC", A_ACC), ("LO", A_LO), ("RESCAN", A_RESCAN), ("SINK", A_SINK)):
    assert src.count(a) == 1, f"锚点 {tag} 命中 {src.count(a)} 次"

# ----  pristine 的旗标/管道条数与 InitBuffer 数：改完必须一根手指都没多碰 ----
FLAG0 = len(re.findall(r"(?:Set|Wait)Flag<HardEvent::\w+>\(\d+\)", src))
BUF0 = src.count("InitBuffer")

if N == 0:
    io.open(OUT, "w", encoding="utf-8").write(src)
    print(f"N=0 MARK=0（pristine 直拷）  FLAG={FLAG0} BUFF={BUF0}")
    sys.exit(0)

INS_ACC = A_ACC + ("        uint64_t p121Acc = 0ull;   // P121 计量器：本 query 累加的下标读之和（sink 在函数尾）\n")

INS_LO = A_LO + ("            const uint32_t p121Lo = tokIdx;   // P121：本 chunk 扫描的起点（终点 = 内层循环后的 tokIdx）\n")

INS_RESCAN = ("                if (curBegin >= curEnd) { hasBlock = false; }\n"
              "            }\n"
              "            // P121 扫描计量器：把【本 chunk 刚扫过的那段下标】再读 {N} 遍 —— 同基址、同步进、\n"
              "            //   同条数，只累加不解释、不写任何暂存 ⇒ 输出逐位不变。\n"
              "            //   ⚠️ 地址里那个 `p121Acc >> 63` 运行时恒 0（见函数尾的界），但编译器无法证明 ⇒\n"
              "            //   各遍地址符号上不相等，标量读不会被合并成一遍（否则 N=3 与 N=1 同一份机器码）。\n"
              "            for (uint32_t p121r = 0u; p121r < {N}u; ++p121r) {{\n"
              "                const uint64_t p121Off = p121Acc >> 63;\n"
              "                for (uint32_t p121i = p121Lo; p121i < tokIdx; p121i += tokStep) {{\n"
              "                    const int32_t p121v = idxGm_.GetValue(idxBase + p121Off + p121i);\n"
              "                    p121Acc += static_cast<uint64_t>(static_cast<uint32_t>(p121v) & 0xFFFFu);\n"
              "                }}\n"
              "            }}\n"
              "            if (cnt == 0u) { break; }").replace("{N}", str(N)).replace("{{", "{").replace("}}", "}")

INS_SINK = ("        if (pend > 0u) { FlushChunk(q, o, kb, kr, sc, ml, nbCur, pend, rowBase, pendRuns); }\n"
            "\n"
            "        // P121 sink：把累加值喂给一个【可证明永不成立】的分支，逼编译器保留上面那些标量读。\n"
            "        //   界：每项 & 0xFFFF ≤ 65535，每 query 总项数 ≤ {N}·sparseCount ≤ {N}·65536 ≤ 255·65536\n"
            "        //   ⇒ acc < 2^40，而哨兵 = UINT64_MAX ⇒ store 在任何输入下都不可达 ⇒ 这一族的唯一效果\n"
            "        //   就是「多读了 {N} 遍下标」，连 scratch 归属都不用论证。\n"
            "        if (p121Acc == 0xFFFFFFFFFFFFFFFFull) {{\n"
            "            kfBuf_.Get<float>().SetValue(scGrp_ * D_ - 1u, 1.0f);\n"
            "        }}\n"
            "\n"
            "        // 归一化 + 写回").replace("{N}", str(N)).replace("{{", "{").replace("}}", "}")

src = src.replace(A_ACC, INS_ACC, 1).replace(A_LO, INS_LO, 1)
src = src.replace(A_RESCAN, INS_RESCAN, 1).replace(A_SINK, INS_SINK, 1)

assert src.count("P121") >= 4, src.count("P121")
FLAG1 = len(re.findall(r"(?:Set|Wait)Flag<HardEvent::\w+>\(\d+\)", src))
BUF1 = src.count("InitBuffer")
assert FLAG1 == FLAG0 == 36, (FLAG0, FLAG1)
assert BUF1 == BUF0 == 13, (BUF0, BUF1)
assert src.count("idxGm_.GetValue") == 2, src.count("idxGm_.GetValue")
io.open(OUT, "w", encoding="utf-8").write(src)
print(f"N={N} MARK={src.count('P121 扫描计量器')}/1 FLAG={FLAG1}（应 {FLAG0}） BUFF={BUF1}（应 {BUF0}） "
      f"READS={src.count('idxGm_.GetValue')} bytes={len(src.encode('utf-8'))}")
