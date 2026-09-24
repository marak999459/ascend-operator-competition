#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P124 = **P123 的死目的端孪生**：同样 {N} 圈 × 每圈 p124n 条、同 `Muls`、同 `rowC` 宽度、
同位置（`SoftmaxPv` 尾）、同链距，唯一改变的变量 = 目的端从**活累加器 `o`** 换成
**`kfBuf_` 的前 p124n 行**（本函数返回后即死内存；且 `×1.0f` 逐位不改值 ⇒ 双重惰性）。

为什么这一发值得花（P123 读完之后剩下的唯一一条能一次解两个未知数的路）：
  P119（死端，加 `m·nbCur` 条）平台 **+2.9 %**；P123（活端，加 `63·nbCur` 条）平台 **+22.7 %**。
  两个读数、三个未知（每圈死端价 p_D、活端价 p_L、平台每 chunk 的 token 数 m）⇒ 差一个方程。
  本发补上它：Δ124 = {N}·p_D，而 p_D = 2.9/m ⇒ **m = {N}·2.9/Δ124 = 182.7/Δ124**，
  顺带给出 **p_L/p_D = Δ123/Δ124**。⇒ 一发同时回答"平台的 chunk 到底填多满"和
  **"目的端死活是不是一个 8× 的轴"** —— 后一条关系到 P119/P112/P116 三把押在死写上的尺子
  是不是系统性偏低（§1.1 那句"向量线只值 ≈8~11 %"整个建在这三把上）。

⚠️ 界：`kfBuf_` 只有 `scGrp_` 行（= `scGrp_·D_` 个 float），而 `nbCur ≤ nb_` 可以大于 `scGrp_`
   ⇒ 内圈长度取 `p124n = min(nbCur, scGrp_)`（无除法、无取模，只做一次比较），
   越界的可能为 0；若平台真出现 `nbCur > scGrp_` 则本发的条数是 `{N}·scGrp_` 而非 `{N}·nbCur`，
   方向已知、读的时候按这个口径折。

用法：p124_deadrep.py N 输出文件 [pristine 路径]  （N ∈ 1..255；N=0 ⇒ 原样直拷 pristine）
"""
import io
import re
import sys

N = int(sys.argv[1])
OUT = sys.argv[2]
PRISTINE = sys.argv[3] if len(sys.argv) > 3 else \
    "/home/fszqsn/ops_comp/ascend-operator-competition/code 3/probes/backup/p114_pre_probe/sparse_flash_attention.cpp"
src = io.open(PRISTINE, encoding="utf-8").read()

# 锚点 = SoftmaxPv 第 5 段（PV 累加）的收尾 + 函数右括号（与 P123 同一处、同一锚点）
A_TAIL = ("                    Axpy<float, float>(o[i * rowC], vt, p.GetValue(i * nBlk_ + j), rowC);\n"
          "                }\n"
          "            }\n"
          "        }\n"
          "    }\n")
assert src.count(A_TAIL) == 1, f"锚点 TAIL 命中 {src.count(A_TAIL)} 次"

FLAG0 = len(re.findall(r"(?:Set|Wait)Flag<HardEvent::\w+>\(\d+\)", src))
BUF0 = src.count("InitBuffer")
MULS0 = src.count("Muls(")

if N == 0:
    io.open(OUT, "w", encoding="utf-8").write(src)
    print(f"N=0 MARK=0（pristine 直拷）  FLAG={FLAG0} BUFF={BUF0} MULS={MULS0}")
    sys.exit(0)

INS = ("                    Axpy<float, float>(o[i * rowC], vt, p.GetValue(i * nBlk_ + j), rowC);\n"
       "                }\n"
       "            }\n"
       "        }\n"
       "        // P124 死端孪生计量器：与活端那一发只差**一个变量** = 目的端（活累加器 → 本函数\n"
       "        //   返回后即死的 kfBuf_ 行）。同指令、同宽度（rowC 个 fp32）、同位置、同链距、\n"
       "        //   同条数（{N}×min(nbCur,scGrp_)），且 `×1.0f` 逐位不改值 ⇒ 双重惰性。\n"
       "        //   ⇒ 与另外两把尺子合解：每 chunk 的 token 数 m = {N}×2.9/Δ，活∶死 = Δ活/Δ。\n"
       "        const uint32_t p124n = (nbCur < scGrp_) ? nbCur : scGrp_;   // kfBuf_ 只有 scGrp_ 行\n"
       "        for (uint32_t p124r = 0u; p124r < {N}u; ++p124r) {{\n"
       "            for (uint32_t p124i = 0u; p124i < p124n; ++p124i) {{\n"
       "                LocalTensor<float> dk = vf[p124i * rowC];\n"
       "                Muls(dk, dk, 1.0f, rowC);\n"
       "            }}\n"
       "        }}\n"
       "    }\n").replace("{N}", str(N)).replace("{{", "{").replace("}}", "}")

src = src.replace(A_TAIL, INS, 1)

assert src.count("P124 死端孪生计量器") == 1
FLAG1 = len(re.findall(r"(?:Set|Wait)Flag<HardEvent::\w+>\(\d+\)", src))
BUF1 = src.count("InitBuffer")
MULS1 = src.count("Muls(")
assert FLAG1 == FLAG0 == 36, (FLAG0, FLAG1)
assert BUF1 == BUF0 == 13, (BUF0, BUF1)
assert MULS1 == MULS0 + 1, (MULS0, MULS1)
assert src.count("Muls(dk, dk, 1.0f, rowC)") == 1
assert "vf[p124i * rowC]" in src
io.open(OUT, "w", encoding="utf-8").write(src)
print(f"N={N} MARK={src.count('P124 死端孪生计量器')} LOOP={src.count('p124r = 0u')} "
      f"INNER={src.count('Muls(dk, dk, 1.0f, rowC)')} FLAG={FLAG1}（应 {FLAG0}） "
      f"BUFF={BUF1}（应 {BUF0}） MULS=+{MULS1-MULS0} bytes={len(src.encode('utf-8'))}")
