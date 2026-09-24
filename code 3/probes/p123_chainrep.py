#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P123 计量器：给**依赖链**定价 —— §1.1 那 ≈71 % 在 (c) 静态关闭后唯一剩下的具名落点。

为什么这一发不再是"复制同样的活"：P114~P121 八把尺子全是"多干一份同样的活"，它们的 Δ 是
**边际发射成本**；而复制体若落在别人影子里（流水还空着、或在等别的事），Δ 就是 0 —— P119/P120/P121
那三个 ≈0 全是这个形状。⇒ 这一族**构造上读不到"等待"**。本尺子的做法是复制一份**必须串行**的活：
目的端 = 真累加器 `o` 本身（`Muls(oc, oc, 1.0f, rowC)`），`dst == src` ⇒ N 份彼此排队，
且下一个 chunk 的 PV 必须等它们 ⇒ 量的是"链上多一圈"，不是"多一批指令"。

三道论证：
  ① 逐位恒等：`x * 1.0f` 在 IEEE 下对**所有**浮点值逐位不变（本文件 `ZeroPaddingOut` 上方那句
     注释就是靠它做 fp16→fp32→fp16 透传的）⇒ 不写 scratch、不改任何值 ⇒ 输出逐位相同。
  ② 与 P119 只差一个变量：同指令类（`Muls`/向量乘）、同宽度（`rowC = D_` 个 fp32）、同位置
     （`SoftmaxPv` 尾部）、同目的端语义（真累加器）—— P119 的复制体写 `kfBuf_` 末行 ⇒ 可自由重叠。
  ③ 无 sink、无 scratch 归属问题：写的就是活内存。⚠️ 唯一失效模式 = 编译器证明 `x*1 ⇒ x` 并折叠
     ⇒ **本地四臂标定是必经步骤**（N=0/1/3/7 必须 Δ∝N，否则尺子不存在，这一发不许发）。

用法：p123_chainrep.py N 输出文件 [pristine 路径]  （N ∈ 1..255；N=0 ⇒ 原样直拷 pristine）
"""
import io
import re
import sys

N = int(sys.argv[1])
OUT = sys.argv[2]
PRISTINE = sys.argv[3] if len(sys.argv) > 3 else \
    "/home/fszqsn/ops_comp/ascend-operator-competition/code 3/probes/backup/p114_pre_probe/sparse_flash_attention.cpp"
src = io.open(PRISTINE, encoding="utf-8").read()

# 锚点 = SoftmaxPv 第 5 段（PV 累加）的收尾 + 函数右括号
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
       "        // P123 依赖链计量器：把【真累加器自己乘 1】再串 {N} 圈 —— 与本函数上方第 4 段那道真重缩放\n"
       "        //   同指令、同宽度（rowC 个 fp32）、同目的端，唯一区别是乘数恒 1（IEEE 下逐位恒等 ⇒ 输出不变）。\n"
       "        //   ⚠️ 关键是 dst == src ⇒ 这 {N} 圈彼此排队、且下一 chunk 的 PV 必须等它们 ⇒ 读到的是\"链上多\n"
       "        //   一圈\"的价，而不是\"多一批指令\"的价（P119 的复制体写 kfBuf_ 末行 ⇒ 可自由重叠）。\n"
       "        for (uint32_t p123r = 0u; p123r < {N}u; ++p123r) {{\n"
       "            for (uint32_t p123i = 0u; p123i < nbCur; ++p123i) {{\n"
       "                LocalTensor<float> oc = o[p123i * rowC];\n"
       "                Muls(oc, oc, 1.0f, rowC);\n"
       "            }}\n"
       "        }}\n"
       "    }\n").replace("{N}", str(N)).replace("{{", "{").replace("}}", "}")

src = src.replace(A_TAIL, INS, 1)

assert src.count("P123 依赖链计量器") == 1
FLAG1 = len(re.findall(r"(?:Set|Wait)Flag<HardEvent::\w+>\(\d+\)", src))
BUF1 = src.count("InitBuffer")
MULS1 = src.count("Muls(")
assert FLAG1 == FLAG0 == 36, (FLAG0, FLAG1)
assert BUF1 == BUF0 == 13, (BUF0, BUF1)
assert MULS1 == MULS0 + 1, (MULS0, MULS1)
assert src.count("Muls(oc, oc, 1.0f, rowC)") == 1
io.open(OUT, "w", encoding="utf-8").write(src)
print(f"N={N} MARK={src.count('P123 依赖链计量器')} LOOP={src.count('p123r = 0u')} "
      f"INNER={src.count('Muls(oc, oc, 1.0f, rowC)')} FLAG={FLAG1}（应 {FLAG0}） "
      f"BUFF={BUF1}（应 {BUF0}） MULS=+{MULS1-MULS0} bytes={len(src.encode('utf-8'))}")
