#!/usr/bin/env python3
# P110 消融补丁：把 ComputeScores 在原位再跑 (N-1) 遍。
#
# 为什么这一发能回答问题：重跑写的是**同一块 sc**、输入（kb/kr 的 fp16、q）一遍没动
# ⇒ 结果逐位不变 ⇒ 这是一个**语义惰性**的计算量扰动。于是"时间对计算量的斜率"直接读出
# 向量流水线到底饱和没有：
#   t1 = A + S（A = 与 score 无关的那部分墙，S = score 计算段）
#   tN = A + N·S  ⇒  S = (tN − t1)/(N − 1)，A = t1 − S
#   ⇒ S/t1 ≈ 1 ⇒ 向量墙（要动就得改精度/改架构）；S/t1 ≪ 申报的 51~55 % ⇒ 有空隙（微优化还有肉）
# ⚠️ 只作用在远端副本。本地提交源一个字不落。
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PATH = sys.argv[2] if len(sys.argv) > 2 else "code/op_kernel/sparse_flash_attention.cpp"
PRISTINE = sys.argv[3] if len(sys.argv) > 3 else "../kernel_pristine.cpp"

src = open(PRISTINE, "r", encoding="utf-8").read()
anchor = "        ComputeScores(q, kb, kr, sc, nbCur, m);\n"
assert src.count(anchor) == 1, "ComputeScores anchor hits=%d" % src.count(anchor)
if N > 1:
    # 第二对 MTE2_V 旗标：V 搬进的是 kb 自己那块 UB（P32），重跑必须插在 V 的 MTE2 之前
    src = src.replace(anchor, anchor + "".join(
        "        ComputeScores(q, kb, kr, sc, nbCur, m);  // P110 rep%d\n" % (r + 2)
        for r in range(N - 1)), 1)
open(PATH, "w", encoding="utf-8").write(src)
print("REP=%d  extra_calls=%d  HITS=%d" % (N, N - 1, src.count("P110 rep")))
