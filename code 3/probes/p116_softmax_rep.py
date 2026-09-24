#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P116 探针补丁：把 `SoftmaxPv` 里**幂等的前半段**原地重跑 N 遍 ⇒ 逐位惰性，买的是 softmax 段时间。

为什么是这一段（`sparse_flash_attention.cpp` 的 1016~1054，锚点见下）：
  · 第 1 步 行最大：`WholeReduceMax` 读 `sc` 写 `mx`；那条标量循环 `mx.SetValue(i, max(mx, ml))`
    读的是 **还没被改过的 `ml`** ⇒ 重跑一遍写回的是同一个数 ⇒ 幂等。
  · 第 2 步 `P = exp(sc - mNew)`：只读 `sc`/`mx`、只写 `p` ⇒ 幂等。
  · 第 3 步的**向量部分**：`Duplicate(r0,0)` + `WholeReduceSum` + `Add` ⇒ 幂等（`r0` 每次自己归零）。
  ⇒ 三段的输出在第二遍里逐位不变。
⛔ 第 3 步的**标量部分**（`av.SetValue(ml-mx)` 之后紧跟 `ml.SetValue(mx)`）**不幂等**：第二遍此时
   `ml` 已经是 `mx` ⇒ alpha 会算成 `exp(0)=1` ⇒ 输出必错。所以区域右界就切在 `ml` 被改写之前。
   同理第 4 步（`o *= a`、`l = l*a + r0`）与第 5 步（`Axpy` 累加 `O`）都不可重复 ⇒ **PV 段没法用"重复"定价**，
   那一段只能靠消融（P117）或减法。
读数：Δ% = softmax 段占该档墙的比例（P110 已证"加工作量 1:1 涨"）。
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PATH = sys.argv[2] if len(sys.argv) > 2 else "code/op_kernel/sparse_flash_attention.cpp"
PRISTINE = sys.argv[3] if len(sys.argv) > 3 else "../kernel_pristine.cpp"
assert 1 <= N <= 4, "N must be 1..4"

BEGIN = "        // ---- 1) 行最大：一条 WholeReduceMax 做 nbCur 行，列数超过 RED_SLAB 时分块取 max\n"
END = "            Add(r0, r0, r1, nbCur);\n        }\n"

src = open(PRISTINE, "r", encoding="utf-8").read()
for a in (BEGIN, END):
    assert src.count(a) == 1, "anchor hits=%d: %r" % (src.count(a), a[:48])
i0, i1 = src.index(BEGIN), src.index(END) + len(END)
region = src[i0:i1]
# 区域必须正好包含这三条不变量，否则切错了界（尤其 ⛔ 不能把 ml.SetValue 圈进来）
# ⚠️ 数**代码拼写**而不是标识符：BEGIN 那行注释里也写着 "WholeReduceMax" ⇒ 数标识符会多 1。
assert region.count("WholeReduceMax<float>") == 2, \
    "region 里 WholeReduceMax<float> 调用数=%d（应 2：k0==0 与 else 两支）" % region.count("WholeReduceMax<float>")
assert region.count("WholeReduceSum<float>") == 1, "region 切错"
assert region.count("Duplicate(r0") == 1 and region.count("Add(r0, r0, r1") == 1, "region 少了第 3 步向量部分"
assert region.count("Exp<float>") == 1, "region 只该含第 2 步那一条 exp（`av` 那条在 ml 改写之后 ⇒ 界切错了）"
assert "Maxs(pi" in region, "region 少了第 2 步的下限夹取"
assert "ml.SetValue" not in region and "Axpy" not in region and "Muls(oi" not in region, "region 吃进了非幂等段"
if N > 1:
    ind = " " * 8
    body = "".join(("    " + ln if ln.strip() else ln) for ln in region.splitlines(keepends=True))
    src = src[:i0] + (
        ind + "{                                              // P116 softmax-rep x%d\n" % N
        + ind + "    for (uint32_t rP116_ = 0u; rP116_ < %du; ++rP116_) {\n" % N
        + body
        + ind + "    }\n"
        + ind + "}\n" + src[i1:])
open(PATH, "w", encoding="utf-8").write(src)
print("REP=%d  HITS_P116=%d  EXPECT=%d" % (N, src.count("P116 softmax-rep"), 1 if N > 1 else 0))
