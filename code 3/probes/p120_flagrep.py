#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P120 探针补丁：在 `FlushChunk` 末尾追加 **N 对同核 `SetFlag/WaitFlag<HardEvent::MTE2_V>(4)`**
⇒ 将"每 chunk 一对旗标"的单价从**从没量过**变成**量过**。

为什么这是第一把量"开销"而不是"工作"的尺子（前七把都在量段）：
  主循环每 chunk 压着 **4 对**同核旗标（L965/966、972/973、983/984、987/988），而 §4 那条
  0.42 µs 是**跨核** `CrossCoreSetFlag` 的价 —— 同核这一对的单价从没进过账。L197 那句
  "代价是每 chunk 多一次 V 排空 + MTE2 旗标对"是**断言不是读数**，本尺就是去读它。

为什么惰性是**构造级**的（前七把都得论证"这块内存没人读"，这把不用）：
  · 两条指令都不碰内存、不碰 UB、不碰寄存器堆 ⇒ 没有任何数据被读或被写。
  · event 线 `MTE2_V(4)` 全文件从没出现过（`MTE2_V` 只用了 0 与 1）⇒ 不与任何现有配对交错。
  · 严格 Set→Wait 交替（同 id 一次 outstanding）⇒ 与 L965 那几对同构，不可能死锁。
  · **不新增任何 buffer** ⇒ UB 预算 / 环 / tiling 三个混淆变量一个都不动。
  唯一的副作用：给 V 流水线多插 N 个"等 MTE2 走到这个点"的耦合 —— 而这正是要定价的东西。

⚠️ 与"复制段"那族的方向差别（读数时别搞混）：复制体只加**长度**不加**依赖**，而本尺加的是
   真实的管道耦合 ⇒ 大 Δ 会**高估**现有 4 对的价（它们是各自必需的耦合，不 N 份可叠加）；
   小 Δ 则仍然可靠地否证（4 对 ≈0 的价）。⇒ 本尺的判读按"关闭方向单向可用"来用。

用法：p120_flagrep.py N DST PRISTINE     （N = 追加几对，0 = pristine 直拷）
    发出去的那一档 = 127（本地标定：31 对只 +0.8~2.0 %，会落进平台 ±2 % 带内）
"""
import re
import shutil
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PATH = sys.argv[2] if len(sys.argv) > 2 else "code/op_kernel/sparse_flash_attention.cpp"
PRISTINE = sys.argv[3] if len(sys.argv) > 3 else "../kernel_pristine.cpp"
assert 0 <= N <= 255, "N must be 0..255"

# FlushChunk 的收口那三行 = 全文件唯一锚点（V_MTE2(0) 那对只在 635/987 出现，三行连排只此一处）
TAIL = ("        SoftmaxPv(o, vb, sc, ml, nbCur, m);\n"
        "        SetFlag<HardEvent::V_MTE2>(0);\n"
        "        WaitFlag<HardEvent::V_MTE2>(0);\n"
        "    }\n")
PAIR = ("        SetFlag<HardEvent::MTE2_V>(4);\n"
        "        WaitFlag<HardEvent::MTE2_V>(4);\n")
HEAD = "        // ---- P120 flag meter ----\n"

if N == 0:
    shutil.copyfile(PRISTINE, PATH)
    print("N=0 MARK=0（pristine 直拷）")
    sys.exit(0)

src = open(PRISTINE, "r", encoding="utf-8").read()
assert src.count(TAIL) == 1, "anchor hits=%d" % src.count(TAIL)
assert "P120" not in src, "pristine 里已有 P120 残留"
# 门：event 线 MTE2_V 的 id=4 必须空闲（现有使用者只能落在 0..3）
used = re.findall(r"(Set|Wait)Flag<HardEvent::(\w+)>\((\d+)\)", src)
assert len(used) == 36, "旗标条数 != 36（18 对）：%d" % len(used)
for kind, ev, idn in used:
    assert int(idn) <= 3, "id>3 已被占用：%s(%s)" % (ev, idn)
    assert not (ev == "MTE2_V" and idn == "4"), "MTE2_V(4) 不是空闲线"
src = src.replace(TAIL, TAIL[:-len("    }\n")] + HEAD + PAIR * N + "    }\n", 1)
open(PATH, "w", encoding="utf-8").write(src)
print("N=%d MARK=%d/%d SET=%d/%d WAIT=%d/%d bytes=%d" % (
    N, src.count("P120 flag meter"), 1,
    src.count("SetFlag<HardEvent::MTE2_V>(4)"), N,
    src.count("WaitFlag<HardEvent::MTE2_V>(4)"), N,
    len(src.encode("utf-8"))))
