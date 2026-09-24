#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P125 计量器 = 给"每个单元付一次的那两段 I/O"定价（双臂，同一份 pristine 现生成）。

  ARM=q（prologue）：`LoadQ(...)` + 三条 `Duplicate`（o 清零 / m=MIN / l=0）整段 ×N。
  ARM=w（epilogue）：`WriteOut` 的**两段 DMA 尾巴**各 ×N —— ① 每组头的
      `V_MTE3(1)` + `CopyUb2Gm(outGm_)` + `MTE3_V(1)`；② `lseOn_` 那对
      `DataCopyPad(maxGm_/sumGm_)` + `V_MTE3/MTE3_V(2)`。⛔ `Muls`/`PackFromF32`/
      `GetValue` 一遍都不多跑（`Muls(oi, oi, 1/l)` 是**原位**写，重复 = 除 l²）。

两臂的可证幂等（细节与出处在 `p125_design.txt` 第 2 条）：prologue 是 GM→UB→Cast 的纯函数、
`kb`/`kr` 到后面的 while 循环才被 KV 覆盖；epilogue 重复段只**从 `st` 读、往同一批 GM 地址写**
同一份字节。⇒ 数值逐位不变，但照旧要过本地逐位门，不靠证明放行。

用法：p125_pin.py N ARM OUT [PRISTINE]     （N ∈ 0..255；N=0 ⇒ 原样直拷 pristine）
每步都带命中数断言（§7 第 2 条）：旗标/缓冲/`Muls` 的**静态**条数在两臂都不许变
（重复靠 `for` 包住，不是把语句抄 N 遍 ⇒ 文本命中数不变才是对的）。
"""
import hashlib
import io
import re
import sys

REPO = "/home/fszqsn/ops_comp/ascend-operator-competition"
N = int(sys.argv[1])
ARM = sys.argv[2]
OUT = sys.argv[3]
PRISTINE = sys.argv[4] if len(sys.argv) > 4 else \
    REPO + "/code 3/code/op_kernel/sparse_flash_attention.cpp"

SHA_PRISTINE = "f815bf1eaba0f8bc"          # 榜上那份 kernel 的前 16 位（当场核，⛔ 不信文档）
src = io.open(PRISTINE, encoding="utf-8").read()
assert hashlib.sha256(src.encode("utf-8")).hexdigest()[:16] == SHA_PRISTINE, \
    "pristine 不是榜上那份 kernel：%s" % PRISTINE

FLAG_RE = re.compile(r"(?:Set|Wait)Flag<HardEvent::\w+>\(\d+\)")
FLAG0 = len(FLAG_RE.findall(src))
BUF0 = src.count("InitBuffer")
MULS0 = src.count("Muls(")
LOADQ0 = src.count("LoadQ(q, kb, kr,")
DUP0 = src.count("Duplicate(o, 0.0f, nbCur")
COPY0 = src.count("CopyUb2Gm(outGm_")
PAD0 = src.count("DataCopyPad(maxGm_")
REP0 = src.count("p125r = 0u")
# ⚠️ `CopyUb2Gm(outGm_` / `DataCopyPad(maxGm_` 各有 3 处（WriteOut / 折叠轮 / padding 路径），本发只打
#    **WriteOut** 那两处 ⇒ 放行靠"锚点整段唯一"（wrap 里断言），不靠这两个计数。
assert (FLAG0, BUF0, LOADQ0, DUP0, COPY0, PAD0, REP0) == (36, 13, 1, 1, 3, 3, 0), \
    "基线锚点数不对：%s" % ((FLAG0, BUF0, LOADQ0, DUP0, COPY0, PAD0, REP0),)

# ---- 臂 q：prologue 整段 ----
Q_ANCHOR = ("        // Q 与 Qrope 拼接：前 512 content、后 64 rope —— 整批 MTE2 + Cast（P5b）\n"
            "        LoadQ(q, kb, kr, s1Base, ropeBase, n0, nbCur);\n"
            "\n"
            "        // O 累加器清零；m = SOFTMAX_MIN_NUM、l = 0\n"
            "        Duplicate(o, 0.0f, nbCur * static_cast<uint32_t>(D_));\n"
            "        Duplicate(ml, sfa::SOFTMAX_MIN_NUM, nbCur);\n"
            "        Duplicate(ml[halfOff_], 0.0f, nbCur);\n")
Q_BODY = ("            LoadQ(q, kb, kr, s1Base, ropeBase, n0, nbCur);\n"
          "            Duplicate(o, 0.0f, nbCur * static_cast<uint32_t>(D_));\n"
          "            Duplicate(ml, sfa::SOFTMAX_MIN_NUM, nbCur);\n"
          "            Duplicate(ml[halfOff_], 0.0f, nbCur);\n")

# ---- 臂 w：epilogue 两段 DMA 尾巴 ----
def ind4(seg):
    """整段右移 4 空格（逐行前缀，⛔ 不用 str.replace —— 那会把续行里的 24 空格也翻倍）。"""
    return "".join(("    " + ln if ln.strip() else ln) + "\n" for ln in seg.splitlines())


W1_ANCHOR = ("            SetFlag<HardEvent::V_MTE3>(1);\n"
             "            WaitFlag<HardEvent::V_MTE3>(1);\n"
             "            CopyUb2Gm(outGm_[s1Base + (n0 + i0) * rowC], st, g * rowC);\n"
             "            SetFlag<HardEvent::MTE3_V>(1);\n"
             "            WaitFlag<HardEvent::MTE3_V>(1);\n")
W1_BODY = ind4(W1_ANCHOR)
W2_ANCHOR = ("            // 尾部那对 MTE3_V 保证：下一个头块重写 lseBuf_ 之前，本块的 DataCopyPad 已读完。\n"
             "            SetFlag<HardEvent::V_MTE3>(2);\n"
             "            WaitFlag<HardEvent::V_MTE3>(2);\n"
             "            DataCopyPad(maxGm_[lseOff], lse,\n"
             "                        DataCopyExtParams{1, static_cast<uint32_t>(nbCur * sizeof(float)), 0, 0, 0});\n"
             "            DataCopyPad(sumGm_[lseOff], lse[halfOff_],\n"
             "                        DataCopyExtParams{1, static_cast<uint32_t>(nbCur * sizeof(float)), 0, 0, 0});\n"
             "            SetFlag<HardEvent::MTE3_V>(2);\n"
             "            WaitFlag<HardEvent::MTE3_V>(2);\n")
W2_BODY = ind4(W2_ANCHOR)

HDR = ("        // P125 计量器（%s 臂）：本段每单元只付一次，这里整段重跑 N=%d 遍（N=1 = 原样一遍，\n"
       "        // 是对照组）⇒ Δ/(N−1) 就是一圈（=多付一次本段）的价。可证幂等：同地址、同字节、源不被\n"
       "        // 本段改动 ⇒ 数值逐位不变。⛔ 这一发是尺子、不是杠杆，读到 0 也是结论。\n")


def wrap(anchor, body, tag):
    """把 anchor 段包进 `for (rep) { … }`，返回替换后的整段（含两行说明注释）。"""
    assert src.count(anchor) == 1, "%s 锚点命中 %d 次" % (tag, src.count(anchor))
    return HDR % (tag, N) + "        for (uint32_t p125r = 0u; p125r < %du; ++p125r) {\n" % N + body + "        }\n"


if N == 0:
    io.open(OUT, "w", encoding="utf-8").write(src)
    print("N=0 MARK=0 REP=0（pristine 直拷） FLAG=%d BUFF=%d MULS=%d" % (FLAG0, BUF0, MULS0))
    sys.exit(0)

assert 1 <= N <= 255, "N 要在 1..255"
if ARM == "q":
    src = src.replace(Q_ANCHOR, wrap(Q_ANCHOR, Q_BODY, "prologue"), 1)
elif ARM == "w":
    src = src.replace(W1_ANCHOR, wrap(W1_ANCHOR, W1_BODY, "epilogue-①"), 1)
    src = src.replace(W2_ANCHOR, wrap(W2_ANCHOR, W2_BODY, "epilogue-②"), 1)
else:
    sys.exit("未知 ARM：%s（q|w）" % ARM)

FLAG1 = len(FLAG_RE.findall(src))
REP1 = src.count("p125r = 0u")
MARK1 = src.count("P125 计量器")
want_rep = 1 if ARM == "q" else 2
assert (FLAG1, src.count("InitBuffer"), src.count("Muls("), REP1, MARK1) == \
    (FLAG0, BUF0, MULS0, want_rep, want_rep), \
    "补丁后静态条数变了：%s" % ((FLAG1, src.count("InitBuffer"), src.count("Muls("), REP1, MARK1),)
assert src.count("LoadQ(q, kb, kr,") == 1 and src.count("CopyUb2Gm(outGm_") == COPY0
assert src.count("Muls(oi, oi, 1.0f / l, rowC)") == 1, "原位 Muls 被动了"
if ARM == "w":
    assert src.count("Duplicate(oi, 0.0f, rowC)") == 1
io.open(OUT, "w", encoding="utf-8").write(src)
print("ARM=%s N=%d MARK=%d REP=%d FLAG=%d（应 %d） BUFF=%d MULS=%d bytes=%d"
      % (ARM, N, MARK1, REP1, FLAG1, FLAG0, src.count("InitBuffer"), src.count("Muls("),
         len(src.encode("utf-8"))))
