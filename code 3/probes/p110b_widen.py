#!/usr/bin/env python3
# P110b：给"每头重做一次 K/K-rope 加宽"这件事定价（ComputeScores 里 `for i in nbCur` 那两条
# WidenToF32 —— 就地折叠把 kf 用掉了，所以 nb=4 时同一份 K 要从 fp16 加宽 4 遍）。
# 两臂互相校验：
#   wid2   再加宽一遍（写回同一块 kf ⇒ 逐位惰性）⇒ 读出**一遍加宽的边际成本** S_w
#   hoist  i>0 时干脆不加宽（**输出必错**，只看时间）⇒ 直接读出"若加宽提到头循环外"能省的**上界**
#          预测：hoist 省下的 ≈ (nbCur-1)·S_w ⇒ 两臂对不上就说明加宽与后面的 Mul 之间有重叠，
#          那"省下来的时间"并不真能省 ⇒ 这一条比单读 hoist 更老实。
# ⚠️ 只作用在远端副本，且每臂都从 ../kernel_pristine.cpp 重新生成。
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "rep1"
PATH = sys.argv[2] if len(sys.argv) > 2 else "code/op_kernel/sparse_flash_attention.cpp"
PRISTINE = sys.argv[3] if len(sys.argv) > 3 else "../kernel_pristine.cpp"

src = open(PRISTINE, "r", encoding="utf-8").read()
A1 = "                WidenToF32(kf, kb[g0 * rowC], static_cast<int32_t>(g * rowC));\n"
A2 = "                WidenToF32(krf, kr[g0 * rowR], static_cast<int32_t>(g * rowR));\n"
assert src.count(A1) == 1 and src.count(A2) == 1, "widen anchors %d/%d" % (src.count(A1), src.count(A2))

if MODE == "rep1":
    pass
elif MODE == "wid2":
    src = src.replace(A1 + A2, A1 + A2 +
                      "                " + A1.strip() + "  // P110b wid2\n"
                      "                " + A2.strip() + "  // P110b wid2\n", 1)
elif MODE == "hoist":
    src = src.replace(A1 + A2,
                      "                if (i == 0u) {   // P110b hoist：输出必错，只看时间\n"
                      + A1 + A2 + "                }\n", 1)
else:
    raise SystemExit("MODE ∈ {rep1,wid2,hoist}")

open(PATH, "w", encoding="utf-8").write(src)
print("MODE=%s  HITS=%d" % (MODE, src.count("P110b")))
