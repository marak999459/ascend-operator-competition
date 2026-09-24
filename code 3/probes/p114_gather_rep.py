#!/usr/bin/env python3
# P114 探针补丁：把每 chunk 的三条 gather DataCopy（K、K_rope、V）原地再发 (N-1) 遍。
#
# 为什么惰性：重发的 src/dst/长度与原有那条**逐字符相同** ⇒ 写回 UB 的字一个字没变，
# 输出逐位不变 ⇒ 这是一个纯 MTE2 搬运量的扰动，与 P110/P112 的计算量扰动同构：
#   t1 = G + R（G = gather 段，R = 其余墙）
#   tN = G·N + R  ⇒  G/t1 = (tN − t1)/((N − 1)·t1)
# ⇒ 同一份 Δ% 在本地读"本地靶族的 gather 占比"，在平台读"平台六点的 gather 占比"，
#   两边一对就知道 P112 那个"score 段只占 6 %"的落差到底是不是搬运侧的不对称。
# ⚠️ 只作用在远端副本（从 pristine 重新生成，不叠补丁）。本地提交源由 p114_gate.sh 铺。
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PATH = sys.argv[2] if len(sys.argv) > 2 else "code/op_kernel/sparse_flash_attention.cpp"
PRISTINE = sys.argv[3] if len(sys.argv) > 3 else "../kernel_pristine.cpp"

src = open(PRISTINE, "r", encoding="utf-8").read()
ANCHORS = [
    "                CopyGm2Ub(kb[done * D_],  kGm_[kOff],  run * D_);\n",
    "                CopyGm2Ub(kr[done * Dr_], krGm_[rOff], run * Dr_);\n",
    "                CopyGm2Ub(vb[done * D_], vGm_[(rowBase + stageBeg_[j]) * D_], run * D_);\n",
]
for a in ANCHORS:
    assert src.count(a) == 1, "anchor hits=%d for %r" % (src.count(a), a[:60])
if N > 1:
    for a in ANCHORS:
        extra = "".join(a.rstrip("\n") + "  // P114 rep%d\n" % (r + 2) for r in range(N - 1))
        src = src.replace(a, a + extra, 1)
open(PATH, "w", encoding="utf-8").write(src)
print("REP=%d  HITS_P114=%d  EXPECT=%d" % (N, src.count("P114 rep"), 3 * (N - 1)))
