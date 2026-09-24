#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P115 探针补丁：把每 chunk 的三条 gather 各**等分成 N 条**（同总字节、调用数 ×N、逐位惰性）。

与 P114（字节 ×2、调用 ×2）配对才读得出"那 17.8 % 里字节与调用各占多少"：
  · P114 = 字节 ×2、调用 ×2  ⇒ Δ114 ≈ 字节成本 + 调用成本
  · P115 = 字节 ×1、调用 ×(N−1) ⇒ Δ115 只值调用成本
  ⇒ 平台若 Δ115 ≈ 0 ⇒ 平台的 gather 是**字节/带宽**限制 ⇒ 少搬才有的赚（跨行去重那一档大事）；
    平台若 Δ115 明显非零 ⇒ 是**命令发射**限制 ⇒ 反向操作"整段超读换长拷贝"就有肉。
为什么惰性：拆分点落在同一批元素上（`h` 取到 16 个元素 = 32 B 的整数倍，两端仍 256 bit 对齐），
  两条半长写回的地址区间与原条**完全相同** ⇒ UB 里的字一个字没变。
⚠️ 只作用在远端副本（从 pristine 重生成，不叠补丁）；本地提交源由 p115_gate.sh 铺。
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PATH = sys.argv[2] if len(sys.argv) > 2 else "code/op_kernel/sparse_flash_attention.cpp"
PRISTINE = sys.argv[3] if len(sys.argv) > 3 else "../kernel_pristine.cpp"
# N>4 会把 kr 的一段（run=1 ⇒ 64 元素）切成 pu_=0 ⇒ DataCopy blockCount=0 直接挂死
assert N in (1, 2, 3, 4), "N must be 1..4 (got %d)" % N

SITES = [  # (原句, dst 表达式, src 表达式, 元素数的表达式)
    ("                CopyGm2Ub(kb[done * D_],  kGm_[kOff],  run * D_);\n",
     "kb[done * D_]", "kGm_[kOff]", "run * D_"),
    ("                CopyGm2Ub(kr[done * Dr_], krGm_[rOff], run * Dr_);\n",
     "kr[done * Dr_]", "krGm_[rOff]", "run * Dr_"),
    ("                CopyGm2Ub(vb[done * D_], vGm_[(rowBase + stageBeg_[j]) * D_], run * D_);\n",
     "vb[done * D_]", "vGm_[(rowBase + stageBeg_[j]) * D_]", "run * D_"),
]

src = open(PRISTINE, "r", encoding="utf-8").read()
for line, dst, gsrc, nelem in SITES:
    assert src.count(line) == 1, "anchor hits=%d: %r" % (src.count(line), line[:52])
if N > 1:
    ind = " " * 16
    for line, dst, gsrc, nelem in SITES:
        # dst/gsrc 现在是 "kb[done * D_]" / "kGm_[kOff]" 这种下标式，追加偏移走同一个下标，
        # ⛔ 不用 operator+ —— 下标式是本文件已在用的形态，不引入新假设。
        assert dst.endswith("]") and gsrc.endswith("]"), dst
        dbase, gbase = dst[:-1], gsrc[:-1]
        blk = ("%s{\n" % ind
               + "%s    const uint32_t pn_ = %s;                     // P115 split x%d\n" % (ind, nelem, N)
               + "%s    const uint32_t pu_ = (pn_ / %du) & ~15u;     // 每段元素数，保持 32B 倍数\n" % (ind, N)
               + "%s    for (uint32_t q_ = 0u; q_ < %du - 1u; ++q_) {\n" % (ind, N)
               + "%s        CopyGm2Ub(%s + q_ * pu_], %s + q_ * pu_], pu_);\n" % (ind, dbase, gbase)
               + "%s    }\n" % ind
               + "%s    CopyGm2Ub(%s + (%du - 1u) * pu_], %s + (%du - 1u) * pu_], pn_ - (%du - 1u) * pu_);\n"
                 % (ind, dbase, N, gbase, N, N)
               + "%s}\n" % ind)
        src = src.replace(line, blk, 1)
open(PATH, "w", encoding="utf-8").write(src)
print("SPLIT=%d  HITS_P115=%d  EXPECT=%d" % (N, src.count("P115 split"), 3 if N > 1 else 0))
