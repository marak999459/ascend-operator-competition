#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P119 探针补丁：把 `SoftmaxPv` 第 5 段（PV 累加）的 j 循环体**原地重跑 N 遍**，
但**目的端改指一块死 scratch** ⇒ 累加器 `o` 一个字节都不动 ⇒ 对输出逐位惰性。

为什么"带累加器的段"这次能被复制（`#15.96` 那句"只能靠消融"是错的）：
  幂等只是"复制同一段"的**充分**条件。真实那段是 `o_i += p_ij * v_j`；复制体写成
  `sk += p_ij * v_j`（sk 全程没人读）⇒ 指令条数、每条宽度、标量读次数、源操作数与真那段
  **逐项相同**，唯一的区别是目的寄存器不同 + 结果作废。⇒ 一段一次发次同时定价两个候选：
    ① `Axpy` 本身（本地：每 head·token 64 条，占 AIV 31~34 %）；
    ② 每条 `Axpy` 前那一条 **97 ns 标量 UB 读** `p.GetValue(i*nBlk_+j)`（§4 的反模式，从没在平台量过）。

为什么逐位惰性（三条都要成立）：
  · 目的：`sk = vf[(scGrp_-1)*rowC]` = `kfBuf_` 最后一行 fp32（在分配界内：`kfBuf_` 恰好
    `scGrp_*D_` 个 float）。`kfBuf_` 在本函数返回后是死内存，而它后面**每一个**读者都是先整块覆写再读：
    `ComputeScores` 的 `WidenToF32(kf, …)`、`WriteOut`/`MergeToken` 的 `PackFromF32(st, …)`、
    `ZeroPaddingOut` 的 `Duplicate(zf, …)`。同 chunk 内 §5 之后无人读它（§5 是 `SoftmaxPv` 最后一段）。
  · 源：只读 `vf`（本组 V 的 fp32 展开）与 `p`（本 chunk 的 P），都不被写。
    ⚠️ 复制体会把 `sk` 那一行（= `t = scGrp_-1` 时的 `vt`）越写越脏 ⇒ **只影响复制体自己的结果**，
    真那段已经在它之前跑完 ⇒ 不影响 `o`。
  · 环 / tiling / UB 预算：**一个新 buffer 都不加**（scratch 借现有 `kfBuf_`）⇒ 只有一个变量在动。

用法：p119_pvrep.py N DST PRISTINE     （N = 额外重跑几遍，0 = 不铺补丁直接拷 pristine）
"""
import shutil
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PATH = sys.argv[2] if len(sys.argv) > 2 else "code/op_kernel/sparse_flash_attention.cpp"
PRISTINE = sys.argv[3] if len(sys.argv) > 3 else "../kernel_pristine.cpp"
assert 0 <= N <= 6, "N must be 0..6"

VF = "        LocalTensor<float> vf = kfBuf_.Get<float>();   // ComputeScores 之后即空闲，借一组\n"
TAIL = ("                for (uint32_t i = 0; i < nbCur; ++i) {\n"
        "                    Axpy<float, float>(o[i * rowC], vt, p.GetValue(i * nBlk_ + j), rowC);\n"
        "                }\n"
        "            }\n"
        "        }\n")

if N == 0:
    shutil.copyfile(PRISTINE, PATH)
    print("N=0 MARK=0/0（pristine 直拷）")
    sys.exit(0)

src = open(PRISTINE, "r", encoding="utf-8").read()
for a in (VF, TAIL):
    assert src.count(a) == 1, "anchor hits=%d: %r" % (src.count(a), a[:48])
assert src.count("Axpy<float, float>(o[i * rowC], vt,") == 1, "PV 内层必须唯一"
assert "P119" not in src, "pristine 里已有 P119 残留"

src = src.replace(VF, VF + "        LocalTensor<float> skP119 = vf[(scGrp_ - 1u) * rowC];   // P119 PV meter\n", 1)
body = (
    "            // ---- P119 PV meter：同样 %d 遍 `nbCur×gv` 条 Axpy + 同样多次标量 p 读，目的端 = kfBuf_ 末行 ----\n"
    "            for (uint32_t rP119 = 0u; rP119 < %du; ++rP119) {\n"
    "                for (uint32_t t = 0; t < gv; ++t) {\n"
    "                    const LocalTensor<float> vt2 = vf[t * rowC];\n"
    "                    const uint32_t j2 = j0 + t;\n"
    "                    for (uint32_t i = 0; i < nbCur; ++i) {\n"
    "                        Axpy<float, float>(skP119, vt2, p.GetValue(i * nBlk_ + j2), rowC);\n"
    "                    }\n"
    "                }\n"
    "            }\n") % (N, N)
# TAIL 的最后两行是 t 循环与 j0 循环的收尾；计量器要落在两者之间
src = src.replace(TAIL, TAIL[:-len("        }\n")] + body + "        }\n", 1)
open(PATH, "w", encoding="utf-8").write(src)
print("N=%d MARK=%d/%d  bytes=%d" % (N, src.count("P119 PV meter"), 2, len(src.encode("utf-8"))))
