#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P118 探针补丁：每个 chunk 额外发 `N*m` 组 gather（每组 3 条、与真搬运同宽度），
源行可选**冷**（本 batch KV 池里伪随机行）或**暖**（本 chunk 刚读过的那一行）⇒ 二者只差冷暖。

为什么是这个体积：一组 = 512 fp16(k) + 64 fp16(kr) + 512 fp16(v) = 3 条 / 2176 B，
N=1 时**与 P114 加的量逐条同尺寸**（那条臂在平台上的读数 = 17.8 %）。于是
  Δ118(cold) − Δ118(warm)      = 纯冷溢价（同位置、同条数、同字节、同 dst，只有冷暖不同）
  Δ118(warm) vs 17.8 % (P114)  = 位置差的量级（P114 的拷贝插在 MTE2 窗口里，这里在 chunk 尾）

锚点：`FlushChunk` 末尾，即 `SoftmaxPv(...)` + `SetFlag/WaitFlag<V_MTE2>(0)` 之后。

为什么逐位惰性（三条都要成立）：
  · 目的：三条写进 `kfBuf_` 的前 1024/1024 B 与 `krfBuf_` 的前 128 B（同 dst 反复覆写，反正没人读）。
    `kfBuf_`/`krfBuf_` 在 `SoftmaxPv` 之后是死内存，下一 chunk 的 `ComputeScores` 第一件事就是
    `WidenToF32(kf/krf, kb/kr[…], g*D_/g*Dr_)` 覆写 `[0, g*D_)`（g = m，因主循环
    `while (cnt < nBlk_)` 保证 m ≤ nBlk_ = scGrp_）⇒ 前 2048/256 B 必被重新写下，我们只碰 1024/128 B。
  · 源：只读 `kGm_/krGm_/vGm_`，行号由本地计数器算出、不进任何输出下标。
  · 管道：写在 `V_MTE2(0)` 之后 ⇒ 不与本 chunk 的 V 读竞争；下一 chunk 的 `WaitFlag<MTE2_V>(0)`
    排在 `ComputeScores` 之前，MTE2 队列按序 ⇒ 写必在 V 覆写 kfBuf_ 前退休。
  · ⛔ 目的绝不能换成 `vb`（= `kBuf_`）：V 补搬只覆盖 `[0, m*D_)`，`kfBuf_` 的前 1024 B 会被读到。
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MODE = (sys.argv[2] if len(sys.argv) > 2 else "cold").lower()
PATH = sys.argv[3] if len(sys.argv) > 3 else "code/op_kernel/sparse_flash_attention.cpp"
PRISTINE = sys.argv[4] if len(sys.argv) > 4 else "../kernel_pristine.cpp"
assert 1 <= N <= 4, "N must be 1..4"
assert MODE in ("cold", "warm"), "MODE must be cold|warm"

TAIL = ("        SoftmaxPv(o, vb, sc, ml, nbCur, m);\n"
        "        SetFlag<HardEvent::V_MTE2>(0);\n"
        "        WaitFlag<HardEvent::V_MTE2>(0);\n"
        "    }\n")
DECL = "    TBuf<TPosition::VECCALC> kfBuf_, krfBuf_, rdBuf_;\n"
# 冷：xorshift32 混淆后乘移取行号（**不带除法**），天然落在 [0, S2_) 内、每 chunk 互不相同。
# 暖：本 chunk 第一段的首行 —— K 在主循环搬过、V 在 FlushChunk 里刚搬完，必然还在缓存里。
ROW = ("                const uint32_t hP118 = (ctrP118_ * 2654435761u + i * 40503u) | 1u;\n"
       "                const uint32_t kP118 = (hP118 >> 15) ^ hP118;\n"
       "                const uint32_t wP118 = (kP118 >> 13) ^ kP118;\n"
       "                const int64_t rowP118 = static_cast<int64_t>(\n"
       "                    (static_cast<uint64_t>(wP118) * S2_) >> 32);\n") if MODE == "cold" else (
       "                const int64_t rowP118 = stageBeg_[0];\n")

src = open(PRISTINE, "r", encoding="utf-8").read()
for a in (TAIL, DECL):
    assert src.count(a) == 1, "anchor hits=%d: %r" % (src.count(a), a[:48])
assert src.count("SoftmaxPv(o, vb, sc, ml, nbCur, m);") == 1, "SoftmaxPv 调用点必须唯一"
assert src.count("int32_t stageBeg_[SFA_STAGE_MAX];") == 1, "stageBeg_ 必须存在（暖臂的锚）"
assert "P118" not in src, "pristine 里已有 P118 残留"

src = src.replace(DECL, DECL + "    uint32_t ctrP118_ = 0u;   // P118 gather meter\n", 1)
body = (
    "        // ---- P118 gather meter：%s × %d*m 组，每组 3 条与真 gather 同宽 ----\n"
    "        {\n"
    "            ++ctrP118_;\n"
    "            LocalTensor<DT_QUERY> dkP118 = kfBuf_.Get<DT_QUERY>();\n"
    "            LocalTensor<DT_QUERY> drP118 = krfBuf_.Get<DT_QUERY>();\n"
    "            const uint32_t nP118 = static_cast<uint32_t>(m) * %du;\n"
    "            for (uint32_t i = 0u; i < nP118; ++i) {\n"
    "%s"
    "                CopyGm2Ub(dkP118, kGm_[(rowBase + rowP118) * D_], D_);\n"
    "                CopyGm2Ub(drP118, krGm_[(rowBase + rowP118) * Dr_], Dr_);\n"
    "                CopyGm2Ub(dkP118, vGm_[(rowBase + rowP118) * D_], D_);\n"
    "            }\n"
    "            SetFlag<HardEvent::MTE2_V>(2);\n"
    "            WaitFlag<HardEvent::MTE2_V>(2);\n"
    "        }\n") % ("冷行" if MODE == "cold" else "暖行", N, N, ROW)
src = src.replace(TAIL, TAIL[:-len("    }\n")] + body + "    }\n", 1)

open(PATH, "w", encoding="utf-8").write(src)
print("N=%d MODE=%s  MARK=%d/%d  GROUPS/chunk=%d  CALLS/chunk=%d"
      % (N, MODE, src.count("P118 gather meter"), 2, N, 3 * N))
