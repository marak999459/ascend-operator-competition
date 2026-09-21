#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成"AIV 吞吐微基准"版的 SFA kernel（code3.md §15.11）。

裁判的问题：一个 576-MAC 的点积实测要 270~370 cycle，其中"纯 Mul+Cast"就占 ~183 cycle。
  - 解释 A：§11.1 的 fp32 峰值口径（LANES_PER_REP=64 × 1.8 GHz × 40 核 = 9.2 TFLOP/s）
    本身太乐观，向量机实际只有十几 lane ⇒ 我们已经贴真 roofline，只剩 Cube。
  - 解释 B：每条向量指令有 ~20-40 cycle 的发射/停顿开销 ⇒ 把 fold/归约做成宽指令有数倍空间。
两者指向完全相反的路线，所以要量"一条 fp32 Mul 要多少 cycle"。

一次实验分不开 a（每条指令固定开销）和 b（每 rep 边际开销），所以插的是**同一条指令、
三种 calcNum**：跑 3 次构建得 3 个 (reps/条, cycle/条) 点，最小二乘拟合 cycle/条 = a + b·reps。
  calcNum=64   -> 1 rep    calcNum=512 -> 8 reps    calcNum=2048 -> 32 reps
  （LANES_PER_REP=64 是 kernel 自己的常量：256 B / 4 B）

kLoop：若给 `-` 则不插循环，输出干净的提交源（用来恢复远端）。

用法：
  python3 mk_bench_mul.py 512      > /tmp/bench512.cpp
  python3 mk_bench_mul.py -        > /tmp/clean.cpp
只写远端副本，本地提交源不动；跑完 `dev.sh build`（会重新 sync）即恢复。
"""
import io
import os
import sys

REPO = "/home/fszqsn/ops_comp/ascend-operator-competition"
SRC = os.path.join(REPO, "code 3", "code", "op_kernel", "sparse_flash_attention.cpp")

ITERS = 4000
LANES_PER_REP = 64         # = kernel 里的 sfa::LANES_PER_REP
CLOCK_HZ = 1.8e9
MAX_CALCNUM = 4096         # kfBuf_ = SFA_SC_GRP(8) * D_(512) * 4B = 4096 个 fp32

BLOCK = """
        // ==== SFA_BENCH_MULS：AIV 吞吐微基准（只在远端探针副本里，不进提交源）====
        {
            LocalTensor<float> bd = kfBuf_.Get<float>();
            LocalTensor<float> bs = pfBuf_.Get<float>();
            for (uint32_t it = 0; it < SFA_BENCH_MULS; ++it) {
                Mul(bd, bs, bs, SFA_BENCH_N);
            }
            PipeBarrier<PIPE_V>();
        }
"""

ANCHOR = "        uint32_t actQ = 0, actKV = 0;\n"


def main():
    calc = sys.argv[1] if len(sys.argv) > 1 else "512"
    with io.open(SRC, encoding="utf-8") as f:
        text = f.read()
    if calc == "-":
        sys.stdout.write(text)
        sys.stderr.write("输出干净提交源（无探针）\n")
        return 0
    n = int(calc)
    if n > MAX_CALCNUM:
        sys.stderr.write("calcNum %d 越界（kfBuf_ 只有 %d 个 fp32）\n" % (n, MAX_CALCNUM))
        return 1
    if text.count(ANCHOR) != 1:
        sys.stderr.write("锚点不唯一：%d\n" % text.count(ANCHOR))
        return 1
    head = "#define SFA_BENCH_MULS %d\n#define SFA_BENCH_N %du\n" % (ITERS, n)
    out = head + text.replace(ANCHOR, BLOCK + ANCHOR, 1)
    sys.stdout.write(out)
    reps = n // LANES_PER_REP
    sys.stderr.write("已插入 %d 次 Mul(%d) = %d rep/条；"
                     "cycle/条 = (T_probe - T_base) * %.3e / %d；"
                     "拟合需 b = (C32 - C1)/31，a = C1 - b\n"
                     % (ITERS, n, reps, CLOCK_HZ, ITERS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
