#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成"AIV 归约吞吐微基准"版的 SFA kernel（code3.md §15.12）。

§15.11 量出 `Mul` 的成本 = 29.0 + 1.12 × rep。但 P7 的两级归约只换来 8.5%，说明
`WholeReduceSum`（硬件 vcadd）每个 repeat 远不止 1.12 cycle。这个脚本就是在 ProcessToken
入口插两段**定长**归约循环，反解出 reduce 的 cycle/repeat：

    A: WholeReduceSum(dst, src, mask=64 lane, repeatTime=R_A, 1, 1, 8)   × ITERS 次
    B: 同样的调用，repeatTime=R_B

同一条指令、两个 repeat 数 ⇒ cycle/条 = 28.98 + X × R，两个方程任选其一都能解 X；
两段放在**同一次构建**里（各自 ITERS 次），用总 Δ 反解，省一次构建。

dst 借 kfBuf_（16 KB，纯 scratch，探针跑完立刻被真实流程重写），src 借 pfBuf_。
用法： python3 mk_bench_reduce.py > /tmp/bench_red.cpp      # 只写远端副本
      python3 mk_bench_reduce.py - > /tmp/clean.cpp          # 输出干净提交源
"""
import io
import os
import sys

REPO = "/home/fszqsn/ops_comp/ascend-operator-competition"
SRC = os.path.join(REPO, "code 3", "code", "op_kernel", "sparse_flash_attention.cpp")

ITERS = 2000
RA = 8
RB = 64
CLOCK_HZ = 1.8e9
A_OVERHEAD = 28.98     # §15.11 量出来的每条向量调用固定开销

BLOCK = """
        // ==== SFA_BENCH_RED：归约吞吐微基准（只在远端探针副本里，不进提交源）====
        {
            LocalTensor<float> rdst = kfBuf_.Get<float>();
            LocalTensor<float> rsrc = pfBuf_.Get<float>();
            for (uint32_t it = 0; it < SFA_BENCH_RED; ++it) {
                WholeReduceSum(rdst, rsrc, 64u, %d, 1, 1, 8);
                WholeReduceSum(rdst, rsrc, 64u, %d, 1, 1, 8);
            }
            PipeBarrier<PIPE_V>();
        }
""" % (RA, RB)

ANCHOR = "        uint32_t actQ = 0, actKV = 0;\n"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "go"
    with io.open(SRC, encoding="utf-8") as f:
        text = f.read()
    if mode == "-":
        sys.stdout.write(text)
        sys.stderr.write("输出干净提交源（无探针）\n")
        return 0
    if text.count(ANCHOR) != 1:
        sys.stderr.write("锚点不唯一：%d\n" % text.count(ANCHOR))
        return 1
    head = "#define SFA_BENCH_RED %d\n" % ITERS
    sys.stdout.write(head + text.replace(ANCHOR, BLOCK + ANCHOR, 1))
    calls = ITERS * 2
    sys.stderr.write(
        "已插入 %d 次 ×2 条归约（R=%d 与 R=%d）；每条固定开销 %.2f\n"
        "  cycle/条 平均值 = Δ * %.3e / %d\n"
        "  X = (C_平均 - %.2f) / ((%d+%d)/2)  -> reduce 每 repeat 的 cycle\n"
        % (ITERS, RA, RB, A_OVERHEAD, CLOCK_HZ, calls, A_OVERHEAD, RA, RB))
    return 0


if __name__ == "__main__":
    sys.exit(main())
