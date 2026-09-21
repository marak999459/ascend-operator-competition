#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成"fp32 模板实例自检"探针版 SFA kernel（code3.md §15.13）。

`SFA_F32=1` 真机首验（harness 把同一批用例的输入按 float 送进去）发现 DT_QUERY=float
那条实例的输出与参考完全不符。这个探针在 `ProcessToken` 最前面插一段自报家门，
一次构建回答三个问题：

  Q1 配置：fp32 实例真正拿到的 tiling/成员值（nb_/nBlk_/stageMax_/redW_/D_/Dr_/N1_/S2_
     以及 scBatch_ 与 sizeof(DT_QUERY)）—— 验证"host 在 fp32 下选 (8,16)"是不是真的。
  Q2 原语：`Cast<float>(dst, src<float>, CAST_RINT, n)`（即 WidenToF32 在 fp32 实例里
     走的那条）到底是不是恒等搬运。写 0..63 进去，读回来对不对得上。
  Q3 搬运：`DataCopy`(GM->UB) + 之后的 `DataCopy`(UB->GM) 在 4 字节元素上是否完整，
     直接把 query 的第 0 行 16 个元素原样搬到输出区。

输出布局（写进 attentionOut 的第 0 行起，用 float/half 各自的元素类型）：
  [0..15]   Q1 配置
  [16..79]  Q2 原语回读（期望 0,1,2,...,63）
  [80..95]  Q3 query 前 16 个元素

只在远端副本里跑（探针不进提交源）：
    python3 mk_probe_f32cfg.py > /tmp/probe_f32.cpp     # 打探针的远端副本
    python3 mk_probe_f32cfg.py - > /tmp/clean.cpp       # 输出干净提交源
"""
import io
import sys

REPO = "/home/fszqsn/ops_comp/ascend-operator-competition"
SRC = REPO + "/code 3/code/op_kernel/sparse_flash_attention.cpp"

BLOCK = """
#ifdef SFA_PROBE_F32CFG
        // ==== 自检探针（只在远端副本里，见 code 3/probes/mk_probe_f32cfg.py）====
        if (GetBlockIdx() == 0u && tok == tokBegin_) {
            LocalTensor<DT_QUERY> stage = kBuf_.Get<DT_QUERY>();
            LocalTensor<float> aa = pfBuf_.Get<float>();
            LocalTensor<float> bb = krfBuf_.Get<float>();
            LocalTensor<DT_QUERY> qs = vBuf_.Get<DT_QUERY>();
            for (uint32_t i = 0; i < 64u; ++i) { aa.SetValue(i, static_cast<float>(i)); }
            WidenToF32(bb, aa, 64);                                 // Q2：fp32 实例里是 float->float
            PipeBarrier<PIPE_V>();                                  // 标量读 V 写的区之前先对齐流水线
            CopyGm2Ub(qs, qGm_[0u], 16u);                           // Q3：GM->UB
            PipeBarrier<PIPE_ALL>();                                // MTE2 写完才许标量读
            // Q1：配置自报
            float rep[16];
            rep[0] = static_cast<float>(nb_);
            rep[1] = static_cast<float>(nBlk_);
            rep[2] = static_cast<float>(stageMax_);
            rep[3] = static_cast<float>(redW_);
            rep[4] = static_cast<float>(D_);
            rep[5] = static_cast<float>(Dr_);
            rep[6] = static_cast<float>(N1_);
            rep[7] = static_cast<float>(S2_);
            rep[8] = scBatch_ ? 1.0f : 0.0f;
            rep[9] = static_cast<float>(sizeof(DT_QUERY));
            rep[10] = static_cast<float>(nHeadBlk_);
            rep[11] = static_cast<float>(sbs_);
            rep[12] = static_cast<float>(sparseCount_);
            rep[13] = 0.0f;
            rep[14] = 0.0f;
            rep[15] = 0.0f;
            for (uint32_t i = 0; i < 16u; ++i) { stage.SetValue(i, static_cast<DT_QUERY>(rep[i])); }
            for (uint32_t i = 0; i < 64u; ++i) { stage.SetValue(16u + i, static_cast<DT_QUERY>(bb.GetValue(i))); }
            for (uint32_t i = 0; i < 16u; ++i) { stage.SetValue(80u + i, qs.GetValue(i)); }
            PipeBarrier<PIPE_ALL>();
            CopyUb2Gm(outGm_[0u], stage, 96u);
            PipeBarrier<PIPE_ALL>();
            return;
        }
#endif
"""

ANCHOR = "    __aicore__ inline void ProcessToken(uint32_t b, uint32_t s)\n    {\n"


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
    sys.stdout.write("#define SFA_PROBE_F32CFG 1\n" + text.replace(ANCHOR, ANCHOR + BLOCK, 1))
    sys.stderr.write("已插入自检探针：out[0..15]=配置 [16..79]=Cast 回读 [80..95]=query 前 16\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
