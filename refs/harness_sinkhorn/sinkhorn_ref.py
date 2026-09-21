#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mHC-Sinkhorn 参考实现与对拍

用法:
  # 生成输入
  python3 sinkhorn_ref.py gen <batch> <n> <iters> <eps> <out.bin> <ref.bin> [seed] [--f32]

  # 对拍（默认读 fp16 的 /tmp/mhc_out.bin；--f32 读 /tmp/mhc_out_f32.bin）
  python3 sinkhorn_ref.py check <batch> <n> <iters> <eps> <in.bin> <ref.bin> [--f32]

参考实现（与 kernel 逻辑一致）：
  a = logits
  行 softmax: a[i,:] = softmax(a[i,:])       # exp(x - rowmax) / sum
  行归一化:   a[i,:] /= (rowsum_i + eps)
  列归一化:   a[:,k] /= (colsum_k + eps)
  迭代 iters-1 次: 行归一化 -> 列归一化
"""
import struct
import sys
import math
import random


def read_fp16(path, count):
    with open(path, "rb") as f:
        d = f.read()
    vals = list(struct.unpack("<%de" % (len(d) // 2), d))
    assert len(vals) == count, "期望 %d 个，实际 %d" % (count, len(vals))
    return vals


def read_fp32(path, count):
    with open(path, "rb") as f:
        d = f.read()
    vals = list(struct.unpack("<%df" % (len(d) // 4), d))
    assert len(vals) == count, "期望 %d 个，实际 %d" % (count, len(vals))
    return vals


def f16_bits(v):
    b = struct.unpack("<I", struct.pack("<f", v))[0]
    s = (b >> 31) & 1
    e = ((b >> 23) & 0xFF) - 127
    m = b & 0x7FFFFF
    if e < -14:
        return s << 15
    if e > 15:
        return (s << 15) | (0x1F << 10)
    return (s << 15) | ((e + 15) << 10) | (m >> 13)


def f16_from_bits(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]


def gen(batch, n, iters, eps, out_in, out_ref, seed, f32=False):
    random.seed(seed)
    total = batch * n * n
    logits = [round(random.uniform(-4.0, 4.0), 4) for _ in range(total)]
    with open(out_in, "wb") as f:
        f.write(struct.pack("<%df" % total, *logits))

    ref = sinkhorn(logits, batch, n, iters, eps, f32)
    with open(out_ref, "wb") as f:
        if f32:
            f.write(struct.pack("<%df" % total, *ref))
        else:
            f.write(struct.pack("<%dH" % total, *[f16_bits(v) for v in ref]))
    print("生成: %s (%d 元素), 参考: %s dtype=%s"
          % (out_in, total, out_ref, "fp32" if f32 else "fp16"))


def sinkhorn(logits, batch, n, iters, eps, f32=False):
    """在 fp16（默认）或 fp32 语义下模拟 kernel"""
    out = []
    for m in range(batch):
        if f32:
            a = list(logits[m * n * n:(m + 1) * n * n])

            def quant(x):
                return x
        else:
            # fp16 量化输入（模拟 kernel 从 GM 读 fp16）
            a = [f16_from_bits(f16_bits(logits[m * n * n + i])) for i in range(n * n)]

            def quant(x):
                return f16_from_bits(f16_bits(x))

        # 行 softmax
        rowmax = [max(a[i * n + k] for k in range(n)) for i in range(n)]
        for i in range(n):
            for k in range(n):
                a[i * n + k] = quant(math.exp(a[i * n + k] - rowmax[i]))
        rowsum = [sum(a[i * n + k] for k in range(n)) for i in range(n)]
        for i in range(n):
            for k in range(n):
                a[i * n + k] = quant(a[i * n + k] / quant(rowsum[i] + eps))
        colsum = [sum(a[i * n + k] for i in range(n)) for k in range(n)]
        for i in range(n):
            for k in range(n):
                a[i * n + k] = quant(a[i * n + k] / quant(colsum[k] + eps))

        # 交替迭代
        for _ in range(1, iters):
            rowsum = [sum(a[i * n + k] for k in range(n)) for i in range(n)]
            for i in range(n):
                inv = quant(1.0 / quant(rowsum[i] + eps))
                for k in range(n):
                    a[i * n + k] = quant(a[i * n + k] * inv)
            colsum = [sum(a[i * n + k] for i in range(n)) for k in range(n)]
            for i in range(n):
                for k in range(n):
                    a[i * n + k] = quant(a[i * n + k] / quant(colsum[k] + eps))

        out.extend(a)
    return out


def check(batch, n, iters, eps, in_bin, ref_bin, got_bin=None, f32=False):
    total = batch * n * n
    if got_bin is None:
        got_bin = "/tmp/mhc_out_f32.bin" if f32 else "/tmp/mhc_out.bin"
    rd = read_fp32 if f32 else read_fp16
    ref = rd(ref_bin, total)
    got = rd(got_bin, total)

    bad_mats = []
    worst = 0.0
    for m in range(batch):
        blk_r = ref[m * n * n:(m + 1) * n * n]
        blk_g = got[m * n * n:(m + 1) * n * n]
        dev = max(abs(a - b) for a, b in zip(blk_r, blk_g))
        worst = max(worst, dev)
        if dev > 1e-2:
            bad_mats.append((m, dev))

    nbad = len(bad_mats)
    print("batch=%d n=%d iters=%d dtype=%s -> 异常矩阵 %d/%d | 元素级最大偏差 = %.3e"
          % (batch, n, iters, "fp32" if f32 else "fp16", nbad, batch, worst))

    if bad_mats:
        m = bad_mats[0][0]
        print("\n首个异常矩阵 %d（元素级对比）:" % m)
        print("  %-4s %-24s %-24s" % ("idx", "参考", "实际"))
        for i in range(min(n * 2, 12)):
            idx = m * n * n + i
            flag = "  <-- 差异" if abs(ref[idx] - got[idx]) > 1e-2 else ""
            print("  %-4d %-24.8f %-24.8f%s" % (i, ref[idx], got[idx], flag))
        # 判断末尾是否整体为 0
        blk_g = got[m * n * n:(m + 1) * n * n]
        zeros = sum(1 for x in blk_g if x == 0.0)
        print("\n  该矩阵零元素个数: %d / %d" % (zeros, n * n))
    else:
        print("PASS: 全部矩阵与参考一致")
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    F32 = "--f32" in sys.argv
    a = [x for x in sys.argv if x != "--f32"]
    if len(a) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = a[1]
    if cmd == "gen":
        gen(int(a[2]), int(a[3]), int(a[4]), float(a[5]), a[6], a[7],
            int(a[8]) if len(a) > 8 else 1234, F32)
    elif cmd == "check":
        sys.exit(check(int(a[2]), int(a[3]), int(a[4]), float(a[5]), a[6], a[7], None, F32))
    else:
        print(__doc__)
        sys.exit(1)
