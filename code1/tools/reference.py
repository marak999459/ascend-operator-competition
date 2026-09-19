#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mHC-expand 算子参考实现与精度对拍脚本

赛题：【B组简单题】mHC-expand 算子（前向与反向）

形状约定:
    前向 (backward=False):  x [S, D]    -> o       [S, m, D]
    反向 (backward=True) :  x [S, m, D] -> x_grad  [S, D]

公式:
    前向: o[s, k, j] = x[s, j]                    对所有 k in [0, m)
    反向: x_grad[s, j] = sum_k o_grad[s, k, j]      <-- 求和（赛题口径，非平均）

用法:
    # 仅生成用例并打印参考结果统计（无 NPU 也能跑）
    python reference.py --list
    python reference.py --selftest

    # 导出用例到磁盘，供算子工程读取
    python reference.py --export ./cases

    # 把 NPU 上跑出的结果与参考对拍
    python reference.py --check-forward  ./cases/case_00_o.npy      --case 0
    python reference.py --check-backward ./cases/case_00_xgrad.npy  --case 0

注意:
    本脚本的 backward 默认使用 SUM（赛题口径）。若平台按 vLLM 的 hc_contract
    口径（mean）判定，用 --mean 切换，脚本会同时报告两者的差异倍数，便于快速定位。
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import torch
except ImportError:  # pragma: no cover
    sys.stderr.write("需要 PyTorch：pip install torch\n")
    raise


# --------------------------------------------------------------------------
# 用例矩阵
# --------------------------------------------------------------------------
# 覆盖: 常规大 shape / 非对齐 D / 非对齐 S / 极小 shape / 单 token / m=1
CASES = [
    # (name,               S,    D,    m)
    ("example_small",       4,    8,   2),   # 赛题示例1/2
    ("example_large",    4096, 7168,   4),   # 赛题示例3（大模型典型规模）
    ("unaligned_d",         7,  100,   2),   # D 非 16 倍数 -> 行首非 32B 对齐
    ("unaligned_d_odd",  1000, 7167,   3),   # D 奇数且非对齐
    ("unaligned_s",        13,  128,   4),   # S 非 align/非核数倍数
    ("tiny",                1,    1,   2),   # 极小
    ("single_token",        1, 4096,   4),   # 单 token，S 方向无法切核
    ("m_one",             128,  256,   1),   # m=1 退化为纯拷贝
    ("m_large",            64,  512,  16),   # m 很大
    ("d_one",              32,    1,   4),   # D=1，完全非对齐
    ("d_align_16",         64,   16,   8),   # D 恰好 16
    ("bandwidth",        8192, 7168,   4),   # 带宽压测
]

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# --------------------------------------------------------------------------
# 参考实现
# --------------------------------------------------------------------------
def ref_forward(x: "torch.Tensor", m: int) -> "torch.Tensor":
    """[S, D] -> [S, m, D]，第 1 维复制 m 份。"""
    s, d = x.shape
    assert m >= 1, f"mhc_mult 必须 >= 1, got {m}"
    # unsqueeze(1).expand(...).clone() 与赛题示例完全一致
    return x.unsqueeze(1).expand(s, m, d).clone()


def ref_backward_sum(o_grad: "torch.Tensor", m: int) -> "torch.Tensor":
    """[S, m, D] -> [S, D]，沿第 1 维求和。赛题口径。"""
    assert o_grad.dim() == 3, f"backward 输入应为 3 维 [S, m, D], got {o_grad.shape}"
    assert o_grad.shape[1] == m, f"输入第 1 维 {o_grad.shape[1]} 与 mhc_mult={m} 不一致"
    return o_grad.sum(dim=1)


def ref_backward_mean(o_grad: "torch.Tensor", m: int) -> "torch.Tensor":
    """[S, m, D] -> [S, D]，沿第 1 维求平均。vLLM hc_contract 口径，仅用于排查评分口径。"""
    return o_grad.mean(dim=1)


def ref_backward_fp32(o_grad: "torch.Tensor", m: int) -> "torch.Tensor":
    """
    [S, m, D] -> [S, D]，FP32 域求和后转回原 dtype。

    这是 Ascend kernel 应当采用的数值策略（kernel 内 FP32 累加）。
    当评测为逐 bit 比对时，应以本函数为参考，而不是 ref_backward_sum。
    """
    return ref_backward_sum(o_grad.to(torch.float32), m).to(o_grad.dtype)


# --------------------------------------------------------------------------
# 用例构造
# --------------------------------------------------------------------------
def make_case(s: int, d: int, m: int, dtype: "torch.dtype", seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(s, d, dtype=torch.float32, generator=g).to(dtype)
    o_grad = torch.randn(s, m, d, dtype=torch.float32, generator=g).to(dtype)
    return x, o_grad


def check_parity(x, o, label, rtol=1e-3, atol=1e-3):
    """比较两个张量，返回 (是否通过, 描述)。"""
    if tuple(o.shape) != tuple(x.shape):
        return False, f"{label}: 形状不一致 ref={tuple(x.shape)} got={tuple(o.shape)}"
    a = x.to(torch.float32)
    b = o.to(torch.float32)
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    max_abs = (a - b).abs().max().item() if a.numel() else 0.0
    return ok, f"{label}: allclose={ok} max_abs_diff={max_abs:.3e}"


# --------------------------------------------------------------------------
# 自检：验证参考实现本身自洽
# --------------------------------------------------------------------------
def cmd_selftest(args) -> int:
    print("=" * 78)
    print("参考实现自检")
    print("=" * 78)
    failures = 0
    for name, s, d, m in CASES:
        for dt_name, dt in DTYPES.items():
            x, o_grad = make_case(s, d, m, dt)

            # --- 前向 ---
            o = ref_forward(x, m)
            exp_shape = (s, m, d)
            ok_shape = tuple(o.shape) == exp_shape
            # 每个副本必须与输入逐 bit 相同
            ok_copy = all(torch.equal(o[:, k, :], x) for k in range(m))

            # --- 反向 ---
            xg_sum = ref_backward_sum(o_grad, m)
            xg_mean = ref_backward_mean(o_grad, m)
            ok_bwd_shape = tuple(xg_sum.shape) == (s, d)
            # sum 应为 mean 的 m 倍
            ok_scale = torch.allclose(
                xg_sum.to(torch.float32),
                (xg_mean.to(torch.float32) * m),
                rtol=1e-3, atol=1e-3,
            )
            # 往返一致性: sum_k expand(x) == m * x
            ok_roundtrip = torch.allclose(
                ref_forward(x, m).to(torch.float32).sum(dim=1),
                x.to(torch.float32) * m,
                rtol=1e-3, atol=1e-3,
            )

            good = ok_shape and ok_copy and ok_bwd_shape and ok_scale and ok_roundtrip
            if not good:
                failures += 1
            flag = "OK  " if good else "FAIL"
            print(f"[{flag}] {name:<14} {dt_name:<4} S={s:<5} D={d:<5} m={m:<3} "
                  f"fwd_shape={ok_shape} fwd_exact={ok_copy} "
                  f"bwd_shape={ok_bwd_shape} sum==m*mean={ok_scale} roundtrip={ok_roundtrip}")

    print("-" * 78)
    print(f"失败用例数: {failures}")
    return 1 if failures else 0


# --------------------------------------------------------------------------
# 列出用例
# --------------------------------------------------------------------------
def cmd_list(args) -> int:
    print(f"{'idx':<4} {'name':<16} {'S':>6} {'D':>6} {'m':>3}   forward        backward")
    print("-" * 78)
    for i, (name, s, d, m) in enumerate(CASES):
        print(f"{i:<4} {name:<16} {s:>6} {d:>6} {m:>3}   "
              f"[{s},{d}]->[{s},{m},{d}]   [{s},{m},{d}]->[{s},{d}]")
    print("-" * 78)
    print(f"共 {len(CASES)} 个 shape 用例 × {len(DTYPES)} 种 dtype = {len(CASES)*len(DTYPES)} 组")
    return 0


# --------------------------------------------------------------------------
# 导出用例（供算子工程 / NPU 上板读取）
# --------------------------------------------------------------------------
def cmd_export(args) -> int:
    outdir = args.export
    os.makedirs(outdir, exist_ok=True)
    meta_lines = []
    n = 0
    for i, (name, s, d, m) in enumerate(CASES):
        for dt_name, dt in DTYPES.items():
            x, o_grad = make_case(s, d, m, dt, seed=i)

            o_ref = ref_forward(x, m)
            xg_ref = ref_backward_sum(o_grad, m)
            xg_ref_fp32 = ref_backward_fp32(o_grad, m)

            stem = f"case{i:02d}_{name}_{dt_name}"
            torch.save(x, os.path.join(outdir, f"{stem}_x.pt"))
            torch.save(o_grad, os.path.join(outdir, f"{stem}_ograd.pt"))
            torch.save(o_ref, os.path.join(outdir, f"{stem}_o_ref.pt"))
            torch.save(xg_ref, os.path.join(outdir, f"{stem}_xgrad_ref.pt"))
            torch.save(xg_ref_fp32, os.path.join(outdir, f"{stem}_xgrad_ref_fp32.pt"))

            meta_lines.append(
                f"{i},{name},{dt_name},{s},{d},{m},{stem}"
            )
            n += 1

    with open(os.path.join(outdir, "meta.csv"), "w", encoding="utf-8") as f:
        f.write("idx,name,dtype,S,D,m,stem\n")
        f.write("\n".join(meta_lines) + "\n")

    print(f"已导出 {n} 组用例到 {outdir}")
    print(f"清单: {os.path.join(outdir, 'meta.csv')}")
    print()
    print("每组包含 5 个文件:")
    print("  <stem>_x.pt               前向输入 [S, D]  / 也是反向的期望输出形状")
    print("  <stem>_ograd.pt           反向输入 [S, m, D]")
    print("  <stem>_o_ref.pt           前向参考输出 [S, m, D]")
    print("  <stem>_xgrad_ref.pt       反向参考输出 [S, D]  (逐元素口径 = 赛题 sum)")
    print("  <stem>_xgrad_ref_fp32.pt  反向参考输出 [S, D]  (FP32 累加后回写，逐 bit 比对用)")
    return 0


# --------------------------------------------------------------------------
# 对拍
# --------------------------------------------------------------------------
def _load(path):
    t = torch.load(path, map_location="cpu", weights_only=False)
    if t.is_cuda:
        t = t.cpu()
    return t


def cmd_check(args) -> int:
    idx = args.case
    if idx < 0 or idx >= len(CASES):
        sys.stderr.write(f"--case 超出范围 [0, {len(CASES)-1}]\n")
        return 2
    name, s, d, m = CASES[idx]
    dt_name, dt = args.dtype, DTYPES[args.dtype]

    x, o_grad = make_case(s, d, m, dt, seed=idx)

    if args.check_forward:
        got = _load(args.check_forward)
        exp = ref_forward(x, m)
        ok, msg = check_parity(exp, got, f"前向 case={idx}({name}) {dt_name}")
        print(msg)
        return 0 if ok else 1

    if args.check_backward:
        got = _load(args.check_backward)
        fn = ref_backward_mean if args.mean else ref_backward_sum
        exp = fn(o_grad, m)
        ok, msg = check_parity(exp, got, f"反向 case={idx}({name}) {dt_name}"
                                          f"{' [mean]' if args.mean else ' [sum]'}")
        print(msg)
        if not ok:
            exp_alt = ref_backward_mean(o_grad, m) if not args.mean else ref_backward_sum(o_grad, m)
            alt_name = "mean" if not args.mean else "sum"
            ok2, msg2 = check_parity(exp_alt, got, f"      改用 {alt_name} 口径")
            print(msg2)
            if ok2:
                print(f"  >>> 提示: 实际结果与 {alt_name} 口径吻合，"
                      f"{'加' if alt_name == 'mean' else '去掉'} --mean 重跑")
                print(f"  >>> 两种口径相差 {m} 倍，请核对赛题评分口径！")
        return 0 if ok else 1

    sys.stderr.write("请指定 --check-forward 或 --check-backward\n")
    return 2


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="mHC-expand 算子参考实现与精度对拍",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true", help="列出全部用例")
    p.add_argument("--selftest", action="store_true", help="自检参考实现")
    p.add_argument("--export", metavar="DIR", help="导出用例到目录")
    p.add_argument("--check-forward", metavar="NPY", help="对拍前向结果")
    p.add_argument("--check-backward", metavar="NPY", help="对拍反向结果")
    p.add_argument("--case", type=int, default=0, help="用例编号，默认 0")
    p.add_argument("--dtype", choices=list(DTYPES), default="fp16", help="dtype，默认 fp16")
    p.add_argument("--mean", action="store_true",
                   help="反向按 vLLM hc_contract 的 mean 口径对拍（默认 sum，即赛题口径）")
    args = p.parse_args()

    if args.selftest:
        return cmd_selftest(args)
    if args.export:
        return cmd_export(args)
    if args.check_forward or args.check_backward:
        return cmd_check(args)
    return cmd_list(args)


if __name__ == "__main__":
    raise SystemExit(main())
