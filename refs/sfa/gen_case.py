#!/usr/bin/env python3
"""
用当前 sfa_ref.py 重新生成一整套 SFA 回归用例到 ./cases/。

为什么需要它：
  refs/sfa 下的旧 c*.bin 是 `SFA_CASE 1` 格式（旧版生成），payload 比当前
  write_case() 少 16 字节 → 喂给重建版 test_sfa_real.cpp 会布局不符。
  本脚本用**当前**格式生成用例，保证与 harness 一致。

用法：
  python3 gen_case.py            # 生成到 ./cases/
  python3 gen_case.py <输出目录>

只依赖标准库（struct），不需要 numpy。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sfa_ref as S  # noqa: E402

# (名称, B, S1, S2, N1, D, SBS, MODE, COUNT, nblk) —— 与 run.sh 的用例表一致
CASES = [
    ("r1_min",     1, 1, 16, 1, 512, 1, 0, 16, 4),
    ("r2_chunk",   1, 1, 64, 1, 512, 1, 0, 64, 8),
    ("r3_mode3",   1, 4, 32, 1, 512, 1, 3, 32, 4),
    ("r4_shortkv", 1, 4,  4, 1, 512, 1, 3, 16, 2),
    ("r5_blocks",  1, 2, 64, 1, 512, 4, 0, 32, 4),
    ("r6_multiB",  2, 2, 32, 2, 512, 1, 3, 32, 4),
    ("r7_norope",  1, 1, 32, 1, 512, 1, 0, 32, 4),
    ("r8_heads",   1, 2, 32, 8, 512, 1, 3, 32, 4),
]
EXTRA = [
    ("case_small", 1, 2, 32, 2, 512, 1, 3, 8, 4),
    ("mini",       1, 1, 16, 1, 512, 1, 0, 8, 4),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cases")
    os.makedirs(outdir, exist_ok=True)

    n_ok = 0
    for (name, B, S1, S2, N1, D, SBS, MODE, CN, nblk) in CASES + EXTRA:
        zero_rope = (name == "r7_norope")
        case = S.gen_case(B, S1, S2, N1, D, SBS, MODE, CN,
                          seed=hash(name) % 10000, nblk=nblk, zero_rope=zero_rope)
        path = os.path.join(outdir, "%s.bin" % name)
        S.write_case(path, case)
        print("  生成 %-12s B=%d S1=%d S2=%d N1=%d SBS=%d MODE=%d COUNT=%d  -> %d B"
              % (name, B, S1, S2, N1, SBS, MODE, CN, os.path.getsize(path)))
        n_ok += 1

    print("\n完成：%d 个用例 -> %s" % (n_ok, outdir))
    print("下一步：把 cases/ 推到真机 sfa_real/cases/，再跑 run.sh")


if __name__ == "__main__":
    main()
