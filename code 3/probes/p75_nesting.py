#!/usr/bin/env python3
"""P75：M1e 的**成立条件**探针（纯本地，不碰真机）。

M1e 的全部收益押在一句话上：一个单元的 16 条 m-lane 可以装【同一 batch 的 RPW 行 × N1 个头】，
因为它们共享同一份 K gather。共享的前提是"最长那一行的 token 表覆盖其余行"，即
  (i) **嵌套**：token 表按 s 单调（s 行的表 ⊆ s+1 行的表，前缀即可）；
  (ii) **窗口内长度差不大**：否则 AIC 为一个单元搬的 K 远多于各行自己需要的 ⇒ 白搬。
mode3 的因果口径"应该是"嵌套的（§15.36 的推断），但这句话从没在**真实用例的 idx 张量**上量过。
这一发就把两条都量出来：对 RPW ∈ {2,4,8,16} 各算
  · 违反前缀嵌套的行对数（应为 0）
  · 单元数 ÷ 每单元片数（= AIC 侧握手轮数），与"各行各走自己表"的轮数比 ⇒ **轮数 ÷ 几**
  · 空转格占比 = 1 - Σ(有效列)/（16 lane × 片数 × n_blk）⇒ AIC 白算的比例
判据口径：轮数比 > 2.5 且空转 < 25 % ⇒ M1e 值得做；否则 M1e 判死、Cube 线改走"深度靠 GM 计数"。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_case import load, tokens_of   # noqa: E402

NBLK = 48


def check(name, path):
    C = load(path)
    B, S1, N1 = C['B'], C['S1'], C['N1']
    toks = [[tokens_of(C, b, s) for s in range(S1)] for b in range(B)]
    lens = [[len(t) for t in tb] for tb in toks]
    print('== %s  B=%d S1=%d N1=%d D=%d  行数=%d' % (name, B, S1, N1, C['D'], B * S1))
    # (i) 前缀嵌套：相邻行必须是"短的是长的前缀"
    bad = 0
    for b in range(B):
        for s in range(S1 - 1):
            a, c = toks[b][s], toks[b][s + 1]
            if len(a) > len(c) or a != c[:len(a)]:
                bad += 1
                if bad <= 3:
                    print('   !! 嵌套违反 b=%d s=%d->%d  len %d->%d' % (b, s, s + 1, len(a), len(c)))
    print('   前缀嵌套违反对数 = %d / %d   %s' % (bad, B * (S1 - 1), 'OK' if bad == 0 else '🔴 不成立'))
    if bad:
        return
    # (ii) 窗口
    for rpw in (2, 4, 8, 16):
        if S1 < rpw:
            continue
        rounds_u = rounds_p = 0
        cells_used = cells_all = 0
        for b in range(B):
            for s0 in range(0, S1, rpw):
                win = lens[b][s0:s0 + rpw]
                mx = max(win)
                lanes = len(win) * N1
                rounds_u += (mx + NBLK - 1) // NBLK
                rounds_p += sum((x + NBLK - 1) // NBLK for x in win)
                for x in win:
                    cells_used += x * N1
                    cells_all += mx * N1 * 1.0
        units = B * ((S1 + rpw - 1) // rpw)
        waste = 1.0 - (cells_used / cells_all if cells_all else 0.0)
        print('   RPW=%2d 单元=%-4d 轮数 %4d (逐行 %4d ⇒ ÷%.2f)  空转格 %5.1f %%  每单元lane=%d' %
              (rpw, units, rounds_u, rounds_p, rounds_p / max(rounds_u, 1), 100.0 * waste, rpw * N1))


if __name__ == '__main__':
    for p in sys.argv[1:] or ['/tmp/p75/w3.bin', '/tmp/p75/big1.bin']:
        check(os.path.basename(p).split('.')[0], p)
