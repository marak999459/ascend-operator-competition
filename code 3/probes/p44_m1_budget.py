#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P44：把 §15.57 的段账换算成"每轮（一次 AIC↔AIV 交接）"的货币，再拿 §15.43 实测的
交接税 1.33~1.37 µs/轮去卡 M1（score 上 Cube）的**现实落点**。

为什么要这一屏：§15.57(6) 给的 1.85~2.05× 是"把 score 整段删掉"的**乐观上界**，它默认交接免费。
M1 的真实结构是**每 flush 一轮**（AIC 产 16×64 score → Fixpipe → AIV 读回做 softmax），
所以判据不是"score 值多少钱"，而是"**1.35 µs 的税 vs AIV 每花在 score 上的钱**"。
纯算术，不碰真机。税与 AIC 生产成本取自 §15.43(b)(c)(d)（`cubexfer` 锁步档）。
"""

# 实测：§15.57(3) 主表（批量 ms，AUTO 档）
FULL = {'big1': 0.6577, 'd2048': 5.0530, 'w4': 9.4417}
NOSC = {'big1': 0.3561, 'd2048': 2.6933, 'w4': 4.6032}   # 删 score 的乐观上界落点
NOPV = {'big1': 0.4723, 'd2048': 3.5804, 'w4': 6.3952}   # 删 PV
NOCALC = {'big1': 0.1026, 'd2048': 0.7171, 'w4': 0.7510}  # 地板

# 形状/档位（SFA_PICK 的 AUTO 档：P39 的真机对账；rows=B*Q_S；nb=每单元头数，N1=8）
CASE = {
    # name: (rows, nb, n_blk, 有效 token 数/单元, 扫描条目/单元)
    'big1':  (128, 4, 48, 256, 2048),
    'd2048': (128, 4, 48, 2048, 2048),
    'w4':    (128, 4, 48, 4096, 4096),   # SBS=2 ⇒ 每条展开 2 个 token
}
CORES = 40            # 910B3 的 AIV 核数；MIX 下 BD=20 组 × 2 AIV
TAX_US = 1.35         # §15.43(c) 锁步交接税（1.33~1.37 取中）
AIC_TILE_US = 0.35    # §15.43(d)：真实 tile 16×64 的 AIC 每轮生产成本 ≈ 探针的 1/8

print('%-6s %6s %6s %8s %9s | %8s %8s %8s | %8s' % (
    '案', '单元', '轮/单', '轮/核', 'µs/轮总', 'µs/轮score', '税/轮', '税/score', 'M1落点ms'))
for name, (rows, nb, k, valid, scan) in CASE.items():
    units = rows * ((8 + nb - 1) // nb)
    flushes = -(-valid // k)                      # 每单元几次 flush = 几轮
    rounds = units * flushes
    per_core_ideal = rounds / CORES
    waves = -(-units // CORES)
    per_core_quant = waves * flushes              # 最慢那核要跑满一整波
    full, sc = FULL[name], FULL[name] - NOSC[name]
    us_round = full * 1e3 / per_core_quant
    us_score = sc * 1e3 / per_core_quant
    tax_total = TAX_US * per_core_quant / 1e3     # 税全暴露在关键路径上的最坏情形（ms）
    m1_worst = NOSC[name] + tax_total             # 删了 score 但税一分不省
    m1_best = NOSC[name]                          # 税被完全重叠吃掉
    print('%-6s %6d %6d %8.0f %9.2f | %8.2f %8.2f %7.0f%% | %6.3f ~ %.3f  (%.2fx ~ %.2fx)' % (
        name, units, flushes, per_core_quant, us_round, us_score, TAX_US,
        100 * TAX_US / us_score, m1_best, m1_worst, full / m1_worst, full / m1_best))

print()
for name in FULL:
    rows, nb, k, valid, _ = CASE[name]
    units = rows * ((8 + nb - 1) // nb)
    pr = (-(-units // CORES)) * (-(-valid // k))
    us_score = (FULL[name] - NOSC[name]) * 1e3 / pr
    both = NOSC[name] + NOPV[name] - FULL[name]      # full − score边际 − PV边际
    print('  %-6s score %.2f µs/轮 vs 税 1.35 µs/轮 ⇒ 完全隐藏时 %.2f×、全不隐藏时 %.2f×；'
          'M1+M2：乐观 %.2f× / 保守(两轮交接各付一次税) %.2f×' % (
              name, us_score,
              FULL[name] / NOSC[name],
              FULL[name] / (NOSC[name] + TAX_US * pr / 1e3),
              FULL[name] / both,
              FULL[name] / (both + 2 * TAX_US * pr / 1e3)))
