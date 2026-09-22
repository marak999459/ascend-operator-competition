#!/usr/bin/env python3
# P39：把 host 的选档代价模型原样搬到 Python，用 P37 的真机三维网格当靶子，
# 看模型在【大形状】上把 nb 排错在哪一步。只读不改代码 —— 目的是给 P38 定靶。
# 模型项抄自 code 3/code/op_host/sparse_flash_attention.cpp:140-245。
import math

QD, DR, E = 512, 64, 2
GATHER_DIV = 660
UB_SAFE = 186485
STAGE_MAX = 48
CORES = 40
NB_CAND = [32, 16, 8, 4, 2, 1]
NBLK_CAND = [128, 64, 48, 40, 32, 16, 8, 4, 2, 1]


def align(x):
    return (x + 31) // 32 * 32


def half_elems(nb):
    return (nb + 7) // 8 * 8


def red_w(nb, k):
    return max(nb, k, 16)


def ub_need(nb, k):
    return (align(nb * (QD + DR) * 4) + align(nb * QD * 4) + align(k * QD * E)
            + align(k * DR * E) + align(nb * k * 4) + align(nb * k * 4)
            + align(3 * half_elems(nb) * 4) + 2 * align(half_elems(nb) * 4)
            + align(k * QD * 4) + align(k * DR * 4) + 3 * align(red_w(nb, k) * 4))


def gather_calls(nb, k):
    return k * (2 * QD + DR) * E // (2 * nb * GATHER_DIV)


def unit_calls(nb, k):
    widen = nb * k * (QD + DR) * 1123 // (64 * 36000)
    return nb * 22 + widen + (3 + 4 * nb + k * nb) + gather_calls(nb, k)


def ks2_allowed(units, lse_elems, count, k):
    return (units >= 1 and units * 2 <= CORES and lse_elems >= 8
            and lse_elems % 8 == 0 and count >= 2 * k)


SHAPES = {   # name: (rows, qN, count)
    'w1': (4, 4, 2048),
    'w2': (32, 4, 2048),
    'w3': (128, 4, 2048),
    'w4': (128, 8, 2048),
}

# P37 pass-1 真机实测（ms），键 (case, nb, k)；ks 两臂相同故不列。
MEAS = {
    ('w2', 1, 32): 1.9388, ('w2', 1, 40): 1.8352, ('w2', 1, 48): 1.7823,
    ('w2', 2, 32): 1.6166, ('w2', 2, 40): 1.5629, ('w2', 2, 48): 1.5086,
    ('w2', 4, 32): 1.4832, ('w2', 4, 40): 1.4070, ('w2', 4, 48): 1.3692,
    ('w3', 1, 32): 6.2897, ('w3', 1, 40): 5.9581, ('w3', 1, 48): 5.7790,
    ('w3', 2, 32): 5.6012, ('w3', 2, 40): 5.4091, ('w3', 2, 48): 5.2190,
    ('w3', 4, 32): 5.8509, ('w3', 4, 40): 5.5466, ('w3', 4, 48): 5.4009,
    ('w4', 1, 32): 12.5769, ('w4', 1, 40): 11.9032, ('w4', 1, 48): 11.5513,
    ('w4', 2, 32): 10.4016, ('w4', 2, 40): 10.0446, ('w4', 2, 48): 9.6912,
    ('w4', 4, 32): 10.2575, ('w4', 4, 40): 9.7230, ('w4', 4, 48): 9.4604,
    ('w4', 8, 32): 10.9141, ('w4', 8, 40): 10.4933,
}

for name, (rows, qn, count) in SHAPES.items():
    lse = rows * qn
    print(f"\n### {name}  rows={rows} qN={qn} count={count}")
    print(f"{'nb':>3} {'k':>4} {'units':>6} {'waves':>5} {'UB':>7} {'calls':>7} "
          f"{'cost':>8} {'meas':>8} {'ms/unit-wave':>12}")
    for nb in NB_CAND:
        if nb > max(1, qn):
            continue
        for k in sorted(NBLK_CAND, reverse=True):
            if k < 16 or k > STAGE_MAX:
                continue
            if ub_need(nb, k) > UB_SAFE:
                continue
            units = rows * ((qn + nb - 1) // nb)
            waves = (units + CORES - 1) // CORES
            c = unit_calls(nb, k)
            cost = waves * c
            meas = MEAS.get((name, nb, k))
            print(f"{nb:>3} {k:>4} {units:>6} {waves:>5} {ub_need(nb,k):>7} {c:>7} "
                  f"{cost:>8} {('%8.4f'%meas) if meas else '     ---'} "
                  f"{('%.4f' % (meas / (units/waves) )) if meas else ''}")

    # 复刻 CalcBlocking 的决策（含 ks 臂与 0.95 迟滞），与实测最优对照
    best = None
    for nb in [x for x in NB_CAND if x <= max(1, qn)]:
        units = rows * ((qn + nb - 1) // nb)
        waves = (units + CORES - 1) // CORES
        for pas in (0, 1):
            hit = False
            for k in sorted(NBLK_CAND, reverse=True):
                if k < 16 or k > STAGE_MAX:
                    continue
                if pas == 0 and nb * k < 2:
                    continue
                if ub_need(nb, k) > UB_SAFE:
                    continue
                c = unit_calls(nb, k)
                cost = waves * c
                ks = 1
                if ks2_allowed(units, lse, count, k):
                    w2 = (units * 2 + CORES - 1) // CORES
                    c2 = ((w2 * c // 2) * 108) // 100 + 1
                    if c2 < cost:
                        cost, ks = c2, 2
                if best is None or cost * 20 < best[0] * 19:
                    best = (cost, nb, k, ks)
                hit = True
                break
            if hit:
                break
    meas_ok = {kk: vv for kk, vv in MEAS.items() if kk[0] == name}
    bm = min(meas_ok, key=lambda x: meas_ok[x]) if meas_ok else None
    print(f"  模型选 nb={best[1]}/k={best[2]}/ks={best[3]} cost={best[0]}"
          + (f"   实测最优 {bm} = {meas_ok[bm]:.4f} ms，模型这一档 = "
             f"{meas_ok.get((name,best[1],best[2]), float('nan')):.4f} ms" if bm else ""))
