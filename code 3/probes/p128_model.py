#!/usr/bin/env python3
"""P128 离线判别器：在 `p127_model.py` 那份**逐函数镜像**上多挂一个 `mode`，回答三件事 ——
  ① "in incumbent 单元数护栏"（FLOOR）把六点族切成哪两半，哪一半 = P127、哪一半 = auto；
  ② 这条切法能不能把 P127 那发里"pin 反而慢 76 % 的那一点"排除掉、同时留住三点负响应；
  ③ FLOOR 取 40 / 64 / 80 / 128 时结论有多稳（不稳就别自欺）。

⛔ 镜像不是权威（权威 = `code/op_host/sparse_flash_attention.cpp` 的 CalcBlocking）。
   唯一被改的一行是迟滞判定；其余（UB 预算 / UnitCalls / KsAllowed / FOLD_TAX）从 127 那份 import。
"""
from p127_model import (NB_CAND, NBLK_CAND, KS_CAND, NB_MIN, NBLK_MIN, SFA_STAGE_MAX,
                        UB_PHYS, UB_SAFE_PCT, FOLD_TAX, CORES, align, ub_need,
                        unit_calls, ks_allowed)


def calc(sbs, qd, dr, qn, e, rows, count, kvs, mode, floor=0):
    """mode: 'auto' = 0.95 迟滞（现状）；'pin' = P127（首位可行者直接胜出）；
       'guard' = P128（incumbent 单元数 >= floor 时行为 = pin，< floor 时行为 = auto）。"""
    ub_safe = UB_PHYS * UB_SAFE_PCT // 100
    toks = min(count * sbs, kvs)
    cap = max(NB_MIN, qn)
    lse = rows * qn
    best = None                              # (cost, nb, nblk, ks, units, waves)
    for nb in NB_CAND:
        if nb > cap:
            continue
        units = rows * ((qn + nb - 1) // nb)
        waves = (units + CORES - 1) // CORES
        for take_sbs in (True, False):
            hit = None
            for k in NBLK_CAND:
                if k < NBLK_MIN or k > SFA_STAGE_MAX:
                    continue
                if take_sbs and nb * k < sbs:
                    continue
                if ub_need(nb, k, qd, dr, e) > ub_safe:
                    continue
                hit = k
                break
            if not hit:
                continue
            k = hit
            chunks = (toks + k - 1) // k or 1
            calls = unit_calls(nb, k, qd, dr, e)
            cost = waves * calls * chunks
            ks = 1
            if ks_allowed(units, lse, count, k, 2):
                for m in KS_CAND:
                    if not ks_allowed(units, lse, count, k, m):
                        continue
                    uk = units * m
                    wk = (uk + CORES - 1) // CORES
                    ck = max((toks // m + k - 1) // k or 1, 1)
                    cost_k = wk * calls * ck + (m - 1) * FOLD_TAX
                    if cost_k < cost:
                        cost, ks = cost_k, m
            if best is None:
                replace = True
            elif mode == 'pin':
                replace = False
            elif mode == 'guard':
                replace = (best[4] < floor) and (cost * 20 < best[0] * 19)
            else:                            # 'auto'
                replace = cost * 20 < best[0] * 19
            if replace:
                best = (cost, nb, k, ks, units, waves)
            break
    return best


def fmt(x):
    return '—' if x is None else 'nb=%-2d k=%-2d ks=%-2d u=%-4d' % (x[1], x[2], x[3], x[4])


LOCAL = [('p1', 1, 4, 4, 2048, 8192), ('p2', 1, 4, 8, 2048, 8192),
         ('p4', 1, 4, 16, 2048, 8192), ('p6', 1, 4, 32, 2048, 8192),
         ('big1', 1, 8, 128, 2048, 8192), ('w3', 2, 4, 128, 2048, 8192),
         ('r64c65', 1, 1, 64, 65536, 65536),
         ('pn16a', 1, 16, 4, 2048, 8192), ('pn32a', 1, 32, 8, 2048, 8192)]


def three(name, sbs, qn, rows, cnt, kvs, floor=80):
    a = calc(sbs, 512, 64, qn, 2, rows, cnt, kvs, 'auto')
    b = calc(sbs, 512, 64, qn, 2, rows, cnt, kvs, 'pin')
    g = calc(sbs, 512, 64, qn, 2, rows, cnt, kvs, 'guard', floor)
    tag = ('guard=auto' if g[1:] == a[1:] else
           ('guard=pin' if g[1:] == b[1:] else '第三态!'))
    return a, b, g, tag


if __name__ == '__main__':
    print('=== ① 本地靶：auto / pin(P127) / guard(80) 三臂（对表 p127_gaterun.txt 的 pick 行）===')
    for name, sbs, qn, rows, cnt, kvs in LOCAL:
        a, b, g, tag = three(name, sbs, qn, rows, cnt, kvs)
        print('%-8s auto[%s] pin[%s] guard80[%s]  %s' % (name, fmt(a), fmt(b), fmt(g), tag))

    print('=== ② plausible 平台族（sbs=1 ∧ fp16 ∧ rows>=41）：护栏把哪一半交给 pin ===')
    for floor in (40, 64, 80, 128):
        npin = nauto = nchg = 0
        for qn in (2, 4, 8, 16, 32):
            for rows in (41, 48, 64, 96, 128, 160, 256):
                for kvs in (2048, 8192):
                    a, b, g, tag = three('x', 1, qn, rows, 2048, kvs, floor)
                    if tag == 'guard=pin' and b[1:] != a[1:]:
                        npin += 1
                    elif tag == 'guard=auto':
                        nauto += 1
                    else:
                        nchg += 1
        print('FLOOR=%-4d 交给 pin（且 pin≠auto）%3d 格 / 交给 auto %3d 格 / 第三态 %d 格' %
              (floor, npin, nauto, nchg))

    print('=== ③ 逐格明细（FLOOR=80，S2=8192）：P=交给 pin、A=交给 auto、括号里 auto→pin 的单元数 ===')
    for qn in (2, 4, 8, 16, 32):
        row = []
        for rows in (41, 48, 64, 96, 128, 160, 256):
            a, b, g, tag = three('x', 1, qn, rows, 2048, 8192, 80)
            mark = 'P' if tag == 'guard=pin' else ('A' if tag == 'guard=auto' else '?')
            row.append('%d:%s%s' % (rows, mark, '' if a[1:] == b[1:] else
                                    '(%d→%d)' % (a[4], b[4])))
        print('qN=%-3d %s' % (qn, ' '.join(row)))
