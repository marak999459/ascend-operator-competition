#!/usr/bin/env python3
"""P127 离线判别器：**逐函数镜像** `code/op_host/sparse_flash_attention.cpp` 的 CalcBlocking，
把迟滞常数做成参数，回答一个问题 —— "0.95 → 0.90 到底改不改选档？改的话往哪边改？"

⚠️ 这是**镜像不是权威**（权威是那份 .cpp；改了 host 规则要改这里重跑）。
⚠️ 与 `p10_pick.py` 的区别：那份停在 P13 时代的 UB/UnitCalls 口径（还有 pf/prf、组宽 16/32、
   迟滞 0.85），已经过期；这份按 P105 的源码逐项抄（`HalfElems`/`RedWOf`/`GatherCalls`
   的 `e/2`/`UnitCalls` 的 `widenVol`/`SFA_STAGE_MAX=48`/`KsAllowed` 三条门 + `FOLD_TAX`）。
"""

UB_PHYS, UB_SAFE_PCT = 196608, 95
UB_BLK = 32
UB_BLK_F32 = UB_BLK // 4          # 8
NB_CAND = [32, 16, 8, 4, 2, 1]
NBLK_CAND = [128, 64, 48, 40, 32, 16, 8, 4, 2, 1]
KS_CAND = [2, 3, 4, 5, 6, 8, 10]
NB_MIN, NBLK_MIN, SFA_STAGE_MAX = 1, 16, 48
COST_NS = 40
FOLD_TAX = 10000 // COST_NS       # 250
GATHER_DIV = 660
RED_MIN_W = 2 * UB_BLK_F32
CORES = 40


def align(x):
    return (x + 31) // 32 * 32


def half_elems(nb):
    return (nb + UB_BLK_F32 - 1) // UB_BLK_F32 * UB_BLK_F32


def red_w(nb, k):
    return max(max(k, nb), RED_MIN_W)


def ub_need(nb, k, qd, dr, e):
    return (align(nb * (qd + dr) * 4) + align(nb * qd * 4)
            + align(k * qd * e) + align(k * dr * e)
            + align(nb * k * 4) + align(nb * k * 4)
            + align(3 * half_elems(nb) * 4) + 2 * align(half_elems(nb) * 4)
            + align(k * qd * 4) + align(k * dr * 4)
            + 3 * align(red_w(nb, k) * 4))


def gather_calls(nb, k, qd, dr, e):
    return k * (2 * qd + dr) * e // (2 * nb * GATHER_DIV)


def unit_calls(nb, k, qd, dr, e):
    widen = nb * k * (qd + dr) * 1123 // (64 * 36000)
    return nb * 22 + widen + (3 + 4 * nb + k * nb) + gather_calls(nb, k, qd, dr, e)


def ks_allowed(units, lse, count, k, ks):
    return (units >= 1 and units * ks <= CORES
            and lse >= 8 and lse % 8 == 0 and count >= ks * k)


def calc_blocking(sbs, qd, dr, qn, e, rows, count, kvs, hyst_num, hyst_den):
    ub_safe = UB_PHYS * UB_SAFE_PCT // 100
    toks = min(count * sbs, kvs)
    cap = max(NB_MIN, qn)
    lse = rows * qn
    best = None            # (cost, nb, nblk, ks)
    for nb in NB_CAND:
        if nb > cap:
            continue
        units = rows * ((qn + nb - 1) // nb)
        waves = (units + CORES - 1) // CORES
        for take_sbs in (True, False):        # pass 0 / pass 1
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
                    ck = (toks // m + k - 1) // k or 1
                    ck = max(ck, 1)
                    cost_k = wk * calls * ck + (m - 1) * FOLD_TAX
                    if cost_k < cost:
                        cost, ks = cost_k, m
            if best is None or cost * hyst_den < best[0] * hyst_num:
                best = (cost, nb, k, ks, units, waves)
            break
    if best is None:
        return None
    return best


def line(tag, sbs, qd, dr, qn, e, rows, count, kvs):
    a = calc_blocking(sbs, qd, dr, qn, e, rows, count, kvs, 19, 20)   # 0.95 现状
    b = calc_blocking(sbs, qd, dr, qn, e, rows, count, kvs, 9, 10)    # 0.90 提案
    def fmt(x):
        return 'nb=%-2d k=%-2d ks=%-2d units=%-4d' % (x[1], x[2], x[3], x[4]) if x else '—'
    flag = 'SAME' if a[1:] == b[1:] else ('⬆抬 nb' if b[1] > a[1] else '⬇降 nb')
    print('%-30s 0.95[%s] 0.90[%s]  %s' % (tag, fmt(a), fmt(b), flag))
    return a[1:] != b[1:]


if __name__ == '__main__':
    print('=== 本地已有靶（对表用：这一列必须与 p126_ab.txt 的 pick 行一致）===')
    for name, sbs, qn, rows, cnt in [('p1', 1, 4, 4, 2048), ('p2', 1, 4, 8, 2048),
                                     ('p4', 1, 4, 16, 2048), ('p6', 1, 4, 32, 2048),
                                     ('big1', 1, 8, 128, 2048), ('w3', 2, 4, 128, 2048),
                                     ('r64c65', 1, 1, 64, 65536)]:
        kvs = 8192 if name != 'r64c65' else 65536
        line(name, sbs, 512, 64, qn, 2, rows, cnt, kvs)
    print('=== plausible 平台族（sbs=1 ∧ rows>=41 ∧ fp16，扫 qN 与 KV 总长）===')
    for qn in (2, 4, 8, 16):
        for rows in (41, 48, 64, 96, 128, 160):
            for kvs in (2048, 8192):
                line('qN=%d rows=%d S2=%d' % (qn, rows, kvs),
                     1, 512, 64, qn, 2, rows, 2048, kvs)
            print()
