#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""代价模型的离线复验：把 host 的 CalcBlocking 在 Python 里逐行复刻，
用 /tmp/grid_P21.txt、/tmp/grid_P24.txt 的**实测 64 格**当判据，
比较不同 waves/迟滞口径的"选档后悔值"。只在本地跑，不上真机。

⚠️ **只有 `ub_need()` 可信**（它与 host 的 `CalcUbNeed` 逐项同源，改 host 必须同步）。
   `pick()` 的 **nb 列已判定不可信**（§15.53(5)：它没复刻 host 两轮 `pass` 与迟滞的实际
   执行顺序，会漏掉 k=40/48 这类可行格 ⇒ 曾把 P29 的 p1 自选档误报成 `2/32/2`，
   真机实测是 `1/40/2`）。要基准就用真机自选档，别引用这里的 `pick()`。
"""
import re, sys, math

UB = 196352
UB_SAFE_PCT = 95
CORES = 40
UB_BLK = 32
UB_BLK_F32 = UB_BLK // 4
NB_CAND = [32, 16, 8, 4, 2, 1]
NBLK_CAND = [128, 64, 48, 40, 32, 16, 8, 4, 2, 1]  # P29 起与 host 同源：改 host 必须同步这里与 STAGE_MAX
NBLK_MIN = 16
STAGE_MAX = 48   # P32：V 复用 kBuf_ 之后 48 第一次过 UB 预算（§15.53(3)）
RED_MIN_W = 2 * UB_BLK_F32
GATHER_DIV = 660
WIDEN_DIV = 64 * 36000
KS_TAX = 108


def align(x):
    return (x + 31) // 32 * 32


def half(nb):
    return (nb + UB_BLK_F32 - 1) // UB_BLK_F32 * UB_BLK_F32


def redw(nb, k):
    return max(nb, k, RED_MIN_W)


def ub_need(nb, k, qD, dr, e):
    # P32：V 复用 kBuf_（同一份 align(k*qD*e)），所以这里只有一项 —— 与 host 的
    #      CalcUbNeed 逐项同源，改 host 必须同步改这里。
    return (align(nb * (qD + dr) * 4) + align(nb * qD * 4)
            + align(k * qD * e) + align(k * dr * e)
            + align(nb * k * 4) + align(nb * k * 4)
            + align(3 * half(nb) * 4) + 2 * align(half(nb) * 4)
            + align(k * qD * 4) + align(k * dr * 4)
            + 3 * align(redw(nb, k) * 4))


def gather(nb, k, qD, dr, e):
    return k * (2 * qD + dr) * e // (2 * nb * GATHER_DIV)


def calls(nb, k, qD, dr, e, widen=True):
    n = nb
    wv = (n * k * (qD + dr) * 1123) // WIDEN_DIV if widen else 0
    return n * 22 + wv + (3 + 4 * n + k * n) + gather(nb, k, qD, dr, e)


def ks2_allowed(units, cores, lse, count, k):
    return units >= 1 and units * 2 <= cores and lse >= 8 and lse % 8 == 0 and count >= 2 * k


def pick(sh, waves_mode='ceil', hyst=0.85, widen=True):
    """返回 (nb, k, ks, cost) 与全候选表；复刻 host 的两轮 pass 与迟滞。"""
    rows, qN, qD, dr, e, count, lse, sbs = sh
    ub_safe = UB // 100 * UB_SAFE_PCT
    nb_cap = max(qN, 1)
    best = None
    table = {}
    for nb in NB_CAND:
        if nb > nb_cap:
            continue
        units = rows * ((qN + nb - 1) // nb)
        w = ((units + CORES - 1) // CORES) if waves_mode == 'ceil' else max(units / CORES, 1.0)
        for pas in (0, 1):
            found = False
            for k in NBLK_CAND:
                if k < NBLK_MIN or k > STAGE_MAX:
                    continue
                if pas == 0 and nb * k < sbs:
                    continue
                if ub_need(nb, k, qD, dr, e) > ub_safe:
                    continue
                c = calls(nb, k, qD, dr, e, widen)
                cost = w * c
                ks = 1
                if ks2_allowed(units, CORES, lse, count, k):
                    u2 = units * 2
                    w2 = ((u2 + CORES - 1) // CORES) if waves_mode == 'ceil' else max(u2 / CORES, 1.0)
                    cost2 = (w2 * c // 2) * KS_TAX // 100 + 1
                    if cost2 < cost:
                        cost = cost2
                        ks = 2
                table[(nb, k, ks)] = cost
                found = True
                if best is None or cost < best[3] * hyst:
                    best = (nb, k, ks, cost)
                break
            if found:
                break
    return best, table


def load(p):
    d = {}
    for ln in open(p, errors='replace'):
        m = re.match(r'(\S+)\s+nb=(\d+)\s+kb=(\d+)\s+ks=(\d+)\s+\|\s+平均\s+([\d.]+)\s+ms\s+超差\s+(\d+)/(\d+)', ln)
        if m:
            d[(m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))] = float(m.group(5))
    return d


# 名字: rows=B*S1, qN=N1, qD, dr, elemSize, count(=2048), lse=B*S1*N1, sbs
SHAPES = {
    'p1':   (4, 4, 512, 64, 2, 2048, 16, 1),
    'p4':   (16, 4, 512, 64, 2, 2048, 64, 1),
    'p6':   (32, 4, 512, 64, 2, 2048, 128, 1),
    'q2h':  (4, 4, 512, 64, 2, 2048, 16, 2),
    'big1': (128, 8, 512, 64, 2, 2048, 1024, 1),
}

if __name__ == '__main__':
    grids = {'P21': load('/tmp/grid_P21.txt'), 'P24': load('/tmp/grid_P24.txt')}
    for tag, g in grids.items():
        print(f'\n===== 实测档 {tag} =====')
        for wm in ('ceil', 'exact'):
            for hy in (0.85, 0.95, 1.01):
                tot = 0.0
                det = []
                for cs, sh in SHAPES.items():
                    b, _ = pick(sh, wm, hy)
                    meas = {k[1:]: v for k, v in g.items() if k[0] == cs}
                    feas = {kk: vv for kk, vv in meas.items()
                            if ub_need(kk[0], kk[1], 512, 64, 2) <= UB // 100 * 95 and kk[1] >= 32}
                    got = meas.get(b[:3])
                    bm = min(feas.values())
                    det.append(f'{cs}:{"/".join(map(str,b[:3]))}={got:.4f}(best {bm:.4f})')
                    tot += got / bm
                print(f'  waves={wm:5} hyst={hy:4}  后悔和={tot/len(SHAPES):.4f}   ' + ' '.join(det))
