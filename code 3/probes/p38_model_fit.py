#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P38 前置：给代价模型补上【每单元 chunk 数】这一维，然后用两份数据同时验收：
   ① 大形状实测网格（p34 + p37，0.24~16 ms，= 平台六点的量级）→ 比"选档后悔值"；
   ② 全部 27 个本地用例形状 → 只问"新模型会不会把已经验过的自选档改坏"（不改档 = 零风险）。

为什么是这一维：`cost = waves × UnitCalls`，而 `UnitCalls` 的注释自己写着"处理**一个 KV
chunk** 的代价"，`waves` 却是【整个单元】的波数 ⇒ 中间少了"这个单元要扫多少个 chunk"
= `ceil(token/单元 / n_blk)`。少掉它的直接后果（§15.50 已经点过、但当时判停在"本地量不到"）：
模型在 k 轴上只剩"k 越大每次调用越贵"⇒ **反指**；而在 nb 轴上，"nb 越大 → 最大可行 k 越小
→ 每 chunk 越便宜"这条假收益没人对冲 ⇒ 大 nb 被系统性偏袒。P37 的网格正好把这条抓了现行：
w4 上模型选 nb=8/k=40（cost 2512），实测它是第 3 名（10.49 ms），全场最快是 nb=4/k=48 = 9.46 ms。
"""
import math
import os
import re

QD, DR, E = 512, 64, 2
UB_SAFE = 186485
STAGE_MAX = 48
CORES = 40
NBLK_MIN = 16
NB_CAND = [32, 16, 8, 4, 2, 1]
NBLK_CAND = [128, 64, 48, 40, 32, 16, 8, 4, 2, 1]
GATHER_DIV = 660
KS_TAX = 108
HYST = 19          # 迟滞：cost*20 < best*HYST 才换档（host 现在是 19 = 0.95）

# ---------------- 与 host 同源的 UB 预算 / 调用数 ----------------
def align(x):
    return (x + 31) // 32 * 32

def half(nb):
    return (nb + 7) // 8 * 8

def ub_need(nb, k):
    return (align(nb * (QD + DR) * 4) + align(nb * QD * 4) + align(k * QD * E)
            + align(k * DR * E) + align(nb * k * 4) + align(nb * k * 4)
            + align(3 * half(nb) * 4) + 2 * align(half(nb) * 4)
            + align(k * QD * 4) + align(k * DR * 4)
            + 3 * align(max(nb, k, 16) * 4))

def calls(nb, k):
    """一个工作单元处理【一个 chunk】的等效向量调用条数（= host UnitCalls）。"""
    widen = nb * k * (QD + DR) * 1123 // (64 * 36000)
    gather = k * (2 * QD + DR) * E // (2 * nb * GATHER_DIV)
    return nb * 22 + widen + (3 + 4 * nb + k * nb) + gather

def ks2_allowed(units, lse, count, k):
    return (1 <= units and units * 2 <= CORES and lse >= 8 and lse % 8 == 0
            and count >= 2 * k)

def pick(rows, qn, count, sbs, lse, mode='old', hyst=HYST, S2=8192):
    """mode: 'old' = 现状 cost = waves × calls；'new' = 再乘每单元 chunk 数。"""
    toks = min(count * sbs, S2)             # 每行最多扫这么多 token（host 只见得到这些）
    best = None
    for nb in [x for x in NB_CAND if x <= max(1, qn)]:
        units = rows * ((qn + nb - 1) // nb)
        waves = (units + CORES - 1) // CORES
        for pas in (0, 1):
            hit = False
            for k in [x for x in NBLK_CAND if NBLK_MIN <= x <= STAGE_MAX]:
                if pas == 0 and nb * k < sbs:
                    continue
                if ub_need(nb, k) > UB_SAFE:
                    continue
                c = calls(nb, k)
                mult = math.ceil(toks / k) if mode == 'new' else 1
                cost = waves * c * mult
                ks = 1
                if ks2_allowed(units, lse, count, k):
                    w2 = (units * 2 + CORES - 1) // CORES
                    m2 = math.ceil(toks / (2 * k)) if mode == 'new' else 1
                    c2 = ((w2 * c * m2 // (2 if mode == 'old' else 1)) * KS_TAX) // 100 + 1
                    if c2 < cost:
                        cost, ks = c2, 2
                if best is None or cost * 20 < best[0] * hyst:
                    best = (cost, nb, k, ks)
                hit = True
                break
            if hit:
                break
    return best

# ---------------- 实测网格（p34 + p37 两趟取最小）----------------
SHAPE = {          # case: rows, qN, count, sbs   （lse = rows*qN）
    'w1': (4, 4, 2048, 2), 'w2': (32, 4, 2048, 2), 'w3': (128, 4, 2048, 2),
    'w4': (128, 8, 2048, 2),
}
MEAS = {}
pat = re.compile(r'^(?:AUTO|GRID|G) (\w+).*SFA_PICK nb=(\d+) nblk=(\d+) ks=(\d+) '
                 r'平均 ([\d.]+) ms 平均 ([\d.]+) ms')
here = os.path.dirname(os.path.abspath(__file__))
for fn in ('p34_big.txt', 'p37_fullbig.txt'):
    for line in open(os.path.join(here, fn), encoding='utf-8'):
        m = pat.match(line.strip())
        if not m:
            continue
        cs, nb, k, ks, t1, t2 = m.groups()
        nb, k, ks = int(nb), int(k), int(ks)
        MEAS.setdefault((cs, nb, k, ks), []).extend([float(t1), float(t2)])
# 同一 (case,nb,k) 下 ks=1 优先（P37 量到 ks=1/2 差 <0.2 %，p34 那批只有 ks=2）
M = {}
for (cs, nb, k, ks) in sorted(MEAS, key=lambda x: x[3]):
    if (cs, nb, k) in M:
        continue
    M[(cs, nb, k)] = min(MEAS[(cs, nb, k, ks)])

print("=== 大形状选档后悔值（亏 = 模型自选档比全场实测最快慢多少）===")
print(f"{'case':5} {'实测最优档':>20}  {'旧模型':>10} {'亏':>9}   {'新模型':>10} {'亏':>9}")
for cs, (rows, qn, count, sbs) in SHAPE.items():
    lse = rows * qn
    arm = {kk[1:]: v for kk, v in M.items() if kk[0] == cs}
    if not arm:
        continue
    bk = min(arm, key=arm.get)
    bt = arm[bk]
    po, pn_ = pick(rows, qn, count, sbs, lse, 'old'), pick(rows, qn, count, sbs, lse, 'new')
    def fmt(p):
        t = arm.get((p[1], p[2]))
        return ('nb=%d/k=%d' % (p[1], p[2]),
                ('%+.1f %%' % (100 * (t / bt - 1))) if t else '未测')
    so, ro = fmt(po)
    sn, rn = fmt(pn_)
    print('%-5s nb=%d/k=%d=%.4f   %10s %9s   %10s %9s'
          % (cs, bk[0], bk[1], bt, so, ro, sn, rn))

print("\n=== 两份模型在 31 个本地用例形状上的自选档（新==旧 ⇒ 对已验过的档位零影响）===")
LOCAL = [  # name, rows, qN, count, sbs
    ('r1_min', 1, 1, 16, 1), ('r2_chunk', 1, 1, 64, 1), ('r3_mode3', 4, 1, 32, 1),
    ('r4_shortkv', 4, 1, 16, 1), ('r5_blocks', 2, 1, 32, 4), ('r6_multiB', 4, 2, 32, 1),
    ('r7_norope', 1, 1, 32, 1), ('r8_heads', 2, 8, 32, 1),
    ('p1', 4, 4, 2048, 1), ('p1s1', 4, 4, 2048, 1), ('p1s1h', 4, 4, 2048, 1),
    ('p1s2', 4, 4, 2048, 2), ('p1s2h', 4, 4, 2048, 2), ('p1s4', 4, 4, 2048, 4),
    ('p1s8', 4, 4, 1024, 8), ('p2', 8, 2, 2048, 1), ('p2s2', 8, 2, 2048, 2),
    ('p4', 16, 4, 2048, 1), ('p4s2', 16, 4, 2048, 2), ('p6', 32, 4, 2048, 1),
    ('p6s2', 32, 4, 2048, 2), ('p_n64', 128, 8, 2048, 1), ('p_n512', 128, 8, 2048, 1),
    ('p_n1024', 128, 8, 2048, 1), ('q1h', 4, 4, 2048, 1), ('q2h', 4, 4, 2048, 2),
    ('q3h', 8, 4, 2048, 1), ('e1empty', 4, 4, 2048, 1), ('e6many', 32, 4, 2048, 1),
    ('e7padq', 8, 4, 2048, 1), ('big1', 128, 8, 2048, 1),
]
diff = 0
for nm, rows, qn, count, sbs in LOCAL:
    lse = rows * qn
    po = pick(rows, qn, count, sbs, lse, 'old')
    pn_ = pick(rows, qn, count, sbs, lse, 'new')
    same = (po[1], po[2], po[3]) == (pn_[1], pn_[2], pn_[3])
    diff += not same
    print('%-10s rows=%-4d qN=%d sbs=%d count=%-5d  旧 nb=%d/k=%d/ks=%d  '
          '新 nb=%d/k=%d/ks=%d%s'
          % (nm, rows, qn, sbs, count, po[1], po[2], po[3], pn_[1], pn_[2], pn_[3],
             '' if same else '   <<< 改档'))
print('\n改档数 %d / %d' % (diff, len(LOCAL)))
