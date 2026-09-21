#!/usr/bin/env python3
"""复刻 host 的 CalcBlocking（P10 代价模型），离线打印每个形状选中的 (nb, n_blk)。

用途：真机上没有探针（§15.14(e) 特别强调"远端从未出现带探针的构建"），选中值只能
由 host 规则算出来。⚠️ 这是**镜像**，不是权威 —— 权威是 `code/op_host/*.cpp`；
改了 host 规则就改这里重跑，别拿它反过来当实测证据。
"""

SFA_SC_GRP = 16     # 多头单位的组宽
SFA_SC_GRP_W = 32   # P13：单头单位（nb==1）的组宽，就地广播乘省掉 pf/prf 换来的
NB_CAND = [32, 16, 8, 4, 2, 1]
NBLK_CAND = [128, 64, 32, 16, 8, 4, 2, 1]
NB_MIN, NBLK_MIN, UB_BLK = 1, SFA_SC_GRP, 32
UB_PHYS, UB_SAFE_PCT = 196608, 95


def align(x):
    return (x + 31) // 32 * 32


def sc_grp(nb):
    """host 的 ScGrpOf / kernel 的 scGrp_ 镜像：nb==1 时组宽翻倍。"""
    return SFA_SC_GRP_W if nb == 1 else SFA_SC_GRP


def half_elems(nb):
    return (nb + UB_BLK // 4 - 1) // (UB_BLK // 4) * (UB_BLK // 4)


def ub_need(nb, k, qd, dr, es):
    g = sc_grp(nb)
    scratch = align(g * qd * 4) + align(g * dr * 4)      # kfBuf_ + krfBuf_
    if nb != 1:
        scratch += align(g * qd * 4) + align(g * dr * 4)  # pfBuf_ + prfBuf_（P13：nb==1 不分配）
    return (align(nb * (qd + dr) * 4) + align(nb * qd * 4)
            + align(k * qd * es) + align(k * qd * es) + align(k * dr * es)
            + align(nb * k * 4) + align(nb * k * 4)
            + align(3 * half_elems(nb) * 4) + 2 * align(half_elems(nb) * 4)
            + scratch
            + 3 * align(max(nb, g) * 4))


def gather_calls(nb, k, qd, dr, es):
    """P13b：搬运项 —— 每单元每 chunk 搬 K+V+Krope，代价 ∝ k·(2qD+dr)·e/nb。
    系数 1/660 由 §15.16 的 p4/p6/big1 三个实测点各自反解（1.42e-3/1.52e-3/1.68e-3）。"""
    return k * (2 * qd + dr) * es // (2 * nb * 660)


def unit_calls(nb, k, qd=512, dr=64, es=2):
    # P13：三段重列 —— 每组一次(2 加宽) + 每头每组 20 + softmax 侧(2+4nb+nG) + PV(k·nb)
    nG = (k + sc_grp(nb) - 1) // sc_grp(nb)
    return nG * (2 + 20 * nb) + (2 + 4 * nb + nG + k * nb) + gather_calls(nb, k, qd, dr, es)


def pick(sbs, qd, dr, qn, es, rows, cores=40, ub_safe=None):
    ub_safe = ub_safe if ub_safe is not None else UB_PHYS * UB_SAFE_PCT // 100
    nb_cap = max(NB_MIN, qn)
    best = None
    for nb in NB_CAND:
        if nb > nb_cap:
            continue
        units = rows * ((qn + nb - 1) // nb)
        waves = (units + cores - 1) // cores
        for take_sbs in (True, False):
            hit = None
            for k in NBLK_CAND:
                if k < NBLK_MIN:
                    continue
                if take_sbs and nb * k < sbs:
                    continue
                if ub_need(nb, k, qd, dr, es) > ub_safe:
                    continue
                hit = k
                break
            if hit:
                cost = waves * unit_calls(nb, hit, qd, dr, es)
                if best is None or cost * 20 < best[0] * 17:   # 只认 >15% 的明显更优
                    best = (cost, nb, hit, units, waves)
                break
    return best


CASES = [  # 名字, sparseBlockSize, qD, dr, Q_N, elemSize, rows  （§5.8.3 的平台形状 + big1）
    ("p1  rows=4  N1=4", 2048, 512, 64, 4, 2, 4),
    ("p2  rows=8  N1=2", 2048, 512, 64, 2, 2, 8),
    ("p4  rows=16 N1=4", 2048, 512, 64, 4, 2, 16),
    ("p6  rows=32 N1=4", 2048, 512, 64, 4, 2, 32),
    ("big1 rows=128 N1=8 fp16", 2048, 512, 64, 8, 2, 128),
    ("big1 rows=128 N1=8 fp32", 2048, 512, 64, 8, 4, 128),
]

if __name__ == "__main__":
    print(f"{'用例':26} {'nb':>3} {'n_blk':>6} {'单元':>5} {'波':>3} {'UB 需求':>9} {'cost':>7}")
    for name, sbs, qd, dr, qn, es, rows in CASES:
        cost, nb, k, units, waves = pick(sbs, qd, dr, qn, es, rows)
        print(f"{name:26} {nb:3d} {k:6d} {units:5d} {waves:3d} "
              f"{ub_need(nb, k, qd, dr, es):9,d} {cost:7d}")
        for cand in NB_CAND:
            if cand > max(NB_MIN, qn):
                continue
            for kk in NBLK_CAND:
                if kk >= NBLK_MIN and ub_need(cand, kk, qd, dr, es) <= UB_PHYS * UB_SAFE_PCT // 100:
                    break
            u = rows * ((qn + cand - 1) // cand)
            w = (u + 39) // 40
            flag = " <== 选中" if cand == nb else ""
            print(f"{'':26} {cand:3d} {kk:6d} {u:5d} {w:3d} "
                  f"{ub_need(cand, kk, qd, dr, es):9,d} {w * unit_calls(cand, kk, qd, dr, es):7d}{flag}")
