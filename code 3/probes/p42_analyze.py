#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P42 读数分析：按"有效 token 数 d"分组，列出四档强制档 + AUTO 的实测时间，
算 AUTO 相对全场最快的后悔值，并和离线复刻（p38_model_fit.pick 用 S2=d 模拟"host 看得见有效项"）对账。
判读口径：§15.55(7) 预测 —— 只有 d ≲ 2×n_blk（≈96）时 chunk 数会饱和、盲区才可能翻档。
"""
import importlib.util, io, contextlib, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else '/tmp/p42_density.txt'

spec = importlib.util.spec_from_file_location('m', os.path.join(HERE, 'p38_model_fit.py'))
mod = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(mod)

pat = re.compile(r'^(AUTO|G) (\w+)(?: nb=(\d+) kb=(\d+) ks=(\d+))? :: .*?'
                 r'SFA_PICK nb=(\d+) nblk=(\d+) ks=(\d+) 平均 ([\d.]+) ms')
DENS = {'d64': 64, 'd256': 256, 'd1024': 1024, 'd2048': 2048}
best = {}     # (case, arm) -> ms ；arm = (nb, k, ks)
auto = {}     # case -> (arm, ms)
for line in open(SRC, encoding='utf-8'):
    m = pat.match(line.strip())
    if not m:
        continue
    kind, cs, fnb, fk, fks, pnb, pk, pks, t = m.groups()
    arm = (int(pnb), int(pk), int(pks))
    t = float(t)
    if kind == 'AUTO':
        cur = auto.get(cs)
        if cur is None or t < cur[1]:
            auto[cs] = (arm, t)
    else:
        key = (cs, arm)
        best[key] = min(best.get(key, 1e9), t)

print(f"{'案':6} {'d(有效token)':>10} {'AUTO档':>10} {'AUTO ms':>9} {'最快档':>10} {'最快 ms':>9} {'亏':>7}  {'理想模型(d)选档':>14}")
for cs, d in DENS.items():
    if cs not in auto:
        print(f"{cs:6} {d:>10} —— 无读数")
        continue
    a, at = auto[cs]
    arms = {arm: t for (c, arm), t in best.items() if c == cs}
    if not arms:
        print(f"{cs:6} {d:>10} {a[0]}/{a[1]}/ks{a[2]} {at:.4f}  无强制档读数")
        continue
    barma = min(arms, key=lambda x: arms[x])
    bt = arms[barma]
    ideal = mod.pick(128, 8, 2048, 1, 1024, mode='new', S2=d)
    print(f"{cs:6} {d:>10} {a[0]:>3}/{a[1]:<2}/ks{a[2]} {at:>9.4f} "
          f"{barma[0]:>3}/{barma[1]:<2}/ks{barma[2]} {bt:>9.4f} "
          f"{(at/bt-1)*100:>+6.1f} %  {ideal[1]:>3}/{ideal[2]:<2}/ks{ideal[3]}"
          + ('   <<< AUTO 比最快档慢 >3 %' if at / bt - 1 > 0.03 else ''))
    for arm in sorted(arms):
        note = ''
        if arm != barma:
            note = f"（比最快慢 {(arms[arm]/bt-1)*100:+.1f} %）"
        print(f"       nb={arm[0]:<2} k={arm[1]:<3} ks={arm[2]}  {arms[arm]:.4f} ms {note}")
