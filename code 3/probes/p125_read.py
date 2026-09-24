#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P125 标定读数器：把 `p125_calib*.txt` 里"每臂每 N × 每例"的 ms 收成 Δ 表，并把 `p125_design.txt`
第 4 条那三条硬门**算成结论**（人眼看大表最容易自欺）。

⚠️ 本版对**门 A 做了一次口径订正**（先说清楚，别让它藏在数里）：
   旧写法 = "N=1 与 N=2 时间差 <0.3 % ⇒ 判被吸收 ⇒ 作废"。这一条在本发**结构上必然失败**：
   这里的 N=1 就是"一遍都不多跑"（pristine 的量），Δ(1)≈0 是**构造如此**，不是折叠的证据。
   ⇒ N=1 的真实身份是**对照组**（字节变了、工作量没变 ⇒ 它的 |Δ| 就是本场的噪声底），
     防折叠改由"每圈单价 Δ(N)/(N−1) 跨 N 是否稳定"来判（折叠 ⇒ 单价随 N 塌向 0）。
三条门（口径写在 `p125_design.txt`，这里只把数算出来）：
  门 A（噪声底）：Δ(N=1) 逐例的 |Δ| = 本场"字节不同、工作量相同"的对照 ⇒ 记为 σ；
                 σ > 1.5 % ⇒ 本地尺子没有分辨率，本场的低档位读数一律不作依据。
  门 B（防折叠）：**最敏感例**的每圈单价 u(N) = Δ(N)/(N−1) 在最大跨度上 u_max/u_min ≤ 4 ⇒ 认线性，
                 Δ 才可折回"一圈的价"；否则 ⇒ 判被折叠或被吸收，不许发货。
  门 C（选档）  ：取"七例里最差那例 Δ ≤ +15 %"的最大 N（P119/P121/P123 同一约定）。
用法：python3 p125_read.py [log ...]   （默认读 p125_calib.txt + p125_calib_hi.txt）
"""
import os
import re
import sys

REPO = "/home/fszqsn/ops_comp/ascend-operator-competition"
LOGS = sys.argv[1:] or [os.path.join(REPO, "code 3/probes/" + f)
                        for f in ("p125_calib.txt", "p125_calib_hi.txt")]

pat_arm = re.compile(r"^############ ARM=(\w)")
pat_n = re.compile(r"^##########\s+(\w) N=(\d+)")
pat_ms = re.compile(r"^\s+(\S+)\s+([0-9]+\.[0-9]+) ms\s*$")

data = {}                       # (log, arm, n) -> {case: ms}
for lg in LOGS:
    if not os.path.exists(lg):
        print("跳过（不存在）：%s" % lg)
        continue
    arm = n = None
    for line in open(lg, encoding="utf-8", errors="replace"):
        m = pat_arm.match(line)
        if m:
            arm = m.group(1)
            continue
        m = pat_n.match(line)
        if m:
            n = int(m.group(2))
            data.setdefault((os.path.basename(lg), arm or m.group(1), n), {})
            continue
        m = pat_ms.match(line.rstrip("\n"))
        if m and n is not None:
            data[(os.path.basename(lg), arm, n)][m.group(1)] = float(m.group(2))

if not data:
    sys.exit("日志里一行时间都没解析到：%s" % LOGS)

print("解析到 %d 个（场次, 臂, N）档" % len(data))
for key in sorted(data):
    lg, a, n = key
    if not data[key]:
        print("  %s 臂 %s N=%d：无时间行 ⇒ 忽略" % (lg, a, n))

# 逐（场次, 臂）出表；Δ 只在同一场内算（跨场次的 N=0 不可比）
for lg in sorted({k[0] for k in data}):
    for a in sorted({k[1] for k in data if k[0] == lg}):
        ns = sorted({k[2] for k in data if k[0] == lg and k[1] == a and data[(lg, a, k[2])]})
        if 0 not in ns:
            continue
        base = data[(lg, a, 0)]
        ladder = [x for x in ns if x > 0]
        print("\n=== 场次 %s / 臂 %s（基准 N=0，阶梯 %s）===" % (lg, a, ladder))
        print("%-9s %9s" % ("例", "基准 ms") + "".join(" %8s" % ("N=%d" % x) for x in ladder))
        worst = {}
        dmax = {}
        for c in sorted(base):
            row = []
            for x in ladder:
                v = data[(lg, a, x)].get(c)
                d = (v / base[c] - 1.0) * 100.0 if v else float('nan')
                row.append(d)
                worst.setdefault(x, []).append(d)
                dmax[c] = max(dmax.get(c, -99.0), d)
            print("%-9s %9.4f" % (c, base[c]) + "".join(" %8.2f" % d for d in row))
        print("%-9s %9s" % ("最差"  , "—") + "".join(" %8.2f" % max(worst[x]) for x in ladder))
        print("%-9s %9s" % ("中位"  , "—") + "".join(" %8.2f" % sorted(worst[x])[len(worst[x]) // 2] for x in ladder))
        if 1 in ladder:
            s_worst = max(abs(d) for d in worst[1])
            s_med = sorted(abs(d) for d in worst[1])[len(worst[1]) // 2]
            print("门 A：对照 σ（N=1 = 字节变了、工作量没变）= 逐例 |Δ| 最大 %.2f %% / 中位 %.2f %% ⇒ %s"
                  % (s_worst, s_med, "本地尺子够用" if s_worst <= 1.5 else "**σ>1.5 % ⇒ 本场低档读数不作依据**"))
        # 每圈单价 u = Δ/(N−1)，**逐例**打印：七例里只有部分对这一段敏感（r64c65/w3/p6 几乎是平的），
        # "逐例中位"会被不敏感例拉向 0 ⇒ 门判在**最敏感那一例**（Δ 最大 ⇒ 离噪声最远），中位当参考。
        # 折叠/吸收的表现 = 最敏感例的 u 随 N 塌向 0。
        sens = max(dmax, key=lambda c: dmax[c])   # 例，不是档
        print("门 B：逐例每圈单价 u(N) = Δ/(N−1)（pp/圈）｜最敏感例 = %s" % sens)
        for c in sorted(base):
            print("  %-9s" % c + "".join(" N=%-3d %6.3f" % (
                x, (data[(lg, a, x)].get(c, base[c]) / base[c] - 1) * 100 / (x - 1)) for x in ladder if x > 1))
        sc = base[sens]
        ub = [(data[(lg, a, x)][sens] / sc - 1) * 100 / (x - 1) for x in ladder if x > 1 and sens in data[(lg, a, x)]]
        um = [u for u in ub if u > 0.02]
        if len(um) >= 2:
            hi, lo = max(um), min(um)
            print("  ⇒ %s 臂 u = %s ⇒ u_max/u_min = %.2f ⇒ %s"
                  % (sens, [round(u, 3) for u in ub], hi / lo,
                     "认线性（尺子在）" if hi / lo <= 4 else "**单价随 N 塌 ⇒ 判折叠/吸收，不许发货**"))
        else:
            print("  ⇒ 可用档不足（u 全 ≤0.02 ⇒ 这一臂本地读不出单价）")
        ok = [x for x in ladder if max(worst[x]) <= 15.0]
        print("门 C：本地最差档 Δ ≤ +15 %% 的档 = %s ⇒ 候选 N = %s"
              % (ok, max(ok) if ok else "无（阶梯要往下缩）"))
