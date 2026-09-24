#!/usr/bin/env python3
"""P115 读数器：同字节、把每条 gather 拆成 N 条 ⇒ 纯"调用条数"计量器。
   与 P114（字节 ×2 且调用 ×2）对照才能把"字节"和"条数"分开。
   形状一律从设备上的 .bin header 读（⛔ 不手抄），故 N1>1 案的 gather 条数按
   rows·COUNT·ceil(N1/nb) 的上界给（nb 未知 ⇒ 只报 Δ%，不报绝对单价）；
   绝对 ns/条只对 **N1=1** 的案子报（此时 nb 被 forced=1，条数确定）。
"""
import re
import subprocess
import sys

H = "devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
S = f"{__import__('os').environ['HOME']}/.atomgitdevenv/.ssh/config"
TXT = sys.argv[1] if len(sys.argv) > 1 else "p115_calib_run.txt"
CORES = 40
BYTES_PER_TOK_CMD = 3  # kb + kr + vb

meta = {}

arms = {}
cur = None
for line in open(TXT, encoding="utf-8", errors="replace"):
    m = re.match(r"^#+ ARM split=(\d+)", line)
    if m:
        cur = int(m.group(1))
        arms[cur] = {}
        continue
    t = re.match(r"^\s+([A-Za-z0-9_]+)\s+([\d.]+) ms(?: ms)?\s*$", line)
    if t and cur:
        arms[cur][t.group(1)] = float(t.group(2))

out = subprocess.run(["ssh", "-F", S, "-o", "ConnectTimeout=25", H,
                      "cd ~/sfa_real/cases && for c in g64c1 g64c4 g64c16 v64c8 r64c65 p6 big1 w3;"
                      " do printf '%s ' $c; head -c 128 $c.bin | tr -d '\\0' | tr '\\n' ' '; echo; done"],
                     capture_output=True, text=True, errors="replace", timeout=180).stdout
for line in out.splitlines():
    mm = re.match(r"^(\S+)\s+SFA_CASE \d+ B (\d+) S1 (\d+) S2 (\d+) N1 (\d+) D (\d+) SBS (\d+) COUNT (\d+)", line)
    if mm:
        k = "B S1 S2 N1 D SBS COUNT".split()
        meta[mm.group(1)] = dict(zip(k, [int(mm.group(i)) for i in range(2, 9)]))
        meta[mm.group(1)]["name"] = mm.group(1)

print(f"{'case':9s} {'B':>2s} {'S1':>6s} {'S2':>7s} {'N1':>2s} {'CNT':>6s} {'tok':>8s} "
      f"{'base ms':>9s} {'x2 Δ%':>7s} {'x4 Δ%':>7s} {'x4/x2':>6s} {'ns/条(核均)':>11s}")
for name, e in meta.items():
    if name not in arms[1]:
        continue
    tok = e["B"] * e["S1"] * min(e["COUNT"], e["S2"])
    b, s2, s4 = arms[1][name], arms[2][name], arms[4][name]
    d2, d4 = (s2 / b - 1) * 100, (s4 / b - 1) * 100
    per = "  n/a"
    if e["N1"] == 1 and e["SBS"] == 1:
        cmd = BYTES_PER_TOK_CMD * tok
        per = f"{(s2 - b) * 1e6 * CORES / cmd:6.2f}"
    print(f"{name:9s} {e['B']:2d} {e['S1']:6d} {e['S2']:7d} {e['N1']:2d} {e['COUNT']:6d} {tok:8d} "
          f"{b:9.4f} {d2:7.2f} {d4:7.2f} {d4 / d2 if d2 else 0:6.2f} {per:>11s}")

print("\nP114（字节 ×2 = 调用 ×2）同案读数，供对照：")
PLAT = "平台：Δ计算×2 = +6.1 % / Δgather×2(字节+条数) = +17.8 %  ⇒ P115(split=2) 若 ≈0 ⇒ 纯字节/带宽；若 ≈17.8 ⇒ 纯发射"
print("  g64c1 +13.61 / g64c4 +15.68 / g64c16 +14.87 / v64c8 +20.74 / r64c65 +22.52 / p6 +6.10 / big1 +3.87")
print("  ⚠️ 那一列的 gather 臂是 P113/P114 的'整条重发一遍'（L2 已热），本列是'原地拆半'（长度也变了）")
print("  ⇒ 本表 Δ 随长度变短而**超线性**（x4/x2 > 2）是已知混淆：拆半同时动了条数与每条字节数。")
print(PLAT)
