#!/usr/bin/env python3
"""P113 读数器：从 p113_proxy_run.txt 抽 header + pick + 三臂时间，算两件事 ——
   ① 两个惰性计量器各自的 Δ%（＝该段占该档墙的比例，加工作 1:1 涨已由 P110 证）；
   ② 该档的 **DMA 请求** 吞吐 = 每 launch 请求字节 / 时间，和对 KV 池的**复用度**（请求/池）。
判据（写在 p113_proxy.sh 头上）：Δ_计算 ≤10 % 且 Δ_gather ≥14 % ⇒ 该族是平台的离线代理。
⚠️ 比值 Δ_计算/Δ_gather 才是**与稀释无关**的诊断量（平台/本地各有一个未知的"墙外常数"）：
   平台 = 6.1/17.8 = 0.34，本地 w 族 ≈ 11，本次靶族见输出。
形状一律从 .bin header 与 kernel 自己打的 pick 行读，⛔ 不手抄。
"""
import re
import sys

TXT = sys.argv[1] if len(sys.argv) > 1 else "p113_proxy_run.txt"
PLAT = (6.1, 17.8)   # P112 / P114 平台侧合读数（%）

blocks = {}
cur = None
for line in open(TXT, encoding="utf-8", errors="replace"):
    m = re.match(r"^#+ ARM (\w+)", line)
    if m:
        cur = m.group(1)
        blocks[cur] = {}
        continue
    if cur is None:
        continue
    t = re.match(r"^\s+([A-Za-z0-9_]+)\s+([\d.]+) ms(?: ms)?\s*$", line)
    if t:
        blocks[cur].setdefault(t.group(1), {})["t"] = float(t.group(2))
        continue
    h = re.search(r"case=(\S+?) B=(\d+) S1=(\d+) S2=(\d+) N1=(\d+) D=(\d+) SBS=(\d+) COUNT=(\d+)", line)
    if h:
        name = re.sub(r"\.bin$", "", h.group(1))
        e = blocks[cur].setdefault(name, {})
        e.update(dict(zip("B S1 S2 N1 D SBS COUNT".split(), [int(h.group(i)) for i in range(2, 9)])))
    p = re.search(r"pick\[nb=(\d+) nblk=(\d+) ks=(\d+) sbs=(\d+) rows=(\d+) count=(\d+)", line)
    if p:
        nm = re.sub(r"\.bin$", "", h.group(1)) if h else re.match(r"^\s*([A-Za-z0-9_]+)", line)
        nm = nm if isinstance(nm, str) else (nm.group(1) if nm else None)
        if nm:
            e = blocks[cur].setdefault(nm, {})
            e.update(dict(zip("nb nblk ks".split(), [int(p.group(i)) for i in (1, 2, 3)])))
    if "不**逐位一致**" in line or "golden 缺失" in line:
        nm = re.match(r"^\s*([A-Za-z0-9_]+)", line)
        if nm:
            blocks[cur].setdefault(nm.group(1), {})["inert"] = False
    elif "逐位一致" in line:
        nm = re.match(r"^\s*([A-Za-z0-9_]+)", line)
        if nm:
            blocks[cur].setdefault(nm.group(1), {})["inert"] = True

names = [n for n in blocks.get("base", {})
         if all(n in blocks[a] and "t" in blocks[a][n] for a in ("base", "score", "gather")) and "B" in blocks["base"][n]]
print("%-13s %-27s %-13s %8s %8s %8s %6s %7s %6s %s" %
      ("case", "shape", "pick(nb,blk,ks)", "base ms", "Δ计算×2", "Δgather", "比值", "请求GB/s", "复用", "代理?"))
for n in sorted(names, key=lambda x: -blocks["base"][x]["t"]):
    b = blocks["base"][n]
    ds = 100.0 * (blocks["score"][n]["t"] - b["t"]) / b["t"]
    dg = 100.0 * (blocks["gather"][n]["t"] - b["t"]) / b["t"]
    tok = min(b["COUNT"], b["S2"])
    grp = (b["N1"] + b.get("nb", 1) - 1) // b.get("nb", 1)
    req = b["B"] * b["S1"] * grp * tok * (2 * b["D"] + 64) * 2.0
    pool = b["B"] * b["S2"] * (2 * b["D"] + 64) * 2.0
    verdict = "✅" if (ds <= 10 and dg >= 14) else ("◑方向对" if dg > ds else "✗")
    inert = all(blocks[a][n].get("inert", True) for a in ("base", "score", "gather"))
    print("%-13s B%d r%d N%d S%d D%d sbs%d c%d  nb=%-2d blk=%-3d ks=%-2d %8.4f %+7.2f%% %+7.2f%% %6.2f %7.1f %6.1fx %s %s" %
          (n, b["B"], b["S1"], b["N1"], b["S2"], b["D"], b["SBS"], b["COUNT"],
           b.get("nb", 0), b.get("nblk", 0), b.get("ks", 0), b["t"], ds, dg,
           (ds / dg if dg else float("nan")), req / (b["t"] * 1e6), req / pool, verdict,
           "" if inert else "⚠️输出不一致"))
print("\n参照：平台六点 Δ计算×2 = +%.1f %%、Δgather = +%.1f %% ⇒ 比值 %.2f（gather 那条是下界）" % (PLAT[0], PLAT[1], PLAT[0] / PLAT[1]))
