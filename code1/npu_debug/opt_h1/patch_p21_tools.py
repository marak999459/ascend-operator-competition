#!/usr/bin/env python3
# [题1/R21 工装，非提交面] p21 读数补齐：**只动 npu_debug/prof_matrix.sh**，提交面四文件一字不改。
# 原过滤只留 ^[PROF 与 === ，逐用例成绩行 `[fwd-fp16-small] ... mismatch= maxdiff=` 被丢掉，
# 于是 run_p21_h5a.sh 里那格号称带的 mismatch 字段实际永远读不到（计时与数值在同一进程里跑了，
# 只是没落进日志）。§26.8-④ 要"数值门禁先于计时"，就得让每一发的数值判据进日志。
#
# 为什么不给成绩行加 bitmis= 字段（那是更强的判据）：test_mhc_expand_npu.cpp 在**冻结面四文件
# md5 之列**（c45bec9f9a70），为了一个已有替代证据的字段去动冻结面不划算。替代证据：
#   默认 fill 输入值 = ((i*37+11)%29-14)*0.125 ⇒ 全部是 0.125 的整数倍，
#   而 TOL=1e-2 ⇒ 任何"取错行/漏写"的差都 ≥0.125 ≫ TOL ⇒ mismatch=0 在该输入族上就是位级恒等
#   （唯一混叠是 ±0，本族无 -0；漏写区预置 0xffff=NaN 由 nan_cnt 单独判）。
#   两臂各自 mismatch=0 ⇒ 两臂都逐 bit 等于同一 CPU 参考 ⇒ 传递出 on≡off。
# usage: patch_p21_tools.py check|on
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
F_PROF = os.path.join(ROOT, "npu_debug", "prof_matrix.sh")

EDITS = [
 (F_PROF,
  'MHC_PCASE=$i "$BIN" 50 64 prof2 2>&1 | grep -E "^\\[PROF|=== " | sed "s/^/c$i /"',
  'MHC_PCASE=$i "$BIN" 50 64 prof2 2>&1 | grep -E "^\\[PROF|=== |mismatch=" | sed "s/^/c$i /"'),
 (F_PROF,
  'grep -E "^\\[PROF|=== |Profiling|Failed|error" | sed "s/^/c$i /"',
  'grep -E "^\\[PROF|=== |Profiling|Failed|error|mismatch=" | sed "s/^/c$i /"'),
]

def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read().replace(b"\r\n", b"\n")).hexdigest()

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE not in ("check", "on"):
    print("usage: patch_p21_tools.py check|on"); sys.exit(1)

texts = {}
fails = 0
for path, old, new in EDITS:
    txt = texts.get(path)
    if txt is None:
        txt = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    if new in txt:
        texts[path] = txt
        print("### skip（已打过）%s" % os.path.basename(path)); continue
    n = txt.count(old)
    if n != 1:
        print("ANCHOR FAIL in %s (%d hits) :: %r" % (path, n, old[:64])); fails += 1; continue
    texts[path] = txt.replace(old, new, 1)
if fails or MODE == "check":
    print("### check edits=%d fails=%d" % (len(EDITS), fails)); sys.exit(1 if fails else 0)

for path in texts:
    bak = path + ".bak_p21"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
for path, txt in texts.items():
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    os.utime(path, None)          # 必须顶 mtime：shutil.copy2 会带回旧时间戳 ⇒ 构建漏判
    print("### wrote %-34s md5=%s" % (os.path.relpath(path, ROOT), md5(path)[:12]))
print("### p21 tools: mismatch_in_grep=%d file=%s" % (
    sum(1 for t in texts.values() for L in t.splitlines()
        if "grep -E" in L and "mismatch=" in L), os.path.basename(F_PROF)))
