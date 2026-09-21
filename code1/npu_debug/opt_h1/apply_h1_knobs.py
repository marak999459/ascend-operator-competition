# -*- coding: utf-8 -*-
# [题1/R15 工装，非提交面] 给隔离树加四个运行期探针，使**一次构建**就能扫
# (核数 × 合批开关 × ROW 资格) 三维 —— 因为 A/B 变成"同一份机器码换环境变量"，
# 跨构建漂移（§20.5 量到 0.2us/7%）和"pkg 静默跑旧核"（§18.7 判据）两类坑一起消失。
#
# 为什么要它：R14 上线的合批把每核 DMA 发起数与"每核行数"解耦，于是 §14/R13 那条
# "每核至少 8KB IO 才值得开这个核"的拐点**在合批态不再成立**（code1.md §22.7）。
# 但拐点该往哪移、移多少，只有把 blk 当自变量重扫才知道 —— 所以需要一个能强制 blk 的探针。
# 顺带用 MHC_FORCE_ROW 原型化 H4（S<40 的前向今天走 STREAM，每行付 2m 条发起 + x 被读 m 次）。
#
# ⚠️ 只在隔离树用；getenv 属归因注入，**提交前整段删除**（§14.6 同一条纪律）。
#   python3 apply_h1_knobs.py on      <- 打补丁
#   python3 apply_h1_knobs.py off     <- 还原三面并复核 md5
#
# 探针语义：
#   MHC_MIN_IO=<bytes>    覆盖 min_io_per_core（走 host 原式，含 core_floor=8）
#   MHC_FORCE_BLK=<n>     硬覆盖 block_dim（可低于 core_floor，用于阶梯）
#   MHC_FORCE_ROW         允许 S<num_aiv 也走 ROW 切分（H4 原型）
#   MHC_NO_MERGE          kernel 的 MergeRows() 直接 return 0（同构建的 base 臂）
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
F_HOST = os.path.join(ROOT, "op_host", "mhc_expand.cpp")
F_KERN = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
F_HDR  = os.path.join(ROOT, "op_kernel", "mhc_expand_tiling.h")
BASE = {F_HOST: "6c1fdacdf788e09e02f0ddb23dc33318",   # 提交 7 的 host 面（R13 core_floor=8）
        F_KERN: "442c12f4448f65dfdaa1f90010f91056",   # 提交 7 的 kernel 面（R14 合批）
        F_HDR:  "f759a052a865facbb889149ea1aa37e3"}   # tiling.h，与提交 7 相同
MARK = "R15 PROBE ONLY"

# (file, old, new)
EDITS = [
 (F_HDR,
  "    uint32_t splitMode;    // 0 = ROW_SPLIT, 1 = ROW_STREAM_SPLIT(前向), 2 = ELEMENT_SPLIT\n};\n",
  "    uint32_t splitMode;    // 0 = ROW_SPLIT, 1 = ROW_STREAM_SPLIT(前向), 2 = ELEMENT_SPLIT\n"
  "    uint32_t probeNoMerge; // [R15 PROBE ONLY] 1 => MergeRows 直接返回 0（同构建 A/B 的 base 臂）\n};\n"),

 (F_HOST,
  "#include <algorithm>\n#include <cstdint>\n",
  "#include <algorithm>\n#include <cstdint>\n#include <cstdlib>   // [R15 PROBE ONLY] getenv\n"),

 (F_HOST,
  "        if (S >= num_aiv) {\n",
  "        if (S >= num_aiv || std::getenv(\"MHC_FORCE_ROW\") != nullptr) {   // [R15 PROBE ONLY] H4 原型\n"),

 (F_HOST,
  "        const uint64_t min_io_per_core = backward ? 6144 : 8192;   // 反向 6KB/核、前向 8KB/核\n",
  "        uint64_t min_io_per_core = backward ? 6144 : 8192;   // 反向 6KB/核、前向 8KB/核\n"
  "        if (const char *e = std::getenv(\"MHC_MIN_IO\")) {            // [R15 PROBE ONLY]\n"
  "            const int v = std::atoi(e);\n"
  "            if (v > 0) min_io_per_core = static_cast<uint64_t>(v);\n"
  "        }\n"),

 (F_HOST,
  "        if (core_cap < block_dim) block_dim = static_cast<uint32_t>(core_cap);\n",
  "        if (core_cap < block_dim) block_dim = static_cast<uint32_t>(core_cap);\n"
  "        // [R15 PROBE ONLY] 强制核数阶梯 + 把[关合批]位经 tiling 传给 kernel\n"
  "        if (const char *fb = std::getenv(\"MHC_FORCE_BLK\")) {\n"
  "            const int v = std::atoi(fb);\n"
  "            if (v > 0) block_dim = static_cast<uint32_t>(v);\n"
  "        }\n"
  "        tiling->probeNoMerge = std::getenv(\"MHC_NO_MERGE\") ? 1u : 0u;\n"),

 (F_KERN,
  "        if (tiling_.splitMode != 0 || tiling_.dTileNum != 1) return 0;\n",
  "        if (tiling_.probeNoMerge) return 0;   // [R15 PROBE ONLY] 同构建 base 臂\n"
  "        if (tiling_.splitMode != 0 || tiling_.dTileNum != 1) return 0;\n"),
]

def md5(p):
    # 口径与仓库一致：md5 of CR-stripped bytes（core.autocrlf=true，见 AGENT.MD）
    with open(p, "rb") as f:
        return hashlib.md5(f.read().replace(b"\r\n", b"\n")).hexdigest()

def crlf_fix(txt):
    return txt.replace("\r\n", "\n")

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE not in ("on", "off"):
    print("usage: apply_h1_knobs.py on|off"); sys.exit(1)

if MODE == "off":
    bad = 0
    for path, want in BASE.items():
        bak = path + ".bak_r15"
        cur = md5(path)
        if cur == want:
            print("### %s 已是提交面 %s" % (os.path.basename(path), want[:12])); continue
        if not os.path.exists(bak):
            print("### %s NO BACKUP (.bak_r15) 且 md5=%s != %s" % (path, cur, want)); bad = 1; continue
        shutil.copy2(bak, path)
        got = md5(path)
        print("### %s restored -> %s %s" % (os.path.basename(path), got[:12], "OK" if got == want else "MISMATCH"))
        if got != want: bad = 1
    sys.exit(1 if bad else 0)

texts, applied = {}, 0
for path, old, new in EDITS:
    txt = texts.get(path)
    first = txt is None
    if first:
        txt = crlf_fix(open(path, encoding="utf-8").read())
    if new in txt:                       # 逐编辑判幂等（不能拿全文件标记判，否则一个文件多编辑会互相吞）
        texts[path] = txt
        print("### skip（该编辑已在）%s" % os.path.relpath(path, ROOT)); continue
    n = txt.count(old)
    if n != 1:
        print("ANCHOR FAIL in %s (%d hits) :: %r" % (path, n, old[:60])); sys.exit(3)
    if first:
        base_now = md5(path)
        if base_now != BASE[path] and MARK not in txt:
            print("BASE MD5 MISMATCH %s: %s != %s" % (path, base_now, BASE[path])); sys.exit(4)
        if not os.path.exists(path + ".bak_r15"):
            shutil.copy2(path, path + ".bak_r15")
    texts[path] = txt.replace(old, new, 1)
    applied += 1
for path, txt in texts.items():
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    print("### wrote %-38s md5=%s" % (os.path.relpath(path, ROOT), md5(path)[:12]))
print("### R15 knobs applied edits=%d getenv=%d force_row=%d" % (
    applied, sum(1 for t in texts.values() for L in t.splitlines() if "getenv" in L),
    sum(1 for L in texts[F_HOST].splitlines() if "MHC_FORCE_ROW" in L)))
