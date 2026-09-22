# -*- coding: utf-8 -*-
# [题1/R15~R16 工装，非提交面] 给隔离树加运行期探针，使**一次构建**就能扫
# (核数 × 合批开关 × ROW 资格) 三维 —— 因为 A/B 变成"同一份机器码换环境变量"，
# 跨构建漂移（§20.5 量到 0.2us/7%）和"pkg 静默跑旧核"（§18.7 判据）两类坑一起消失。
#
# 为什么要它：R14 上线的合批把每核 DMA 发起数与"每核行数"解耦，于是 §14/R13 那条
# "每核至少 8KB IO 才值得开这个核"的拐点**在合批态不再成立**（code1.md §22.7）。
# 但拐点该往哪移、移多少，只有把 blk 当自变量重扫才知道 —— 所以需要一个能强制 blk 的探针。
#
# 🔧 **R16 修订（本节就是 code1.md §23.14-8 记的那笔工装债，动手前先读）**：
#   旧版 BASE 只认提交 6/7 的 host `6c1fdacdf788`，任何新面一律 rc=4 整段退出（这条失败方式是好的：
#   实测过，它不会留下半截补丁）。同时两条锚点随代码演进死了 —— `if (S >= num_aiv) {`（R15 起加了
#   `|| merge_ok`）与 `const uint64_t min_io_per_core = backward ? 6144 : 8192;`（R15 被 merge_cap 替掉、
#   R16 反向支又换成核数律）。⇒ 现在锚到**提交 10 的面**，并把 MHC_MIN_IO 换成两个仍有物理意义的旋钮：
#   MHC_CORE_FLOOR（尾句那个无条件 8）与 MHC_MERGE_FLOOR（合批段的 8 初值）。
#   ⚠️ 还有一条：探针现在**同时打在 op_host 与 npu_debug 的镜像式上**。理由不是"怕哪条路没生效"，
#   而是这两处一旦只改一边，harness 打印的 `blk=` 与设备实际上报就会分家 —— 而"拿 harness 打印的
#   blk 当设备核数"是 §19.6 明写过的那类错。两边同步 ⇒ 打印值可以当 canary 读。
#
# ⚠️ 只在隔离树用；getenv 属归因注入，**提交前整段删除**（§14.6 同一条纪律）。
#   python3 apply_h1_knobs.py on      <- 打补丁
#   python3 apply_h1_knobs.py off     <- 还原并复核 md5
#
# 探针语义：
#   MHC_FORCE_BLK=<n>     硬覆盖 block_dim（可低于任何下界，用于阶梯）—— 主旋钮
#   MHC_CORE_FLOOR=<n>    把尾句 `if (core_cap < 8)` 的 8 换掉（H1″：下界放开到 1~4 的原式检验）
#   MHC_MERGE_FLOOR=<n>   把 sqrt 合批律的**初值** 8 换掉（之后仍会被 while 长上去，只在 io 小段有效）
#   MHC_FORCE_ROW         允许 S<num_aiv 且不合批也走 ROW 切分（H4 原型）
#   MHC_NO_MERGE          kernel 的 MergeRows() 直接 return 0（同构建的 base 臂）
#   MHC_SHAPE=<fwd|bwd>,<fp16|bf16>,<S>,<D>,<m>
#                         R17：prof2 分支里换掉整条形状 ⇒ 新形状只改环境变量、不再重编译。
#                         配套仍需 MHC_PCASE=<合法下标>(只用来取默认 reps) + MHC_REPS=<n>。
#                         为什么必须有它：R17 之前每加一档形状就要动 pcs[] 表 + 重建（一次 6~8 分钟），
#                         而反向 µs 档（case5 是分数最贵的一条）的 blk<8 从来没进过扫描 ——
#                         要扫 4~5 条形状 × 8 档核数，重编译的时间比扫描本身还长。
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
F_HOST = os.path.join(ROOT, "op_host", "mhc_expand.cpp")
F_KERN = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
F_HDR  = os.path.join(ROOT, "op_kernel", "mhc_expand_tiling.h")
F_HARN = os.path.join(ROOT, "npu_debug", "test_mhc_expand_npu.cpp")
BASE = {F_HOST: "f5b0c733aba4d57b5af627663911fa6d",   # 在榜面 = 提交 10/12 的 host（反向 io/12288 律，无发起数上界；提交 11 那一面已撤回，见 code1.md §23.20/§23.23）
        F_KERN: "442c12f4448f65dfdaa1f90010f91056",   # 提交 7~12 的 kernel 面（R14 合批），六版逐字节相同
        F_HDR:  "f759a052a865facbb889149ea1aa37e3",   # tiling.h
        F_HARN: "c45bec9f9a70ad1dfac31b9fad8f3546"}   # 真机 harness（镜像式所在；提交 11 的 dTileNum==1 域限制随之撤回）
MARK = "PROBE ONLY"

# (file, old, new)
EDITS = [
 # ---- tiling 头：给 kernel 一个"关合批"的位 ----
 (F_HDR,
  "    uint32_t splitMode;    // 0 = ROW_SPLIT, 1 = ROW_STREAM_SPLIT(前向), 2 = ELEMENT_SPLIT\n};\n",
  "    uint32_t splitMode;    // 0 = ROW_SPLIT, 1 = ROW_STREAM_SPLIT(前向), 2 = ELEMENT_SPLIT\n"
  "    uint32_t probeNoMerge; // [PROBE ONLY] 1 => MergeRows 直接返回 0（同构建 A/B 的 base 臂）\n};\n"),

 # ---- host：include + 三个旋钮 + 路由 + 强制 blk ----
 (F_HOST,
  "#include <algorithm>\n#include <cstdint>\n",
  "#include <algorithm>\n#include <cstdint>\n#include <cstdlib>   // [PROBE ONLY] getenv\n"),

 (F_HOST,
  "        if (S >= num_aiv || merge_ok) {\n",
  "        if (S >= num_aiv || merge_ok || std::getenv(\"MHC_FORCE_ROW\") != nullptr) {   // [PROBE ONLY] H4 原型\n"),

 (F_HOST,
  "        uint64_t merge_cap = 8;\n",
  "        uint64_t merge_cap = 8;\n"
  "        if (const char *e = std::getenv(\"MHC_MERGE_FLOOR\")) {        // [PROBE ONLY] sqrt 律初值\n"
  "            const int v = std::atoi(e);\n"
  "            if (v > 0) merge_cap = static_cast<uint64_t>(v);\n"
  "        }\n"),

 (F_HOST,
  "        if (core_cap < 8) core_cap = 8;\n",
  "        uint64_t core_floor = 8;                                        // [PROBE ONLY] R13 那条无条件下界\n"
  "        if (const char *e = std::getenv(\"MHC_CORE_FLOOR\")) {\n"
  "            const int v = std::atoi(e);\n"
  "            if (v > 0) core_floor = static_cast<uint64_t>(v);\n"
  "        }\n"
  "        if (core_cap < core_floor) core_cap = core_floor;\n"),

 (F_HOST,
  "        if (core_cap < block_dim) block_dim = static_cast<uint32_t>(core_cap);\n",
  "        if (core_cap < block_dim) block_dim = static_cast<uint32_t>(core_cap);\n"
  "        // [PROBE ONLY] 强制核数阶梯 + 把[关合批]位经 tiling 传给 kernel\n"
  "        if (const char *fb = std::getenv(\"MHC_FORCE_BLK\")) {\n"
  "            const int v = std::atoi(fb);\n"
  "            if (v > 0) block_dim = static_cast<uint32_t>(v);\n"
  "        }\n"
  "        tiling->probeNoMerge = std::getenv(\"MHC_NO_MERGE\") ? 1u : 0u;\n"),

 # ---- kernel：同构建 base 臂 ----
 (F_KERN,
  "        if (tiling_.splitMode != 0 || tiling_.dTileNum != 1) return 0;\n",
  "        if (tiling_.probeNoMerge) return 0;   // [PROBE ONLY] 同构建 base 臂\n"
  "        if (tiling_.splitMode != 0 || tiling_.dTileNum != 1) return 0;\n"),

 # ---- harness 镜像式：与 host 同步，否则打印的 blk 会骗人（§19.6） ----
 (F_HARN,
  "    if (S >= num_aiv || merge_ok) {\n",
  "    if (S >= num_aiv || merge_ok || std::getenv(\"MHC_FORCE_ROW\") != nullptr) {   // [PROBE ONLY] 与 host 同步\n"),

 (F_HARN,
  "    uint64_t merge_cap = 8;   // 下界 8：§23.8b（提交 8 取 4 时平台唯一出带的一条 +38.8%）\n",
  "    uint64_t merge_cap = 8;   // 下界 8：§23.8b（提交 8 取 4 时平台唯一出带的一条 +38.8%）\n"
  "    if (const char *e = std::getenv(\"MHC_MERGE_FLOOR\")) {        // [PROBE ONLY] 与 host 同步\n"
  "        const int v = std::atoi(e);\n"
  "        if (v > 0) merge_cap = (uint64_t)v;\n"
  "    }\n"),

 (F_HARN,
  "    if (core_cap < 8) core_cap = 8;   // 合批段的 merge_cap 已 >=8 ⇒ 与 host 一样不必再分向\n",
  "    uint64_t core_floor = 8;                                        // [PROBE ONLY] 与 host 同步\n"
  "    if (const char *e = std::getenv(\"MHC_CORE_FLOOR\")) {\n"
  "        const int v = std::atoi(e);\n"
  "        if (v > 0) core_floor = (uint64_t)v;\n"
  "    }\n"
  "    if (core_cap < core_floor) core_cap = core_floor;\n"),

 (F_HARN,
  "    if (core_cap < block_dim) block_dim = (uint32_t)core_cap;\n",
  "    if (core_cap < block_dim) block_dim = (uint32_t)core_cap;\n"
  "    if (const char *fb = std::getenv(\"MHC_FORCE_BLK\")) {           // [PROBE ONLY] 与 host 同步\n"
  "        const int v = std::atoi(fb);\n"
  "        if (v > 0) block_dim = (uint32_t)v;\n"
  "    }\n"),

 # ---- harness prof2 分支：环境变量换形状（R17，加一档形状不必重编译） ----
 (F_HARN,
  "        RUNP(c.nm, c.bwd, c.dt, c.S, c.D, c.m, reps);\n",
  "        if (const char *shp = getenv(\"MHC_SHAPE\")) {   // [PROBE ONLY] 形如 bwd,fp16,64,256,2\n"
  "            const int b = (std::strncmp(shp, \"bwd\", 3) == 0);\n"
  "            int dtype = DT_FP16; unsigned Sh = 0, Dh = 0, mh = 0;\n"
  "            const char *q = std::strchr(shp, ',');\n"
  "            if (q) {\n"
  "                if (std::strncmp(q + 1, \"bf16\", 4) == 0) dtype = DT_BF16;\n"
  "                q = std::strchr(q + 1, ',');\n"
  "            }\n"
  "            if (q && sscanf(q + 1, \"%u,%u,%u\", &Sh, &Dh, &mh) == 3 && Sh && Dh && mh) {\n"
  "                RUNP(b ? \"probe-bwd\" : \"probe-fwd\", b != 0, dtype, Sh, Dh, mh, reps);\n"
  "            } else {\n"
  "                printf(\"MHC_SHAPE must be fwd|bwd,fp16|bf16,S,D,m\\n\");\n"
  "                return 2;\n"
  "            }\n"
  "        } else {\n"
  "            RUNP(c.nm, c.bwd, c.dt, c.S, c.D, c.m, reps);\n"
  "        }\n"),
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
print("### R16 knobs applied edits=%d getenv=%d probe_blk=%d" % (
    applied,
    sum(1 for t in texts.values() for L in t.splitlines() if "getenv" in L and MARK in L),
    sum(1 for t in texts.values() for L in t.splitlines() if "MHC_FORCE_BLK" in L)))
