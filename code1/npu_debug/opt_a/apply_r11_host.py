# [题1/R11-B 工装，非提交面] 给 op_host 打一个"强制 block_dim"探针补丁，用来重扫 (核数 × 批大小) 组合。
#
# 为什么需要它：§14.2 那条"blk 谷底在 12~16"是在 **每核一道 barrier** 的驱动下测的；R11-A 证明
# 攒批能拿 0.40us、且 empty kernel 的地板随核数以 ~96ns/核 上升（blk=16 -> 2.0us，blk=40 -> 4.3us）。
# 两者一叠加，(核数, 批大小) 的最优点必须重新联合扫，不能沿用 6KB/核 的旧拐点。
#
# ⚠️ 本补丁**只服务探针**：getenv 属归因注入，**提交前必须整段删除**（§14.6 同一条纪律），
#    且只在隔离树 optR11 里打，绝不打进 ~/ops_comp/code1。
#
#   python3 apply_r11_host.py on      ← 打补丁（MHC_FORCE_BLK=<v> 生效）
#   python3 apply_r11_host.py off     ← 还原提交面 host
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HJ = os.path.join(ROOT, "op_host", "mhc_expand.cpp")
BASE_MD5 = "61543c1b17f94cc7c99c4bf4178fc197"    # 提交 5 + §19.6 注释修订后的 host（R13 探针轮使用）
                                                # 历史：R11/R12 用的是提交 4 的 3c2dadc8f2cdfebc1d11062f674407eb
PATCHED = None  # 运行时算，见下

A1 = """#include <algorithm>
#include <cstdint>
"""
N1 = """#include <algorithm>
#include <cstdint>
#include <cstdlib>   // [R11-B PROBE ONLY] getenv
"""

A2 = """        if (core_cap < 1) core_cap = 1;
        if (core_cap < block_dim) block_dim = static_cast<uint32_t>(core_cap);
"""
N2 = A2 + """
        // [R11-B PROBE ONLY —— 提交前整段删除] 强制 block_dim，重扫 (核数 × 批大小)
        if (const char *fb = std::getenv("MHC_FORCE_BLK")) {
            const int v = std::atoi(fb);
            if (v > 0) block_dim = static_cast<uint32_t>(v);
        }
"""

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE not in ("on", "off"):
    print("usage: apply_r11_host.py on|off"); sys.exit(1)

with open(HJ, encoding="utf-8") as f:
    txt = f.read()
now = hashlib.md5(txt.encode()).hexdigest()

if MODE == "off":
    if now == BASE_MD5:
        print("### host 已是提交面 %s，无需还原" % BASE_MD5); sys.exit(0)
    bak = HJ + ".bak_r11host"
    if not os.path.exists(bak):
        print("NO BACKUP and md5=%s (expected %s)" % (now, BASE_MD5)); sys.exit(2)
    shutil.copy2(bak, HJ)
    print("### restored host -> %s" % hashlib.md5(open(HJ, "rb").read()).hexdigest())
    sys.exit(0)

if now == BASE_MD5:
    for old, new in [(A1, N1), (A2, N2)]:
        n = txt.count(old)
        if n != 1:
            print("ANCHOR FAIL (%d hits) :: %r" % (n, old[:56])); sys.exit(3)
    bak = HJ + ".bak_r11host"
    if not os.path.exists(bak):
        shutil.copy2(HJ, bak)
    for old, new in [(A1, N1), (A2, N2)]:
        txt = txt.replace(old, new, 1)
    with open(HJ, "w", encoding="utf-8") as f:
        f.write(txt)
    PATCHED = hashlib.md5(txt.encode()).hexdigest()
    print("### applied R11-host-probe  host md5 %s -> %s" % (BASE_MD5, PATCHED))
else:
    PATCHED = now
    print("### host 已是补丁态 md5=%s（跳过重复注入）" % now)

# 让 run_r11b.sh 能把补丁态记进缓存目录名：固定格式，便于 grep
print("### HOST_PROBE_MD5=%s  getenv_lines=%d" % (
    PATCHED, sum(1 for L in txt.splitlines() if "MHC_FORCE_BLK" in L)))
