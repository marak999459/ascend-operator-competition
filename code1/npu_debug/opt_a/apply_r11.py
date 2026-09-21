# [题1/R11 工装，非提交面] 小档"成本地板"探针 + 小 tile 攒批候选。
#
# 为什么要这一轮：小档(c0 fwd-fp16-small, 96KB)实测 2.8~3.0us，三个模型给出互相矛盾的归因：
#   A) 从 c6 外推的"每道 PIPE_ALL 0.26~0.28us" => 4 道 barrier 占 1.1us，攒批值 −25~−40%
#   B) §18.3 的同尺反证：c4 每核 26 道 barrier、B=2 减半实测 +0.1/-0.2/-0.1us(零)
#      => barrier 在 <12KB 的档上近乎免费，2.8us 全是派发+序言
#   C) §14.1 的 mte2=1.1us / 每核 2 单元 => 每次 512B 载入往返 ~0.55us，
#      barrier 把它排成 4 次串行；贵的不是指令而是**序列化了的延迟**
# 一个构建同时出三支判据：base / b2s / b4s(候选) + nob/prol/empty(地板探针)。
#
#   python3 apply_r11.py base    ← 提交面 038d65a0 原样（A/B 对照，不打补丁）
#   python3 apply_r11.py b2s     ← 小 tile 走 B=2 / 环 4：c0 的 4 道 barrier -> 2 道
#   python3 apply_r11.py b4s     ← 小 tile 走 B=4 / 环 8：c0 每核正好 4 个单元 => **只剩 1 道 barrier**
#   python3 apply_r11.py nob     ← [PROBE 故意错] 小 tile 路径删掉 barrier => barrier+往返的收益上界
#   python3 apply_r11.py prol    ← [PROBE 故意错] Init 跑完、Process 直接 return => 派发+序言地板
#   python3 apply_r11.py empty   ← [PROBE 故意错] Init 与 Process 都直接 return => 纯启动地板
#
# base md5 不符 / 锚点不唯一 => 直接退出，不产生半拉子源码。只在隔离树里跑。
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KJ = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
BASE_MD5 = "038d65a08b71e1654f9dcee794950662"       # 提交 4 的 kernel（§18.4 溯源链末端）

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
#             BS_SMALL  NOB  PROL  EMPTY
CONF = {"base":  None,
        "b2s":   (2,     0,   0,    0),
        "b4s":   (4,     0,   0,    0),
        "nob":   (1,     1,   0,    0),
        "prol":  (1,     0,   1,    0),
        "empty": (1,     0,   0,    1)}
PROBE = {"nob", "prol", "empty"}
if MODE not in CONF:
    print("usage: apply_r11.py base|b2s|b4s|nob|prol|empty"); sys.exit(1)
if MODE == "base":
    print("### base = 不打补丁（提交面原样），kernel md5 应为 %s" % BASE_MD5)
    sys.exit(0)
BS_SMALL, NOB, PROL, EMPTY = CONF[MODE]

# ---- 1) 编译期常量（BS_SMALL 走 constexpr：换成运行时值会让内层循环失去展开，§18.2 sweep2 实测）----
A1 = """    static constexpr uint32_t FWD_BATCH = 2;      // 大 tile 前向每次攒 2 块、共用一道 barrier
    static constexpr uint32_t FWD_THRESH = 12288; // 单块字节数阈值：过了才攒批
"""
N1 = A1 + """    static constexpr uint32_t R11_BS_SMALL = %d;  // [R11] 小 tile 路径批大小；1 = 提交面形态
    static constexpr uint32_t R11_NOB   = %d;     // [PROBE] 1 = 小 tile 路径不打 barrier
    static constexpr uint32_t R11_PROL  = %d;     // [PROBE] 1 = Init 之后不做任何搬运
    static constexpr uint32_t R11_EMPTY = %d;     // [PROBE] 1 = 连 Init 也不做
""" % (BS_SMALL, NOB, PROL, EMPTY)

# ---- 2) 环缓冲成员：8 格恒备（未 InitBuffer 的格只是句柄，不占 UB）----
A2 = """    TBuf<TPosition::VECCALC> fwd_b3_;
"""
N2 = """    TBuf<TPosition::VECCALC> fwd_b3_;
    TBuf<TPosition::VECCALC> fwd_b4_;
    TBuf<TPosition::VECCALC> fwd_b5_;
    TBuf<TPosition::VECCALC> fwd_b6_;
    TBuf<TPosition::VECCALC> fwd_b7_;
"""

# ---- 3) Init：小 tile 路径按环深建缓冲（BS=2 => 4 格，BS=4 => 8 格）----
A3 = """            // 后两块只有攒批路径会用到；小 tile 走 BS=1，不白占 UB
            if (FwdBigTile()) {
                pipe_.InitBuffer(fwd_b2_, tiling_.dTileLen * elem_size_);
                pipe_.InitBuffer(fwd_b3_, tiling_.dTileLen * elem_size_);
            }
"""
N3 = """            // [R11] 大 tile 走 4 格；小 tile 走 2*BS_SMALL 格（BS_SMALL=1 时仍只要 2 格）
            if (FwdBigTile() || R11_BS_SMALL >= 2) {
                pipe_.InitBuffer(fwd_b2_, tiling_.dTileLen * elem_size_);
                pipe_.InitBuffer(fwd_b3_, tiling_.dTileLen * elem_size_);
            }
            if constexpr (R11_BS_SMALL >= 4) {
                if (!FwdBigTile()) {
                    pipe_.InitBuffer(fwd_b4_, tiling_.dTileLen * elem_size_);
                    pipe_.InitBuffer(fwd_b5_, tiling_.dTileLen * elem_size_);
                    pipe_.InitBuffer(fwd_b6_, tiling_.dTileLen * elem_size_);
                    pipe_.InitBuffer(fwd_b7_, tiling_.dTileLen * elem_size_);
                }
            }
"""

# ---- 4) 小 tile 支路的批大小 = BS_SMALL ----
A4 = """        if (FwdBigTile()) ProcessForwardN<FWD_BATCH>();
        else ProcessForwardN<1>();
"""
N4 = """        if (FwdBigTile()) ProcessForwardN<FWD_BATCH>();
        else ProcessForwardN<R11_BS_SMALL>();
"""

# ---- 5) 槽位表扩到 8 格（BS_SMALL<4 时该支路被 if constexpr 丢弃，不会实例化）----
A5 = """        return slot == 0 ? fwd_b0_.Get<DT_X>() : (slot == 1 ? fwd_b1_.Get<DT_X>() :
               (slot == 2 ? fwd_b2_.Get<DT_X>() : fwd_b3_.Get<DT_X>()));
"""
N5 = """        if constexpr (R11_BS_SMALL >= 4) {
            return slot == 0 ? fwd_b0_.Get<DT_X>() : (slot == 1 ? fwd_b1_.Get<DT_X>() :
                   (slot == 2 ? fwd_b2_.Get<DT_X>() : (slot == 3 ? fwd_b3_.Get<DT_X>() :
                    (slot == 4 ? fwd_b4_.Get<DT_X>() : (slot == 5 ? fwd_b5_.Get<DT_X>() :
                     (slot == 6 ? fwd_b6_.Get<DT_X>() : fwd_b7_.Get<DT_X>()))))));
        }
        return slot == 0 ? fwd_b0_.Get<DT_X>() : (slot == 1 ? fwd_b1_.Get<DT_X>() :
               (slot == 2 ? fwd_b2_.Get<DT_X>() : fwd_b3_.Get<DT_X>()));
"""

# ---- 6) [PROBE] 小 tile 支路去掉 barrier（BS==1 只出现在小 tile 路径 => 大档不受影响）----
A6 = """            PipeBarrier<PIPE_ALL>();
"""
N6 = """            if constexpr (!(R11_NOB && BS == 1)) PipeBarrier<PIPE_ALL>();
"""

# ---- 7) [PROBE] Process / Init 早退 ----
A7 = """    __aicore__ inline void Process() {
        if (task_begin_ >= task_end_) return;
"""
N7 = """    __aicore__ inline void Process() {
        if constexpr (R11_PROL || R11_EMPTY) return;   // [PROBE]
        if (task_begin_ >= task_end_) return;
"""
A8 = """    __aicore__ inline void Init(GM_ADDR x, GM_ADDR o, GM_ADDR workspace, const MhcExpandTilingData &tiling) {
        tiling_ = tiling;
"""
N8 = """    __aicore__ inline void Init(GM_ADDR x, GM_ADDR o, GM_ADDR workspace, const MhcExpandTilingData &tiling) {
        if constexpr (R11_EMPTY) return;   // [PROBE]
        tiling_ = tiling;
"""

PATCHES = [(A1, N1), (A2, N2), (A3, N3), (A4, N4), (A5, N5), (A6, N6), (A7, N7), (A8, N8)]

with open(KJ, encoding="utf-8") as f:
    txt = f.read()
before = hashlib.md5(txt.encode()).hexdigest()
if before != BASE_MD5:
    print("BASE MISMATCH: kernel md5=%s, expected %s" % (before, BASE_MD5))
    sys.exit(2)
for old, new in PATCHES:
    n = txt.count(old)
    if n != 1:
        print("ANCHOR FAIL (%d hits) :: %r" % (n, old[:56]))
        sys.exit(3)

bak = KJ + ".bak_r11base"
if not os.path.exists(bak):
    shutil.copy2(KJ, bak)
elif hashlib.md5(open(bak, "rb").read()).hexdigest() != BASE_MD5:
    print("BAK MISMATCH: %s 不是 %s" % (bak, BASE_MD5))
    sys.exit(4)

for old, new in PATCHES:
    txt = txt.replace(old, new, 1)
with open(KJ, "w", encoding="utf-8") as f:
    f.write(txt)

print("### applied R11-%s (BS_SMALL=%d NOB=%d PROL=%d EMPTY=%d)%s  kernel md5 %s -> %s" % (
    MODE, BS_SMALL, NOB, PROL, EMPTY, "  [PROBE: 故意破坏数值，只取时间]" if MODE in PROBE else "",
    before, hashlib.md5(txt.encode()).hexdigest()))
print("### sanity: PIPE_ALL=%d InitBuffer_fwd=%d ProcessForwardN=%d R11_EMPTY_guard=%d" % (
    txt.count("PipeBarrier<PIPE_ALL>()"),
    txt.count("pipe_.InitBuffer(fwd_b"),
    txt.count("ProcessForwardN<"),
    txt.count("if constexpr (R11_EMPTY) return;")))
