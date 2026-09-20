# [题1/R10 工装，非提交面] 前向 barrier 批处理：每批 B 块共用一道 PIPE_ALL，环给到 2B 块。
#
# 为什么这条路不需要手写事件（§17 已经把事件判死）：
#   PIPE_ALL 是全排空 ⇒ 它同时买断 RAW（等本批 copy-in 落地）和 WAR（等上一批 copy-out 读完）。
#   唯一会被它误伤的是"上一批的 MTE3 还没读完，本批的 MTE2 就覆写同一格"。
#   把环开到 2B 格，本批用的格永远是上一批没在用的格 ⇒ 序的语义一条不少，barrier 却少一半，
#   而且上一批 MTE3 的吐出与本批 MTE2 的灌入第一次真正并行（这是 §14.9 那块 34% 的理论肉）。
#
#   python3 apply_r10.py b1      ← B=1/RING=2：语义与采纳版逐条等价，作"驱动改写本身值不值"的对照
#   python3 apply_r10.py b1r4    ← B=1/RING=4：归因对照——medium 上那 +0.8% 是"环开深"还是"攒批"造成的
#   python3 apply_r10.py b2      ← B=2/RING=4：候选
#   python3 apply_r10.py ad      ← 运行时按单块字节数选 B：小 tile 走 B=1、大 tile 走 B=2/RING=4
#   python3 apply_r10.py fin     ← ad + 删掉已死的 ForwardOneBlock/slot_（= 拟提交形态，必须自己过门禁）
# 锚点不唯一 / base md5 不符 => 直接退出，不产生半拉子源码。只在隔离树里跑。
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KJ = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
BASE_MD5 = "daf2b8ed3995d4698c435c2d42d67b23"   # 采纳版（A1+A2，= 提交 3 的 kernel）

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
CONF = {"b1": (1, 2, 0, 0), "b1r4": (1, 4, 0, 0), "b2": (2, 4, 0, 0),
        "ad": (2, 4, 1, 0), "fin": (2, 4, 1, 1)}
if MODE not in CONF:
    print("usage: apply_r10.py b1|b1r4|b2|ad|fin"); sys.exit(1)
BATCH, RING, ADAPT, CLEAN = CONF[MODE]

# ---- 1) 成员：环缓冲 + 编译期批次参数 ----
A1 = """    TBuf<TPosition::VECCALC> fwd_b0_;
    TBuf<TPosition::VECCALC> fwd_b1_;
"""
N1 = """    static constexpr uint32_t FWD_BATCH = %d;
    static constexpr uint32_t FWD_RING = %d;
    static constexpr uint32_t FWD_ADAPT = %d;
    static constexpr uint32_t FWD_THRESH = 12288;   // 单块 >= 12KB 才值得攒批（c4=8KB / c6=14KB）
    TBuf<TPosition::VECCALC> fwd_b0_;
    TBuf<TPosition::VECCALC> fwd_b1_;
    TBuf<TPosition::VECCALC> fwd_b2_;
    TBuf<TPosition::VECCALC> fwd_b3_;
""" % (BATCH, RING, ADAPT)

# ---- 2) Init：按环深建缓冲（自适应时小 tile 只建 2 块，不白占 UB） ----
A2 = """            pipe_.InitBuffer(fwd_b0_, tiling_.dTileLen * elem_size_);
            pipe_.InitBuffer(fwd_b1_, tiling_.dTileLen * elem_size_);
"""
N2 = """            pipe_.InitBuffer(fwd_b0_, tiling_.dTileLen * elem_size_);
            pipe_.InitBuffer(fwd_b1_, tiling_.dTileLen * elem_size_);
            if constexpr (FWD_RING > 2) {
                if constexpr (FWD_ADAPT == 0) {
                    pipe_.InitBuffer(fwd_b2_, tiling_.dTileLen * elem_size_);
                    pipe_.InitBuffer(fwd_b3_, tiling_.dTileLen * elem_size_);
                } else if (tiling_.dTileLen * elem_size_ >= FWD_THRESH) {
                    pipe_.InitBuffer(fwd_b2_, tiling_.dTileLen * elem_size_);
                    pipe_.InitBuffer(fwd_b3_, tiling_.dTileLen * elem_size_);
                }
            }
"""

# ---- 3) Process 前向改走批处理驱动 ----
A3 = """    __aicore__ inline void Process() {
        if (task_begin_ >= task_end_) return;
"""
N3 = """    __aicore__ inline void Process() {
        if (task_begin_ >= task_end_) return;
        if constexpr (!BACKWARD) { ProcessForward(); return; }   // [R10]
"""

# ---- 4) 摘掉循环里的前向分支（该分支只剩 BACKWARD=true 的 discarded 位置） ----
A4 = """            } else {
                // 前向：ROW -> task=s(全k); STREAM -> task=s*m+k(单k全jt); ELEMENT -> task=s*m*dTileNum+k*dTileNum+jt
                uint32_t s_idx, k, jt;
                if (smode == 0) { s_idx = t; k = m; jt = d_tile_num; } // k=m 表示全k
                else if (smode == 1) { s_idx = t / m; k = t % m; jt = d_tile_num; }
                else { uint32_t stride = m * d_tile_num; s_idx = t / stride; uint32_t r = t % stride; k = r / d_tile_num; jt = r % d_tile_num; }
                if (jt == d_tile_num) {
                    for (uint32_t j = 0; j < d_tile_num; ++j) {
                        const uint32_t cur_h = (j == d_tile_num - 1) ? tiling_.dTailLen : tiling_.dTileLen;
                        ForwardOneBlock(s_idx, j, cur_h, k);
                    }
                } else {
                    const uint32_t cur_h = (jt == d_tile_num - 1) ? tiling_.dTailLen : tiling_.dTileLen;
                    ForwardOneBlock(s_idx, jt, cur_h, k);
                }
            }
        }
    }
"""
N4 = """            }
        }
    }
"""

# ---- 5) 新驱动 + 三个小助手，插在反向单块之前 ----
A5 = """    // 反向单块：逐 m 读 o_grad[i, k, jt] -> Cast 到 float 累加 -> Cast 回原 dtype 写 x_grad[i, jt]
"""
N5 = """    // [R10] 前向批处理驱动：每次攒 BS 个"块"，一起灌 -> 一道 barrier -> 一起吐。
    // 环深 = 2*BS ⇒ 本批写入的格必然不是上一批 MTE3 正在读的格，
    // 于是"上一批的吐出"和"本批的灌入"可以并行，而序的强度与逐块 barrier 完全一样。
    // BS 走模板参数而不是运行时变量：实测把 BS 变成运行时值会让 nb 循环失去展开，
    // medium 上 13.0 -> 13.6us（sweep2 的 b1/ad 同形对照）⇒ 自适应必须编译期分派。
    template <uint32_t BS>
    __aicore__ inline void ProcessForwardN() {
        constexpr uint32_t MASK = BS * 2 - 1;
        uint32_t t = task_begin_, j = 0, r = 0;
        while (t < task_end_) {
            uint32_t ii[BS], jj[BS], hh[BS], kk[BS];
            uint32_t nb = 0;
            while (nb < BS) {
                if (!NextUnit(t, j, ii[nb], jj[nb], hh[nb], kk[nb])) break;
                ++nb;
            }
            if (nb == 0) break;
            for (uint32_t u = 0; u < nb; ++u) FwdIn((r + u) & MASK, ii[u], jj[u], hh[u]);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t u = 0; u < nb; ++u) FwdOut((r + u) & MASK, ii[u], jj[u], hh[u], kk[u]);
            r += nb;
        }
    }

    __aicore__ inline bool FwdBigTile() const {
        return tiling_.dTileLen * elem_size_ >= FWD_THRESH;   // 单块 >= 12KB 才值得攒批
    }

    __aicore__ inline void ProcessForward() {
        if constexpr (FWD_ADAPT == 0) {
            ProcessForwardN<FWD_BATCH>();
        } else if (FwdBigTile()) {
            ProcessForwardN<FWD_BATCH>();     // 大 tile：攒批赚（large −2.4%）
        } else {
            ProcessForwardN<1>();             // 小 tile：攒批亏（medium +0.4us）
        }
    }

    // 取下一个"块"= (行 i, d 方向第 jt 块, 高 cur_h, 副本上限 k_limit)。
    // 任务解码与采纳版一致：ROW -> task=s(全k全jt); STREAM -> task=s*m+k; ELEMENT -> task=s*m*num+jt
    __aicore__ inline bool NextUnit(uint32_t &t, uint32_t &j, uint32_t &i, uint32_t &jt,
                                    uint32_t &cur_h, uint32_t &k_limit) {
        const uint32_t d_tile_num = tiling_.dTileNum;
        const uint32_t m = tiling_.m;
        const uint32_t smode = tiling_.splitMode;
        if (t >= task_end_) return false;
        uint32_t s_idx, k, all_j;
        if (smode == 0) { s_idx = t; k = m; all_j = 1; }
        else if (smode == 1) { s_idx = t / m; k = t % m; all_j = 1; }
        else {
            uint32_t stride = m * d_tile_num;
            s_idx = t / stride; uint32_t r = t % stride;
            k = r / d_tile_num; jt = r % d_tile_num; all_j = 0;
        }
        if (all_j) {
            jt = j;
            if (++j >= d_tile_num) { j = 0; ++t; }
        } else {
            ++t;
        }
        i = s_idx;
        k_limit = k;
        cur_h = (jt == d_tile_num - 1) ? tiling_.dTailLen : tiling_.dTileLen;
        return true;
    }

    __aicore__ inline auto FwdBuf(uint32_t r) {
        if constexpr (FWD_RING == 2) {
            return (r & 1) ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        }
        const uint32_t s = r % FWD_RING;
        return s == 0 ? fwd_b0_.Get<DT_X>() : (s == 1 ? fwd_b1_.Get<DT_X>() :
               (s == 2 ? fwd_b2_.Get<DT_X>() : fwd_b3_.Get<DT_X>()));
    }

    __aicore__ inline void FwdIn(uint32_t r, uint32_t i, uint32_t jt, uint32_t cur_h) {
        const int64_t src_off = static_cast<int64_t>(i) * tiling_.D + jt * tiling_.dTileLen;
        DataCopyExtParams cp{1, static_cast<uint32_t>(cur_h * static_cast<int32_t>(sizeof(DT_X))), 0, 0, 0};
        DataCopyPadExtParams<DT_X> pp{false, 0, 0, static_cast<DT_X>(0)};
        DataCopyPad(FwdBuf(r), x_gm_[src_off], cp, pp);
    }

    __aicore__ inline void FwdOut(uint32_t r, uint32_t i, uint32_t jt, uint32_t cur_h, uint32_t k_limit) {
        DataCopyExtParams cp{1, static_cast<uint32_t>(cur_h * static_cast<int32_t>(sizeof(DT_X))), 0, 0, 0};
        auto x_local = FwdBuf(r);
        const uint32_t k_begin = (k_limit == tiling_.m) ? 0 : k_limit;
        const uint32_t k_end   = (k_limit == tiling_.m) ? tiling_.m : k_limit + 1;
        for (uint32_t k = k_begin; k < k_end; ++k) {
            const int64_t dst_off = static_cast<int64_t>(i) * tiling_.m * tiling_.D +
                                    static_cast<int64_t>(k) * tiling_.D + jt * tiling_.dTileLen;
            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
    }

    // 反向单块：逐 m 读 o_grad[i, k, jt] -> Cast 到 float 累加 -> Cast 回原 dtype 写 x_grad[i, jt]
"""

PATCHES = [(A1, N1), (A2, N2), (A3, N3), (A4, N4), (A5, N5)]

# ---- 6) fin 专用：删掉 A3/A4 之后已经没人调用的 ForwardOneBlock 与 slot_ ----
# 为什么单独一模态：提交面不该留死函数；但它改了源码就必须自己重过门禁+重测，
# 不能拿 ad 的读数代替 fin 的读数（本项目已因"跑的是另一份二进制"废过一整轮，§17.2）。
A6 = """    // 前向单块：读 x[i, jt] 一次，写 k_begin..k_end-1 个副本
    // k=m 表示全 k（ROW 模式 UB 复用），k<m 表示只写第 k 个（STREAM/ELEMENT 模式拆核）
    __aicore__ inline void ForwardOneBlock(uint32_t i, uint32_t jt, uint32_t cur_h, uint32_t k_limit) {
        const int64_t src_off = static_cast<int64_t>(i) * tiling_.D + jt * tiling_.dTileLen;
        DataCopyExtParams cp{1, static_cast<uint32_t>(cur_h * static_cast<int32_t>(sizeof(DT_X))), 0, 0, 0};
        DataCopyPadExtParams<DT_X> pp{false, 0, 0, static_cast<DT_X>(0)};

        // 前向纯搬运：两块 UB 轮转，不经 TQue（省每任务的队列簿记）
        auto x_local = (slot_ & 1) ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // 这道 barrier 同时管两件事：等 MTE2 落地再让 MTE3 读，以及保证两任务前对同一缓冲
        // 的上一次读已排空（本路径没有 VEC 动作，TQue 的事件序本来就挂不上，换 TBuf 不减少序）
        PipeBarrier<PIPE_ALL>();

        const uint32_t k_begin = (k_limit == tiling_.m) ? 0 : k_limit;
        const uint32_t k_end   = (k_limit == tiling_.m) ? tiling_.m : k_limit + 1;
        for (uint32_t k = k_begin; k < k_end; ++k) {
            const int64_t dst_off = static_cast<int64_t>(i) * tiling_.m * tiling_.D +
                                    static_cast<int64_t>(k) * tiling_.D + jt * tiling_.dTileLen;
            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
    }

"""
A7 = """    uint32_t slot_;
"""
A8 = """        slot_ = 0;
"""
if CLEAN:
    PATCHES += [(A6, ""), (A7, ""), (A8, "")]


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


with open(KJ, encoding="utf-8") as f:
    txt = f.read()
before = md5(txt)
if before != BASE_MD5:
    print("BASE MISMATCH: kernel md5=%s, expected %s" % (before, BASE_MD5))
    sys.exit(2)
for old, new in PATCHES:
    n = txt.count(old)
    if n != 1:
        print("ANCHOR FAIL (%d hits) :: %r" % (n, old[:56]))
        sys.exit(3)

bak = KJ + ".bak_pre_r10"
if not os.path.exists(bak):
    shutil.copy2(KJ, bak)
for old, new in PATCHES:
    txt = txt.replace(old, new, 1)
with open(KJ, "w", encoding="utf-8") as f:
    f.write(txt)

print("### applied R10-%s (BATCH=%d RING=%d ADAPT=%d)  kernel md5 %s -> %s" % (MODE, BATCH, RING, ADAPT, before, md5(txt)))
print("### sanity: PIPE_ALL=%d InitBuffer_fwd=%d ProcessForward=%d NextUnit=%d" % (
    txt.count("PipeBarrier<PIPE_ALL>()"),
    txt.count("pipe_.InitBuffer(fwd_b"),
    txt.count("ProcessForward()"),
    txt.count("NextUnit(")))
