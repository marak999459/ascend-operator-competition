// Kernel侧核函数实现：mHC-expand（前向复制广播 / 反向求和归约）
#include "kernel_operator.h"
#include "mhc_expand_tiling.h"
#include "tiling_key_mhc_expand.h"

using namespace AscendC;

template <typename DT_X, bool BACKWARD>
class KernelMhcExpand {
public:
    __aicore__ inline KernelMhcExpand() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR o, GM_ADDR workspace, const MhcExpandTilingData &tiling) {
        tiling_ = tiling;
        elem_size_ = sizeof(DT_X);
        const uint32_t s = tiling_.S;
        const uint32_t d = tiling_.D;
        const uint32_t m = tiling_.m;

        if constexpr (BACKWARD) {
            // 反向：x = o_grad [S, m, D]；o = x_grad [S, D]
            x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x),
                                  static_cast<int64_t>(s) * m * d);
            o_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(o),
                                  static_cast<int64_t>(s) * d);
        } else {
            // 前向：x = [S, D]；o = [S, m, D]
            x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x),
                                  static_cast<int64_t>(s) * d);
            o_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(o),
                                  static_cast<int64_t>(s) * m * d);
        }

        // 按 host 的 splitMode 拆核，小 shape 时把 m 维/元素维也铺到多核
        uint32_t core_num = tiling_.blockDim;
        if (core_num == 0) core_num = 1;
        uint32_t tpc = tiling_.rowsPerCore;
        if (tpc == 0) tpc = 1;
        uint32_t block_idx = GetBlockIdx();
        task_begin_ = block_idx * tpc;
        task_end_ = task_begin_ + tpc;
        // total_tasks 由 host 算好：ROW=S, STREAM=S*m, ELEMENT=S*m*dTileNum(前向)/S*dTileNum(反向)
        uint32_t total_tasks;
        if constexpr (BACKWARD) {
            total_tasks = (tiling_.splitMode == 2) ? s * tiling_.dTileNum : s;
        } else {
            if (tiling_.splitMode == 1) total_tasks = s * m;
            else if (tiling_.splitMode == 2) total_tasks = s * m * tiling_.dTileNum;
            else total_tasks = s;
        }
        if (task_end_ > total_tasks) task_end_ = total_tasks;

        // UB 缓冲：反向走 TQue 流水线（in/out 双缓冲 + 两块 float 暂存）；
        // 前向是纯搬运，按环深建轮转缓冲：小 tile 走 BS=1 只要 2 块，攒批路径要 2*BS 块。
        // 每块上限 host 的 ub_size/4，所以环最深 4 块正好铺满 UB，加深环要先动 host 预算。
        if constexpr (BACKWARD) {
            pipe_.InitBuffer(in_que_, 2, tiling_.dTileLen * elem_size_);
            pipe_.InitBuffer(out_que_, 2, tiling_.dTileLen * elem_size_);
            pipe_.InitBuffer(acc_buf_, tiling_.dTileLen * sizeof(float));
            pipe_.InitBuffer(tmp_buf_, tiling_.dTileLen * sizeof(float));
        } else {
            const uint32_t tile_bytes = tiling_.dTileLen * elem_size_;
            merge_rows_ = MergeRows(tile_bytes);
            if (merge_rows_ >= 2) {
                // 合批只用两格轮换，占字节 (2*merge_rows_*tile_bytes) <= 2*FWD_MERGE_BYTES
                merge_bytes_ = merge_rows_ * tile_bytes;
                pipe_.InitBuffer(fwd_b0_, merge_bytes_);
                pipe_.InitBuffer(fwd_b1_, merge_bytes_);
            } else {
                const uint32_t slots = FwdBatch() * 2;   // 环深 = 2*BS：本批写的格必不是上批 MTE3 读的格
                pipe_.InitBuffer(fwd_b0_, tile_bytes);
                pipe_.InitBuffer(fwd_b1_, tile_bytes);
                // 只有攒批路径要多占的格子；BS=1 时不白占 UB
                if (slots > 2) {
                    pipe_.InitBuffer(fwd_b2_, tile_bytes);
                    pipe_.InitBuffer(fwd_b3_, tile_bytes);
                }
                if (slots > 4) {
                    pipe_.InitBuffer(fwd_b4_, tile_bytes);
                    pipe_.InitBuffer(fwd_b5_, tile_bytes);
                    pipe_.InitBuffer(fwd_b6_, tile_bytes);
                    pipe_.InitBuffer(fwd_b7_, tile_bytes);
                }
            }
        }
    }

    __aicore__ inline void Process() {
        if (task_begin_ >= task_end_) return;
        if constexpr (BACKWARD) ProcessBackward();
        else ProcessForward();
    }

    // 反向任务循环：ROW -> task=s(整行,遍历全部 jt); ELEMENT -> task=s*dTileNum+jt
    __aicore__ inline void ProcessBackward() {
        const uint32_t d_tile_num = tiling_.dTileNum;
        const uint32_t smode = tiling_.splitMode;
        for (uint32_t t = task_begin_; t < task_end_; ++t) {
            uint32_t s_idx, jt;
            if (smode == 2) { s_idx = t / d_tile_num; jt = t % d_tile_num; }
            else            { s_idx = t; jt = d_tile_num; } // jt=d_tile_num 表示遍历全部
            if (jt == d_tile_num) {
                for (uint32_t j = 0; j < d_tile_num; ++j) {
                    const uint32_t cur_h = (j == d_tile_num - 1) ? tiling_.dTailLen : tiling_.dTileLen;
                    BackwardOneBlock(s_idx, j, cur_h);
                }
            } else {
                const uint32_t cur_h = (jt == d_tile_num - 1) ? tiling_.dTailLen : tiling_.dTileLen;
                BackwardOneBlock(s_idx, jt, cur_h);
            }
        }
    }

private:
    // 合批资格：整行(dTileNum==1) + 按行 32B 对齐 + 小 tile + 本核拿到 >=2 行。
    // 字节门沿用 FWD_SMALL_THRESH(6144)，于是 medium(8192B)/large(14336B) 两档**逐字节不变**，
    // 可以直接当 A/B 的恒等对照组；D 方向多 tile 的形状走原路径，不在这次改动范围内。
    __aicore__ inline uint32_t MergeRows(uint32_t tile_bytes) const {
        if (tiling_.splitMode != 0 || tiling_.dTileNum != 1) return 0;
        if (tile_bytes > FWD_SMALL_THRESH || (tile_bytes & 31u) != 0) return 0;
        const uint32_t n = task_end_ - task_begin_;
        if (n < 2) return 0;
        const uint32_t cap = FWD_MERGE_BYTES / tile_bytes;
        return (n < cap) ? n : cap;
    }

    // 合批前向：一批 = 本核连续 L 行。读侧一条 blockCount=1 的 DMA（这 L 行在 x 里本来就连续，
    // 与逐行读只差长度）；写侧每个副本一条 DMA，用 blockCount=L + GM 侧 gap 表达"落点跨 m 行跳"。
    // 字段口径两条都由 §14.9 的单核探针钉死：DataCopyPad 下 blockLen/gap 单位=字节，
    // 且 stride=0 的含义是"该侧按 blockLen 前进"（不是冻结）——这里要的正是源侧前进。
    __aicore__ inline void ProcessForwardMerged() {
        const uint32_t rb = tiling_.dTileLen * elem_size_;
        const uint32_t d = tiling_.D;
        const uint32_t m = tiling_.m;
        DataCopyPadExtParams<DT_X> pp{false, 0, 0, static_cast<DT_X>(0)};
        uint32_t t = task_begin_;
        uint32_t slot = 0;
        bool first = true;
        while (t < task_end_) {
            uint32_t L = task_end_ - t;
            if (L > merge_rows_) L = merge_rows_;
            auto buf = (slot == 0) ? fwd_b0_.Get<DT_X>() : fwd_b1_.Get<DT_X>();
            if (!first) PipeBarrier<PIPE_ALL>();   // 两格轮换：读回本格之前先等上一批 MTE3 腾空
            first = false;
            DataCopyExtParams rd{1, L * rb, 0, 0, 0};
            DataCopyPad(buf, x_gm_[static_cast<int64_t>(t) * d], rd, pp);
            PipeBarrier<PIPE_ALL>();               // 本批落地之后才允许 MTE3 读它
            // {count, len, srcStride, dstStride, rsv}：源连续前进，目的每块跳 m 行（gap=(m-1)*rb 字节）
            DataCopyExtParams wr{static_cast<uint16_t>(L), rb, 0, (m - 1) * rb, 0};
            for (uint32_t k = 0; k < m; ++k) {
                const int64_t dst = (static_cast<int64_t>(t) * m + k) * d;
                DataCopyPad(o_gm_[dst], buf, wr);
            }
            t += L;
            slot ^= 1u;
        }
    }

    // 前向批处理驱动：攒 BS 个"块"一起灌 -> 一道 PIPE_ALL -> 一起吐。
    // 环深取 2*BS：本批写入的格必然不是上一批 MTE3 正在读的格，于是"上一批的吐出"和
    // "本批的灌入"第一次真正并行，而序的强度与逐块 barrier 完全一样（这条纯搬运路径上
    // arch22 只认 PIPE_ALL，手工事件对不产生序，见 code1.md §17）。
    // BS 走模板参数：换成运行时变量会让内层循环失去展开，小档实测更慢。
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

    // 攒批的批大小。环占用 = 2*BS 格 × 单块字节，两支各自的界不一样（§18.1 那条锁的反向使用）：
    // BS=2 用"tile >= FWD_THRESH(12KB)"这个**下界**（R10 原样），BS=4 用"tile <= FWD_SMALL_THRESH(6KB)"
    // 这个**上界**，于是 BS=4 的环最坏 8*6144=48KB，恰好等于 BS=2 支路最小的环 4*12288=48KB
    // —— 新增这一支没有多占过 R10 已经占过的字节，host 的 ub_size/4 预算一个字节也没放宽。
    // BS=4 只在整行模式(dTileNum==1)放开：多 tile 形状每核单元数多、barrier 本就被摊薄，
    // medium 上攒批实测 0 收益（§18.3），不给未测过的形状自由度。
    __aicore__ inline uint32_t FwdBatch() const {
        const uint32_t tile_bytes = tiling_.dTileLen * elem_size_;
        if (tile_bytes >= FWD_THRESH) return FWD_BATCH;
        if (tiling_.dTileNum == 1 && tile_bytes <= FWD_SMALL_THRESH) return FWD_BATCH_SMALL;
        return 1;
    }

    __aicore__ inline void ProcessForward() {
        if (merge_rows_ >= 2) { ProcessForwardMerged(); return; }
        // 三条支路各自编译期展开（BS 换成运行时变量会丢掉内层循环展开，实测更慢）
        const uint32_t bs = FwdBatch();
        if (bs >= FWD_BATCH_SMALL) ProcessForwardN<FWD_BATCH_SMALL>();
        else if (bs >= FWD_BATCH) ProcessForwardN<FWD_BATCH>();
        else ProcessForwardN<1>();
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

    // 环槽位 -> UB 缓冲：0/1 两格恒在，2..7 只在对应批大小成立时才被 InitBuffer
    __aicore__ inline auto FwdBuf(uint32_t slot) {
        if (slot == 0) return fwd_b0_.Get<DT_X>();
        if (slot == 1) return fwd_b1_.Get<DT_X>();
        if (slot == 2) return fwd_b2_.Get<DT_X>();
        if (slot == 3) return fwd_b3_.Get<DT_X>();
        if (slot == 4) return fwd_b4_.Get<DT_X>();
        if (slot == 5) return fwd_b5_.Get<DT_X>();
        if (slot == 6) return fwd_b6_.Get<DT_X>();
        return fwd_b7_.Get<DT_X>();
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
    // 双缓冲流水线：DMA 读 k+1 与 VEC 算 k 并行，隐藏 DMA 延迟
    __aicore__ inline void BackwardOneBlock(uint32_t i, uint32_t jt, uint32_t cur_h) {
        const uint32_t m = tiling_.m;
        const int64_t base = static_cast<int64_t>(i) * m * tiling_.D + jt * tiling_.dTileLen;
        DataCopyExtParams cp{1, static_cast<uint32_t>(cur_h * static_cast<int32_t>(sizeof(DT_X))), 0, 0, 0};
        DataCopyPadExtParams<DT_X> pp{false, 0, 0, static_cast<DT_X>(0)};

        auto acc = acc_buf_.Get<float>();
        Duplicate(acc, 0.0f, cur_h);

        // 预取第 0 个副本
        auto next_dma_buf = in_que_.AllocTensor<DT_X>();
        DataCopyPad(next_dma_buf, x_gm_[base], cp, pp);
        in_que_.EnQue(next_dma_buf);

        for (uint32_t k = 0; k < m; ++k) {
            auto cur_dma_buf = next_dma_buf;
            auto compute_buf = in_que_.DeQue<DT_X>();

            // 启动下一个副本的 DMA（与当前 VEC 计算并行）
            if (k < m - 1) {
                next_dma_buf = in_que_.AllocTensor<DT_X>();
                DataCopyPad(next_dma_buf,
                            x_gm_[base + static_cast<int64_t>(k + 1) * tiling_.D],
                            cp, pp);
                in_que_.EnQue(next_dma_buf);
            }

            auto tmp = tmp_buf_.Get<float>();
            Cast(tmp, compute_buf, RoundMode::CAST_NONE, cur_h);
            Add(acc, acc, tmp, cur_h);
            in_que_.FreeTensor(compute_buf);
        }

        auto out_buf = out_que_.AllocTensor<DT_X>();
        Cast(out_buf, acc, RoundMode::CAST_RINT, cur_h);
        out_que_.EnQue(out_buf);
        auto o_local = out_que_.DeQue<DT_X>();
        DataCopyPad(o_gm_[static_cast<int64_t>(i) * tiling_.D + jt * tiling_.dTileLen], o_local, cp);
        out_que_.FreeTensor(o_local);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> in_que_;
    TQue<QuePosition::VECOUT, 2> out_que_;
    TBuf<TPosition::VECCALC> acc_buf_;
    TBuf<TPosition::VECCALC> tmp_buf_;
    static constexpr uint32_t FWD_BATCH = 2;              // 大 tile 前向每次攒 2 块、共用一道 barrier
    static constexpr uint32_t FWD_BATCH_SMALL = 4;        // 整行小 tile：攒 4 块（R11，见 code1.md §19）
    static constexpr uint32_t FWD_THRESH = 12288;         // 单块字节数：过了才走大 tile 攒批
    static constexpr uint32_t FWD_SMALL_THRESH = 6144;    // 单块字节数：<= 此值才够开 8 格环
    static constexpr uint32_t FWD_MERGE_BYTES = 16384;    // 合批单格 UB 上限；两格轮换 = 32KB <= 48KB 预算
    TBuf<TPosition::VECCALC> fwd_b0_;
    TBuf<TPosition::VECCALC> fwd_b1_;
    TBuf<TPosition::VECCALC> fwd_b2_;
    TBuf<TPosition::VECCALC> fwd_b3_;
    TBuf<TPosition::VECCALC> fwd_b4_;
    TBuf<TPosition::VECCALC> fwd_b5_;
    TBuf<TPosition::VECCALC> fwd_b6_;
    TBuf<TPosition::VECCALC> fwd_b7_;
    GlobalTensor<DT_X> x_gm_;
    GlobalTensor<DT_X> o_gm_;
    MhcExpandTilingData tiling_;
    uint32_t elem_size_;
    uint32_t task_begin_;
    uint32_t task_end_;
    uint32_t merge_rows_ = 0;
    uint32_t merge_bytes_ = 0;
};

template <typename DT_X, bool BACKWARD>
__global__ __aicore__ void mhc_expand(GM_ADDR x, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcExpandTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcExpandTilingData, tiling_data, tiling);
    KernelMhcExpand<DT_X, BACKWARD> op;
    op.Init(x, o, workspace, tiling_data);
    op.Process();
}

// 显式实例化模板（由 TilingKey 选择）
template __aicore__ void mhc_expand<half, false>(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template __aicore__ void mhc_expand<half, true>(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template __aicore__ void mhc_expand<bfloat16_t, false>(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template __aicore__ void mhc_expand<bfloat16_t, true>(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
