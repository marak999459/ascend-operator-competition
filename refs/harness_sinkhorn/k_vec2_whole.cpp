// Kernel: mHC-Sinkhorn (Sinkhorn-Knopp double stochastic), vectorized step-1
//   + whole-tile row reduction (one vcadd instead of n per-row ReduceSum/ReduceMax)
//
// 910B3 + CANN 9.0.0. Layout invariant: every UB row occupies RS=8 floats (=32B),
// so all vector operands stay 32B aligned for n in {4,6,8}.
//
// Per-row max/sum are produced by a single WholeReduceMax/WholeReduceSum with
// count=n, repeatTime=n and srcRepStride=1 (one 32B block per row), which packs the
// n results at red_[0..n-1]. dstRepStride counts elements, srcRepStride counts
// 32B blocks. Because count masks the padding lanes, the per-row reductions never
// see the NEG_PAD tail of a row.
//
// Those n results feed one Brcb, which replicates each of 8 source scalars across a
// whole 32B block -- including that row's padding lanes. 0/(s+eps) == 0, so padding
// held at 0 is a fixed point of both normalizations and a whole-region Div replaces
// one Div per row. Padding is seeded with a large negative sentinel at load time so
// exp() flushes it to 0 and row-max is never taken over the sentinel.
//
// Column sums need no replication at all: they live in one block that every row's
// Div reuses as its shared third operand.
//
// Only in-core PipeBarrier is used (never SyncAll: it is a cross-core barrier and
// deadlocks whenever batch is not a multiple of coreNum).
#include "kernel_operator.h"

#include "mhc_sinkhorn_tiling.h"
#include "tiling_key_mhc_sinkhorn.h"

#include <type_traits>

using namespace AscendC;

template <typename DT_LOGITS>
class KernelMhcSinkhorn {
public:
    static constexpr int32_t N_MAX = 8;
    static constexpr int32_t RS = 8;    // UB row stride, floats -> exactly 32B
    static constexpr int32_t RH = 16;   // UB row stride, halfs    -> exactly 32B
    // Padding sentinel: far below any plausible logit, far enough from the fp32
    // limit that (sentinel - rowmax) cannot overflow.
    static constexpr float NEG_PAD = -1e30f;

    __aicore__ inline KernelMhcSinkhorn() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR weights,
                                const MhcSinkhornTilingData &tiling) {
        n_ = tiling.n;
        num_iters_ = tiling.numIters;
        eps_ = tiling.eps;

        const int64_t total = static_cast<int64_t>(tiling.batch) * n_ * n_;
        x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(logits), total);
        o_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(weights), total);

        const uint32_t per_core = (tiling.batchPerCore == 0u) ? 1u : tiling.batchPerCore;
        const uint32_t idx = GetBlockIdx();
        m_begin_ = idx * per_core;
        m_end_ = m_begin_ + per_core;
        if (m_end_ > tiling.batch) m_end_ = tiling.batch;
        if (m_begin_ > tiling.batch) m_begin_ = tiling.batch;

        const uint32_t nf = N_MAX * RS * sizeof(float);
        pipe_.InitBuffer(mat_buf_, nf);
        pipe_.InitBuffer(red_buf_, nf);
        pipe_.InitBuffer(bcast_buf_, nf);
        pipe_.InitBuffer(cs_buf_, nf);
        pipe_.InitBuffer(hin_buf_, N_MAX * RH * sizeof(DT_LOGITS));
        pipe_.InitBuffer(hout_buf_, N_MAX * RH * sizeof(DT_LOGITS));
    }

    __aicore__ inline void Process() {
        for (uint32_t m = m_begin_; m < m_end_; ++m) {
            ProcessOneMatrix(m);
        }
    }

private:
    __aicore__ inline void LoadRows(int32_t n, int64_t base, LocalTensor<float> mat) {
        Duplicate(mat, NEG_PAD, n * RS);
        DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(DT_LOGITS)), 0, 0, 0};
        DataCopyPadExtParams<DT_LOGITS> pp{false, 0, 0, static_cast<DT_LOGITS>(0)};
        if constexpr (std::is_same<DT_LOGITS, half>::value) {
            LocalTensor<DT_LOGITS> hin = hin_buf_.Get<DT_LOGITS>();
            for (int32_t i = 0; i < n; ++i) {
                DataCopyPad(hin[i * RH], x_gm_[base + static_cast<int64_t>(i) * n], cp, pp);
            }
            PipeBarrier<PIPE_ALL>();
            for (int32_t i = 0; i < n; ++i) {
                Cast(mat[i * RS], hin[i * RH], RoundMode::CAST_NONE, n);
            }
        } else {
            for (int32_t i = 0; i < n; ++i) {
                DataCopyPad(mat[i * RS], x_gm_[base + static_cast<int64_t>(i) * n], cp, pp);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreRows(int32_t n, int64_t base, LocalTensor<float> mat) {
        DataCopyExtParams wo{1, static_cast<uint32_t>(n * sizeof(DT_LOGITS)), 0, 0, 0};
        if constexpr (std::is_same<DT_LOGITS, half>::value) {
            LocalTensor<DT_LOGITS> hout = hout_buf_.Get<DT_LOGITS>();
            for (int32_t i = 0; i < n; ++i) {
                Cast(hout[i * RH], mat[i * RS], RoundMode::CAST_RINT, n);
            }
        }
        PipeBarrier<PIPE_ALL>();   // VEC -> MTE1: last Div/Cast must land before DMA
        if constexpr (std::is_same<DT_LOGITS, half>::value) {
            LocalTensor<DT_LOGITS> hout = hout_buf_.Get<DT_LOGITS>();
            for (int32_t i = 0; i < n; ++i) {
                DataCopyPad(o_gm_[base + static_cast<int64_t>(i) * n], hout[i * RH], wo);
            }
        } else {
            for (int32_t i = 0; i < n; ++i) {
                DataCopyPad(o_gm_[base + static_cast<int64_t>(i) * n], mat[i * RS], wo);
            }
        }
        PipeBarrier<PIPE_ALL>();   // the next matrix's MTE2 reuses these UB blocks
    }

    // red_[0..n-1] = per-row max, replicated by one Brcb, then subtracted wholesale.
    __aicore__ inline void SubtractRowMax(int32_t n, LocalTensor<float> mat, LocalTensor<float> red,
                                          LocalTensor<float> bcast) {
        WholeReduceMax<float>(red, mat, n, n, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        Brcb<float>(bcast, red, 1, BrcbRepeatParams());
        PipeBarrier<PIPE_V>();
        Sub(mat, mat, bcast, n * RS);
    }

    __aicore__ inline void RowNormalize(int32_t n, LocalTensor<float> mat, LocalTensor<float> red,
                                        LocalTensor<float> bcast) {
        WholeReduceSum<float>(red, mat, n, n, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        Adds(red, red, eps_, n);
        Brcb<float>(bcast, red, 1, BrcbRepeatParams());
        PipeBarrier<PIPE_V>();
        Div(mat, mat, bcast, n * RS);
    }

    // One row-wide block holds every column sum, so each row's Div can reuse it as
    // the shared third operand and the column axis needs no broadcast. Lanes >= n
    // of cs end up as plain eps, which keeps the padding lanes of mat pinned at 0.
    __aicore__ inline void ColNormalize(int32_t n, LocalTensor<float> mat, LocalTensor<float> cs) {
        Duplicate(cs, 0.0f, RS);
        for (int32_t i = 0; i < n; ++i) {
            Add(cs, cs, mat[i * RS], RS);
        }
        PipeBarrier<PIPE_V>();
        Adds(cs, cs, eps_, RS);
        for (int32_t i = 0; i < n; ++i) {
            Div(mat[i * RS], mat[i * RS], cs, RS);
        }
    }

    __aicore__ inline void ProcessOneMatrix(uint32_t m) {
        const int32_t n = static_cast<int32_t>(n_);
        const int64_t base = static_cast<int64_t>(m) * n * n;

        LocalTensor<float> mat = mat_buf_.Get<float>();
        LocalTensor<float> red = red_buf_.Get<float>();
        LocalTensor<float> bcast = bcast_buf_.Get<float>();
        LocalTensor<float> cs = cs_buf_.Get<float>();

        LoadRows(n, base, mat);
        SubtractRowMax(n, mat, red, bcast);
        Exp(mat, mat, n * RS);
        RowNormalize(n, mat, red, bcast);
        ColNormalize(n, mat, cs);

        for (uint32_t it = 1; it < num_iters_; ++it) {
            RowNormalize(n, mat, red, bcast);
            ColNormalize(n, mat, cs);
        }

        StoreRows(n, base, mat);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> mat_buf_;
    TBuf<TPosition::VECCALC> red_buf_;
    TBuf<TPosition::VECCALC> bcast_buf_;
    TBuf<TPosition::VECCALC> cs_buf_;
    TBuf<TPosition::VECCALC> hin_buf_;
    TBuf<TPosition::VECCALC> hout_buf_;
    GlobalTensor<DT_LOGITS> x_gm_;
    GlobalTensor<DT_LOGITS> o_gm_;
    uint32_t n_;
    uint32_t num_iters_;
    float eps_;
    uint32_t m_begin_;
    uint32_t m_end_;
};

template <typename DT_LOGITS>
__global__ __aicore__ void mhc_sinkhorn(GM_ADDR logits, GM_ADDR mask, GM_ADDR weights,
                                        GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcSinkhornTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcSinkhornTilingData, tiling_data, tiling);
    KernelMhcSinkhorn<DT_LOGITS> op;
    op.Init(logits, weights, tiling_data);
    op.Process();
}

template __aicore__ void mhc_sinkhorn<half>(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template __aicore__ void mhc_sinkhorn<float>(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
