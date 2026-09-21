// Kernel: mHC-Sinkhorn (Sinkhorn-Knopp double stochastic) - vectorized version
//
// Design for 910B3 + CANN 9.0.0:
//  - UB rows are strided RS=8 floats (32B) so every row start is 32B aligned,
//    satisfying Exp/ReduceSum/Cast alignment requirements.
//  - GM<->UB via per-row DataCopyPad into matF[i*RS] (handles unaligned GM rows
//    for n=4 single row=16B, n=6 non-aligned).
//  - Row softmax: Exp -> ReduceSum -> Muls(1/(sum+eps)).
//  - Column normalize: accumulate rows into colsum via Add, then Div.
//  - Alternate row/col normalize num_iters-1 more rounds.
//  - Only in-core PipeBarrier<PIPE_ALL>(); no SyncAll. No debug output.
#include "kernel_operator.h"

#include "mhc_sinkhorn_tiling.h"
#include "tiling_key_mhc_sinkhorn.h"

#include <limits>
#include <type_traits>

using namespace AscendC;

template <typename DT_LOGITS>
class KernelMhcSinkhorn {
public:
    __aicore__ inline KernelMhcSinkhorn() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR weights,
                                const MhcSinkhornTilingData &tiling) {
        tiling_ = tiling;
        n_ = tiling_.n;
        num_iters_ = tiling_.numIters;
        eps_ = tiling_.eps;

        const int64_t batch = static_cast<int64_t>(tiling_.batch);
        const int64_t tile = static_cast<int64_t>(n_) * n_;
        x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(logits), batch * tile);
        o_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(weights), batch * tile);

        const uint32_t batch_per_core = (tiling_.batchPerCore == 0) ? 1u : tiling_.batchPerCore;
        const uint32_t block_idx = GetBlockIdx();
        m_begin_ = block_idx * batch_per_core;
        m_end_ = m_begin_ + batch_per_core;
        if (m_end_ > tiling_.batch) m_end_ = tiling_.batch;
        if (m_begin_ > tiling_.batch) m_begin_ = tiling_.batch;

        pipe_.InitBuffer(matF_buf_, N_MAX * RS * sizeof(float));
        pipe_.InitBuffer(hIn_buf_, N_MAX * 16 * sizeof(DT_LOGITS));
        pipe_.InitBuffer(hOut_buf_, N_MAX * 16 * sizeof(DT_LOGITS));
        pipe_.InitBuffer(rowsum_buf_, N_MAX * RS * sizeof(float));
        pipe_.InitBuffer(colsum_buf_, N_MAX * sizeof(float));
        pipe_.InitBuffer(work_buf_, 64 * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t m = m_begin_; m < m_end_; ++m) {
            ProcessOneMatrix(m);
        }
    }

private:
    static constexpr int32_t N_MAX = 8;
    static constexpr int32_t RS = 8;  // row stride in floats = 32B

    __aicore__ inline void ProcessOneMatrix(uint32_t m) {
        const int32_t n = static_cast<int32_t>(n_);
        const int64_t base = static_cast<int64_t>(m) * n * n;

        LocalTensor<float> matF = matF_buf_.Get<float>();
        LocalTensor<float> rowsum = rowsum_buf_.Get<float>();
        LocalTensor<float> colsum = colsum_buf_.Get<float>();
        LocalTensor<float> work = work_buf_.Get<float>();

        // load: per-row DataCopyPad, row start 32B aligned (matF[i*RS])
        if constexpr (std::is_same<DT_LOGITS, half>::value) {
            LocalTensor<DT_LOGITS> hIn = hIn_buf_.Get<DT_LOGITS>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(n * (int32_t)sizeof(DT_LOGITS)), 0, 0, 0};
            DataCopyPadExtParams<DT_LOGITS> pp{false, 0, 0, static_cast<DT_LOGITS>(0)};
            for (int32_t i = 0; i < n; ++i) {
                DataCopyPad(hIn[i * 16], x_gm_[base + i * n], cp, pp);
            }
            PipeBarrier<PIPE_ALL>();
            for (int32_t i = 0; i < n; ++i) {
                Cast(matF[i * RS], hIn[i * 16], RoundMode::CAST_NONE, n);
            }
        } else {
            DataCopyExtParams cp{1, static_cast<uint32_t>(n * (int32_t)sizeof(DT_LOGITS)), 0, 0, 0};
            DataCopyPadExtParams<DT_LOGITS> pp{false, 0, 0, static_cast<DT_LOGITS>(0)};
            for (int32_t i = 0; i < n; ++i) {
                DataCopyPad(matF[i * RS], x_gm_[base + i * n], cp, pp);
            }
        }
        PipeBarrier<PIPE_ALL>();

        // init: row softmax
        for (int32_t i = 0; i < n; ++i) {
            Exp(matF[i * RS], matF[i * RS], n);
        }
        for (int32_t i = 0; i < n; ++i) {
            ReduceSum<float>(rowsum[i * RS], matF[i * RS], work, n);
        }
        PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < n; ++i) {
            const float inv = 1.0f / (rowsum.GetValue(i * RS) + eps_);
            Muls(matF[i * RS], matF[i * RS], inv, n);
        }

        // column normalize
        Duplicate(colsum, 0.0f, n);
        for (int32_t i = 0; i < n; ++i) {
            Add(colsum, colsum, matF[i * RS], n);
        }
        Adds(colsum, colsum, eps_, n);
        for (int32_t i = 0; i < n; ++i) {
            Div(matF[i * RS], matF[i * RS], colsum, n);
        }
        PipeBarrier<PIPE_ALL>();

        // alternate row/col normalize
        for (uint32_t it = 1; it < num_iters_; ++it) {
            for (int32_t i = 0; i < n; ++i) {
                ReduceSum<float>(rowsum[i * RS], matF[i * RS], work, n);
            }
            PipeBarrier<PIPE_ALL>();
            for (int32_t i = 0; i < n; ++i) {
                const float inv = 1.0f / (rowsum.GetValue(i * RS) + eps_);
                Muls(matF[i * RS], matF[i * RS], inv, n);
            }
            Duplicate(colsum, 0.0f, n);
            for (int32_t i = 0; i < n; ++i) {
                Add(colsum, colsum, matF[i * RS], n);
            }
            Adds(colsum, colsum, eps_, n);
            for (int32_t i = 0; i < n; ++i) {
                Div(matF[i * RS], matF[i * RS], colsum, n);
            }
            PipeBarrier<PIPE_ALL>();
        }

        // store: per-row DataCopyPad
        if constexpr (std::is_same<DT_LOGITS, half>::value) {
            LocalTensor<DT_LOGITS> hOut = hOut_buf_.Get<DT_LOGITS>();
            for (int32_t i = 0; i < n; ++i) {
                Cast(hOut[i * 16], matF[i * RS], RoundMode::CAST_RINT, n);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams wo{1, static_cast<uint32_t>(n * (int32_t)sizeof(DT_LOGITS)), 0, 0, 0};
            for (int32_t i = 0; i < n; ++i) {
                DataCopyPad(o_gm_[base + i * n], hOut[i * 16], wo);
            }
        } else {
            DataCopyExtParams wo{1, static_cast<uint32_t>(n * (int32_t)sizeof(DT_LOGITS)), 0, 0, 0};
            for (int32_t i = 0; i < n; ++i) {
                DataCopyPad(o_gm_[base + i * n], matF[i * RS], wo);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> matF_buf_;
    TBuf<TPosition::VECCALC> hIn_buf_;
    TBuf<TPosition::VECCALC> hOut_buf_;
    TBuf<TPosition::VECCALC> rowsum_buf_;
    TBuf<TPosition::VECCALC> colsum_buf_;
    TBuf<TPosition::VECCALC> work_buf_;
    GlobalTensor<DT_LOGITS> x_gm_;
    GlobalTensor<DT_LOGITS> o_gm_;
    MhcSinkhornTilingData tiling_;
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
