// mHC-Sinkhorn kernel 修复版
// 修复：原版在 per-matrix 循环内调用 SyncAll()，而各核循环次数 =
//       ceildiv(batch, coreNum) 并不相同（batch 不能整除核数时，尾部核次数更少，
//       甚至为 0），导致先完成的核在栅栏处永久等待 -> 死锁 -> 设备超时(507035)。
// 方案：所有核执行相同的轮数 maxChunksPerCore，无活的轮次只参与 SyncAll 不做计算。
#include "kernel_operator.h"

#include "mhc_sinkhorn_tiling.h"
#include "tiling_key_mhc_sinkhorn.h"

#include <limits>

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

        pipe_.InitBuffer(mat_buf_, N_MAX * N_MAX * sizeof(DT_LOGITS));
        pipe_.InitBuffer(sum_buf_, N_MAX * sizeof(DT_LOGITS));
    }

    __aicore__ inline void Process() {
        // 【修复点】所有核执行相同的轮数；且 SyncAll() 必须放在 if 之外，
        // 保证无论本核是否有活都无条件到达栅栏（放 if 内会被编译器按常量分支消除）。
        const uint32_t myChunks = m_end_ - m_begin_;
        const uint32_t nChunks = (tiling_.maxChunksPerCore == 0) ? 1u : tiling_.maxChunksPerCore;

        for (uint32_t c = 0; c < nChunks; ++c) {
            // 先计算，再无条件栅栏
            if (c < myChunks) {
                ProcessOneMatrix(m_begin_ + c);
            }
            // 注意：不能写成 if (c < myChunks) { ...; SyncAll(); }
            AscendC::SyncAll();
        }

        // 兜底栅栏：确保所有核都走到同一点再退出
        AscendC::SyncAll();
    }

private:
    // 行归约（dst[i] = Σ_k src[i*n+k]）
    __aicore__ inline void RowSumCompact(const LocalTensor<DT_LOGITS> &dst,
                                         const LocalTensor<DT_LOGITS> &src) {
        for (int32_t i = 0; i < static_cast<int32_t>(n_); ++i) {
            float acc = 0.0f;
            for (int32_t k = 0; k < static_cast<int32_t>(n_); ++k) {
                acc += static_cast<float>(src.GetValue(i * static_cast<int32_t>(n_) + k));
            }
            dst.SetValue(i, static_cast<DT_LOGITS>(acc));
        }
    }

    // 单矩阵 Sinkhorn 迭代（紧凑 n*n 布局）
    __aicore__ inline void ProcessOneMatrix(uint32_t m) {
        auto mat = mat_buf_.Get<DT_LOGITS>();
        auto sumv = sum_buf_.Get<DT_LOGITS>();
        const int32_t nn = static_cast<int32_t>(n_) * static_cast<int32_t>(n_);
        const int64_t base = static_cast<int64_t>(m) * nn;

        // ---- 读入 x[m]（紧凑 n*n）：n=4/8 DataCopy（32B 对齐），n=6 标量 ----
        if (n_ != 6) {
            DataCopy(mat, x_gm_[base], nn);
            AscendC::SyncAll();  // 等读入完成再标量访问
        } else {
            for (int32_t idx = 0; idx < nn; ++idx) {
                mat.SetValue(idx, static_cast<DT_LOGITS>(x_gm_.GetValue(base + idx)));
            }
        }

        // ---- 初始化阶段：行 softmax ----
        for (int32_t i = 0; i < static_cast<int32_t>(n_); ++i) {
            float mx = static_cast<float>(mat.GetValue(i * static_cast<int32_t>(n_)));
            for (int32_t k = 1; k < static_cast<int32_t>(n_); ++k) {
                float v = static_cast<float>(mat.GetValue(i * static_cast<int32_t>(n_) + k));
                mx = (v > mx) ? v : mx;
            }
            sumv.SetValue(i, static_cast<DT_LOGITS>(mx));
        }
        for (int32_t i = 0; i < static_cast<int32_t>(n_); ++i) {
            const float mx = static_cast<float>(sumv.GetValue(i));
            for (int32_t k = 0; k < static_cast<int32_t>(n_); ++k) {
                const int32_t idx = i * static_cast<int32_t>(n_) + k;
                mat.SetValue(idx, static_cast<DT_LOGITS>(
                    static_cast<float>(mat.GetValue(idx)) - mx));
            }
        }
        Exp(mat, mat, nn);
        AscendC::SyncAll();  // 等 exp 完成再标量归约
        RowSumCompact(sumv, mat);
        for (int32_t i = 0; i < static_cast<int32_t>(n_); ++i) {
            const float s = static_cast<float>(sumv.GetValue(i));
            for (int32_t k = 0; k < static_cast<int32_t>(n_); ++k) {
                const int32_t idx = i * static_cast<int32_t>(n_) + k;
                mat.SetValue(idx, static_cast<DT_LOGITS>(
                    static_cast<float>(mat.GetValue(idx)) / s + eps_));
            }
        }
        for (int32_t k = 0; k < static_cast<int32_t>(n_); ++k) {
            float acc = 0.0f;
            for (int32_t i = 0; i < static_cast<int32_t>(n_); ++i) {
                acc += static_cast<float>(mat.GetValue(i * static_cast<int32_t>(n_) + k));
            }
            sumv.SetValue(k, static_cast<DT_LOGITS>(acc + eps_));
        }
        for (int32_t i = 0; i < static_cast<int32_t>(n_); ++i) {
            for (int32_t k = 0; k < static_cast<int32_t>(n_); ++k) {
                const int32_t idx = i * static_cast<int32_t>(n_) + k;
                mat.SetValue(idx, static_cast<DT_LOGITS>(
                    static_cast<float>(mat.GetValue(idx)) /
                    static_cast<float>(sumv.GetValue(k))));
            }
        }

        // ---- 交替迭代阶段（i = 1 .. numIters-1）----
        for (uint32_t it = 1; it < num_iters_; ++it) {
            RowSumCompact(sumv, mat);
            for (int32_t k = 0; k < static_cast<int32_t>(n_); ++k) {
                sumv.SetValue(k, static_cast<DT_LOGITS>(
                    static_cast<float>(sumv.GetValue(k)) + eps_));
            }
            for (int32_t i = 0; i < static_cast<int32_t>(n_); ++i) {
                const float s = static_cast<float>(sumv.GetValue(i));
                for (int32_t k = 0; k < static_cast<int32_t>(n_); ++k) {
                    const int32_t idx = i * static_cast<int32_t>(n_) + k;
                    mat.SetValue(idx, static_cast<DT_LOGITS>(
                        static_cast<float>(mat.GetValue(idx)) / s));
                }
            }
            for (int32_t k = 0; k < static_cast<int32_t>(n_); ++k) {
                float acc = 0.0f;
                for (int32_t i = 0; i < static_cast<int32_t>(n_); ++i) {
                    acc += static_cast<float>(mat.GetValue(i * static_cast<int32_t>(n_) + k));
                }
                sumv.SetValue(k, static_cast<DT_LOGITS>(acc + eps_));
            }
            for (int32_t i = 0; i < static_cast<int32_t>(n_); ++i) {
                for (int32_t k = 0; k < static_cast<int32_t>(n_); ++k) {
                    const int32_t idx = i * static_cast<int32_t>(n_) + k;
                    mat.SetValue(idx, static_cast<DT_LOGITS>(
                        static_cast<float>(mat.GetValue(idx)) /
                        static_cast<float>(sumv.GetValue(k))));
                }
            }
        }

        // ---- 输出 weights（紧凑 n*n）：n=4/8 DataCopy，n=6 标量 ----
        if (n_ != 6) {
            DataCopy(o_gm_[base], mat, nn);
            AscendC::SyncAll();
        } else {
            for (int32_t idx = 0; idx < nn; ++idx) {
                o_gm_.SetValue(base + idx, mat.GetValue(idx));
            }
        }
    }

    static constexpr int32_t N_MAX = 8;

    TPipe pipe_;
    TBuf<TPosition::VECCALC> mat_buf_;
    TBuf<TPosition::VECCALC> sum_buf_;
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
