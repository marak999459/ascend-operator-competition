// mHC-Sinkhorn kernel —— 纯标量实现（无 SyncAll、无向量指令）
//
// 【问题根因】
// 原实现用 SyncAll() 做「向量(DataCopy/Exp) → 标量(GetValue/SetValue)」的流水同步。
// 但 SyncAll 是核间栅栏，要求所有核到达同一点，而各核处理的矩阵数
// = ceil(batch/coreNum) 并不相同，先做完的核永久等待 → 死锁。
//
// 实测证据（910B3, coreNum=40）：
//   batch=8 / 64 / 100 / 1 / 8192  -> 死锁超时（不能整除 40）
//   batch=40 / 80 / 120            -> 成功（恰好整除，各核轮数一致）
//   去掉全部 SyncAll                -> 任何 batch 都成功
//
// 【修复方案】
// 每个核负责的矩阵彼此不相交，本就不需要核间同步。n<=8 时矩阵最多 64 个元素，
// 全部标量完成，既不跨核同步、也没有「向量写→标量读」的流水竞争。
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

        // 三个【完全独立】的 UB 缓冲，避免 fp16 原始区与 float 计算区复用同一块内存
        //   raw_buf_  : nn 个 DT_LOGITS（最多 64 个 = 128B）
        //   mat_buf_  : 64 个 float（矩阵）
        //   sum_buf_  : 16 个 float（rowf[8] + colf[8]）
        pipe_.InitBuffer(raw_buf_, N_MAX * N_MAX * sizeof(DT_LOGITS));
        pipe_.InitBuffer(mat_buf_, N_MAX * N_MAX * sizeof(float));
        pipe_.InitBuffer(sum_buf_, 2 * N_MAX * sizeof(float));
    }

    __aicore__ inline void Process() {
        // 无任何跨核同步：本核区间与其他核不相交
        for (uint32_t m = m_begin_; m < m_end_; ++m) {
            ProcessOneMatrix(m);
        }
    }

private:
    static constexpr int32_t N_MAX = 8;

    // 设备侧没有 exp()/expf()，自行实现：范围归约 + 泰勒展开
    //   x = k*ln2 + r, |r| <= ln2/2, e^x = 2^k * e^r
    __aicore__ inline float ExpPoly(float x) const {
        const float LN2 = 0.6931471805599453f;
        const float INV_LN2 = 1.4426950408889634f;
        if (x < -87.0f) {
            return 0.0f;
        }
        if (x > 88.0f) {
            x = 88.0f;
        }
        float kf = static_cast<float>(static_cast<int32_t>(x * INV_LN2));
        if (kf > x * INV_LN2) {
            kf -= 1.0f;
        }
        const float r = x - kf * LN2;
        float p = 1.0f / 120.0f;
        p = p * r + (1.0f / 24.0f);
        p = p * r + (1.0f / 6.0f);
        p = p * r + 0.5f;
        p = p * r + 1.0f;
        p = p * r + 1.0f;
        int32_t k = static_cast<int32_t>(kf);
        int32_t bits = (k + 127) << 23;
        if (bits <= 0) {
            return 0.0f;
        }
        if (bits >= 0x7F800000) {
            bits = 0x7F7FFFFF;
        }
        const float pow2k = *reinterpret_cast<float *>(&bits);
        return p * pow2k;
    }

    __aicore__ inline void ProcessOneMatrix(uint32_t m) {
        const int32_t n = static_cast<int32_t>(n_);
        const int32_t nn = n * n;
        const int64_t base = static_cast<int64_t>(m) * nn;

        LocalTensor<float> a = mat_buf_.Get<float>();                 // mat n*n（独立缓冲）
        LocalTensor<float> rowf = sum_buf_.Get<float>();              // rowf[8]
        LocalTensor<float> colf = sum_buf_.Get<float>()[N_MAX];       // colf[8]

        // ---- 读入 ----
        // 关键：nn*sizeof(DT_X) 必须是 32B 的整数倍才能用 DataCopy。
        //   n=4: 16*2=32B  ✔   n=8: 64*2=128B ✔   n=6: 36*2=72B ✘（非 32B 倍数，
        //   且奇数矩阵起点 72m 非对齐）-> 必须用 DataCopyPad。
        {
            LocalTensor<DT_LOGITS> raw = raw_buf_.Get<DT_LOGITS>();
            if (n_ == 6) {
                DataCopyExtParams cp{1, static_cast<uint32_t>(nn * (int32_t)sizeof(DT_LOGITS)), 0, 0, 0};
                DataCopyPadExtParams<DT_LOGITS> pp{false, 0, 0, static_cast<DT_LOGITS>(0)};
                DataCopyPad(raw, x_gm_[base], cp, pp);
            } else {
                DataCopy(raw, x_gm_[base], nn);
            }
            // DataCopy 走 MTE2 流水，标量读之前必须加【核内】栅栏。
            // 原来错用 SyncAll（核间栅栏）来做这件事 -> batch 不能整除核数时死锁。
            PipeBarrier<PIPE_ALL>();
            for (int32_t i = 0; i < nn; ++i) {
                a.SetValue(i, static_cast<float>(raw.GetValue(i)));
            }
        }

        // ---- 初始化：行 softmax ----
        for (int32_t i = 0; i < n; ++i) {
            float mx = a.GetValue(i * n);
            for (int32_t k = 1; k < n; ++k) {
                const float v = a.GetValue(i * n + k);
                mx = (v > mx) ? v : mx;
            }
            for (int32_t k = 0; k < n; ++k) {
                a.SetValue(i * n + k, a.GetValue(i * n + k) - mx);
            }
        }
        for (int32_t i = 0; i < nn; ++i) {
            a.SetValue(i, ExpPoly(a.GetValue(i)));
        }
        for (int32_t i = 0; i < n; ++i) {
            float s = 0.0f;
            for (int32_t k = 0; k < n; ++k) {
                s += a.GetValue(i * n + k);
            }
            rowf.SetValue(i, s + eps_);
        }
        for (int32_t i = 0; i < n; ++i) {
            for (int32_t k = 0; k < n; ++k) {
                a.SetValue(i * n + k, a.GetValue(i * n + k) / rowf.GetValue(i));
            }
        }
        for (int32_t k = 0; k < n; ++k) {
            float s = 0.0f;
            for (int32_t i = 0; i < n; ++i) {
                s += a.GetValue(i * n + k);
            }
            colf.SetValue(k, s + eps_);
        }
        for (int32_t i = 0; i < n; ++i) {
            for (int32_t k = 0; k < n; ++k) {
                a.SetValue(i * n + k, a.GetValue(i * n + k) / colf.GetValue(k));
            }
        }

        // ---- 交替迭代：行归一化 / 列归一化 ----
        for (uint32_t it = 1; it < num_iters_; ++it) {
            for (int32_t i = 0; i < n; ++i) {
                float s = 0.0f;
                for (int32_t k = 0; k < n; ++k) {
                    s += a.GetValue(i * n + k);
                }
                rowf.SetValue(i, s + eps_);
            }
            for (int32_t i = 0; i < n; ++i) {
                const float inv = 1.0f / rowf.GetValue(i);
                for (int32_t k = 0; k < n; ++k) {
                    a.SetValue(i * n + k, a.GetValue(i * n + k) * inv);
                }
            }
            for (int32_t k = 0; k < n; ++k) {
                float s = 0.0f;
                for (int32_t i = 0; i < n; ++i) {
                    s += a.GetValue(i * n + k);
                }
                colf.SetValue(k, s + eps_);
            }
            for (int32_t i = 0; i < n; ++i) {
                for (int32_t k = 0; k < n; ++k) {
                    a.SetValue(i * n + k, a.GetValue(i * n + k) / colf.GetValue(k));
                }
            }
        }

        // ---- 写回：先写 UB，再用与读入对称的向量搬运写出 ----
        // 实测：逐元素 o_gm_.SetValue() 在 n=4/6 时会出现"前 N 个元素正确、其余为 0"
        //       的写入截断现象，故改为写 UB 后一次性搬出。
        {
            LocalTensor<DT_LOGITS> oraw = raw_buf_.Get<DT_LOGITS>();
            for (int32_t i = 0; i < nn; ++i) {
                oraw.SetValue(i, static_cast<DT_LOGITS>(a.GetValue(i)));
            }
            PipeBarrier<PIPE_ALL>();
            if (n_ == 6) {
                DataCopyExtParams wo{1, static_cast<uint32_t>(nn * (int32_t)sizeof(DT_LOGITS)), 0, 0, 0};
                DataCopyPad(o_gm_[base], oraw, wo);
            } else {
                DataCopy(o_gm_[base], oraw, nn);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> raw_buf_;
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
