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

        // UB 缓冲（双缓冲）
        pipe_.InitBuffer(in_que_, 2, tiling_.dTileLen * elem_size_);
        pipe_.InitBuffer(out_que_, 2, tiling_.dTileLen * elem_size_);
        pipe_.InitBuffer(acc_buf_, tiling_.dTileLen * sizeof(float));
        pipe_.InitBuffer(tmp_buf_, tiling_.dTileLen * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (task_begin_ >= task_end_) return;
        const uint32_t d_tile_num = tiling_.dTileNum;
        const uint32_t s = tiling_.S;
        const uint32_t m = tiling_.m;
        const uint32_t smode = tiling_.splitMode;

        for (uint32_t t = task_begin_; t < task_end_; ++t) {
            if constexpr (BACKWARD) {
                // 反向：ROW -> task=s; ELEMENT -> task=s*dTileNum+jt
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
            } else {
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

private:
    // 前向单块：读 x[i, jt] 一次，写 k_begin..k_end-1 个副本
    // k=m 表示全 k（ROW 模式 UB 复用），k<m 表示只写第 k 个（STREAM/ELEMENT 模式拆核）
    __aicore__ inline void ForwardOneBlock(uint32_t i, uint32_t jt, uint32_t cur_h, uint32_t k_limit) {
        const int64_t src_off = static_cast<int64_t>(i) * tiling_.D + jt * tiling_.dTileLen;
        DataCopyExtParams cp{1, static_cast<uint32_t>(cur_h * static_cast<int32_t>(sizeof(DT_X))), 0, 0, 0};
        DataCopyPadExtParams<DT_X> pp{false, 0, 0, static_cast<DT_X>(0)};

        auto in_buf = in_que_.AllocTensor<DT_X>();
        DataCopyPad(in_buf, x_gm_[src_off], cp, pp);
        in_que_.EnQue(in_buf);
        auto x_local = in_que_.DeQue<DT_X>();

        const uint32_t k_begin = (k_limit == tiling_.m) ? 0 : k_limit;
        const uint32_t k_end   = (k_limit == tiling_.m) ? tiling_.m : k_limit + 1;
        for (uint32_t k = k_begin; k < k_end; ++k) {
            const int64_t dst_off = static_cast<int64_t>(i) * tiling_.m * tiling_.D +
                                    static_cast<int64_t>(k) * tiling_.D + jt * tiling_.dTileLen;
            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
        in_que_.FreeTensor(x_local);
    }

    // 反向单块：逐 m 读 o_grad[i, k, jt] -> Cast 到 float 累加 -> Cast 回原 dtype 写 x_grad[i, jt]
    __aicore__ inline void BackwardOneBlock(uint32_t i, uint32_t jt, uint32_t cur_h) {
        const uint32_t m = tiling_.m;
        const int64_t base = static_cast<int64_t>(i) * m * tiling_.D + jt * tiling_.dTileLen;
        // 真机坑：行首/尾块非 32B 对齐，GM<->UB 一律用 DataCopyPad（blockLen 单位为字节）。
        DataCopyExtParams cp{1, static_cast<uint32_t>(cur_h * static_cast<int32_t>(sizeof(DT_X))), 0, 0, 0};
        DataCopyPadExtParams<DT_X> pp{false, 0, 0, static_cast<DT_X>(0)};

        // 1. 累加器清零
        auto acc = acc_buf_.Get<float>();
        Duplicate(acc, 0.0f, cur_h);

        // 2. 逐副本读入 + Cast 到 float + 累加
        for (uint32_t k = 0; k < m; ++k) {
            auto in_buf = in_que_.AllocTensor<DT_X>();
            DataCopyPad(in_buf, x_gm_[base + static_cast<int64_t>(k) * tiling_.D], cp, pp);
            in_que_.EnQue(in_buf);
            auto grad_local = in_que_.DeQue<DT_X>();
            auto tmp = tmp_buf_.Get<float>();
            Cast(tmp, grad_local, RoundMode::CAST_NONE, cur_h);
            Add(acc, acc, tmp, cur_h);
            in_que_.FreeTensor(grad_local);
        }

        // 3. Cast 回原 dtype 写回 x_grad
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
    GlobalTensor<DT_X> x_gm_;
    GlobalTensor<DT_X> o_gm_;
    MhcExpandTilingData tiling_;
    uint32_t elem_size_;
    uint32_t task_begin_;
    uint32_t task_end_;
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
