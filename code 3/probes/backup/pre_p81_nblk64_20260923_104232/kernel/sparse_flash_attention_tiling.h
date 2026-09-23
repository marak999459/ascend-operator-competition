// Tiling结构体定义的头文件
//
// ⚠️ 相对题目原始骨架的扩展（原骨架只有 7 个字段）：
//    新增 sparseBlockSize / sparseCount / nb / nBlk 四个字段。
//    原因：原骨架没有传递 sparseBlockSize（题面必选属性！）与分块尺寸，
//    kernel 无法工作。这四个值必须由 host 计算下发。
//    这属于"必须做的扩展"，不是随意改动 —— 原骨架本身不足以实现本题。
#pragma once

#include <cstdint>

// P24 起【组宽不再是常数】：kernel 的 scGrp_ = n_blk（一组 = 整个 chunk），
// 部分积就地写在 kf/krf 上，所以不需要在两边共享的头里放一个组宽常量。
// 历史：P8 取 16、P13 对 nb==1 取 32（那两档的实测结论见 code3.md §15.12/§15.16）。
// ⚠️ 唯一还要求两边同源的是 **rdBuf_ 的切片宽度** redW_ = max(nb, n_blk, 2 个 UB 块)
//    （kernel Init 段 / host RedWOf 同一口径，改一边就会真机越界）。

// P21：一个 chunk 的 gather **段数**上限 = kernel 里"扫描前置"那块定长暂存数组的尺寸。
// host 的 n_blk 候选因此钳在 <= 它：一段至少贡献 1 个 token，所以 n_blk 个 token 最多
// n_blk 段。⇒ 这一钳就是"最大 chunk"的上限，真正的-binding 项是 host 的 UB 预算式。
//   P29 时代每多一个 token 要 512×2(K) + 512×2(V) + 64×2(kr) + 512×4(kf 组宽=整个
//   chunk，P24 起) + 64×4 ≈ 4.4 KB，加上 q/o/s/p/ml/lse/rd 的固定项，`nb==1` 档塞到
//   **40** 个 token（184,512 B / ubSafe 186,485 B）就到顶，`nb>=2` 档 32 就是上限。
// P32 让 V 复用 kb 自己那块（见 kernel Init 与 §15.53）⇒ 每 token 少 512×2 = 1 KB，
//   `nb==1/2` 到 48 都还留有余地，`nb==4` 也刚好挤进（185,568 B，余 917 B）；
//   `nb==8` 与 k=56 仍然装不下，所以 48 就是这一钳的新上限。
// P29 抬到 40 的实测口径见 code3.md §15.49；P32 抬到 48 的实测见 §15.53。
// ⚠️ 必须是 8（一个 UB 块的 fp32 个数）的整数倍：分数矩阵行间距 = n_blk 个 float，
//    否则行起点落在块中间 -> 真机 ADDR_MISALIGN（与 nb<8 同一类，见 host 的 NBLK_MIN 注释）。
constexpr uint32_t SFA_STAGE_MAX = 48;
// cube 路径（M1d）的段登记上限【单独放宽】：AIV 在 cube 形态下不搬 K/K-rope、也不算
// ComputeScores ⇒ krBuf_/kfBuf_/krfBuf_ 三块（n_blk=48、nb=8 时约 117 KB）整段空转，
// 让给 V 与 s/p 之后 n_blk 能到 64。上限卡在 **64** 而不是环容量(16·N1)：arch22 一条
// `Mmad` 的 n 最大 64（L0C 只有 64 列），要 128 得先把 n 轴切成两半 ⇒ 另发再做。
// ⚠️ 只放宽 cube 侧：向量侧仍按 SFA_STAGE_MAX=48 钳着，否则 P31 那 184 格网格收敛出来的
//    选档轴会被重新打开（64/128 两档在向量路径实测是亏的）。
constexpr uint32_t SFA_STAGE_MAX_CUBE = 64;

struct SparseFlashAttentionTilingData {
    // ---- 原骨架字段（保持顺序不变）----
    uint32_t B;
    uint32_t Q_S;
    uint32_t KV_S;
    uint32_t Q_N;
    uint32_t Q_D;
    uint32_t Dr;
    uint32_t sparse_size;
    float    scale_value;
    uint32_t sparse_mode;
    uint32_t batch_per_core;

    // ---- 新增字段（本题实现必需）----
    uint32_t sparse_block_size;   // 题面必选属性 sparseBlockSize
    uint32_t sparse_count;        // sparse_indices 末维（官方固定 2048）
    uint32_t nb;                  // 每次处理多少个 query 头（UB 分块）
    uint32_t n_blk;               // 每个 KV chunk 容纳的 token 数（UB 分块）
    uint32_t attention_mode;      // 题面必选属性（仅支持 2）
    uint32_t total_tokens;        // B * Q_S，用于核间切分

    // ---- 变长（官方 BSND 语义：per-batch arr[b]；长度为 1 时广播 arr[0]）----
    uint32_t actual_q_len_size;   // actual_seq_lengths_query 元素个数（0=未传）
    uint32_t actual_kv_len_size;  // actual_seq_lengths_kv  元素个数（0=未传）

    // ---- P11v2：sparse 列表（KV 轴）切给几个核 ----
    // 只有 1（不切）与 2（切成前后两半，两半各一个核）两种取值。
    // 归并通道是**本算子自己的输出张量**（分片 0 照常写 attention_out / LSE，分片 1 在
    // SyncAll 之后读回来并覆盖写），所以不能用 workspace —— arch22 的自定义算子路径上
    // GetUserWorkspace() 返回的是恒定的未映射地址（§15.21），而跨核**标量** SetValue/GetValue
    // 也读不到（§15.32 实测：40 块里只有 2~3 块看得见，批量 DataCopy 才是全通的）。
    // ⚠️ 追加在结构体末尾：前面的字段顺序是原骨架口径，device 侧按字节布局读，不能重排。
    uint32_t kv_shard;

    // ---- P19-M1d：score 交给 cube 核算（§15.70(f) / §15.71(c)(d)）----
    // 0 ⇒ 与 P38【逐字节同一条路径】，既是回滚位也是同场次 A/B 位；host 的形态门
    //   （fp16 / Q_N<=16 / D=512 / Dr=64 / 不切 KV / 环装得下）任一条不满足就一定是 0。
    // 1 ⇒ 同一次 launch 里 cube 侧按 chunk 产 score、向量侧只吃回程环，且强制
    //   nb = Q_N（一个单位吃整组头，否则同一份 K 会被每个头块各 gather 一遍）、
    //   kv_shard = 1（归并通道借的正是下面那两行暂存）。
    // tile 宽度不进 tiling：它恒等于 n_blk，环预算式 2*align16(nb)*n_blk*4 已在 host 核过。
    uint32_t cube_on;
};
