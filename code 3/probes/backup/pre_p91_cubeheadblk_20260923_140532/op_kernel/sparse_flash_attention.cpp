// Kernel侧核函数实现：SparseFlashAttention（分块 + 在线 softmax）
//
// ============================================================
// 语义依据：官方 SFA kernel 源码（见 _sfa/SEMANTICS.md，逐条有出处）
//   - sparseIndices 是【块号】: token 区间 = [blk*SBS, blk*SBS+SBS)
//   - -1 是无效哨兵，【遇到即停止扫描】
//   - 块被 thr 截断: end = min(begin+SBS, thr)
//   - mode3: thr = (act_kv - act_q) + s + 1  ← 等价于官方因果掩码判据
//   - 展开时按 thr 截断即已实现掩码，无需逐元素比较
//   - 全 mask 行 -> 输出全 0, LSE = (0, 0)（平台口径；-2e38 是废弃写法，见 code3.md §6）
//   - 变长: actual_seq_lengths 是 per-batch 值 arr[b]（长度 1 时广播）
//
// 结构：
//   核间：工作单元 = (query 行, 头块[, sparse 分片])，按单元数平均分给各核。
//         P10 起头块也是一维单元；P11v2 起，在"单元数 ×2 仍不超过核数"时把 sparse 列表
//         也切成两半分给两个核（tiling 的 kv_shard，只有 1/2 两种取值；P18 起按**下标奇偶**
//         交错分，不按连续区间 —— 见 ProcessToken）。
//         分片是唯一需要核间通信的一维：分片 0 照常把半行结果写进【本算子自己的输出张量】，
//         所有块 SyncAll 一次，分片 1 再把它读回归并、覆盖写。
//         ⚠️ 跨核传值只认批量 DataCopy：标量 SetValue/GetValue 不是跨核通道（§15.32 实测
//         40 块里只有 2~3 块看得见别人写的标量），所以分片 0 的 (m,l) 用"整块 8 个 float"
//         的对齐窗口 DataCopy 读回，不是 GetValue。
//         ⚠️ SyncAll 口径（§15.16/§15.17/§15.32）：只在单元循环之外无条件调用一次是安全的
//         （arch22 + AIV-only，所有块都会走到）；上道题死锁的根因是"各核调用次数不齐"。
//         代价 ≈2~4 µs；块数 1/16/40 实测都能过 barrier ⇒ 与 P14 收缩块数不冲突。
//         ⚠️ workspace 仍然不可用（§15.18/§15.21）：AIV-only 路径上 GetUserWorkspace() 丢掉
//         入参、返回 __get_kfc_workspace_addr() + 16 MB，而自定义算子那条路那个基址寄存器
//         恒为 0 ⇒ 形参永远是未映射的 0x1000000，写几个字节就报 MTE DDR 越界。
//         ⇒ 归并只能借输出张量，不能借 workspace。
//   核内：for tok -> for head-block(nb) -> for KV-chunk(n_blk)
//         在线 softmax：只保留 O[nb,D] + m[nb] + l[nb]，与稀疏长度 m 无关
//
// 当前状态：score、softmax、PV 三段都已向量化（P3a/P2/P7/P8/P12/P13），Q 装载与 O 写回
//   改成整批 DataCopy（P5b），KV chunk 搬运已聚合 + 双缓冲（P1/P4）。
// ============================================================
#include "kernel_operator.h"
// MIX 形态下构建侧生成的入口桩会调 matmul::clearWorkspace(workspace)，而配套的
// matmul_intf.h 只在 c310+dump 时才被写进生成源（asc_op_compile_base/compile_op.py:576）
// ⇒ arch22 上必须自己给这个符号一个实现。本算子只用裸 Mmad/Fixpipe，不碰 KFC 消息队列，
// 空实现在语义上是安全的（真清空间的那条框架路径会把下一次 launch 毒化，见 code3.md §15.24）。
namespace matmul { __aicore__ inline void clearWorkspace(GM_ADDR) {} }

#include "sparse_flash_attention_tiling.h"
#include "tiling_key_sparse_flash_attention.h"

using namespace AscendC;

namespace sfa {
constexpr float  SOFTMAX_MIN_NUM = -2e38f;   // 官方 SOFTMAX_MIN_NUM
constexpr int32_t HQ_DIM   = 512;            // Q_D / KV_D
constexpr int32_t ROPE_DIM = 64;             // Dr
constexpr int32_t QD_ROPE  = HQ_DIM + ROPE_DIM;   // 576
constexpr uint32_t UB_BLK  = 32u;            // UB 最小块 = 256 bit
constexpr uint32_t F32_PER_BLK = UB_BLK / 4u;   // 一个块里的 fp32 个数 = 8
constexpr uint32_t LANES_PER_REP = 64u;         // 向量一次 repeat 的 fp32 lane 数 = 256B/4B

// 归约类 API（WholeReduceSum/Max）一次 repeat 最多处理的 fp32 元素数。
// 结构上 2201 的 low/high mask 合计可到 128，但官方 softmax 实现按 64 用，
// 这里也取保守值：超过就分块再跨块归约（当前 tiling 的 n_blk<=64，只走一块）。
constexpr uint32_t RED_SLAB = 64u;
// repeat 模式接口（Add/Mul 的 mask[] 重载）的满掩码低半：mask[0]=低 64bit 全 1、mask[1]=0，
// 对应 fp32 一次 repeat 的 64 个 lane（官方 SetMask 在 len==64 时也是这个值）。
// ⚠️ 形参是 `uint64_t mask[]`（非 const），constexpr 数组绑不上 ⇒ 调用处声明局部数组。
constexpr uint64_t MASK_LOW_FULL = 0xffffffffffffffffULL;
// 硬件 vexp 的输入下限：指数小于它就先夹到该值，exp(-88)≈1e-38 等价于 0。
// 没有这个下限，标量版时代由手写 ExpPoly 的 `x < -87 -> 0` 承担同一职责。
constexpr float EXP_FLOOR = -88.0f;

// ---- P19-M1d：cube 生产者 ↔ 同组两个向量消费者之间的旗标 ----
// ⚠️ 语义边界（三条全部是真机结论，不是推的）：
//   · AIC→AIV 是【广播】：一次 set，同组两颗 AIV 的收件箱各 +1（`xcoremm3` 里两 AIV 都
//     wait(5)、AIC 每轮只 set 一次还跑通了 ⇒ 一颗 set 够两颗 wait）。
//     ⇒ 所以**两颗 AIV 必须消费同一条 READY 序列**，不能各等各的号：只要有一侧多消费
//     或少消费，它留下的残值会在【下一次 launch】把第一片直接放行（不挂，静默读半片）。
//     这就是 M1d 的并行形态定成"一组 = 一个单元、两个 AIV 各吃一半头"的原因（§15.72(a)）。
//   · AIV→AIC 是【扇入】：同组两颗 AIV 各 set 同一个 id 一次，AIC 的一只 wait 才被叫醒
//     （三次独立实测：§15.26(b) 推 + `xcorec`/`xcorep` 真机、§15.68(a) `m1stage` 真机、
//     §15.72(i) 的 P66 `sub0only`（只让 sub0 交一张 ⇒ 单线单等照样 `rc=124`））。
//     ⇒ **两侧同号、AIC 每轮只等一次**，两颗各交一张正好凑齐一次扇入。旧版这里按
//     "每条线绑一颗、AIC 每片等两条线"写 ⇒ AIC 等的每条线都只有一颗会交 ⇒ 全线死锁。
//   · 跨 launch 不残留的前提：每个 set 有且仅有一个 wait 消费（§15.30(j)(2)）。
// ⚠️ ID 的可选区间比看上去窄，两头都有主：
//   · 下界 0~3 —— 核内 SetFlag<HardEvent::…> 用的就是那几位（本文件里是 0/1/2/3），
//     两侧撞号会互相吞计数。
//   · 上界 11~14 —— dav_c220 的同步实现里 SYNC_AIC_FLAG=11 / SYNC_AIV_FLAG=12 /
//     SYNC_AIC_AIV_FLAG=13 / SYNC_AIV_ONLY_ALL=14 是框架 super-kernel 自动屏障的私有旗标
//     （GetffstMsg 又把 flagId 截到 4 bit ⇒ 全空间只有 0~15）⇒ 能挑的中间带就是 4~10。
//     官方 SFA 的 MIX 版用的正是 4/5/7/8/9 这一带，这里只占 4/5 两个号。
// mode 取 2：与官方 arch2201 SFA 的 SFA_SYNC_MODE2 同值（也是 §15.30(j) 探针实测跑通的那个）。
constexpr uint32_t CF_READY = 4u;   // AIC -> 同组两颗 AIV（广播）："本片已落地"
constexpr uint32_t CF_CRED  = 5u;   // 两颗 AIV -> AIC（扇入）：本片的环槽已读走、可以覆写
// 回程环深度：AIC 最多领先消费者 RING 片。⚠️ 在【扇入】语义下 RING≥2 不安全：一只 wait
//   只要求"两颗合计"交够数，允许一颗领先到 2 片、另一颗还泡在要被覆写的那槽里（§15.72(i)
//   的收支推导）⇒ 先锁步 RING=1（每片一轮 ≈1.35 µs，§15.60 价目表）。要拿回流水深度就得换
//   按颗的 GM 计数轮询（38.6 ns/次，且天然分得清是谁交的），不是换旗标编号。
constexpr uint32_t SFA_RING = 1u;
// L0A/L0B 的分片宽度。**128 是量出来的最优点，不是照抄官方**：
//   · 往宽改（P83：256 ⇒ content 2 片 + rope 1 片 = 3 片下发）在 AIC 受限形态上实测
//     **慢 2.5 %**（同场次 3-vs-5 片，w3 +2.5 % / p6 +2.6 % / w2 +2.5 %，AIV 受限的 w4 持平；
//     两臂输出逐位相同 ⇒ 纯成本差）。⇒ 每片的成本【不是】下发条数主导，Mmad/LoadData 少两条
//     换不来任何东西，字节数与 MAC 数才是账（P82 的"每片 ≈1.5 %"是真实计算+搬运时间，别当税额）。
//   · 往窄改更亏（P82 的 5→13 片放大 = 每多一片真活 +11.8 %/8 ≈ 1.5 %）。
// rope 那一片按实际 64 列走 ⇒ 一共 5 片，bufA/bufB 只需装下最宽的一片。
constexpr uint32_t CUBE_KSLICE = 128u;

// InitBuffer 的入参向上对齐到 UB 块；host 的预算式用同样的对齐口径
__aicore__ inline uint32_t UbAlignBuf(uint32_t len)
{
    return (len + UB_BLK - 1u) / UB_BLK * UB_BLK;
}


}  // namespace sfa

template <typename DT_QUERY>
class KernelSparseFlashAttention {
public:
    __aicore__ inline KernelSparseFlashAttention() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                GM_ADDR sparse_indices, GM_ADDR actual_seq_lengths_query,
                                GM_ADDR actual_seq_lengths_kv, GM_ADDR query_rope, GM_ADDR key_rope,
                                GM_ADDR attention_out, GM_ADDR softmax_max_out, GM_ADDR softmax_sum_out,
                                const SparseFlashAttentionTilingData &tiling_data, GM_ADDR trace)
    {
        B_    = tiling_data.B;
        S1_   = tiling_data.Q_S;
        S2_   = tiling_data.KV_S;
        N1_  = tiling_data.Q_N;
        D_   = static_cast<int32_t>(tiling_data.Q_D);
        Dr_  = static_cast<int32_t>(tiling_data.Dr);
        scale_ = tiling_data.scale_value;
        mode_  = tiling_data.sparse_mode;
        sbs_   = tiling_data.sparse_block_size;
        sparseCount_ = tiling_data.sparse_count;
        nb_    = (tiling_data.nb   == 0) ? 1u : tiling_data.nb;
        nBlk_  = (tiling_data.n_blk == 0) ? 1u : tiling_data.n_blk;
        nHeadBlk_ = (N1_ + nb_ - 1) / nb_;
        // ml / LSE 的"第二半区"偏移（l 与 sum）。原先直接用 nb_，那要求 nb_ 是 8 的
        // 倍数才 256bit 对齐 —— 这就是 host 侧 NB_MIN=8 的唯一由来（见其注释）。
        // P10 要把头块摊到更多核上、nb_ 会掉到 1/2/4，所以半区偏移改成"向上取整到
        // 一个 UB 块的元素数"，两半区各占整块，DataCopyPad 的 UB 源地址仍然对齐。
        halfOff_ = static_cast<uint32_t>(sfa::UbAlignBuf(nb_ * sizeof(float)) / sizeof(float));
        // 一次"整批装载/写回"能暂存的行数（每行 D_ 个元素）。写回 scratch 借的是 kfBuf_
        //（P24 起容量 = **整个 chunk** 行 fp32），装载 scratch 借 kBuf_/krBuf_（n_blk 行
        // 原生 dtype），三者取最小才不会越界，且 host 的 UB 预算式不需要为 P5b 新增任何一项。
        stageMax_ = (nb_ < nBlk_) ? nb_ : nBlk_;
        lseOn_ = (softmax_max_out != nullptr) && (softmax_sum_out != nullptr);
        // ⚠️ 官方 BSND 变长语义：actual_seq_lengths 是【per-batch 值】arr[b]；
        //    数组长度为 1 时广播 arr[0]；未传时用满 padded 的 Q_S / KV_S
        //    （见 SEMANTICS.md §3）。
        //    注意：host 若探测不到数组长度会下发 0；此处对"缓冲存在但长度未知"
        //    按 1（广播）处理 —— 因为回退到 padded 长度会静默算错。
        qLenSize_  = tiling_data.actual_q_len_size;
        kvLenSize_ = tiling_data.actual_kv_len_size;

        if (trace != nullptr) {
            traceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(trace));
            traceOn_ = true;
            // ⚠️ 真机 ccec 禁止 aicore 函数里 float<->整数 的隐式转换，
            //    所以这里必须显式 static_cast<float>。
            const float bidx = static_cast<float>(GetBlockIdx());
            const float bnum = static_cast<float>(GetBlockNum());
            if ASCEND_IS_AIV {
                TRec(9.0f, bidx, bnum, 1.0f, 0.0f);
            } else {
                TRec(9.0f, bidx, bnum, 0.0f, 0.0f);
            }
        }

        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(query));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(key));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(value));
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sparse_indices));
        qrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(query_rope));
        krGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(key_rope));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(attention_out));
        // P19-M1d：回程环的 fp32 别名（只在 cube 路径上用）。【单槽】，就借本单元自己的
        // 那一行输出：两张表的行距逐字节相同（都是 N1*D 个 DT_QUERY），所以"第几个 float"
        // = s1Base * sizeof(DT_QUERY) / 4，s1Base 是 D=512 的整数倍 ⇒ 既整除又 32B 对齐。
        // ⚠️ P73 起环【绝不】再借 query：那一行的字节区间正好盖住本单元 A tile 的
        //   lane 0（=head0 的 query 行），Fixpipe 写进去的 fp16-as-NaN 会把下一次
        //   ND2NZ 读到的操作数毒掉，整条 L0C 就出 NaN（§15.72(i) 的 p70/p71/p72 三段论）。
        //   代价：环容量从"两行"降到"一行"⇒ host 的 CubeGate 补一条 nTile<=16*N1。
        ringOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(attention_out));
        if (softmax_max_out != nullptr) { maxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(softmax_max_out)); }
        if (softmax_sum_out != nullptr) { sumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(softmax_sum_out)); }
        // 变长长度张量：int32 一维，BSND 下是 per-batch 值
        if (actual_seq_lengths_query != nullptr) {
            qLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(actual_seq_lengths_query));
            qLenOn_ = true;
            if (qLenSize_ == 0u) { qLenSize_ = 1u; }   // 长度未知 -> 按广播处理
        }
        if (actual_seq_lengths_kv != nullptr) {
            kvLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(actual_seq_lengths_kv));
            kvLenOn_ = true;
            if (kvLenSize_ == 0u) { kvLenSize_ = 1u; } // 长度未知 -> 按广播处理
        }

        // ---- 核间切分：工作单元 = (query 行, 头块)，按单元数平均分给各核 ----
        // ⚠️ 只有向量核做工（本算子无 Cube 计算）。用 ASCEND_IS_AIV 与官方一致。
        // P10 之前单元只到"行"这一级（头块在核内循环），于是比赛平台那 6 个点
        //    （§5.8.3 反推：B*Q_S = 4 / 8 / 4 / 16 / 4 / 32 行）最多只让 4~32 个核干活。
        //    把头块也摊出去之后，同一份 K/V 搬运会在 ceil(Q_N/nb) 个核上重复，但
        //    §15.11 已裁定本机是"发射-bound"（AIV 只用到的峰值 4.5%，MTE 不是瓶颈）
        //    ⇒ 用重复搬运换核数是净赚（code3.md §15.14）。
        // ---- P11v2：sparse 列表（KV 轴）切成两半分给两个核 ----
        // host 只在"切完仍是【一核一单元】"时才下发 kv_shard=2；这里再核一遍，因为归并
        // 要用的分片 1 部分和是留在 UB 里的（同核再领一个单元就被覆盖），而 lseOn_=false
        // （平台没给 LSE 输出）时根本没有跨核通道。任一条件不满足就整个退回 1，退回是
        // 【安全】的：多启动的那一半块直接空转，剩下这些块各自算完整行，结果与不切分逐位相同。
        // ⚠️ 必须在 ASCEND_IS_AIC 的提前 return 之前算：SyncAll 要所有块调用次数一致。
        const uint32_t total0  = B_ * S1_ * nHeadBlk_;
        // MIX：GetBlockNum() 给的是【组数】（1 组 = 1 AIC + 2 AIV），AIV 块号 = group*2 + sub
        // ⇒ AIV 并行度 = 2·组数。ratio 写死 2 不查 GetTaskRatio()：AIC 侧它返回 1，
        //   两档核数会在同一份 tiling 下算歪（code3.md §15.37(b)）。
        const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum()) * 2u;
        const uint32_t coreIdx = GetBlockIdx();
        if (tiling_data.kv_shard == 2u && lseOn_ && total0 > 0u && coreNum == 2u * total0) {
            ks_ = 2u;
        }
        // ---- P19-M1d：本核是否走"cube 产 score、向量侧只吃回程环"的形态 ----
        // host 的形态门（op_host 的 CubeGate）已经核过一遍，这里再自证一遍 kernel 侧看得见的
        // 那几条，任何一条不成立就整个退回向量路径 —— 退回是【安全】的：cube_on 只是把
        // ComputeScores 换成读环，其余语义（扫描 / softmax / 写回）两条路径逐字节相同。
        //   · kv_shard 必须为 1：环借的正是"本单元自己的输出行"，而分片 1 的归并通道也是它。
        //   · nb_ == N1_ 且 2<=N1_<=16 且【N1_ 为偶数】：一个 L0C tile 装得下整组头（否则同一
        //     份 K 要被每个头块各 gather 一遍，且 M 轴要再切一层），偶数头才切得平两颗 AIV。
        //   · n_blk 是 16 的整数倍：NZ 的行按 16 个组成"分形行组"，dstNzC0Stride 与 Mmad 的
        //     n 都按整组算（host 的 CubeGate 同一口径）。
        //   · 环装得下：一片写 16(定死的 L0C 行)×n_blk 个 fp32，必须整块落在本单元那一行
        //     输出（N1*D 个 DT_QUERY）之内，否则就踩到【相邻单元】的输出行 —— 相邻单元可能
        //     正活在别的核上（host 的 CubeGate 同一口径，P73 加）。
        //   · 只有 fp16 实例进 cube 路径（§15.70(f)4）—— 与 AIC 侧的 if constexpr 配对，
        //     少一侧就会让 AIV 等一条永远不会有人 set 的旗标（§15.70(f) 记过这条）。
        cubeOn_ = (tiling_data.cube_on != 0u) && (tiling_data.kv_shard == 1u) &&
                  (nb_ == N1_) && (N1_ >= 2u) && (N1_ <= 16u) && ((N1_ & 1u) == 0u) &&
                  (D_ == sfa::HQ_DIM) && (Dr_ == sfa::ROPE_DIM) &&
                  ((nBlk_ % 16u) == 0u) && (sizeof(DT_QUERY) == 2u) &&
                  (nBlk_ <= SFA_STAGE_MAX_CUBE) &&
                  (16ull * nBlk_ * sizeof(float) <=
                   static_cast<uint64_t>(N1_) * static_cast<uint64_t>(D_) * sizeof(DT_QUERY));
        if (cubeOn_) { ks_ = 1u; }
        sub_ = coreIdx & 1u;   // MIX：AIV 块号 = group*2 + sub
        if ASCEND_IS_AIC {
            ks_ = 1u;   // AIC 不进纯 AIV 的 SyncAll（那是向量核硬件屏障，AIC 调它等一块不存在的旗标）
            unitBegin_ = 0; unitEnd_ = 0; unitStep_ = 1;
            return;
        }
        // ---- P19-M1d：cube 形态下【一组共一个单元】，两颗 AIV 各吃一半头 ----
        // READY 是广播的（见 sfa::CF_READY 那一段），所以两颗 AIV 必须消费同一条片序列 ⇒
        // 单元按【组】切、不再按【块】切。头这一维天生可分（softmax/PV/LSE/输出全按头独立），
        // 所以分法就是 sub_0 拿前一半、sub_1 拿后一半。
        // ⚠️ N1 必须是【偶数】：奇数会让 sub_1 拿到 ⌊N1/2⌋=0 个头（N1=1）而变成"只交旗标
        //    不做向量活"的搭档，那种 AIV 要把 Duplicate/WriteOut 的零长度分支一路特判下去，
        //    收益却是零（所有目标形状 N1=4/8 都是偶数）⇒ 直接在形态门里排除，退回向量路径。
        // ⚠️ nb_ 在这里折半 ⇒ 下面所有 UB 尺寸/stageMax_ 自动按半份头算，而 host 的预算是按
        //    【整份 N1】算的 ⇒ 只会更空，不会越界。halfOff_ 在上面已按整份算好 ⇒ mlBuf_/
        //    lseBuf_ 仍留整份宽度（更空，同样安全）。
        if (cubeOn_) {
            headBase_ = (sub_ == 0u) ? 0u : (N1_ >> 1);
            nb_ = N1_ >> 1;
        }
        const uint32_t total = cubeOn_ ? total0 : (total0 * ks_);
        if (coreNum == 0 || coreIdx >= coreNum) { unitBegin_ = 0; unitEnd_ = 0; unitStep_ = 1; return; }
        // ⚠️ 跨步而不是连续：单元编号 u = tok * nHeadBlk_ + hb，所以连续分配会把
        // 同一行（代价相近）整片塞给一个核；mode==3 的因果掩码下越靠后的行 token
        // 越多，连续分配的临界路径 = 最后那几个核，实测能把切分的收益全部吃掉。
        // 跨步（coreIdx, coreIdx+coreNum, ...）让每个核拿到一前一后的行，负载均衡。
        // cube 形态下步长是【组数】、起点是【组号】= coreIdx>>1（两颗 AIV 同一个单元）。
        unitBegin_ = cubeOn_ ? (coreIdx >> 1) : coreIdx;
        unitEnd_   = total;
        unitStep_  = cubeOn_ ? (coreNum >> 1) : coreNum;

        // ---- UB 缓冲（大小由 host 的 UB 预算反算保证）----
        // ⚠️ 每个尺寸都自己向上对齐到一个 UB 块（32B = 256bit）再交给 InitBuffer：
        //    这样每个缓冲的起始地址必然 256bit 对齐，DataCopy/DataCopyPad 才不会
        //    报 ADDR_MISALIGN；不对齐时行为取决于 TPipe 内部的取整方向，不可依赖。
        //    host 侧 CalcUbNeed 用同一口径逐项对齐，两边必须一致。
        pipe_.InitBuffer(qBuf_,   sfa::UbAlignBuf(nb_   * (D_ + Dr_) * sizeof(float)));
        pipe_.InitBuffer(oBuf_,   sfa::UbAlignBuf(nb_   * D_  * sizeof(float)));
        pipe_.InitBuffer(kBuf_,   sfa::UbAlignBuf(nBlk_ * D_  * sizeof(DT_QUERY)));
        // P32：**不再有独立的 vBuf_**。V 只在 SoftmaxPv 里被读，而那时 kb 已经读完
        //   （ComputeScores 的最后一加宽就是它对 kb 的最后一次读）⇒ 让 V 落在 kb 自己
        //   那块 n_blk×D×2 字节上（同 DT_QUERY、同 32B 对齐口径，尺寸逐字节相同），
        //   代价是每 chunk 多一次"V 排空 + MTE2 旗标对"，换来的是 host 预算里省掉
        //   align(n_blk×D×2) ⇒ nb≥2 的臂第一次装得下 n_blk=48。见 code3.md §15.53。
        // P81：cube 形态下 AIV 既不搬 K-rope（K/K-rope 由 AIC 自己 ND2NZ 进 L1）也不算
        //   ComputeScores ⇒ krBuf_/krfBuf_ 两块【整段不分配】。kfBuf_ 【不能】不分配 —— 它还是
        //   SoftmaxPv 第 5) 步 V 加宽的落点，而 PV 仍在向量侧；能动的是它的【行数】：
        //   P24 把 scGrp_ 抬到整个 chunk 的理由是"部分积就地压在 kf 上"，cube 侧没有
        //   ComputeScores ⇒ 理由消失 ⇒ 组宽钳回 SFA_SC_GRP_CUBE（账见 tiling.h）。
        //   host 的 CalcUbNeed(cube=true) 与这里逐项同口径，差一项就是真机越界。
        const uint32_t krBytes = cubeOn_ ? 0u : sfa::UbAlignBuf(nBlk_ * Dr_ * sizeof(DT_QUERY));
        if (krBytes != 0u) { pipe_.InitBuffer(krBuf_, krBytes); }
        pipe_.InitBuffer(sBuf_,   sfa::UbAlignBuf(nb_   * nBlk_ * sizeof(float)));
        pipe_.InitBuffer(pBuf_,   sfa::UbAlignBuf(nb_   * nBlk_ * sizeof(float)));
        pipe_.InitBuffer(mlBuf_,  sfa::UbAlignBuf(3 * halfOff_ * sizeof(float)));
        // 两个半区各占整块：sum 半区的 UB 地址因此必然 256bit 对齐，
        // DataCopyPad 才不会报 ADDR_MISALIGN。
        pipe_.InitBuffer(lseBuf_, 2u * sfa::UbAlignBuf(halfOff_ * sizeof(float)));
        // ---- 向量化 scratch ----
        // P24：**组宽 = 整个 chunk**（scGrp_ = n_blk），且不再要单独的"部分积"缓冲 ——
        // 把 P13 的"单头单位就地广播乘"推广到所有 nb：kf/krf 每来一个头就从 fp16 的
        // kBuf_/krBuf_ **重加宽一次**，然后部分积就地压在 kf/krf 上折叠。
        // 省的是"每组一次"的那批 mul/fold/reduce/add/muls 调用：每 chunk 从
        // nbCur×⌈n_blk/16⌉ 组降到 nbCur×1 组；代价是 kf 的加宽量随 nbCur 翻倍
        // （每头都要重铺一遍 32 行）。谁赢取决于 nb —— 本地实测见 code3.md §15.45。
        // ⚠️ 乘积**必须留在 fp32 域**：两个 fp16 相乘在 fp32 里逐位精确，而 P23 那种
        //    "点积全程走 fp16 原生域"把 LSE 的绝对误差从 1.8e-4 抬到 1.7e-1（**约 1000×**），
        //    比赛平台 Case1~3 当场判成 precision_ratio 0.03~0.18 —— 判据比本地的
        //    rtol=1e-2 严得多，标定见 code3.md §15.44。
        // rdBuf_：三个宽度为 redW_ 的归约工作区，P3a 用 [0]=content 行和、[redW_]=rope 行和，
        //   P2 复用同两块（[0]=Σp、[redW_]=分块临时）并另用 [2*redW_]=alpha 向量。
        //   ⚠️ 宽度是 redW_ = max(nb_, scGrp_, 2 块)，不是 nb_：score 的一条 WholeReduceSum
        //   一次出 **g 行**行和（g 最大 = 整个 chunk），P2 的归约一次出 **nbCur 行** ⇒
        //   两边都得放得下；只按 nb_ 定宽时组宽超过 nb_ 会压到下一切片（§15.13：GRP=16 时
        //   r5/big1 全错）。下限"2 块"是 MergeToken 要的 2×8 个 exp 自变量（合并窗口一个
        //   UB 块 = 8 个头）。
        scGrp_ = cubeOn_ ? ((nBlk_ < SFA_SC_GRP_CUBE) ? nBlk_ : SFA_SC_GRP_CUBE)
                         : nBlk_;
        // P81：krf 只有 ComputeScores 用（rope 部分积）⇒ cube 形态不分配；kf 保留，宽度
        //   = 上面的 scGrp_。两条口径都在 host 的 CalcUbNeed(cube=true) 里逐项对。
        const uint32_t kfBytes = sfa::UbAlignBuf(scGrp_ * D_ * sizeof(float));
        const uint32_t krfBytes = cubeOn_ ? 0u : sfa::UbAlignBuf(scGrp_ * Dr_ * sizeof(float));
        // kfBuf_ 同时是 WriteOut / ZeroPaddingOut 的 fp16/fp32 落点（一次一批 stageMax_ 行），
        // 所以批宽不能超过它的行数。今天 stageMax_ ≤ N1_ ≤ 16 < 32 恒成立，这一钳是防"以后
        // 抬 N1_ 或抬 SFA_SC_GRP_CUBE"时把两处口径错开。
        if (cubeOn_ && stageMax_ > scGrp_) { stageMax_ = scGrp_; }
        pipe_.InitBuffer(kfBuf_, kfBytes);
        if (krfBytes != 0u) { pipe_.InitBuffer(krfBuf_, krfBytes); }
        redW_ = (nb_ > scGrp_) ? nb_ : scGrp_;
        {
            const uint32_t mergeMin = 2u * sfa::F32_PER_BLK;
            if (redW_ < mergeMin) { redW_ = mergeMin; }
        }
        pipe_.InitBuffer(rdBuf_,  3u * sfa::UbAlignBuf(redW_ * sizeof(float)));
        // ---- P7：批量 fold / 广播乘的成立条件（见 ComputeScores 与 code3.md §15.12）。----
        // 一条向量调用的固定开销 ≈29 cycle，而 reduce 的每个 repeat 要 ≈8 cycle ⇒
        // 便宜的写法是"用少量 Add（1.12 cycle/rep）把行折到 64 lane，再一次 reduce"。
        // 批量写法要求：行宽是 64 的整数倍（一次 repeat 正好一个 64-lane 块），
        // 且行距换算成 32B 块后不超 uint8_t 的 255 上限。
        scBatch_ = (D_ > 0 && Dr_ > 0 &&
                    (D_ % static_cast<int32_t>(sfa::LANES_PER_REP)) == 0 &&
                    (Dr_ % static_cast<int32_t>(sfa::LANES_PER_REP)) == 0 &&
                    (D_ / static_cast<int32_t>(sfa::F32_PER_BLK)) <= 255 &&
                    (Dr_ / static_cast<int32_t>(sfa::F32_PER_BLK)) <= 255);
    }

    __aicore__ inline void Process()
    {
        // ---- P19-M1d：AIC 侧。cube_on=0 时这里什么都不做（与 P38 逐字节相同：AIC 空转）----
        if ASCEND_IS_AIC {
            if (cubeOn_) { CubeProduce(); }
            return;
        }
        // ---- 阶段 A：各分片算自己那一段 sparse 列表 ----
        // 单元编号 u = ((tok * nHeadBlk_ + hb) * ks_) + shard ⇒ 分片是【最低维】，
        // 这样下面 tok/hb 的解码口径与 P10/P14 完全一致（host 的 blockDim 反算也同一口径）。
        // ks_>=2 时 host 保证 coreNum == 单元总数 ⇒ 每核恰好一个单元，循环只走一圈。
        for (uint32_t u = unitBegin_; u < unitEnd_; u += unitStep_) {
            const uint32_t unit  = u / ks_;
            const uint32_t shard = u - unit * ks_;
            const uint32_t tok = unit / nHeadBlk_;
            const uint32_t hb  = unit - tok * nHeadBlk_;
            const uint32_t b = tok / S1_;
            const uint32_t s = tok - b * S1_;
            ProcessToken(b, s, hb, shard);
        }
        if (ks_ < 2u) { return; }
        // ---- barrier：等分片 0 的输出张量落到 GM ----
        // 无条件、且所有块都恰好走到这一次（ks_ 由各块同一份 tiling 推出）⇒ 不存在
        // §15.16 那类"各核调用次数不齐"的死锁形态。
        SyncAll();
        // ---- 阶段 B：只有最后一个分片负责归并写回（读回前面分片的结果 + 覆盖写）----
        for (uint32_t u = unitBegin_; u < unitEnd_; u += unitStep_) {
            const uint32_t unit  = u / ks_;
            const uint32_t shard = u - unit * ks_;
            if (shard + 1u != ks_) { continue; }
            const uint32_t tok = unit / nHeadBlk_;
            const uint32_t hb  = unit - tok * nHeadBlk_;
            const uint32_t b = tok / S1_;
            const uint32_t s = tok - b * S1_;
            MergeToken(b, s, hb);
        }
    }

    // 有效长度：官方 BSND 变长语义 = per-batch 值 arr[b]（见 SEMANTICS.md §3）
    //   size==1 -> 广播 arr[0]；未传 -> 用满 padded 的 Q_S / KV_S
    __aicore__ inline void GetActualLens(uint32_t b, uint32_t &actQ, uint32_t &actKV) const
    {
        actQ = S1_;
        if (qLenOn_ && qLenSize_ > 0) {
            const uint32_t idx = (qLenSize_ == 1) ? 0u : b;
            if (idx < qLenSize_) {
                const int32_t v = qLenGm_.GetValue(idx);
                if (v >= 0 && static_cast<uint32_t>(v) < actQ) {
                    actQ = static_cast<uint32_t>(v);
                }
            }
        }
        actKV = S2_;
        if (kvLenOn_ && kvLenSize_ > 0) {
            const uint32_t idx = (kvLenSize_ == 1) ? 0u : b;
            if (idx < kvLenSize_) {
                const int32_t v = kvLenGm_.GetValue(idx);
                if (v >= 0 && static_cast<uint32_t>(v) < actKV) {
                    actKV = static_cast<uint32_t>(v);
                }
            }
        }
    }

private:
    // GM -> UB 整批搬运。前提：elems*sizeof(DT_QUERY) 是 32B 的整数倍，
    // 且两端地址都 256bit 对齐 —— 由 D_=512 / Dr_=64 与 UB 分块对齐共同保证。
    __aicore__ inline void CopyGm2Ub(const LocalTensor<DT_QUERY> &dst,
                                     const GlobalTensor<DT_QUERY> &src, uint32_t elems) const
    {
        DataCopy(dst, src, DataCopyParams{1, static_cast<uint16_t>(elems * sizeof(DT_QUERY) / sfa::UB_BLK), 0, 0});
    }

    // UB -> GM 整批搬运。对齐前提同 CopyGm2Ub（长度是 32B 整数倍、两端 256bit 对齐）。
    __aicore__ inline void CopyUb2Gm(const GlobalTensor<DT_QUERY> &dst,
                                     const LocalTensor<DT_QUERY> &src, uint32_t elems) const
    {
        DataCopy(dst, src, DataCopyParams{1, static_cast<uint16_t>(elems * sizeof(DT_QUERY) / sfa::UB_BLK), 0, 0});
    }

    // fp32 的整块搬运（一个 UB 块 = 8 个 float）。P11v2 归并读回分片 0 的 (m,l) 用：
    // 跨核只能走批量 DataCopy（§15.32），而 LSE 一个单元不足 8 个 ⇒ 只能连块一起读。
    __aicore__ inline void CopyGm2UbF32(const LocalTensor<float> &dst,
                                        const GlobalTensor<float> &src) const
    {
        DataCopy(dst, src, DataCopyParams{1, 1, 0, 0});
    }

    // DT_QUERY -> fp32 的加宽搬运（K/V/Q 进向量化工作区前都要走这一步）。
    // ⚠️ 2201 上 float 进 fp32 工作区**不能用 Cast**：
    //    · float->float + CAST_NONE 根本没有指令（`dav_c220/kernel_operator_vec_vconv_impl.h:441`
    //      就是 ASCENDC_ASSERT(false)）⇒ 运行期不落指令，目标区留着垃圾；
    //    · float->float 剩下的 5 个模式（CAST_RINT/FLOOR/CEIL/ROUND/TRUNC）**全是"舍入到整数"**，
    //      没有一条是恒等搬运 —— 拿 CAST_RINT 当恒等用会把每个输入截断成整数，
    //      整个 fp32 实例的 score 偏 ~15%、输出只能是 0/±1（code3.md §15.13 真机实测）。
    //    ⇒ 同宽时改走 `Muls(x, 1.0f)`：IEEE 乘法下 x*1.0f 对**所有**浮点值逐位恒等
    //      （含 ±0、±inf、NaN），代价就是一条 vmul（≈29+1.12×rep cycle，与 Cast 同级）。
    //    half->float 反过来只开了 CAST_NONE 一条 ⇒ 按 sizeof 分派。
    template <typename T>
    __aicore__ inline void WidenToF32(const LocalTensor<float> &dst, const LocalTensor<T> &src,
                                      int32_t n) const
    {
        if constexpr (sizeof(T) == 4u) {
            Muls(dst, src, 1.0f, n);
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, n);
        }
    }

    // fp32 工作区 -> DT_QUERY 输出/暂存。同上：DT_QUERY=float 时 Cast 的任何模式都不是恒等，
    // CAST_RINT 会把累加好的 O 直接舍成整数（真机上表现就是"输出非 0 即 ±1"）。
    template <typename T>
    __aicore__ inline void PackFromF32(const LocalTensor<T> &dst, const LocalTensor<float> &src,
                                       int32_t n) const
    {
        if constexpr (sizeof(T) == 4u) {
            Muls(dst, src, 1.0f, n);
        } else {
            Cast(dst, src, RoundMode::CAST_RINT, n);
        }
    }


    // ---- traceGm_ 保留为空实现的写入路径（正式提交时 trace == nullptr，编译期即消除）----
    __aicore__ inline void TRec(float tag, float a, float b, float c, float d) const
    {
        if (!traceOn_) { return; }
        GlobalTensor<float> &g = const_cast<GlobalTensor<float> &>(traceGm_);
        uint32_t &o = const_cast<uint32_t &>(traceOff_);
        if (o + 5 > 4096) { return; }
        g.SetValue(o + 0, tag);
        g.SetValue(o + 1, a);
        g.SetValue(o + 2, b);
        g.SetValue(o + 3, c);
        g.SetValue(o + 4, d);
        o += 5;
    }

    __aicore__ inline int64_t CalcThreshold(uint32_t s, uint32_t actQ, uint32_t actKV) const
    {
        if (mode_ == 3) {
            return static_cast<int64_t>(actKV) - static_cast<int64_t>(actQ)
                   + static_cast<int64_t>(s) + 1;
        }
        return static_cast<int64_t>(actKV);
    }

    // 读一个有效稀疏块 -> token 区间 [begin, end)；false 表示扫描结束
    // tokEnd = 本分片可扫到的最后一个下标；tokStep = 相邻两个"归本分片"的下标间距
    //          （不切分时都是整表步进 1，见 ProcessToken）
    __aicore__ inline bool NextTokenBlock(uint64_t idxBase, uint32_t tokEnd, uint32_t tokStep,
                                          uint32_t &tokIdx,
                                          int64_t thr, int64_t &begin, int64_t &end) const
    {
        while (tokIdx < tokEnd) {
            const int32_t blk = idxGm_.GetValue(idxBase + tokIdx);
            tokIdx += tokStep;
            if (blk < 0) { return false; }                 // 官方：遇 -1 即停
            const int64_t b0 = static_cast<int64_t>(blk) * static_cast<int64_t>(sbs_);
            if (b0 >= thr) { continue; }                    // 官方此处是 continue 而非 break
            int64_t e0 = b0 + static_cast<int64_t>(sbs_);
            if (e0 > thr) { e0 = thr; }                     // 块被 threshold 截断
            begin = b0; end = e0;
            return true;
        }
        return false;
    }

    // 一个工作单元 = (query 行 b,s) 的一个头块 headBlk 的一个 sparse 分片 shard。
    // P10 把头块摊到各核，P11v2 再把 sparse 列表切成 ks_ 份分给各核（P18 起是**奇偶交错**
    // 而不是连续区间，原因见 ProcessToken 里 tokBeg/tokStep 那段）：
    //   shard 0 —— 照常 WriteOut，它的 attention_out / LSE 既是最终结果（分片 1 空时）
    //              也是分片 1 归并时要读的"已发布半行"（所以这里不需要任何新的发布代码）；
    //   shard 1 —— 不写回，部分和 O(未归一) + m + l 留在 oBuf_/mlBuf_ 里，等 SyncAll
    //              之后由 MergeToken 收口（每核一个单元 ⇒ 中间不会被覆盖，见 Init 的门）。
    __aicore__ inline void ProcessToken(uint32_t b, uint32_t s, uint32_t headBlk, uint32_t shard)
    {
        uint32_t actQ = 0, actKV = 0;
        GetActualLens(b, actQ, actKV);

        const uint32_t s1Base   = static_cast<uint32_t>(((static_cast<uint64_t>(b) * S1_ + s) * N1_) * D_);
        const uint32_t ropeBase = static_cast<uint32_t>(((static_cast<uint64_t>(b) * S1_ + s) * N1_) * Dr_);
        const uint64_t idxBase  = (static_cast<uint64_t>(b) * S1_ + s) * sparseCount_;

        // ⚠️ padding query 行（s >= 真实 query 长度）：输出全 0，LSE 取官方哨兵值。
        //    官方参考语义见 SEMANTICS.md 附录 C：`if s >= s1: continue`（输出保持 0）。
        //    两个 Zero* 都是【整行】写入，所以头块摊到各核后只由 headBlk==0 那个单元做
        //    一次，其余单元直接返回（同一行不会被两个核同时写）。
        //    cube 形态下 nHeadBlk_==1 ⇒ headBlk 恒 0，此时改由【sub_】这一维定谁写（两颗
        //    AIV 同一单元，只让拿到第 0 号头的那颗写整行，避免同地址双写）。
        const uint64_t lseBase = (static_cast<uint64_t>(b) * S1_ + s) * N1_;
        if (s >= actQ) {
            if (headBlk == 0 && (sub_ == 0u || !cubeOn_)) {
                ZeroPaddingOut(s1Base);
                if (lseOn_) { ZeroPaddingLse(lseBase); }
            }
            return;
        }

        const int64_t thr = CalcThreshold(s, actQ, actKV);

        LocalTensor<float> q  = qBuf_.Get<float>();
        LocalTensor<float> o  = oBuf_.Get<float>();
        LocalTensor<DT_QUERY> kb = kBuf_.Get<DT_QUERY>();
        LocalTensor<DT_QUERY> kr = krBuf_.Get<DT_QUERY>();
        LocalTensor<float> sc = sBuf_.Get<float>();
        LocalTensor<float> ml = mlBuf_.Get<float>();
        LocalTensor<float> lse = lseBuf_.Get<float>();   // sum 半区在 lse[halfOff_]，见 Init

        // cube 形态：一个单元 = 一行【全部 N1 个头】，两颗 AIV 各拿连续的一半（headBase_，
        // 形态门保证 N1 是偶数 ⇒ 两边都不空）；非 cube：单元 = 头块，n0 = 头块起始头。
        const uint32_t n0 = cubeOn_ ? headBase_ : (headBlk * nb_);
        uint32_t nbCur = N1_ - n0;
        if (nbCur > nb_) { nbCur = nb_; }

        // Q 与 Qrope 拼接：前 512 content、后 64 rope —— 整批 MTE2 + Cast（P5b）
        // P19-M1d：cube 路径上 score 由 AIC 算，qBuf_ 从头到尾没人读（SoftmaxPv 只用
        // o / sc / vb / ml）⇒ 这次装载整个省掉，它借的 kBuf_/krBuf_ 也留给 V。
        if (!cubeOn_) {
            LoadQ(q, kb, kr, s1Base, ropeBase, n0, nbCur);
        }

        // O 累加器清零；m = SOFTMAX_MIN_NUM、l = 0
        Duplicate(o, 0.0f, nbCur * static_cast<uint32_t>(D_));
        Duplicate(ml, sfa::SOFTMAX_MIN_NUM, nbCur);
        Duplicate(ml[halfOff_], 0.0f, nbCur);

        // 本分片负责的下标子集 = { pos | pos ≡ shard (mod ks_) } ∩ [0, sparseCount_)。
        // P11v2 原来切的是【连续区间】[shard·count/ks, (shard+1)·count/ks) —— 平台语义是
        // "遇 -1 即停"，而官方数据最常见的写法就是"前 V 项有效 + 尾部填 -1"，于是
        // V < count/2 时分片 1 一上手就读到 -1、整片空转，1.9× 退化成 1.0×（真机
        // p1s1h 实测复现，见 code3.md §15.35(e)）。
        // ⚠️ 停止条件与"垃圾项"的关系，两种切法**并不相同**，别当成免费午餐：
        //    连续切法下分片 1 从 count/2 起扫，只有"有效项全部挤在前半"这一种分布会失效；
        //    交错切法下第一个 -1 之后的表项**两侧都可能**被吞进来（偶/奇各自停在自己的
        //    第一个 -1），所以它依赖的是同一条官方约定——"-1 之后不再有有效项"，
        //    只是不再额外假设"有效项必须铺满前半张表"。
        //    两个分片各自扫到自己的那个 -1 就停，拿到的有效项数恒为 ⌈V/ks⌉ / ⌊V/ks⌋。
        uint32_t tokBeg = 0u;
        uint32_t tokEnd = sparseCount_;
        uint32_t tokStep = 1u;
        if (ks_ > 1u) {
            tokBeg = shard;
            tokStep = ks_;
        }

        uint32_t tokIdx = tokBeg;
        int64_t curBegin = 0, curEnd = 0;
        bool hasBlock = false;
        uint32_t pend = 0;       // UB 里"已搬好、还没发 flush"的 token 数（P21 的流水深度 = 1 chunk）
        uint32_t pendRuns = 0;   // 那批 token 由几段连续区间拼成（P32：V 的搬运要用同一张表）
        const int64_t rowBase = static_cast<int64_t>(b) * S2_;   // KV 在 GM 里的行基址（batch 维）

        // ⚠️ 必须支持"一个 sparse block 跨越多个 chunk"（sparseBlockSize 最大 128，
        //    而 UB 只能放下 n_blk 个 token；nBlk_ < sbs_ 时旧写法会越界/错算）。
        //    做法：缓冲区满就 flush，稀疏块的断点（curBegin/curEnd/hasBlock）跨 flush 保留，
        //    由 FlushChunk 的在线 softmax（mOld/mNew 重缩放）保证分段累加等价。
        //    当 nBlk_ >= sbs_ 时（如 SBS=8、nBlk=16）此循环与原逻辑逐位等价。
        //
        // 搬运：块内 token 在 GM 里是【连续】的（KV_N=1，key 布局 (B,S2,1,D)），
        //   所以一段连续区间用一条 DataCopy 整批搬进 UB，代替逐元素 GetValue/SetValue。
        //   一次 query 要 gather ~sparse_count 个 token、每 token 1088 个元素，
        //   这条聚合是"标量搬运 -> MTE 块搬运"的开关（P1）。
        //   对齐前提：D_/Dr_ 与 DT_QUERY 的乘积必为 32B 的整数倍（512/64 × 2/4 都满足），
        //   且 chunk 内每段起点偏移 = done*D_*sizeof，同样是 32B 的倍数。
        while (true) {
            // 1) P21：先把【上一轮】搬进 UB 的那个 chunk 的向量流水发出去。
            //    这一步必须在扫描之前 —— 只有 flush 已经在飞，下面那些"不带管道障碍"的
            //    标量 GM 读才有地方躲。老写法是"扫一段 → 搬运 → (满了才)flush"，于是
            //    扫描 / 搬运 / 计算三段实测完全串行（§15.38(c) 那 98~103 % 的可加性就是它），
            //    每单元 1024 次标量读 = 39.5 µs 整个摊在墙钟上（§15.39 的地板拆解）。
            if (pend > 0u) {
                FlushChunk(q, o, kb, kr, sc, ml, nbCur, pend, rowBase, pendRuns, s1Base);
                pend = 0u;
            }
            // 2) 扫满一个 chunk：只把 (token 起点, 长度) 登记进定长暂存，不碰 UB、不碰管道。
            //    段数上限 = n_blk ≤ SFA_STAGE_MAX（host 的 CalcBlocking 有这道钳，见 tiling.h）。
            uint32_t nRun = 0u;
            uint32_t cnt = 0u;
            while (cnt < nBlk_) {
                if (!hasBlock) {
                    if (!NextTokenBlock(idxBase, tokEnd, tokStep, tokIdx, thr, curBegin, curEnd)) { break; }
                    hasBlock = true;
                }
                uint32_t run = static_cast<uint32_t>(curEnd - curBegin);
                const uint32_t space = nBlk_ - cnt;      // >0：循环条件保证
                if (run > space) { run = space; }
                stageBeg_[nRun] = static_cast<int32_t>(curBegin);
                stageLen_[nRun] = run;
                ++nRun;
                cnt += run;
                curBegin += run;
                if (curBegin >= curEnd) { hasBlock = false; }
            }
            if (cnt == 0u) { break; }    // 表扫尽；UB 里那个 chunk 已在 1) 收口，pend=0
            // 3) 整批搬运。第一条 CopyGm2Ub 的隐式全管道等待等的是 1) 那条 flush，而它
            //    已经把 2) 的扫描挡在身后 ⇒ 每 chunk 的墙钟从"扫描+计算+搬运"变成
            //    "max(扫描, 计算)+搬运"。搬运次序、chunk 边界、flush 次数与老写法逐位相同。
            //    P32：这里只搬 K 与 K_rope，V 延后到 FlushChunk 里、ComputeScores 之后
            //    搬进 kb 自己那块（同尺寸、同 dtype）⇒ host 预算少一项 align(n_blk×D×2)。
            uint32_t done = 0u;
            for (uint32_t j = 0u; j < nRun; ++j) {
                const int64_t beg = static_cast<int64_t>(stageBeg_[j]);
                const uint32_t run = stageLen_[j];
                const int64_t kOff = (static_cast<int64_t>(b) * S2_ + beg) * D_;
                const int64_t rOff = (static_cast<int64_t>(b) * S2_ + beg) * Dr_;
                // P19-M1d：cube 路径上 K/K-rope 由 AIC 自己按段 ND2NZ 进 L1，AIV 只搬 V
                //（V 在 FlushChunk 里延后搬进 kb 自己那块，见 P32）⇒ 这两条搬运整个省掉，
                // 它们实测占每 chunk 墙钟的 26~31 %（§15.53 的 noK 档）。
                if (!cubeOn_) {
                    CopyGm2Ub(kb[done * D_],  kGm_[kOff],  run * D_);
                    CopyGm2Ub(kr[done * Dr_], krGm_[rOff], run * Dr_);
                }
                done += run;
            }
            pend = cnt;
            pendRuns = nRun;
        }
        if (pend > 0u) { FlushChunk(q, o, kb, kr, sc, ml, nbCur, pend, rowBase, pendRuns, s1Base); }

        // 归一化 + 写回：Muls(1/l) -> Cast 暂存 -> 整批 DataCopy（P5b）
        // ⚠️ 只有分片 0 发布：它写下的就是 MergeToken 要读回的"已发布半行"。分片 1 的
        //    部分和（未归一的 O、m、l）留在 UB 里等 SyncAll 之后收口。
        if (shard == 0u) {
            WriteOut(o, ml, lse, s1Base, lseBase + n0, n0, nbCur);
        }
    }

    // 把一行 len 个连续 fp32 就地折成 <=64 个有效 lane（向量一次 repeat 的宽度）：
    // 每步一条原地块加 dst[0:h] += dst[h:2h]。512 -> 3 步、64 -> 0 步。
    __aicore__ inline void FoldRow(LocalTensor<float> row, uint32_t len) const
    {
        for (uint32_t h = len / 2u; h >= sfa::LANES_PER_REP; h >>= 1) {
            Add(row, row, row[h], h);
        }
    }

    // 折叠后每行的有效 lane 数（FoldRow 的同一条折半规则，两处必须一致）
    __aicore__ inline uint32_t FoldedLen(uint32_t len) const
    {
        while (len > sfa::LANES_PER_REP) { len >>= 1; }
        return len;
    }

    /**
     * P7: 一次调用折 g 行（code3.md §15.12）。
     * repeat 模式的 `Add` 一个 repeat 固定 8 块 = 64 个 fp32，行距用 srcRepStride 表达，
     * 所以"第 o 块 += 第 o+step 块"这种块对，g 行可以一次做完：
     *      调用数 = rowC/64 - 1（512 -> 7 条），与行数 g 无关；
     *      原来 FoldRow 是每行 log2(rowC/64) 条（512 -> 3 条 × g 行 = 24 条）。
     * 块对划分与 FoldRow 逐级一致 ⇒ 求和顺序不变（末位也应一致），区别于 §15.12 第一版的
     * 两级归约（那条按块内顺序求和，会漂移）。
     */
    __aicore__ inline void FoldRowsBatch(LocalTensor<float> rows, uint32_t len, uint32_t g) const
    {
        const uint8_t rowBlk = static_cast<uint8_t>(len / sfa::F32_PER_BLK);   // 行距（32B 块）
        const BinaryRepeatParams prm(1, 1, 1, rowBlk, rowBlk, rowBlk);
        uint64_t msk[2] = {sfa::MASK_LOW_FULL, 0ULL};
        const uint8_t rep = static_cast<uint8_t>(g);
        for (uint32_t step = len / 2u; step >= sfa::LANES_PER_REP; step >>= 1) {
            for (uint32_t o = 0; o < step; o += sfa::LANES_PER_REP) {
                Add(rows[o], rows[o], rows[o + step], msk, rep, prm);
            }
        }
    }

    /**
     * P7: 一行 q 广播乘 g 行 k（code3.md §15.12）。
     * ⚠️ 2201 上一次 repeat 固定 8 块 = 64 个 fp32（vadd/vmul 不吃 blockNumber），所以
     * "g 行 × len 列"这块矩形只能按 64 列一刀刀切：调用数 = len/64，每条出 g 行同一块。
     * 广播靠 src1RepStride=0（官方 rmsnorm 同口径）。
     * 对 len=64（rope 侧）是 1 条顶 g 条；对 len=512 与每行一条的写法等价，胜在统一。
     */
    __aicore__ inline void MulRowsBroadcast(LocalTensor<float> dst, LocalTensor<float> src,
                                            LocalTensor<float> qRow, uint32_t len, uint32_t g) const
    {
        const uint8_t rowBlk = static_cast<uint8_t>(len / sfa::F32_PER_BLK);
        const BinaryRepeatParams prm(1, 1, 1, rowBlk, rowBlk, 0);
        uint64_t msk[2] = {sfa::MASK_LOW_FULL, 0ULL};
        const uint8_t rep = static_cast<uint8_t>(g);
        for (uint32_t c = 0; c < len; c += sfa::LANES_PER_REP) {
            Mul(dst[c], src[c], qRow[c], msk, rep, prm);
        }
    }

    /**
     * P5b: Q / Qrope 整批装载（取代每头 576 次标量 GetValue）。
     * BSND 布局下同一 token 的相邻头在 GM 里是连续的 ⇒ 一条 DataCopy 能搬 stageMax_ 个头。
     * 暂存区借 kBuf_/krBuf_（同 DT_QUERY 类型、天然 32B 对齐），再 Cast 成 fp32 拼进 q 的
     * [content | rope] 两段。kBuf_/krBuf_ 紧接着要被 KV chunk 覆盖，所以尾部必须用
     * V_MTE2 确认 Cast 已经读完（set+wait 同 id 严格配对）。
     */
    __aicore__ inline void LoadQ(LocalTensor<float> &q, LocalTensor<DT_QUERY> &kb,
                                 LocalTensor<DT_QUERY> &kr, uint32_t s1Base, uint32_t ropeBase,
                                 uint32_t n0, uint32_t nbCur)
    {
        const uint32_t rowC = static_cast<uint32_t>(D_);
        const uint32_t rowR = static_cast<uint32_t>(Dr_);
        const uint32_t qRow = rowC + rowR;
        for (uint32_t i0 = 0; i0 < nbCur; i0 += stageMax_) {
            uint32_t g = nbCur - i0;
            if (g > stageMax_) { g = stageMax_; }
            CopyGm2Ub(kb, qGm_[s1Base + (n0 + i0) * rowC], g * rowC);
            CopyGm2Ub(kr, qrGm_[ropeBase + (n0 + i0) * rowR], g * rowR);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            for (uint32_t t = 0; t < g; ++t) {
                LocalTensor<float> dst = q[(i0 + t) * qRow];
                WidenToF32(dst, kb[t * rowC], rowC);
                WidenToF32(dst[rowC], kr[t * rowR], rowR);
            }
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
    }

    /**
     * P5b: 归一化 + attention_out 整批写回 + LSE 写回。
     * 标量版每个头要 D_ 次 GetValue 再 D_ 次 SetValue（且标量 GM 写偶发丢行），这里改成
     *   Muls(1/l) -> Cast 到 kfBuf_ 暂存 -> 一条 DataCopy 覆盖 stageMax_ 个连续头。
     * 除改成乘倒数：两者相差 <=1 个 fp32 ULP（相对 6e-8），远小于 fp16 输出精度；
     * 向量管线没有标量除法（2201 上 Divs 不存在），乘倒数是唯一可向量化写法。
     * 事件：V 写暂存 -> MTE3 读，用 V_MTE3；MTE3 读完 -> 下一组 V 再写同一块暂存，用 MTE3_V。
     */
    __aicore__ inline void WriteOut(LocalTensor<float> &o, LocalTensor<float> &ml,
                                    LocalTensor<float> &lse, uint32_t s1Base, uint64_t lseOff,
                                    uint32_t n0, uint32_t nbCur)
    {
        const uint32_t rowC = static_cast<uint32_t>(D_);
        LocalTensor<DT_QUERY> st = kfBuf_.Get<DT_QUERY>();
        for (uint32_t i0 = 0; i0 < nbCur; i0 += stageMax_) {
            uint32_t g = nbCur - i0;
            if (g > stageMax_) { g = stageMax_; }
            for (uint32_t t = 0; t < g; ++t) {
                const uint32_t i = i0 + t;
                const float l = ml.GetValue(halfOff_ + i);
                LocalTensor<float> oi = o[i * rowC];
                if (l > 0.0f) {
                    Muls(oi, oi, 1.0f / l, rowC);
                } else {
                    Duplicate(oi, 0.0f, rowC);      // 整行被 mask：输出 0
                }
                PackFromF32(st[t * rowC], oi, rowC);
                if (lseOn_) {
                    // ⚠️ softmaxMax 写【未缩放】的行最大（ml[i] 里存的就是 mNew，scale 只进
                    //    score，不进 LSE）——真机双对拍实测过：乘 scale_ 会偏小 22.6 倍。
                    lse.SetValue(i, (l > 0.0f) ? ml.GetValue(i) : 0.0f);
                    lse.SetValue(halfOff_ + i, (l > 0.0f) ? l : 0.0f);
                }
            }
            SetFlag<HardEvent::V_MTE3>(1);
            WaitFlag<HardEvent::V_MTE3>(1);
            CopyUb2Gm(outGm_[s1Base + (n0 + i0) * rowC], st, g * rowC);
            SetFlag<HardEvent::MTE3_V>(1);
            WaitFlag<HardEvent::MTE3_V>(1);
        }
        if (lseOn_) {
            // 尾部那对 MTE3_V 保证：下一个头块重写 lseBuf_ 之前，本块的 DataCopyPad 已读完。
            SetFlag<HardEvent::V_MTE3>(2);
            WaitFlag<HardEvent::V_MTE3>(2);
            DataCopyPad(maxGm_[lseOff], lse,
                        DataCopyExtParams{1, static_cast<uint32_t>(nbCur * sizeof(float)), 0, 0, 0});
            DataCopyPad(sumGm_[lseOff], lse[halfOff_],
                        DataCopyExtParams{1, static_cast<uint32_t>(nbCur * sizeof(float)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(2);
            WaitFlag<HardEvent::MTE3_V>(2);
        }
    }

    /**
     * P11v2: 分片 1 在 SyncAll 之后收口 —— 读回分片 0 已发布的半行，与自己的半行合并，
     * 覆盖写 attention_out 与 LSE。
     *
     * 在线 softmax 的合并式（Ô0 = 分片 0 已归一写回的输出，O1/m1/l1 = 本分片 UB 里的
     * 未归一部分和）：
     *     m  = max(m0, m1)
     *     w0 = l0 * exp(m0 - m)         u1 = exp(m1 - m)         L = w0 + l1 * u1
     *     O  = (Ô0 * w0 + O1 * u1) / L
     * 与"整行一次算完"的差别只有求和顺序，外加 Ô0 多走了一次 DT_QUERY 往返（fp16 一个
     * ULP 量级，远小于判分口径 atol=2e-3/rtol=1e-2）⇒ 切分的用例不再与参考实现【逐位】
     * 相同，这正是 host 把 kv_shard 的门卡得很紧（只给"一核一单元"的小形状）的原因。
     *
     * ⚠️ 分片 0 的 (m0,l0) 只能批量读：GM 标量 GetValue 不是跨核通道（§15.32 实测标量
     *    发布 40 块里只有 2~3 块看得见）。而 DataCopy 的最小粒度是一个 UB 块 = 8 个 fp32
     *    ⇒ 只能"按 8 对齐的窗口"整块读，一个窗口最多带 8 个头，且窗口起点 ab 必须满足
     *    ab <= lseOff、ab+8 <= nlse。两头都成立要求 **nlse 是 8 的整数倍**（host 的门），
     *    此时 ab = lseOff 向下取整到 8 一定不越界；落在窗口外的头留给下一个窗口。
     *    （第一版按"顶格贴住右边界再取整"滑窗口，off=8/need=1 时窗口停在 [0,8) 却要读
     *    第 8 个元素 —— p1/p2 当场错 38% 的元素，code3.md §15.33。）
     * ⚠️ 半边为空时必须分情况：WriteOut 在 l<=0 时把 LSE 写成哨兵 (0,0)，所以"空的那半"
     *    的 m 已被抹成 0，不能再拿去做 max —— 否则 LSE 的 m 会被抬到 0（输出值本身是
     *    scale-invariant 的、不会被带坏，但 m 会错）。
     *      l1 <= 0：分片 0 写的就是最终值 ⇒ 按原值透传（同址同字节，重写无害）；
     *      l0 <= 0：最终值取本分片 ⇒ m=m1、L=l1、O=O1/l1（分片 0 那项权重为 0，不读它）。
     */
    __aicore__ inline void MergeToken(uint32_t b, uint32_t s, uint32_t headBlk)
    {
        uint32_t actQ = 0, actKV = 0;
        GetActualLens(b, actQ, actKV);
        if (s >= actQ) { return; }      // padding 行：分片 0 已整行写 0、LSE 写 (0,0)

        const uint32_t s1Base  = static_cast<uint32_t>(((static_cast<uint64_t>(b) * S1_ + s) * N1_) * D_);
        const uint64_t lseBase = (static_cast<uint64_t>(b) * S1_ + s) * N1_;
        const uint32_t n0 = headBlk * nb_;
        uint32_t nbCur = N1_ - n0;
        if (nbCur > nb_) { nbCur = nb_; }
        const uint64_t lseOff  = lseBase + n0;
        const uint64_t headEnd = lseOff + nbCur;

        const uint32_t rowC = static_cast<uint32_t>(D_);
        LocalTensor<float> o   = oBuf_.Get<float>();        // 本分片未归一的 O
        LocalTensor<float> ml  = mlBuf_.Get<float>();
        LocalTensor<float> q   = qBuf_.Get<float>();        // chunk 循环后即空闲，借来放 Ô0 的 fp32
        LocalTensor<float> lse = lseBuf_.Get<float>();
        LocalTensor<float> rd  = rdBuf_.Get<float>();       // m0 窗口
        LocalTensor<float> rdL = rd[redW_];                 // l0 窗口
        LocalTensor<float> rdE = rd[2 * redW_];             // 2*g 个 exp 自变量（g<=8, redW_>=16）
        LocalTensor<DT_QUERY> kb0 = kBuf_.Get<DT_QUERY>();  // 分片 0 输出的落地区（chunk 循环后空闲）
        LocalTensor<DT_QUERY> st  = kfBuf_.Get<DT_QUERY>();

        constexpr uint64_t WIN = sfa::F32_PER_BLK;          // 一个 UB 块 = 8 个 fp32
        for (uint64_t ab = lseOff & ~(WIN - 1ULL); ab < headEnd; ab += WIN) {
            // 本窗口带回来的头：[ab, ab+8) 与 [lseOff, headEnd) 的交集
            const uint32_t w  = (ab > lseOff) ? 0u : static_cast<uint32_t>(lseOff - ab);
            uint64_t last = ab + WIN;
            if (last > headEnd) { last = headEnd; }
            const uint32_t g  = static_cast<uint32_t>(last - (ab + w));
            const uint32_t i0 = static_cast<uint32_t>(ab + w - lseOff);

            // 1) 批量读回分片 0 发布的 (m,l) 与归一化输出 Ô0
            CopyGm2UbF32(rd,  maxGm_[ab]);
            CopyGm2UbF32(rdL, sumGm_[ab]);
            CopyGm2Ub(kb0, outGm_[s1Base + static_cast<uint64_t>(n0 + i0) * rowC], g * rowC);
            // 标量要读 MTE2 刚写进 UB 的窗口，而标量读不受 SetFlag/WaitFlag 约束 ⇒ 全流水栅栏
            PipeBarrier<PIPE_ALL>();

            // 2) 定合并系数（标量部分）：最终 m 落到 lse，两个指数自变量落到 rdE，再一次 Exp
            for (uint32_t t = 0; t < g; ++t) {
                const uint32_t i = i0 + t;
                const float m0 = rd.GetValue(w + t);
                const float l0 = rdL.GetValue(w + t);
                const float m1 = ml.GetValue(i);
                const float l1 = ml.GetValue(halfOff_ + i);
                if (l1 <= 0.0f) {                       // 本分片没活：分片 0 的结果就是最终结果
                    lse.SetValue(i, (l0 > 0.0f) ? m0 : 0.0f);
                    lse.SetValue(halfOff_ + i, (l0 > 0.0f) ? l0 : 0.0f);
                    continue;
                }
                const float m = ((l0 > 0.0f) && (m0 > m1)) ? m0 : m1;
                lse.SetValue(i, m);
                rdE.SetValue(2 * t, (l0 > 0.0f) ? (m0 - m) : 0.0f);
                rdE.SetValue(2 * t + 1, m1 - m);
            }
            Maxs(rdE, rdE, sfa::EXP_FLOOR, static_cast<int32_t>(2 * g));
            Exp<float>(rdE, rdE, static_cast<int32_t>(2 * g));
            PipeBarrier<PIPE_V>();

            // 3) 加权合并 + 暂存到写回区
            for (uint32_t t = 0; t < g; ++t) {
                const uint32_t i = i0 + t;
                LocalTensor<float> oi = o[i * rowC];
                LocalTensor<float> t0 = q[i * rowC];
                WidenToF32(t0, kb0[t * rowC], rowC);
                const float l0 = rdL.GetValue(w + t);
                const float l1 = ml.GetValue(halfOff_ + i);
                if (l1 <= 0.0f) {
                    PackFromF32(st[t * rowC], t0, rowC);   // 透传（fp16->fp32->fp16 逐位还原）
                    continue;
                }
                if (l0 <= 0.0f) {
                    Muls(oi, oi, 1.0f / l1, rowC);         // 只有本分片有活
                    PackFromF32(st[t * rowC], oi, rowC);
                    lse.SetValue(halfOff_ + i, l1);
                    continue;
                }
                const float w0 = l0 * rdE.GetValue(2 * t);
                const float u1 = rdE.GetValue(2 * t + 1);
                const float L  = w0 + l1 * u1;
                const float inv = 1.0f / L;
                Muls(t0, t0, w0 * inv, rowC);
                Muls(oi, oi, u1 * inv, rowC);
                Add(oi, oi, t0, rowC);
                PackFromF32(st[t * rowC], oi, rowC);
                lse.SetValue(halfOff_ + i, L);
            }

            SetFlag<HardEvent::V_MTE3>(3);
            WaitFlag<HardEvent::V_MTE3>(3);
            CopyUb2Gm(outGm_[s1Base + static_cast<uint64_t>(n0 + i0) * rowC], st, g * rowC);
            SetFlag<HardEvent::MTE3_V>(3);
            WaitFlag<HardEvent::MTE3_V>(3);
        }
        // 4) LSE 一次整批写回（源地址是 lseBuf_ 的两个半区起点，必然 256bit 对齐）
        if (lseOn_) {
            SetFlag<HardEvent::V_MTE3>(2);
            WaitFlag<HardEvent::V_MTE3>(2);
            DataCopyPad(maxGm_[lseOff], lse,
                        DataCopyExtParams{1, static_cast<uint32_t>(nbCur * sizeof(float)), 0, 0, 0});
            DataCopyPad(sumGm_[lseOff], lse[halfOff_],
                        DataCopyExtParams{1, static_cast<uint32_t>(nbCur * sizeof(float)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(2);
            WaitFlag<HardEvent::MTE3_V>(2);
        }
    }

    // P5b: padding query 行的输出整批写 0。标量逐元素写在多核并发下偶发整行丢失
    //      （VARSBS8 复现 0xAA 残留），改成"UB 零块 + DataCopy"后既快又不再复现。
    //      fp32 的 0.0 按字节全 0，所以同一块暂存对 half/fp32 输出都是合法的 0。
    __aicore__ inline void ZeroPaddingOut(uint32_t s1Base)
    {
        const uint32_t rowC = static_cast<uint32_t>(D_);
        LocalTensor<float> zf = kfBuf_.Get<float>();
        Duplicate(zf, 0.0f, stageMax_ * rowC);
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        LocalTensor<DT_QUERY> st = kfBuf_.Get<DT_QUERY>();
        for (uint32_t n0 = 0; n0 < N1_; n0 += stageMax_) {
            uint32_t g = N1_ - n0;
            if (g > stageMax_) { g = stageMax_; }
            CopyUb2Gm(outGm_[s1Base + n0 * rowC], st, g * rowC);
        }
        SetFlag<HardEvent::MTE3_V>(1);
        WaitFlag<HardEvent::MTE3_V>(1);
    }

    // padding 行的 LSE = (0, 0)，官方哨兵值；两个半区一次清零、按头块整批写
    __aicore__ inline void ZeroPaddingLse(uint64_t lseBase)
    {
        LocalTensor<float> lseP = lseBuf_.Get<float>();
        Duplicate(lseP, 0.0f, nb_);
        Duplicate(lseP[halfOff_], 0.0f, nb_);
        SetFlag<HardEvent::V_MTE3>(2);
        WaitFlag<HardEvent::V_MTE3>(2);
        for (uint32_t hb = 0; hb < nHeadBlk_; ++hb) {
            const uint32_t n0 = hb * nb_;
            uint32_t nbCur = N1_ - n0;
            if (nbCur > nb_) { nbCur = nb_; }
            DataCopyPad(maxGm_[lseBase + n0], lseP,
                        DataCopyExtParams{1, static_cast<uint32_t>(nbCur * sizeof(float)), 0, 0, 0});
            DataCopyPad(sumGm_[lseBase + n0], lseP[halfOff_],
                        DataCopyExtParams{1, static_cast<uint32_t>(nbCur * sizeof(float)), 0, 0, 0});
        }
        SetFlag<HardEvent::MTE3_V>(2);
        WaitFlag<HardEvent::MTE3_V>(2);
    }

    /**
     * score[i][j] = ( q_i[0:D] · k_j[0:D] + q_i[D:D+Dr] · k_rope_j[0:Dr] ) * scale
     *
     * 向量化口径（取代原先每对 (i,j) 576 次标量乘加）：
     *   1) 本组 g 个 token 的 K / K-rope 整行 Cast 成 fp32 —— **每个头做一次**（因为
     *      部分积就地压回 kf/krf，见下面第 3 点）。fp16 -> fp32 是精确扩展，不引入误差。
     *   2) 每条 512/64 维点积 = 一次整条 `Mul` + 折半块加 + 一次 `WholeReduceSum`，
     *      且 `WholeReduceSum` 的 repeatTime=组内 token 数 ⇒ **一条指令出 G 个行和**，
     *      结果紧凑落在 sc 的行切片上（dst 步长单位=元素，已实测）。
     *   3) 归约维（512）超过一次 repeat 的 64 lane ⇒ 先折到 64 再归约。P7 起折叠与乘法都走
     *      repeat 模式（MulRowsBroadcast / FoldRowsBatch）：一条调用同时做完组内 g 行的同一块，
     *      调用数从 g×log2 降到 len/64 - 1。行宽不是 64 的整数倍时 scBatch_=false，退回 1)+2) 的
     *      逐行写法（FoldRow）。
     *   4) P24：组宽 = 整个 chunk（scGrp_ = nBlk_），部分积**就地**写在 kf/krf 上（P13 那个
     *      "nb==1 才就地"的条件到此作废）。就地写下每个 lane 的两个源操作数地址相同、
     *      无跨 lane 依赖 ⇒ 与写到别的缓冲逐位等价。
     * 与标量版的差别只有 fp32 求和顺序（树形 vs 顺序），量级 ~1e-7 相对。
     */
    __aicore__ inline void ComputeScores(LocalTensor<float> &q, LocalTensor<DT_QUERY> &kb,
                                         LocalTensor<DT_QUERY> &kr, LocalTensor<float> &sc,
                                         uint32_t nbCur, uint32_t m)
    {
        LocalTensor<float> kf = kfBuf_.Get<float>();
        LocalTensor<float> krf = krfBuf_.Get<float>();
        // P24：部分积就地下在 kf/krf 上（pfBuf_/prfBuf_ 已经根本不分配了）。
        LocalTensor<float> pf = kf;
        LocalTensor<float> prf = krf;
        LocalTensor<float> rd = rdBuf_.Get<float>();
        LocalTensor<float> rd2 = rd[redW_];

        const uint32_t rowC = static_cast<uint32_t>(D_);
        const uint32_t rowR = static_cast<uint32_t>(Dr_);
        const uint32_t vc = FoldedLen(rowC);
        const uint32_t vr = FoldedLen(rowR);

        for (uint32_t g0 = 0; g0 < m; g0 += scGrp_) {
            const uint32_t g = (m - g0 < scGrp_) ? (m - g0) : scGrp_;
            for (uint32_t i = 0; i < nbCur; ++i) {
                // 就地把 kf/krf 用掉了，所以每个头都要从 fp16 的 kb/kr 重加宽一次。
                // P12 的同一条理由：本组 g 行在 kb 与在 kf 里都是**连续**排布的 ⇒
                // g 条加宽并成 1 条 count=g*rowC 的加宽，写入地址与取值顺序逐位不变。
                WidenToF32(kf, kb[g0 * rowC], static_cast<int32_t>(g * rowC));
                WidenToF32(krf, kr[g0 * rowR], static_cast<int32_t>(g * rowR));
                const uint32_t qb = i * (rowC + rowR);
                LocalTensor<float> qc = q[qb];
                LocalTensor<float> qr = q[qb + rowC];
                if (scBatch_) {
                    // P7（code3.md §15.12）：广播乘 + 一次调用折 g 行
                    MulRowsBroadcast(pf, kf, qc, rowC, g);
                    MulRowsBroadcast(prf, krf, qr, rowR, g);
                    FoldRowsBatch(pf, rowC, g);      // rowC=512 -> 7 条；rowR=64 -> 0 条
                    FoldRowsBatch(prf, rowR, g);
                } else {
                    for (uint32_t t = 0; t < g; ++t) {
                        Mul(pf[t * rowC], kf[t * rowC], qc, rowC);
                        Mul(prf[t * rowR], krf[t * rowR], qr, rowR);
                        FoldRow(pf[t * rowC], rowC);
                        FoldRow(prf[t * rowR], rowR);
                    }
                }
                // 行与行的块间距 = rowC / 每块元素数；一次调用做完本组 g 行
                LocalTensor<float> out = sc[i * nBlk_ + g0];
                WholeReduceSum(rd, pf, vc, g, 1, 1, rowC / sfa::F32_PER_BLK);
                WholeReduceSum(rd2, prf, vr, g, 1, 1, rowR / sfa::F32_PER_BLK);
                Add(out, rd, rd2, g);
                Muls(out, out, scale_, g);
            }
        }
    }

    /**
     * P19-M1d：score 不再由本核算，改成读同组 AIC 放进 GM 的回程环。
     *
     * 环槽位置：本单元【自己的那一行输出】，全程只有这一槽（片与片之间靠 SFA_RING=1 的
     * credit 串行，不存在"两片同时在环里"⇒ 不需要乒乓）。环借的是输出行的【头部字节】，
     * 本单元收工时 WriteOut 会把整行 N1*D 个 fp16 覆回去，所以留在里面的垃圾出不了门。
     * 两张表的行距逐字节相同（N1*D 个 DT_QUERY），所以"第几个 float"= s1Base>>1。
     *
     * 布局：AIC 用 Fixpipe 写出 [16 行(头) × nTile 列(token)] 的 fp32 ND，行距 = nTile，
     * 所以本函数按【每头一条】DataCopy 读回 nTile 个 float，落到 sc 的行（行距 nBlk_）。
     *   ⚠️ 不能一次读 nbCur*nTile：那是"行距 nTile 的 ND"，而 sBuf_ 的行距是 nBlk_，
     *      只有 nTile==nBlk_ 时才连续 —— 末片（cnt<n_blk ⇒ nTile<n_blk）就会串行错行。
     *   ⚠️ 每头一条 = nbCur 次 DataCopy 调用，固定税 ≈50 ns/次（§15.70(d)）⇒ 4 头 0.2 µs，
     *      相对被替换掉的 ComputeScores（实测每 chunk 数 µs 级）是净赚。
     *
     * 对齐：dst 偏移 i*nBlk_ 个 float（nBlk_ 是 8 的倍数 ⇒ 32B 对齐），
     *       src 偏移 (s1Base>>1 + i*nTile)*4 B（s1Base 是 D_=512 的倍数 ⇒ 32B 对齐）。
     * 尾列：nTile 向上取整到 16，多出来的列是 AIC 写的垃圾/AIV 从不读（SoftmaxPv 只看前 m 列）。
     */
    __aicore__ inline void ScoreFromRing(LocalTensor<float> &sc, uint32_t nbCur, uint32_t m,
                                         uint32_t s1Base)
    {
        CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_READY);
        const uint32_t nTile = (m + 15u) & ~15u;
        const uint64_t base = static_cast<uint64_t>(s1Base) >> 1;
        const uint32_t blkPerRow = nTile * static_cast<uint32_t>(sizeof(float)) / sfa::UB_BLK;
        const DataCopyParams dcp{1, static_cast<uint16_t>(blkPerRow), 0, 0};
        for (uint32_t i = 0u; i < nbCur; ++i) {
            // 环里躺的是本行【整份 N1 个头】，本颗 AIV 只挑自己那半 ⇒ 行号 = headBase_+i。
            DataCopy(sc[i * nBlk_],
                     ringOutGm_[base + static_cast<uint64_t>(headBase_ + i) * nTile], dcp);
        }
        // ⚠️ DataCopy 是 MTE2 队列的异步搬：不插 MTE2_V 对的话，下面那条 Muls 会先在
        //    【还没落地的 UB】上做一次乘 scale，然后 DMA 才把**未乘 scale**的原值盖回来
        //    ⇒ SoftmaxPv 读到的是裸的 QKᵀ（大 22.63 倍 ⇒ softmax 退化成 one-hot）。
        //    真机 p1 的指纹：head 3 的 softmaxMax = 26.6135 = 本地参考的**未缩放**行最大
        //    逐位相同，而 softmaxSum = 1.22（one-hot）。
        SetFlag<HardEvent::MTE2_V>(3);
        WaitFlag<HardEvent::MTE2_V>(3);
        // scale 留在向量侧乘：与 ComputeScores 的最后一道 `Muls(out, out, scale_, g)` 同一
        // 条指令、同一个次序（先点后乘），所以 LSE 与输出的语义逐位不变。
        Muls(sc, sc, scale_, nbCur * nBlk_);
    }

    /** 处理一个 KV chunk：score -> (V 补搬) -> 在线 softmax 更新 (m, l, O) */
    __aicore__ inline void FlushChunk(LocalTensor<float> &q, LocalTensor<float> &o,
                                      LocalTensor<DT_QUERY> &kb,
                                      LocalTensor<DT_QUERY> &kr, LocalTensor<float> &sc,
                                      LocalTensor<float> &ml, uint32_t nbCur, uint32_t m,
                                      int64_t rowBase, uint32_t nRun, uint32_t s1Base)
    {
        // ⚠️ 跨流水线同步（CANN 9.0.0 / arch2201 不插自动同步，实测反汇编里
        //    没有任何 set_flag/wait_flag）：
        //   MTE2 -> V：本 chunk 的 kb/kr 是刚由 DataCopy 搬进来的，V 读之前
        //     必须等 MTE2 队列走到这个标记点。
        //   V -> MTE2：调用方下一次 DataCopy 会覆盖同一块 UB，必须先等本函数的
        //     V 读完。
        //  两对都是"紧挨着的 set+wait"，同 id 严格配对 —— 不会出现 wait 多于 set
        //  的死锁（上道题 SyncAll 就是这类坑）。P4 双缓冲时把第二对换成对侧槽位。
        //  P19-M1d 之后仍然成立：cube 路径上这两对退化成"空 set + 等它"，语义不变。
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        LocalTensor<DT_QUERY> vb = kBuf_.Get<DT_QUERY>();
        // id=1 那两对是给这次 kb 复用用的：
        //   V_MTE2 保证【上一处】发给 kb 的向量读已经退休，MTE2 才准写；
        //   MTE2_V 保证 V 搬完，SoftmaxPv 才准读。两条都是"紧挨着"，与另两对同构。
        if (cubeOn_) {
            // P79：cube 路径把 V 的搬运提到【等 CF_READY 之前】。V 只依赖 stageLen_/GM，
            //   不依赖本片的 score ⇒ 这段 DMA 正好压在 AIC 算本片 score 的时间里跑；
            //   旧写法是"先等 READY、再搬 V"，于是 AIV 在 READY 之前整段空转 —— 与 (g)
            //   在 AIC 侧抓到的那笔空转同构（那边每片省 1.23~1.71×）。
            //   MTE2 队列是 FIFO：环拷贝排在 V 之后，两者都在 SoftmaxPv 之前被 wait 齐；
            //   而交还 credit 的位置没动（仍是 SoftmaxPv 读完 sc 之后）⇒ 握手记账逐字不变。
            SetFlag<HardEvent::V_MTE2>(1);
            WaitFlag<HardEvent::V_MTE2>(1);
            {
                uint32_t done = 0u;
                for (uint32_t j = 0u; j < nRun; ++j) {
                    const uint32_t run = stageLen_[j];
                    CopyGm2Ub(vb[done * D_], vGm_[(rowBase + stageBeg_[j]) * D_], run * D_);
                    done += run;
                }
            }
            SetFlag<HardEvent::MTE2_V>(1);       // 只 set，wait 挪到 ScoreFromRing 之后
            ScoreFromRing(sc, nbCur, m, s1Base);
            WaitFlag<HardEvent::MTE2_V>(1);
        } else {
            // 1) score = (q·k + q_rope·k_rope) * scale —— 向量路径（P32/P38 原次序）
            ComputeScores(q, kb, kr, sc, nbCur, m);
            SetFlag<HardEvent::V_MTE2>(1);
            WaitFlag<HardEvent::V_MTE2>(1);
            {
                uint32_t done = 0u;
                for (uint32_t j = 0u; j < nRun; ++j) {
                    const uint32_t run = stageLen_[j];
                    CopyGm2Ub(vb[done * D_], vGm_[(rowBase + stageBeg_[j]) * D_], run * D_);
                    done += run;
                }
            }
            SetFlag<HardEvent::MTE2_V>(1);
            WaitFlag<HardEvent::MTE2_V>(1);
        }
        // 3) 行最大 + 在线 softmax + PV 累加 —— 向量化（P2）
        SoftmaxPv(o, vb, sc, ml, nbCur, m);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        // P19-M1d：到这里本片的 sc 已经被 V 全程读过 ⇒ 环槽可以覆写，交还一张 credit。
        //   ⚠️ notify 只能挂 MTE3：官方 arch22 的 SFA（同一套 MIX 形态、同一批 FFTS 旗标）
        //   里 AIV→AIC 方向的 CrossCoreSetFlag 无例外全是 PIPE_MTE3
        //   （`refs/sfa/cann_builtin_900/sparse_flash_attention_service_vector_mla.h:1102,1107`），
        //   AIC→AIV 方向全是 PIPE_FIX。挂 PIPE_V 的 notify 实测**到不了对侧**（AIC 一 wait
        //   就永挂，单片用例也挂 —— §15.72(g) 的 gatecut/rdyonly 对照就是这个形状）。
        //   又因为上面那对 V_MTE2(0) 只挡【MTE2 队列】、挡不住 MTE3 ⇒ 这里补一对 V_MTE3，
        //   让"V 把 sc 读干净"成为这次 notify 的真正前置。
        if (cubeOn_) {
            SetFlag<HardEvent::V_MTE3>(1);
            WaitFlag<HardEvent::V_MTE3>(1);
            CrossCoreSetFlag<2, PIPE_MTE3>(sfa::CF_CRED);   // 两颗同号 ⇒ AIC 一 wait 收齐
        }
    }

    /**
     * 在线 softmax 的后半段，全向量化（取代原先每 chunk nbCur*m*D_ 次标量乘加）：
     *   mNew = max(mOld, rowmax(S))
     *   P    = exp(S - mNew)                  （硬件 vexp）
     *   O    = O * alpha + P · V,  l = l * alpha + ΣP,  alpha = exp(mOld - mNew)
     *
     * 与标量版的差别只有：求和/求最大从顺序改成树形（~1e-7 相对），
     * exp 从手写泰勒改成硬件 vexp（|x|<=88 内相对误差 ~1e-6，远小于判据 rtol=1e-2）。
     *
     * ⚠️ 向量管线是异步的：凡"向量写完 -> 标量读同一块 UB"处都要 PipeBarrier<PIPE_V>，
     *    否则会读到旧值（这里三处，每 chunk 共 3 条 barrier，成本可忽略）。
     */
    __aicore__ inline void SoftmaxPv(LocalTensor<float> &o, LocalTensor<DT_QUERY> &vb,
                                     LocalTensor<float> &sc, LocalTensor<float> &ml,
                                     uint32_t nbCur, uint32_t m)
    {
        LocalTensor<float> p = pBuf_.Get<float>();
        LocalTensor<float> rd = rdBuf_.Get<float>();
        LocalTensor<float> r0 = rd;                    // P3a: content 行和 / P2: ΣP
        LocalTensor<float> r1 = rd[redW_];             // P3a: rope 行和    / P2: 分块临时
        LocalTensor<float> av = rd[2 * redW_];         // alpha 向量
        LocalTensor<float> mx = ml[2 * halfOff_];      // 本 chunk 的行最大 = 跨 chunk 的 mNew
        const int32_t rowStep = static_cast<int32_t>(nBlk_ / sfa::F32_PER_BLK);  // 行间距（块）
        const uint32_t rowC = static_cast<uint32_t>(D_);

        // ---- 1) 行最大：一条 WholeReduceMax 做 nbCur 行，列数超过 RED_SLAB 时分块取 max
        for (uint32_t k0 = 0; k0 < m; k0 += sfa::RED_SLAB) {
            const uint32_t w = (m - k0 < sfa::RED_SLAB) ? (m - k0) : sfa::RED_SLAB;
            if (k0 == 0) {
                WholeReduceMax<float>(mx, sc[k0], static_cast<int32_t>(w),
                                      static_cast<int32_t>(nbCur), 1, 1, rowStep,
                                      ReduceOrder::ORDER_ONLY_VALUE);
            } else {
                WholeReduceMax<float>(r1, sc[k0], static_cast<int32_t>(w),
                                      static_cast<int32_t>(nbCur), 1, 1, rowStep,
                                      ReduceOrder::ORDER_ONLY_VALUE);
                Max(mx, mx, r1, nbCur);
            }
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < nbCur; ++i) {        // mNew 还要和历史行最大 mOld 取 max
            const float mOld = ml.GetValue(i);
            const float v = mx.GetValue(i);
            mx.SetValue(i, (v > mOld) ? v : mOld);
        }

        // ---- 2) P = exp(S - mNew)：每行一次 Adds + 一次 vexp（先夹下限，避免 vexp 下溢域外）
        //      2201 没有标量减的 `Subs`（那个重载只在 3510/5102/3003/3113 编译），
        //      取负后走 `Adds` —— IEEE 下两者逐位等价。
        for (uint32_t i = 0; i < nbCur; ++i) {
            LocalTensor<float> pi = p[i * nBlk_];
            Adds(pi, sc[i * nBlk_], -mx.GetValue(i), m);
            Maxs(pi, pi, sfa::EXP_FLOOR, m);
            Exp<float>(pi, pi, static_cast<int32_t>(m));
        }

        // ---- 3) ΣP（分块累加）与 alpha
        Duplicate(r0, 0.0f, static_cast<int32_t>(nbCur));
        for (uint32_t k0 = 0; k0 < m; k0 += sfa::RED_SLAB) {
            const uint32_t w = (m - k0 < sfa::RED_SLAB) ? (m - k0) : sfa::RED_SLAB;
            WholeReduceSum<float>(r1, p[k0], static_cast<int32_t>(w),
                                  static_cast<int32_t>(nbCur), 1, 1, rowStep);
            Add(r0, r0, r1, nbCur);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < nbCur; ++i) {         // alpha 的指数 = mOld - mNew <= 0
            av.SetValue(i, ml.GetValue(i) - mx.GetValue(i));
            ml.SetValue(i, mx.GetValue(i));            // m 更新为 mNew
        }
        Maxs(av, av, sfa::EXP_FLOOR, nbCur);
        Exp<float>(av, av, static_cast<int32_t>(nbCur));
        PipeBarrier<PIPE_V>();

        // ---- 4) O 重缩放 + l 更新
        for (uint32_t i = 0; i < nbCur; ++i) {
            const float a = av.GetValue(i);
            LocalTensor<float> oi = o[i * rowC];
            Muls(oi, oi, a, rowC);
            ml.SetValue(halfOff_ + i, ml.GetValue(halfOff_ + i) * a + r0.GetValue(i));
        }

        // ---- 5) PV 累加：O_i += P_ij * V_j。V_j 对全部头复用，所以 j 在外层。
        //      Axpy = dst += src*scalar，一条指令顶替原来的 Muls + Add 两条。
        // P12：V 的加宽按 scGrp_ 行一批（kfBuf_ 正好装得下一整组 fp32 展开），
        //      j 的升序不变 ⇒ Axpy 的累加顺序逐位不变，只省掉 (g-1) 条固定开销。
        LocalTensor<float> vf = kfBuf_.Get<float>();   // ComputeScores 之后即空闲，借一组
        for (uint32_t j0 = 0; j0 < m; j0 += scGrp_) {
            const uint32_t gv = (m - j0 < scGrp_) ? (m - j0) : scGrp_;
            WidenToF32(vf, vb[j0 * rowC], static_cast<int32_t>(gv * rowC));
            for (uint32_t t = 0; t < gv; ++t) {
                const LocalTensor<float> vt = vf[t * rowC];
                const uint32_t j = j0 + t;
                for (uint32_t i = 0; i < nbCur; ++i) {
                    Axpy<float, float>(o[i * rowC], vt, p.GetValue(i * nBlk_ + j), rowC);
                }
            }
        }
    }

    // ==================== P19-M1d：AIC 侧的 Cube 生产者 ====================
    /**
     * 一组一份的工作区句柄 + 参数模板。
     *
     * L1 布局（元素号，DT_QUERY=fp16）：
     *   [0, 16·(D+Dr))          A content/rope：本单元那一行的【全部 N1 个头】
     *   [16·(D+Dr), +nBlk·D)    B content：n_tile 个 token × 512 列（每片重新 gather）
     *   [..., +nBlk·Dr)         B rope
     * ⚠️ A 侧是【整份头】而不是半份：tile 的 M 轴就是一行 token 的全部头，两颗 AIV 各自
     *    只读自己那几行（headBase_ 起 nb_ 行）。分成两份 A 是上一版"一 AIV 一单元"留下的
     *    形状，那一版因为 READY 是广播的而必挂（§15.72(a)）。
     * ⚠️ B 侧两块按【最大 nBlk_】错开，而每片的 dstNzC0Stride 按【本片 n_tile】算 ——
     *    末片 n_tile<nBlk_ 时中间留一段空隙，只会浪费 L1，不会串位（每片的 LoadData
     *    只在自己的 n_tile·D 段里走）。
     */
    struct CubeCtx {
        LocalTensor<DT_QUERY> l1qa, l1qr, l1ka, l1kr;
        // L0A/L0B 各【两份】乒乓，口径同官方 arch22 mm1（`InitBuffer(tmpBufL0A,
        // L0A_PP_SIZE * 2)` + `aL0TensorPingPong[(abL0BufIter % 2) * …]`）。
        // 单缓冲在真机上的表现是 DFX `L0B read/write conflict in the MTE (same
        // address)`（retCode=0x26 / InnerCode=0x7150026，§15.72(f)）：链式 Mmad 的
        // unitFlag=0b10 表示"本单元的操作数还要被链上的后续单元用"，所以 M 队列
        // 排空并不等于 L0A/L0B 已释放，下一个切片再写同一地址就撞上。
        LocalTensor<DT_QUERY> l0a[2], l0b[2];
        LocalTensor<float> l0c;
        Nd2NzParams nzA, nzAr, nzKa, nzKr;
        LoadData2DParams ldA, ldB;
        MmadParams mp;
        FixpipeParamsV220 fx;
    };

    /** 换一个工作单元：算基址/阈值、重置扫描游标，并把这一行的 Q 装进 L1 的 A tile */
    __aicore__ inline void CubeUnitBegin(CubeCtx &ctx)
    {
        const uint32_t unit = cu_;
        const uint32_t tok = unit / nHeadBlk_;
        const uint32_t b = tok / S1_;
        const uint32_t s = tok - b * S1_;
        uint32_t actQ = 0, actKV = 0;
        GetActualLens(b, actQ, actKV);
        cskip_ = (s >= actQ) ? 1u : 0u;
        cthr_ = CalcThreshold(s, actQ, actKV);
        ctok_ = 0u;
        csegB_ = 0; csegE_ = 0; chas_ = 0u;
        const uint64_t row = static_cast<uint64_t>(b) * S1_ + s;
        cs1_  = static_cast<uint32_t>(row * N1_ * static_cast<uint64_t>(D_));
        cidx_ = row * sparseCount_;
        crb_  = static_cast<int64_t>(b) * S2_;
        if (cskip_ != 0u) { return; }          // padding 行：ProcessToken 那边也不产片
        const uint64_t ropeBase = row * N1_ * static_cast<uint64_t>(Dr_);
        ctx.nzA.nValue = N1_;                  // ⚠️ 整份头，不是 AIV 的半份 nb_
        DataCopy(ctx.l1qa, qGm_[cs1_], ctx.nzA);
        ctx.nzAr.nValue = N1_;
        DataCopy(ctx.l1qr, qrGm_[ropeBase], ctx.nzAr);
    }

    /**
     * 产一个 chunk 的 score 放进回程环。false ⇒ 当前单元已产完（调用方换单元）。
     *
     * 数值口径：score = (Q·Kᵀ + Qrope·Kropeᵀ)·scale，Mmad 两边都是"行 = k 轴"的 NZ，
     * 所以【不需要转置】(ifTranspose=false，§15.70(f))；累加维从向量版的"折半树形求和"
     * 换成 cube 的 fp16 乘 / fp32 累加，误差量级与向量的 fp32 树形求和同级（~1e-7 相对），
     * 且 scale 仍然留在向量侧乘 ⇒ 与 ComputeScores 的差异只有求和顺序（见 §15.72(e) 的
     * 判据口径：超差应为 0，"逐位一致"这一列预期变红）。
     */
    __aicore__ inline bool CubeOneChunk(CubeCtx &ctx)
    {
        if (cskip_ != 0u) { return false; }
        // 1) 扫描：与 ProcessToken 的 chunk 切分【同一套判据、同一个步进】——
        //    两侧的片数必须逐片相等，否则 AIV 的 wait 与 AIC 的 set 就对不上（§15.71(d)）。
        uint32_t nRun = 0u;
        uint32_t cnt = 0u;
        while (cnt < nBlk_) {
            if (chas_ == 0u) {
                int64_t bg = 0, en = 0;
                if (!NextTokenBlock(cidx_, sparseCount_, 1u, ctok_, cthr_, bg, en)) {
                    break;
                }
                csegB_ = bg; csegE_ = en; chas_ = 1u;
            }
            uint32_t run = static_cast<uint32_t>(csegE_ - csegB_);
            const uint32_t space = nBlk_ - cnt;      // >0：循环条件保证
            if (run > space) { run = space; }
            stageBeg_[nRun] = static_cast<int32_t>(csegB_);
            stageLen_[nRun] = run;
            ++nRun;
            cnt += run;
            csegB_ += run;
            if (csegB_ >= csegE_) { chas_ = 0u; }
        }
        if (cnt == 0u) { return false; }              // 本单元产完（表扫尽）
        const uint32_t nTile = (cnt + 15u) & ~15u;    // NZ 的行按 16 个一组 ⇒ 向上取整
        // 3) 逐段 ND2NZ 收 K / K-rope 进 L1 的 B tile。行偏移 = 已收行数（官方 DataCopyPA
        //    同一口径：块起点不是 16 的倍数也成立，m1g 档实测 mismatchGather=0）。
        {
            uint32_t done = 0u;
            for (uint32_t j = 0u; j < nRun; ++j) {
                const uint64_t rn = static_cast<uint64_t>(crb_) +
                                    static_cast<uint64_t>(stageBeg_[j]);
                const uint32_t len = stageLen_[j];
                ctx.nzKa.nValue = len; ctx.nzKa.dstNzC0Stride = nTile;
                ctx.nzKr.nValue = len; ctx.nzKr.dstNzC0Stride = nTile;
                DataCopy(ctx.l1ka[done * 16u], kGm_[rn * static_cast<uint64_t>(D_)], ctx.nzKa);
                DataCopy(ctx.l1kr[done * 16u], krGm_[rn * static_cast<uint64_t>(Dr_)], ctx.nzKr);
                done += len;
            }
        }
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        // 4) 五个 k 切片：content 4×128 + rope 1×64，全部累加进同一块 L0C。
        //    切片基址与官方 ComputeMm1 同式：A 侧 `mSize·kSlice·j`、B 侧 `nSize·kSlice·j`
        //    （NZ tile 里列分形在外、行在内 ⇒ 一个 k 切片就是连续一段）。
        for (uint32_t j = 0u; j < 5u; ++j) {
            const bool rp = (j == 4u);
            const uint32_t kk = rp ? static_cast<uint32_t>(Dr_) : sfa::CUBE_KSLICE;
            const uint32_t nF = kk / 16u;
            const uint32_t pb = (cbu_ + j) & 1u;      // L0A/L0B 乒乓槽（见 CubeCtx 的注释）
            const LocalTensor<DT_QUERY> l0a = ctx.l0a[pb];
            const LocalTensor<DT_QUERY> l0b = ctx.l0b[pb];
            ctx.ldA.repeatTimes = static_cast<uint8_t>(nF);              // A 只有一个行分形
            ctx.ldB.repeatTimes = static_cast<uint8_t>((nTile / 16u) * nF);
            if (rp) {
                LoadData(l0a, ctx.l1qr, ctx.ldA);
                LoadData(l0b, ctx.l1kr, ctx.ldB);
            } else {
                LoadData(l0a, ctx.l1qa[j * 16u * kk], ctx.ldA);
                LoadData(l0b, ctx.l1ka[j * nTile * kk], ctx.ldB);
            }
            SetFlag<HardEvent::MTE1_M>(1);
            WaitFlag<HardEvent::MTE1_M>(1);
            ctx.mp.k = static_cast<uint16_t>(kk);
            ctx.mp.n = static_cast<uint16_t>(nTile);
            ctx.mp.cmatrixInitVal = (j == 0u);
            // arch22 的"链尾"标志：非末片 0b10（继续累加），末片 0b11（可以搬出）。
            // 官方 ComputeMm1 同一取值，注释原文就是"累加最后一次翻转 flag，表示可以搬出"。
            ctx.mp.unitFlag = rp ? 0b11u : 0b10u;
            Mmad(ctx.l0c, l0a, l0b, ctx.mp);
            // m/16 × n/16 = 3 < 10 的小方块：M 队列不插自动同步 ⇒ 显式栅栏（官方同处理）
            PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>(2);
            WaitFlag<HardEvent::M_MTE1>(2);
        }
        cbu_ += 5u;
        // 5) L0C -> GM 回程环：ND 写出，行距 = nTile 个 float（AIV 按头逐行读回）
        //    2) credit 门挪到了这里（P76）：环只有 SFA_RING=1 个槽，写【本片】前要等
        //       两颗 AIV 把【上一片】读走（两侧同号，一 wait 收齐两张 = 扇入，§15.72(j)）。
        //    ⚠️ 位置就是本轮的全部变量：旧版这道门在 gather 之前 ⇒ AIC 一进门就泡在等待里，
        //       本片的搬 K + 5 次 Mmad（≈AIC 每片成本）白白排在 AIV 的尾巴后面，两台机器
        //       从不重叠。挪到 Fixpipe 之前 ⇒ 搬 K/Mmad 与 AIV 消费上一片并行，AIC 只在
        //       "真的要覆写环槽"那一刻才停。等待次数、收支、排空口径一字未动。
        //       L0C 不用乒乓：本片 Mmad 与本片 Fixpipe 之间已有 M_FIX 对，而下一片的
        //       Mmad 排在【上次】的 PIPE_ALL 栅栏之后 ⇒ L0C 的覆写天然被挡住。
        if (cp_ >= sfa::SFA_RING) {
            CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_CRED);
        }
        SetFlag<HardEvent::M_FIX>(3);
        WaitFlag<HardEvent::M_FIX>(3);
        const uint64_t fbase = static_cast<uint64_t>(cs1_) >> 1;   // DT_QUERY 元素号 -> float 号
        ctx.fx.nSize = static_cast<uint16_t>(nTile);
        ctx.fx.dstStride = nTile;
        // 单槽：本单元自己的输出行（P73 起不再按 cp_ 奇偶翻到 query —— 那正是 NaN 的源头）。
        Fixpipe(ringOutGm_[fbase], ctx.l0c, ctx.fx);
        // Fixpipe 是 FIX 队列的异步批量写：旗标前必须把整条队列排空，否则 AIV 会读到
        // 半片（§15.71 记的验证链就是这个形状：M_FIX 对 -> Fixpipe -> PIPE_ALL 栅栏 -> set）。
        PipeBarrier<PIPE_ALL>();
        // 一条【广播】旗标同时武装本组两颗 AIV ⇒ 两侧各自的片序列必须逐片等长（形态门
        // 保证一颗 AIV 一份单元、两颗同吃一片，§15.72(a)）。
        CrossCoreSetFlag<2, PIPE_FIX>(sfa::CF_READY);
        ++cp_;
        return true;
    }

    /**
     * AIC 主循环：【一组一条单元流】。
     *
     * 认领式与 AIV 侧逐字同式（Init：unitBegin_=coreIdx>>1、unitStep_=组数）：AIC 的块号
     * 就是组号，跨步就是组数 ⇒ 两颗 AIV（块号 2g/2g+1）合起来正好覆盖本组那条流，一片不多
     * 一片不少。上一版这里是"两个搭档各一条流、按【片】轮转"，那是"一 AIV 一单元"的形状，
     * 而 READY 是广播的 ⇒ 两颗 AIV 会消费到对方的片号，跨启动留残值（§15.72(a)）。
     * 稳态无损的条件：AIC 每片 ≪ 两颗 AIV 合起来每片（预算 3 µs vs 实测 ~24 µs/头，8× 余量）。
     */
    __aicore__ inline void CubeProduce()
    {
        // 只有 fp16 实例进这条路（host 的 CubeGate 与 kernel Init 的自证门同一口径）。
        // ⚠️ 必须用 if constexpr 包住整个函数体：Process() 里那一句已经 odr-use 了本函数，
        //    fp32 实例也会生成它 —— 而它引用的 Mmad/Fixpipe 组合在 float 实例下 L0C
        //    布局假设完全不同（而且那条路永远不会有旗标来）。
        if constexpr (sizeof(DT_QUERY) == 2u) {
            const uint32_t total = B_ * S1_ * nHeadBlk_;
            cstep_ = static_cast<uint32_t>(GetBlockNum());   // 组数 = AIC 块数 = 单元流条数
            cu_ = static_cast<uint32_t>(GetBlockIdx());
            cp_ = 0u;
            cbu_ = 0u;
            const uint32_t rowA = static_cast<uint32_t>(D_);
            const uint32_t rowR = static_cast<uint32_t>(Dr_);
            const uint32_t aPart = 16u * (rowA + rowR);   // A tile：本单元那一行的全部 N1 个头
            const uint32_t bPart = nBlk_ * (rowA + rowR); // B tile（两块按最大档错开）

            TBuf<TPosition::A1> bufL1;
            TBuf<TPosition::A2> bufA;
            TBuf<TPosition::B2> bufB;
            TBuf<TPosition::CO1> bufC;
            // L0A/L0B 各【两份】：乒乓槽尺寸 = 一个切片的字节数 ×2（口径同官方 L0A_PP_SIZE*2）。
            const uint32_t l0aPP = 16u * sfa::CUBE_KSLICE * static_cast<uint32_t>(sizeof(DT_QUERY));
            const uint32_t l0bPP = nBlk_ * sfa::CUBE_KSLICE * static_cast<uint32_t>(sizeof(DT_QUERY));
            pipe_.InitBuffer(bufL1, (aPart + bPart) * static_cast<uint32_t>(sizeof(DT_QUERY)));
            pipe_.InitBuffer(bufA, l0aPP * 2u);
            pipe_.InitBuffer(bufB, l0bPP * 2u);
            pipe_.InitBuffer(bufC, 16u * nBlk_ * static_cast<uint32_t>(sizeof(float)));

            CubeCtx ctx;
            const LocalTensor<DT_QUERY> l1 = bufL1.Get<DT_QUERY>();
            ctx.l1qa = l1;
            ctx.l1qr = l1[16u * rowA];
            ctx.l1ka = l1[aPart];
            ctx.l1kr = l1[aPart + nBlk_ * rowA];
            const LocalTensor<DT_QUERY> a0 = bufA.Get<DT_QUERY>();
            const LocalTensor<DT_QUERY> b0 = bufB.Get<DT_QUERY>();
            ctx.l0a[0] = a0;
            ctx.l0a[1] = a0[16u * sfa::CUBE_KSLICE];
            ctx.l0b[0] = b0;
            ctx.l0b[1] = b0[nBlk_ * sfa::CUBE_KSLICE];
            ctx.l0c = bufC.Get<float>();

            // ---- 参数模板：只有 nValue / dstNzC0Stride / repeatTimes / n / dstStride
            //      随片变化，其余整个启动期固定（各取一次赋值，省掉每片的标量写）----
            ctx.nzA.ndNum = 1u;
            ctx.nzA.dValue = rowA;
            ctx.nzA.srcDValue = rowA;          // Q 的行距就是 D_（BSND 下同 token 相邻头连续）
            ctx.nzA.dstNzC0Stride = 16u;       // A tile 只有 16 行
            ctx.nzA.dstNzNStride = 1u;
            ctx.nzA.srcNdMatrixStride = 0u;
            ctx.nzA.dstNzMatrixStride = 0u;
            ctx.nzAr = ctx.nzA;
            ctx.nzAr.dValue = rowR;
            ctx.nzAr.srcDValue = rowR;
            ctx.nzKa = ctx.nzA;                // B 侧：srcDValue 同 KV 的行距（KV_N=1 ⇒ =D_）
            ctx.nzKr = ctx.nzAr;

            ctx.ldA.startIndex = 0;
            ctx.ldA.srcStride = 1;             // 一次 repeat 一个 16×16 分形，源里分形连续
            ctx.ldA.dstGap = 0;
            ctx.ldA.ifTranspose = false;       // ← 关键：score 不需要转置（§15.70(f)）
            ctx.ldA.sid = 0;
            ctx.ldA.addrMode = 0;
            ctx.ldB = ctx.ldA;

            ctx.mp.m = 16u;                    // 头数向上取整到一个分形（nb_ ≤ 16）
            ctx.mp.cmatrixSource = false;      // C 在 L0C（不是 bias 表）

            ctx.fx.mSize = 16u;
            ctx.fx.srcStride = 16u;            // L0C 沿 m 的行距 = mSize
            ctx.fx.ndNum = 1u;
            ctx.fx.srcNdStride = 0u;
            ctx.fx.dstNdStride = 0u;
            ctx.fx.unitFlag = 0b11u;
            ctx.fx.reluEn = false;

            for (; cu_ < total; cu_ += cstep_) {
                CubeUnitBegin(ctx);
                while (CubeOneChunk(ctx)) { }
            }
            // ---- 收工：排空 credit 侧那 SFA_RING 轮余量 ----
            // 每消费一片【两颗】AIV 各交一张（合计 2C 张），生产侧每轮 wait 收两张、只在
            // "第 ≥RING 片"前等 ⇒ 恒定多出 2·min(RING, C) 张 = min(RING, C) 轮。不排空就把
            // 残值留给下一次启动（收件箱计数是【相对】还是【绝对】没有官方说法，赌它不如
            // 配平 —— §15.72(b) 的收支表）。
            const uint32_t d = (cp_ < sfa::SFA_RING) ? cp_ : sfa::SFA_RING;
            for (uint32_t i = 0u; i < d; ++i) {
                CrossCoreWaitFlag<2, PIPE_S>(sfa::CF_CRED);
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> qBuf_, oBuf_, kBuf_, krBuf_, sBuf_, pBuf_, mlBuf_, lseBuf_;
    TBuf<TPosition::VECCALC> kfBuf_, krfBuf_, rdBuf_;
    GlobalTensor<DT_QUERY> qGm_, kGm_, vGm_, qrGm_, krGm_, outGm_;
    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<int32_t> qLenGm_, kvLenGm_;
    GlobalTensor<float> maxGm_, sumGm_, traceGm_;

    uint32_t B_ = 0, S1_ = 0, S2_ = 0, N1_ = 0;
    int32_t D_ = 512, Dr_ = 64;   // 从 tiling 取，默认 512/64
    uint32_t sbs_ = 1, sparseCount_ = 2048, nb_ = 1, nBlk_ = 1, nHeadBlk_ = 1;
    // ml / LSE 第二半区的元素偏移（= nb_ 向上取整到一个 UB 块），见 Init 的说明
    uint32_t halfOff_ = 8;
    uint32_t stageMax_ = 1;      // P5b：一次整批装载/写回的行数上限（见 Init 的钳制）
    // P7：score 的 fold/rope 乘走"一次调用折 g 行"的批量形态（见 FoldRowsBatch/ComputeScores）。
    bool scBatch_ = false;
    uint32_t redW_ = 1;      // rdBuf_ 每条切片的宽度 = max(nb_, scGrp_, 2 块)，见 Init
    uint32_t scGrp_ = 1;     // 一组多少个 token = **整个 chunk**（n_blk），见 Init 的 P24 段
    float scale_ = 1.0f;
    uint32_t mode_ = 3;
    bool lseOn_ = false;
    bool qLenOn_ = false, kvLenOn_ = false;
    uint32_t qLenSize_ = 0, kvLenSize_ = 0;
    bool traceOn_ = false;
    uint32_t traceOff_ = 0;
    // 本核负责的【工作单元】跨步序列：单元 = (query 行, 头块, sparse 分片)，见 Init 的核间切分
    uint32_t unitBegin_ = 0, unitEnd_ = 0, unitStep_ = 1;
    // P11v2：sparse 列表切几段（1=不切分；2=按下标奇偶交错分两半，由 host 的门决定，见 Init）
    uint32_t ks_ = 1;
    // P21：一个 chunk 的 gather 段暂存（扫描前置用）。段数上限 = n_blk，host 已把 n_blk
    //      钳在 SFA_STAGE_MAX 内（见 tiling.h 与该文件 CalcBlocking），所以这里定长即可。
    //      放对象里而不是 UB：UB 的标量读实测比 GM 的标量读还贵 2.5 倍（§15.40 P20 二分）。
    int32_t stageBeg_[SFA_STAGE_MAX_CUBE];
    uint32_t stageLen_[SFA_STAGE_MAX_CUBE];

    // ---- P19-M1d：cube 路径的状态 ----
    // cubeOn_ 两侧都要有：AIV 用它把 ComputeScores 换成"读环"，AIC 用它决定要不要当生产者。
    bool cubeOn_ = false;
    uint32_t sub_ = 0;        // 本 AIV 在同组里的编号（0/1）⇒ 拿哪一半头、交哪条 credit
    uint32_t headBase_ = 0u;  // 本 AIV 在本单元里负责的第一个头（cube 形态；非 cube 恒 0）
    // AIC 侧【当前单元】的游标。一组只有一条片流（两颗 AIV 共用），所以全是标量。
    uint32_t cu_ = 0u;      // 下一个待领的单元号（= 本组两颗 AIV 的 unitBegin_ 口径）
    uint32_t cp_ = 0u;      // 累计已产片数：定环槽位 + credit 配平（跨单元【不清零】）
    uint32_t cbu_ = 0u;     // 累计 Mmad 单元数：只取奇偶 ⇒ L0A/L0B 乒乓槽轮转（跨片不清零）
    int64_t cthr_ = 0;      // 因果阈值（CalcThreshold）
    uint32_t ctok_ = 0u;    // sparse 下标游标（= ProcessToken 的 tokIdx）
    int64_t csegB_ = 0, csegE_ = 0;   // 跨 chunk 未吃完的那个块（= curBegin/curEnd）
    uint32_t chas_ = 0u;    // 上面那一段还有效
    uint32_t cskip_ = 0u;   // 本单元是 padding 行（不产任何片，与 ProcessToken 同口径）
    uint32_t cstep_ = 1u;   // 单元跨步 = 组数（与 AIV 的 unitStep_ 同一式）
    uint32_t cs1_ = 0u;     // s1Base：DT_QUERY 元素号；环槽的 float 号 = cs1_>>1
    uint64_t cidx_ = 0u;    // 本单元 sparse_indices 的起始下标
    int64_t crb_ = 0;       // KV 的 batch 行基址（= b*S2_）

    // 回程环的 fp32 别名（环就住在"本单元自己的输出行"里，见 CubeGate 的预算）
    GlobalTensor<float> ringOutGm_;
};

// kernel 入口：与原骨架写法一致 —— 【模板 + __global__ __aicore__，不要 extern "C"】
// ⚠️ 踩坑记录：加 extern "C" 会把它变成非模板函数，导致编译器生成的
//    `sparse_flash_attention<TEMPLATE_PARAMS>(...)` 报
//    "'..._tilingkey' does not name a template but is followed by template arguments"
template <typename DT_QUERY>
__global__ __aicore__ void sparse_flash_attention(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparseIndices,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKV,
    GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR attentionOut,
    GM_ADDR softmaxMaxOut, GM_ADDR softmaxSumOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(SparseFlashAttentionTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionTilingData, tiling_data, tiling);
    KernelSparseFlashAttention<DT_QUERY> op;
    op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,
            queryRope, keyRope, attentionOut, softmaxMaxOut, softmaxSumOut,
            tiling_data, nullptr);
    op.Process();
}

template __aicore__ void sparse_flash_attention<half>(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template __aicore__ void sparse_flash_attention<float>(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
