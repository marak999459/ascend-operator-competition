// 自动生成：probes/p106_modeldiff.py —— 不要手改，改模型区请改 op_host 后重跑
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
constexpr uint32_t SFA_STAGE_MAX = 48;
constexpr uint32_t HQ_DIM = 512;      // Q_D / KV_D
constexpr uint32_t ROPE_DIM = 64;     // Dr
constexpr uint32_t QD_ROPE = HQ_DIM + ROPE_DIM;   // 576
constexpr uint32_t OFFICIAL_SPARSE_COUNT = 2048;  // 官方固定值
// 单核 UB 容量从平台查询后按 95% 留安全余量，不写死 ——
// 写死会在换芯片/换 CANN 版本时静默算出错误分块。
constexpr uint64_t UB_DEFAULT_BYTE = 196608ULL;   // 910B3 实测可用
constexpr uint64_t UB_SAFE_PCT     = 95ULL;

// 头分块候选（从大到小取第一个能装下的）
constexpr uint32_t NB_CAND[] = {32, 16, 8, 4, 2, 1};
constexpr uint32_t NBLK_CAND[] = {128, 64, 48, 40, 32, 16, 8, 4, 2, 1};
// P105：kv_shard 的候选深度（原来只有 1/2 两档）。上限 10 是两件事夹出来的：
//   ① 门 ① 要 units0*ks <= 核数(40)，平台那 6 个 decode 形点 units0 = 4~32 ⇒ 有意义的
//      深度本来就是 2~10；
//   ② 每深一档就多付一轮跨核折叠（实测 1.7~11.8 µs/轮，见下面的 FOLD_TAX），而每单元的
//      chunk 数按 1/ks 缩 —— 摊到不足一个 chunk 之后，加深只在付 barrier 不再省活。
// 遍历按【由浅到深】+ 只在严格更优时替换 ⇒ 并列时保留浅的（少一轮 fp16 往返 = 精度余量）。
constexpr uint32_t KS_CAND[] = {2, 3, 4, 5, 6, 8, 10};
// 代价模型的货币换算：一条"等效向量调用"值多少纳秒。用【一个】实测点反解，不是推导 ——
// pm41（rows=4、Q_N=4、sbs=1、count=4096，host 选 nb=1/ks=2）实测 313.8 µs，同档
// cost = UnitCalls(1,48)=169 × chunks(⌈2048/48⌉)=43 ≈ 7267 ⇒ 43 ns，取整 40。
// ⚠️ 只有一个标定点，所以这个数**只**用来把"绝对时间的折叠税"折算进模型，不参与各项
//    之间的相对排序（模型内部的排序与 COST_NS 无关）。税被低估/高估一倍的后果由 P105
//    的 nb×ks 实测网格兜（LOG#15.86 的口径）。
constexpr uint64_t COST_NS = 40ULL;
// 一轮跨核折叠的墙钟成本。消融臂（把阶段 B 整段再走一遍、输出必错只读时间）在 6 个
// 用例上实测 +1.7 ~ +11.8 µs，且与 payload 无关（rows=16 那份 4 倍 payload 只付 1.7 µs）
// ⇒ 延迟主导，按【每轮常数】记，取最不利端 10 µs。
constexpr uint64_t FOLD_TAX = 10000ULL / COST_NS;   // = 250 等效调用
// UB 一个块 = 256 bit = 32 B；host 预算与 kernel InitBuffer 都按它向上取整。
constexpr uint64_t UB_BLK = 32ULL;
// ⚠️ nb 原先有 8 的下限（NB_MIN）：kernel 把 LSE 的 sum 半区放在 UB 偏移 nb 个 float 处，
// nb<8 时半区源地址不是 256bit 对齐 -> aivec ADDR_MISALIGN（当年实测 r1~r7 全挂）。
// P10 按当年注释里指的路把 kernel 布局改成"半区各占整块"（偏移 = nb 向上取整到一个块，
// 见 kernel 的 halfOff_），下限随之解除 —— 头块要能摊到更多核上就必须要 nb 能小。
constexpr uint32_t NB_MIN = 1U;
constexpr uint64_t UB_BLK_F32 = UB_BLK / 4ULL;        // 一个 UB 块 = 8 个 fp32
static inline uint64_t HalfElems(uint32_t nb)
{
    return (static_cast<uint64_t>(nb) + UB_BLK_F32 - 1ULL) / UB_BLK_F32 * UB_BLK_F32;
}
// ⚠️ n_blk 的【下限】同样是"一个 UB 块里的 fp32 个数"，且必须是它的整数倍。
// kernel 的分数矩阵按 [nb][n_blk] 排布，行间距 = n_blk 个 float；P3a 之后每行由
// `WholeReduceSum` + `Add` 以 32B 块为单位写入，n_blk 不是 8 的倍数时行起点就落在
// 块中间 -> 与 nb<8 同一类真机 ADDR_MISALIGN。
// P24 之前这里写的是 SFA_SC_GRP(=16)，因为"一组 token 要装得下"；P24 起组宽就是
// n_blk，那条理由自动消失，但下限仍取 16（= 2 个块）：候选空间与 P21 保持一致，
// 别让一次结构改动同时改两条寻优的腿（§15.45 的对照口径）。
constexpr uint32_t NBLK_MIN = 16U;

// elemSize = DT_QUERY 的字节数（fp16=2 / fp32=4）。K/V/Krope 在 UB 里就是 DT_QUERY，
// 写死 2 字节会让 fp32 模板实例的 InitBuffer 总量越过物理 UB（code3.md §13.14：
// nb=32/nBlk=16 时 fp32 实际需要 213696 B > 196608 B）。
// 每项单独向上对齐 32B —— kernel 的 InitBuffer 也是按块分配的，两边口径必须一致。
static inline uint64_t UbAlignBuf(uint64_t len)
{
    return (len + 31ULL) / 32ULL * 32ULL;
}

// ⭐ kernel 里 `scGrp_` 的镜像（op_kernel/…cpp 的 Init 段）：P24 起组宽 = **整个 chunk**
// ⇒ 没有独立常数了，直接用 n_blk；归约工作区宽度 = max(nb, n_blk, 2 个 UB 块)。
constexpr uint64_t RED_MIN_W = 2ULL * UB_BLK_F32;   // kernel 的 mergeMin

static inline uint64_t RedWOf(uint32_t nb, uint64_t nBlk)
{
    const uint64_t g = (nBlk > nb) ? nBlk : nb;
    return (g > RED_MIN_W) ? g : RED_MIN_W;
}

static inline uint64_t CalcUbNeed(uint32_t nb, uint64_t nBlk, uint32_t qD, uint32_t dr,
                                  uint32_t elemSize)
{
    const uint64_t e = elemSize;
    const uint64_t n = nb;
    const uint64_t k = nBlk;
    // 与 kernel Init 的 InitBuffer 逐项对应，漏一项就会真机越界
    // P32：kernel 不再有 vBuf_ —— V 在 ComputeScores 之后才由 MTE2 搬进 kBuf_ 自己那块
    //      （同 dtype、同 size、同对齐口径），所以这里少一项 align(n_blk*D*e)。
    //      省下的 40,960 B(n_blk=40) 正好让 n_blk=48 在 nb=1/2/4 上第一次过门。
    return UbAlignBuf(n * (qD + dr) * 4ULL)      // qBuf_ : Q+Qrope 恒 fp32
         + UbAlignBuf(n * qD * 4ULL)             // oBuf_ : 累加器 fp32
         + UbAlignBuf(k * qD * e)                // kBuf_（V 复用它，见上）
         + UbAlignBuf(k * dr * e)                // krBuf_
         + UbAlignBuf(n * k * 4ULL)              // sBuf_
         + UbAlignBuf(n * k * 4ULL)              // pBuf_
         + UbAlignBuf(3ULL * HalfElems(nb) * 4ULL)  // mlBuf_ : m/l/mnew，半区各占整块（kernel halfOff_）
         + 2ULL * UbAlignBuf(HalfElems(nb) * 4ULL)  // lseBuf_ : max/sum 两个半区各自对齐整块
         // ---- P24 的 scratch：一组 = **整个 chunk** 个 token 的 fp32 展开 + 三条归约工作区，
         //      宽度 = RedWOf(nb, n_blk)（kernel 的 redW_ 口径，两边必须一致）。
         //      部分积就地写在 kf/krf 上 ⇒ pfBuf_/prfBuf_ 根本不分配。nb==1 档的字节数
         //      与 P21 相同（那时 kf 已经是 32 行）；nb>1 档回收 pf/prf 两份，多出的是
         //      kf 从 16 行变 n_blk 行。----
         + UbAlignBuf(k * qD * 4ULL)             // kfBuf_
         + UbAlignBuf(k * dr * 4ULL)             // krfBuf_
         + 3ULL * UbAlignBuf(RedWOf(nb, k) * 4ULL);
}

// 反算分块。
//   ⚠️ 不能再要求 nBlk >= sparseBlockSize。
//   sparseBlockSize 合法范围是 [1,128]，而 nBlk=128 时仅 kBuf+vBuf 就要 2*128*512*2
//   = 262144 B > 整个 UB(196608 B) —— 即"整块读"对 SBS=128 在硬件上不可能成立。
//   kernel 侧支持"一个 sparse block 跨多个 chunk"（分段 online softmax），
//   所以"一次 flush 覆盖一整块"只是省 flush 的偏好，不是正确性前提：
//   第一轮按此偏好选，装不下时第二轮放开。
//
//   ⭐ P10：nb 不再由"UB 能装多大就多大"决定，而是按【代价模型】选。
//   kernel 的工作单元是 (query 行, 头块)，单元数 = rows * ceil(Q_N / nb)，
//   墙钟时间 ∝ ceil(单元数 / coreNum) × 每单元的向量调用条数。
//   §5.8.3 由探针反推出比赛平台 6 个点只有 4/8/4/16/4/32 行 —— 单元只到"行"这一级
//   时最多 4~32 个 AIV 在干活，另外的核整个算子期间空转。把头块也摊出去就能填满。
//   但切分的收益有上限：nb 变小后"每组 token 只做一次的共享搬运/展开"（K 的 Cast、
//   V 的加宽）被 ceil(Q_N/nb) 个单元各做一遍，每单元调用数随之上升。实测（§15.14）
//   第一版规则"仍能填满核的最大 nb，填不满就降到 1"在 rows=32 的点上反而慢了 17%
//   （nb=1 把 32 行切成 128 单元 = 4 波，每单元贵 2.2 倍，而 32 行本来 1 波就跑完）。
//   所以这里改成对候选 (nb, n_blk) 直接比 ceil(waves) × UnitCalls()，取最小代价；
//   并列时取较大的 nb（单元少 = 固定开销少、也更省 UB 之外的搬运）。
//   ⚠️ P38 更正：上面那行式子当时【少乘了一维】—— UnitCalls 是"一个 chunk"的代价，waves 是
//   "整个单元"的波数，中间还要乘"这个单元扫几个 chunk"。小形状上这一维近似常数所以没出事，
//   大形状上它把 nb 一路推过头（w4 慢 11 %、big1 慢 5~10 %），补法见下面的 UnitCost。
//   返回 false 表示连最小分块都装不下 -> 调用方必须安全降级，绝不返回 tiling 错误。
// 一个工作单元（nb 个头 × n_blk 个 token）处理一个 KV chunk 的【代价】，单位是
// "等效向量调用条数"。只用于在候选之间比大小，绝对值没有意义（2201 上实测一条向量
// 指令 ≈ 28.98 + 1.123 × rep 个周期，见 §15.13），所以这里把 rep 摊成常数、只保留随
// (nb, n_blk) 变化的项。P24 之后按"每头一次 + 每 token 一次"两段重列：
//   组宽 = 整个 chunk ⇒ nG 恒为 1，P21 的"每组一次"项全部并进"每头一次"。
//   K 侧：每头 2 条加宽（K、K-rope）+ 每头 20 条
//                    = 8 条广播乘 + 1 条 rope 乘 + 7 条折行 + 2 条归约 + 1 加 + 1 乘 scale
//   Q 侧：softmax 的 max/exp/alpha/重缩放 ≈ 4·nb + 2，V 的加宽 1 条（P24 起整个
//         chunk 一批），PV 每个 (头, token) 一条 Axpy = n_blk·nb ← 这项**不能**摊薄
//   搬运：+ GATHER_DIV 见下面 GatherCalls 的注释（P13 那一版漏了它，翻车三次）
static inline uint64_t GatherCalls(uint32_t nb, uint64_t nBlk, uint32_t qD, uint32_t dr,
                                   uint32_t elemSize)
{
    // 每单元每 chunk 要把 K+V+Krope 从 GM 搬进 UB：nb 越小单元越多，同一段 KV 就被
    // ceil(Q_N/nb) 个核各搬一遍 ⇒ 搬运代价 ∝ n_blk·(2·qD+dr)·elemSize / nb。
    // P13 第一版没有这一项，host 于是把 p4/p6/big1 全推到 nb=1，实测
    // 1.196× / 1.394× / 1.247× **变慢**（§15.16）。三个点各自反解系数得
    // 1.42e-3 / 1.52e-3 / 1.68e-3（跨度 ±10%）⇒ 取 1/660 = 1.515e-3，三档偏差 <7%。
    // ⚠️ 这是标定不是推导：换 D 或 SBS 量级要重新用实测点验一遍。
    constexpr uint64_t GATHER_DIV = 660ULL;
    return nBlk * (2ULL * qD + dr) * elemSize / (2ULL * nb * GATHER_DIV);   // e/2 ⇒ fp16 记 1
}

static inline uint64_t UnitCalls(uint32_t nb, uint64_t nBlk, uint32_t qD, uint32_t dr,
                                 uint32_t elemSize)
{
    const uint64_t n = nb;
    const uint64_t k = nBlk;
    // P24：组宽 = 整个 chunk ⇒ "每组一次"的那 20·nb 条每 chunk 只剩一轮。
    // 但 kf/krf 的加宽从"每组 2 条"变成"每头 2 条、每条 k 行"，**数据量**第一次
    // 随 nb 增长 ⇒ 这一项不能再把 rep 摊成常数（P21 那时加宽量与 nb 无关，摊掉无损）。
    // 一条 content 加宽 = k·qD 个 fp32 = k·qD/64 个 repeat ⇒ 1.123·k·qD/64 周期；一条
    // 调用按 ≈36 周期（固定 28.98 + 平均 6 rep）换算成"等效调用条数"。
    const uint64_t widenVol = n * k * (qD + dr) * 1123ULL / (64ULL * 36000ULL);
    return n * 22ULL + widenVol + (3ULL + 4ULL * n + k * n)
         + GatherCalls(nb, k, qD, dr, elemSize);
}

// P38：一个工作单元扫完自己那段 sparse 列表要发起多少次"整 chunk"的处理 =
//   ceil(本单元 token 数 / n_blk)，乘到代价里。
// ⚠️ 这一维在此前的模型里【整个都没有】：`cost = waves × UnitCalls`，可 `UnitCalls` 的口径
//   是"处理【一个 chunk】的调用数"，`waves` 却是【整个单元】的波数 ⇒ 中间少乘了"几个 chunk"。
//   少掉它有两个后果，都被 P37 的三维实测网格抓到（code3.md §15.55）：
//     ① **k 轴反指**：模型只剩"k 越大 → 每 chunk 调用越多 ⇒ 越贵"，表达不出"k 越大 → flush
//        越少 → 越便宜"，与 §15.51(2) 那 24/24 格同臂"k 越大越快"直接反号。当时靠"每个 nb
//        臂取最大可行 k"的贪心把符号错误常数化掉了，代价是①的兄弟②没人对冲。
//     ② **nb 轴被系统性偏袒**：nb 变大 → 最大可行 k 变小 → 每 chunk 显得越便宜 → cost 越小，
//        于是模型一路上推 nb。w4（rows=128、Q_N=8、4096 token/行）模型选 nb=8/k=40，实测它
//        是第 3 名，比全场最快的 nb=4/k=48 慢 **11.0 %**；big1（同 rows/Q_N、256 token/行）
//        从 P29 起就一直停在 nb=8，而 §15.51 的网格早就量到它的 nb=4 快 5 % 以上。
//   token 数取 `min(sparse_count*sbs, KV_S)`：kernel 会跳过 -1 项，host 看不到"每行实际
//   有几个有效块"（big1 的 2048 项里只有 256 个有效），所以这是一个【上界】。它不影响
//   选档：误差只通过 ceil 的取整进入，而各 nb 臂吃的是同一个上界。
static inline uint64_t UnitCost(uint32_t nb, uint64_t nBlk, uint32_t qD, uint32_t dr,
                                uint32_t elemSize, uint64_t toks)
{
    uint64_t chunks = (toks + nBlk - 1ULL) / nBlk;
    if (chunks == 0ULL) { chunks = 1ULL; }
    return UnitCalls(nb, nBlk, qD, dr, elemSize) * chunks;
}

// P16/P105：kv_shard（把 sparse 列表切成 ks 半分给 ks 个核）这一档【能不能开】。
// 三条门与 kernel 的 Init 同一口径，任何一条不满足就只能 ks=1：
//   ① units0*ks <= coreNum —— 切完仍在【一波】内。多出一波等于白切：U 个单元 / C 核是
//      ceil(U/C) 波 × T，切成 k 份是 ceil(Uk/C) 波 × T/k，只有 Uk<=C 才真的降时间。
//      ⚠️ 这条同时就是 kernel 那句 `coreNum == ks * total0` 的镜像：kernel 要求"一核一单元"
//      是因为分片的部分和留在 UB 里，同核再领一个单元就被覆盖。
//   ② LSE 元素数 >= 8 且是 8 的整数倍 —— 归并走「整块 8 个 float」的对齐窗口读累加器
//      的 (m,l)，窗口起点向下取整到 32B、尾部不越出张量，这两件事只对 8 的整数倍同时成立。
//   ③ sparse_count >= ks*n_blk —— 每片至少摊得下一个完整 chunk。
static inline bool KsAllowed(uint64_t units0, uint32_t coreNum, uint64_t lseElems,
                             uint64_t sparseCount, uint64_t nBlk, uint64_t ks)
{
    return units0 >= 1ULL && units0 * ks <= static_cast<uint64_t>(coreNum) &&
           lseElems >= 8ULL && (lseElems % 8ULL) == 0ULL && sparseCount >= ks * nBlk;
}


static bool CalcBlocking(uint32_t sparseBlockSize, uint64_t ubSafe, uint32_t qD, uint32_t dr,
                         uint32_t qN, uint32_t elemSize, uint32_t coreNum, uint32_t rows,
                         uint64_t sparseCount, uint64_t lseElems, uint64_t kvS,
                         uint32_t &nbOut, uint32_t &nBlkOut, uint32_t &ksOut, uint64_t &costOut)
{
    constexpr size_t NB_N = sizeof(NB_CAND) / sizeof(NB_CAND[0]);
    constexpr size_t NBLK_N = sizeof(NBLK_CAND) / sizeof(NBLK_CAND[0]);
    // P38：每行最多扫这么多 token —— 上界，口径与 UnitCost 的注释同源（host 看不到
    //      每行实际有效几个块，big1 那类"sparse 表 2048 项、有效 256 项"的用例会高估）。
    uint64_t toks = sparseCount * static_cast<uint64_t>(sparseBlockSize);
    if (kvS < toks) { toks = kvS; }
    // nb 超过头数没有意义：kernel 的 nbCur = min(Q_N - n0, nb) 会把它夹回来，
    // 多出来的 qBuf_/oBuf_/sBuf_ 全是死缓冲。
    const uint64_t nbCap = (static_cast<uint64_t>(qN) < NB_MIN) ? NB_MIN : qN;

    uint64_t bestCost = 0;
    uint32_t bestNb = 0, bestBlk = 0, bestKs = 1U;
    // NB_CAND 从大到小遍历，且只在【严格更优】时替换 -> 并列时代价模型自动保留
    // 较大的 nb（单元少 = 每单元固定开销少，也更省 UB）。
    for (size_t i = 0; i < NB_N; ++i) {
        const uint32_t nb = NB_CAND[i];
        if (static_cast<uint64_t>(nb) > nbCap) { continue; }
        const uint64_t units = static_cast<uint64_t>(rows) * ((qN + nb - 1ULL) / nb);
        const uint64_t waves = (coreNum == 0) ? units : (units + coreNum - 1ULL) / coreNum;
        // 同一个 nb 下 UnitCalls 对 n_blk 单调增，所以每个 nb 只需它的最优 n_blk：
        // 第一轮保留"一次 flush 覆盖一整块"的偏好，装不下时第二轮放开。
        for (uint32_t pass = 0; pass < 2; ++pass) {
            bool found = false;
            for (size_t j = 0; j < NBLK_N; ++j) {
                const uint32_t nBlk = NBLK_CAND[j];
                if (nBlk < NBLK_MIN) { continue; }
                // P21：kernel 把"扫满一个 chunk"的段登记进定长暂存（尺寸 SFA_STAGE_MAX），
                //      一段至少 1 个 token ⇒ n_blk 不能超过它。P32 之前这道钳对选档是空操作
                //      （D=512 下 64/128 两档本来就过不了下面的 UB 预算）；P32 省下 vBuf_ 后
                //      48 变成了"预算装得下、暂存数组装不下"的第一档，所以这道钳现在是
                //      binding 的（口径与 kernel 的 stageBeg_ 同源，见 tiling.h）。
                if (nBlk > SFA_STAGE_MAX) { continue; }
                if (pass == 0 && nb * nBlk < sparseBlockSize) { continue; }
                if (CalcUbNeed(nb, nBlk, qD, dr, elemSize) > ubSafe) { continue; }
                const uint64_t calls = UnitCalls(nb, nBlk, qD, dr, elemSize);
                uint64_t cost = waves * UnitCost(nb, nBlk, qD, dr, elemSize, toks);
                uint32_t ks = 1U;
                // P16/P105：kv_shard 是【一等候选】，不是选完档之后再判的附加开关。
                //      切完之后每单元的 token 数按 1/ks 缩 ⇒ 波数按 ks*units 重算，再按
                //      【每多一轮折叠】付一笔 FOLD_TAX。
                //      ⚠️ 少了这一档，模型会挑"不切时最便宜"的那一档 nb，而那一档往往
                //      是切分后最贵的（p4：模型选 nb=2/ks=1 = 0.4905，实测 nb=4/ks=2 =
                //      0.4252 ⇒ 慢 15 %，见 §15.34）。
                // P38：省的是【flush 次数】不是"调用数"—— 一半 token 少一半 chunk，而每 chunk
                //      的调用数不变。原式把 calls 整个折半，连"每头一次"那批固定项也折了半，
                //      是同一个"漏乘 chunk 维度"带出来的近似；补上维度后按 chunk 折半才是对的形式。
                // P105：税的记法从"×108/100"换成"每轮绝对值"—— 8 % 是把 ks=2 的一个点
                //      当成比例，而 ks 深到 10 时分子是【轮数】不是 token 数：一轮 barrier
                //      该核群只有一批核干活，时间与 payload 无关（消融实测 1.7~11.8 µs），
                //      比例式会把它算成随 chunk 数缩放，深档必然被低估。
                if (KsAllowed(units, coreNum, lseElems, sparseCount, nBlk, 2ULL)) {
                    constexpr size_t KS_N = sizeof(KS_CAND) / sizeof(KS_CAND[0]);
                    for (size_t m = 0; m < KS_N; ++m) {
                        const uint64_t k = KS_CAND[m];
                        if (!KsAllowed(units, coreNum, lseElems, sparseCount, nBlk, k)) { continue; }
                        const uint64_t unitsK = units * k;
                        const uint64_t wavesK = (coreNum == 0)
                            ? unitsK : (unitsK + coreNum - 1ULL) / coreNum;
                        // 每单元的 token 数按 1/k 缩（P38 那条"省的是 flush 次数"在这里成立）
                        uint64_t chunksK = (toks / k + nBlk - 1ULL) / nBlk;
                        if (chunksK == 0ULL) { chunksK = 1ULL; }
                        const uint64_t costK = wavesK * calls * chunksK
                                             + (k - 1ULL) * FOLD_TAX;
                        if (costK < cost) { cost = costK; ks = static_cast<uint32_t>(k); }
                    }
                }
                // 迟滞：只有【明显更优】（< 0.95×）才从较大的 nb 改档，留一点余量给
                // 未建模的每单元方差（并列时保留大 nb = 单元少 = 同一份 K 被更少的核重复加宽）。
                // ⚠️ 这一档从 0.85 抬到 0.95：0.85 是 §15.14 时代加的，那时模型还**缺**搬运项，
                // 波数 ceil 的误差靠迟滞兜底。P24 把模型重列成"每头 + 每 token"两段后，
                // 用真机 5 用例 × 4 档 nb 的实测网格复验（probes/p25_model_check.py）：
                // **20 组相邻档比较 0 次顺序反转** ⇒ 模型自己已经够准，迟滞不再是误差兜底，
                // 而 0.85 会实打实吃掉 big1 的 5 %（模型判 nb=4 比 nb=8 便宜 9 %，
                // 但 0.91 > 0.85 被压回 nb=8，实测 0.7194 vs 0.6846 ms）。取 0.95 后
                // 五个本地点的选档全部落在实测最优点上（后悔和 1.0001 vs 0.85 的 1.0103）。
                if (bestCost == 0 || cost * 20ULL < bestCost * 19ULL) {
                    bestCost = cost;
                    bestNb = nb;
                    bestBlk = nBlk;
                    bestKs = ks;
                }
                found = true;
                break;
            }
            if (found) { break; }
        }
    }
    if (bestCost == 0) { return false; }
    nbOut = bestNb;
    nBlkOut = bestBlk;
    ksOut = bestKs;
    costOut = bestCost;
    return true;
}


static bool CalcBlockingP38(uint32_t sparseBlockSize, uint64_t ubSafe, uint32_t qD, uint32_t dr,
                         uint32_t qN, uint32_t elemSize, uint32_t coreNum, uint32_t rows,
                         uint64_t sparseCount, uint64_t lseElems, uint64_t kvS,
                         uint32_t &nbOut, uint32_t &nBlkOut, uint32_t &ksOut, uint64_t &costOut)
{
    constexpr size_t NB_N = sizeof(NB_CAND) / sizeof(NB_CAND[0]);
    constexpr size_t NBLK_N = sizeof(NBLK_CAND) / sizeof(NBLK_CAND[0]);
    // P38：每行最多扫这么多 token —— 上界，口径与 UnitCost 的注释同源（host 看不到
    //      每行实际有效几个块，big1 那类"sparse 表 2048 项、有效 256 项"的用例会高估）。
    uint64_t toks = sparseCount * static_cast<uint64_t>(sparseBlockSize);
    if (kvS < toks) { toks = kvS; }
    // nb 超过头数没有意义：kernel 的 nbCur = min(Q_N - n0, nb) 会把它夹回来，
    // 多出来的 qBuf_/oBuf_/sBuf_ 全是死缓冲。
    const uint64_t nbCap = (static_cast<uint64_t>(qN) < NB_MIN) ? NB_MIN : qN;

    uint64_t bestCost = 0;
    uint32_t bestNb = 0, bestBlk = 0, bestKs = 1U;
    // NB_CAND 从大到小遍历，且只在【严格更优】时替换 -> 并列时代价模型自动保留
    // 较大的 nb（单元少 = 每单元固定开销少，也更省 UB）。
    for (size_t i = 0; i < NB_N; ++i) {
        const uint32_t nb = NB_CAND[i];
        if (static_cast<uint64_t>(nb) > nbCap) { continue; }
        const uint64_t units = static_cast<uint64_t>(rows) * ((qN + nb - 1ULL) / nb);
        const uint64_t waves = (coreNum == 0) ? units : (units + coreNum - 1ULL) / coreNum;
        // 同一个 nb 下 UnitCalls 对 n_blk 单调增，所以每个 nb 只需它的最优 n_blk：
        // 第一轮保留"一次 flush 覆盖一整块"的偏好，装不下时第二轮放开。
        for (uint32_t pass = 0; pass < 2; ++pass) {
            bool found = false;
            for (size_t j = 0; j < NBLK_N; ++j) {
                const uint32_t nBlk = NBLK_CAND[j];
                if (nBlk < NBLK_MIN) { continue; }
                // P21：kernel 把"扫满一个 chunk"的段登记进定长暂存（尺寸 SFA_STAGE_MAX），
                //      一段至少 1 个 token ⇒ n_blk 不能超过它。P32 之前这道钳对选档是空操作
                //      （D=512 下 64/128 两档本来就过不了下面的 UB 预算）；P32 省下 vBuf_ 后
                //      48 变成了"预算装得下、暂存数组装不下"的第一档，所以这道钳现在是
                //      binding 的（口径与 kernel 的 stageBeg_ 同源，见 tiling.h）。
                if (nBlk > SFA_STAGE_MAX) { continue; }
                if (pass == 0 && nb * nBlk < sparseBlockSize) { continue; }
                if (CalcUbNeed(nb, nBlk, qD, dr, elemSize) > ubSafe) { continue; }
                const uint64_t calls = UnitCalls(nb, nBlk, qD, dr, elemSize);
                uint64_t cost = waves * UnitCost(nb, nBlk, qD, dr, elemSize, toks);
                uint32_t ks = 1U;
                // P16/P105：kv_shard 是【一等候选】，不是选完档之后再判的附加开关。
                //      切完之后每单元的 token 数按 1/ks 缩 ⇒ 波数按 ks*units 重算，再按
                //      【每多一轮折叠】付一笔 FOLD_TAX。
                //      ⚠️ 少了这一档，模型会挑"不切时最便宜"的那一档 nb，而那一档往往
                //      是切分后最贵的（p4：模型选 nb=2/ks=1 = 0.4905，实测 nb=4/ks=2 =
                //      0.4252 ⇒ 慢 15 %，见 §15.34）。
                // P38：省的是【flush 次数】不是"调用数"—— 一半 token 少一半 chunk，而每 chunk
                //      的调用数不变。原式把 calls 整个折半，连"每头一次"那批固定项也折了半，
                //      是同一个"漏乘 chunk 维度"带出来的近似；补上维度后按 chunk 折半才是对的形式。
                // P105：税的记法从"×108/100"换成"每轮绝对值"—— 8 % 是把 ks=2 的一个点
                //      当成比例，而 ks 深到 10 时分子是【轮数】不是 token 数：一轮 barrier
                //      该核群只有一批核干活，时间与 payload 无关（消融实测 1.7~11.8 µs），
                //      比例式会把它算成随 chunk 数缩放，深档必然被低估。
                if (KsAllowed(units, coreNum, lseElems, sparseCount, nBlk, 2ULL)) {
                    const uint64_t units2 = units * 2ULL;
                    const uint64_t waves2 = (coreNum == 0)
                        ? units2 : (units2 + coreNum - 1ULL) / coreNum;
                    uint64_t chunks2 = (toks / 2ULL + nBlk - 1ULL) / nBlk;
                    if (chunks2 == 0ULL) { chunks2 = 1ULL; }
                    const uint64_t cost2 = ((waves2 * calls * chunks2) * 108ULL) / 100ULL + 1ULL;
                    if (cost2 < cost) { cost = cost2; ks = 2U; }
                }
                // 迟滞：只有【明显更优】（< 0.95×）才从较大的 nb 改档，留一点余量给
                // 未建模的每单元方差（并列时保留大 nb = 单元少 = 同一份 K 被更少的核重复加宽）。
                // ⚠️ 这一档从 0.85 抬到 0.95：0.85 是 §15.14 时代加的，那时模型还**缺**搬运项，
                // 波数 ceil 的误差靠迟滞兜底。P24 把模型重列成"每头 + 每 token"两段后，
                // 用真机 5 用例 × 4 档 nb 的实测网格复验（probes/p25_model_check.py）：
                // **20 组相邻档比较 0 次顺序反转** ⇒ 模型自己已经够准，迟滞不再是误差兜底，
                // 而 0.85 会实打实吃掉 big1 的 5 %（模型判 nb=4 比 nb=8 便宜 9 %，
                // 但 0.91 > 0.85 被压回 nb=8，实测 0.7194 vs 0.6846 ms）。取 0.95 后
                // 五个本地点的选档全部落在实测最优点上（后悔和 1.0001 vs 0.85 的 1.0103）。
                if (bestCost == 0 || cost * 20ULL < bestCost * 19ULL) {
                    bestCost = cost;
                    bestNb = nb;
                    bestBlk = nBlk;
                    bestKs = ks;
                }
                found = true;
                break;
            }
            if (found) { break; }
        }
    }
    if (bestCost == 0) { return false; }
    nbOut = bestNb;
    nBlkOut = bestBlk;
    ksOut = bestKs;
    costOut = bestCost;
    return true;
}


struct Row { uint32_t rows, qN, sbs; uint64_t count, kvS; };
static void emit(const Row &r, uint32_t elemSize, bool quietSame)
{
    uint64_t lse = (uint64_t)r.rows * r.qN;
    uint32_t nb, k, ks, nb2, k2, ks2;
    uint64_t c105 = 0, c38 = 0;
    bool ok  = CalcBlocking(r.sbs, 186568, 512, 64, r.qN, elemSize, 40, r.rows, r.count, lse, r.kvS, nb, k, ks, c105);
    bool ok2 = CalcBlockingP38(r.sbs, 186568, 512, 64, r.qN, elemSize, 40, r.rows, r.count, lse, r.kvS, nb2, k2, ks2, c38);
    if (!ok || !ok2) { printf("FAIL  "); return; }
    bool same = (nb == nb2 && k == k2 && ks == ks2);
    if (quietSame && same) return;
    printf("rows=%-4u qN=%-2u sbs=%-4u count=%-6llu %-4s | P38 nb=%-2u k=%-3u ks=%-2u  ->  P105 nb=%-2u k=%-3u ks=%-2u"
           "  模型预测加速 %5.2fx %s\n",
           r.rows, r.qN, r.sbs, (unsigned long long)r.count, elemSize == 2 ? "fp16" : "fp32",
           nb2, k2, ks2, nb, k, ks, (double)c38 / (double)c105, same ? "" : "  <<< 不同档");
}
int main(int argc, char **argv)
{
    const bool big = (argc > 1 && strcmp(argv[1], "big") == 0);
    const std::vector<uint32_t> ROWS = {2,4,8,16,32,64,128,256,512,1024,4096};
    const std::vector<uint32_t> QN   = {1,2,4,8,16};
    const std::vector<uint64_t> CNT  = big ? std::vector<uint64_t>{2048,8192,32768,65536,262144}
                                           : std::vector<uint64_t>{256,1024,2048,4096,16384,65536};
    const std::vector<uint32_t> SBS  = big ? std::vector<uint32_t>{1} : std::vector<uint32_t>{1,16,64,128};
    for (uint32_t rows : ROWS) for (uint32_t qn : QN) for (uint64_t cnt : CNT) for (uint32_t sbs : SBS) {
        Row r{rows, qn, sbs, cnt, 1ULL << 30};
        emit(r, 2, big);
    }
    return 0;
}
