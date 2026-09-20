// mhc_expand 仿真机对拍 harness（本地组装，tar 管道推本地/云端仿真机）
// 与提交 kernel 完全同源：直接 #include "op_kernel/mhc_expand.cpp"
// tiling 决策镜像 op_host TilingFunc 的关键路径（行切/流切/元素切 + UB 预算）
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <sys/mman.h>
#include <vector>
#include <map>

#include "kernel_operator.h"
#include "tikicpulib.h"
using namespace AscendC;

#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(T, name, ptr) \
    const T& name = *reinterpret_cast<const T*>(ptr)
#endif

// 镜像 op_host TilingFunc：UB 预算 = ub_size/4（真机 910B 为 256KB/4=64KB）；
// ub48 模式模拟 host ub_size==0 兜底 192KB/4=48KB 的路径
static uint64_t g_ub_budget = 64 * 1024;

#include "mhc_expand_tiling.h"
#include "mhc_expand.cpp"

// ===== host tiling 镜像（与 op_host/mhc_expand.cpp 决策一致） =====
static const uint32_t SPLIT_ROW = 0;
static const uint32_t SPLIT_ROW_STREAM = 1;
static const uint32_t SPLIT_ELEMENT = 2;

struct TilingPlan {
    MhcExpandTilingData t;
    uint32_t total_tasks;
};

static TilingPlan plan_tiling(uint32_t S, uint32_t D, uint32_t m, bool backward,
                              uint32_t num_aiv, uint32_t elem_size) {
    const uint64_t ub_budget = g_ub_budget;

    MhcExpandTilingData t;
    memset(&t, 0, sizeof(t));
    t.S = S; t.D = D; t.m = m; t.backward = backward ? 1u : 0u;

    uint32_t d_tile_len;
    if ((uint64_t)D * elem_size <= ub_budget) {
        d_tile_len = D;
    } else {
        uint64_t tl = std::min<uint64_t>(2048, D);
        uint64_t max_t = ub_budget / (uint64_t)elem_size;
        tl = std::min(tl, max_t);
        if (tl > 16) tl = (tl / 16) * 16;
        if (tl < 16) tl = 16;
        if (tl > D) tl = D;
        if (tl < 1) tl = 1;
        d_tile_len = (uint32_t)tl;
    }
    t.dTileLen = d_tile_len;
    t.dTileNum = (D + d_tile_len - 1) / d_tile_len;
    t.dTailLen = D - (t.dTileNum - 1) * d_tile_len;

    uint64_t total_tasks = 0;
    uint32_t split_mode = SPLIT_ROW;
    uint32_t block_dim = num_aiv;

    if (S >= num_aiv) {
        split_mode = SPLIT_ROW;
        total_tasks = S;
    } else if (!backward) {
        uint64_t stream_tasks = (uint64_t)S * m;
        if (stream_tasks >= num_aiv) {
            split_mode = SPLIT_ROW_STREAM;
            total_tasks = stream_tasks;
        } else {
            split_mode = SPLIT_ELEMENT;
            total_tasks = (uint64_t)S * m * t.dTileNum;
        }
    } else {
        split_mode = SPLIT_ELEMENT;
        total_tasks = (uint64_t)S * t.dTileNum;
    }

    if (total_tasks < num_aiv) block_dim = (uint32_t)total_tasks;
    if (block_dim == 0) block_dim = 1;

    t.splitMode = split_mode;
    t.blockDim = block_dim;
    t.rowsPerCore = (uint32_t)((total_tasks + block_dim - 1) / block_dim);
    uint64_t tail = total_tasks - (uint64_t)t.rowsPerCore * (block_dim - 1);
    t.tailRows = (uint32_t)tail;

    TilingPlan p;
    p.t = t;
    p.total_tasks = (uint32_t)total_tasks;
    return p;
}

// ===== 参考实现 =====
template <typename T>
static void ref_forward(const T* x, T* o, int S, int D, int m) {
    for (int s = 0; s < S; s++)
        for (int k = 0; k < m; k++)
            for (int j = 0; j < D; j++)
                o[(size_t)s * m * D + (size_t)k * D + j] = x[(size_t)s * D + j];
}
template <typename T>
static void ref_backward(const T* x, T* o, int S, int D, int m) {
    for (int s = 0; s < S; s++)
        for (int j = 0; j < D; j++) {
            float acc = 0.0f;
            for (int k = 0; k < m; k++) acc += (float)x[(size_t)(s * m + k) * D + j];
            o[(size_t)s * D + j] = (T)acc;
        }
}

// ===== 运行单个用例 =====
static void* shared_alloc(size_t sz) {
    void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    memset(p, 0, sz);
    return p;
}

static size_t g_sim_elems = 0;   // 累计 VEC 元素当量（Add/Cast 等逐元素 op）
static size_t g_dma_elems = 0;   // 累计 DMA 搬运元素当量（GM<->UB）

template <typename T, bool BACKWARD>
static int run_one(const char* name, uint32_t S, uint32_t D, uint32_t m,
                   uint32_t num_aiv, float tol, int fill_mode = 0) {
    const size_t in_elems  = BACKWARD ? (size_t)S * m * D : (size_t)S * D;
    const size_t out_elems = BACKWARD ? (size_t)S * D : (size_t)S * m * D;
    g_sim_elems += in_elems + out_elems;
    const size_t in_sz  = in_elems * sizeof(T);
    const size_t out_sz = out_elems * sizeof(T);

    T* x = (T*)shared_alloc(in_sz);
    T* o = (T*)shared_alloc(out_sz);
    // 有符号取值：uint32 直接 -14 会回绕出 ~5e8 溢出值，混入饱和路径干扰对拍
    // fill_mode: 0=ramp 有符号  1=常数 1.0  2=常数 32768（fp16 饱和测试）
    for (size_t i = 0; i < in_elems; i++)
        x[i] = (fill_mode == 1) ? (T)1.0f
              : (fill_mode == 2) ? (T)32768.0f
              : (T)((float)((int64_t)((i * 37 + 11) % 29) - 14) * 0.125f);

    TilingPlan p = plan_tiling(S, D, m, BACKWARD, num_aiv, sizeof(T));
    MhcExpandTilingData* t = (MhcExpandTilingData*)shared_alloc(sizeof(MhcExpandTilingData));
    memcpy(t, &p.t, sizeof(*t));

    typedef void (*KFunc)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
    KFunc kf = reinterpret_cast<KFunc>(&mhc_expand<T, BACKWARD>);
    const clock_t tk0 = clock();
    ICPU_RUN_KF(kf, p.t.blockDim, (GM_ADDR)x, (GM_ADDR)o, (GM_ADDR)nullptr, (GM_ADDR)t);
    const double kern_s = (double)(clock() - tk0) / CLOCKS_PER_SEC;
    // DMA 当量 = 读+写全部元素；VEC 当量 = 反向 m 次 Cast + m 次 Add + 1 Cast 输出
    g_dma_elems += in_elems + out_elems;
    g_sim_elems += BACKWARD ? out_elems * (2 * m + 1) : 0;

    std::vector<T> oref(out_elems);
    if (BACKWARD) ref_backward<T>(x, oref.data(), (int)S, (int)D, (int)m);
    else          ref_forward<T>(x, oref.data(), (int)S, (int)D, (int)m);

    // 饱和测试（fill_mode==2）：x 全 32768，和 32768*m 超 fp16 量程时内核
    // CAST_RINT 饱和到 0x7bff(65504)。宿主 half 转换在 [65536,131072) 走非 IEEE
    // 路径产出 0x7fff，与内核约定不同，故直接对拍饱和期望而非宿主 ref。
    if (fill_mode == 2 && BACKWARD) {
        float sum = 32768.0f * (float)m;
        float expected = (sum > 65504.0f) ? 65504.0f : sum;
        for (size_t i = 0; i < out_elems; i++) oref[i] = (T)expected;
    }

    size_t mismatch = 0; float maxdiff = 0;
    std::vector<uint32_t> bad_idx;
    for (size_t i = 0; i < out_elems; i++) {
        float d = fabsf((float)o[i] - (float)oref[i]);
        if (d > maxdiff) maxdiff = d;
        if (d > tol) { mismatch++; if (bad_idx.size() < 12) bad_idx.push_back((uint32_t)i); }
    }
    printf("[%s] S=%u D=%u m=%u blk=%u mode=%u tile=%u kern=%.1fs -> maxdiff=%.5f mismatch=%zu/%zu %s\n",
           name, S, D, m, p.t.blockDim, p.t.splitMode, p.t.dTileLen, kern_s,
           maxdiff, mismatch, out_elems, mismatch ? "FAIL" : "PASS");
    if (!bad_idx.empty()) {
        printf("    first mismatches:");
        for (uint32_t idx : bad_idx) {
            uint32_t row = BACKWARD ? idx / D : idx / (m * D);
            printf(" [i=%u row=%u got=%.4f ref=%.4f]", idx, row, (float)o[idx], (float)oref[idx]);
        }
        printf("\n");
    }

    munmap(x, in_sz); munmap(o, out_sz); munmap(t, sizeof(*t));
    return mismatch ? 1 : 0;
}

// ===== 诊断模式：反向 fp16 深挖（输入前后位级快照 + 坏值位模式统计） =====
static float h2f(uint16_t bits) {
    half h;
    memcpy(&h, &bits, sizeof(h));
    return (float)h;
}

static void diag_bwd_fp16(uint32_t S, uint32_t D, uint32_t m, uint32_t num_aiv, int fill_mode) {
    typedef half T;
    const size_t in_elems = (size_t)S * m * D;
    const size_t out_elems = (size_t)S * D;
    auto shared_alloc = [](size_t sz) {
        void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { perror("mmap"); exit(1); }
        memset(p, 0, sz);
        return p;
    };

    T* x = (T*)shared_alloc(in_elems * sizeof(T));
    T* o = (T*)shared_alloc(out_elems * sizeof(T));
    T* x_save = (T*)malloc(in_elems * sizeof(T));
    // 刻意保留无符号回绕填充：复现历史 FAIL（宿主/内核 fp16 饱和约定差异），仅诊断用
    for (size_t i = 0; i < in_elems; i++)
        x[i] = (fill_mode == 1) ? (T)1.0f
                                : (T)((float)(((uint32_t)(i * 37 + 11) % 29) - 14) * 0.125f);
    memcpy(x_save, x, in_elems * sizeof(T));

    size_t bad_fill = 0;
    for (size_t i = 0; i < in_elems; i++)
        if (fabsf((float)x[i]) > 2.0f) bad_fill++;

    TilingPlan p = plan_tiling(S, D, m, true, num_aiv, sizeof(T));
    MhcExpandTilingData* t = (MhcExpandTilingData*)shared_alloc(sizeof(MhcExpandTilingData));
    memcpy(t, &p.t, sizeof(*t));
    printf("[diag] S=%u D=%u m=%u blk=%u mode=%u tile=%u fill=%d bad_fill=%zu/%zu\n",
           S, D, m, p.t.blockDim, p.t.splitMode, p.t.dTileLen, fill_mode, bad_fill, in_elems);

    typedef void (*KFunc)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
    KFunc kf = reinterpret_cast<KFunc>(&mhc_expand<T, true>);
    ICPU_RUN_KF(kf, p.t.blockDim, (GM_ADDR)x, (GM_ADDR)o, (GM_ADDR)nullptr, (GM_ADDR)t);

    // 1) 内核是否写坏了输入 x（跑核前后位级快照比对）
    size_t x_dirty = 0;
    std::vector<size_t> dirty_pos;
    for (size_t i = 0; i < in_elems; i++) {
        if (memcmp(&x[i], &x_save[i], sizeof(T)) != 0) {
            x_dirty++;
            if (dirty_pos.size() < 12) dirty_pos.push_back(i);
        }
    }
    printf("[diag] x corrupted after kernel: %zu/%zu\n", x_dirty, in_elems);
    for (size_t i : dirty_pos) {
        uint32_t row = (uint32_t)(i / D), s = row / m, k = row % m, j = (uint32_t)(i % D);
        printf("    xdirty[i=%zu s=%u k=%u j=%u] bits %04x -> %04x (val %.5f -> %.5f)\n",
               i, s, k, j,
               *(uint16_t*)&x_save[i], *(uint16_t*)&x[i],
               h2f(*(uint16_t*)&x_save[i]), h2f(*(uint16_t*)&x[i]));
    }
    if (x_dirty) {
        std::map<uint16_t, size_t> pat;
        for (size_t i = 0; i < in_elems; i++)
            if (memcmp(&x[i], &x_save[i], sizeof(T)) != 0) pat[*(uint16_t*)&x[i]]++;
        printf("    corrupted-value bit patterns:");
        for (auto& kv : pat) printf(" %04x(x%zu)", kv.first, kv.second);
        printf("\n");
    }

    // 2) 分别用跑核前 / 跑核后的 x 算参考，切分「内核算错」vs「参考读到坏数据」
    std::vector<T> oref_save(out_elems), oref_post(out_elems);
    ref_backward<T>(x_save, oref_save.data(), (int)S, (int)D, (int)m);
    ref_backward<T>(x, oref_post.data(), (int)S, (int)D, (int)m);
    size_t mm_save = 0, mm_post = 0;
    for (size_t i = 0; i < out_elems; i++) {
        if (fabsf((float)o[i] - (float)oref_save[i]) > 1e-2f) mm_save++;
        if (fabsf((float)o[i] - (float)oref_post[i]) > 1e-2f) mm_post++;
    }
    printf("[diag] o vs ref(x_pre)=%zu mism, o vs ref(x_post)=%zu mism / %zu\n",
           mm_save, mm_post, out_elems);

    // 3) 位级 dump 前 12 个 o-vs-ref(x_pre) 不匹配点（含两个 k 位置的输入前后位）
    size_t shown = 0;
    for (size_t i = 0; i < out_elems && shown < 12; i++) {
        if (fabsf((float)o[i] - (float)oref_save[i]) > 1e-2f) {
            uint32_t s = (uint32_t)(i / D), j = (uint32_t)(i % D);
            size_t i0 = ((size_t)s * m + 0) * D + j, i1 = ((size_t)s * m + 1) * D + j;
            printf("    mm[i=%zu s=%u j=%u] o=%04x(%.4f) ref=%04x(%.4f) | xpre k0=%04x(%.4f) k1=%04x(%.4f) | xpost k0=%04x(%.4f) k1=%04x(%.4f)\n",
                   i, s, j,
                   *(uint16_t*)&o[i], (float)o[i],
                   *(uint16_t*)&oref_save[i], (float)oref_save[i],
                   *(uint16_t*)&x_save[i0], (float)x_save[i0],
                   *(uint16_t*)&x_save[i1], (float)x_save[i1],
                   *(uint16_t*)&x[i0], (float)x[i0],
                   *(uint16_t*)&x[i1], (float)x[i1]);
            shown++;
        }
    }
    free(x_save);
    munmap(x, in_elems * sizeof(T)); munmap(o, out_elems * sizeof(T)); munmap(t, sizeof(*t));
}

int main(int argc, char** argv) {
    // 真机 910B3 AIV 核数；可用参数覆盖（小值可强制触发流切/元素切路径）
    uint32_t num_aiv = 50;
    if (argc > 1) num_aiv = (uint32_t)atoi(argv[1]);
    // 模式：quick（默认，小用例全量）/ medium（官方中档）/ large（官方 fwd 大档）/ mtile（大 D 多 tile）/ full
    const char* mode = (argc > 2) ? argv[2] : "quick";
    bool run_quick  = !strcmp(mode, "quick") || !strcmp(mode, "full");
    bool run_medium = !strcmp(mode, "medium") || !strcmp(mode, "full");
    bool run_large  = !strcmp(mode, "large")  || !strcmp(mode, "full");
    bool run_mtile  = !strcmp(mode, "mtile")  || !strcmp(mode, "full") || !strcmp(mode, "ub48");
    if (!strcmp(mode, "ub48")) g_ub_budget = 48 * 1024;   // host ub_size==0 兜底 192KB 的路径

    printf("=== mhc_expand CPU matrix test (aiv=%u mode=%s) ===\n", num_aiv, mode);
    const clock_t t0 = clock();
    int fail = 0;
    const float TOL = 1e-2f;

    // 诊断模式：只跑反向 fp16 深挖，不跑矩阵
    if (argc > 2 && strcmp(argv[2], "diag") == 0) {
        diag_bwd_fp16(64, 256, 2, num_aiv, 0);
        printf("--- m sweep (S=64 D=256 fp16 bwd, ramp fill) ---\n");
        for (uint32_t mm : {1u, 2u, 3u, 4u, 5u, 8u, 16u, 32u})
            run_one<half, true>("bwd-msweep", 64, 256, mm, num_aiv, TOL);
        printf("--- D sweep (S=64 m=2 fp16 bwd, ramp fill) ---\n");
        for (uint32_t dd : {64u, 128u, 256u, 512u, 1024u})
            run_one<half, true>("bwd-Dsweep", 64, dd, 2, num_aiv, TOL);
        printf("--- const fill (value-dependence control) ---\n");
        run_one<half, true>("bwd-const1", 64, 256, 2, num_aiv, TOL, 1);
        run_one<half, true>("bwd-const1-D100", 4, 100, 2, num_aiv, TOL, 1);
        printf("=== diag done ===\n");
        return 0;
    }

    if (run_quick) {
        // 官方小规模 + 非对齐 + 边界 + 饱和 + bf16
        fail += run_one<half, false>("fwd-fp16-small",   64, 256, 2, num_aiv, TOL);
        fail += run_one<half, false>("fwd-fp16-D100",    7, 100, 2, num_aiv, TOL);
        fail += run_one<half, false>("fwd-fp16-D7167",   3, 7167, 2, num_aiv, TOL);
        fail += run_one<half, false>("fwd-fp16-S1D1",    1, 1, 2, num_aiv, TOL);
        fail += run_one<half, false>("fwd-fp16-m1",      5, 33, 1, num_aiv, TOL);
        fail += run_one<half, false>("fwd-fp16-m16",     64, 512, 16, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-small",   64, 256, 2, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-D100",    4, 100, 2, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-D7167",   2, 7167, 2, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-S1D1",    1, 1, 2, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-m1",      5, 33, 1, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-m16",     64, 512, 16, num_aiv, TOL);
        // 显式饱和：x 全 32768；m=1 和在量程内精确，m=2 和 65536 超量程应饱和到 65504
        fail += run_one<half, true> ("bwd-fp16-sat-m1",  4, 128, 1, num_aiv, TOL, 2);
        fail += run_one<half, true> ("bwd-fp16-sat-m2",  4, 128, 2, num_aiv, TOL, 2);
        fail += run_one<bfloat16_t, false>("fwd-bf16-small",  64, 256, 2, num_aiv, TOL);
        fail += run_one<bfloat16_t, false>("fwd-bf16-D7167",  3, 7167, 2, num_aiv, TOL);
        fail += run_one<bfloat16_t, true> ("bwd-bf16-small",  64, 256, 2, num_aiv, TOL);
        fail += run_one<bfloat16_t, true> ("bwd-bf16-D7167",  2, 7167, 2, num_aiv, TOL);
        // 小核数：强制走 SPLIT_ROW_STREAM / SPLIT_ELEMENT 路径
        printf("--- forced small-core modes (aiv=4) ---\n");
        fail += run_one<half, false>("fwd-fp16-STREAM", 8, 7168, 2, 4, TOL);
        fail += run_one<half, false>("fwd-fp16-ELEMENT", 3, 7168, 2, 4, TOL);
        fail += run_one<half, true> ("bwd-fp16-ELEMENT", 3, 7168, 2, 4, TOL);
        fail += run_one<half, true> ("bwd-fp16-ELEM-D7167", 2, 7167, 2, 4, TOL);
        // 整行多副本（dTileNum=1）：前向 ROW 读 1 写 m、反向双缓冲 m=2~5
        fail += run_one<half, false>("fwd-fp16-ROW-m8",   4, 2048, 8, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-m2",       8, 256, 2, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-m3",       8, 256, 3, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-m4",       8, 256, 4, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-m5",       8, 256, 5, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-D7167-m5", 2, 7167, 5, num_aiv, TOL);
        fail += run_one<bfloat16_t, true> ("bwd-bf16-m4", 8, 256, 4, num_aiv, TOL);
    }

    // 官方中档（优化后整行路径 dTileNum=1）
    if (run_medium) {
        printf("--- medium group ---\n");
        fail += run_one<half, false>("fwd-fp16-medium",  1024, 4096, 4, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-medium",  1024, 4096, 4, num_aiv, TOL);
        fail += run_one<bfloat16_t, false>("fwd-bf16-medium", 1024, 4096, 4, num_aiv, TOL);
        fail += run_one<bfloat16_t, true> ("bwd-bf16-medium", 1024, 4096, 4, num_aiv, TOL);
    }

    // 官方大档：fwd + bwd 实测仿真仅 ~25s / ~10min，不必留给真机
    if (run_large) {
        printf("--- large group ---\n");
        fail += run_one<half, false>("fwd-fp16-large",   8192, 7168, 8, num_aiv, TOL);
        fail += run_one<bfloat16_t, false>("fwd-bf16-large",  8192, 7168, 8, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-large",   8192, 7168, 8, num_aiv, TOL);
        fail += run_one<bfloat16_t, true> ("bwd-bf16-large",  8192, 7168, 8, num_aiv, TOL);
    }

    if (run_mtile) {
        // 旧 MT 组（quick 版按 m 因子预算）：优化后转整行路径，保留作回归
        printf("--- multi-tile legacy (now full-row under fixed tiling) ---\n");
        fail += run_one<half, true> ("bwd-fp16-MT-ELEM",   4, 4096, 16, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-MT-ODD",    2, 7167, 8, num_aiv, TOL);
        fail += run_one<half, false>("fwd-fp16-MT-STREAM", 3, 7168, 16, 4, TOL);
        fail += run_one<half, false>("fwd-fp16-MT-ELEMENT",1, 7168, 4, num_aiv, TOL);
        fail += run_one<bfloat16_t, true> ("bwd-bf16-MT-ELEM", 4, 4096, 16, num_aiv, TOL);
        fail += run_one<bfloat16_t, true> ("bwd-bf16-MT-ODD",  2, 7167, 8, num_aiv, TOL);
        // 新条件（D 因子预算）下逼出 dTileNum>1 的大 D 用例。
        // 注意 host else 分支固定 tile = min(2048, D)，所以只有 D 为奇数时尾块才是奇数
        printf("--- multi-tile large-D group (fixed tiling condition) ---\n");
        fail += run_one<half, true> ("bwd-fp16-MTL-D70000-m8",  2, 70000, 8, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-MTL-D70000-m2",  2, 70000, 2, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-MTL-D33000-m8",  1, 33000, 8, num_aiv, TOL);
        fail += run_one<bfloat16_t, true>("bwd-bf16-MTL-D70000-m2", 2, 70000, 2, num_aiv, TOL);
        fail += run_one<half, false>("fwd-fp16-MTL-D70000-m2",  1, 70000, 2, num_aiv, TOL);
        fail += run_one<half, false>("fwd-fp16-MTL-D33000-m8",  1, 33000, 8, 4, TOL);
        // 奇数尾块多 tile：D 为奇数 → tail = D - 2048*(n-1) 为奇数
        fail += run_one<half, true> ("bwd-fp16-MTL-D70001-m2",  2, 70001, 2, num_aiv, TOL);
        fail += run_one<bfloat16_t, true>("bwd-bf16-MTL-D33001-m8", 1, 33001, 8, num_aiv, TOL);
        fail += run_one<half, false>("fwd-fp16-MTL-D70001-m2",  1, 70001, 2, num_aiv, TOL);
        // 预算边界一对：D*2 == 64KB 走整行（<=），D*2 == 64KB+2 退化多 tile 且尾块只剩 1 元素
        printf("--- budget-boundary pair (D*elem == ub_budget vs +1) ---\n");
        fail += run_one<half, true> ("bwd-fp16-BND-D32768-m2",   2, 32768, 2, num_aiv, TOL);
        fail += run_one<half, true> ("bwd-fp16-BND-D32769-m2",   2, 32769, 2, num_aiv, TOL);
        fail += run_one<half, false>("fwd-fp16-BND-D32769-m4",   1, 32769, 4, num_aiv, TOL);
    }

    // host ub_size==0 兜底 192KB（预算 48KB）：D∈(24576,32768] 由整行退化为多 tile
    if (!strcmp(mode, "ub48")) {
        printf("--- ub48 fallback-budget group ---\n");
        fail += run_one<half, true> ("bwd-fp16-UB48-D26001-m2",  2, 26001, 2, num_aiv, TOL);
        fail += run_one<half, false>("fwd-fp16-UB48-D26000-m4",  2, 26000, 4, num_aiv, TOL);
    }

    // 诊断 sweep：同一反向用例扫不同核数（定位错误与核数/区间的关系）
    if (run_quick && argc > 3 && strcmp(argv[3], "sweep") == 0) {
        printf("--- bwd-fp16-small block sweep ---\n");
        for (uint32_t b : {1u, 2u, 4u, 8u, 16u, 20u})
            run_one<half, true>("bwd-sweep", 64, 256, 2, b, TOL);
    }

    {
        double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("=== %s | dma_elems=%.1fM vec_elems=%.1fM cpu_s=%.0f dma_rate=%.1fM/s ===\n",
               fail ? "HAS FAIL" : "ALL PASS",
               (double)g_dma_elems / 1e6, (double)g_sim_elems / 1e6, secs,
               secs > 0 ? (double)g_dma_elems / 1e6 / secs : 0.0);
    }
    return fail;
}
