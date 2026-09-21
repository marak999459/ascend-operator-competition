// mhc_sinkhorn 仿真机对拍 harness（同源包含提交 kernel，零改动）
// 用法: ./test_sinkhorn_cpu [num_aiv] [quick|full|det|mag]
// 参考实现按官方题面 fp32 口径写在文件内（softmax 减行最大 -> /(sum+eps) 交替归一化），
// 与 kernel 的差异只在 exp 的多项式实现与末位舍入 -> 判据 maxdiff <= 1e-2。
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <sys/mman.h>
#include <vector>

#include "kernel_operator.h"
#include "tikicpulib.h"
using namespace AscendC;

#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(T, name, ptr) \
    const T& name = *reinterpret_cast<const T*>(ptr)
#endif

#include "mhc_sinkhorn_tiling.h"
#include "mhc_sinkhorn.cpp"

// ===== 镜像 op_host/mhc_sinkhorn.cpp 的 TilingFunc 关键路径 =====
struct Plan {
    MhcSinkhornTilingData t;
    uint32_t block_dim;
};

static Plan plan_tiling(uint32_t batch, uint32_t n, uint32_t iters, float eps,
                        uint32_t num_aiv) {
    Plan p;
    memset(&p.t, 0, sizeof(p.t));
    p.t.batch = batch;
    p.t.n = n;
    p.t.numIters = iters;
    p.t.eps = eps;
    p.t.coreNum = num_aiv;
    p.t.batchPerCore = (batch + num_aiv - 1u) / num_aiv;
    p.block_dim = num_aiv;   // host 侧恒为 SetBlockDim(coreNum)，含 batch < 核数的空转核
    return p;
}

// ===== 参考实现（fp32 域，与 kernel 同结构：行 softmax -> 列归一 -> 交替） =====
static void ref_sinkhorn(const std::vector<float>& xin, std::vector<float>& a,
                         uint32_t batch, uint32_t n, uint32_t iters, float eps) {
    a = xin;
    const uint32_t steps = (iters == 0) ? 1u : iters;
    for (uint32_t m = 0; m < batch; ++m) {
        float* p = a.data() + (size_t)m * n * n;
        for (uint32_t step = 0; step < steps; ++step) {
            if (step == 0) {   // 初始化：行 softmax（减行最大）
                for (uint32_t i = 0; i < n; ++i) {
                    float mx = p[i * n];
                    for (uint32_t j = 0; j < n; ++j) mx = std::max(mx, p[i * n + j]);
                    float s = 0.0f;
                    for (uint32_t j = 0; j < n; ++j) { p[i * n + j] = expf(p[i * n + j] - mx); s += p[i * n + j]; }
                    s += eps;
                    for (uint32_t j = 0; j < n; ++j) p[i * n + j] /= s;
                }
            } else {           // 交替迭代：行归一化
                for (uint32_t i = 0; i < n; ++i) {
                    float s = eps;
                    for (uint32_t j = 0; j < n; ++j) s += p[i * n + j];
                    for (uint32_t j = 0; j < n; ++j) p[i * n + j] /= s;
                }
            }
            for (uint32_t j = 0; j < n; ++j) {   // 每步收尾做一次列归一化
                float s = eps;
                for (uint32_t i = 0; i < n; ++i) s += p[i * n + j];
                for (uint32_t i = 0; i < n; ++i) p[i * n + j] /= s;
            }
        }
    }
}

static void* shared_alloc(size_t sz) {
    void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    memset(p, 0, sz);
    return p;
}

// 确定性伪随机（跨 x86_64 / aarch64 一致，避免 rand() 实现差异）
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    float uniform(float lo, float hi) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const uint32_t r = (uint32_t)((s >> 33) & 0xFFFFFFFFu);
        return lo + (hi - lo) * ((float)r / 4294967295.0f);
    }
};

template <typename T>
static int run_one(const char* name, uint32_t batch, uint32_t n, uint32_t iters,
                   float eps, uint32_t num_aiv, float tol, float lo, float hi,
                   uint64_t seed, int* bad_matrix_out = nullptr) {
    const size_t total = (size_t)batch * n * n;
    std::vector<float> xin(total);
    Rng rng(seed);
    for (size_t i = 0; i < total; ++i) xin[i] = rng.uniform(lo, hi);

    // 输入先量化到 T（kernel 读入即 T 域），参考实现从同一份量化值出发
    std::vector<float> qin(total);
    for (size_t i = 0; i < total; ++i) qin[i] = (float)(T)xin[i];

    std::vector<float> ref;
    ref_sinkhorn(qin, ref, batch, n, iters, eps);

    T* x = (T*)shared_alloc(total * sizeof(T));
    T* o = (T*)shared_alloc(total * sizeof(T));
    for (size_t i = 0; i < total; ++i) x[i] = (T)xin[i];

    Plan p = plan_tiling(batch, n, iters, eps, num_aiv);
    MhcSinkhornTilingData* t = (MhcSinkhornTilingData*)shared_alloc(sizeof(MhcSinkhornTilingData));
    memcpy(t, &p.t, sizeof(*t));

    typedef void (*KFunc)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
    KFunc kf = reinterpret_cast<KFunc>(&mhc_sinkhorn<T>);
    const clock_t tk0 = clock();
    ICPU_RUN_KF(kf, p.block_dim, (GM_ADDR)x, (GM_ADDR)nullptr, (GM_ADDR)o,
                (GM_ADDR)nullptr, (GM_ADDR)t);
    const double kern_s = (double)(clock() - tk0) / CLOCKS_PER_SEC;

    size_t mismatch = 0;
    float maxdiff = 0.0f;
    int bad_matrix = -1;
    for (uint32_t m = 0; m < batch; ++m) {
        float worst = 0.0f;
        for (size_t k = 0; k < (size_t)n * n; ++k) {
            const size_t i = (size_t)m * n * n + k;
            const float d = fabsf((float)o[i] - (float)(T)ref[i]);
            if (d > maxdiff) maxdiff = d;
            if (d > worst) worst = d;
            if (d > tol) mismatch++;
        }
        if (worst > tol && bad_matrix < 0) bad_matrix = (int)m;
    }

    printf("[%-18s] batch=%-5u n=%u iters=%-3u aiv=%-3u [%g,%g] sim=%.2fs -> maxdiff=%.5f mism=%zu/%zu %s",
           name, batch, n, iters, num_aiv, lo, hi, kern_s, maxdiff, mismatch, total,
           mismatch ? "FAIL" : "PASS");
    if (bad_matrix >= 0) printf("  first_bad_matrix=%d", bad_matrix);
    printf("\n");

    if (bad_matrix_out) *bad_matrix_out = bad_matrix;
    munmap(x, total * sizeof(T)); munmap(o, total * sizeof(T)); munmap(t, sizeof(*t));
    return mismatch ? 1 : 0;
}

// 确定性：同一输入跑两次，要求逐 bit 一致
static int run_determinism(uint32_t batch, uint32_t n, uint32_t iters, uint32_t num_aiv) {
    const float eps = 1e-6f;
    const size_t total = (size_t)batch * n * n;
    Rng rng(777);
    std::vector<half> x(total);
    for (size_t i = 0; i < total; ++i) x[i] = (half)rng.uniform(-4.0f, 4.0f);

    Plan p = plan_tiling(batch, n, iters, eps, num_aiv);
    std::vector<half> a(total), b(total);
    typedef void (*KFunc)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
    KFunc kf = reinterpret_cast<KFunc>(&mhc_sinkhorn<half>);
    for (int round = 0; round < 2; ++round) {
        half* dx = (half*)shared_alloc(total * sizeof(half));
        half* dobf = (half*)shared_alloc(total * sizeof(half));
        MhcSinkhornTilingData* t = (MhcSinkhornTilingData*)shared_alloc(sizeof(MhcSinkhornTilingData));
        memcpy(t, &p.t, sizeof(*t));
        memcpy(dx, x.data(), total * sizeof(half));
        ICPU_RUN_KF(kf, p.block_dim, (GM_ADDR)dx, (GM_ADDR)nullptr, (GM_ADDR)dobf,
                    (GM_ADDR)nullptr, (GM_ADDR)t);
        memcpy(round ? b.data() : a.data(), dobf, total * sizeof(half));
        munmap(dx, total * sizeof(half)); munmap(dobf, total * sizeof(half)); munmap(t, sizeof(*t));
    }
    size_t diff = 0;
    for (size_t i = 0; i < total; ++i)
        if (memcmp(&a[i], &b[i], sizeof(half)) != 0) diff++;
    printf("[determinism       ] batch=%u n=%u iters=%u -> bitdiff=%zu %s\n",
           batch, n, iters, diff, diff ? "FAIL" : "PASS");
    return diff ? 1 : 0;
}

// 题面要求"输入含 -inf/inf/nan 时对应位置输出 nan"：只记录，不计入 FAIL
static void run_special_values(uint32_t n, uint32_t num_aiv) {
    const uint32_t batch = 1, iters = 20;
    const float eps = 1e-6f;
    const size_t total = (size_t)n * n;
    std::vector<half> in(total, (half)1.0f);
    in[0] = (half)INFINITY;
    in[n + 1] = (half)(-INFINITY);
    in[2 * n + 2] = (half)NAN;

    half* dx = (half*)shared_alloc(total * sizeof(half));
    half* dobf = (half*)shared_alloc(total * sizeof(half));
    MhcSinkhornTilingData* t = (MhcSinkhornTilingData*)shared_alloc(sizeof(MhcSinkhornTilingData));
    Plan p = plan_tiling(batch, n, iters, eps, num_aiv);
    memcpy(t, &p.t, sizeof(*t));
    memcpy(dx, in.data(), total * sizeof(half));
    typedef void (*KFunc)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
    ICPU_RUN_KF(reinterpret_cast<KFunc>(&mhc_sinkhorn<half>), p.block_dim,
                (GM_ADDR)dx, (GM_ADDR)nullptr, (GM_ADDR)dobf, (GM_ADDR)nullptr, (GM_ADDR)t);
    printf("[special-values    ] n=%u 输入含 inf/-inf/nan -> 首矩阵前 6 个输出:", n);
    for (size_t i = 0; i < 6 && i < total; ++i) printf(" %.5g", (double)(float)dobf[i]);
    printf("  (仅记录，官方要求对应位置为 nan)\n");
    munmap(dx, total * sizeof(half)); munmap(dobf, total * sizeof(half)); munmap(t, sizeof(*t));
}

int main(int argc, char** argv) {
    uint32_t num_aiv = 20;   // 仿真机核数上限；真机 910B3 是 40，用参数覆盖
    if (argc > 1) num_aiv = (uint32_t)atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "quick";
    const float TOL = 1e-2f;
    int fail = 0;
    const clock_t t0 = clock();
    printf("=== mhc_sinkhorn 仿真机对拍 (aiv=%u mode=%s) ===\n", num_aiv, mode);

    if (!strcmp(mode, "det")) {
        fail += run_determinism(20, 6, 20, num_aiv);
        fail += run_determinism(1, 4, 1, num_aiv);
        run_special_values(6, num_aiv);
        printf("=== %s ===\n", fail ? "HAS FAIL" : "ALL PASS");
        return fail;
    }
    if (!strcmp(mode, "mag")) {   // 量级扫：验证减行最大防溢出这条路径
        for (uint32_t n : {4u, 6u, 8u}) {
            fail += run_one<half>("mag-small", 20, n, 20, 1e-6f, num_aiv, TOL, -4.0f, 4.0f, 11);
            fail += run_one<half>("mag-large", 20, n, 20, 1e-6f, num_aiv, TOL, -60.0f, 60.0f, 12);
            fail += run_one<half>("mag-nonneg", 20, n, 20, 1e-6f, num_aiv, TOL, 0.0f, 1.0f, 13);
            fail += run_one<float>("fp32-mag-large", 20, n, 20, 1e-6f, num_aiv, TOL, -60.0f, 60.0f, 14);
        }
        printf("=== %s | cpu_s=%.0f ===\n", fail ? "HAS FAIL" : "ALL PASS",
               (double)(clock() - t0) / CLOCKS_PER_SEC);
        return fail;
    }

    // quick：正确性回归主矩阵（对齐 code2.md §5.3 的 n × batch 覆盖口径）
    for (uint32_t n : {4u, 6u, 8u}) {
        for (uint32_t batch : {1u, 7u, 20u, 40u, 64u, 100u}) {
            fail += run_one<half>("fp16", batch, n, 20, 1e-6f, num_aiv, TOL, -4.0f, 4.0f, 1234);
        }
        fail += run_one<float>("fp32", 20, n, 20, 1e-6f, num_aiv, TOL, -4.0f, 4.0f, 1234);
        fail += run_one<half>("fp16-it1", 20, n, 1, 1e-6f, num_aiv, TOL, -4.0f, 4.0f, 55);
        fail += run_one<half>("fp16-it100", 7, n, 100, 1e-6f, num_aiv, TOL, -4.0f, 4.0f, 56);
    }
    if (!strcmp(mode, "full")) {
        printf("--- full: 大批量 + 多 seed ---\n");
        for (uint32_t n : {4u, 6u, 8u}) {
            fail += run_one<half>("fp16-big", 1024, n, 20, 1e-6f, num_aiv, TOL, -4.0f, 4.0f, 7);
            fail += run_one<half>("fp16-seed2", 20, n, 20, 1e-6f, num_aiv, TOL, -4.0f, 4.0f, 2);
            fail += run_one<half>("fp16-seed3", 20, n, 20, 1e-6f, num_aiv, TOL, -4.0f, 4.0f, 3);
            fail += run_one<float>("fp32-big", 256, n, 20, 1e-6f, num_aiv, TOL, -4.0f, 4.0f, 8);
        }
        fail += run_determinism(20, 6, 20, num_aiv);
        run_special_values(6, num_aiv);
    }

    printf("=== %s | cpu_s=%.0f ===\n", fail ? "HAS FAIL" : "ALL PASS",
           (double)(clock() - t0) / CLOCKS_PER_SEC);
    return fail;
}
