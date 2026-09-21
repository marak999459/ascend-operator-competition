// mhc_sinkhorn 真机耗时测量（第三题 refs/sfa/bench.cpp 的同款口径：预热 + 逐次计时 + 连发计时）
// 用法: ./bench_sinkhorn <batch> <n> <iters> [reps] [fp16|fp32]
// 编译（真机，先按 run.sh 组装好 vendor）:
//   g++ -std=c++17 -O2 bench_sinkhorn.cpp -o bench_sink \
//     -I$CANN/aarch64-linux/include -I$V/op_api/include \
//     -L$CANN/aarch64-linux/lib64 -L$V/op_api/lib -lascendcl -lnnopbase -lcust_opapi
// ⚠ 连发计时含 ~2-5µs 的启动开销下界；要拿纯 kernel 时长用 msprof 包一层
//   （本目录 run.sh 的 PROF=1 分支已做包装，见其 msprof 段）。
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_mhc_sinkhorn.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <vector>
#include <string>

#define CHK(expr, msg)                                                        \
    do {                                                                      \
        aclError _e = (expr);                                                 \
        if (_e != ACL_SUCCESS) { printf("[FAIL] %s -> %d\n", msg, (int)_e); return 1; } \
    } while (0)

static double now_us()
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static uint16_t f32_to_f16(float v)
{
    uint32_t b; memcpy(&b, &v, 4);
    const uint32_t s = (b >> 31) & 1u;
    int32_t e = (int32_t)((b >> 23) & 0xFFu) - 127;
    uint32_t m = b & 0x7FFFFFu;
    if (e < -14) return (uint16_t)(s << 15);
    if (e > 15)  return (uint16_t)((s << 15) | (0x1Fu << 10));
    return (uint16_t)((s << 15) | ((uint32_t)(e + 15) << 10) | (m >> 13));
}

int main(int argc, char** argv)
{
    const int64_t batch = (argc > 1) ? atoll(argv[1]) : 20;
    const int64_t n     = (argc > 2) ? atoll(argv[2]) : 6;
    const int64_t iters = (argc > 3) ? atoll(argv[3]) : 20;
    const int     reps  = (argc > 4) ? atoi(argv[4]) : 50;
    const bool    f32   = (argc > 5) && !strcmp(argv[5], "fp32");
    const int64_t per   = n * n;
    const size_t  bytes = (size_t)batch * per * (f32 ? 4 : 2);
    const float   eps   = 1e-6f;

    CHK(aclInit(nullptr), "aclInit");
    CHK(aclrtSetDevice(0), "setDevice");
    aclrtStream stream = nullptr;
    CHK(aclrtCreateStream(&stream), "createStream");

    std::vector<uint8_t> hin(bytes);
    uint32_t seed = 12345u;
    for (size_t i = 0; i < (size_t)batch * per; ++i) {
        seed = seed * 1103515245u + 12345u;
        const float v = ((float)(seed >> 8) / 8388608.0f) - 4.0f;   // [-4, 4)
        if (f32) memcpy(&hin[i * 4], &v, 4);
        else { const uint16_t h = f32_to_f16(v); memcpy(&hin[i * 2], &h, 2); }
    }

    void *dx = nullptr, *dy = nullptr;
    CHK(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc x");
    CHK(aclrtMalloc(&dy, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc y");
    CHK(aclrtMemcpy(dx, bytes, hin.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "h2d");

    int64_t dims[3] = {batch, n, n};
    int64_t strides[3] = {per, n, 1};
    auto x = aclCreateTensor(dims, 3, f32 ? ACL_FLOAT : ACL_FLOAT16, strides, 0,
                             ACL_FORMAT_ND, dims, 3, dx);
    auto y = aclCreateTensor(dims, 3, f32 ? ACL_FLOAT : ACL_FLOAT16, strides, 0,
                             ACL_FORMAT_ND, dims, 3, dy);
    if (!x || !y) { printf("[FAIL] aclCreateTensor\n"); return 1; }

    uint64_t wsSize = 0; aclOpExecutor* ex = nullptr;
    aclnnStatus st = aclnnMhcSinkhornGetWorkspaceSize(x, nullptr, iters, eps, y, &wsSize, &ex);
    if (st != 0) { printf("[FAIL] GetWorkspaceSize %d\n", (int)st); return 1; }
    void* ws = nullptr;
    if (wsSize > 0) CHK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws");

    for (int i = 0; i < 3; ++i) aclnnMhcSinkhorn(ws, wsSize, ex, stream);
    CHK(aclrtSynchronizeStream(stream), "warmup sync");

    double mn = 1e18, mx = 0, sum = 0;
    for (int i = 0; i < reps; ++i) {
        const double t0 = now_us();
        aclnnMhcSinkhorn(ws, wsSize, ex, stream);
        aclrtSynchronizeStream(stream);
        const double dt = now_us() - t0;
        sum += dt; if (dt < mn) mn = dt; if (dt > mx) mx = dt;
    }
    const double single_avg = sum / reps;

    CHK(aclrtSynchronizeStream(stream), "pre-burst sync");
    const double t0 = now_us();
    for (int i = 0; i < reps; ++i) aclnnMhcSinkhorn(ws, wsSize, ex, stream);
    CHK(aclrtSynchronizeStream(stream), "burst sync");
    const double burst_avg = (now_us() - t0) / reps;

    printf("batch=%ld n=%ld iters=%ld %s | single_avg=%.2fus single_min=%.2fus burst_avg=%.2fus\n",
           (long)batch, (long)n, (long)iters, f32 ? "fp32" : "fp16",
           single_avg, mn, burst_avg);

    aclDestroyTensor(x); aclDestroyTensor(y);
    aclrtFree(dx); aclrtFree(dy);
    if (ws) aclrtFree(ws);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
