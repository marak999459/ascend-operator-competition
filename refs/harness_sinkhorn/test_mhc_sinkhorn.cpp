// mHC-Sinkhorn 崩溃复现 harness
// 用法: ./test_mhc_sinkhorn <batch> <n> <iters> <eps> [输入文件]
//       DT=fp32 环境变量切到 float32 通路（比赛契约是 fp32 进 fp32 出）
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_mhc_sinkhorn.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>

#define CHK(expr, msg)                                                       \
    do {                                                                     \
        aclError _e = (expr);                                                \
        if (_e != ACL_SUCCESS) {                                             \
            printf("[FAIL] %s -> aclError=%d\n", msg, (int)_e);              \
            return 1;                                                        \
        }                                                                    \
    } while (0)

static int64_t env_int(const char *k, int64_t d) {
    const char *v = getenv(k);
    return v ? atoll(v) : d;
}

int main(int argc, char **argv) {
    const int64_t batch = (argc > 1) ? atoll(argv[1]) : 8;
    const int64_t n     = (argc > 2) ? atoll(argv[2]) : 8;
    const int64_t iters = (argc > 3) ? atoll(argv[3]) : 20;
    const double  eps   = (argc > 4) ? atof(argv[4])  : 1e-6;
    const int64_t per   = n * n;
    const int64_t total = batch * per;
    // DT=fp32 走比赛契约的 float32 通道（默认仍是历史 fp16 口径）
    const bool    f32   = getenv("DT") && !strcmp(getenv("DT"), "fp32");
    const size_t  esz   = f32 ? 4 : 2;
    const uint32_t ONE  = f32 ? 0x3F800000u : 0x3C00u;   // 常量输入 = 1.0

    printf("=== mHC-Sinkhorn 测试: batch=%ld n=%ld iters=%ld eps=%g dtype=%s (total=%ld) ===\n",
           (long)batch, (long)n, (long)iters, eps, f32 ? "fp32" : "fp16", (long)total);
    fflush(stdout);

    CHK(aclInit(nullptr), "aclInit");
    CHK(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    CHK(aclrtCreateStream(&stream), "aclrtCreateStream");

    // ---- host 数据 ----
    // 支持从文件读入随机数据（第 5 个参数）：每行一个 float，共 batch*n*n 行
    // 不指定则用常量 1.0（Sinkhorn 收敛到全 1/n，便于自检）
    std::vector<uint32_t> hx(total, ONE);
    if (argc > 5) {
        FILE *fi = fopen(argv[5], "rb");
        if (!fi) { printf("[FAIL] 打不开输入文件 %s\n", argv[5]); return 1; }
        std::vector<float> fin((size_t)total);
        size_t got = fread(fin.data(), sizeof(float), (size_t)total, fi);
        fclose(fi);
        if ((int64_t)got != total) {
            printf("[FAIL] 输入文件元素数 %zu != %ld\n", got, (long)total);
            return 1;
        }
        for (int64_t i = 0; i < total; ++i) {
            const float v = fin[i];
            if (f32) {                                   // float32 原样落位模式
                memcpy(&hx[i], &v, 4);
                continue;
            }
            // float -> fp16 位模式
            uint32_t b;
            memcpy(&b, &v, 4);
            const uint32_t s = (b >> 31) & 1u;
            int32_t e = (int32_t)((b >> 23) & 0xFFu) - 127;
            uint32_t m = b & 0x7FFFFFu;
            uint16_t h;
            if (e < -14) {                       // 下溢 -> 0
                h = (uint16_t)(s << 15);
            } else if (e > 15) {                 // 上溢 -> inf
                h = (uint16_t)((s << 15) | (0x1Fu << 10));
            } else {
                h = (uint16_t)((s << 15) | ((uint32_t)(e + 15) << 10) | (m >> 13));
            }
            hx[i] = h;
        }
        printf("[info] 已从 %s 读入 %ld 个元素\n", argv[5], (long)total);
    }
    std::vector<uint32_t> ho(total, 0);

    void *dx = nullptr, *dy = nullptr;
    CHK(aclrtMalloc(&dx, total * esz, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc x");
    CHK(aclrtMalloc(&dy, total * esz, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc y");
    CHK(aclrtMemcpy(dx, total * esz, hx.data(), total * esz, ACL_MEMCPY_HOST_TO_DEVICE), "H2D x");

    // ---- 描述符 ----
    int64_t viewDims[3] = {batch, n, n};
    int64_t strides[3]  = {per, n, 1};
    aclDataType dt = f32 ? ACL_FLOAT : ACL_FLOAT16;
    auto x = aclCreateTensor(viewDims, 3, dt, strides, 0, ACL_FORMAT_ND,
                             viewDims, 3, dx);
    auto y = aclCreateTensor(viewDims, 3, dt, strides, 0, ACL_FORMAT_ND,
                             viewDims, 3, dy);
    if (!x || !y) { printf("[FAIL] aclCreateTensor\n"); return 1; }

    // ---- 取 workspace 并执行 ----
    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus st = aclnnMhcSinkhornGetWorkspaceSize(x, nullptr, iters, eps, y,
                                                     &wsSize, &executor);
    printf("[info] GetWorkspaceSize status=%d wsSize=%llu\n", (int)st,
           (unsigned long long)wsSize);
    fflush(stdout);
    if (st != 0) { printf("[FAIL] GetWorkspaceSize\n"); return 1; }

    void *ws = nullptr;
    if (wsSize > 0) {
        CHK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc ws");
    }

    printf("[info] 调用 aclnnMhcSinkhorn ...\n");
    fflush(stdout);
    st = aclnnMhcSinkhorn(ws, wsSize, executor, stream);
    printf("[info] aclnnMhcSinkhorn status=%d\n", (int)st);
    fflush(stdout);
    if (st != 0) { printf("[FAIL] aclnnMhcSinkhorn launch\n"); return 1; }

    printf("[info] aclrtSynchronizeStream ... (若死锁/崩溃会卡在这里)\n");
    fflush(stdout);
    aclError e = aclrtSynchronizeStream(stream);
    if (e != ACL_SUCCESS) {
        printf("[FAIL] aclrtSynchronizeStream ret=%d  <== 复现崩溃!\n", (int)e);
        return 2;
    }
    printf("[OK] 同步成功，kernel 执行完成\n");

    CHK(aclrtMemcpy(ho.data(), total * esz, dy, total * esz, ACL_MEMCPY_DEVICE_TO_HOST), "D2H y");

    // ---- 打印输出 + 自检 ----
    // fp16 -> fp32：必须用 powf，不能用 (1u << (exp-15))（exp<15 时移位为负 = UB）
    auto f16 = [](uint16_t hv) -> float {
        const uint32_t sign = (hv >> 15) & 1u, exp = (hv >> 10) & 0x1Fu, man = hv & 0x3FFu;
        float f;
        if (exp == 0) {
            f = (man ? ((float)man / 1024.0f) : 0.0f) * (1.0f / 16384.0f);   // 次正规
        } else if (exp == 31) {
            f = 0.0f;                                                        // Inf/NaN 当作 0
        } else {
            f = (1.0f + (float)man / 1024.0f) * std::ldexp(1.0f, (int)exp - 15);
        }
        return sign ? -f : f;
    };
    // 线格式 -> 浮点：fp32 直接还原作者，fp16 走上面的解码
    auto dec_h = [&](uint32_t w) -> float {
        if (!f32) return f16((uint16_t)w);
        float f;
        memcpy(&f, &w, 4);
        return f;
    };

    printf("[info] 第 0 个矩阵的 n*n 输出:\n");
    for (int64_t i = 0; i < n; ++i) {
        printf("   ");
        for (int64_t k = 0; k < n; ++k) {
            printf("%9.5f", dec_h(ho[i * n + k]));
        }
        printf("\n");
    }

    // ---- 同一次运行内并排打印输入 vs 输出（避免跨进程文件时序问题）----
    printf("[cmp] 输入 vs 输出（前 3 个矩阵，每矩阵前 6 个元素）:\n");
    printf("  %-6s %-6s %-14s %-14s\n", "矩阵", "idx", "输入", "输出");
    for (int64_t mm = 0; mm < 3 && mm < batch; ++mm) {
        for (int64_t i = 0; i < 6 && i < n * n; ++i) {
            const int64_t idx = mm * n * n + i;
            printf("  %-6ld %-6ld %-14.6f %-14.6f %s\n", (long)mm, (long)i,
                   dec_h(hx[idx]), dec_h(ho[idx]),
                   (fabs(dec_h(hx[idx]) - dec_h(ho[idx])) < 1e-6) ? "<-- 相同!" : "");
        }
    }

    // ---- 自检：行和/列和应为 ~1（双随机），且应等于全 1/n 矩阵 ----
    double max_err = 0.0;
    for (int64_t mm = 0; mm < batch; ++mm) {
        for (int64_t i = 0; i < n; ++i) {
            double rs = 0.0, cs = 0.0;
            for (int64_t k = 0; k < n; ++k) {
                rs += dec_h(ho[mm * n * n + i * n + k]);
                cs += dec_h(ho[mm * n * n + k * n + i]);
            }
            double e1 = (rs - 1.0) < 0 ? (1.0 - rs) : (rs - 1.0);
            double e2 = (cs - 1.0) < 0 ? (1.0 - cs) : (cs - 1.0);
            if (e1 > max_err) max_err = e1;
            if (e2 > max_err) max_err = e2;
        }
    }
    printf("[check] 双随机性：行和/列和与 1.0 的最大偏差 = %.3e\n", max_err);

    // ---- 逐矩阵列出异常者及其行和（定位失败模式）----
    {
        printf("[bad] 逐矩阵行和（偏差 > 1e-2 才列出）:\n");
        int shown = 0;
        for (int64_t mm = 0; mm < batch && shown < 6; ++mm) {
            double worst = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                double rs = 0.0;
                for (int64_t k = 0; k < n; ++k) {
                    rs += dec_h(ho[mm * n * n + i * n + k]);
                }
                const double e = fabs(rs - 1.0);
                if (e > worst) worst = e;
            }
            if (worst > 1e-2) {
                printf("   矩阵 %-4ld 最大行和偏差 %.4f | 行和:", (long)mm, worst);
                for (int64_t i = 0; i < n; ++i) {
                    double rs = 0.0;
                    for (int64_t k = 0; k < n; ++k) {
                        rs += dec_h(ho[mm * n * n + i * n + k]);
                    }
                    printf(" %.4f", rs);
                }
                printf("\n");
                ++shown;
            }
        }
        if (shown == 0) printf("   （全部矩阵正常）\n");
    }

    // 每个元素与 1/n 的偏差（常数输入时 Sinkhorn 收敛到全 1/n）
    double max_dev = 0.0;
    for (int64_t i = 0; i < batch * n * n; ++i) {
        double d = dec_h(ho[i]) - 1.0 / (double)n;
        if (d < 0) d = -d;
        if (d > max_dev) max_dev = d;
    }
    printf("[check] 与 1/n=%.5f 的最大偏差 = %.3e  %s\n", 1.0 / (double)n, max_dev,
           (max_dev < 5e-3) ? "<<< PASS" : "<<< FAIL");

    // ---- 导出原始结果，供外部（Python）独立校验；fp16 文件名与字节格式保持历史口径 ----
    {
        FILE *fp = fopen(f32 ? "/tmp/mhc_out_f32.bin" : "/tmp/mhc_out.bin", "wb");
        if (fp) {
            fwrite(ho.data(), esz, (size_t)total, fp);
            fclose(fp);
            printf("[dump] 原始 %s 已写入 %s (%ld 个)\n", f32 ? "fp32" : "fp16",
                   f32 ? "/tmp/mhc_out_f32.bin" : "/tmp/mhc_out.bin", (long)total);
        }
    }


    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(dx);
    aclrtFree(dy);
    if (ws) aclrtFree(ws);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    printf("=== DONE ===\n");
    return 0;
}
