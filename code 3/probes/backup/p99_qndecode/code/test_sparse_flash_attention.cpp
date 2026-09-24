// CPU 仿真测试程序：SparseFlashAttention（最简版）
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <sys/mman.h>

// 必须先包含 tikicpulib
#include "tikicpulib.h"

// 提供 libcpudebug 中缺失的符号 stub
#include <string>
namespace AscendC {
    std::string GetCoreName(int idx) { return "Core" + std::to_string(idx); }
}
static std::vector<std::string> g_tmpFileNames;
std::vector<std::string>& GetTmpFileName() { return g_tmpFileNames; }

// pv_* 仿真器调试接口 stub
extern "C" {
    void pv_init() {}
    void pv_step() {}
    void pv_mem_read(uint64_t addr, void* buf, size_t len) {}
    void pv_mem_write(uint64_t addr, const void* buf, size_t len) {}
    void pv_reg_read(int reg, void* val) {}
    void pv_reg_write(int reg, const void* val) {}
    void pv_launch_sub_core(int core_id) {}
    void set_read_record(int enable) {}
}

#ifndef C_DT_FLOAT
#define C_DT_FLOAT 0
#endif
#ifndef C_DT_FLOAT16
#define C_DT_FLOAT16 1
#endif

#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(T, name, ptr) \
    const T& name = *reinterpret_cast<const T*>(ptr)
#endif

#include "sparse_flash_attention.cpp"

// 共享内存分配
static void* shared_alloc(size_t size) {
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    memset(p, 0, size);
    return p;
}
static void shared_free(void* p, size_t size) { munmap(p, size); }

// 参考实现
template<typename T>
void reference_sparse_flash_attention(
    const T* query, const T* key, const T* value,
    const int32_t* sparse_indices,
    T* output,
    int B, int Q_S, int KV_S, int Q_N, int Q_D,
    int sparse_size, float scale) {
    
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < Q_S; s++) {
            // 读 sparse_indices
            std::vector<int> indices;
            for (int i = 0; i < sparse_size; i++) {
                int idx = sparse_indices[(int64_t)b * Q_S * 1 * sparse_size + (int64_t)s * 1 * sparse_size + i];
                if (idx >= 0 && idx < KV_S) {
                    indices.push_back(idx);
                }
            }
            int valid_count = indices.size();
            
            // 读 query: (Q_N, Q_D)
            std::vector<float> q(Q_N * Q_D);
            for (int n = 0; n < Q_N; n++) {
                for (int d = 0; d < Q_D; d++) {
                    q[n * Q_D + d] = (float)query[(int64_t)b * Q_S * Q_N * Q_D + (int64_t)s * Q_N * Q_D + (int64_t)n * Q_D + d];
                }
            }
            
            // Gather K̃ 和 Ṽ: (valid_count, Q_D)
            std::vector<float> k_sel(valid_count * Q_D);
            std::vector<float> v_sel(valid_count * Q_D);
            for (int i = 0; i < valid_count; i++) {
                for (int d = 0; d < Q_D; d++) {
                    k_sel[i * Q_D + d] = (float)key[(int64_t)b * KV_S * 1 * Q_D + (int64_t)indices[i] * 1 * Q_D + d];
                    v_sel[i * Q_D + d] = (float)value[(int64_t)b * KV_S * 1 * Q_D + (int64_t)indices[i] * 1 * Q_D + d];
                }
            }
            
            // 计算 score = Q @ K̃^T * scale: (Q_N, valid_count)
            std::vector<float> score(Q_N * valid_count);
            for (int n = 0; n < Q_N; n++) {
                for (int i = 0; i < valid_count; i++) {
                    float s = 0.0f;
                    for (int d = 0; d < Q_D; d++) {
                        s += q[n * Q_D + d] * k_sel[i * Q_D + d];
                    }
                    score[n * valid_count + i] = s * scale;
                }
            }
            
            // Softmax: 数值稳定
            std::vector<float> attn(Q_N * valid_count);
            for (int n = 0; n < Q_N; n++) {
                float mx = score[n * valid_count];
                for (int i = 1; i < valid_count; i++) {
                    mx = std::max(mx, score[n * valid_count + i]);
                }
                float sum = 0.0f;
                for (int i = 0; i < valid_count; i++) {
                    float e = expf(score[n * valid_count + i] - mx);
                    attn[n * valid_count + i] = e;
                    sum += e;
                }
                float inv_sum = 1.0f / (sum + 1e-6f);
                for (int i = 0; i < valid_count; i++) {
                    attn[n * valid_count + i] *= inv_sum;
                }
            }
            
            // 计算 out = attn @ Ṽ: (Q_N, Q_D)
            std::vector<float> out(Q_N * Q_D);
            for (int n = 0; n < Q_N; n++) {
                for (int d = 0; d < Q_D; d++) {
                    float s = 0.0f;
                    for (int i = 0; i < valid_count; i++) {
                        s += attn[n * valid_count + i] * v_sel[i * Q_D + d];
                    }
                    out[n * Q_D + d] = s;
                }
            }
            
            // 写回输出
            for (int n = 0; n < Q_N; n++) {
                for (int d = 0; d < Q_D; d++) {
                    output[(int64_t)b * Q_S * Q_N * Q_D + (int64_t)s * Q_N * Q_D + (int64_t)n * Q_D + d] = (T)out[n * Q_D + d];
                }
            }
        }
    }
}

// 测试用例
void test_case(int B, int Q_S, int KV_S, int Q_N, int Q_D, int sparse_size, float scale) {
    printf("Test: B=%d, Q_S=%d, KV_S=%d, Q_N=%d, Q_D=%d, sparse_size=%d\n",
           B, Q_S, KV_S, Q_N, Q_D, sparse_size);
    
    // 分配输入
    int64_t q_size = (int64_t)B * Q_S * Q_N * Q_D;
    int64_t kv_size = (int64_t)B * KV_S * 1 * Q_D;
    int64_t idx_size = (int64_t)B * Q_S * 1 * sparse_size;
    
    float* query = (float*)shared_alloc(q_size * sizeof(float));
    float* key = (float*)shared_alloc(kv_size * sizeof(float));
    float* value = (float*)shared_alloc(kv_size * sizeof(float));
    int32_t* sparse_indices = (int32_t*)shared_alloc(idx_size * sizeof(int32_t));
    float* output_kernel = (float*)shared_alloc(q_size * sizeof(float));
    float* output_ref = (float*)shared_alloc(q_size * sizeof(float));
    
    // 初始化输入
    srand(42);
    for (int64_t i = 0; i < q_size; i++) {
        query[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    for (int64_t i = 0; i < kv_size; i++) {
        key[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        value[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < Q_S; s++) {
            for (int i = 0; i < sparse_size; i++) {
                sparse_indices[(int64_t)b * Q_S * 1 * sparse_size + (int64_t)s * 1 * sparse_size + i] = rand() % KV_S;
            }
        }
    }
    
    // 调用 kernel
    SparseFlashAttentionTilingData tiling;
    tiling.B = B;
    tiling.Q_S = Q_S;
    tiling.KV_S = KV_S;
    tiling.Q_N = Q_N;
    tiling.Q_D = Q_D;
    tiling.Dr = 64;
    tiling.sparse_size = sparse_size;
    tiling.scale_value = scale;
    tiling.sparse_mode = 0;
    tiling.batch_per_core = B;  // 单 core
    
    // 调用 kernel（CPU 仿真）
    sparse_flash_attention<float>(
        (GM_ADDR)query, (GM_ADDR)key, (GM_ADDR)value,
        (GM_ADDR)sparse_indices, nullptr, nullptr,
        nullptr, nullptr,  // query_rope, key_rope
        (GM_ADDR)output_kernel, nullptr, nullptr,
        nullptr, (GM_ADDR)&tiling);
    
    // 调用参考实现
    reference_sparse_flash_attention<float>(
        query, key, value, sparse_indices, output_ref,
        B, Q_S, KV_S, Q_N, Q_D, sparse_size, scale);
    
    // 比较结果
    float max_err = 0.0f;
    int bad_count = 0;
    for (int64_t i = 0; i < q_size; i++) {
        float err = fabsf(output_kernel[i] - output_ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-3f) bad_count++;
    }
    
    printf("  Max error: %.6f, Bad count: %ld / %ld\n", max_err, bad_count, q_size);
    if (bad_count == 0) {
        printf("  PASS!\n");
    } else {
        printf("  FAIL!\n");
    }
    
    // 释放
    shared_free(query, q_size * sizeof(float));
    shared_free(key, kv_size * sizeof(float));
    shared_free(value, kv_size * sizeof(float));
    shared_free(sparse_indices, idx_size * sizeof(int32_t));
    shared_free(output_kernel, q_size * sizeof(float));
    shared_free(output_ref, q_size * sizeof(float));
}

int main() {
    printf("=== SparseFlashAttention CPU 仿真测试 ===\n\n");
    
    // 测试用例
    test_case(1, 2, 4, 2, 64, 2, 1.0f / sqrtf(64));
    test_case(1, 4, 8, 4, 64, 4, 1.0f / sqrtf(64));
    test_case(2, 4, 8, 2, 32, 4, 1.0f / sqrtf(32));
    
    printf("\n=== 测试完成 ===\n");
    return 0;
}
