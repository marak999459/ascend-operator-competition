// One-shot instrument: verify WholeReduceSum/Max on __NPU_ARCH__=2201.
// Lives under refs/harness_sinkhorn/cpu_debug/ on purpose - never in a submission dir.
// Question it answers: does one call reduce n rows, and do the n results land
// packed at dst[0..n-1] (what a single Brcb needs) or one-per-32B-block?
#include <cstdint>
#include <cstdio>
#include <sys/mman.h>

#include "kernel_operator.h"
#include "tikicpulib.h"

using namespace AscendC;

static constexpr int32_t ROWS = 4;
static constexpr int32_t RS = 8;  // one 32B block of fp32 per row, same as the kernel

__global__ __aicore__ void probe_run(GM_ADDR out_sum, GM_ADDR out_max) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> src_buf, dst_sum, dst_max;
    pipe.InitBuffer(src_buf, ROWS * RS * sizeof(float));
    pipe.InitBuffer(dst_sum, ROWS * RS * sizeof(float));
    pipe.InitBuffer(dst_max, ROWS * RS * sizeof(float));

    LocalTensor<float> src = src_buf.Get<float>();
    LocalTensor<float> dsum = dst_sum.Get<float>();
    LocalTensor<float> dmax = dst_max.Get<float>();
    for (int32_t i = 0; i < ROWS * RS; ++i) {
        src.SetValue(i, static_cast<float>(i));
    }

    // (dst, src, mask/count, repeatTime, dstRepStride, srcBlkStride, srcRepStride)
    WholeReduceSum<float>(dsum, src, 4, ROWS, 1, 1, 1);
    WholeReduceMax<float>(dmax, src, 4, ROWS, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();

    GlobalTensor<float> gsum, gmax;
    gsum.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out_sum), ROWS * RS);
    gmax.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out_max), ROWS * RS);
    DataCopyExtParams cp{1, static_cast<uint32_t>(ROWS * RS * sizeof(float)), 0, 0, 0};
    DataCopyPad(gsum, dsum, cp);
    DataCopyPad(gmax, dmax, cp);
    PipeBarrier<PIPE_ALL>();
}

static void *alloc_gm(size_t sz) {
    void *p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    return p;
}

int main() {
    float *os = static_cast<float *>(alloc_gm(4096));
    float *om = static_cast<float *>(alloc_gm(4096));

    typedef void (*ProbeFunc)(GM_ADDR, GM_ADDR);
    ProbeFunc kf = reinterpret_cast<ProbeFunc>(&probe_run);
    ICPU_RUN_KF(kf, 1, (GM_ADDR)os, (GM_ADDR)om);

    // Expected for row i, count=4 lanes: sum = 4*(i*8) + (0+1+2+3) = 32i + 6, max = i*8 + 3.
    printf("[probe] dst_sum[0..7]  =");
    for (int i = 0; i < 8; ++i) printf(" %.1f", os[i]);
    printf("\n[probe] dst_sum[8..31]=");
    for (int i = 8; i < 32; ++i) printf(" %.1f", os[i]);
    printf("\n[probe] dst_max[0..7]  =");
    for (int i = 0; i < 8; ++i) printf(" %.1f", om[i]);
    printf("\n[probe] expect 6 38 70 102 (sum) / 3 11 19 27 (max), either packed or at block heads\n");

    bool sum_pack = (os[0] == 6.f && os[1] == 38.f && os[2] == 70.f && os[3] == 102.f);
    bool blk_head = (os[0] == 6.f && os[8] == 38.f && os[16] == 70.f && os[24] == 102.f);
    printf("[probe] VERDICT sum: packed@dst[0..3]=%s  per-block-head@dst[0,8,16,24]=%s  => %s\n",
           sum_pack ? "YES" : "no", blk_head ? "YES" : "no",
           (sum_pack || blk_head) ? "OK" : "NEITHER - semantics differ, do not adopt");
    printf("[probe] max: packed=%s per-block=%s\n",
           (om[0] == 3.f && om[1] == 11.f && om[2] == 19.f && om[3] == 27.f) ? "YES" : "no",
           (om[0] == 3.f && om[8] == 11.f && om[16] == 19.f && om[24] == 27.f) ? "YES" : "no");
    return 0;
}
