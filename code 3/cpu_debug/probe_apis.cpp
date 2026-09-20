/*
 * Probe kernel for problem 3 (SFA): compile-only adjudication of Ascend C APIs
 * on ascend910b / CANN 9.0.0, driven through the CPU-side compile gate
 * (code3.md 13.6). It is never executed -- it only has to TYPE-CHECK, so the
 * LocalTensors are left default-constructed on purpose.
 *
 * Entry signature mirrors code 3/code/op_kernel/sparse_flash_attention.cpp:486
 * so the build system generates the same kernel variants.
 * ASCII only (code3.md 4.4).
 */
#include "kernel_operator.h"
#include "sparse_flash_attention_tiling.h"
#include "tiling_key_sparse_flash_attention.h"

using namespace AscendC;

/* same shape the official arch22 SFA uses (cann_builtin_900/...common.h:25) */
constexpr SoftmaxConfig PROBE_SFA_CFG = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

__aicore__ inline void ProbeVectorOps()
{
    LocalTensor<half> h0, h1;
    LocalTensor<float> f0;
    LocalTensor<int32_t> i0;
    half hv;
    uint64_t mcount = 0xFFFFFFFFFFFFULL;
    uint64_t marr[2] = {0xFFFFFFFFFFFFULL, 0ULL};
    /* kernel_struct_unary.h: only a 4-arg {dstBlkStride, srcBlkStride, dstRepStride, srcRepStride}
       and a 5-arg (+halfBlock) ctor exist -- there is no 2-arg form */
    UnaryRepeatParams up{1, 1, 0, 0};
    BrcbRepeatParams bp{1, 8};

    /* 910B branch of vec_unary_intf.h: (dst, src, mask[], mask, uint8_t repeatTime, params) */
    Exp<half>(h0, h1, marr, 8, up);
    Exp<half>(h0, h1, mcount, 8, up);
    Reciprocal<half>(h0, h1, mcount, 8, up);
    Muls<half>(h0, h1, hv, mcount, 8, up);
    Cast<float>(f0, h1, RoundMode::CAST_NONE, mcount, 8, up);
    Cast<int32_t>(i0, h1, RoundMode::CAST_FLOOR, mcount, 8, up);

    /* Brcb: repeatTime is uint8_t, params are {blkStride, repStride} -- official uses {1, 8} */
    Brcb<half>(h0, h1, (uint8_t)8, bp);

    /* the old 11.4 open question: WholeReduceSum with explicit strides */
    WholeReduceSum<half>(h0, h1, marr, 8, 1, 1, 1);
}

__aicore__ inline void ProbeCrossCore()
{
    /* basic_api/kernel_operator_block_sync_intf.h:87-90 */
    CrossCoreSetFlag<2, PIPE_FIX>((uint16_t)0);
    CrossCoreWaitFlag<2, PIPE_FIX>((uint16_t)0);
}

__aicore__ inline void ProbeSoftmaxFlashV2()
{
    LocalTensor<half> h0, h1;
    LocalTensor<uint8_t> u8;
    SoftMaxTiling tl;
    SoftMaxShapeInfo si{8, 16, 8, 16};

    /* form A: 9 args, no shared tmp buffer (softmaxflashv2.h:79) */
    SoftmaxFlashV2<half, true, true, false, false, PROBE_SFA_CFG>(
        h0, h0, h0, h1, h0, h0, h0, tl, si);

    /* form B: verbatim arg count used by the built-in arch22 SFA source
       (cann_builtin_900/..._service_vector_mla.h:540) -- 10 args */
    SoftmaxFlashV2<half, true, true, false, false, PROBE_SFA_CFG>(
        h0, h0, h0, h1, h0, h0, h0, u8, tl, si);

    /* form C: 11 args, with outReduceMax + shared tmp (softmaxflashv2.h:291) */
    SoftmaxFlashV2<half, true, true, false, false, PROBE_SFA_CFG>(
        h0, h0, h0, h0, h1, h0, h0, h0, u8, tl, si);
}

template <typename DT_QUERY>
__global__ __aicore__ void sparse_flash_attention(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparseIndices,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKV,
    GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR attentionOut,
    GM_ADDR softmaxMaxOut, GM_ADDR softmaxSumOut, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SparseFlashAttentionTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionTilingData, tiling_data, tiling);
    /* opaque condition: keeps every call alive through instantiation without
       pretending the tensors hold valid data */
    if (workspace != nullptr && tiling_data.B > 0xFFFFFFFFu) {
        ProbeVectorOps();
        ProbeCrossCore();
        ProbeSoftmaxFlashV2();
    }
}

template __aicore__ void sparse_flash_attention<half>(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template __aicore__ void sparse_flash_attention<float>(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
