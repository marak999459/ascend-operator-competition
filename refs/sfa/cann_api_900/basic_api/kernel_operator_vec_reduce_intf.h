/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_operator_vec_reduce_intf.h
 * \brief
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_REDUCE_INTF_H__
#endif

#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCE_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCE_INTERFACE_H

#include "kernel_macros.h"
#include "kernel_tensor.h"
#include "utils/kernel_utils_mode.h"

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include <cstdint>
#include "stub_def.h"
#endif

namespace AscendC {
#pragma begin_pipe(V)
/* *************** BlockReduceMax /BlockReduceMin /BlockReduceSum PairReduceSum ********************* */
/*
 * @ingroup BlockReduceSum
 * @brief Sum all elements in each block
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] repeatTime repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t repeatTime, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

/*
 * @ingroup BlockReduceMax
 * @brief Maximize all elements in each block
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] repeatTime repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t repeatTime, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

/*
 * @ingroup BlockReduceMin
 * @brief Find the minimum value of all elements in each block
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] repeatTime repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t repeatTime, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

/*
 * @ingroup PairReduceSum
 * @brief Sum of adjacent inner pair (parity) elements
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] repeatTime repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t repeatTime, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t repeatTime, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t repeatTime, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t repeatTime, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t repeatTime, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

#if (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3113)
template <typename T, bool isSetMask = true, typename U = T>
__aicore__ inline void RepeatReduceSum(const LocalTensor<U>& dst, const LocalTensor<T>& src,
    const int32_t repeatTime, const int32_t mask, const int32_t dstBlkStride, const int32_t srcBlkStride,
    const int32_t dstRepStride, const int32_t srcRepStride);
#else
template <typename T, bool isSetMask = true>
__aicore__ inline void RepeatReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t repeatTime, const int32_t mask, const int32_t dstBlkStride, const int32_t srcBlkStride,
    const int32_t dstRepStride, const int32_t srcRepStride);
#endif

/* **************************************** Whole Reduce Interface ****************************************** */
/*
 * @ingroup WholeReduceSum
 * @brief Sum of all effective elements in each repeatTime
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] repeatTime repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
#if (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3113)
template <typename T, bool isSetMask = true, typename U = T>
__aicore__ inline void WholeReduceSum(const LocalTensor<U>& dst, const LocalTensor<T>& src,
    const uint64_t mask[], const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);
#else
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const uint64_t mask[], const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);
#endif

/*
 * @ingroup WholeReduceMax
 * @brief Index of the maximum value of all elements in each repeat
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] repeatTime repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const uint64_t mask[], const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);

/*
 * @ingroup WholeReduceMin
 * @brief Index of the minimum value of all elements in each repeat
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] repeatTime repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const uint64_t mask[], const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);

#if (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3113)
template <typename T, bool isSetMask = true, typename U = T>
__aicore__ inline void WholeReduceSum(const LocalTensor<U>& dst, const LocalTensor<T>& src,
    const int32_t mask, const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);
#else
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t mask, const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);
#endif

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t mask, const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t mask, const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);

/* **************************************** Reduce Interface ****************************************** */
/*
 * @ingroup ReduceMax Level 0
 * @brief Index of the maximum value of all input elements
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] sharedTmpBuffer LocalTensor to store the intermediate results
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] srcRepStride src repeat stride
 * @param [in] calIndex Specify whether to get the index with the highest value
 */
template <typename T>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const int32_t mask, const int32_t repeatTime, const int32_t srcRepStride,
    bool calIndex = 0);

/*
 * @ingroup ReduceMin
 * @brief Index of the minimum value of all input elements
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] sharedTmpBuffer LocalTensor to store the intermediate results
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] srcRepStride src repeat stride
 * @param [in] calIndex Specify whether to get the index with the highest value
 */
template <typename T>
__aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const int32_t mask, const int32_t repeatTime, const int32_t srcRepStride,
    bool calIndex = 0);

/*
 * @ingroup ReduceSum
 * @brief sum all input elements
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] sharedTmpBuffer LocalTensor to store the intermediate results
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] srcRepStride src repeat stride
 */
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const int32_t mask, const int32_t repeatTime, const int32_t srcRepStride);

template <typename T>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[], const int32_t repeatTime, const int32_t srcRepStride,
    bool calIndex = 0);
template <typename T>
__aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[], const int32_t repeatTime, const int32_t srcRepStride,
    bool calIndex = 0);
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[], const int32_t repeatTime, const int32_t srcRepStride);

/*
 * @ingroup ReduceMin Level 2
 * @brief Index of the minimum value of all input elements
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] sharedTmpBuffer LocalTensor to store the intermediate results
 * @param [in] count Number of data involved in calculation
 * @param [in] calIndex Specify whether to get the index with the highest value
 */
template <typename T>
__aicore__ inline void ReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const int32_t count, bool calIndex = 0);

/*
 * @ingroup ReduceMax Level 2
 * @brief Index of the maximum value of all input elements
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] sharedTmpBuffer LocalTensor to store the intermediate results
 * @param [in] count Number of data involved in calculation
 * @param [in] calIndex Specify whether to get the index with the highest value
 */
template <typename T>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const int32_t count, bool calIndex = 0);

/*
 * @ingroup ReduceSum Level 2
 * @brief sum all input elements
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] sharedTmpBuffer LocalTensor to store the intermediate results
 * @param [in] count Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const int32_t count);
#pragma end_pipe
template <typename T>
__aicore__ inline __inout_pipe__(S) void GetReduceRepeatMaxMinSpr(T &maxMinValue, T &maxMinIndex);

template <typename T>
__aicore__ inline __inout_pipe__(S) void GetReduceRepeatMaxMinSpr(T &maxMinValue);

// GetReduceMaxMinCount has been updated, please use GetReduceRepeatMaxMinSpr instead.
template <typename T>
__aicore__ inline __inout_pipe__(S) void GetReduceMaxMinCount(T &maxMinValue, T &maxMinIndex);

// GetReduceMaxMinCount has been updated, please use GetReduceRepeatMaxMinSpr instead.
template <typename T>
__aicore__ inline __inout_pipe__(S) void GetReduceMaxMinCount(T &maxMinValue);

template <typename T>
__aicore__ inline __inout_pipe__(S) T GetReduceRepeatSumSpr();

// GetAccVal has been updated, please use GetReduceRepeatSumSpr instead.
template <typename T>
__aicore__ inline __inout_pipe__(S) T GetAccVal();
} // namespace AscendC

#include "../../impl/basic_api/kernel_operator_vec_reduce_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_VEC_REDUCE_INTERFACE_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_REDUCE_INTF_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_REDUCE_INTF_H__
#endif
