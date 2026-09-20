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
 * \file kernel_operator_vec_brcb_intf.h
 * \brief
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_BRCB_INTF_H__
#endif

#ifndef ASCENDC_MODULE_OPERATOR_VEC_BRCB_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_BRCB_INTERFACE_H

#include "kernel_macros.h"
#include "kernel_tensor.h"
#include "kernel_struct_brcb.h"

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include <cstdint>
#include "stub_def.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/*
 * @ingroup brcb Level 0
 * @brief this function fetches 8 b16/b32 data from src0, broadcast each data into one 32B block,
 * @brief then finally writes these 8 blocks into dst continously.
 * @brief gather element in the uint of block
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 */
template <typename T>
__aicore__ inline void Brcb(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const uint8_t repeatTime,
    const BrcbRepeatParams& repeatParams);
} // namespace AscendC
#pragma end_pipe

#include "../../impl/basic_api/kernel_operator_vec_brcb_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_VEC_BRCB_INTERFACE_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_BRCB_INTF_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_BRCB_INTF_H__
#endif
