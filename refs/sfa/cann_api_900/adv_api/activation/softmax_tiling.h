/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file softmax_tiling.h
 * \brief
 */

#ifndef LIB_SOFTMAX_SOFTMAX_TILING_H
#define LIB_SOFTMAX_SOFTMAX_TILING_H
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "softmax_tilingdata.h"
#include "kernel_tiling/kernel_tiling.h"
namespace AscendC {
/*
 * @ingroup GetSoftMaxMaxTmpSize
 * @brief get softmax api calculate need max temporary local space size
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize : input dstMax Tensor and expSum Tensor DType size
 * @param [in] isReuseSource : whether to reuse the src Tensor
 * @return max temporary local space size
 */
uint32_t GetSoftMaxMaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isReuseSource);
/*
 * @ingroup GetSoftMaxMinTmpSize
 * @brief get softmax api calculate need min temporary local space size
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize : input dstMax Tensor and expSum Tensor DType size
 * @param [in] isReuseSource : whether to reuse the src Tensor
 * @return min temporary local space size
 */
uint32_t GetSoftMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isReuseSource);
/*
 * @ingroup SoftMaxTilingFunc
 * @brief calculate SoftMax api need tiling
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize : input dstMax Tensor and expSum Tensor DType size
 * @param [in] localWorkSpaceSize : the temporary local space size for SoftMax api, unit is Byte
 * @param [out] softmaxTiling : SoftMax api tiling
 */
void SoftMaxTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
    optiling::SoftMaxTiling& softmaxTiling);

void SoftMaxTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
    AscendC::tiling::SoftMaxTiling& softmaxTiling);
/*
 * @ingroup GetSoftMaxFlashV3MaxMinTmpSize
 * @brief calculate SoftmaxFlashV3 api need min/max temporary local space size
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize1 : input src Tensor and dst Tensor DType size
 * @param [in] dataTypeSize2 : input dstMax Tensor and expSum Tensor DType size
 * @param [in] isUpdate : whether to enable the flash calculation
 * @param [in] isBasicBlock : whether enable basicblock calculate
 */
void GetSoftMaxFlashV3MaxMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1,
    const uint32_t dataTypeSize2, uint32_t& maxValue, uint32_t& minValue, const bool isUpdate,
    const bool isBasicBlock = false);
/*
 * @ingroup SoftMaxFlashV3TilingFunc
 * @brief calculate SoftmaxFlashV3 api need tiling
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize1 : input src Tensor and dst Tensor DType size
 * @param [in] dataTypeSize2 : input dstMax Tensor and expSum Tensor DType size
 * @param [in] localWorkSpaceSize : the temporary local space size for SoftmaxFlashV3 api, unit is Byte
 * @param [in] isUpdate : whether to enable the flash calculation
 * @param [in] isBasicBlock : whether enable basicblock calculate
 * @param [out] softmaxFlashTiling : SoftmaxFlashV3 api tiling
 */
void SoftMaxFlashV3TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2,
    const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxFlashV3Tiling, const bool isUpdate,
    const bool isBasicBlock = false);

void SoftMaxFlashV3TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2,
    const uint32_t localWorkSpaceSize, AscendC::tiling::SoftMaxTiling& softmaxFlashV3Tiling, const bool isUpdate,
    const bool isBasicBlock = false);
/*
 * @ingroup GetSoftMaxFlashMaxTmpSize
 * @brief calculate SoftmaxFlash api need max temporary local space size
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize : input dstMax Tensor and expSum Tensor DType size
 * @param [in] isUpdate : whether to enable the flash calculation
 * @param [in] isReuseSource : whether to reuse the src Tensor
 * @return max temporary local space size
 */
uint32_t GetSoftMaxFlashMaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isUpdate,
    const bool isReuseSource);
/*
 * @ingroup GetSoftMaxFlashMinTmpSize
 * @brief calculate SoftmaxFlash api need min temporary local space size
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize : input dstMax Tensor and expSum Tensor DType size
 * @param [in] isUpdate : whether to enable the flash calculation
 * @param [in] isReuseSource : whether to reuse the src Tensor
 * @return min temporary local space size
 */
uint32_t GetSoftMaxFlashMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isUpdate,
    const bool isReuseSource);
/*
 * @ingroup SoftMaxFlashTilingFunc
 * @brief calculate SoftmaxFlash api need tiling
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize : input dstMax Tensor and expSum Tensor DType size
 * @param [in] localWorkSpaceSize : the temporary local space size for SoftmaxFlash api, unit is Byte
 * @param [in] isUpdate : whether to enable the flash calculation
 * @param [out] softmaxFlashTiling : SoftmaxFlash api tiling
 */
void SoftMaxFlashTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
    optiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate = false);

void SoftMaxFlashTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
    AscendC::tiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate = false);

/*
 * @ingroup GetSoftMaxGradMaxTmpSize
 * @brief get SoftmaxGrad api need max temporary local space size
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize : input dstMax Tensor and expSum Tensor DType size
 * @param [in] isReuseSource : whether to reuse the src Tensor
 * @param [in] isFront : whether to enable the front calculation
 * @return max temporary local space size
 */
uint32_t GetSoftMaxGradMaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isFront,
    const bool isReuseSource);
/*
 * @ingroup GetSoftMaxGradMinTmpSize
 * @brief get SoftmaxGrad api need min temporary local space size
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize : input dstMax Tensor and expSum Tensor DType size
 * @param [in] isReuseSource : whether to reuse the src Tensor
 * @param [in] isFront : whether to enable the front calculation
 * @return min temporary local space size
 */
uint32_t GetSoftMaxGradMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isFront,
    const bool isReuseSource);
/*
 * @ingroup SoftMaxGradTilingFunc
 * @brief calculate SoftmaxGrad api need tiling
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize : input dstMax Tensor and expSum Tensor DType size
 * @param [in] localWorkSpaceSize : the temporary local space size for SoftmaxGrad api, unit is Byte
 * @param [in] isFront : whether to enable the front calculation
 * @param [out] softmaxGradTiling : SoftmaxGrad api tiling
 */
void SoftMaxGradTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
    optiling::SoftMaxTiling& softmaxGradTiling, const bool isFront = false);

void SoftMaxGradTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
    AscendC::tiling::SoftMaxTiling& softmaxGradTiling, const bool isFront = false);
/*
 * @ingroup IsBasicBlockInSoftMax
 * @brief judge tiling is basicBlock or not
 * @param [in] tiling : input softmax series api tiling
 * @param [in] dataTypeSize : input src Tensor and dst Tensor DType size
 * @return [bool] true is basicBlock, else not
 */
bool IsBasicBlockInSoftMax(optiling::SoftMaxTiling& tiling, const uint32_t dataTypeSize = 2);

bool IsBasicBlockInSoftMax(AscendC::tiling::SoftMaxTiling& tiling, const uint32_t dataTypeSize = 2);
/*
 * @ingroup GetSoftMaxFlashV2MinTmpSize
 * @brief get SoftmaxFlashV2 api need min temporary local space size
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize1 : input src Tensor and dst Tensor DType size
 * @param [in] dataTypeSize2 : input dstMax Tensor and expSum Tensor DType size
 * @param [in] isUpdate : whether to enable the flash calculation
 * @param [in] isBasicBlock : whether enable basicblock calculate
 * @param [in] isFlashOutputBrc : whether enable output data broadcast
 * @return min temporary local space size
 */
uint32_t GetSoftMaxFlashV2MinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1,
    const uint32_t dataTypeSize2, const bool isUpdate, const bool isBasicBlock = false,
    const bool isFlashOutputBrc = false);
/*
 * @ingroup GetSoftMaxFlashV2MaxTmpSize
 * @brief get SoftmaxFlashV2 api need max temporary local space size
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize1 : input src Tensor and dst Tensor DType size
 * @param [in] dataTypeSize2 : input dstMax Tensor and expSum Tensor DType size
 * @param [in] isUpdate : whether to enable the flash calculation
 * @param [in] isBasicBlock : whether enable basicblock calculate
 * @param [in] isFlashOutputBrc : whether enable output data broadcast
 * @return max temporary local space size
 */
uint32_t GetSoftMaxFlashV2MaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1,
    const uint32_t dataTypeSize2, const bool isUpdate, const bool isBasicBlock = false,
    const bool isFlashOutputBrc = false);
/*
 * @ingroup SoftMaxFlashV2TilingFunc
 * @brief calculate SoftmaxFlashV2 api need tiling
 * @param [in] srcShape : input src Tensor shape
 * @param [in] dataTypeSize1 : input src Tensor and dst Tensor DType size
 * @param [in] dataTypeSize2 : input dstMax Tensor and expSum Tensor DType size
 * @param [in] localWorkSpaceSize : the temporary local space size for SoftmaxFlashV2 api, unit is Byte
 * @param [in] isUpdate : whether to enable the flash calculation
 * @param [in] isBasicBlock : whether enable basicblock calculate
 * @param [in] isFlashOutputBrc : whether enable output data broadcast
 * @param [out] softmaxFlashTiling : SoftmaxFlashV2 api tiling
 */
void SoftMaxFlashV2TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2,
    const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate,
    const bool isBasicBlock = false, const bool isFlashOutputBrc = false);

void SoftMaxFlashV2TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2,
    const uint32_t localWorkSpaceSize, AscendC::tiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate,
    const bool isBasicBlock = false, const bool isFlashOutputBrc = false);
}
#endif // LIB_SOFTMAX_SOFTMAX_TILING_H