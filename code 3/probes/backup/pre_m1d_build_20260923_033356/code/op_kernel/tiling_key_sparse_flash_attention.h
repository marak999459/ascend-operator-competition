// TilingKey模板定义的头文件
// 与原骨架写法一致（DECL 与 SEL 各一块，见 HANDOFF.md §七：别在不同算子间混用写法）
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(SparseFlashAttention,
    ASCENDC_TPL_DATATYPE_DECL(DT_QUERY, C_DT_FLOAT, C_DT_FLOAT16),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_QUERY, C_DT_FLOAT, C_DT_FLOAT16),
    ),
);
