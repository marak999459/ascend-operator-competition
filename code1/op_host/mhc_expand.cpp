// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

#include "../op_kernel/mhc_expand_tiling.h"
#include "../op_kernel/tiling_key_mhc_expand.h"

namespace {
// 切分模式枚举（与 TilingData.splitMode 一致）
constexpr uint32_t SPLIT_ROW = 0;          // 按 token 行切核（首选）
constexpr uint32_t SPLIT_ROW_STREAM = 1;   // 前向：把 (s, k) 展开为 S*m 个写任务
constexpr uint32_t SPLIT_ELEMENT = 2;      // 极端小 shape：按 (s, jt) 扁平切
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // ---- 平台信息 ----
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        if (ub_size == 0) {
            ub_size = 192 * 1024;   // 防御：默认 910B UB 约 192KB
        }

        // ---- 输入信息 ----
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        const gert::StorageShape *in_storage_shape = context->GetInputShape(0);
        const gert::Shape &in_shape = in_storage_shape->GetStorageShape();
        size_t in_rank = in_shape.GetDimNum();

        // ---- 算子属性 ----
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_mhc_mult = attrs->GetInt(0);
        const bool *attr_backward = attrs->GetBool(1);
        int64_t m = (attr_mhc_mult != nullptr) ? *attr_mhc_mult : 2;
        bool backward = (attr_backward != nullptr) ? *attr_backward : false;

        if (m < 1) {
            return ge::GRAPH_FAILED;   // 扩展倍数必须 >= 1
        }

        // ---- 语义消歧：按 backward + rank 解析 S / D ----
        uint32_t S = 0, D = 0;
        if (!backward) {
            if (in_rank != 2) return ge::GRAPH_FAILED;   // 前向期望 [S, D]
            S = static_cast<uint32_t>(in_shape.GetDim(0));
            D = static_cast<uint32_t>(in_shape.GetDim(1));
        } else {
            if (in_rank != 3) return ge::GRAPH_FAILED;   // 反向期望 [S, m, D]
            if (static_cast<int64_t>(in_shape.GetDim(1)) != m) return ge::GRAPH_FAILED;
            S = static_cast<uint32_t>(in_shape.GetDim(0));
            D = static_cast<uint32_t>(in_shape.GetDim(2));
        }
        if (S == 0 || D == 0) return ge::GRAPH_FAILED;

        // ---- 配置 tiling key（dtype 模板域：fp16 / bf16；backward 布尔模板域） ----
        uint32_t DT_X = 0;
        if (dtype_x == ge::DT_FLOAT16) {
            DT_X = C_DT_FLOAT16;
        } else if (dtype_x == ge::DT_BF16) {
            DT_X = C_DT_BF16;
        } else {
            return ge::GRAPH_FAILED;
        }
        uint32_t BACKWARD = backward ? 1u : 0u;
        ASCENDC_TPL_SEL_PARAM(context, DT_X, BACKWARD);

        // ---- 填充 TilingData ----
        MhcExpandTilingData *tiling = context->GetTilingData<MhcExpandTilingData>();
        tiling->S = S;
        tiling->D = D;
        tiling->m = static_cast<uint32_t>(m);
        tiling->backward = backward ? 1u : 0u;

        // D 方向大 tile：32B 对齐（fp16/bf16 = 16 元素），典型 512~2048；
        // 前向攒批路径在 UB 里同时握 2*BS 份 tile 做环（见 op_kernel 的 fwd_b0_~b7_），
        // kernel 侧按"环占用字节"反推 BS：整行且 tile<=6KB 用 8 格、tile>=12KB 用 4 格（R10）。
        // 所以这个 /4 是**环占用的字节上限**：放宽预算等于放弃前向 barrier 批处理，两者必须一起改。
        const uint64_t elem_size = static_cast<uint64_t>(dtype_size_x);
        const uint64_t ub_budget = ub_size / 4;
        uint32_t d_tile_len = 0;
        if (static_cast<uint64_t>(D) * elem_size <= ub_budget) {
            d_tile_len = D;                              // 整行模式：dTileNum = 1
        } else {
            uint64_t t = std::min<uint64_t>(2048, D);
            uint64_t max_t = ub_budget / elem_size;
            t = std::min(t, max_t);
            if (t > 16) t = (t / 16) * 16;               // 32B 对齐
            if (t < 16) t = 16;
            if (t > D) t = D;
            if (t < 1) t = 1;                            // 极端兜底
            d_tile_len = static_cast<uint32_t>(t);
        }
        tiling->dTileLen = d_tile_len;
        tiling->dTileNum = (D + d_tile_len - 1) / d_tile_len;
        tiling->dTailLen = D - (tiling->dTileNum - 1) * d_tile_len;

        // ---- 切分决策（"切分最优"得分点，DESIGN.md §3.5） ----
        uint32_t num_aiv = static_cast<uint32_t>(num_cores_aiv);
        if (num_aiv == 0) num_aiv = 1;
        uint64_t total_tasks = 0;
        uint32_t split_mode = SPLIT_ROW;
        uint32_t block_dim = num_aiv;

        if (S >= num_aiv) {
            // 行数足够 → 按 token 行切核（首选，每核负责连续若干整行）
            split_mode = SPLIT_ROW;
            total_tasks = S;
        } else if (!backward) {
            uint64_t stream_tasks = static_cast<uint64_t>(S) * m;
            if (stream_tasks >= num_aiv) {
                // 前向 S 小：把 (s, k) 展开为 S*m 个写任务，充分用核
                split_mode = SPLIT_ROW_STREAM;
                total_tasks = stream_tasks;
            } else {
                // 前向极端小 shape：按 (s, k, jt) 扁平切
                split_mode = SPLIT_ELEMENT;
                total_tasks = static_cast<uint64_t>(S) * m * tiling->dTileNum;
            }
        } else {
            // 反向小 S：按 (s, jt) 切（每个输出元素由唯一核负责，免跨核归约，无需 workspace）
            split_mode = SPLIT_ELEMENT;
            total_tasks = static_cast<uint64_t>(S) * tiling->dTileNum;
        }

        if (total_tasks < num_aiv) block_dim = static_cast<uint32_t>(total_tasks);
        if (block_dim == 0) block_dim = 1;

        // ---- 小档少开核（真机实测 2026-09-20 §14 定则，2026-09-21 R11 §19 复核并分向）----
        // 每多启动一个 block 要多付一次派发（R11-B 用空 kernel 直接量出这条地板：
        // 固定 1.2us + 每核约 75~90ns），所以"每核分到的字节"低于拐点时开核反而更慢：
        //   fwd-fp16-small(96KB) blk=40 -> 4.5us，blk=16 -> 3.0us，blk=12 -> 2.9us，blk=4 -> 3.9us
        //   bwd-fp16-small(96KB) blk=40 -> 4.8us，blk=24 -> 3.7us，blk=16 -> 3.8us，blk=8 -> 4.3us
        // 前向 IO = S*D + S*m*D、反向同样 = (1+m)*S*D（谁当输入不影响总量），故一条式子覆盖两向。
        // 拐点**分向**取：前向在 R11 把 barrier 按块摊薄（整行小 tile 攒 4 块）之后谷底左移到
        // 8KB/核 = 96KB/12，c0/c2 各三轮交替读到 2.6us（同形 BS=1@16 为 3.0us）；反向不经攒批
        // 那条路，谷底仍在 6KB/核。medium(42MB)/large(8.4GB) 远不到拐点，仍按 num_aiv 满开核。
        const uint64_t io_bytes = static_cast<uint64_t>(m + 1) * static_cast<uint64_t>(S) *
                                  static_cast<uint64_t>(D) * elem_size;
        const uint64_t min_io_per_core = backward ? 6144 : 8192;   // 反向 6KB/核、前向 8KB/核
        uint64_t core_cap = io_bytes / min_io_per_core;
        if (core_cap < 1) core_cap = 1;
        if (core_cap < block_dim) block_dim = static_cast<uint32_t>(core_cap);


        tiling->splitMode = split_mode;
        tiling->blockDim = block_dim;
        tiling->rowsPerCore = static_cast<uint32_t>((total_tasks + block_dim - 1) / block_dim);
        // 最后一个核的任务数（uint64 计算避免溢出）
        uint64_t tail = total_tasks - static_cast<uint64_t>(tiling->rowsPerCore) * (block_dim - 1);
        tiling->tailRows = static_cast<uint32_t>(tail);

        // ---- 启动核数与 workspace ----
        context->SetBlockDim(block_dim);
        size_t *current_workspace = context->GetWorkspaceSizes(1);
        current_workspace[0] = 0;   // 本实现不需要 workspace
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *in_shape = context->GetInputShape(0);
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_mhc_mult = attrs->GetInt(0);
        const bool *attr_backward = attrs->GetBool(1);
        int64_t m = (attr_mhc_mult != nullptr) ? *attr_mhc_mult : 2;
        bool backward = (attr_backward != nullptr) ? *attr_backward : false;

        size_t in_rank = in_shape->GetDimNum();
        gert::Shape *out_shape = context->GetOutputShape(0);
        if (!backward) {
            // 前向： [S, D] -> [S, m, D]
            if (in_rank != 2 || m < 1) return GRAPH_FAILED;
            out_shape->SetDimNum(3);
            out_shape->SetDim(0, in_shape->GetDim(0));
            out_shape->SetDim(1, m);
            out_shape->SetDim(2, in_shape->GetDim(1));
        } else {
            // 反向： [S, m, D] -> [S, D]
            if (in_rank != 3 || m < 1) return GRAPH_FAILED;
            if (in_shape->GetDim(1) != m) return GRAPH_FAILED;
            out_shape->SetDimNum(2);
            out_shape->SetDim(0, in_shape->GetDim(0));
            out_shape->SetDim(1, in_shape->GetDim(2));
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        ge::DataType in_dtype = context->GetInputDataType(0);   // 返回值类型是 ge::DataType（值）
        context->SetOutputDataType(0, in_dtype);                // 参数类型也是 ge::DataType（值）
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class MhcExpand : public OpDef {
    public:
        explicit MhcExpand(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("o")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("mhc_mult").AttrType(OPTIONAL).Int(2);
            this->Attr("backward").AttrType(OPTIONAL).Bool(false);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(MhcExpand);
}  // namespace ops
