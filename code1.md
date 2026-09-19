# code1 —— 第一题 `mhc_expand`（前向与反向）

> **本文件随题目推进持续更新**：状态 / 已排除 / 下一步证据。
> 通用流程与通用坑见根目录 `算子开发工作流.md`；连接参数见根目录 `连接信息.md`。
> ⛔ 本题目录 `code1/` 与其它两题**严禁跨界**。

> **相关文档**：[AGENT.MD](AGENT.MD)（入口） · [算子开发工作流.md](算子开发工作流.md)（通用流程/坑/纪律） · [连接信息.md](连接信息.md)（连接参数） · [official_problem_statement.md](official_problem_statement.md#第一题b组简单题mhc-expand-算子前向与反向)（**本题官方题面**） · [code2.md](code2.md) · [code3.md](code3.md) · [refs/README.md](refs/README.md)

---
## 0. 工作约束（**必须遵守**）

| # | 约束 |
|---|---|
| 1 | **每完成一轮对话，必须把进展更新到本文件（code1.md）** —— 新结论、新排除的假设、下一步方向，都要写进来 |
| 2 | 只改 code 1/ 目录下的文件，**不动其他目录**（code3 / code2 / 根目录），除非用户明确要求 |
| 3 | 不要 printf / flush / cout 调试输出，用其他手段（dump 到文件、返回值对比等） |
| 4 | 每次改动代码前先备份（cp xxx xxx.bak） |
| 5 | 一步一步验证，每步单独确认结果，不要一次性改多处 |

---
## 1. 状态速览

| 项 | 值 |
|---|---|
| 算子名 | `mhc_expand`（赛题：《【B组简单题】mHC-expand 算子（前向与反向）》） |
| 工程目录 | `code1/`（`op_host/`、`op_kernel/`、`tools/`） |
| 目标芯片 | `ascend910b`（真机 910B3，单卡 NPU ID=2，CANN 9.0.0） |
| 源码状态 | 【实测】`op_host/mhc_expand.cpp` / `op_kernel/mhc_expand.cpp` 已含**完整实现**（TilingFunc / InferShape / InferDataType / BF16 OpDef / TPipe+双缓冲 kernel），**不是**骨架 |
| 真机验证 | ⬜ **从未编译、从未上机** —— 这是本题**唯一的未知项** |
| 平台提交 | 无记录（未提交过） |
| 性能 | 无任何真机数据（无耗时 / 无带宽 / 无 msprof） |

**一句话**：代码看起来是完成品，但**"能不能编过、跑对"完全没验证过**。投入小、能拿到确定结果，是三题里性价比最高的下一步。

> ⭐ **权威来源**：官方题面全文见 `official_problem_statement.md`（第一题章节）。下表与官方**逐条核对一致**。

---

## 2. 输入契约（**动手前先看这张表**）

| 项 | 值 | 官方题面 |
|---|---|---|
| 输入名 | `x`（REQUIRED） | `x`，必选输入 ✅ |
| 输出名 | `o`（REQUIRED） | `o` ✅ |
| dtype | `ge::DT_FLOAT16`、`ge::DT_BF16`（InferDataType 直通：输出 dtype = 输入 dtype） | bfloat16、float16 ✅ 一致 |
| format | `ge::FORMAT_ND`（行主序连续，**无 batch 维、无 N 维**） | ND ✅ 一致 |
| 属性 | `mhc_mult`：OPTIONAL Int，默认 **2**；`backward`：OPTIONAL Bool，默认 **false** | 前向/反向为**两个独立接口**（`MHC Expand Forward` / `Backward`），本实现用 `backward` 属性合一 ⚠️ 见 §5-E5 |

### 2.1 语义

| `backward` | 输入 shape | 输出 shape | 语义 |
|---|---|---|---|
| `false`（前向） | `[S, D]`（rank 2） | `[S, m, D]`（rank 3） | `o[s, k, j] = x[s, j]`，沿**倒数第 2 维**复制 m 份 |
| `true`（反向） | `[S, m, D]`（rank 3，且必须 `in_shape[1] == mhc_mult`） | `[S, D]`（rank 2） | `x_grad[s, j] = Σ_{k=0}^{m-1} o_grad[s, k, j]`，沿 m 维**求和** |

官方原文：

```
前向：o[i, m, j] = x[i, j]    对所有 m ∈ [0, mhc_mult)
反向：x_grad[i, j] = Σ_{m=0}^{mhc_mult-1} o_grad[i, m, j]
```

官方形状约束：`o_grad` 的形状**必须与前向输出的形状一致**；`x_grad` 的形状**必须与前向输入的形状一致**。

- `m` 插在**倒数第 2 维**（不是最后一维、不是首维）。
- 轴语义：`S` = token 数，`D` = 隐藏维，`m` = `mhc_mult`。**本算子没有 batch 维**。
- 本算子**无任何可学习参数**，是纯数据搬运算子（memory-bound）。
- 关键布局性质：`[S, m, D]` 在行主序下等价于 `(S*m, D)`，即 **token s 的 m 份副本在 GM 上连续占 `m*D` 个元素**。

### 2.2 ⭐ 反向是**求和**，不是平均

- 赛题 2.2 原文给出 `x_grad[i, j] = Σ o_grad[i, m, j]`，示例代码 `o_grad.sum(dim=1)`。**赛题是唯一权威。**
- 数学上：**广播的伴随就是求和**。
- vLLM 的 `hc_contract` 用 `mean(dim=-2)`，但那是"往返可复原"的独立归约原语，**不是**本算子的反向。
- **两者数值相差恰好 `m` 倍 → 方向选错必定零分。**

---

## 3. 提交文件清单

| 提交字段 | 文件 | 位置 |
|---|---|---|
| `kernel_cpp` | `mhc_expand.cpp` | `op_kernel/` |
| `tiling_h` | `mhc_expand_tiling.h` | `op_kernel/` |
| `tiling_key_h` | `tiling_key_mhc_expand.h` | `op_kernel/` |
| `host_cpp` | `mhc_expand.cpp` | `op_host/` |

- 另有三个 `CMakeLists.txt`（`code1/`、`op_host/`、`op_kernel/`）。
- ⛔ **`op_host/` 与 `op_kernel/` 下是同名但不同内容的两个 `mhc_expand.cpp`，别传错。**
- ⛔ 提交前扫描：`grep -n "printf\|fflush\|cout\|调试" <四个文件>` 必须为空。
- 提交前**先 `--dry-run`**，确认四个文件都被正确识别。

---

## 4. 已排除 / 已确认（**别重复走**）

### 4.1 已确认（有源码级证据）

| 项 | 结论 |
|---|---|
| 源码落地状态 | **已完整实现**，不是骨架（实测 `op_host`/`op_kernel` 源码） |
| tiling 结构体 | **11 字段** `MhcExpandTilingData`（全 `uint32_t`），非"只有一个 `length`" |
| `InferShape` / `InferDataType` | **已完整实现**（前向设 3 维、反向设 2 维、失败返 `GRAPH_FAILED`；dtype 直通） |
| OpDef dtype | **已同时声明 `DT_FLOAT16` / `DT_BF16`** |
| tiling key 项数 | **4 项** = 2(dtype) × 2(backward)：`fp16/false`、`fp16/true`、`bf16/false`、`bf16/true` |
| 模板域 | `template <typename DT_X, bool BACKWARD>`（**backward 已进模板域**），4 个显式实例化 |
| 反向实现 | **逐副本 `for k` 各一次 `DataCopyPad`**（`blockCount` 固定 1），**不是**单次 `blockCount=m` |
| `Init` 签名 | `(GM_ADDR x, GM_ADDR o, GM_ADDR workspace, const MhcExpandTilingData &tiling)`（**多一个 `workspace` 形参**；host 恒置 `workspace[0]=0`，实际不需要 workspace） |
| 反向小 S 切分 | `SPLIT_ELEMENT`，`total_tasks = S * dTileNum`，**免跨核归约** |
| 核数/UB | `GetCoreNumAiv()` + `SetBlockDim(block_dim)`；UB 预算 = `ub_size/4`，`ub_size==0` 时兜底 192KB；kernel 用 `TQue` 双缓冲 + 2 个 fp32 buffer |

### 4.2 ⚠️ 已排除的**过时结论**（原文曾如是说，**已被源码推翻，不要再信**）

> 以下都出自已删除的 `code1/DESIGN.md`（设计阶段的计划文档），而落地后的源码与之不符。
> 该文档**已删除**（其可用的设计意图已并入本文件的 §2/§3/§4）：

| 设计文档曾写 | 实际情况 |
|---|---|
| "设计已定稿，**尚未落地到源码**，`op_host/`、`op_kernel/` 仍为原始骨架" | ❌ **已完整落地** |
| "当前只有一个 `uint32_t length`，必须整体替换" | ❌ 已是 11 字段结构体 |
| "`InferShape` 为**空实现**（直接 return GRAPH_SUCCESS）" | ❌ 已完整实现 |
| "当前只声明了 `C_DT_FLOAT16`，需补 BF16" | ❌ 已含 `C_DT_FLOAT16, C_DT_BF16` |
| "**`backward` 不进模板域**（否则实例从 2 个翻到 4 个）" | ❌ **`backward` 就在模板域**，实例就是 4 个 |
| "反向用一次 DMA 描述读 m 份（`blockCount=m`）" | ❌ 实际是逐副本循环，`blockCount=1` |
| "`Init` 签名必须换成接收全量 tiling" | ❌ 早已是全量 tiling（且多一个 `workspace`） |

> ⚠️ **不可信项**：文档称"文件与网站**通过版逐字节一致**"，但**没有任何 md5 / diff 记录**支撑（文档里的 md5 全是第三题的）。此结论目前只是**转述**，需重新校验（见 §5）。

### 4.3 反例：这些**还没有**被排除

- ❌ 从未在真机编译过 → **能否编过完全未知**
- ❌ 从未上机跑过 → 数值正确性、对齐行为、UB 容量全部未验证
- ❌ `sum` vs `mean` 的平台口径未知（本实现按 `sum`）
- ❌ 平台精度判据未知（`allclose(rtol=1e-3)` 还是**逐 bit**）
- ❌ 平台是按"单算子 + `backward` 属性"调用，还是按两个独立算子名调用，未知

---

## 5. ⭐ 下一步证据（按优先级，每条都写清"判据"）

### E1【最高优先】真机验证：能否编过、跑对

- **做什么**：把 `code1/` 的四个源文件推到真机构建目录，编译 + 跑用例矩阵。
- **判据（二值，不含糊）**：
  - 编译：出现**完整**编译器输出，`error:` 行为空 → 通过；有 `error:` → 逐条修
  - 运行：用例矩阵全 PASS → 本题可交；有 FAIL → 记录错误率与形状
- **前置**：先请用户建立 NPU 隧道（见 `连接信息.md` §2）。

### E2 复核"逐字节一致"（**低成本，但必须先做**）

- **做什么**：对比赛网站下载的 zip 重新做 `md5sum` / `diff` 并**落盘记录**。
- **判据**：四个源文件 md5 与 zip 内一致 → 基准可信；不一致 → 以 zip 为基准重新对齐（**先问用户**）。
- ⚠️ 没有这一步，"通过版"这个前提就是**未检验的假设**（正是第三题翻车的那类错误）。

### E3 必测风险点（有明确翻车机理）

| 风险 | 机理 | 怎么测 |
|---|---|---|
| **反向 `sum` / `mean`** | 差恰好 `m` 倍，选错**必定零分**。✅ 已由官方题面确认是 **sum** | 参考实现同时算两种口径，对拍时两种都比一遍，看哪种过 |
| **非 32B 对齐** | `D` 非 16 倍数时行首非 32B 对齐，裸 `DataCopy` 会崩 | 必测 `D = 100`、`D = 7167`、`D = 1`（**这是最易翻车处**，官方也点名"非对齐维度"） |
| **逐 bit 比对** | BF16 只有 8 位尾数，`m=4` 时 fp16/bf16 直接累加与 fp32 累加差最后 1 ulp | 反向**必须 fp32 域累加**（已实现），并对比 `xgrad_ref_fp32`。官方测试覆盖里点名"float16 累加精度、bfloat16 累加精度" |
| **UB 容量** | `m` 很大（如 16）时整行是否放得下 | 测 `m=16`（`S=64, D=512, m=16`） |
| **小 shape 用不满核** | `S=8, D=7168, m=2` 按行只能用到 8 个核 | 测小 S，并确认 `SPLIT_ROW_STREAM` / `SPLIT_ELEMENT` 生效 |
| **ccec 隐式转换** | `uint32_t`→`float` 隐式转换在**真机**报错，CPU 仿真不管 | 真机自然暴露；提交前先自查所有 `static_cast` |
| **`m=1` 退化** | 官方扩展倍数只列 `m=2/4/8`，但 `m=1` 应退化为纯拷贝 | 测 `m=1`，确认不越界、不退化成错误路径 |

### E4 用例矩阵

**官方题面给出的覆盖范围**（§6）：

| 维度 | 官方取值 |
|---|---|
| 数据类型 | bfloat16、float16 |
| 小规模 | `S=64, D=256, m=2` |
| 中规模 | `S=1024, D=4096, m=4` |
| 大规模 | `S=8192, D=7168, m=8` |
| 扩展倍数 | `m=2`、`m=4`、`m=8` |
| 边界场景 | `S=1`（单 token）、`D=1`（单维度）、非对齐维度 |
| 精度场景 | float16 累加精度、bfloat16 累加精度 |
| 前向验证 | 输出每个副本与输入是否一致 |
| 反向验证 | 梯度归约求和是否正确 |

**本地 `tools/reference.py` 内置 12 组 shape × 2 dtype = 24 组用例**（**超集**，含官方三档规模）：

```
(4,8,m=2) (4096,7168,m=4) (7,100,m=2) (1000,7167,m=3) (13,128,m=4) (1,1,m=2)
(1,4096,m=4) (128,256,m=1) (64,512,m=16) (32,1,m=4) (64,16,m=8) (8192,7168,m=4)
```

> ⚠️ 官方的中规模 `(1024,4096,m=4)` 与大规模 `(8192,7168,m=8)` **未被 reference.py 精确覆盖**（最接近的是 `(8192,7168,m=4)`）。上机时**建议补上官方这两组**。

- 判据：`torch.allclose(rtol=1e-3, atol=1e-3)`（两侧先转 fp32），前向另需 `torch.equal`（**逐 bit**）
- CLI：`python3 tools/reference.py --list | --selftest | --export DIR | --check-forward NPY | --check-backward NPY --case N --dtype {fp16,bf16}`

### E5 单算子 vs 双算子（**官方题面到手后新增的确认项**）

官方题面把前向与反向写成**两个独立算子**（`MHC Expand Forward` / `MHC Expand Backward`，各自有独立的输入输出规格表），而本实现是**单算子 `MhcExpand` + `backward` 属性**。

- 若平台按**两个独立算子名**调用 → 需**多注册一个 OpDef**，kernel 代码可 100% 复用
- 若平台按**单算子 + backward 属性**调用 → 现状即可
- **判据**：平台首个 `Compile Error` / `Wrong Answer` 的具体报错，或从提交后平台反馈的算子名判断

> ⚠️ 这是当前**无法在本地判定**的项，只能通过提交或询问评测侧确认。

---

## 6. 工程材料索引（本题相关）

| 路径 | 是什么 |
|---|---|
| `code1/op_host/mhc_expand.cpp` | host tiling（**提交文件**） |
| `code1/op_kernel/mhc_expand.cpp` | kernel（**提交文件**） |
| `code1/op_kernel/mhc_expand_tiling.h` | tiling 结构体（**提交文件**） |
| `code1/op_kernel/tiling_key_mhc_expand.h` | tiling key（**提交文件**） |
| `code1/tools/reference.py` | Python 参考实现 + 用例导出 + 对拍自检（**权威判据来源**） |

> 本题**无其它留存材料** —— 早期的接收/提交代码快照（`_incoming/q1`、`_incoming/q2`、`_submission326701`、`_payload.tar.gz`）与第二题算子包（`.run`）**已在本轮整理中删除**（内容与提交源重复，判据一律用 `md5sum`）。

---

## 7. 本题目录边界

整理后 **`code1/` 已只含第一题的东西**：

```
code1/
├─ op_host/mhc_expand.cpp            ← 提交文件
├─ op_kernel/mhc_expand.cpp          ← 提交文件
├─ op_kernel/mhc_expand_tiling.h     ← 提交文件
├─ op_kernel/tiling_key_mhc_expand.h ← 提交文件
├─ tools/reference.py                ← 参考实现与对拍
└─ CMakeLists.txt
```

原先混在本目录下的**第二题、第三题材料已分流**：

| 原位置 | 现状 | 归属 |
|---|---|---|
| `code1/_sfa/` | → `refs/sfa/`（只留有价值的：参考实现 / 用例 / 脚本） | **第三题** |
| `code1/_harness/` | → `refs/harness_sinkhorn/` | **第二题** 测试脚手架 |
| `code1/_incoming/`、`code1/_submission326701/`、`code1/_payload.tar.gz` | ❌ **已删除** | 早期快照 |
| `code1/mhc_sinkhorn_CPU全过_修复版.run` | ❌ **已删除** | **第二题** 算子包 |

> 各题自己的 `codeN.md` 才是该题的当前口径。本文件只讲**第一题**。
