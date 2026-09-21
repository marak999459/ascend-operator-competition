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
| 目标芯片 | `ascend910b` —— **已实测坐实**：真机 `02aeb` = **Ascend 910B3，NPU ID=7**（旧记录 ID=2 作废），驱动 25.5.0，CANN 9.0.0，HBM 64GB（§11.8） |
| 源码状态 | 【实测】`op_host/mhc_expand.cpp` / `op_kernel/mhc_expand.cpp` 已含**完整实现 + 性能优化**（tiling 整行条件修正 + 反向双缓冲流水线，见 §9），**不是**骨架 |
| 真机验证 | ✅ **编译门首过**（§11.8）+ ✅ **算子已在真机跑通并出全量精度矩阵**（2026-09-20 20:00–20:25，`02aeb` / 910B3 / 50 AIV，5 组 38 用例，§11.9）。⭐ **反向 24/24 全 PASS 且逐位精确**（含仿真侧三次没收口的 `bwd-*-large` 8192×7168×8 = **0/58720256**）；⛔ **前向 14/14 全 FAIL 且非确定** —— 机理已坐实为 `ForwardOneBlock` 的 **MTE1 读 UB 与 MTE2 写 UB 无序**（`op_kernel/mhc_expand.cpp:104-122`，纯搬运无 VEC 指令 ⇒ VECIN 队列的序不生效；CPU 仿真因 `ICPU` 不实现事件而原理上看不见，§11.9.4）。修法属改代码，**三候选已列给用户待拍板（§11.9.6）**，提交源本轮零改动。msprof 未采（前向未正确性定档， timing 无意义） |
| 仿真机验证 | ✅ **第二轮 5 组已全部收口**（§9.5 / §9.5.2）：`tpm0u` **并未被回收**（上一轮"容器被回收/日志丢失/机器被 large 压满"的判断**作废**）—— 5 组实际在 18:12–18:14 **两分钟内跑完**，5 份日志 + `driver.log` 已于 20:30 tar 捞回本地并 **6 份 md5 逐字节复验**（`code1/cpu_debug/recovered_r4/`）。结果：quick / medium / ub48 **ALL PASS**；`large` 前向 2 条 **0/469762048 位级精确 PASS**，反向 2 条未跑完（**已由真机收口，不再补跑**）；mtile **唯一 1 条 `bwd-fp16-BND-D32768-m2` 判为仿真假失败**（真机同参位级精确 PASS，§9.5.2）。⛔ **重大口径修正**：仿真对**前向无保护力** —— 前向仿真全绿、真机全红（§11.9.4），日志里 `[TmSim]: Run in serial mode.` 就是机理证据 ⇒ **本题正确性判据以真机为准** |
| 比赛平台提交 | 无记录（未提交过） |
| 性能 | ⚠️ **仅有真机 kernel 墙钟粗测**（启动器 sync 前后计时，分辨率 1ms）：反向 `large` 8192×7168×8（读 939.5MB + 写 117.4MB ≈ 1.06GB）→ `kern=0.001s`，含 `medium` 也一律印 0.001s ⇒ 单靠它**不足以支撑 §9 两项优化的 A/B 结论**。正式基线仍待 `msprof`（§11.6），**建议等前向修完再一次性采**（否则前向数字无意义、反向要重采） |

**一句话**：真机已接通并跑完 38 条矩阵 —— **反向位级精确全绿（含 `large` 与全部 UB 预算边界）、前向 14/14 全红**，根因已定位到 `ForwardOneBlock` 缺 MTE2→MTE1 序（§11.9.4），**修法是本题当前唯一的阻塞项，等用户拍板**（§11.9.6）；仿真侧 5 组已随 tpm0u 日志捞回而全部收口（§9.5.2），且已证明**仿真对前向无保护力**。提交源四文件 md5 与 §6.1 基线逐字节一致（本轮**零改动**）。

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
| 合规扫描（2026-09-20） | 四个提交文件 `grep "printf\|fflush\|cout\|fprintf\|TODO\|FIXME\|#if 0\|调试"` 为空 ✅ 可随时提交 |
| 静态审查（2026-09-20） | 全文通读 4 个提交文件：host/kernel 两侧任务编码（ROW/STREAM/ELEMENT）一致；反向 fp32 累加（`CAST_NONE` 入 / `CAST_RINT` 出）；GM↔UB 全部走 `DataCopyPad`（非 32B 对齐已处理）；切分决策无 uint32 溢出路径。⚠️ **其"未发现阻塞性问题"的结论已被真机推翻**（§11.9.4）：静态审查看不见流水线事件序 —— 前向 `ForwardOneBlock` 用 VECIN 缓冲直接喂 MTE1，**没有任何 MTE2→MTE1 依赖**，这是阻塞性缺陷。⇒ **教训：纯搬运 kernel 的队列 `QuePosition` 选择必须逐条对照"谁写这块 UB / 谁读这块 UB"来推，静态过一遍不够** |
| md5 基线（2026-09-20） | 完整值见 §6.1（**优化后口径**：host `a16c371d…` · kernel `0fb9e6ec…` · tiling.h `f759a052…` · tiling_key `267e0125…`）；20:30 真机全轮结束后**再次核对，四值未变 ⇒ 提交源本轮零改动**。后续判断"文件是否被动过"以此为准 |
| 真机 561002 的根因（2026-09-20） | ⭐ **「Do not find tiling func」不是算子缺陷，是加载方式**：`REGISTER_OP_LIB` 的注册器只在**框架自己 dlopen** `libcust_opapi.so` 时才把 `LocalRegistry` 提交进全局表；启动器把它做成**链接期依赖** ⇒ 注册表空 ⇒ 全红。**判据/解法已固化在 §11.9.2，下次直接照抄，不要再从环境变量猜** |
| 前向 vs 反向的真机分歧（2026-09-20） | ⭐ **反向全绿、前向全红是同一件事的两面**：反向在 VECIN 缓冲上有 `Cast`/`Add`（VEC 消费 ⇒ 序成立）且写出走 VECOUT；前向是**纯搬运、中间没有 VEC 指令** ⇒ VECIN 的序对 MTE1 完全不生效。**别再逐条查前向的偏移/对齐算术** —— `out` 总字节 939MB < 2^31、偏移全为 `int64_t`、`tile*2` 恒 32B 对齐，均已排除（§11.9.4） |
|本地仿真机对拍（2026-09-20） | quick 矩阵 **28 用例 ALL PASS**（前向/反向 × fp16/bf16；官方小规模、非对齐 D=100/7167、边界 S=1/D=1/m=1/m=16、强制 STREAM/ELEMENT 切分、fp16 饱和 m=1/m=2、多块切分 6 用例）。本地仿真机全量矩阵 fwd-medium/large、bwd-medium PASS。方法与历史坑见 §8；**提交内核零改动**参与同源验证 |

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

> ⚠️ **不可信项**：文档称"文件与比赛平台**通过版逐字节一致**"，但**没有任何 md5 / diff 记录**支撑（文档里的 md5 全是第三题的）。此结论目前只是**转述**，需重新校验（见 §5）。

### 4.3 反例：这些**还没有**被排除

- ~~❌ 从未在真机编译过 → 能否编过完全未知~~ → **已排除**：910B3 上 `ascend910b` 口径 20s 编过，4 个 `.o` + `libcust_opapi.so` 产出（§11.8）
- ~~❌ 从未上机跑过 → 数值正确性、对齐行为、UB 容量全部未验证~~ → **已验证并出结论**（§11.9）：反向 24 条位级精确 PASS；**非对齐 D=7167 / D=100 / D=1 全过**；**UB=256KB ⇒ 64KB 预算口径被真机反证成立**（D=32768 整行 vs 32769 退化 tile=2048，与 §9.1 推导吻合）。⚠️ 但由此**新增一条未排除项**：**前向在真机 14/14 全 FAIL**，机理已定位（§11.9.4），**修完之后需重跑全量矩阵**
- ⚠️ **新未排除项**：真机 UB 用量是否逼近上限 —— 前向整行路径 `tile=32768` 时 `in_que_(2×64KB)+out_que_(2×64KB)+acc+tmp(2×128KB)` 名义上远超 256KB，真机却 `D=32768` 反向 PASS ⇒ 说明这些缓冲**并未同时存活**或 `InitBuffer` 有复用；**若后续调整 tile 策略需重新核算这一点**，不能假定"256KB 够用"
- ❌ `sum` vs `mean` 的比赛平台口径未知（本实现按 `sum`）
- ❌ 比赛平台精度判据未知（`allclose(rtol=1e-3)` 还是**逐 bit**）—— ⚠️ 本轮新增一个必须靠它来裁的具体分叉：**fp16 溢出取 `inf` 还是 `65504`**（§11.9.5）
- ❌ 比赛平台是按"单算子 + `backward` 属性"调用，还是按两个独立算子名调用，未知
- ❌ **新**：比赛平台**是否会在真机之外用 CPU 仿真跑**（若是，§9.5.2 那条"仿真假失败 `BND-D32768`"就会白丢分）—— 口径未知

---

## 5. ⭐ 下一步证据（按优先级，每条都写清"判据"）

### E1【最高优先】真机验证：能否编过、跑对 —— ✅ **2026-09-20 已执行完毕，结论见 §11.9**

- **做什么**：把 `code1/` 的四个源文件推到真机构建目录，编译 + 跑用例矩阵。
- **判据（二值，不含糊）**：
  - 编译：出现**完整**编译器输出，`error:` 行为空 → 通过；有 `error:` → 逐条修
  - 运行：用例矩阵全 PASS → 本题可交；有 FAIL → 记录错误率与形状
- **前置**：先请用户建立 NPU 隧道（见 `连接信息.md` §2）。
- **实测结果**：编译门 ✅ 首过（§11.8）；运行 **不满足"全 PASS"** —— **反向 24/24 位级精确 PASS、前向 14/14 全 FAIL**（§11.9.3），错误率/形状已逐条记录 ⇒ 按判据**本题当前不可交**，阻塞点是 §11.9.6 的前向修法决策。

### E2 复核"逐字节一致"（**现状已变化**）

- **2026-09-20 更新**：比赛平台下载的 zip 及早期快照（`_incoming/`、`_submission326701/` 等）**已在整理中删除，本地无副本** → "与 zip 逐字节一致"**无法本地复核**。
- **替代做法**：已记录当前四个提交文件的 md5 基线（§4.1 / §6.1），以当前文件为唯一事实源。
- 若用户能**重新下载**比赛平台 zip → 再做一次 `md5sum` / `diff` 复核（先问用户，不自动对齐）。

### E3 必测风险点（有明确翻车机理）

| 风险 | 机理 | 怎么测 |
|---|---|---|
| **反向 `sum` / `mean`** | 差恰好 `m` 倍，选错**必定零分**。✅ 已由官方题面确认是 **sum** | 参考实现同时算两种口径，对拍时两种都比一遍，看哪种过 |
| **非 32B 对齐** | `D` 非 16 倍数时行首非 32B 对齐，裸 `DataCopy` 会崩 | 必测 `D = 100`、`D = 7167`、`D = 1`（**这是最易翻车处**，官方也点名"非对齐维度"） |
| **逐 bit 比对** | BF16 只有 8 位尾数，`m=4` 时 fp16/bf16 直接累加与 fp32 累加差最后 1 ulp | 反向**必须 fp32 域累加**（已实现），并对比 `xgrad_ref_fp32`。官方测试覆盖里点名"float16 累加精度、bfloat16 累加精度" |
| **UB 容量** | `m` 很大（如 16）时整行是否放得下 | 测 `m=16`（`S=64, D=512, m=16`）。✅ 仿真已覆盖：优化 A 去掉 m 因子后 `D=4096, m=16` 走整行 `tile=4096` 且 PASS（§9.3 mtile 组） |
| **小 shape 用不满核** | `S=8, D=7168, m=2` 按行只能用到 8 个核 | 测小 S，并确认 `SPLIT_ROW_STREAM` / `SPLIT_ELEMENT` 生效。✅ 仿真 mode=1/2 用例全 PASS |
| **ccec 隐式转换** | `uint32_t`→`float` 隐式转换在**真机**报错，本地仿真机不管 | 真机自然暴露；提交前先自查所有 `static_cast` |
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

- 若比赛平台按**两个独立算子名**调用 → 需**多注册一个 OpDef**，kernel 代码可 100% 复用
- 若比赛平台按**单算子 + backward 属性**调用 → 现状即可
- **判据**：比赛平台首个 `Compile Error` / `Wrong Answer` 的具体报错，或从提交后比赛平台反馈的算子名判断

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
| `code1/cpu_debug/` |本地仿真机对拍 harness + 构建/运行脚本（**非提交**；脚本已做成本地/云端仿真机通用，见 §8 与工作流 §3.6） |
| `code1/cpu_debug/quick_matrix_cloud_20260920.log` | 云端仿真机28 用例逐行结果留档（**非提交**，ALL PASS 证据） |
| `code1/cpu_debug/recovered_r4/` | **tpm0u 第二轮 5 组日志捞回归档**（**非提交**，§9.5.2，md5 已双侧复验） |
| `code1/npu_debug/` | **真机侧**（**非提交**）：`test_mhc_expand_npu.cpp` 启动器 + `build_npu.sh` + `run_npu.sh`（组 OPP 包）+ `preflight_npu.sh` + `pkg/`（运行期组出的 custom OPP 包） |
| `code1/npu_debug/logs/` | 真机历轮日志 30+ 份（`preflight` / `build_gate` / `npu_quick*` / `npu_diag*` / `npu_det*` / `npu_matrix1` / `npu_large`，§11.9），**精度结论的唯一原始出处** |

> 本题**无其它留存材料** —— 早期的接收/提交代码快照（`_incoming/q1`、`_incoming/q2`、`_submission326701`、`_payload.tar.gz`）与第二题算子包（`.run`）**已在本轮整理中删除**（内容与提交源重复，判据一律用 `md5sum`）。

### 6.1 md5 基线（2026-09-20 记录，**2026-09-20 优化后更新**）

```
a16c371d1bb39a81d0efecd24db71b49  code1/op_host/mhc_expand.cpp        ← 优化后（tiling 条件修正）
0fb9e6ec007579425c563f54fe61e92f  code1/op_kernel/mhc_expand.cpp      ← 优化后（反向双缓冲）
f759a052a865facbb889149ea1aa37e3  code1/op_kernel/mhc_expand_tiling.h
267e012564ba3d18cfabc729cc214e88  code1/op_kernel/tiling_key_mhc_expand.h
e4230a47ccb6d3a8616287a4e2449bcc  code1/tools/reference.py（非提交文件）
--- 以下 cpu_debug/ 非提交 ---
4668173231e634e3d448962cc50dc965  code1/cpu_debug/test_mhc_expand_cpu.cpp  ← 在 99db13d2 基础上加 3 个预算边界用例（§9.4）；tiling 镜像同步 + 模式分组(quick/medium/large/mtile/ub48) + 逐用例计时；**第二轮已在 tpm0u 上机跑（§9.5）**
1fe721462b86e1ff00ce95fb3a4aa13f  code1/cpu_debug/build_cpu.sh
dca1abe8130b508764c5e595a142e60a  code1/cpu_debug/run_cpu.sh
deb5bd526794498c321f64c4a4fe0fce  code1/cpu_debug/run_all_groups.sh  ← 分组驱动（组名白名单 + 真实用例计数）；**正在跑的那份远端版本 = 033b9d2d…，只差计数器取 `maxdiff=` 还是 `^\[`**
b770ff2af9e758fdbfc6f207cf7936f3  code1/cpu_debug/probe_args.sh  ← 诊断：打印脚本实际收到的位置参数（定位 GROUPS 保留变量坑用）
--- 以下 npu_debug/ 非提交（真机侧，2026-09-20 20:30 记录）---
c7a3e69899cf0748ec1a8e498ef18fdc  code1/npu_debug/test_mhc_expand_npu.cpp  ← 真机 ACL 启动器（§11.9.1）：**dlopen 取 aclnn 入口，不做链接期依赖**
6b4030a55335f541cb689b4dbf4244d5  code1/npu_debug/build_npu.sh             ← 探测 set_env + 时间戳/`nm -D` 防假成功 + 链接 `-lnnopbase -lascendcl -ldl`
417ceccbbd9c4bfc20f13dec5d0f68a2  code1/npu_debug/run_npu.sh               ← 运行期组 custom OPP 包 + 设 `ASCEND_CUSTOM_OPP_PATH`（解 561002）
b488fed4f41cb02972f4bf2682e4dd0f  code1/npu_debug/preflight_npu.sh         ← 到手第一条只读命令（§11.3）
--- tpm0u 捞回的 r4 日志（§9.5.2，只读留证）---
4e605205ba796010d68d1f8c3de51c4f  code1/cpu_debug/recovered_r4/quick_r4_20260920_1812.log
3af4a0c75f26f249a5a975eb8add9d35  code1/cpu_debug/recovered_r4/medium_r4_20260920_1812.log
060c7388b4a6be5c480b6241250d0c34  code1/cpu_debug/recovered_r4/mtile_r4_20260920_1812.log
40b0a9723889f828bc7d40f1ce6d16a0  code1/cpu_debug/recovered_r4/ub48_r4_20260920_1812.log
8ccfd1d3a589271abee5bcee94afefd6  code1/cpu_debug/recovered_r4/large_r4_20260920_1812.log   ← 前向 2 条位级 PASS，反向 2 条未跑完
759952580f2a1754d4f615480925624e  code1/cpu_debug/recovered_r4/driver.log
```

> ⛔ **`op_kernel/` 三个文件的 md5 是"提交内核零改动"的证据链**：推到任何调试环境都用 **tar 管道 + md5 复验**，**不得对它们做格式化/行尾转换**。远端只是副本，**本地 `code1/` 永远是唯一权威源**（容器销毁不影响任何文件）。
> ⚠️ 行尾状态**用 `tr -dc '\r' < f | wc -c` 判定**，不要用 `grep -c $'\r'`（模式会退化、把"含字母 r 的行数"当成 CRLF 计数 → 本轮据此误判过一次"全文件 CRLF"，实际全部纯 LF）。
> ⚠️ **本仓库 `core.autocrlf=true` 且无 `.gitattributes`**：内核文件在 **git 索引里是 LF**，但 `checkout` / 重新 clone 会把工作区写成 **CRLF** → **上面这份 md5 会集体失配（代码其实没变）**。**已实测**：`git checkout-index` 导出的 `op_kernel/mhc_expand.cpp` = `28f7760d454b99d660f11ee9796a2ff4`（含 183 个 CR），与工作区/索引的 `d9af56115380fba67550bf76a886084e` 不同。核对口径三选一：① 比对前归一化 `tr -d '\r' < f | md5sum`；② `git cat-file -p :<路径>` 直接读索引字节；③ 加 `.gitattributes` 写 `* -text` 关闭转换（属仓库配置，**需用户同意**）。

---

## 7. 本题目录边界

整理后 **`code1/` 已只含第一题的东西**：

```
code1/
├─ op_host/mhc_expand.cpp            ← 提交文件
├─ op_kernel/mhc_expand.cpp          ← 提交文件
├─ op_kernel/mhc_expand_tiling.h     ← 提交文件
├─ op_kernel/tiling_key_mhc_expand.h ← 提交文件
├─ cpu_debug/                        ← 仿真机对拍（非提交；tar 管道推本地仿真机 / 云端仿真机）
│   ├─ test_mhc_expand_cpu.cpp       ← 同源 harness：#include op_kernel 内核 + 镜像 TilingFunc
│   ├─ build_cpu.sh / run_cpu.sh     ← CANN 路径探测 + `$(uname -m)-linux` 自适应（本地 / 云端仿真机通用）
│   └─ quick_matrix_cloud_20260920.log ← 云端仿真机28 用例逐行结果（ALL PASS 证据）
├─ npu_debug/                        ← 真机侧（非提交）：ACL 启动器 + 构建/组包脚本 + preflight
│   ├─ test_mhc_expand_npu.cpp       ← dlopen 取 aclnn 入口（⛔ 不得做成链接期依赖，否则 561002，§11.9.2）
│   ├─ build_npu.sh / run_npu.sh     ← 编译 + 运行期组 custom OPP 包
│   ├─ pkg/custom/…                  ← run_npu.sh 自动组出的 OPP 包（ASCEND_CUSTOM_OPP_PATH 指这里）
│   └─ logs/npu_*.log                ← 真机历轮原始日志（§11.9）
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

---

## 8. 仿真机验证（2026-09-20，本地仿真机 + 云端仿真机）

**方法（同源对拍，提交内核零改动）**：`cpu_debug/test_mhc_expand_cpu.cpp` 直接 `#include "op_kernel/mhc_expand.cpp"`，tiling 决策镜像 op_host `TilingFunc`；`ICPU_RUN_KF` 以 blockDim≤20 跑核（CPU sim 核数上限 <50）。

**两个调试环境**：
- 本地仿真机 `192.168.101.128`（fszqsn，x86_64，6vCPU/24G）：CANN 在 `~/Ascend/cann/cann-9.0.0`（⚠️ 部分文档写 `/usr/local/Ascend/...`，在本地仿真机上**不存在**）。
- 云端仿真机 devenv 容器（aarch64，16 核/32G，`npu-smi` 不存在 = 纯 CPU 仿真）：CANN 在 `~/Ascend/cann-9.0.0`，`libpem_davinci.so` 在 `aarch64-linux/simulator/dav_2201/lib/`。

构建/运行脚本 `cpu_debug/build_cpu.sh` / `run_cpu.sh` 已固化全部编译坑（`-DASCENDC_CPU_DEBUG -D__NPU_ARCH__=2201 -D_GLIBCXX_USE_CXX11_ABI=0` + `-lpem_davinci -lcpudebug*` 系列），并做了**CANN 路径自动探测 + 架构自适应**（`$(uname -m)-linux`），本地仿真机/云端仿真机通用。工程推送用 tar 管道 + md5 校验。

### 8.1 结果

- **quick 矩阵 28 用例 ALL PASS**（云端仿真机16v/32G aarch64，2026-09-20）：前向/反向 × fp16/bf16；官方小规模 (64,256,2)；非对齐 D=100/7167；边界 S=1/D=1/m=1/m=16；强制 STREAM/ELEMENT 切分（aiv=4）；**fp16 饱和**（x 全 32768：m=1 和在量程内精确 / m=2 和 65536 饱和到 0x7bff=65504 ✅ 符合 IEEE 惯例）；**多块切分 6 用例**（bwd MT tile=1536×3 块尾块 1024、MT-ODD 尾块 1023 奇数、fwd MT-STREAM/ELEMENT、bf16 MT×2）——bwd-large 独有的 dTileNum>1 路径已小规模锁定。
- 本地仿真机（6vCPU）全量矩阵跑到 bwd-medium 后按用户决策终止（exit 137）：**fwd-medium / fwd-large(8192×7168×8, 469M 元素) / bwd-medium 均 PASS**；bwd-large 改由小规模多块用例覆盖 + 真机验证。
- 云端仿真机全量矩阵（含 medium/large 档）**未跑成**：nohup 启动后隧道即抖动，重连时进程与 `/tmp` 日志一并消失（隧道断会带走容器后台任务）；随后**该容器开机即异常、由用户删除重建**（2026-09-20）。→ 结论：大规模正确性以 **本地仿真机的 fwd-large PASS + 等价 tiling 路径小规模覆盖** 背书，剩余交给真机（真机毫秒级，比仿真划算）。
- 逐用例结果留档：`cpu_debug/quick_matrix_cloud_20260920.log`（33 行，含每例 tile/mode/blk 与实际失配数，0 失败）。跨环境搭建配方与踩坑已归入 `算子开发工作流.md` §3.6 / §4.2 / §4.3 / §6。

### 8.2 ⭐ 历史坑：反向 fp16 一度 FAIL 的根因（**内核无 bug，别再查**）

- 现象：`bwd-fp16` m=2~5 FAIL（`got=65504(0x7bff) ref=131008(0x7fff)`），与核数无关（blk=1..20 失配数恒为 1698/16384）。
- 根因：**harness 填充表达式 `(uint32_t)(i*37+11)%29 - 14` 是无符号运算** —— residue<14 回绕成 ~4.29e9，×0.125 = 5.37e8，超 fp16 量程。宿主 `(half)` 转换在 [65536,131072) 产出 0x7fff（非 IEEE，读回原值），内核 `CAST_RINT` 饱和到 0x7bff=65504；两种**合法**约定仅对超量程值不一致。
- 证据链：① `bad_fill=15819/32768` 恰为 14/29（坏值在跑核**前**就存在）；② x 跑核前后位级快照完全一致（0/32768，内核不写输入）；③ m-sweep（m=1/8/16/32 PASS，m=2/3/4/5 FAIL）失配数逐个被剩余定理算准；④ 常数填充 PASS；⑤ bf16 全 PASS（8 位指数装得下 5e8）。
- 修复：填充改有符号 `(int64_t)((i*37+11)%29) - 14`（最大 |和| = 14×0.125×32 = 56，量程内）；另加显式饱和用例（`fill_mode==2`）锁定内核饱和行为。

### 8.3 对真机的启示

- 官方用例只要输入在 fp16 量程内（评测数据通常如此），反向求和路径不会触饱和分歧；即使触发，内核按 IEEE 饱和（0x7bff）也是标准行为。
- 本地仿真机已覆盖：三种切分模式、非 32B 对齐、尾部块、m 边界、双 dtype。**未覆盖**（只能真机验）：真机 DMA 行为、ccec 隐式转换、UB 实际容量、多核调度时序。

---
## 9. 性能优化（2026-09-20）

### 9.1 优化 A：tiling 条件修正（host）

**问题**：原条件 `m * D * elem_size <= ub_budget` 过于保守。核内同时只持有 **1 份 D**（前向读 1 次写 m 次、反向逐副本读），不需要 m 份同时在 UB。

**修正**：`D * elem_size <= ub_budget`；`max_t = ub_budget / elem_size`（不再除以 m）。

**效果（大档 S=8192, D=7168, m=8, fp16）**：

| | 修正前 | 修正后 |
|---|---|---|
| dTileLen | 2048 | 7168 |
| dTileNum | 4 | 1 |
| 前向 DMA/行 | 4 read + 32 write = 36 | 1 read + 8 write = 9 |
| 反向 DMA/行 | 16 read + 4 write = 20 | 8 read + 1 write = 9 |
| 内层 D 循环 | 4 次 | 0（消除） |

中档 (1024, 4096, 4) 同样受益：dTileNum 4→1。

### 9.2 优化 B：反向双缓冲流水线（kernel）

**问题**：原反向逐副本串行——DMA 读 k → VEC Cast+Add → DMA 读 k+1 → VEC Cast+Add → ……，DMA 与 VEC 不重叠。

**修正**：预取 k+1 的 DMA 与 k 的 VEC 计算并行。双缓冲交替使用 `in_que_` 的两个 slot：
- 循环前预取 k=0
- 每轮：DeQue 当前 → AllocTensor+DataCopyPad 预取下一个 → Cast+Add → FreeTensor
- 最后一轮不启动 DMA（`k < m-1` 守卫）

**效果**：DMA 延迟被 VEC 计算隐藏。每轮耗时 ≈ max(DMA, Cast+Add) 而非 DMA + Cast + Add。

### 9.3 云端仿真机重跑（2026-09-20，容器 xq82l aarch64/16核）

**优化后分组重跑，已跑完的 73 个用例逐条 `maxdiff=0.00000 mismatch=0`**：

| 组 | 用例数 | 结果 | 关键证据 | 日志（容器 `cpu_debug/`） |
| --- | --- | --- | --- | --- |
| quick | 35 | ✅ ALL PASS | 全部 `tile=D`（整行路径生效）；反向双缓冲 m=1/2/3/4/5/8/16 全覆盖 | `quick_r2_20260920.log` |
| medium（1024×4096×4） | 4 | ✅ ALL PASS | fwd/bwd × fp16/bf16，`tile=4096`、`mode=0`(ROW)，**46 秒**跑完 | `medium_r2_20260920.log` |
| large（8192×7168×8） | 前向 2/4 | ⚠️ 前向 ✅、反向 ⏳ | 前向 `mismatch=0/469762048`（4.7 亿元素逐位一致），**22 秒**；反向 fp16/bf16 跑到中途容器 SSH 掉线，结果未取回 | `large_r2_20260920.log`（掉线前只落前向两行） |
| mtile（多 tile + 大 D） | 15 | ✅ ALL PASS | 含**奇数尾块**多 tile：`D=70001→tail=369`、`D=33001→tail=233`、`bf16 D=70000` | `mtile_r2_20260920.log` |
| ub48（host `ub_size==0` 兜底 48KB 预算） | 17 | ✅ ALL PASS | `D=26001/26000` 在 48KB 预算下由整行**退化**为 `tile=2048` 多 tile，仍逐位 PASS | `ub48_r2_20260920.log` |

> ⚠️ 上表 5 份日志**只落在容器 xq82l 上**；该容器已于同日回收（`devspace_tunnel.ps1` 自举时服务端报 `development environment no longer exists`）→ **日志已不可取回**。73 个用例的逐条结果以上表与当时输出为准（本会话已读到），重跑时**必须把 stdout 直接落到本地 `code1/cpu_debug/logs/`**，不再在容器上攒。

**三条值得记住的事实**：
1. **仿真耗时此前被高估了一个量级**：旧文档按"中/大档小时级"推断，实测 medium 全组 **46s**、large 前向（10.6 亿 DMA 元素）**22s**。根因：日志里的 `kern=` / `cpu_s` 是 `clock()`，多线程仿真下**只计主线程**，拿它推墙钟必然离谱。→ 中大档**不必留给真机**；large 反向按 medium 反向外推约 5–10 分钟（未实测收尾）。
2. **多 tile 的尾块在 host 侧恒为偶数**：`op_host/mhc_expand.cpp:90` 的 else 分支写死 `t = std::min(2048, D)`（不是按预算算出的 24576/32768），`2048×k` 为偶 ⇒ 只有 **D 取奇数**才能逼出奇数尾块。新增的 `D=70001/33001` 两个用例正是补这个缺口。
3. **整行/多 tile 的分界就是 `D*elem_size ≤ ub_size/4`**：910B 真机 UB=256KB ⇒ 预算 64KB ⇒ fp16/bf16 整行上限 `D ≤ 32768`；官方三档 D（256/4096/7168）全部走整行，多 tile 分支只在兜底或超大 D 时生效。

### 9.4 待跑用例与 tiling 预期（2026-09-20 用纯算术钉死，等到环境直接对答案）

`op_host/mhc_expand.cpp:85-101` 的 tile 决策是**纯整数运算**，无需上机即可预判。用镜像该函数的 Python 脚本复算，新增 3 个预算边界用例的预期如下：

| 用例 | S×D×m | ub_budget | 预期 tile | 预期 num | 预期 tail | 压的是哪条路 |
|---|---|---|---|---|---|---|
| `bwd-fp16-BND-D32768-m2` | 2×32768×2 | 64KB | 32768 | 1 | 32768 | `D*2 == budget`，`<=` 取等号 ⇒ **整行**最后一格 |
| `bwd-fp16-BND-D32769-m2` | 2×32769×2 | 64KB | 2048 | 17 | **1** | 超预算 2 字节即退化多 tile，**尾块只剩 1 元素** |
| `fwd-fp16-BND-D32769-m4` | 1×32769×4 | 64KB | 2048 | 17 | 1 | 同上走前向（S=1 ⇒ SPLIT_ELEMENT） |

同法复核已有用例：`D=70001→tile=2048/num=35/tail=369`、`D=33001→17/233`、ub48 组 `D=26001→13/1425`（48KB 预算下 `D=24576` 恰好取等整行、`D=24577→num=13/tail=1`）——与 §9.3 表里容器实际输出的 `tile=` 一致，说明 harness 的 tiling 镜像与 op_host 同源无漂移。

mtile 组用例数因此从 15 → **18**（全矩阵待跑合计 76）。

**待办**：
- [x] 中档复跑（medium 4 用例）
- [ ] **large 反向 2 用例 + 3 个新边界用例**（`run_cpu.sh 20 large` / `run_cpu.sh 20 mtile`）——任一活着的 CPU 仿真机上跑，**每组 stdout 同时 tee 到本地 `code1/cpu_debug/logs/<组>_<日期>.log`**
- [ ] 真机验证（E1）：编译 + 跑用例矩阵
- [ ] 真机 msprof 测性能基线（优化 A/B 的收益只能在这里定量）

> ⚠️ **环境状态（2026-09-20 收尾，`devspace_tunnel.ps1 -List` 实测）**：三条算力路径当日全部不可用——
> ① **本地仿真机** VMware 未开机（`vmware-vmx` 进程不存在，`192.168.101.128:22` 超时）；
> ② **云端 CPU 环境** `e6z6k` / 原 xq82l 已回收（服务端 `no longer exists`，只能用户在 IDE 点 Start）、`tpm0u` 在线但正给**另一个会话**当通道（全量仿真是 ~60 进程压 16 核，会把它的 sshd 饿死 ⇒ 不能派活）；
> ③ **真机** `02aeb` 隧道未自举，且上机按纪律先问用户。
> → 剩余待办**卡在环境，不卡在代码**。用户开任意一个 CPU 环境后，本轮工作一步接上。

> 本轮容器推送清单 md5 已复验（本地=远端逐字节一致，CR=0）：kernel `0fb9e6ec…` / host `a16c371d…` / tiling.h `f759a052…` / tiling_key `267e0125…` / harness `99db13d2…` / build_cpu.sh `1fe72146…`。环境下次接上时**先把本地 `code1/` 整目录 tar 管道推过去并逐文件复验 md5**（当前 harness 已变为 `4668173…`，见 §6.1）。

### 9.5 第二轮全量仿真（2026-09-20 18:12，云端仿真机 `tpm0u` aarch64 / 16 核 / 30G）

题2 释放 `tpm0u` 后接上：整目录推送 8 文件 md5 逐字节一致，构建产物 `test_expand_cpu` 830080 字节 / 18:02 新鲜（非假成功）。分组驱动 `cpu_debug/run_all_groups.sh` 一次跑 5 组，`driver.log` 实测：

| 组 | 起止 | rc | 结果状态（**20:30 已全部捞回本地，见 §9.5.2**） |
| --- | --- | --- | --- |
| quick | 18:12:41 → 47（6s） | 0 | ✅ **`=== ALL PASS ===`**（30 条用例行） |
| medium | 18:12:47 → 18:13:28（41s） | 0 | ✅ **`=== ALL PASS ===`**，`dma_elems=83.9M / vec_elems=159.4M / cpu_s=2` |
| mtile（含 3 个新 BND 用例） | 18:13:28 → 38（10s） | **1** | ⚠️ **`HAS FAIL` —— 唯一一条 `bwd-fp16-BND-D32768-m2` 仿真 FAIL(63276/65536)，真机同参位级精确 PASS ⇒ 判为仿真假失败**（§9.5.2） |
| ub48 | 18:13:38 → 49（11s） | 0 | ✅ **`ALL PASS`**（含 `D=24576/24577` 48KB 兜底边界对） |
| large（反向 2 用例 = 本轮缺口） | 18:13:49 起，**未跑完** | — | 🟡 捞回 285 行：**`fwd-fp16-large` / `fwd-bf16-large` 各 0/469762048 位级精确 PASS**，第 3 条 `bwd-*-large` 起始处被截断。反向 2 条**已由真机收口**（§11.9.3），不再补跑 |

⚠️ 原判断"这 4 份日志只存在容器上、取回前不得计为已验证"**成立且已执行**：**20:30 已 tar 管道捞回本地并 6 份 md5 逐字节复验**，详见 §9.5.2。

⚠️ **一个已成模式的事实（两轮独立观察）**：**两台不同的容器都是在 `large` 组（尤其反向 fp16/bf16 两用例）运行期间变得不可达** —— xq82l 当日端口拒连、5 份日志丢失；tpm0u 这次是 banner exchange 超时。⇒ §9.3 里"large 反向按 medium 外推 5–10 分钟"的估计**不成立**，该组在 CPU 仿真上的真实代价远超预估，且会把机器压到 sshd 无法应答。**下一步该改成"降并发单独跑 large"**（`run_cpu.sh 4 large` ⇒ 仿真子进程从 ~16 个降到 ~4 个，留出 sshd 的 CPU），而不是继续按 `20` 压满。

**本轮挖到的坑：`GROUPS` 是 bash 保留变量 + harness 对未知组名"假 ALL PASS"**

前两次启动（18:04 / 18:08）驱动都打印 `1000 START → DONE rc=0 → ALL PASS`，看起来全绿，实际**一个用例都没跑**：

1. 驱动第一版用 `GROUPS=(quick medium …)` 存组名。`GROUPS` 是 bash 的**内置特殊变量**（当前进程的 gid 列表），赋值会被 bash 立即重置回 `(1000)` —— 容器用户 `developer` 的 gid 正好是 1000。于是循环只迭代出一个不存在的组名 `1000`，与传参无关（这也是"两次都得到同一个魔法数"的原因）。
2. 更致命的是 harness **不拒绝未知 mode**：`test_mhc_expand_cpu.cpp:303-306` 的五个布尔全 false ⇒ 零用例，末尾照样 `=== ALL PASS ===` + `exit 0`。驱动只看 rc 就报成功 → **假成功串成了链**。
3. 已修（工具层，非提交代码）：变量改名 `MODES`；驱动加组名白名单（不合法记 `SKIP bad-mode`）+ 每组用 `grep -c 'maxdiff='` 统计**真实执行**的用例数，为 0 记 `SUSPECT zero-cases`。诊断脚本 `cpu_debug/probe_args.sh` 留在仓库，下次怀疑参数问题一步定位。
4. **待用户拍板（代码改动，我不自动改）**：harness 侧把"未知 mode"从静默通过改成硬失败（非零退出），这样任何拼错的组名都会立刻暴露。

**顺带核实**：`ssh -n` 会把 stdin 接成 `/dev/null`，用它做 `本地文件 | ssh 'cat > 远端'` 推送会写出空文件（远端 md5 恰好是空串的 `d41d8cd9…`）—— 推送类命令一律不加 `-n`。

#### 9.5.1 收尾事故（18:54–18:56 实测，`large` 跑到第 41 分钟时环境侧出事）

- `pull_watch.log`：18:36→18:54 共 7 轮全部 `ssh busy/unreachable`，本地 `driver_r4.log` / `remote_md5_r4.txt` **0 字节 ⇒ 一行都没取回**（那 4 组结果迄今只在容器上）。
- **发现自己在双开轮询**：每个 attempt 打两遍（间隔 27s）。根因是 `TaskStop` 只杀掉了外层 shell，第一版脚本进程（PID 10924）仍活着 → 已 `Stop-Process` 清掉（复查 `Count=0`），频率恢复 3 分钟一次。
- 18:54 端口层从 `UP` 变 **`down`** ⇒ 形态从"sshd 被压满"变成"转发已退出"。
- 18:56 `devspace_tunnel.ps1 -Role cpu -Diag`：**服务端换不出 `connect_url`，`development environment no longer exists`**；扩展日志给出更精确的判据 —— **当前登录账号的环境列表里查不到 `devEnvId 724ca0f5…`（tpm0u）**。
- 同一时段（18:53:26）日志里却有 **`e6z6k`（devEnvId `d611a7d4…`）`forward.ready localPort=48254`** 成功建立 —— **与上午完全反过来**（上午 tpm0u 可用、e6z6k 查不到）。
- ⇒ 结论修正：这**不像 tpm0u 被回收**，更像**桌面 VS Code 的登录账号又切了一次**（现在这个账号看得见 e6z6k、看不见 tpm0u）。两种情形的处置完全不同，且**只有用户能区分**（控制台看 tpm0u 是"运行中"还是已消失）。
- ⚠️ 关键风险：若只是账号可见性，切回原账号后隧道可自举、**驱动（`setsid nohup` 起，脱离 ssh 会话）可能仍在跑 large，4 份日志还在容器上能捞**；若容器真被回收，则第二轮结果与 xq82l 一样再次丢失。

#### 9.5.2 ⭐ 定档：**tpm0u 并未丢失，5 组日志已全部捞回**（20:30 实测）

用户 20:2x 交办"tpm0u 暂定为无法再获取，剩下的你看着办"后，我在整理真机日志时顺手看到本机仍挂着一条 **19:57 起的 `ssh -T -D 53090 → devenvc_tpm0u…`** 且 `netstat` 显示其 **LISTENING + ESTABLISHED** ⇒ 只探测一次（**未重试轰炸**）即连通：容器**活着、`/home/developer/ops_comp/code1/cpu_debug/logs/` 五组日志俱在**。

⇒ 所以 §9.5.1 的两种情形里成立的是**"账号可见性"那一种**：容器从未被回收，`large` 也从未"压满 16 核导致 sshd 无法应答"——**真实原因是 `driver.log` 里 5 组早在 18:12–18:14 两分钟内就跑完了**（quick 6s / medium 41s / mtile 10s / ub48 11s），所谓 41 分钟是**我这边隧道不可见**，不是机器在忙。⚠️ **这是一次误判**：把"我连不上"当成了"机器被压满"，并据此写下了与 §9.3 相反的"large 在仿真上要极贵"的结论（上文 ⚠️ 那段"下一步改成降并发单独跑 large"）—— **该结论作废**：`large` 慢只慢在**反向 2 条**（18:13:49 起、至 18:54 仍未出结果 ⇒ 单条确为**十分钟级**），前向 2 条是秒级且已位级通过。降并发的建议只对**反向 large** 有意义，而它已被真机收口，**不再补跑**。

捞回方式与判据（全部通过才算数）：`ssh 'cd …/cpu_debug && tar cf - logs/*_r4_*.log logs/driver.log' | tar xf -`（**不加 `ssh -n`**）→ 本地 `code1/cpu_debug/recovered_r4/` → **远端/本地 md5 六份逐一比对一致**：

```
4e605205ba796010d68d1f8c3de51c4f  quick_r4    3af4a0c75f26f249a5a975eb8add9d35  medium_r4
060c7388b4a6be5c480b6241250d0c34  mtile_r4    40b0a9723889f828bc7d40f1ce6d16a0  ub48_r4
8ccfd1d3a589271abee5bcee94afefd6  large_r4    759952580f2a1754d4f615480925624e  driver.log
```

**捞回后的两个新事实（真机侧看不到、必须靠仿真）**：

1. ⚠️ **`bwd-fp16-BND-D32768-m2` 是仿真的"假失败"**：仿真 `mismatch=63276/65536 maxdiff=1.875` **FAIL**，而真机同参（`S=2 D=32768 m=2 blk=2 mode=2 tile=32768`，整行路径取等点）**`maxdiff=0.00000 mismatch=0/65536` 位级精确 PASS**（§11.9.3）。⇒ 该用例**以真机为准**，仿真侧这条 FAIL 不要再查、也不要为它改核；它是"整行 tile 恰好等于 64KB 预算"这个取等点上仿真器/镜像的产物。**反向其余 23 条两侧一致（全绿）。**
2. ✅ 前向 `large` 在仿真上 **0/469762048 位级精确 PASS**，而真机 **98.9% FAIL**（§11.9.4）—— 两侧**完全反向**的分歧。捞回的日志末尾数十行 `[TmSim]: Run in serial mode.` 就是这件事的**直接书面证据**：仿真器串行执行 ⇒ 流水线竞争**原理上不可能被发现**。**"仿真全绿"对本 kernel 的前向没有任何保护力**，这道题的正确性判据只能是真机。

⇒ **仿真侧缺口至此收口**：任务 #5 完成。历轮"题1 仿真 large 未收口"的说法全部作废，当前口径见 §1。


---

## 10. ⭐ 开源参考池：`ops-transformer-master`（2026-09-20 新增）

> 📌 **定位（用户 2026-09-20 定调）**：这批开源材料**有很大的参考价值**，但**最好不要照抄**——可以抄的是**部分细节**（切分策略、搬运手法、阈值取法这类"怎么做"），实现仍要自己出。下面每条都标了"可借鉴什么"，供改代码时按需查，**不作为提交源**。

> 📦 **本节所引路径/行号依赖外部库，它不入库**（`.gitignore` 已忽略 `ops-transformer-master/`，源码无须上传）。换机器或新 clone 后需自行拉取：`https://gitcode.com/cann/ops-transformer`，本地这份版本 = **9.2.0**（`version.cmake:11`）。⚠️ **行号会随版本漂移** —— 按行号找不到时改用**符号名 grep**（如 `UseReadOnce`、`USE_PERMANENT_X`），别把"找不到"当成"不存在"。

### 10.0 合规前提（主办方要求"不能引用闭源软件"）

`ops-transformer-master/` 是华为 CANN **官方开源** transformer 算子库，许可证为 **CANN Open Software License Agreement Version 2.0**（`LICENSE:1`）。核对条款：

| 条款 | 内容 | 对本比赛的影响 |
|---|---|---|
| `LICENSE:16` | 授权 worldwide、royalty-free，可用于 download / use / **modify** / integrate / distribute；范围限"developing software **solely for use in systems with Huawei AI Processors**" | ✅ 昇腾真机场景**正好落在授权范围内** |
| `LICENSE:19` | 不得用于开发运行在**非华为处理器**上的软件 | ✅ 不受影响 |
| `LICENSE:21` | **不得移除/篡改版权声明** | ⚠️ 若借用了文件骨架，**必须保留原 Huawei 版权头** |
| `LICENSE:24` | 分发须附协议副本、保留 notices | ⚠️ 同上；建议在文件头加一行 `Adapted from CANN ops-transformer (CANN OSL v2.0)` |
| `LICENSE:32-33` | 对华为提专利诉讼即终止授权；无单独专利条款 | 无可操作影响 |

三方依赖清单 `Third_Party_Open_Source_Software_List.yaml:12-32` 只有 abseil / googletest / eigen / makeself / json / protobuf / libboundscheck，且全是 build/test 支撑、**不在算子源码路径内** → **无 GPL 污染、无禁止竞品条款**。`OAT.xml:17-18` 声明全仓 license=CANN-2.0，`OAT.xml:65-74` 的版权头扫描文本就是 `LICENSE:21` 那段。

> ⚠️ 但要注意：这是**开源**（源码可见）而**非 OSI 认证**许可证。"不引用闭源软件"这条规则**满足**；至于"引用开源代码"是否需要在提交里额外声明，题面无明文，**保守做法是保留版权头**。

### 10.1 本题有没有同名实现？—— 没有

全仓库 grep `mhc_expand` / `MhcExpand` / `hc_expand` 命中 **0**；`Tile` / `Repeat` / `Broadcast` / `kv_expand` 亦无同名算子。所以第一题只能找**语义邻近**的算子，不存在"直接对标件"。

（对照：第二题 `mhc/mhc_sinkhorn`、第三题 `attention/sparse_flash_attention` 都有同名实现，见 `code2.md §10` / `code3.md §10`。）

### 10.2 邻近算子候选（按有用程度排序）

本题语义已确认是**纯带宽题**：前向 `x[S,D] → o[S,m,D]` 复制广播（无乘加），反向沿中间轴 m 元求和。据此筛出的候选：

| 路径 | 它算什么 | 重合度 | 可借鉴的"部分细节" |
|---|---|---|---|
| `experimental/mhc/mhc_post/kernel/mhc_post_kernel.cpp` | `out[b*N+n]=x[b]*h[n]`，1→N 广播缩放；`h≡1` 即本题前向 | **高** | 双策略切核（per-stream `:61-76` vs read-once `:163-200`）；`UseReadOnce` 的 **4MB 阈值**（`:405-413`）；bf16 经 fp32 Cast 的路径（`:215-393`）；host 自适应 blockDim（`:482-531`） |
| `experimental/mhc/mhc_pre/kernel/mhc_pre_kernel.cpp` | `out[b]=Σ_s h[s]·x[b*N+s]`，`h≡1` 即本题反向 | **高** | 用 `s==0` 的 `Muls` 代替 `Duplicate` 清零（`:86-93`）；尾块 `DataCopyPad`（`:99-121`）。⚠️ 它**无双缓冲流水**，本题反向已优于它（§9） |
| `mhc/mhc_post/op_kernel/arch22/mhc_post_arch22.h` + `op_host/op_tiling/arch22/mhc_post_tiling_base_arch22.cpp` | 官方正式算子，含 1→n 广播 | **中高** | `USE_PERMANENT_X` 让 x 常驻 UB、内层只写不读（`:189-241`）；tiling 折半 dOuter + fp32 常驻判定（`:508-536`）；`SetL2CacheHint(CACHE_MODE_DISABLE)`（`:115-118`） |
| `mhc/mhc_post_backward/op_kernel/arch22/mhc_post_backward_arch22.h` | `grad_x=H_res·grad_out`，沿 n fp32 累加 | 中 | `[n,tileC]` 缓冲 + `Duplicate` 清零 + 批量 Cast（`:124-146, 202, 257, 265`）——对应本题反向累加段 |
| `attention/nsa_compress_grad/op_kernel/nsa_compress_grad.h:457`、`attention/flash_attention_score_grad/op_kernel/arch22/…_bn2.h:3668,3690` | arch22 上 `blockCount>1 + 一个 stride=0` 的 DMA | 中（纯手法） | 有望**一条 DMA 写 m 份副本**，省掉 m-1 次搬运。⚠️ stride 单位是**字节**、需 32B 对齐，未验证 |
| `mhc/block_attention_residuals(_grad)/op_kernel/arch22/…/block_attention_residuals_hslice.h:112-118` | 残差注意力正/反向 | 低-中 | **Kahan 补偿求和**，直接对应题面 §6 的 bf16/fp16 累加精度场景 |
| `moe/moe_token_unpermute_grad`、`mc2/moe_distribute_combine_v2/op_kernel/arch22/…:593` | scatter-add / 多块搬运 | 低 | 仅切核参考 |

### 10.3 真正需要自出的只有两点

1. **一条 DMA 出 m 份副本**的广播写法（上表倒数第 3 行是可试的手法，官方没有现成的 expand）；
2. **反向的精度补偿求和**（Kahan，或直接沿用现有 fp32 累加，见 §9）。

### 10.4 架构可编性核查（结论：能编）

`mhc/*/op_kernel` 69 个文件全部含 `__aicore__`；仓库根 `CMakeLists.txt:59-61` 明确 **`ascend910b → arch22`**，对应宏 `__CCE_AICORE__ == 220`（全仓 133 处）。`arch35` 是 950 的新式 regbase 写法（276 处），**不可直搬 910B**。

本题 `code1/op_kernel/mhc_expand.cpp` 与官方 `mhc/mhc_post/op_kernel/mhc_post.cpp:18-33` **同风格**（`KERNEL_TASK_TYPE_DEFAULT` + `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT`）→ **910B3 可编**，风格无需调整。

> 📎 顺带修正一条易混事实：官方 `mhc/mhc_sinkhorn` 的 `docs/aclnnMhcSinkhorn.md:9-12` 标注 **A2/A3 不支持**、仅 float32、`n∈{4,6,8}`（`:149-162`）。这与本题第二题的题面口径**不同源**，不要拿官方 sinkhorn 的约束去改自己的契约（详见 `code2.md §10`）。

### 10.5 待办


- [ ] （可选）借 `experimental/mhc/mhc_post` 的 **4MB read-once 阈值**思路，复核本题前向 `SPLIT_ELEMENT` / `SPLIT_STREAM` 的切换点是否还有收益空间
- [ ] （可选）试"一条 DMA 出 m 份副本"的 `blockCount>1 + stride=0` 写法 —— **先仿真机验证 stride 单位与对齐**再上真机
- ⛔ 以上都属于"抄细节不抄实现"，且**均需用户认可后才动 `code1/` 的代码**（§0 约束 1/5）

---

## 11. 真机（NPU）准备清单（2026-09-20 18:58，资源到位前先把路铺平）

> 纪律不变（工作流 §4.1 / README §7 第二步）：**上真机跑什么要先问用户**。本节只准备"到手即可执行"的材料，不代表授权开工。

### 11.1 隧道侧的前置条件（只有用户能做，两步）

1. **新环境必须在 VS Code / devenv 页面里被打开过一次** —— 否则 `~/.atomgitdevenv/.ssh/config` 里没有 Host 别名条目，脚本自举无从下手（`3GFCN` 今天就是这个状态）。
2. 打开后把**短名**告诉我，我补进 `devspace_tunnel.ps1` 的 `$EnvTable`（约定：加新环境只改这一处）。
3. ⚠️ **今天两次实测到的账号可见性问题**：转发凭据由"桌面 VS Code 当前登录账号"签发，所以会出现 `tpm0u` 查不到而 `e6z6k` 可达（18:53/18:56 同时刻反证）。判据已固化进 `-Diag`：**扩展日志写"账号环境列表里查不到 devEnvId xxx" ⇒ 是账号不是回收**。

### 11.2 ⚠️ 第一道闸：芯片口径必须先确认，不要先编译

| 事实 | 位置 |
|---|---|
| 工程两处写死 910B | `code1/CMakeLists.txt:7` `set(ASCEND_COMPUTE_UNIT ascend910b)`、`code1/op_host/mhc_expand.cpp:202` `.AddConfig("ascend910b")` |
| CPU 仿真按 arch22 编 | `code1/cpu_debug/build_cpu.sh:28` `-D__NPU_ARCH__=2201`（= 910B / arch22，见 §10.4） |
| 但控制台里那台 NPU 是 **910C（Atlas A3）** | 用户 2026-09-20 截图 `DevEnvC_3GFCN`：`1*NPU 910C / 40vCPU / 240GiB`，当时"已关机" |

⇒ **若到手的是 910C**：`AddConfig` 里没有它，包会在加载/调优阶段被拒；arch 宏与 UB 预算（本题 tiling 的一切推导都建立在"910B UB=256KB ⇒ 预算 64KB"上，§9.1）也要按真芯片重算。⇒ **先跑 preflight 拿芯片名，再决定动不动那两行**；动属于代码改动，按 §0 约束由用户拍板，且**不能为了本地能跑把比赛平台口径改坏**（口径以题面/平台为准）。

### 11.3 preflight（到手第一条命令，只读不写）

`code1/npu_debug/preflight_npu.sh`（**非提交文件**）一次连接打回全部判据：`npu-smi info` + `-t board`（**芯片名 / NPU-Arch**）、`/dev/davinci*`、`set_env.sh` 候选路径（跨环境循环探测，不硬编码）、`ccec/atc/msopst/msprof/cmake` 是否可用、**CANN 自带的 `ASCConfig*.cmake`**（`npu_op_package` 依赖，决定 §11.4 走 cmake 还是退路）、CANN 里 `ascend910*` / `davinci_22|35` 目录（现场判 910B/910C 可编性）、磁盘余量。

### 11.4 构建（两条候选路径，preflight 决定走哪条）

- **首选**：工程自带的 ASC CMake（`code1/CMakeLists.txt`）——`find_package(ASC REQUIRED)` + `npu_op_package(custom TYPE SHARED)` 产出 op 包。**前提**是 preflight 找得到 `ASCConfig*.cmake`。
- **退路**：`ccec` 单编 kernel 校验可编性（至少拿到"能否过编"的确定答案），构建脚本口径再由用户定。
- ⛔ **防假成功（工作流 §4.6）**：构建前记 `build_out` 时间戳，构建后必须看到产物**新时间戳** + `grep -a "error:"` 未被 `2>/dev/null` 吞掉；只信"产物变了"，不信"脚本回 OK"。

### 11.5 上机用例矩阵（把仿真侧欠的账一起收口）

| 来源 | 用例 | 为什么放真机 |
|---|---|---|
| §9.5 缺口 | `large` 4 条（含 `bwd-fp16/bf16-large` 8192×7168×8） | 两台云端仿真机都在这一条上失联（§9.5.1）；真机上毫秒级，比继续赌仿真划算 |
| §9.4 新增 | `D=32768 / 32769`（64KB 预算取等 vs 尾块 1 元素）、`24576 / 24577`（48KB 兜底） | 预算边界对**真芯片的 UB 大小**最敏感，正是 §9.1 整行优化的收益判定点 |
| 基线 | 官方三档 256/4096/7168 × 正/反向 × fp16/bf16 = 12 | 与 §2 契约逐条对齐 |

⚠️ **真机侧目前缺一个启动器**：`cpu_debug/test_mhc_expand_cpu.cpp` 走的是 `ICPU_RUN_KF`（CPU 仿真专用），上真机需要 ACL 侧"建包→下发 kernel→比对"的启动器（**非提交文件**，preflight 确认 CANN 版本后再写，避免按猜测的 API 版本写一版跑不通的）。

### 11.6 性能基线（E1 的定量，优化 A/B 唯一能证明自己的地方）

`msprof` 采 `aicore_time` / cube-vector 占比 / HBM 带宽利用率；**输出目录写工程内**（`code1/npu_debug/prof/`，不写 `/tmp`），采完**立刻 tar 管道回本地 + md5 复验**（§9.5.1 的教训：证据只留容器 = 会再丢一次）。

### 11.7 时间盒

preflight 5 min → 构建 20 min → 用例矩阵 30 min → msprof 20 min。任一环节隧道 down：**不重试**，先落已产出的日志再报告。

### 11.8 真机 preflight + 编译门实测结果（2026-09-20 19:38–19:53，`02aeb`）

| 判据 | 实测值 |
|---|---|
| 芯片 | **Ascend 910B3**，**NPU ID=7**（不是 §1 旧记录里的 2），驱动 `npu-smi 25.5.0`，HBM 64GB（已用 3203MB），设备节点 `/dev/davinci7` |
| 机器 | aarch64 / 16 vCPU / **122GB** 内存 / `/home` 余 179G |
| CANN | `~/Ascend/cann-9.0.0`（另有 `~/Ascend/ascend-toolkit` 并存），`ccec` / `atc` / `msopst` / `msprof` 齐备，clang 15.0.5，cmake 3.20.5 |
| `ASCEND_COMPUTE_UNIT` | **`ascend910b` 口径正确，提交源四文件零改动**（推送后 md5 逐字节比对 = §6.1 基线） |

⚠️ **两个现场坑**（下次直接照抄，别重新踩）：
1. **`npu-smi` 裸跑报 `libc_sec.so: cannot open shared object file`** —— 它不在 CANN 的 `set_env.sh` 里，在驱动目录：
   `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver` 之后就正常出表。
2. **`find_package(ASC REQUIRED)` 不需要 `ASCConfig*.cmake`**（全 CANN grep 不到这个文件名）；`npu_op_package` 宏实际在
   `~/Ascend/cann-9.0.0/aarch64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake`，**configure 只要带上 `-DCMAKE_PREFIX_PATH=$ASCEND_HOME_PATH` 就能解析**。

**编译门（本题第一次真机侧编译，此前状态是"从未编译"）**：

```bash
source ~/Ascend/cann-9.0.0/set_env.sh
cd ~/ops_comp/code1 && rm -rf build_out
cmake -S . -B build_out -DCMAKE_PREFIX_PATH=$ASCEND_HOME_PATH && cmake --build build_out -j8
```

`rc=0`、**耗时 20 秒**、产物时间戳全新（19:40:44，非"假成功"）：`MhcExpand_ascend910b` / `cust_optiling` / `cust_opapi` 全部 Built，产出 vendor 包（`build_out/tmp/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/` 下 **4 个 `.o`**，对应 `backward×dtype` 四个分支）+ `binary_info_config.json` + `libcust_opapi.so`(1.6MB)。日志本地双写：`code1/npu_debug/logs/{preflight,asc_probe,build_gate,aclnn_sig}_20260920_*.log`。

**设备侧启动器接口已确认**（`build_out/autogen/aclnn_mhc_expand.h:25,41`，标准 aclnn 两段式）：

```c
aclnnStatus aclnnMhcExpandGetWorkspaceSize(const aclTensor *x, int64_t mhcMult, bool backward,
                                           const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnMhcExpand(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

⇒ §11.5 说的"缺 ACL 启动器"现在**只剩写代码这一件事**，接口与芯片口径都不再是未知项。任务 #7 完成，#8 解除阻塞。

### 11.9 ⭐ 真机首跑 + 全量精度矩阵实测（2026-09-20 19:55–20:25，`02aeb` / 910B3 / NPU ID=7 / 50 AIV）

**结论先说**：本题**第一次在真机上跑起算子**（此前从未上过机）。跑完 5 组共 38 条用例后，真机给出一个仿真侧**完全看不见**的事实 ——

> **反向 24 条全 PASS（含逐位精确），前向 14 条全 FAIL，无一例外。** 提交源四文件 md5 仍 = §6.1 基线（**本轮零改动**）。

#### 11.9.1 启动器已建成（`code1/npu_debug/`，三文件均**非提交**）

| 文件 | 作用 |
|---|---|
| `test_mhc_expand_npu.cpp` | ACL/aclnn 启动器：用例矩阵 + 参考实现与 `cpu_debug` **逐条同源**，输出行格式一致便于跨侧对拍；额外字段 `ws=`（workspace 字节）、`nan_unwritten=`（用 `0x7fff`=fp16 NaN 预置输出缓冲，核没写到的区域一眼暴露） |
| `build_npu.sh` | 探测 `set_env.sh` → 按 `uname -m` 拼 include/lib → 校验 op 包**新时间戳 + `nm -D` 见 `aclnnMhcExpand`**（防假成功）→ 链接 `-lnnopbase -lascendcl -ldl` |
| `run_npu.sh` | 运行期组装 custom OPP 包 + 设 `ASCEND_CUSTOM_OPP_PATH` + 启动 |

用法：`bash npu_debug/run_npu.sh {quick|medium|large|mtile|bnd|ub48}`（日志自动 tee，本地已双写 30 份 `npu_debug/logs/npu_*.log`）。

#### 11.9.2 561002「Do not find tiling func」—— 首跑全红，根因是**加载方式**不是算子

现象：25/25 用例 `st=561002`。逐项排除过 `ASCEND_CUSTOM_OPP_PATH` 指 `tmp/vendors/custom`、`build_out`、`build_out/op_host`、直接指 `.so` —— 全无效。

**根因（证据链）**：CANN 9.0 的 `REGISTER_OP_LIB(custom).RegOpLibInit(...)` + `ops::OpAICoreDef::SetTiling` 在 DSO 静态初始化期写 **`LocalRegistry`**，**只有框架自己通过 `ASCEND_CUSTOM_OPP_PATH` dlopen 该 so 时才把这张表提交进全局注册表**；启动器若把 `libcust_opapi.so` 做成**链接期依赖**，它虽然被加载，注册器却只落在本 DSO 的 LocalRegistry ⇒ 框架查不到 tiling func。

**解法（已固化进脚本）**：⛔ **绝不做成链接期依赖** —— 启动器用 `dlopen`+`dlsym` 取入口（路径走 `MHC_OPAPI_SO`），`run_npu.sh` 在**运行前**按 CANN 自带 `tikcpp/ascendc_kernel_cmake/fwk_modules/scripts/install.sh` 的布局组包，让框架自己 dlopen：

```
npu_debug/pkg/custom/op_api/lib/libcust_opapi.so      ← ASCEND_CUSTOM_OPP_PATH=$PKG/custom
npu_debug/pkg/custom/op_impl/ai_core/tbe/{kernel,config}
```

改完 `err=0`，接口层彻底打通。**kernel 侧无需 OPP 路径**：`ACLNN_WITH_BINARY` 已把 4 个 `.o` 与 `binary_info_config.json` 以 `_binary_*_start/end` 符号内嵌进 `libcust_opapi.so`。

#### 11.9.3 精度矩阵（真机实测，`blk=50`）

**反向：24 条全绿，且 `maxdiff` 逐条 = 0.00000（位级精确，不是容差内通过）**

| 覆盖维度 | 用例 | 结果 |
|---|---|---|
| 官方三档 | `small` 64×256×2 / `medium` 1024×4096×4 / **`large` 8192×7168×8** × fp16/bf16 | ✅ 6/6（large 各 **0/58720256**） |
| m 扫 | m=1,2,3,4,5,16 | ✅ |
| 非对齐 D | D=7167（m=2,5,8）、D=100、D=1、D=33 | ✅ |
| 多 tile（tile=2048） | D=70000/70001/33000/33001 × m=2/8，fp16+bf16 | ✅ 8/8 |
| **UB 预算边界对** | D=**32768**(tile=32768 整行) vs **32769**(tile=2048)、D=**24576**(整行) vs **24577**(tile=2048)、D=26001 | ✅ 5/5 |

⇒ §9.5 仿真侧三次都没收口的 **`bwd-*-large`（8192×7168×8）在真机毫秒级通过**（`kern=0.001s`）；§9.4 靠纯算术推出来的预算边界，真机侧全部对得上。

**前向：14 条全红，且非确定**

| 用例 | mismatch | 备注 |
|---|---|---|
| `fwd-fp16-S1D1`（S=1 D=1 m=2，**最小可能形状**） | 1/2 | 连"1 个元素复制 2 份"都错 |
| `fwd-fp16-small` 64×256×2 | 4944/32768 (15%) | 同一用例历轮 8178 / 2046 / 2809 / 4944 ⇒ **每轮不同** |
| `fwd-fp16-medium` / `bf16-medium` | 14806024 / 14807867 of 16777216 (88%) | |
| `fwd-fp16-large` / `bf16-large` | 464593143 / 464566343 of 469762048 (**98.9%**) | S 越大越接近全错 |
| `fwd-fp16-MTL-D70000-m2` | 40033 / 53135 of 140000（同用例两轮） | |
| 其余（m=1/m=16/ROW-m8/D7167/D100/BND/UB48） | 14%~50% | 全 FAIL |

#### 11.9.4 ⭐ 前向失败机理已定位：MTE1 读 UB 与 MTE2 写 UB **之间没有任何序保证**

`op_kernel/mhc_expand.cpp:104-122`（前向是**纯搬运**，中间没有 VEC 运算）：

```cpp
auto in_buf = in_que_.AllocTensor<DT_X>();     // in_que_ = TQue<VECIN,2>  (:168)
DataCopyPad(in_buf, x_gm_[src_off], cp, pp);   // :110  MTE2  GM -> UB
in_que_.EnQue(in_buf); auto x_local = in_que_.DeQue<DT_X>();
for (k...) DataCopyPad(o_gm_[dst_off], x_local, cp);   // :119  MTE1  UB -> GM
in_que_.FreeTensor(x_local);                   // :121  只保证 VEC 已消费，不保证 MTE1 已读完
```

`VECIN` 队列的序是**挂到 VEC 消费**上的；前向没有 VEC 指令，**MTE1 可能在 MTE2 落地前就开读** ⇒ 读到 UB 里的陈旧内容。三条独立证据互锁：

1. **错值是"UB 脏数据"而非"搬错行"**：`large` 用例首帧 `got=0.0078 / 0.0000 / 0.0039`，即位图案 `0x0001 / 0x0000 / 0x0002`（fp16 次正规）。而输入按 §11.9.3 的填充**全是 0.125 的整数倍**，输出里绝无可能出现这些值 ⇒ 不是偏移算错，是**源头就没数据**。
2. **写没落地的区域可区分**：输出预置 `0x7fff`（fp16 NaN），全部用例 `nan_unwritten=0` ⇒ 空间**确实被写过**，只是写的是脏数据。
3. **规模相关性**：ROW 模式每任务把同一块 UB 连发 m 次 MTE1，`large`（S=8192, m=8, tile=7168）窗口最宽 ⇒ 98.9%；小形状只 15%。**同形重复跑计数每次不同** ⇒ 竞争而非算术错误。

**为什么反向不受影响**（`op_kernel/mhc_expand.cpp:126-165`）：反向在 VECIN 缓冲上插了 `Cast`/`Add`（:154-155），**VEC 消费即构成 MTE2→VEC 的序**，写出又走 `TQue<VECOUT>`（:159-164，保护 VEC→MTE1）⇒ 全链有序，实测 24/24 位级精确。**这也解释了为什么仿真全绿**：`ICPU_RUN_KF` 不实现流水线事件，串行执行 ⇒ 该竞争在 CPU 仿真上**原理上不可能被发现**（工作流 §4.6「仿真假通过」的又一实例）。

**排除掉的可能**：`out` 总字节 939,524,096 < 2^31 ⇒ 32 位字节偏移不溢出；`dst_off`/`src_off` 均为 `int64_t`；`tile=7168` → `cur_h*2=14336` 为 32B 对齐，非 pad 分支。**与寻址无关。**

#### 11.9.5 顺带坐实的两个口径

- **真机 UB = 256KB 成立**：host 预算 `ub_size/4`=64KB ⇒ D=32768（fp16 整行 64KB）走整行、D=32769 退化为 tile=2048，与 §9.1 推导**逐位吻合** ⇒ §9 全部 tiling 推导无需重算。
- **fp16 溢出语义分歧（确定性，非竞争）**：`bwd-fp16-sat-m2`（输入 32768，m=2 ⇒ 65536 溢出）真机 `Cast(CAST_RINT)` 给 **+inf**（512/512 全错），CPU 仿真与本地参考给 **65504（饱和）**。同组 `sat-m1`（不溢出）PASS ⇒ 只有溢出分支分歧。**倾向**：PyTorch fp16 溢出也是 inf ⇒ 大概率是**我方参考**该改，而不是核该改；但需按题面/样例核实后再定（见 §11.9.6）。

#### 11.9.6 ⛔ 待用户拍板（均属改代码，按 §0 约束不自动动手）

| # | 决策点 | 候选 |
|---|---|---|
| 1 | **前向竞争怎么修**（**最高优先，当前前向真机 0 分**） | (a) 每次 MTE2 入队后、MTE1 开读前显式 `SetFlag<PIPE_MTE2, PIPE_MTE1>`+`WaitFlag` 配对；(b) 前向也改用 `TQue<VECOUT>` 走"入队即序"；(c) 任务尾 `PipeBarrier<PIPE_ALL>`（最省事，但把 m 次写出串行化，大 S 下带宽利用率有代价） |
| 2 | fp16 溢出：改**核**（加饱和）还是改**仿真参考**（接受 inf） | 需先核对题面样例是否覆盖溢出输入 |
| 3 | `cpu_debug` harness：未知 mode 现在静默返回 0，应改非零退出 | 低风险，可顺手 |

**性能基线（§11.6 msprof）暂不采**：前向正确性未定，`aicore_time` 无意义；反向已可采，但等前向修完一次性出 A/B 更省时间盒。
