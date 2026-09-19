# code2 —— 第二题 `mhc_sinkhorn`

> **本文件随题目推进持续更新**：状态 / 已排除 / 下一步证据。
> 通用流程与通用坑见根目录 `算子开发工作流.md`；连接参数见根目录 `连接信息.md`。
> ⛔ 本题目录 `code2/` 与其它两题**严禁跨界**。

> **相关文档**：[AGENT.MD](AGENT.MD)（入口） · [算子开发工作流.md](算子开发工作流.md)（通用流程/坑/纪律） · [连接信息.md](连接信息.md)（连接参数） · [official_problem_statement.md](official_problem_statement.md#第二题b组中等题mhc-sinkhorn-算子)（**本题官方题面**） · [code1.md](code1.md) · [code3.md](code3.md) · [refs/harness_sinkhorn/](refs/harness_sinkhorn)（测试脚手架）

---
## 0. 工作约束（**必须遵守**）

| # | 约束 |
|---|---|
| 1 | **每完成一轮对话，必须把进展更新到本文件（code2.md）** —— 新结论、新排除的假设、下一步方向，都要写进来 |
| 2 | 只改 code 2/ 目录下的文件，**不动其他目录**（code1 / code3 / 根目录），除非用户明确要求 |
| 3 | 不要 printf / flush / cout 调试输出，用其他手段（dump 到文件、返回值对比等） |
| 4 | 每次改动代码前先备份（cp xxx xxx.bak） |
| 5 | 一步一步验证，每步单独确认结果，不要一次性改多处 |

---
## 1. 状态速览

| 项 | 值 |
|---|---|
| 算子名 | `mhc_sinkhorn` |
| 工程目录 | `code2/`（`op_host/`、`op_kernel/`） |
| 目标芯片 | `ascend910b`（真机 910B3，64GB HBM，**coreNum = 40**，CANN 9.0.0） |
| 正确性 | ✅ **真机 5/5 通过**（平台 5 个测试点全部 Pass） |
| 性能 | ⏳ **303µs**，最优 **1.46µs**（"慢约 200 倍"），目标 **~1.5µs** |
| 病根（已定位） | **指令数**，不是数据量 —— 内层是逐元素标量访问 |
| 平台配置 | `.AddConfig("ascend910b")` |

**一句话**：正确性已经拿到，**只剩性能**。这是三题里唯一"确定性收益"已落袋的题。

> ⚠️ **但契约有未核项**（`official_problem_statement.md` 到手后才暴露）：官方要求**仅 FLOAT32**、且有 `normOut`/`sumOut` 两个可选输出；而本地对拍**全部跑在 fp16 上**，`normOut`/`sumOut` **完全没实现**。详见 §2.1 与 §7。

> ⚠️ **性能数值口径未统一**（历史遗留）：`303µs` 与 `341µs`（带 `batch=20, n=6` 前缀的实测）并存，未说明是否同一配置；目标值 `1.4µs` 与 `~1.5µs` 两处不一致。**下一步任一次上机都应明确记录形状前缀**。

---

## 2. 输入契约 / 算子语义

### 2.1 契约

> ⭐ **权威来源**：官方题面全文见 `official_problem_statement.md`（第二题章节）。下表以**官方题面**为准，代码侧偏差单独标注。

| 项 | 官方题面 | 代码侧现状（实测） |
|---|---|---|
| 输入 | `x`，形状 `(B,S,n,n)` 或 `(T,n,n)`，**仅 FLOAT32** | OpDef 声明 `{DT_FLOAT16, DT_FLOAT}`，名 `logits` ⚠️ **多支持了 fp16** |
| 输入 | `mask` — **官方题面没有这个输入** | OpDef 有 `mask`（OPTIONAL）⚠️ **且 kernel 完全未使用** |
| 输出 | `output`（与 x 一致）/ `normOut` / `sumOut`（后两者可选） | OpDef 只有 `weights` ⚠️ **未实现 normOut / sumOut** |
| 属性 | `eps`（标量 float32，建议 `1e-6`）、`numIters`（标量 int64，范围 **1~100**，建议 20） | `eps`（Float，默认 `1e-06`）、`iterations`（Int，默认 `20`）—— 名称不同、语义一致 |
| 属性校验 | `numIters` 超出 1~100 报参数无效错误 | `num_iters < 1 \|\| num_iters > 100` → `GRAPH_FAILED` ✅ 一致 |
| 逻辑 shape | 仅 3 维 `(T,n,n)` 或 4 维 `(B,S,n,n)`；**`n ∈ {4,6,8}`** | 末两维必须相等且 `n ∈ {4,6,8}`，否则 tiling 返 `GRAPH_FAILED` ✅ 一致 |
| batch 计算 | 前序维度为批量维，各方阵独立计算 | `batch = Π shape[i], i = 0..dim_num-3`（3 维时 = `T`，4 维时 = `B*S`）✅ 一致 |
| dtype | **不支持 float16/double** | 见上 ⚠️ |
| 特殊值 | 输入含 `-inf/inf/nan` → **对应位置输出 nan** | 未验证 |
| 确定性 | **默认确定性实现**，相同输入多次调用结果一致 | 未验证 |

#### ⭐ `normOut` / `sumOut`（**当前完全未实现，官方题面的重要部分**）

| 项 | 官方规格 |
|---|---|
| `normOut` | 逻辑形状 `(2*numIters, …)`；**物理** `size = 2*numIters·n·n_align·(B·S 或 T)` |
| `sumOut` | 逻辑形状 `(2*numIters, …, n)`；**物理** `size = 2*numIters·n_align·(B·S 或 T)` |
| 对齐规则 | Device 内存中 **n 维按 8 对齐**：`n_align = ceil(n/8)*8`（n=4/6/8 时 **`n_align` 均为 8**） |
| ⚠️ 关键 | **`n=4` 时物理元素数是逻辑形状的 2 倍，内存申请必须以物理 size 为准** |
| 索引语义 | `normOut` 从 **0** 开始（`0..2*numIters-1` 均有效）；`sumOut` 从 **1** 开始，**`sumOut[0]` 是占位未定义**（分配空间但不写入），**不作为正确性中间测试点** |
| 可选性 | 两者都是可选输出，传空指针即不输出 |

### 2.2 数学语义

对 batch 中**每个 n×n 矩阵**独立执行（官方 §3 原文口径）：

1. **初始化（第 1 次迭代）**：对 x 沿 **dim=-1（行方向）** 做 softmax → 加 eps 得 `normOut[0]`；对 `normOut[0]` 沿 **dim=-2** 求和 + eps 得 `sumOut[1]`；`normOut[0] / sumOut[1]` 得 `normOut[1]`（列和为 1）
2. **交替迭代**（i = 1 … numIters-1）：行归一化（得 `sumOut[2i]`、`normOut[2i]`）→ 列归一化（得 `sumOut[2i+1]`、`normOut[2i+1]`）
3. **最终输出** `output = normOut[2*numIters-1]`

- `numIters=1` 时**仅执行初始化阶段**，输出 `normOut[1]`（此时仅满足列和为 1）
- 最终得到**双随机矩阵**（行和 = 列和 = 1）

**⚠️ env 口径差异（需注意）**：官方题面中 eps 是**独立加入**的（`softmax(x) + eps`、`sum + eps`），而 kernel 侧的实现是 `rowsum + eps`**再相除**、且在**求行/列和时**就加了 eps。这个差异不是"题面 vs 实现"的精度问题，而是**公式结构差异**，已在 §7 列为待核项。

**对拍前提**：参考实现**每一步中间结果都量化回 fp16**（并先对输入 logits 做一次 fp16 量化）—— ⚠️ **但官方要求 float32**，见 §7 待核项。

### 2.3 元素规模（决定了优化方向）

| n | 每矩阵元素 | 32B 对齐 | 备注 |
|---|---|---|---|
| 4 | 16 | ✅ `16*2 = 32B` | 可用 `DataCopy` |
| 6 | 36 | ❌ `36*2 = 72B` | 起点 `72m` 也非对齐 → **必须 `DataCopyPad`** |
| 8 | 64 | ✅ `64*2 = 128B` | 可用 `DataCopy` |

> ⭐ 关键洞察：**矩阵太小（16/36/64 元素），单矩阵向量化没有收益** —— 一条 128 元素的向量指令就能装下两个 8×8 矩阵。

---

## 3. 提交文件清单

| 提交字段 | 文件 | 位置 |
|---|---|---|
| `kernel_cpp` | `mhc_sinkhorn.cpp` | `op_kernel/` |
| `tiling_h` | `mhc_sinkhorn_tiling.h` | `op_kernel/` |
| `tiling_key_h` | `tiling_key_mhc_sinkhorn.h` | `op_kernel/` |
| `host_cpp` | `mhc_sinkhorn.cpp` | `op_host/` |

- 另有三个 `CMakeLists.txt`。
- ⛔ **`op_host/` 与 `op_kernel/` 下同名不同内容的两个 `mhc_sinkhorn.cpp`，别传错。**
- ⛔ 提交前扫描：`grep -n "printf\|fflush\|cout\|调试" <四个文件>` 必须为空；**先 `--dry-run`**。

### 3.1 历史结论：只需替换 1 个文件

早期排查的结论是"比赛工程只需替换 `op_kernel/mhc_sinkhorn.cpp`，其余三个文件不动"（本地留档版 = `refs/harness_sinkhorn/mhc_sinkhorn_scalar.cpp`）。⚠️ **这条是当时的口径，当前 `code2/` 目录下文件的实际状态需以 `md5sum` + 时间戳为准**（见 §7）。

---

## 4. 已排除 / 已确认（**别重复走**）

### 4.1 ✅ 已修复的两个 Bug

#### Bug 1：`SyncAll()` 死锁 → TLE / Runtime Error **507035**

- **根因**：`SyncAll()` 是**核间**栅栏，却被放在 per-matrix 循环内。各核处理的矩阵数 = `ceil(batch/coreNum)`，`batch` 不整除核数时尾部核次数更少甚至为 0 → 先行完成的核在栅栏处**永久等待**。
- **实测对照实验**（910B3, coreNum=40）：

  | batch | 能整除 40 | 结果 |
  |---|---|---|
  | 8 / 64 / 100 / 1 / 8192 | ❌ | **死锁超时** |
  | 40 / 80 / 120 | ✅ | 成功 |
  | 任意 batch，去掉全部 SyncAll | — | **成功** |

- **修复**：每个核负责的矩阵彼此不相交，**本就不需要核间同步** → 一律用**核内** `PipeBarrier<PIPE_ALL>()`。
- ⛔ **绝对不要再用 `SyncAll`** —— 这是本题最大的坑。

#### Bug 2：写入截断 → Wrong Answer（n=4 / n=6）

- **根因**：逐元素 `o_gm_.SetValue()` 写 GlobalTensor，在 n=4/6 长度下会**截断**（表现为"每个矩阵前 N 个元素正确、其余被写成 0"）。
- **指纹匹配证据**：本地 `n=6 + batch=20` 错误率 **24.31%**，与评测测试点 5 **完全一致**。
- **修复**：改为「先写 UB 缓冲，再用 `DataCopy`/`DataCopyPad` 一次性搬出」。
- **验证**：**17 种 shape × 3 个随机 seed，全部 0.00% 错误**（`batch = 1/2/7/13/20/40/64/100/256/1024/4096` × `n = 4/6/8`）。

### 4.2 ⛔ 已排除的死路

| 尝试 | 结果 |
|---|---|
| 去掉 SyncAll 但保留其余不变 | 死锁解决，但 n=4/6 **仍有错** |
| 用 `PipeBarrier<PIPE_ALL>` 替代 SyncAll | ✅ 死锁解决（正确方向） |
| 用"均匀轮数 + 空转轮"配 SyncAll | 小 batch **仍死锁**，无效 |
| 读路径换 `DataCopyPad` / `DataCopy` / 纯标量读 | **三者结果完全相同** → 问题不在读 |
| 把 fp16 与 float 缓冲合并成一块 | 不是根因 |
| 改成三个完全独立的 UB 缓冲 | 不是根因（但保留，更安全） |
| 在标量写 UB 前后加多处 `PipeBarrier` | 只略微改善，非根治 |
| **写回改用向量搬运** | ✅ **正解** |

### 4.3 ❌ 还没有被排除（未决项）

**A. 契约层（官方题面到手后才暴露，优先级最高）**

| 未决项 | 说明 | 判据 |
|---|---|---|
| **本地对拍全在 fp16** | 官方要求**仅 FLOAT32**；本地 harness 固定 `ACL_FLOAT16`，参考实现也做 fp16 量化模拟 → **官方口径的 fp32 路径从未测过** | 用 fp32 张量重跑 harness，看是否仍 5/5 |
| **`normOut` / `sumOut` 未实现** | 官方题面的两个可选输出（含 `n_align=8` 物理对齐、`n=4` 时物理是逻辑 2 倍、`sumOut[0]` 占位未定义）当前**完全缺失** | 若平台用例要求这两个输出，会直接失败 |
| **`mask` 输入从未使用** | 官方题面**没有 mask 输入**，OpDef 里却有且 kernel 不读 | 若平台真传 mask 且其语义非恒等，会全错 |
| **`eps` 公式结构差异** | 官方：`softmax(x) + eps`、`sum + eps`；kernel：`sum + eps` 后再除、且在**求和时**就加 eps | 1e-3 量级，小于对拍阈值 1e-2 故未暴露；**若平台判据更严会暴露** |
| `-inf/inf/nan` 传播 | 官方要求对应位置输出 `nan` | 未测 |
| 确定性 | 官方要求相同输入多次调用结果一致 | 未测 |

**B. 性能层**

- 【推测】"每批矩阵只栅栏一次"能省多少开销 —— **必须实测验证**
- 【推测】向量化改写（多矩阵打包 + 40 核满核）能否把 303µs 降到 ~1.5µs
- **向量版 kernel 尚不存在**（属未完成工作）
- 平台评测表格 `Pass 0.00%` 列的单位/含义未说明

---

## 5. ⭐ 下一步证据：性能优化

### 5.1 病根推理链（已定位，实测数据支撑）

```
batch=20, n=6：
  每矩阵每迭代约 730 次标量 GetValue/SetValue
  × 20 迭代 × 1~2 矩阵/核 ≈ 1.5 万次标量访问
  UB 标量访问约 20~30 周期/次 → 30 万+ 周期
  @1.65GHz ≈ 180µs   ← 与实测 341µs 同量级，吻合
```

要降到 1.4µs，需把标量访问量**砍掉约 100 倍**。

> 📌 补充（按 `n=6` 逐语句统计，非原文明示）：初始化段约 **672** 次，单轮迭代约 **300** 次 → 19 轮 ≈ 5700，加写回 ≈ **6372 次/矩阵**。原文的"730 × 20 迭代"口径未明确是"平均值"还是"初始化段"。

### 5.2 优化方向（必须"批处理 + 满核"，单纯向量化没用）

```
例：8 个 8×8 矩阵 = 512 元素 → 4 条 128 元素向量指令
```

- 用 `Duplicate` / `Muls` / `Add` / `Exp` 等向量 API
- 读用 `DataCopyPad`（n=6 是 72B 不对齐，**必须 Pad**）
- 写回用 `DataCopyPad`（UB→GM）
- **所有核都要参与**（不要 20 个核处理 20 个矩阵、另 20 个空转）；任务划分保证核间不重叠 → 无需核间同步
- ⛔ **绝对不要用 `SyncAll`**，只用核内 `PipeBarrier<PIPE_ALL>()`
- 改完先备份：`cp mhc_sinkhorn.cpp mhc_sinkhorn.cpp.bak`

### 5.3 判据（每次上机都要给）

| 检查 | 阈值 | 说明 |
|---|---|---|
| 逐矩阵元素级对拍 | `dev = max(abs(ref-got)) > 1e-2` 判异常 | `sinkhorn_ref.py check` |
| 与 `1/n` 的最大偏差 | `max_dev < 5e-3` → PASS | `test_mhc_sinkhorn.cpp` |
| 行和/列和双随机性 | 打印与 1.0 的最大偏差 | 无阈值判定 |
| 回归覆盖 | **n=4/6/8 × 多种 batch，全部 0.00% 错误** | 改动后**必须**跑 |

---

## 6. 测试方法（脚手架在哪）

> ⚠️ 本题的测试脚手架**已归档到 `refs/harness_sinkhorn/`**（原在 `code1/_harness/`，历史原因），**不在 `code2/`**。

| 文件（均在 `refs/harness_sinkhorn/`） | 用途 |
|---|---|
| `mhc_sinkhorn_scalar.cpp` | ⭐ **当前 5/5 正确版 kernel**（纯标量、无 SyncAll、无向量指令） |
| `sinkhorn_ref.py` | Python 参考实现 + 生成/对拍（`gen` / `check` 子命令） |
| `test_mhc_sinkhorn.cpp` | C++ harness（aclnn 调用、fp16 解码、行和自检、导出 `/tmp/mhc_out.bin`） |
| `mhc_sinkhorn_host_orig.cpp` | host tiling 重建桩（6 字段版，复现用） |
| `run.sh` | 一键构建 + 组装 vendor + 跑用例矩阵 |
| `build_test.sh` | `orig` / `fixed` 双 tag 对比构建 |
| `patch_trace.py` / `patch_trace2.py` | 给 kernel 的每个 `SyncAll` 前插入打点，定位死锁位置 |
| `mhc_sinkhorn_fixed.cpp` | ⚠️ **中间版本**（统一轮数 + 保留 SyncAll，**未修写回截断**，与当前结论矛盾） |

> 📌 本题的详细交接文档（原 `code1/HANDOFF.md`，7.2KB）**已删除**，其内容已全部并入本文件。

- 用例命令：`python3 sinkhorn_ref.py gen <batch> <n> <iters> <eps> <out.bin> <ref.bin> [seed]`（默认 seed=1234）
- 对拍命令：`python3 sinkhorn_ref.py check <batch> <n> <iters> <eps> <in.bin> <ref.bin>`
- 一键流程：`run.sh <kernel文件> [host文件] [tiling文件]`（默认用 `fixed_kernel.cpp` / `host_orig.cpp` / `orig_tiling.h`）
- harness 退出码语义：`0` 成功 / `2` 设备崩溃 / `124` 死锁超时（`timeout` 产生）/ 其它 = 原值
- ⚠️ **harness 固定走 fp16**（`aclCreateTensor(..., ACL_FLOAT16, ...)`），与官方"仅 FLOAT32"不一致 —— 做 §4.3-A 的 fp32 复测时需要改 harness
- 真机工作区：`~/mhc_build`（CMake 工程）、`~/mhc_test`（测试脚本与 harness）

---

## 7. ⚠️ 历史遗留：文档自相矛盾与待核项

> 这些是整理时发现的**文档口径冲突**，不是代码问题。下一步上机前先核 `md5sum` + 时间戳。

| 冲突 | 说明 | 处理建议 |
|---|---|---|
| 哪个 kernel 是终版 | `mhc_sinkhorn_fixed.cpp`（统一轮数 + **保留 SyncAll**、**未修 Bug2**）vs `mhc_sinkhorn_scalar.cpp`（**无 SyncAll**、含 Bug2 修复）。后者更晚（9/18 10:41 vs 9/18 01:02）→ 判定 **`scalar.cpp` 是终版**，`fixed.cpp` 是中间产物 | 以 `scalar.cpp` 为基线 |
| 性能数字 `303µs` vs `341µs` | `341µs` 带 `batch=20, n=6` 前缀；`303µs` 无形状前缀 | 每次上机**记录形状前缀** |
| 目标值 `1.4µs` vs `~1.5µs` | 两处不一致 | 统一为 **~1.5µs**（与最优 1.46µs 对应） |
| "20 矩阵 × 2 栅栏 = 40 次" | 实际 `scalar.cpp` 中 `PipeBarrier<PIPE_ALL>()` 每矩阵出现 **4 处** | 该计数描述作废，改按实测 |
| 评测表 `Pass 0.00%` 列 | 疑似把平台页面显示格式原样抄入 | 忽略该列，只看错误占比 |
| `eps` 加的位置 | kernel 在**求行/列和时**加 eps；参考实现在**除法分母**加（且先取倒数再乘） | 偏差约 1e-3 < 对拍阈值 1e-2，故未暴露；若追求逐 bit 一致需统一 |
| `mask` 输入 | kernel **完全未使用** | 若平台用例真的依赖 mask，会全错 —— **待确认** |

---

## 8. 方法论教训（本题贡献的通用经验）

1. **看到"数值全错"先怀疑测量工具**：harness 的 fp16 解码曾用 `1u << (exp-15)`，`exp<15` 时移位为负 = UB，把正确的 `0.125` 读成 `5.37e8`，**误判 kernel 有 bug**，白查好几轮。改用 `std::ldexp` 后偏差立刻为 0。
2. **别用"陈旧构建"下的结论**：几次诊断自相矛盾，根因是变体**没真正编译进去**。每次改完必须确认构建时间戳/产物确实更新。
3. **交叉验证优于单点判断**：用"错误占比指纹"反推出测试点 5 = `n=6 batch=20`（**24.31% 精确匹配**），这是定位问题的关键突破。
4. **控制变量做对照实验**：死锁根因的确认靠的就是"同一 kernel 只改 batch"的对照。
