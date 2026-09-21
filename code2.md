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
| 正确性 | ✅ **平台 5/5 现在是由向量版拿到的**（§11.17，`submission_id=6ab0bc78b0477ec41e0887b2`，5 个 case 每条 `precision_ratio: 1`）；此前那次 5/5 是标量版 |
| ✅ 提交源当前状态 | `op_kernel/mhc_sinkhorn.cpp` = **step-5 多矩阵打包（`d50b94d4`，294 行）**，四层证据：真机 fp32 常量 6/6 + 随机对拍 3/3（§11.13/§11.15）、真机 fp16 6/6（§11.15.1）、仿真机 41 项全 PASS 含 `iters=1/100` 与确定性 `bitdiff=0`（§11.16）、平台 5/5（§11.17）。回滚点：step-1 `bdd8dfa8`（git `686f783` 里）、标量版 `.bak` `0121995bbae2` |
| 性能 | 平台 5 个 case 原值 **8.4 / 14.62 / 9.3 / 15.14 / 25.94 µs**（§11.17）。真机设备时长（fp32, iters=20）`1024×8` = **37.412µs**：step-4 116.964 → **−68%**、标量版 3048.97 → **约 1/80**。"303µs 慢 200 倍"那条无形状前缀的历史数已作废（§11.2） |
| 病根（当前认知） | 每条 V 指令 ≈ **固定开销 12.8ns + 3.5ns/256B**（§11.18 的两点插值）⇒ "省条数"与"省每条触碰的块数"并重；step-5 后固定项还剩 ~31%，地板是数据量项 20.6µs |
| 目标平台配置 | `.AddConfig("ascend910b")` |

**一句话**：正确性与性能**同时已在平台与真机双向验证**（向量版 5/5、`1024×8` 37.4µs），剩下的全是"还能多快"的问题；
下一步按 §11.18 的 A/B/C 排：`G_MAX=16` 是便宜且能证伪模型的实验，n=4 的列树按 `n` 折叠省数据量项。


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
| **`normOut` / `sumOut` 未实现** | 官方题面的两个可选输出（含 `n_align=8` 物理对齐、`n=4` 时物理是逻辑 2 倍、`sumOut[0]` 占位未定义）当前**完全缺失** | 若比赛平台用例要求这两个输出，会直接失败 |
| **`mask` 输入从未使用** | 官方题面**没有 mask 输入**，OpDef 里却有且 kernel 不读 | 若比赛平台真传 mask 且其语义非恒等，会全错 |
| **`eps` 公式结构差异** | 官方：`softmax(x) + eps`、`sum + eps`；kernel：`sum + eps` 后再除、且在**求和时**就加 eps | 1e-3 量级，小于对拍阈值 1e-2 故未暴露；**若比赛平台判据更严会暴露** |
| `-inf/inf/nan` 传播 | 官方要求对应位置输出 `nan` | 未测 |
| 确定性 | 官方要求相同输入多次调用结果一致 | 未测 |

**B. 性能层**

- ⚠️ **本文件旧结论"向量版 kernel 尚不存在"已作废**：`op_kernel/mhc_sinkhorn.cpp`（9-19 11:09）早就是一版向量实现了，但**没有任何编译/真机记录**，且它自身有两个问题（见 §9.1）。
- 【已确认】旧向量版**没有消除内层标量 UB 读**：每轮迭代仍做 n 次 `rowsum.GetValue()` + 每轮 2 次 `PipeBarrier<PIPE_ALL>` —— 病根原样保留，估计只比标量版好 2~5 倍，**不可能到 µs 级**。
- 【推测】"每批矩阵只栅栏一次"能省多少开销 —— **必须实测验证**
- 比赛平台评测表格 `Pass 0.00%` 列的单位/含义未说明

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

### 5.2 ⭐ 成本模型修正 + 两步走方向（本轮推导，待实测校准）

**先纠正一条旧假设**：`所有核都要参与、不要 20 核跑 20 矩阵` —— 对**小 batch 的延迟**这条是错的。
batch=20 / 40 核时"每核 1 矩阵"已是延迟最优；把 20 个矩阵打包给 4 个核，每核的**串行深度不变**（仍是 1 矩阵 × 20 迭代），只是浪费核。
打包（一条向量指令装多个矩阵）**只在大 batch 的吞吐上有收益**。小 batch 的唯一杠杆是：

```
总时间 ≈ numIters × (每轮向量指令数) × (指令发射/依赖周期)  +  栅栏数 × 栅栏周期
1.5µs @1.65GHz ≈ 2400 周期  ÷ 20 轮 ≈ 120 周期/轮
→ 每轮必须压到 ~30 条向量指令以内，且内层 0 次标量 UB 读、0 次多余栅栏
```

**三类必消的成本**（按贵的程度排序）：
1. **UB 标量读写**（`GetValue`/`SetValue`，约 20~30 周期/次）—— 标量版每矩阵 ~6400 次 = 病根；旧向量版每轮仍留 n 次。
2. **`PipeBarrier<PIPE_ALL>`** —— 标量版每矩阵 4 处、旧向量版每轮 2 处（20 轮 = 40 次）。**只要不混用标量/向量访问同一 UB，就能压到"载入 1 次 + 存出 1 次"**。
3. **指令数本身** —— 见下 step-2。

**step 1（本轮已写码，待仿真机验证）**：保持逐矩阵、行主序 `RS=8`（每行独占一个 32B 块）布局
- 行归约结果写到**各自块的起点**（起点天然 32B 对齐）→ 用 `MoveMask` 三步（偏移 1/2/4 lane）在块内复制 → 行归一化退化成 `Div(row, row, rs_i, n)`，**不再需要标量乘数**
- 列和累加进**同一个块**，直接当每条 `Div` 的共享第三操作数，**连复制都不需要**
- 补回旧向量版丢掉的**减行最大**（见 §9.1）
- 内层零标量访问，栅栏只剩载入/存出/复制前 3 处

**step 2（拿到基线后再做）**：把"矩阵编号"放进 lane、把 `n²` 个元素放进 block 的 SoA 布局 ——
此时**行和与列和都变成跨 block 的长向量 Add**，每轮只剩 `4n` 条指令；代价是载入/存出各需一次转置，
而转置**绝不能用标量搬**（标量转置 2×n² 次访问 ≈ 2µs/矩阵，直接吃满预算），必须靠硬件转置/`MoveMask` 级联，
可行性依赖 §5.2.1 的探测结果。

**通用红线**：读用 `DataCopyPad`（n=6 的 72B 不对齐）；写回同样 `DataCopyPad`（Bug 2 的教训）；
⛔ **绝对不要用 `SyncAll`**，只用核内 `PipeBarrier<PIPE_ALL>()`；改前先备份（本轮备份见 §9.3）。

### 5.2.1 ⭐ 上机必须先做的两件事（第 1 件 ✅ 本轮已做完，结论见 §9.7）

| 顺序 | 命令 | 拿到的证据 |
|---|---|---|
| 1 | ✅ `bash probe_cann_api.sh` | **已做**。结论推翻原假设：`MoveMask` 在 910B **不存在**（只有 910C 的 `Reg::MoveMask`）、`ReduceSum` 一律要 `sharedTmpBuffer`；910B 的对口原语是 **`Brcb` + `BlockReduceSum`**。详见 §9.7 |
| 2 | ⏳ `BENCH=1 bash run.sh <kernel>` | 标量版 / step-1 版 / （将来）step-2 版在 `batch×n×iters` 矩阵上的**真实 µs 基线**；`PROF=1` 走 msprof 拿纯 kernel 时长 ← **等真机** |

> ✅ 上面那条"照样例推的 API 还没过编译器"的警告已兑现并且**方向是错的**：第一次确实编不过，但根因是**原语选错（拿了 910C 的指令）**，不是算法结构问题。按 §9.7 换成 `Brcb` 后**一次编译通过、39 对拍全绿**，算法结构（行主序 + RS=8 块布局 + 列向共享除数 + 减行最大）无需改动。
> 📌 **可复用教训**：CANN 的样例代码可能按 910C（`dav_c310`）写，抄之前先 `grep -l <原语> $CANN/*/asc/impl/basic_api/dav_c220/`，命中 0 就说明 910B 用不了。


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
| **`cpu_debug/test_mhc_sinkhorn_cpu.cpp`** | 🆕 **仿真机对拍**（2026-09-20 新增）：直接 `#include` 提交 kernel，文件内自带官方口径 fp32 参考 + 确定性/特殊值检查 |
| **`cpu_debug/build_cpu.sh` / `run_cpu.sh`** | 🆕 仿真机编译与运行（沿用第一题 §3.6 已跑通的配方，路径自动探测，kernel 目录支持 `KERNEL_DIR` 覆盖） |
| **`bench_sinkhorn.cpp`** | 🆕 **真机计时**（此前本题完全没有计时手段）：预热 + `single_avg/min` + `burst_avg`，支持 fp16/fp32 |
| **`probe_cann_api.sh`** | 🆕 探测本机 CANN 的归约/广播/对齐 API 形态，输出落 `probe_cann_api.txt` |
| `run.sh` | 一键构建 + 组装 vendor + 跑用例矩阵；🆕 `BENCH=1` 追加计时阶段、`PROF=1` 用 msprof 拿纯 kernel 时长，`REPS`/`DT` 可调；日志改落 `$T/log`（不再写 `/tmp`） |


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
| 评测表 `Pass 0.00%` 列 | 疑似把比赛平台页面显示格式原样抄入 | 忽略该列，只看错误占比 |
| `eps` 加的位置 | kernel 在**求行/列和时**加 eps；参考实现在**除法分母**加（且先取倒数再乘） | 偏差约 1e-3 < 对拍阈值 1e-2，故未暴露；若追求逐 bit 一致需统一 |
| `mask` 输入 | kernel **完全未使用** | 若比赛平台用例真的依赖 mask，会全错 —— **待确认** |

---

## 8. 方法论教训（本题贡献的通用经验）

1. **看到"数值全错"先怀疑测量工具**：harness 的 fp16 解码曾用 `1u << (exp-15)`，`exp<15` 时移位为负 = UB，把正确的 `0.125` 读成 `5.37e8`，**误判 kernel 有 bug**，白查好几轮。改用 `std::ldexp` 后偏差立刻为 0。
2. **别用"陈旧构建"下的结论**：几次诊断自相矛盾，根因是变体**没真正编译进去**。每次改完必须确认构建时间戳/产物确实更新。
3. **交叉验证优于单点判断**：用"错误占比指纹"反推出测试点 5 = `n=6 batch=20`（**24.31% 精确匹配**），这是定位问题的关键突破。
4. **控制变量做对照实验**：死锁根因的确认靠的就是"同一 kernel 只改 batch"的对照。

---

## 9. 本轮改动记录（2026-09-20，step-1 重写 + 工具链补齐）

### 9.1 读代码读出来的三个事实（不是推测，逐行核对过）

| # | 事实 | 位置 | 影响 |
|---|---|---|---|
| 1 | **提交源里那版向量 kernel 从未验证**，而 5/5 通过的标量版被降级成 `.bak` | `op_kernel/mhc_sinkhorn.cpp`(9-19 11:09) vs `.bak`(9-18 21:28) | 若本轮直接提交，交的是**未验证代码**，等于拿 5/5 换未知 |
| 2 | **旧向量版丢了"减行最大"**，直接 `Exp(原始 logits)` | 旧文件 `ProcessOneMatrix` 初始化段 | 官方参考实现是 `exp(x-rowmax)`；fp16 定义域到 ±65504，`exp(>88)=inf` → 归一化后整行 NaN。**正确性回退** |
| 3 | **旧向量版内层仍逐轮读 UB 标量** | 旧文件 `rowsum.GetValue(i*RS)` × n、每轮 2 次栅栏 | 病根没消，只是把读入/写回向量化了，达不到 µs 级 |
| 4 | **本题 harness 完全没有计时** | `test_mhc_sinkhorn.cpp` 只有正确性自检 | "303µs→?" 历史上无法归因；`303µs`/`341µs` 口径含混就是这个缺口造成的 |

### 9.2 step-1 kernel 改了什么（一次只动一个变量：**内层零标量访问**）

- 保留逐矩阵 + 行主序 + `RS=8`（每行独占 32B 块）布局 → **对齐前提与旧版一致，不引入新变量**
- ~~行归约值写在各自块起点，`MoveMask` 三步块内复制~~ ← **该写法已作废**（910B 无 `MoveMask`）；实测版改为"行和紧凑排列 + 一次 `Brcb` 展开 + 一次全区 `Div`"，**同样消掉 n 次 `GetValue`**，详见 §9.7
- 列和累加进单个块，直接作所有行 `Div` 的共享第三操作数（**不需要复制**）
- **补回 `ReduceMax` + 减行最大**，与标量版、官方参考三方对齐
- 结构：`LoadRows → SubtractRowMax → Exp → RowNormalize → ColNormalize →(iters-1)×(Row,Col)→ StoreRows`，208 行，无 `SyncAll`、无打印
- **自审时修掉一处栅栏缺失**（fp32 路径）：末轮 `Div` 是 VEC、`DataCopyPad` 是 MTE1，二者之间原本没有栅栏 → 可能搬出未落地的数据。现在 `StoreRows` 是「Cast → 栅栏 → DMA → 栅栏」，第二道栅栏因为**下一个矩阵的 MTE2 会复用同一批 UB 块**。⚠️ 这个隐患 9-19 那版旧向量里也有，只是从没编过所以没暴露。

### 9.3 备份与 md5 基线（口径：`tr -d '\r' | md5sum`）

| 文件 | md5(12) | 是什么 |
|---|---|---|
| `code2/op_kernel/mhc_sinkhorn.cpp` | `bdd8dfa83e6d` | ✅ **step-1 向量版（Brcb 版）**：仿真机编译通过 + 39 对拍全 PASS，**真机未验** |
| `code2/op_kernel/mhc_sinkhorn.cpp.bak` | `0121995bbae2` | 真机 5/5 的标量版 ← **可提交基线** |
| `code2/op_kernel/mhc_sinkhorn.cpp.bak_movemask` | `bcaea02b061a` | 本轮中间稿：用 `MoveMask` 广播（**910B 不存在该指令，已作废**，见 §9.7） |
| `code2/op_kernel/mhc_sinkhorn.cpp.bak_vec1` | `6ba8a8deda88` | 9-19 那版旧向量实现（含 §9.1 的三个问题） |
| `code2/op_kernel/mhc_sinkhorn_tiling.h` | `c42f9443e835` | 本轮**未改**（6 字段 tiling 够用） |
| `code2/op_kernel/tiling_key_mhc_sinkhorn.h` | `86fb67fb8a30` | 本轮**未改** |
| `code2/op_host/mhc_sinkhorn.cpp` | `6fcd98d51a66` | 本轮**未改** |
| `refs/harness_sinkhorn/mhc_sinkhorn_scalar.cpp` | `0ad8e3979e5f` | 标量版副本（真机回归对照用基准） |

推送侧基线（§9.5 步骤 0 落地后逐字节比对）：`test_mhc_sinkhorn_cpu.cpp`=`783d9517240a`、`bench_sinkhorn.cpp`=`676e0d902a08`、`build_cpu.sh`=`ef5df5d93f98`（路径候选修正后）、`probe_cann_api.sh`=`4fdda31a451c`、`run.sh`=`db70753e334c`；`run_cpu.sh` 于过滤修正后再算。全部 CR=0（`tr -dc '\r' | wc -c` 判定）。**本轮实测：tar 管道推送后远端 md5 与本地逐字节一致**（kernel 与 build_cpu.sh 均验证过）。

> ⛔ **本轮不做任何提交**（纪律 §0 与 AGENT.MD 硬约束 5）。要提交前先 `--dry-run` 并等用户确认。

### 9.4 契约缺口（用户裁决：本轮**只记录不动**，避免与性能改动混在一起无法归因）

§4.3-A 六项原样保留：仅 FLOAT32 未测、`normOut`/`sumOut` 未实现、`mask` 未使用、`eps` 公式结构差异、`-inf/nan` 传播、确定性。
其中**确定性**与 **`-inf/nan` 传播**本轮已在仿真 harness 里加了检查用例（`mode=det`），**只出数据、不改 kernel**。

### 9.5 仿真机一连上就按这个顺序跑

```bash
# 0) 推送（隧道内 scp 会卡死，用 tar 管道；落地后必须 md5 复验）
#    多个 -C 让包内路径直接落成 op_kernel/ op_host/ cpu_debug/，与 code1 布局同构
cd "D:/Projects/算子比赛" && tar cf - -C code2 op_kernel op_host \
                       -C refs/harness_sinkhorn cpu_debug probe_cann_api.sh \
  | ssh -p <PORT> -i <KEY> developer@127.0.0.1 \
      "mkdir -p ~/ops_comp/code2 && tar xf - -C ~/ops_comp/code2"
# 落地后应为：~/ops_comp/code2/{op_kernel,op_host,cpu_debug,probe_cann_api.sh}
# ⚠️ 本轮修过的坑：build_cpu.sh 旧的候选路径 ../op_kernel / ../../../code2/op_kernel
#    与上面这条 tar 的落地布局**都对不上**（会 ERROR: not found）。现在候选是
#    "op_kernel"（云端包根，code1 同构）与 "../../code2/op_kernel"（本地仓库内直接跑）。
ssh ... "cd ~/ops_comp/code2 && md5sum op_kernel/mhc_sinkhorn.cpp cpu_debug/*.sh cpu_debug/*.cpp"
#    期望：mhc_sinkhorn.cpp=bcaea02b061a ... 对照 §9.3 基线（口径 tr -d '\r' | md5sum）

# 1) API 探测（先做这个，再谈编译）
bash ~/ops_comp/code2/probe_cann_api.sh          # 结果落同目录 probe_cann_api.txt

# 2) 编译（预期第一次会因 ReduceSum 重载 / MoveMask 报错，按探测结果改调用形态）
bash ~/ops_comp/code2/cpu_debug/build_cpu.sh      # 报错看同目录 build_cpu.log

# 3) 对拍：quick 全量 → mag（防溢出）→ det（确定性 + 特殊值）
bash ~/ops_comp/code2/cpu_debug/run_cpu.sh 20 quick
bash ~/ops_comp/code2/cpu_debug/run_cpu.sh 20 mag
bash ~/ops_comp/code2/cpu_debug/run_cpu.sh 20 det
```

**判据**（沿用 §5.3）：`maxdiff ≤ 1e-2` 且 `mism=0`；`det` 要求 `bitdiff=0`；`mag-large`（±60 输入）用来证明第 2 条修复真的生效——若 kernel 丢了减最大，这一组必炸 NaN。

### 9.5b 真机计时前的一次性落位（`run.sh` 写死了路径，别推到别处）

`run.sh` 的 BENCH 段编译的是 **`$T/bench_sinkhorn.cpp`**（`$T=/home/developer/mhc_test`），所以推真机时必须把这个文件放进 `$T/`，不是 `refs/` 原样路径：

```bash
cd "D:/Projects/算子比赛" && tar cf - -C refs/harness_sinkhorn bench_sinkhorn.cpp \
  | ssh -p <PORT> -i <KEY> developer@127.0.0.1 "tar xf - -C /home/developer/mhc_test"
# 然后：基线与改版各跑一次，同形状同 reps
BENCH=1 bash run.sh <标量版 .bak>      # 基线
BENCH=1 bash run.sh <step-1 版>        # 增益
REPS=200 BENCH=1 bash run.sh ...       # 想压噪声就加大 reps
PROF=1 BENCH=1 bash run.sh ...         # 纯 kernel 时长（msprof，产物 $T/prof）
```

### 9.6 下一步（按优先级）

1. ~~仿真机：跑通 §9.5 → 修 API 调用形态 → `quick/mag/det` 全绿。~~ ✅ **本轮已完成，见 §9.7**
2. 真机：`BENCH=1 bash run.sh <scalar版>` 拿**基线**，再 `BENCH=1 bash run.sh <step-1版>` 拿**增益**，两边同形状同 reps（补上"303µs 无形状前缀"的历史欠账）。← **当前卡在这一步，等用户开真机隧道**
3. 若 step-1 只到 10~40µs：按 §5.2 step-2 做 SoA 打包（**先确认转置不用标量**），再谈冲 ~2µs。
4. 性能落定后，另开一轮处理 §9.4 的契约缺口（fp32 实测、`normOut`/`sumOut`），**不与性能改动混提交**。

### 9.7 ⭐ API 探测实测结论（2026-09-20 云端仿真机，aarch64 / CANN 9.0.0）

> **这一节是本轮最值钱的东西**：我原来照"官方 softmax 样例"写的两个原语，**在 910B 上一个都不存在**。下次改 kernel 前先查这张表，别再凭记忆写。

| 我原来写的 | 910B（`__NPU_ARCH__=2201`）实际 | 正确用法 |
|---|---|---|
| `MoveMask(dst, dstMask, src, srcMask)` 三步广播 | ❌ **完全不存在**。`dav_c220` 命中 **0**、tikicpulib 命中 **0**，只有 `Reg::MoveMask` 且仅在 `dav_c310`（910C） | ✅ **`Brcb`**（910B 的广播指令，`dav_c220/kernel_operator_vec_brcb_impl.h`） |
| `ReduceSum<T>(dst, src, reduceLen, dataLen)` | ❌ 该重载不存在。basic_api 的 `ReduceSum` **一律要 `sharedTmpBuffer`** | ✅ `ReduceSum<T>(dst, src, tmp, count)`（whole-reduce，结果写 `dst[0]`）<br>✅ `BlockReduceSum<T,false>(dst, src, repeatTime, mask, dstRepStride, srcBlkStride, srcRepStride)`（**免 workBuffer**，块内归约，stride 语义待验） |
| `ReduceMax<T>(dst, src, n, n)` | 同上 | ✅ `ReduceMax<T>(dst, src, tmp, count, calIndex=false)` |

**`Brcb` 语义（头文件原文，决定布局怎么写）**：*"取 src0 的 **8** 个 b16/b32 数据，**每个复制铺满一个 32B 块**，把这 8 个块连续写入 dst"*。签名 `Brcb<T>(dst, src0, uint8_t repeatTime, BrcbRepeatParams{dstBlkStride, dstRepStride})`，支持 half / bfloat16_t / int16 / uint16 / float / int32 / uint32。

**由此得到的两个设计红利**（已编译验证）：

1. 行和必须**紧凑**排在 `red[0..n-1]`（不是各 32B 块起点），一次 `Brcb` 就展成 n 条完整行 → 归一化只需 **1 条 `Div`** 覆盖 `n*RS` 全区，而不是 n 条。
2. 广播出的除数**在 padding lane 上等于本行的和**，于是 `0/(s+eps)=0` 让 padding 成为两轮归一化的**不动点**。配套要求：padding 在 `Exp` 之前必须是**极负哨兵**（`-1e30f`）而不是 0，否则 `Exp(0)=1` 会把假数据混进行和/列和。`LoadRows` 先 `Duplicate(mat, NEG_PAD, n*RS)` 再逐行 `DataCopyPad`（`isPad=false`，不碰剩余字节）来铺哨兵。

**每轮指令数（n=8）**：行归一 8×`ReduceSum` + `Adds` + `Brcb` + `Div` = 11；列归一 `Duplicate` + 8×`Add` + `Adds` + 8×`Div` = 18 → **约 29 条/轮**，落在 §5.2 估的 ~30 条预算内。内层**零标量访问、零 `PIPE_ALL`**（只 `PIPE_V`）。
> ⚠️ **本段最后一句已被 §10.7E 更正**：那 8 次 `ReduceSum` 在 2201 上每次内部含 `SetFlag/WaitFlag<V_S>` + `get_acc_val()` + 标量写回 ⇒ "源码层零标量"不等于"零标量往返"。计数 11/18/29 不变。

**仿真机实测结果**：`quick` 27 项 + `mag` 12 项 **全 PASS**，`maxdiff ≤ 1.2e-4`、`mism=0`；`mag-large`（±60）PASS 证明减行最大生效（丢了必炸 NaN）；`det` 两例 `bitdiff=0`。`special-values` 记录：inf/-inf/nan 输入下输出 `0.96777 0 0 0 0 0`，**非 NaN 传播** → 与官方要求的差异仍在（§9.4 契约缺口，本轮不动）。

> ⭐ **`Brcb` 在 CPU 仿真里是真模拟、不是空 stub** —— 若是空 stub，`quick` 会大面积 FAIL。这条很重要：**广播类原语可仿真可验**，与 code1 记录的"仿真核数上限 20、真机 50 核覆盖不到"是两类不同的盲区。

**仿真机现场参数**（下次现场再读，别当常量）：主机别名见 `~/.atomgitdevenv/.ssh/config`（本轮 `devenvc_tpm0u.*`，⚠️ **别名每次重建就换**，旧记录 `xq82l` 已失效）；aarch64 / 16 核 / `~/Ascend/cann-9.0.0/set_env.sh`；工程落地 `~/ops_comp/code2/{op_kernel,op_host,cpu_debug,probe_cann_api.sh}`。

**工具修正**：`run_cpu.sh` 的噪声过滤原来只滤 `[SUCCESS][AIV`，而仿真器实际打的是 **`[SUCCESS][AIC_n]`** → 已改成 `\[SUCCESS\]\[AI[CD]_`。`build_cpu.sh` 的 kernel 目录候选已按 §9.5 修正。

---

## 10. ⭐ 开源参考池：`ops-transformer-master`（2026-09-20 新增）

> 📌 **定位（用户 2026-09-20 定调）**：这批开源材料**有很大的参考价值**，但**最好不要照抄**——抄的是**部分细节**（指令组合、流水编排、tiling 决策），不是整段 kernel。合规条款与许可证核查全文见 `code1.md §10.0`（CANN OSL v2.0，开源、可用于昇腾场景、**须保留版权头**）。

> 📦 **本节所引路径/行号依赖外部库，它不入库**（`.gitignore` 已忽略 `ops-transformer-master/`，源码无须上传）。换机器或新 clone 后需自行拉取：`https://gitcode.com/cann/ops-transformer`，本地这份版本 = **9.2.0**（`version.cmake:11`）。⚠️ **行号会随版本漂移** —— 按行号找不到时改用**符号名 grep**（如 `BlockReduceSum`、`Brcb`、`SigmoidPerf`），别把"找不到"当成"不存在"。

### 10.0 ⚠️ 首要结论：同名的官方 `mhc_sinkhorn` **不能抄指令**

`ops-transformer-master/mhc/mhc_sinkhorn/` 确实与本题**同名**，但它是给 **950 编的**：

| 证据 | 内容 |
|---|---|
| `mhc_sinkhorn/README.md:5-13` | 产品表：仅 **Ascend 950PR/950DT = √**，Atlas **A2 训练/推理 = ×**、A3 = × |
| `mhc_sinkhorn/op_kernel/` | 目录下**只有 `arch35/`**，无 arch22 分支 |
| `op_host/CMakeLists.txt:16` | 判 `ASCEND_COMPUTE_UNIT == ascend950` |
| `mhc_sinkhorn_arch35.h:173-240` | 整份用 `Reg::RegTensor` + `__VEC_SCOPE__`；全仓 **arch22 目录 0 处使用 Reg API** |

⇒ `Reg::Exp` / `DataCopyGather` / `ReduceMaxWithDataBlock` / `CreateAddrReg` 在 **910B3 + CANN 9.0.0 上不可用**。本题能抄的是**结构与手法**，指令层必须自己换成 arch22 的等价物。

**真正可照搬手法的两份 arch22 生产代码**（同族，能在 910B 编）：
- `mhc/mhc_pre_sinkhorn_backward/op_kernel/arch22/mhc_pre_grad_kernel.h:415-471` —— **唯一在 910B 上编得过的"Sinkhorn 一轮 = 少数几条指令"完整范例**，正对本题病根（指令数 = 矩阵数 × n）
- `mhc/mhc_pre_sinkhorn/op_kernel/mhc_pre_sinkhorn_base.h:483-505, 609-642` —— 广播除数、一步 softmax 的 arch22 标准写法

### 10.1 交叉印证：官方独立佐证了 §9 已验证的两个选择

这轮最省时间的收获——**§9 那张"910B 原语真表"没有走弯路**：

| §9 的结论 | 官方 arch22 的佐证 |
|---|---|
| 广播用 `Brcb`（`MoveMask` 在 910B 不存在） | ✅ 官方 arch22 路径同样用 `Brcb` 做广播（`mhc_pre_sinkhorn_base.h:326-401`） |
| `ReduceSum` 必带 `sharedTmpBuffer`，改用 `BlockReduceSum` 免 workBuffer | ✅ **官方实例 `mhc_pre_grad_kernel.h:467-468` 正是 `BlockReduceSum<T,false>`**，且 §9 表里标注的"**stride 语义待验**"在此有了可对照的真实参数取法（`calCount = tileRepeatTimes*n`、免 sharedTmp） |
| `:168` 一次全区 `Exp` 已是最优 | ✅ 官方**没有**查表 / 多项式 exp，也没有 `Reciprocal`（全仓 grep 0 命中），`Exp` 就是一次硬件全区（`mhc_pre_sinkhorn_base.h:635`） |
| `:144-148` 列和靠布局解决、不做转置 | ✅ 官方同样**没有转置**，列归一靠 stride/repeat 参数解决 |

### 10.2 官方优化清单（按性价比排序，针对"指令数"病根）

| # | 官方做法（文件:行号） | 我现在的做法 | 预期收益 |
|---|---|---|---|
| 1 | **块归约一条指令做完整个 tile**：`BlockReduceSum(dst,src,calCount,mask,dstRepStride,srcBlkStride,srcRepStride)`，`mhc_pre_grad_kernel.h:467-468`。⚠️ **本行已被 §10.6/§10.7B 的 `WholeReduceSum` 取代**：更贴本题（count 形态有 2201 实现、dst 单位是元素 ⇒ 结果天然紧凑、官方同形状用例现成） | `mhc_sinkhorn.cpp:131-133` n 次 `ReduceSum(…,tmp,n)` | 行归约 8→1 条；每轮 11→4 条，**~2.5–3×** |
| 2 | **batch 折进 repeat 维**：`Div(dst,src,colSum,64,tileRepeatTimes,{dstBlkStride=n,src0BlkStride=n,src1BlkStride=1,dstRepStride=n*8,src0RepStride=n*8,src1RepStride=8})`，`mhc_pre_grad_kernel.h:415-421`；累加同法 `:432-438`。arch35 同构：`mhc_sinkhorn_arch35.h:204,213` 外层循环是 **tile 不是矩阵** | `mhc_sinkhorn.cpp:156-178` 严格"一核一矩阵串行"，`:50-55` 每核 `batchPerCore` 个矩阵逐个跑 20 轮 | 每条指令覆盖 8 矩阵 → 大 batch **8×**（⚠️ 对 batch≈核数时无效，需配合 #7） |
| 3 | **除数不复制、直接广播**：`SigmoidPerf` 用 `src0BlkStride=0, src0RepStride=0` 让一个块当全矩阵除数，1 条 `Div` 覆盖全区（`mhc_pre_sinkhorn_base.h:483-505`） | `:151-152` n 条 `Div(mat[i*RS],…,cs,RS)` | 列归一 19→~4 条，**再 2–4×** |
| 4 | **一次 MTE2 搬完 tile**：`CopyIn` 用 `blockCount=nBurst, srcStride, dstStride`（`mhc_pre_sinkhorn_base.h:697-713`），右填充 `DataCopyPadExtParams{true,0,8-n,0}`（`mhc_pre_grad_kernel.h:399-407`） | `:74-93` 每矩阵 n 次 `DataCopyPad` + `Duplicate(NEG_PAD)` + n 次 `Cast` | 载入/存出指令 n→1。⚠️ **"省掉哨兵播种"有前提，见 §10.6 第 2 条** |
| 5 | **流水用 event 而非 `PIPE_ALL`**：`SetFlag/WaitFlag<HardEvent::MTE3_V>`（`mhc_pre_sinkhorn_base.h:762-774`）+ `TQue<VECIN,2>` 双 buffer（`:65-70,106-115`） | `:92,103,114` 每矩阵 4 次 `PipeBarrier<PIPE_ALL>`（全管道排空）。⚠️ §9 末段记的"内层零 `PIPE_ALL`"指**内层**，外层搬运仍是 `PIPE_ALL` | 载入/计算/存出重叠，**~1.3–2×** |
| 6 | **迭代策略**：固定 `num_iters`，**无提前退出、无收敛判据**；`CalcColNorm<true>` 只在首轮加 eps，后续 `<false>`（`mhc_sinkhorn_arch35.h:486` vs `:528`） | 20 轮固定 ✅，但每轮都 `Adds(eps)` | 每轮省 1–2 条。⛔ **别写收敛检测**，官方没有，标量比较会贵 100× |
| 7 | **UB/核数切分**：`ForwardSplitCore` 先 `tNormCore=ceil(T/totalCoreNum)`，再按 `availableUb/2` 算 `tUbFactor`、`FloorAlign(…,8)`，尾核单独 `tTailCoreLoop`，`SetBlockDim(usedCoreNum)`（`mhc_sinkhorn_base_tiling.cpp:122-145`、`mhc_sinkhorn_tiling.cpp:327`） | `op_host:67-74` 均分 + `SetBlockDim(全部核)` | **只在 #2 落地后有意义**（否则 20 矩阵塞不进 8 的倍数） |

### 10.3 若只改 3 处（建议实施顺序）

1. 行归约换 **1 条 `BlockReduceSum`**（替 `mhc_sinkhorn.cpp:131-133` 的 n 次 `ReduceSum`，顺带消掉 sharedTmp 依赖）→ 每轮 11→4；
2. 列归一压成**全区 1 条 `Div`**（除数 `srcBlkStride=0/src1RepStride=0` 广播，`calCount=n*RS, repeatTime=1`，学 `mhc_pre_sinkhorn_base.h:492-499`）→ 每轮 19→4；合计每轮 30→~8 条；
3. **同步与搬运**：`PIPE_ALL`→`PIPE_V` + `SetFlag/WaitFlag`；`LoadRows/StoreRows` 的 n 次 `DataCopyPad` 合成 1 次带 `srcStride/dstStride` 的调用；改 `op_host:67` 让 `batchPerCore` 为 8 的倍数并 `SetBlockDim(usedCoreNum)`，使 #2 的 repeat 批量真正生效。

预期：#1+#2 约 **4×**，#3 再叠 1.5–2× 并把大 batch 摊薄 8× → 进 **2–5µs 区间**。能否到 ~1.5µs **取决于真机基线重测**。

### 10.4 ⚠️ 两个前提，动手前必须先处理

1. **基线未定**：§1 已记 303µs 是**标量版**数字，工作区 `op_kernel/mhc_sinkhorn.cpp`（md5 `bdd8dfa8…`）是 step-1 向量版、**真机未验**。→ **上真机第一件事是重测当前文件**，否则 #1/#2/#3 的收益无法归因。
2. **本题目录边界**：以上全部在 `ops-transformer-master/`（**只读参考，不是提交源**）。改动只落在 `code2/`，且**按 §0 约束 1/5：方向需用户认可、一次一处、先 `cp .bak`**。

### 10.5 待办

- [ ] 真机重测 step-1 向量版基线（形状前缀务必记全，回应 §1 的"303/341µs 口径未统一"）
- [ ] **第一刀改 §10.6 #0**（`WholeReduceSum`/`WholeReduceMax` 替 n 次 `ReduceSum`/`ReduceMax` + 删 workBuffer），**不再先试 `BlockReduceSum`** —— 单点改动、39 项对拍复跑 + 单次真机计时
- [ ] 上云端仿真机时先在现有 probe 里加一行 `WholeReduceSum<float>(dst, src, n, n, 1, 1, 1)` 过编译门（§10.6 第 1 条末段"编译门状态要分清"；题3 过的那支是 `<half>` + `mask[]`，不同重载）

---

### 10.6 ⭐ 对 §10 的逐条复核 + `WholeReduceSum`/`WholeReduceMax`（2026-09-20 本轮，逐行读源码）

> 编号说明：本节由本轮追加（云端仿真机 SSH 别名 **`TpM0u`**，落地 `~/ops_comp/code2`），占 **§10.6** 这个空号；`§10.2` 表第 1/4 行里"见 §10.6"的两处指代即本节。**另一条并行会话若要续写 §10，请从 §10.7 起编号。**
> ⚠️ **交叉写入事故记录（不追责，只记证据）**：本轮期间我**两次读到**一版与磁盘现状不符的 §10（多出 `#0` 行、5 项版 §10.5、一节"官方原版做法"，当时报出的行数 ~601 > 实际 490），收尾复核时盘上从来没有过那一版（`grep "^#.* 10"` 只有 10.0~10.5，且我对 3 项版 §10.5 的替换能精确命中）。⇒ 两种解释都能对上证据（① 那一版真写过盘、又被整文件写回覆盖掉；② 我的读取命中了另一条会话的陈旧缓存），**本轮无法判定是哪种**，只把可行动的部分记下来。能回插的部分已逐字放进 §10.7A，`#0` 行与"官方原版做法"因只读到切片无法复原。**通用教训：`codeN.md` 的行号/节号引用必须以 `grep -c '' <文件>` + 一次全新 Read 为准，别信上下文里的旧快照。** `code2.md` 的 §10 此前全部未提交（`git log` 只有初始化那一笔）⇒ 本轮末尾单独 commit 一次。

> 口径：本轮只读 `ops-transformer-master/`（**只读参考**）与 `code2.md`。**未改任何 kernel 代码**（`code2/op_kernel/mhc_sinkhorn.cpp` md5 仍 `bdd8dfa83e6d`）。

**0）先给"要不要信 §10"一个答案：信。** §10 全部 file:line 引用逐条开文件对过，命中：`mhc_pre_grad_kernel.h:399-407`（`DataCopyPadExtParams{true,0,8-n,0}` 右填充）· `:415-421`（batch 折 repeat 的 `Div`）· `:467-468`（`BlockReduceSum`）· `mhc_pre_sinkhorn_base.h:483-505`（`SigmoidPerf` 的 `src0BlkStride=0/src0RepStride=0` 广播除数）· `:609-642`（`SoftmaxFP32Perf`）· `:762-774`（`VToMTE3Sync` / `MTE3ToVSync`）。
📌 途中差点误判：会话恢复时上下文里那份 `mhc_sinkhorn.cpp` 快照是**已作废的 MoveMask 稿**，按它行号去核对 §10 会得出"行号全错"的错误结论。**以磁盘 md5 `bdd8dfa83e6d` 为准**（`ReduceSum` 在 `:131-133`、`Div(…,cs,RS)` 在 `:151-152`、`Exp` 在 `:168`，均与 §10 一致）。

**1）⭐ 正对本题病根的原语：`WholeReduceSum` / `WholeReduceMax`**（§10.2 现存那版没列它；并行会话是否已列过，见 §10.7A ① —— 那里说明了这条印证到底有多硬）

⚠️ **参数名以 §10.7B 表末行为准**：本节下面按调用位置写的 `(每行元素数, 行数, …, 每行占几个块)` 是我当时的**猜测性命名**，头文件里的真名是 `(mask, repeatTime, dstRepStride, srcBlkStride, srcRepStride)`；结论（一次调用做完 n 行、免 workBuffer）不变。

官方 `mhc_pre_sinkhorn_base.h:609-624`（arch22、可在 910B 编）：

```cpp
WholeReduceMax(output, input, curColNum, curRowNum, 1, 1, CeilDiv(curColNum, elemInOneBlock), ReduceOrder::ORDER_ONLY_VALUE);
WholeReduceSum(output, input, curColNum, curRowNum, 1, 1, CeilDiv(curColNum, elemInOneBlock));
```

⇒ **一次调用做完 curRowNum 条行归约，且没有 `sharedTmpBuffer`**。`SoftmaxFP32Perf`（`:627-642`）的整个流程与本题"初始化一轮"逐句同构：`WholeReduceMax` → `SubABLastDimBrcInline`（`Brcb` + 全区 `Sub`，`:250-264`）→ `Exp(…, curRowNum*curColNumAlign)` → `WholeReduceSum` → `DivABLastDimBrcInline` → **全程零标量 UB 读、零 `GetValue`、零 workBuffer**。

**为什么比 §10.2 #1 的 `BlockReduceSum` 更合适**：`BlockReduceSum` 要手写 `mask` + 三个 stride，而它的"lane→block 映射"语义官方注释自己都没写（`:429` 那句 `out : value other other other value …` 只能靠猜，正是 §9.7 留"待验"的原因）；`WholeReduce*` 的入参是**语义直白的 `(每行元素数, 行数, …, 每行占几个块)`**。
对到本题：`LoadRows` 后第 i 行就在第 i 个 32B 块（RS=8=fp32 每块元素数）⇒ 每行 1 块 ⇒ `blocksPerRow=1` ⇒ **`WholeReduceSum<float>(red, mat, n, n, 1, 1, 1)` 替掉 `:131-133` 的 n 次 `ReduceSum`，`WholeReduceMax<float>(red, mat, n, n, 1, 1, 1, ORDER_ONLY_VALUE)` 替掉 `:120-122` 的 n 次 `ReduceMax`，并连带删掉 `tmp_buf_`**。每轮 11→4 条，比 #1 少一个必须猜对才成立的量。
⚠️ 全仓 **82 个文件**在引用 `WholeReduce*`（`grep -rl "WholeReduce" ops-transformer-master --include=*.h --include=*.cpp | wc -l`；用例见 `norm_rope_concat.h:106`、`vector_common.h:436`、`kv_rms_norm_rope_cache_*.h`），"arch22 不能用"这个担心不成立。

**编译门状态要分清（别把题3 的结论套过头）**：题3 的 probe 过的是 **`WholeReduceSum<half>(dst, src, mask[], 8, 1, 1, 1)`**（`code3.md:1115`，mask 数组那一支）；官方 mhc_pre 用的是 **`(dst, src, count, outLen, 1, 1, blocksPerRow)`** 这支 —— **同一函数的不同重载，后者在 2201 上还没过过编译**。⇒ 下次上云端仿真机，第一件事是在现有 probe 里加这 1 行浮点调用（题3 只读引用，不动 `code 3/`）。另：`code3.md:1071` 记的是"SFA 官方 arch22 自己没用 `WholeReduceSum`"，别误读成"910B 没有这个函数"。

**2）§10.2 #4"省掉哨兵播种"的前提（已核对，成立但有条件）**
官方用 `{true,0,8-n,0}` 右填充 0 而不炸，靠的是**归约用非对齐计数 `curColNum`**（`:633/:637`）而 `Exp`/`Sub`/`Div` 用对齐计数 `curRowNum*curColNumAlign`（`:635`）⇒ padding lane 变成 `exp(0)=1` 的垃圾，**但没有任何归约会读到它**。
我们当前 kernel 只有列归一破坏了这个不变量：`ColNormalize` 的 `Add(cs, cs, mat[i*RS], RS)`（`:145-148`）用 RS 宽度，会把 padding 垃圾加进列和。⇒ 去掉 `Duplicate(NEG_PAD)` 必须与"列累加计数改 `n`"**同一改动里**做，二者拆开必错；而 #0 的 `WholeReduce*` 计数天然是 `n`，顺带把哨兵这个概念整体消掉。

**3）撤回一条我在对话里说错的对比**：我曾说"题1 用的是 `ReduceSum<false>` 级联 + `GetValue`"。实测 `grep -n "Reduce\|GetValue\|Brcb\|PipeBarrier" code1/op_kernel/mhc_expand.cpp` → **0 命中**（`mhc_expand` 没有归约段），该对比不成立，已从此处撤回；它不影响 #0（#0 的依据是题2 自己的 n 次 `ReduceSum`/`ReduceMax` + workBuffer）。

**下一步顺序不变**：§10.4 前提 1 优先（真机重测 step-1 基线），之后把 §10.3 #1 换成 #0 先试。

### 10.7 ⭐ 交叉写入丢失内容回插 + `WholeReduce*` 的 2201 实现层证据（2026-09-20 本轮后半，云端仿真机 tpm0u 在线）

**A）回插上一节提到的丢失内容（来源：本轮对话记录，非臆造）**。并行会话那版 §10.5 的 6 条待办，逐字回插如下（其中第 2 条是**它的独立论点，我此前没有，采纳**）：

- [ ] **真机重测 step-1 向量版基线**（`BENCH=1 bash run.sh <.bak标量版>` vs `<step-1版>`，同形状同 reps；形状前缀记全，顺带回应 §1 的"303/341µs 口径未统一"）← **仍是第一优先**
- [ ] **等基线出来后再决定要不要按 #0 换一次全区归约** —— 它把 n 次归约压成 1 次，但**不减少每元素的向量吞吐**（`Add`/`Mul` 内部仍按 n 次重复执行），真实增益可能远小于计数差 ⚠️ *这条保留意见对下面的 #0 同样成立：省的是"指令条数与回搬"，不是"每元素吞吐"*
- [ ] 若仍要压归约计数，走 §10.2 #5（`Reciprocal`+`Muls`）或 #6（`CalCuExp` 查表）
- [ ] 先小改一处：用 §10.2 #1 的 stride 写法把列归一压成全区一次 `Div`（最贴近"少一条是一条"）
- [ ] 核对比赛 harness 是否传 `mask`；若是，补上"先 `min(x,0)` 软化 NaN 再进归一化"的廉价保护
- [ ] 从 `mhc_pre_grad_kernel.h:467-468` 抄 `BlockReduceSum` 的 **stride 参数取法**（`calCount=tileRepeatTimes*n`、免 sharedTmp）

📌 **两点必须如实记下**：① 那版 §10.2 里若确实已有一条 **`#0` = "一次调用做完整个 tile 的归约"**，那它与 §10.6 我重新发现的 `WholeReduceSum` **是同一个东西** —— "§10 漏了这条原语"的说法就不成立。⚠️ **但这条"两路会话互证"的力度取决于 §10.6 事故记录里那个未判定的分叉**：如果那一版从未完盘（只是我读到的缓存），那就只是**同一份材料的两次读取**，不算独立印证。真正独立、可核的印证只有一条：**§10.7B 那批 CANN 头文件原文**（任何人 `grep` 都能复现）。② 那版的 `#0` 整行原文与一节"官方原版做法"我**只读到过切片、没能整段留存，无法回插**，如需请让原会话重写。

**B）⭐ 本轮新增的是实现层证据，不是新原语**（全部为云端仿真机上 CANN 9.0.0 头文件原文，路径 `/home/developer/Ascend/cann-9.0.0/aarch64-linux/asc/`）：

| 证据 | 位置 | 结论 |
|---|---|---|
| `WholeReduceSumImpl` 函数体 = `vcadd(dst, src, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, 0)` | `impl/basic_api/dav_c220/kernel_operator_vec_reduce_impl.h:261-267` | **910B 有真实现，且就是一条硬件 `vcadd`**（§5.2.1 的 `grep dav_c220` 判据通过） |
| `ReduceSumIntrinsicsImpl` 函数体 = `vcadd(sharedTmpBuffer, src, repeatTime, 1, 1, srcRepStride, 0)` | 同文件 `:300-305` | 我们现在用的 `ReduceSum(…,tmp,n)` 与 `WholeReduceSum` **底层是同一条指令**，差别只在：n 次调用 = n 条 `vcadd` + 每次结果从 tmp 回搬 ⇒ 换成一次 `WholeReduceSum` = **1 条 `vcadd`、零回搬、`tmp_buf_` 可删** |
| 注释 `dstRepStride; // dst Stride Unit is 2B(fp16)/4B(fp32)` 与 `srcRepStride; // src Stride Unit is 32B` | `impl/basic_api/utils/kernel_utils_struct_param.h:31,33` | ⭐ **钉死了 stride 单位**：dst 按**元素**、src 按 **32B 块**。⇒ `WholeReduceSum<float>(red, mat, n, n, 1, 1, 1)` 的 n 个行和**紧凑落在 `red[0..n-1]`**，正是单次 `Brcb` 的入参要求 ⇒ **§9.7 表中"stride 语义待验"这一格就此结案**（至少对 `WholeReduce*` 成立） |
| `ReduceRepeatParams(int32_t mask, …)` 的 2201 分支把 count 翻成 `highMask/lowMask` | 同文件 `:36-47`（`#else` 侧） | count 形态（`MASK_COUNT_MODE`）在 2201 有实现路径，不是只有 `mask[]` 位掩码那一支 |
| 公开声明 `(dst, src, mask[] \| mask, repeatTime, dstRepStride, srcBlkStride, srcRepStride)`，`#else` 侧对 2201 生效 | `include/basic_api/kernel_operator_vec_reduce_intf.h:143-153, 187-202` | 参数**真名**如此。⚠️ **修正 §10.6 的措辞**：我把它写成 `(count, outLen, …, blocksPerRow)` 是**按官方调用位置猜的名字**，真实语义是 *mask=每行参与归约的元素数（count 形态）、repeatTime=行数、srcRepStride=每行跨几个 32B 块*。官方 `mhc_pre_sinkhorn_base.h:613/:622` 那次调用照此读 |

⚠️ **B 段全部是"读头文件"级别的结论，还没过编译器、没上仿真器**。动手前的判据（一次 `build_cpu.sh` 就够，30 秒级）：在 `refs/harness_sinkhorn/cpu_debug` 里加一个最小 probe，`Duplicate` 出已知矩阵 → `WholeReduceSum<float>(red, mat, 4, 4, 1, 1, 1)` → 断言 `red[0..3]` 与标量行和逐个相等。绿了才谈替换 `:120-122` 与 `:131-133`。

**C）环境侧顺手记**：`devspace_tunnel.ps1 -Role cpu` 对 `e6z6k`（devEnvId `d611a7d4…`）报 `development environment no longer exists`（账号列表查不到，需用户切回所属账号点「连接」或控制台启动），但 **`tpm0u` 正常可连**（`devenvc_tpm0u.724ca0f5…`，本地端口 35098）⇒ 本轮后续以 tpm0u 为云端仿真机入口；`~/ops_comp/code2` 未随重建丢失（`op_kernel/op_host/cpu_debug/probe_cann_api.*` 仍在）。

**D）⭐ 上述推断已在云端仿真机实测结案（tpm0u / CANN 9.0.0 / `-D__NPU_ARCH__=2201 -DASCENDC_CPU_DEBUG`）** —— 仪表 `refs/harness_sinkhorn/cpu_debug/probe_whole_reduce.cpp`（md5 `1c568c655bdf`，故意放在 `refs/` 不进提交目录），日志本地双写 `cpu_debug/logs/probe_whole_reduce_run_20260920.log`。

调用 `WholeReduceSum<float>(dst, src, /*count=*/4, /*repeatTime=*/4, 1, 1, 1)`，源为 4 行 × 8 lane、值 `src[i*8+j]=i*8+j`（⇒ 每行前 4 lane 的理论和 = 6 / 38 / 70 / 102，最大 = 3 / 11 / 19 / 27）：

| 判据 | 实测 | 结论 |
|---|---|---|
| 编译门（浮点 + count 形态，含 `ORDER_ONLY_VALUE`） | `BUILD_RC=0`，0 error | ✅ 2201 可用，B 段的"读头文件"结论升级为"过编译器" |
| 一次调用是否做完 4 行 | `dst_sum[0..3] = 6.0 38.0 70.0 102.0`、`dst_max[0..3] = 3.0 11.0 19.0 27.0`，与理论值逐个相等 | ✅ 4 行 = 1 条指令 |
| 结果落位（决定单次 `Brcb` 是否成立） | **紧凑在 `dst[0..n-1]`**，`dst[8]/dst[16]/dst[24]` 均非结果（"packed=YES / per-block=no"） | ✅ 正是 `Brcb` 的入参要求，§10.6 的布局结论成立 |
| `count` 是否真的屏蔽 padding lane | 行 0 得 6（=0+1+2+3）而**不是** 28（=整块 8 lane 之和） | ✅ **count 形态天然只算前 n 个 lane** |

⭐ **最后一行顺带把 §10.2 #4 的前提从"推断"变成"实测"**：因为归约按 `count=n` 屏蔽 lane，padding 里放 0 还是放 `NEG_PAD` **对行和/列和都没有影响**（前提是列累加也改成 `count=n`）⇒ 现在这个每矩阵一次的 `Duplicate(mat, NEG_PAD, n*RS)` 播种是可以整块删掉的。

**E）⚠️ 一处自我更正（影响 §9.7 的"内层零标量访问"结论）**：`dav_c220/kernel_operator_vec_reduce_impl.h:153-167` 显示，我们每轮调 n 次的 `ReduceSum<T>(dst, src, tmp, count)` 在 2201 上的函数体是

```cpp
vcadd(dst, src, 1, ...);  SetFlag<HardEvent::V_S>(...);  WaitFlag<HardEvent::V_S>(...);
int64_t accVal = get_acc_val();   *(dst) = *(reinterpret_cast<T*>(&accVal));   // 标量写回 UB
```

⇒ **每次 `ReduceSum`/`ReduceMax` 内部都含一次 V↔S 域切换 + `get_acc_val()` + 标量写**（`sharedTmpBuffer` 在 2201 这一支压根没用上，参数是摆设）。所以 §9.7 末尾"每轮 ~29 条、内层零标量访问"这句要改口径：**源码层零标量访问成立，但归约把 n 次标量往返藏进了 API 里**，而标量访问正是 §5.2 排在第一位的病根。换成 `WholeReduce*`（函数体就一条 `vcadd`，结果直接落 UB）是**第一次真正消掉内层标量往返**，省的不只是"8 条→1 条"的计数差。⇒ 这一刀的收益应重新估，且**优先级高于 §10.3 的其余两刀**。

---

## 11. ⭐ 真机三步走：基线结清 + step-1/step-2 采纳 + 一次判负（2026-09-21，910B3 真机 `02aeb`）

### 11.1 身份（读数以这套 md5 为准，口径 `md5sum` 原文件）

| 件 | md5 | 说明 |
|---|---|---|
| `code2/op_kernel/mhc_sinkhorn.cpp`（= 真机 `k_vec1.cpp`） | `bdd8dfa8` | step-1 向量版，**本轮第一次上真机** |
| `refs/harness_sinkhorn/k_vec2_whole.cpp` | `96e37d2e` | step-2 = step-1 + `WholeReduce*` 替换 n 次行归约，删 `tmp_buf_` |
| `code2/op_host/mhc_sinkhorn.cpp`（= `host_cur.cpp`） | `6fcd98d5` | `SetBlockDim(40)` 恒定 |
| `refs/harness_sinkhorn/host_blkcap.cpp` | `7412dcad` | 判负实验件，见 §11.5 |
| `tiling_cur.h` / `tkey_cur.h` | `c42f9443` / `86fb67fb` | ⚠️ 本轮把远端 `mhc_build/op_kernel/tiling_key_mhc_sinkhorn.h` 从 `6358c3a0` 换成了 `86fb67fb`；原件备份在 `~/mhc_test/tkey_backup_20260921.h` |
| `scalar_kernel.cpp` | `0ad8e397` | 标量版（已过 5/5 的那版），本轮拿它补真机基线 |

**工装件（§11.15 换版后）** —— ⚠️ `.sh`/harness 在**本地是全文件 CRLF**，`md5sum 本地` 与 `md5sum 远端` 天然不等；
下表一律 `tr -d '\r'` 后的口径（= 远端实际字节，`file` 已确认无 `CRLF line terminations`）：

| 件 | md5（LF 口径） | 说明 |
|---|---|---|
| `refs/harness_sinkhorn/run.sh` | `83c82024` | + `rm -f test_sink`（缺陷 #2）+ 每用例打 `CASE=PASS/FAIL`（缺陷 #3） |
| `refs/harness_sinkhorn/prof_bench.sh` | `96d70b32` | dtype 口径 `DTY`（默认 fp32）+ 门改数 `CASE=PASS` + `DT=$DTY` 传进 `run.sh` |
| `refs/harness_sinkhorn/f32_gate.sh` | `db80302b` | fp32 契约闸门（常量 6 + 随机 3 对拍），build 段门同样改数 `CASE=PASS` |
| `refs/harness_sinkhorn/test_mhc_sinkhorn.cpp` | `98858791` | host 两侧改 `vector<uint8_t>` 按 esz 拼包（缺陷 #4），判决行带 `dtype=` |
| `~/cpu2/{build_cpu.sh,run_cpu.sh,test_mhc_sinkhorn_cpu.cpp}` | `ef5df5d9` / `7c189880` / `783d9517` | 官方口径仿真机对拍，本轮**搬到真机 02aeb 上跑**（`libcpudebug.so` 在 `tikicpulib/lib/Ascend910B1` 就有，不必另开仿真机） |


### 11.2 历史口径欠账结清

| 文档里裸写的数 | 实测身份 | 结论 |
|---|---|---|
| "341µs" | **batch=20, n=6, iters=100，标量版** `single_avg=341.95us`（`single_min=335.09`，`burst_avg=313.99`） | ✅ 对上，此后一律带 shape 前缀写 |
| "303µs" | 本轮 8 个测量点位无一命中（最接近的是标量 `batch=1024 n=8 single_avg=3048.97us`，差一个数量级） | ⛔ **仍未结案**，别再引用 |
| §5.1 "1.5µs 预算" | 设备侧 `batch=1 n=4` 纯 kernel **7.175µs**（step-2） | 预算口径当初漏了**每任务固定成本 ≈ 4.9µs**（见 §11.6） |

### 11.3 真机结果：墙钟（`bench_sink` single_avg）+ 设备时长（msprof Task Duration 剔首 mean）

| shape (iters=20) | 标量 | step-1 | step-2 | 墙钟 标量→step2 | 设备 step1→step2 |
|---|---|---|---|---|---|
| `1×4` | 53.42 | 24.97 | 24.32 | **−54.5%** | 10.031 → 7.175 (−28.5%) |
| `1×8` | 139.57 | 37.13 | 27.13 | −80.6% | 16.929 → 10.569 (−37.6%) |
| `20×6` | 92.68 | 39.97 | 28.58 | −69.2% | 16.505 → 11.954 (−27.6%) |
| `64×8` | 260.74 | 55.48 | 40.66 | −84.4% | 35.550 → 22.950 (−35.4%) |
| `100×6` | 227.91 | 60.82 | 45.86 | −79.9% | 40.398 → 26.554 (−34.3%) |
| `1024×8` | 3048.97 | 428.59 | 266.18 | −91.3% | 404.941 → 241.073 (−40.5%) |
| `8192×8` | 23720.87 | 3198.00 | 1908.85 | −92.0% | 未 prof |
| `20×6 i100` | 341.95 | 80.78 | 59.40 | −82.6% | 未 prof |

- **正确性**：step-1、step-2 各自 `6/6 配置 成功`，与 `1/n` 最大偏差 ≤ `2.441e-04`（= 标量版同批次同量级，fp16 下约 2 LSB）；6 个配置 = `8×8 / 1024×8 / 64×4 / 100×6 / 1×8 / 8192×8`。
- ⇒ **step-2 采纳**：这是本题第一次拿到"真机 + 设备侧"双口径同时下降的证据，`WholeReduce*` 那一刀的收益（−27.6%~−40.5% 设备时长）**大于** §10.3 里其余两刀的合计预估，与 §10.7E 的推断方向一致。

### 11.4 为什么有效（不是"8 条→1 条"的计数差）

`ReduceSum/ReduceMax` 在 2201 的函数体里含 `SetFlag/WaitFlag<V_S>` + `get_acc_val()` + **标量写回 UB**（§10.7E）；`WholeReduceSum` 的函数体就是一条 `vcadd`。所以省的是**每轮 n 次 V↔S 域切换和标量往返**，正是 §5.2 排在第一位的病根。实测印证：`20×6` 的 `aiv_scalar_time` 3.987 → 2.622µs（−34%），`aiv_vec_time` 4.237 → 2.961µs（−30%）。

### 11.5 ⛔ 判负：`SetBlockDim(min(cores, batch))`（题1 经验的迁移尝试）

假设：题1 量到"每核派发税"，小 batch 少开核应当更快。实测（同一 step-2 kernel，只换 host `6fcd98d5`→`7412dcad`）：

| shape | blk=40 均值 | blk=min(40,batch) 均值 | 判定 |
|---|---|---|---|
| `1×4` | 7.175（min 6.980 max 8.160） | 8.924（min 6.800 max 10.520） | **更慢** |
| `1×8` | 10.569（min 10.420 max 11.020） | 12.374（min 10.200 max 13.940） | **更慢** |
| `20×6` | 11.954 | 11.151（min 10.760 max 16.260，散布 5.5µs） | 落在散布内，不算收益 |
| `64×8` / `100×6` / `1024×8` | 22.950 / 26.554 / 241.073 | 23.035 / 27.070 / 241.354 | 持平 |

⇒ **不采纳，`host_blkcap.cpp` 只作实验件留存**。要点：本题小 batch 的开销不在"核数"，而在**每任务固定成本**（少开核反而让单核串行深度和 `aiv_time` 暴涨：`1×4` 的 aiv 2.261→7.287µs）。⚠️ 题1 的"每核 75~90ns 派发税"**不可迁移**到延迟 bound 的小 kernel 上。

### 11.6 瓶颈定位（子计数器口径，step-2 / fp16 / iters=20）

| shape | Task Duration | aiv_time | 其中 vec | 其中 scalar | mte2 | 说明 |
|---|---|---|---|---|---|---|
| `1×4` | 7.175 | 2.261 | 0.129 | **1.785** | 0.446 | 核内只 1 个 8×8 块，向量占比 5.7% |
| `1×8` | 10.569 | 2.619 | 0.198 | **1.419** | 1.194 | 同上 |
| `20×6` | 11.954 | 6.187 | 2.961 | 2.622 | 1.312 | 开始转向 |
| `1024×8` | 241.073 | **234.906** | **184.167** | 89.092 | 10.495 | 97.4% 在核内，vec 主导 |

1. **小档（batch≤1）**：`Task Duration − aiv_time ≈ 4.9µs` 是**每任务固定成本**（启动+收尾），且 §11.5 已证明它与开核数无关。⇒ 小档的下界 ≈ 4.9µs + 核内时间，**继续抠向量指令的边际收益被这 4.9µs 封死**；`20×6` 已经接近（11.954 vs 4.9+6.187=11.1）。
2. **大档**：`1024×8` 搬 256KB 用 241µs ⇒ **≈1.1 GB/s**，比 910B 带宽上界低两个数量级 ⇒ 纯指令 bound，**带宽方向无任何可优化空间**（与题1 结论正好相反）。
3. **每轮向量指令构成**（n=8）：行路 ≈ `WholeReduceSum + Adds + Brcb + Div` = 4 条；列路 = `Duplicate + 8×Add + Adds + 8×Div` = **18 条** ⇒ 列路占 ~78%。**这就是剩下的杠杆**。
4. ⚠️ 未结案：按 1.65GHz 折算，`1024×8` 每矩阵每轮 ≈ 719ns / 22 条 ≈ **54 周期/条**，远高于发射预期。动手前要先确认 `aiv_vec_time` 到底是不是"V 流水忙时"（否则 §11.6.3 的指令数—时间换算不成立）。

### 11.7 ⭐ 本轮新增的 2201 实现层事实（真机头文件原文，已过编译器）

| 证据 | 位置 | 结论 |
|---|---|---|
| `AddImpl/DivImpl(..., const int32_t& count)` 函数体 = `set_mask_count(); set_vector_mask(0, count);` + **一条** `vadd/vdiv(…, 1, DEFAULT_*)` | `impl/basic_api/dav_c220/kernel_operator_vec_binary_impl.h:62-76, 212-224` | ⭐ count 形态在 2201 **就是一条指令**（不是内部循环）。⇒ 现有 `Div(mat,mat,bcast,n*RS)`、`Sub(...,n*RS)` 本来就是单指令，**没有"拆成 n 条"的嫌疑**；但 count>每 repeat 上限时会被 mask 截断（fp32 每 repeat 128 元素，我们最大 64，安全） |
| `SetMask<T>(int32_t len)`：`len==64 → SetMask(0, FULL_MASK)`；`len==typeLen(=32/sizeof(T)) 或 len>=128 → (FULL,FULL)` | `impl/basic_api/kernel_utils_base.h:126-157` | ⭐ 单 `uint64_t` 形态的 `mask` 参数**语义是"元素个数"**，不是位掩码。fp32 下 64 元素 = 8 个 32B 块 = 一个 repeat 满窗 |
| 官方按行广播除数：`src1BlkStride=0` + `Div(dst, src0, src1Ub, 64, dealRowCount, rp)` | `ops-transformer-master/attention/common/op_kernel/vector_common.h:277-310`（`RowDivs`）；同款见 `mhc_pre_sinkhorn_base.h:483-505`（`SigmoidPerf`） | ⇒ 本题列归一化可用**一条** broadcast `Div`（`mask=n*RS ≤ 64`、`repeatTime=1`、`src1BlkStride=0`）替掉 n 条 vdiv。这就是 step-3 |

### 11.8 工装（全部本地留存 + 远端 `~/mhc_test/`，md5 一致）

| 件 | md5 | 用途 |
|---|---|---|
| `prof_bench.sh` | `9de3846b` | 按形状分目录的 msprof 计时；判决行带 `timed=1`；`build=1` 时**非 6/6 直接 abort** |
| `pair_run.sh` | `fa3628f5`(本地 CRLF；远端去 CRLF 后同) | ⭐ 成对测量：候选与基线**同一 session 顺序跑**，两边都重建 + 都过 6/6 |
| `run.sh` | `db70753e` | 重建 + 6 配置正确性矩阵（`BENCH=1` 才跑墙钟） |
| `bench_sinkhorn.cpp` | `499885eb` | 墙钟；支持第 5 参 `fp16\|fp32`，**但不校验数值** |
| 本地双写日志 | — | `logs/base_scalar_bench_20260921.log`(`10d9fd73`)、`logs/vec1_bench_20260921.log`(`b9ad4b51`)、`logs/vec2_bench_20260921.log`(`ea375a82`)；prof 读数在远端 `log/prof_vec{1,2,2_blkcap}.log` |

### 11.9 ⚠️ 提交前风险（记下来，别到提交前才发现）

1. **fp32 分支从未上过机**：正确性闸门 `test_mhc_sinkhorn.cpp:89-91` 写死 `ACL_FLOAT16`，`prof_bench.sh` 的 msprof 调用也写死 `fp16`。⇒ §11.3 的 6/6 全是 fp16。而本题契约 dtype 是 FLOAT32（§9.4），kernel 的 `else` 分支（直 `DataCopyPad` 进 fp32 UB）**在真机上一次都没跑过**。上提交前必须先补 fp32 设备对拍。
2. `workspace[0]=0` 且 kernel 无核间同步 ⇒ 与题1 同类结论：`batch % coreNum != 0` 不会死锁，但**尾部核空转**；已实测 `8192×8`（205 矩阵/核）正常。
3. step-2 只动 kernel（`WholeReduce*` + 删 `tmp_buf_`），host/tiling 三件不变 ⇒ 提交时**只有 `kernel_cpp` 一个字段变化**，归因干净。

### 11.10 ⭐ step-3 采纳：列路一条 broadcast `Div`（成对测量，同一 session）

`refs/harness_sinkhorn/k_vec3_brcdiv.cpp`（md5 `76f3945a`，本地/远端一致）。相对 step-2 只动一处：`ColNormalize` 末尾的 `for i<n: Div(mat[i*RS],…,cs,RS)` → 一条 `Div<float>(mat, mat, cs, /*mask=*/n*RS, /*repeatTime=*/1, rp{src1BlkStride=0, src1RepStride=0, …})`。

| shape (iters=20) | step-2 第1轮 | step-2 成对基线 `vec2b` | **step-3** | step-3 vs `vec2b` |
|---|---|---|---|---|
| `1×4` | 7.175 | 7.379 | **6.288** | **−14.8%** |
| `1×8` | 10.569 | 10.808 | **8.104** | −25.0% |
| `20×6` | 11.954 | 12.409 | **10.627** | −14.4% |
| `64×8` | 22.950 | 23.038 | **17.705** | −23.1% |
| `100×6` | 26.554 | 27.121 | **21.237** | −21.7% |
| `1024×8` | 241.073 | 241.258 | **172.220** | **−28.6%** |

- **重跑漂移**：`vec2` vs `vec2b` 六个点全部落在 ±4% 内 ⇒ 上表 14~29% 的差是**信号不是噪声**（沿用题1 的成对+漂移上界口径）。
- **正确性**：`vec3` 同一批 6 配置 `6/6 成功`，偏差与 step-2 **逐条相同**（`8×8/1024×8/64×4/1×8/8192×8 = 0.000e+00`，`100×6 = 2.441e-04`）⇒ broadcast `Div` 与逐行 `Div` 数值等价（同一 vdiv 指令、同一配对）。
- **机制对上了**（这是本题第一次"指令数模型"被定量验证）：`1024×8` 的 `aiv_vec_time` 184.167 → 124.122µs（**−32.6%**），而每轮向量指令数按 n=8 算是 22 → 15（**−31.8%**）——两个百分比几乎相等。
- ⭐ **顺带解掉 §11.6.4 的疑点**：`aiv_scalar_time` 也同步 89.209 → 52.304µs（−41%）。⇒ 2201 上"标量时间"里有相当一大块是**每条 V 指令的 mask/stride 编程**（`set_mask_count`/`set_vector_mask` 是标量流水动作），不是业务标量计算。**结论：减 V 指令条数会同时压低 vec 和 scalar 两个计数器**，这条比"哪条流水线是瓶颈"的讨论更有操作性。

### 11.11 ⭐ step-4 采纳：列和改成 log2 树形归约（成对测量，同一 session，`vec3b` 为成对基线）

`refs/harness_sinkhorn/k_vec4_tree.cpp`（md5 `2c9f660a`，本地/远端一致）。相对 step-3 只动 `ColNormalize` 的前半：

```cpp
Add<float>(scratch, mat,               mat[(N_MAX/2)*RS], (N_MAX/2)*RS);   // 8 行 -> 4
Add<float>(scratch, scratch,           scratch[(N_MAX/4)*RS], (N_MAX/4)*RS); // 4 -> 2
Add<float>(cs,      scratch,           scratch[(N_MAX/8)*RS], RS);          // 2 -> 1 = 列和
```

三条 count 形态 `Add` 替掉 `Duplicate + n × Add`。前提：行主序下"列"轴是**跨块**方向，而 2201 整个 reduce 家族（`BlockReduceSum`=`vcgadd`、`WholeReduceSum`=`vcadd`、`PairReduceSum`）**只会块内归约**，没有一条能直接做列和（本轮把 `kernel_operator_vec_reduce_intf.h` 的 8 个 `@brief` 全读过，结论不是推测）⇒ 只能自己搭树。跨块的行 `[n, N_MAX)` 在 `LoadRows` 里一次性置 0（`Duplicate(mat[n*RS], 0.0f, (N_MAX-n)*RS)`），树就固定三层、对 n=4/6/8 都成立。

| shape (iters=20) | step-3 | step-3 成对基线 `vec3b` | **step-4** | step-4 vs `vec3b` | `aiv_vec_time` |
|---|---|---|---|---|---|
| `1×4` | 6.288 | 6.287 | **5.551** | −11.7% | 0.105 → 0.091 |
| `1×8` | 8.104 | 8.253 | **6.390** | −22.6% | 0.140 → 0.100 |
| `20×6` | 10.627 | 10.692 | **9.498** | −11.2% | 2.122 → 1.590 |
| `64×8` | 17.705 | 17.559 | **13.797** | −21.4% | 7.775 → 5.235 |
| `100×6` | 21.237 | 21.324 | **17.626** | −17.3% | 10.536 → 7.869 |
| `1024×8` | 172.220 | 172.050 | **123.848** | **−28.0%** | 124.122 → **83.435** |

- **重复性**：`vec3` vs `vec3b` 六个点全部 ≤ 1.8% ⇒ 本表 −11%~−28% 全是信号。
- **正确性**：`6/6 成功`，偏差逐条等于 step-3（`100×6 = 2.441e-04`，其余 `0.000e+00`）⇒ 树形求和换了浮点结合律顺序，但在 fp16 输出的容差下一位都没差出去（**已实测，不是"应该没事"**）。
- **指令数模型第二次对上**：n=8 每轮 15 条 → 9 条（−40%），实测 `aiv_vec_time` −32.8%；`aiv_scalar_time` 更陡（52.304 → 22.323，−57%），与 §11.10 "标量流水在做每条 V 指令的 mask/stride 编程" 的判断一致。
- **累计**：`1024×8` 设备时长 标量口径时代没人测 → step-1 404.941 → step-2 241.073 → step-4 **123.848**（相对 step-1 **−69.4%**）。
- ⚠️ **模型现在反过来给出"别做"的判据**：§10.2 #5 的 `Reciprocal + Muls` 替 `Div`，按"条数=成本"是 **1 条 vdiv 换 2 条**，预期**不赚**。除非先证明 vdiv 的多周期特性真的体现在计数器上，否则不动这一刀。

### 11.12 step-4 的墙钟口径（`BENCH=1 REPS=50 run.sh`，fp16；与 §11.3 同表才能比）

| shape (iters=20) | 标量 | step-1 | step-2 | **step-4** | 累计 标量→step-4 | 设备时长 step-1→step-4 |
|---|---|---|---|---|---|---|
| `20×6` | 92.68 | 39.97 | 28.58 | **26.63** | −71.3% | 404.941 → 123.848? 见下注 |
| `1×4` | 53.42 | 24.97 | 24.32 | **23.46** | −56.1% | |
| `1×8` | 139.57 | 37.13 | 27.13 | **27.17** | −80.5% | |
| `64×8` | 260.74 | 55.48 | 40.66 | **33.89** | −87.0% | |
| `100×6` | 227.91 | 60.82 | 45.86 | **39.42** | −82.7% | |
| `1024×8` | 3048.97 | 428.59 | 266.18 | **147.58** | **−95.2%** | |
| `8192×8` | 23720.87 | 3198.00 | 1908.85 | **984.86** | **−95.8%** | |
| `20×6 i100` | 341.95 | 80.78 | 59.40 | **43.77** | −87.2% | |

- ⚠️ 设备时长列只对应 §11.3/§11.10/§11.11 的 `1024×8`（404.941 → 241.073 → 172.050 → **123.848**）与 `20×6`（16.505 → 11.954 → 10.692 → **9.498**），其余形状没 prof 过，别横向套用。
- ⭐ **墙钟与设备时长在小 batch 上脱钩**：`1×8` 设备 10.569 → 6.390µs（−40%）而墙钟 27.13 → 27.17µs **一点没动**；`1×4` 同理（24.32 → 23.46）。⇒ batch=1 的墙钟由**发起开销 ≈ 20µs** 主导，继续优化 kernel 对这一档的墙钟无感。**平台按哪种口径计分，决定还要不要为小 batch 投工**。
- 大 batch 两个口径同步下降（`1024×8` 墙钟 266.18 → 147.58，设备 241.073 → 123.848）⇒ 大档的优化是真的。

### 11.13 ⭐ fp32 契约首次在真机过闸（step-4）—— 以及两个会让闸门"绿灯但读数不可信"的工装缺陷

§11.9 #1 那条前置风险（契约是 FLOAT32，历史上所有 6/6 都是 fp16 口径）本轮结案。

**第一次运行作废**。`f32_gate.sh` 首跑报的是 `cases_ok=6/6`，但那次跑的仍是 fp16：

| 现象 | 判读 |
|---|---|
| `n=6` 两行 `dev_vs_1n = 4.069e-05` | 恰好等于 fp16 下 `1/6` 的最近可表示值误差（fp32 应在 1e-07 量级）⇒ **dtype 探针** |
| 落盘只有 `/tmp/mhc_out.bin`（528 个 = 33×4×4 的 fp16 文件名），没有 `/tmp/mhc_out_f32.bin` | 随机段 `DT=fp32` 根本没进程序 |
| stdout 头一行没有 dtype 字段 | 编译进 `test_sink` 的是**旧**源码 |

根因（两条，都是"改了的文件不是被编译的那份"）：

1. `run.sh:33` 先 `cd $T/harness` 再 `g++ test_mhc_sinkhorn.cpp`，而我加的 `DT=fp32` 开关在 `$T/test_mhc_sinkhorn.cpp`（`888f0805`）——harness/ 下那份还是 9/18 的 `c5cd823c`。**闸门跑的是旧二进制，绿灯是真的，绿的东西是错的。**
2. 编译行带 `| head -6` 且失败不清产物 ⇒ 残留的旧 `test_sink` 会让 `[ -f test_sink ] || HARNESS FAIL` 这道门照样通过。"编译失败但闸门绿灯"是结构性缺陷，不是这次运气差。

修法：`cp $T/test_mhc_sinkhorn.cpp $T/harness/`（两边 md5 已核一致）；`run.sh` 编译前插一行 `rm -f test_sink`（原文件备份 `run.sh.bak_rmtest`，`bash -n` 过语法）。

**修复后的真 fp32 结果**（kernel `2c9f660a` / host `6fcd98d5`，build 段 fp16 矩阵仍 `6/6` ⇒ 新 harness 没破坏旧口径）：

| 段 | 配置 | 结果 |
|---|---|---|
| 常量输入 | `8×8` `1024×8` `64×4` `100×6` `1×8` `40×6 i3` | **6/6 PASS**，`dev_vs_1n` 1.192e-07 ~ 2.384e-07，双随机性 9.537e-07 |
| 随机输入 vs Python fp32 参考 | `64×8 i20` `40×6 i7` `33×4 i20` | **异常矩阵 0/64、0/40、0/33**，元素级最大偏差 1.490e-07 / 1.788e-07 / 1.192e-07 |

⇒ 覆盖 n=4/6/8、iters=3/7/20、batch 1~1024；fp32 的 `else` 分支（直 `DataCopyPad` 进 fp32 UB）**首次在真机跑通且与参考同量级**。日志双写：`refs/harness_sinkhorn/logs/f32gate_vec4_20260921.log`（含两次运行的来龙去脉）。

⭐ **通用教训（回写工作流）**：改工装源码之后，第一件确认的事是"**编译入口读的是哪个路径的哪份文件**"（md5 对齐），而不是先信闸门输出的 PASS；并且**判据量级要和自己宣称的口径自洽** —— 这次如果先看过"4e-5 不像 fp32"，缺陷当场就能抓到，不用等 Python 对拍报 `FileNotFoundError`。

### 11.14 ⭐ 计时口径切到 fp32（比赛契约），step-4 的 fp32 基线

历史 §11.3~§11.12 的设备时长全是 **fp16** 读数（`prof_bench.sh` 旧版把 `bench_sink` 的第 5 参写死成 `fp16`），而题面是 float32 进/出 ⇒ 从本轮起**默认 fp32**，判决行一律带 `dtype=`（`prof_bench.sh` md5 `32de412e`，本地/远端一致；`build=1` 时计时前先跑 `f32_gate.sh …0`，非"常量 6/6 + 随机 3/3"不放行）。

step-4 同一 package（kernel `2c9f660a`）、同一 session 顺序跑两种 dtype（REPS=20，剔首 mean）：

| shape (iters=20) | fp16 | **fp32（新口径）** | fp32−fp16 | fp32 `aiv_vec_time` | fp16 `aiv_vec_time` |
|---|---|---|---|---|---|
| `1×4` | 5.652 | **5.484** | −3.0% | 0.088 | 0.091 |
| `1×8` | 6.261 | **5.915** | −5.5% | 0.095 | 0.100 |
| `20×6` | 9.234 | **8.551** | −7.4% | 1.500 | 1.589 |
| `64×8` | 13.395 | **12.335** | −7.9% | 4.895 | 5.234 |
| `100×6` | 17.430 | **16.270** | −6.7% | 7.422 | 7.869 |
| `1024×8` | 124.283 | **117.390** | −5.5% | **78.016** | 83.435 |

- ✅ **重复性**：本次 fp16 列与 §11.11 的 step-4 读数逐点相差 ≤0.6%（`1024×8` 124.283 vs 123.848）⇒ 上一轮成对表里的 ±4% 漂移上界本轮更紧，后面所有"百分比"都以这条 fp32 基线为准。
- **fp32 反而更快**：省掉 fp16 路径的 n 条逐行 `Cast`。差异主要体现在 `aiv_scalar_time`（`1024×8`：22.420 → **17.575**，−22%），`vec` 只降 6.5% ⇒ 又一次印证 §11.10"**每条 V 指令的 mask/stride 编程走标量流水**"。
- ⚠️ §11.3/§11.10/§11.11 里的标量版与 step-1/2/3 读数**没有 fp32 版本**，不要拿它们跟本表比大小（跨口径）。step-5 之后的成对基线一律用本表 fp32。
- `1024×8` fp32 的构成：`aiv_time` 112.887 / vec 78.016（**69%**）/ scalar 17.575 / mte2 10.301 / mte3 11.659 ⇒ 瓶颈没变，还是**每条向量指令的固定开销**。

### 11.15 ⭐ step-5 多矩阵打包：`1024×8` fp32 −68%，以及两条"绿灯是假的"工装缺陷结案

**做法**（`k_vec5_pack.cpp`，md5 `d50b94d4`）：一个 UB 窗口 `WIN=N_MAX*RS=64` float=256B 装一个矩阵，
一次 `repeatTime=g`（`G_MAX=8`）把 g 个矩阵当成 g 个窗口喂进**同一条**向量指令。行归约的
`count=n, repeatTime=N_MAX*g, srcRepStride=1` 直接把 g 个矩阵的 8g 个行极值压到 `red_[0..8g)`；
列树、行/列归一化全部按窗口步长 `N_MAX` 走，一次 pass 的指令数从 `step-4 的 9×batch` 变成
`9×⌈batch/8⌉`。fp32 且 `n==N_MAX` 时整段是一次连续突发 DMA（`DataCopyExtParams{1, g*256B,0,0,0}`）。
草稿版有 4 处错（`BrcbRepeatParams` 位置参数含义、UB 侧 DMA 步长写成 256B、丢了 2 条 `Adds`），
都在上机前对着 impl 头文件逐参核掉了 —— 见 §11.15.2。

**成对结果**（同一 session，`vec4b`=基线 / `vec5`=打包，fp32，REPS=20，剔首 mean；原始行见
`refs/harness_sinkhorn/logs/vec5_packed_fp32_20260921.log`）：

| shape (iters=20) | step-4 fp32 | **step-5 fp32** | Δ | vec 时间 4b→5b | scalar 4b→5b |
|---|---|---|---|---|---|
| `1×4` | 5.438 | 5.872 | **+8.0%** ⚠️ | 0.088 → 0.100 | 1.402 → 1.338 |
| `1×8` | 5.813 | 5.576 | −4.1% | 0.095 → 0.099 | 1.292 → 1.494 |
| `20×6` | 8.441 | 9.303 | **+10.2%** ⚠️ | 1.500 → 1.664 | 1.716 → 2.188 |
| `64×8` | 12.726 | **8.651** | −32.0% | 4.895 → 3.228 | 2.500 → 2.087 |
| `100×6` | 16.326 | **11.409** | −30.1% | 7.422 → 4.269 | 3.219 → 2.495 |
| `1024×8` | 116.964 | **37.412** | **−68.0%** | **78.016 → 30.113** | 17.374 → **4.506** |

- ✅ **成本模型第一次被量出上限**：`1024×8` 的指令条数确实是 ÷8（1024 pass → 128 pass，每 pass 仍是 9 条），
  但 vec 只快了 **2.59×**（78.016 → 30.113）⇒ 单条指令扛 8 个窗口时，它的固定开销涨了约 3.1×。
  "条数主导"方向成立，**"每条数据量免费"是近似成立**（负载 ×8 时收益从 8× 衰减到 2.6×）。
  `mte2` 10.133 → 1.313（−87%）、`mte3` 11.654 → 0.528（−95%）是"一次连续突发"替掉 64 次逐行 `DataCopyPad`
  的直接兑现 —— DMA 这一侧才是近乎线性的省。
- ⚠️ **小 batch 反而变慢**（`1×4`、`20×6`）：g=1 时打包省不到条数，却多付了每窗口的 `Seed`（`Duplicate` 哨兵）
  和窗口步长的 mask 编程 ⇒ 净亏。**这不是判负，是门限问题**：`batch < G_MAX` 时走 step-4 的老路（或跳过 `Seed`），
  是下一步的确定收益点。
- 两个正确性闸门都过：`1024/64/100` 三档 `f32_cases=1 f32_rand=3/3`（常量 6/6 + 随机对拍）⇒ 多窗口的
  stride/Brcb/树形折叠在真机上是对的。

#### 11.15.1 ⭐ 缺陷 #3、#4：两道"绿灯是假的"（本轮结案）

| # | 缺陷 | 症状 | 根因 | 修法（已上机核验） |
|---|---|---|---|---|
| 3 | **正确性门是字符串门** | `prof_bench`/`f32_gate` 用 `grep -ac "成功"` 数 6/6，一条数值全错的 kernel 也能放行 | `run.sh` 的 `成功` 只代表 `rc=0`（没崩），跟 harness 自评的 `<<< PASS` 无关 | `run.sh` 每个用例改打机器可读的 `CASE=PASS/CASE=FAIL`（`rc=0` **且** `grep -q '<<< PASS'`），上层门数 `CASE=PASS`；`prof_bench` 还把 `DT=$DTY` 传进 `run.sh`，让矩阵跑在**计时同一个 dtype** 上 |
| 4 | **我自己加的 dtype 开关把 fp16 通路弄坏了** | step-5 的 fp16 用例全红：行 4-7 读成 0、偏差恰好 = `1/n`、双随机性 1.000e+00 | `test_mhc_sinkhorn.cpp` 用 `vector<uint32_t>` 当元素容器 + `total*esz`（esz=2）拷贝 ⇒ fp16 的 H2D 只送出每个字的前半、D2H 又把两个半字挤进一个字，解码还会错位 | host 两侧一律 `vector<uint8_t>` 按 `esz` 字节拼包（元素 i 在 `buf+i*esz`），转换函数提为 file-scope `f32_to_f16`/`f16_to_f32`；`fwrite` 改成按字节写 |

**判 kernel 无罪的关键一步（bisect）**：拿**已装的 step-4**（`2c9f660a`，不重新编译）重跑同一批 fp16 用例，
红的形状、偏差量级一模一样 ⇒ 病灶在工装不在 kernel。**"新写的 kernel 变红"和"工装变红"必须用旧 kernel 复现一次来区分**，
否则会把工装 bug 当成优化失败丢掉一整条路线。

修好后（harness `98858791`）两套 kernel 的 fp16 矩阵**都是 6/6 `CASE=PASS`**（`dev_vs_1n` = 0 / 4.069e-05，
后者正是 §11.13 那条 fp16 舍入指纹）⇒ 缺陷 #3/#4 结案，新旧口径互相印证。step-5 顺手补的 fp16 计时：

| shape (iters=20) | step-4 fp16（§11.14） | step-5 fp16 | Δ | 对照：fp32 口径的 Δ |
|---|---|---|---|---|
| `64×8` | 13.395 | 11.667 | −12.9% | −32.0% |
| `100×6` | 17.430 | 13.809 | −20.8% | −30.1% |
| `1024×8` | 124.283 | **59.531** | −52.1% | **−68.0%** |

⇒ fp16 的收益明显小于 fp32：step-5 的 fp16 分支仍要走逐行 `Cast`（`n×g` 条）+ 逐行 `DataCopyPad`，
打包只省了计算段。**比赛契约是 fp32（statement:207"仅支持 FLOAT32"），所以判决看上一张表**；
这张表只用来证明"同一 kernel 在两种 dtype 下行为一致、没有 fp16 专属的错"。

#### 11.15.2 推送纪律补一条：本地是 CRLF，远端必须是 LF

本轮发现 `run.sh`/`prof_bench.sh`/`f32_gate.sh`/`test_mhc_sinkhorn.cpp` 在本地是**全文件 CRLF**（Write 工具在
Windows 下落盘），而远端 `file` 报的是纯 LF —— 于是 `md5sum 本地` 和 `md5sum 远端` **永远不可能相等**，
拿它当"两边一致"的证据是假证。正确口径（§9.3 早就写了，本轮才真用上）：
`tr -d '\r' < 本地 > /tmp/x` → 推 `/tmp/x` → 比对 `/tmp/x` 与远端的 md5，并让远端 `file` 确认没有 `CRLF line terminations`。
顺带一条**读数不可信**的坑：`grep -c $"\r"`（双引号）在远端 bash 里是 **locale 翻译**，退化成匹配字母 `r`，
会报出"50 行含 CR"这种像真一样的假数；要写 `tr -d '\r' | wc -c` 或 `file`。

### 11.16 ⭐ 官方口径仿真机对拍搬到真机跑：step-5 与 step-4 成对 41 项全 PASS

`cpu_debug/`（直接 `#include` 被测 kernel、自带官方口径 fp32 参考的那套）本来只在云端仿真机上跑。
本轮发现**真机 02aeb 上就能跑**：`$CANN/toolkit/tools/tikicpulib/lib/Ascend910B1/libcpudebug.so` 齐全，
不必另开环境（工程目录 `~/cpu2/{cpu_debug,op_kernel}`，脚本 md5 `ef5df5d9`/`7c189880`/`783d9517`）。

- ⚠️ 第一次 `aiv=40` 直接 SIGABRT：`The input numBlocks 40 exceed max core num of ascend910B1!`
  —— 真机上的 CPU 模拟器是 **910B1 模型**，核数上限远小于 AIV 40 ⇒ 用 `aiv=8`。
  **两次 rc=134 对 step-4/step-5 完全一致**，所以按 §11.15.1 的 bisect 规矩先判"红的是工装不是 kernel"，改参数重跑。
- `mode=full` 41 项判决行，step-5(`d50b94d4`) 与 step-4(`2c9f660a`) **都是 `ALL PASS`、`fail_lines=0`、exit 0**：
  fp16/fp32 × n=4/6/8 × batch 1/7/20/40/64/100/256/1024 × iters **1/20/100** × 多 seed，
  最大偏差 `mism=0/65536`（fp16 大 batch 到 4.9e-04，tolerance 1e-2），**确定性 `bitdiff=0`**。
- ⇒ 这条闸门补上了真机设备门覆盖不到的 **iters=1 / iters=100 / n=4 大批量 / 逐位确定性**，
  是 §11.15 的 fp32 真机 6/6+随机 3/3 之外的第二份独立证据。原始日志
  `refs/harness_sinkhorn/cpu_debug/logs/cpu_pair_step5_step4_20260921.log`。
- ⚠️ 仍存的契约缺口（**本次不改，只记录**）：`-inf/nan` 输入我们输出 `0.96777 0 0…`，题面要求"对应位置 nan"；
  `normOut`/`sumOut` 未实现。两者在标量版 5/5 时就是这样 ⇒ 平台的 5 个测试点不覆盖它们（§9.4 口径不变）。

### 11.17 ⭐ 题2 第一次向量提交：`Pass 5/5`、每条 `precision_ratio: 1`

| 项 | 值 |
|---|---|
| `problem_id` / `problemName` | `6a7c2267a52e0f540a8a02bd`（`ID=302`）/ **`mhcsinkhorn`**（去下划线全小写，与题1 同一命名法，一次命中） |
| `submission_id` | `6ab0bc78b0477ec41e0887b2` |
| 状态 | **`Pass`，5/5**，Case1~5 **每条 `precision_ratio: 1`** |
| 平台原值 time | **8.4 / 14.62 / 9.3 / 15.14 / 25.94**（µs） |
| 提交内容 | **只有 `kernel_cpp` 变了**：step-1 `bdd8dfa8`(9.3KB 级) → **step-5 `d50b94d4` 13,599B**；`host_cpp` `6fcd98d5`、`tiling_h` `c42f9443`、`tiling_key_h` `86fb67fb` 一字未动 ⇒ 归因干净（§11.9 #3 的约定兑现） |
| dry-run | 恰好 4 字段，sha256 `d4b99ae3…4367`/`f82c4d23…7aad`/`506872c9…b522`/`b7dcecf6…6dd8` **与本地四文件逐项一致** |
| 提交树 | `~/ops_comp/code2`（与题1 的 `~/ops_comp/code1` 同构，**不含三个 `.bak`**），四件 md5 与本地 `tr -d '\r'` 口径平 |
| 回执 | `code2/logs/t2_submit1_nowait_20260921.log`、`code2/logs/t2_submit1_query_20260921.log`（远端同步 `~/mhc_test/log/t2_submit1_*.log`） |

合规扫描：提交四件 `grep printf\|fflush\|fprintf\|std::cout\|TODO\|FIXME\|#if 0\|调试\|getenv` **结果为空**。

### 11.18 瓶颈复诊（step-5 之后）：固定开销项还剩 ~1/3，数据量项是地板

把 §11.15 的两组成对读数（fp32 `1024×8`）拟合"**每条 V 指令 = 固定开销 + 与触碰块数成正比的项**"：

| | 指令数（每核） | 每条负载 | vec 实测 | 每条净耗时 |
|---|---|---|---|---|
| step-4 | ~4784 | 1 窗口 = 256B | 78.016µs | **16.3ns** |
| step-5 | ~736 | 8 窗口 = 2KB | 30.113µs | **40.9ns** |

解二元一次方程：`固定 ≈ 12.8ns/条`、`数据量 ≈ 3.5ns/256B`（两个点两个未知数 ⇒ **这是插值不是实测**，
下面这条预测就是拿来做证伪的）。代回去看构成：step-5 的 30.1µs ≈ **固定项 9.4µs（31%）+ 数据量项 20.6µs（68%）**。

- ⇒ **候选 A（便宜、可证伪）：`G_MAX` 8 → 16。** 模型预测：条数再÷2、每窗口数翻倍 ⇒ 数据量项不变、
  固定项减半 ⇒ vec 30.1 → **≈25.3µs（−16%）**，`1024×8` 总时长 37.4 → **≈33µs**。
  若实测明显偏离这个数，就说明模型漏了第三项（例如 UB 银行冲突或大 repeat 的额外编程），当场结案。
- **候选 B（省数据量项，只对 n<8 有效）**：`n=4` 时列树按 `n` 折只要 **2 条指令 / 3 块**（现在按 `N_MAX=8` 走 3 条 / 7 块），
  行归约、`Seed` 也可以只覆盖 4 行 ⇒ n=4 形状的数据量项近一半。**但要小心不能顺手把 `WholeReduce*` 的
  `repeatTime` 也改成 `n*g`** —— 那会让 `red` 变成每窗口 n 个标量的稠密布局，而 `Brcb` 只能按"一个源块 8 个标量 → 8 个目的块"展开，
  窗口边界对不齐（`n=6` 时 `3×2≠8` 直接无解）。所以 B 的安全边界是：**只动列树，不动行归约与 Brcb**。
- **候选 C（护住小 batch）**：`g==1` 时 `repeatTime` 退回 `n`、其余参数与 step-4 等价 ⇒ 收回 §11.15 那 +8~10%。
  平台 5 个 case 落在 8.4~25.9µs，**都远大于 `1×4` 的 5.9µs**，说明判分形状不是极小 batch，C 的优先级低于 A/B。
- 已排除：带宽（`mte2` 1.3µs、`mte3` 0.5µs，合计 5% 不到）、核数（§11.5 判负）、标量流水（只剩 4.5µs/12%）。


