# code1 —— 第一题 `mhc_expand`（前向与反向）

> **本文件随题目推进持续更新**：状态 / 已排除 / 下一步证据。
> 通用流程与通用坑见根目录 `算子开发工作流.md`；连接参数见根目录 `连接信息.md`。
> ⛔ 本题目录 `code1/` 与其它两题**严禁跨界**。

> **相关文档**：[AGENT.MD](AGENT.MD)（入口） · [算子开发工作流.md](算子开发工作流.md)（通用流程/坑/纪律） · [连接信息.md](连接信息.md)（连接参数） · [official_problem_statement.md](official_problem_statement.md#第一题b组简单题mhc-expand-算子前向与反向)（**本题官方题面**） · [code2.md](code2.md) · [code3.md](code3.md) · [refs/README.md](refs/README.md)

---
## 0. 工作约束（**必须遵守**）

| # | 约束 |
|---|---|
| 1 | **每完成一轮对话，必须把进展更新到本文件（code1.md）** —— 新结论、新排除的假设、下一步方向都要写进来；写完换新会话（会话越长每轮越贵） |
| 2 | 只改 `code1/` 下的文件，**不动其他目录**（code2 / code3 / 根目录），除非用户明确要求 |
| 3 | 提交源里**绝不允许**调试输出（`printf` / `fflush` / `fprintf` / `std::cout`），也不留 `TODO` / `FIXME` / `#if 0` / `调试` 字样；诊断一律走"写进输出张量 + host 读回" |
| 4 | 每次改动代码前先备份（`cp xxx xxx.bak`），改完 md5 复验 |
| 5 | 一步一步验证；但**一次上机改多处**，不要"改一处跑一次"（每次失败诊断都会变成永久历史） |
| 6 | **提交不必逐次请示**（2026-09-20 22:12 用户下达），但**每次提交完必须先分析总结、再问是否继续**；本条不覆盖破坏性/不可逆操作 |
| 7 | 凭据（`密钥.txt` / `public.pem` / 私钥与端口）**绝不进提交、绝不打印到日志或文档**，只用于 API 调用 |

---
## 1. 状态速览

| 项 | 当前结论（细节见括号） |
|---|---|
| 算子名 | `mhc_expand`（赛题《【B组简单题】mHC-expand 算子（前向与反向）》）；工程目录 `code1/` |
| 目标芯片 | **Ascend 910B3 / arch22（`__NPU_ARCH__=2201`）/ NPU ID=7**，驱动 25.5.0，CANN 9.0.0，HBM 64GB，**AIV 实为 40 核**（§14.1）（§11.8） |
| 源码状态 | **完整实现 + 六层优化，不是骨架**：tiling 整行条件修正 + 反向双缓冲（§9）→ 前向 `PipeBarrier<PIPE_ALL>`（§11.10）→ host 少开核定则（§14.3）→ 前向去 `TQue` 改两块 `TBuf` 轮转（A2，§16）→ **前向 barrier 按块摊薄 B=2/环 2B + ≥12KB 门（R10，§18）** |
| 真机 | ✅ **7 轮 89 条 + 规则版 45 条×2 每轮 `ALL PASS fail=0 err=0`**，前后向**位级精确**（`fwd-fp16-large 0/469762048`）（§11.10.3 / §14.4）；⚠️ 2026-09-21 起发现**旧 `prof/prof2` 的 PASS 从不比数值**，已修（§17.1），修后 `base` 在 c4 上仍 `mismatch=0` ⇒ 采纳版结论不受影响。R10 采纳版在**提交树**再过一轮 45 条 + 8 条 `timed=1 mismatch=0`（§18.4） |
| 仿真机 | ✅ 三轮全绿（quick 29 条 + 5 组全量 + 边界组）；⚠️ **对前向流水竞争无保护力**（串行执行），`bwd-fp16-BND-D32768-m2` 是**仿真假失败** ⇒ **正确性判据只认真机**（§8.3） |
| 比赛平台 | ✅✅✅✅ **四次提交均 `状态: Pass`、8/8 用例、每条 `precision_ratio: 1`**（首交 22:15 / 规则版 23:41 / A1+A2 02:09 / R10 05:25；§13 / §14.7 / §16 / §18.5）。⚠️ 榜单 `score=0` **不是我方慢**：平台自 2026-09-03 起对**所有人**停写分数（§13.3） |
| 性能 | ⭐ **中大档已贴带宽墙**：fp16-large 前向 **1089µs = 970 GB/s（910B 规格的 94.7%）**、反向 **1087µs = 972 GB/s（94.9%）**（§18.6；R10 在前向净取 **−2.2%~−2.5%**，四轮九模态同向）。小档经 §14 定则 **case1 −24% / case5 −11%**，再经 §16 的 A2 **case1 再 −7.9%**。**残余缺口 = 小档每任务固定开销 ~1.7µs，投入产出比极低**（§14.8）。对平台固定基准 `best_time`：中大档 6 条 1–4%、两条反超榜首（§14.7）。⚠️ 平台侧本轮未复现 −2.5%（四条 1ms 用例 −0.31%~+0.57%），判"平台用例形状未知 + 几十 µs 档分辨率 ±7%"⇒ 真机是性能判据、平台只判过不过（§18.5） |
| 已关闭路线 | ⛔ **分块+跨步 DMA**（`splitMode=3`，§14.5）；⛔ **arch22 源冻结广播编码**（§14.9）；⛔ **前向用手工事件对替掉 `PIPE_ALL`**（R5~R9，§17：`MTE2_MTE3` 不产生序、`MTE3_MTE2` 产生、窄 barrier 能编译但不管序）；⛔ **前向继续加批次 B=4/环 8 块**（§18.6：收益按 1/B 收敛到 ≤1.2%，且要放宽 host 的 `ub_size/4` 预算 ⇒ 破坏 A/B 可比性）；⛔ **用提交位试探平台用例形状**（§18.5：计分冻结，回报为 0） |
| 唯一未定档项 | §9 两项优化**各自的加速比**未回退重测（§4.3）；平台 8 条用例的**真实形状**未拿到（§18.5 已判定不再追） |

**时间线（2026-09-20 一天内走完）**：真机首跑 ⇒ 反向全绿 / 前向 14 条全红（§11.9）→ 机理 = 前向纯搬运缺 MTE2→MTE1 序 → V1 事件配对**实测不足**、V2 窄 barrier **运行期 trap**、V3 `PipeBarrier<PIPE_ALL>` 全绿（§11.10）→ msprof 定 barrier 代价 **+5.5%**（§11.11）→ **首次提交 8/8 Pass**（§13）→ 小档差距定位 + blockDim 扫描 + "少开核"定则 → **第二次提交 case1 −24% / case5 −11%**（§14）。
**时间线（2026-09-21 凌晨）**：P1 探针定 arch22 DMA 口径（§14.9）→ A1/A2 采纳 + **第三次提交 case1 再 −7.9%**（§16）→ 修两个判据级缺陷（计时档不比数值、pkg 会跑旧核）→ **R5~R9 事件对路线以负结果结案**（§17）→ **R10 barrier 按块摊薄采纳（前向大档净 −2.5%）+ 第四次提交 8/8**（§18）→ 前向路线以"95% 带宽规格"结案 → 现场清理归档：`code1/` 散落 `.bak_*` 收敛为 1 个回滚点 + 1 个 `npu_debug/archive/`，真机 `probe1`/`sub3`/`/tmp` scratch 归档后回收（§18.7）。

**一句话**：题 1 的**正确性**（真机 89+90+45 条 + 平台 8/8×4）与**性能**（中大档双向都到 910B 规格的 94.6%/94.9%，小档累计压掉 30%）双双收口；前向流水的"最后一口"已经**判死在 API 语义上**（§17），能不吃新序原语的"barrier 摊薄"也已经在 §18 吃干（−2.5%，再往下按 1/B 收敛），**题 1 目前没有已知的、可量化的、不需要新硬件语义的待动项**。

---

## 2. 输入契约与语义（**动手前先看这一节**）

| 项 | 值 | 与官方题面 |
|---|---|---|
| 输入 / 输出 | `x`（REQUIRED）→ `o`（REQUIRED） | 一致 |
| dtype | `ge::DT_FLOAT16`、`ge::DT_BF16`；`InferDataType` 直通（输出 dtype = 输入 dtype） | 一致 |
| format | `ge::FORMAT_ND`，行主序连续，**无 batch 维、无 N 维** | 一致 |
| 属性 | `mhc_mult` OPTIONAL Int 默认 **2**；`backward` OPTIONAL Bool 默认 **false** | ⚠️ 官方把前向/反向写成**两个独立算子**，本实现用 `backward` 属性合一 ⇒ ✅ 已由首次提交裁定"可行"（§13.1） |

### 2.1 语义

| `backward` | 输入 shape | 输出 shape | 语义 |
|---|---|---|---|
| `false`（前向） | `[S, D]`（rank 2） | `[S, m, D]`（rank 3） | `o[s,k,j] = x[s,j]`，沿**倒数第 2 维**复制 m 份 |
| `true`（反向） | `[S, m, D]`（rank 3，且 `in_shape[1] == mhc_mult`） | `[S, D]`（rank 2） | `x_grad[s,j] = Σ_{k=0}^{m-1} o_grad[s,k,j]`，沿 m 维**求和** |

官方原文：`前向 o[i,m,j] = x[i,j] 对所有 m ∈ [0, mhc_mult)`；`反向 x_grad[i,j] = Σ_m o_grad[i,m,j]`。
官方形状约束：`o_grad` 必须与前向**输出**同形，`x_grad` 必须与前向**输入**同形。

- 轴语义：`S` = token 数，`D` = 隐藏维，`m` = `mhc_mult`；**本算子没有 batch 维**，`m` 插在**倒数第 2 维**（不是最后一维、不是首维）。
- **无任何可学习参数** ⇒ 纯数据搬运、memory-bound。
- 关键布局性质：`[S, m, D]` 行主序 ≡ `(S*m, D)` ⇒ token s 的 m 份副本在 GM 上**连续占 `m*D` 个元素**。

### 2.2 ⭐ 反向是**求和**，不是平均

- 题面 2.2 原文给 `Σ`、示例代码 `o_grad.sum(dim=1)` ⇒ **题面是唯一权威**；数学上"广播的伴随就是求和"。
- vLLM 的 `hc_contract` 用 `mean(dim=-2)`，但那是"往返可复原"的独立归约原语，**不是**本算子的反向。
- **两者数值恰好差 `m` 倍 → 方向选错必定零分。**
- ✅ 平台口径已由首交裁定：按 `sum` 实现 → **8/8 `precision_ratio=1`**（§13.1）。

---

## 3. 提交文件与提交前闸门

### 3.1 四字段 ↔ 本地文件

| 提交字段 | 文件 | 位置 |
|---|---|---|
| `kernel_cpp` | `mhc_expand.cpp` | `op_kernel/` |
| `tiling_h` | `mhc_expand_tiling.h` | `op_kernel/` |
| `tiling_key_h` | `tiling_key_mhc_expand.h` | `op_kernel/` |
| `host_cpp` | `mhc_expand.cpp` | `op_host/` |

- 另有三个 `CMakeLists.txt`（`code1/`、`op_host/`、`op_kernel/`），**不在提交面**。
- ⛔ `op_host/` 与 `op_kernel/` 下是**同名但内容不同**的两个 `mhc_expand.cpp`，别传错（`--dry-run` 回传的 path 会自动显示，见 §3.4）。

### 3.2 md5 基线与演变链（**判断"文件是否被动过"只认本节**）

```
c76d30be6bd1117e2848d858168d89e3  op_host/mhc_expand.cpp               10,343B  ← ⭐第二次提交版（§14.3 少开核定则）
53ee60e532ee1730b3bd0f1bc713c983  op_kernel/mhc_expand.cpp              9,318B  ← ⭐两次提交共同的内核（§11.10 barrier 版）
f759a052a865facbb889149ea1aa37e3  op_kernel/mhc_expand_tiling.h         1,201B  ← 自始至终未动
267e012564ba3d18cfabc729cc214e88  op_kernel/tiling_key_mhc_expand.h       530B  ← 自始至终未动
```

演变链（每一跳的完整 diff 都已核实）：
- **host**：`a16c371d…`（首交，tiling 条件修正版）→ **`c76d30be…`**（第二次提交）＝ §14.3 那一段 host 侧整数算术，**不含新 API、不碰核函数**。
- **kernel**：`d9af5611…`（原始）→ `0fb9e6ec…`（§9.2 反向双缓冲）→ **`53ee60e5…`**（§11.10 前向 barrier）＝ 上一版 **+3 行**（1 行 `PipeBarrier` + 2 行注释），反向路径与 host 未动。
- **sha256**（`--dry-run` 回传口径，§3.4 / §14.7）：`tiling_h be5f660b02fef22a…` · `tiling_key_h b79c2cbda239ff8d…` · `kernel_cpp a1744d146c71acfb…778b68` · `host_cpp` 首交 `32cce2e9…`(9,143B) → 规则版 `278b7b84…`(10,343B)。

> ⛔ **`op_kernel/` 三个文件是"提交内核零改动"的证据链**：推到任何调试环境都用 **tar 管道 + 逐文件 md5 复验**（工作流 §6），**不得对它们做格式化 / 行尾转换**。远端只是副本，**本地 `code1/` 永远是唯一权威源**（容器销毁不影响任何文件）。
> ⚠️ **本仓库 `core.autocrlf=true` 且无 `.gitattributes`** ⇒ 内核文件在 git 索引里是 LF，`checkout` / 重新 clone 会把工作区写成 CRLF（实测同一文件 `d9af5611…` → `28f7760d…`，多出 183 个 CR）⇒ **上面这份 md5 会集体失配而代码一字未改**。核对口径三选一：① `tr -d '\r' < f | md5sum`；② `git cat-file -p :<路径>` 直接读索引字节；③ 加 `.gitattributes` 写 `* -text`（属仓库配置，**需用户同意**）。行尾状态一律用 `tr -dc '\r' < f | wc -c` 判定，**别用 `grep -c $'\r'`**（模式退化过）。

### 3.3 提交前合规扫描（**必须为空**）

```bash
grep -n "printf\|fflush\|fprintf\|std::cout\|TODO\|FIXME\|#if 0\|调试\|ABL-\|MHC_SK" <四个提交文件>
```
2026-09-20 23:34（第二次提交前）实测**零命中**；首交前、加 barrier 后各扫过一次，均为空。
⚠️ 归因注入（如 `[ABL-SK]` 的 `MHC_SK_T` / `MHC_SK_BLK` 两个 `getenv` 开关和 `#include <cstdlib>`）**用完必须整段删除**再扫（§14.6）。留在 `npu_debug/` 里的 `MHC_PCASE` / `MHC_REPS` / `MHC_OPAPI_SO` 属调试工装，不在提交面内。

### 3.4 `--dry-run` 是防传错的保险步骤（**不创建提交**）

两次提交前都跑过：比赛平台回传的清单**恰好四个字段、无多余文件**，且 sha256 + 字节数与本地逐字节一致 ⇒ **上传链路不做 CRLF 转换、不会把 `.bak` / `npu_debug/` 带上去**（`op_kernel/*.cpp` 与 `op_host/*.cpp` 各只有 1 个匹配；`.bak_pre_fwd_event`、`.events_v1` 后缀不是 `.cpp`，不被 glob 命中）。

---

## 4. 已确认 / 已排除 / 仍未排除

### 4.1 源码结构事实（**均有源码级证据，别重复走**）

| 项 | 结论 |
|---|---|
| 落地状态 | **已完整实现**，不是骨架；OpDef 同时声明 `DT_FLOAT16` / `DT_BF16` |
| tiling 结构体 | **11 字段** `MhcExpandTilingData`（全 `uint32_t`），不是"只有一个 `length`" |
| `InferShape` / `InferDataType` | 已完整实现：前向设 3 维、反向设 2 维、失败返 `GRAPH_FAILED`；dtype 直通 |
| tiling key / 模板域 | **4 项** = 2(dtype) × 2(backward)；`template <typename DT_X, bool BACKWARD>`（**backward 已进模板域**）+ 4 个显式实例化 |
| 反向实现 | **逐副本 `for k` 各一次 `DataCopyPad`（`blockCount` 固定 1）**，**不是**单次 `blockCount=m` |
| `Init` 签名 | `(GM_ADDR x, GM_ADDR o, GM_ADDR workspace, const MhcExpandTilingData &tiling)`（多一个 `workspace` 形参；host 恒置 `workspace[0]=0`，实际不需要 workspace） |
| 反向小 S 切分 | `SPLIT_ELEMENT`，`total_tasks = S * dTileNum` ⇒ **免跨核归约** |
| 核数 / UB | `GetCoreNumAiv()` + `SetBlockDim(block_dim)`；UB 预算 = `ub_size/4`（`op_host:85`）；`ub_size==0` 时兜底 `ub_size=192KB`（`op_host:23-27`）⇒ 预算 48KB；kernel 用 `TQue` 双缓冲 + 2 个 fp32 buffer |
| 静态审查 | host/kernel 三套任务编码（ROW/STREAM/ELEMENT）一致；反向 fp32 累加（`CAST_NONE` 入 / `CAST_RINT` 出）；GM↔UB 全走 `DataCopyPad`（非 32B 对齐已处理）；切分决策无 uint32 溢出路径。⚠️ **其"未发现阻塞性问题"的结论已被真机推翻**（§11.9.4）⇒ 教训见 §4.4 |
| 561002 根因 | ⭐ **「Do not find tiling func」不是算子缺陷，是加载方式**（`REGISTER_OP_LIB` 的注册表只在框架自己 dlopen 时提升）。判据/解法固化在 §11.9.2，**下次直接照抄，不要再从环境变量猜** |
| 前向 vs 反向的真机分歧 | ⭐ 同一件事的两面：反向在 VECIN 缓冲上有 `Cast`/`Add`（VEC 消费 ⇒ 序成立）且写出走 VECOUT；前向**纯搬运、中间没有 VEC 指令** ⇒ VECIN 的序对 MTE1 完全不生效。**别再逐条查前向的偏移/对齐算术** —— `out` 总字节 939,524,096 < 2^31、偏移全为 `int64_t`、`tile*2` 恒 32B 对齐，均已排除（§11.9.4）。✅ 已按此结论修复并真机复跑全绿（§11.10），反向一行未动 |

### 4.2 设计阶段文档 `code1/DESIGN.md` 的全部结论已作废

七条"设计已定稿但源码未落地"的旧断言（仍为骨架 / 只有一个 `uint32_t length` / `InferShape` 空实现 / 只声明 `C_DT_FLOAT16` / `backward` 不进模板域 / 反向用一次 DMA 描述读 m 份 / `Init` 需换签名）**全部被落地后的源码推翻**；该文档**已删除**，可用的设计意图已并入 §2/§3/§4.1。
⇒ 凡遇到引用 DESIGN.md 的说法（含"文件与比赛平台通过版逐字节一致"这句**从来没有 md5 支撑**的转述），一律以 §3.2 / §4.1 为准。

### 4.3 这些**还没有**被排除

- ❌ **§9 两项优化各自的加速比**：§11.11 只做了"加 barrier vs 不加"，§14 只做了 blockDim 扫描，**host tiling 条件修正、反向双缓冲各自没有回退重测** ⇒ "优化有收益"目前只有算术（§9.1 那张 DMA 计数表）与真机 PASS 支撑，**没有定量**。定档需再 2 轮同机 `msprof` A/B（⚠️ 反向回退会改变 `tile` 决策，用例必须固定同一形状）。
- ⚠️ **真机 UB 用量是否逼近上限**：前向整行 `tile=32768` 时名义占用（`in_que_` 2×64KB + `out_que_` 2×64KB + acc + tmp 2×128KB）远超 256KB，真机却 `D=32768` 反向 PASS ⇒ 说明这些缓冲**并未同时存活**或 `InitBuffer` 有复用。**后续任何调整 tile 策略必须重新核算这一点，不能假定"256KB 够用"**。
- ❌ **题面是否覆盖 fp16 溢出输入**：`desc` 全文 4168 字**零处**提及 `inf` / `65504` / 饱和，自报覆盖清单里也没有溢出场景 ⇒ 推断"不被判分覆盖"，**仍未实测**。我方对拍已按真机语义（IEEE `inf`）定档（§11.10.5），两次提交也未被触发 ⇒ 不再为它花提交位。
- ❌ **`sum` 口径的最后一格保留**：不能 100% 排除"平台 8 条里一条反向都没有"；但题面 §6 自报含"反向验证：梯度归约求和是否正确"，且我方反向真机 24 条位级精确 ⇒ 风险极低。

> ✅ **以下已被裁定/关闭，不要再回头查**（旧版在 §4.3、§12 与 §13 三处重复记过同一批口径，现合并到这里）：
> 真机能否编过跑对（§11.8 / §11.9 / §11.10）· 性能是否定量（§11.11 / §14）· 比赛平台跑真机还是仿真（提交版 8/8 `precision_ratio=1`，而**未加 barrier 的 V0 前向在真机 14/14 全错** ⇒ 不可能是"把流水竞争藏起来的串行仿真"；题面自报最大 `D=7168` 也不覆盖 §8.3 那条 `D=32768` 仿真假失败）· 精度判据形式（字段就叫 `precision_ratio`＝**逐元素通过比例**，我方拿到 `1` ⇒ 任何更严口径下同样满分）· 单算子 vs 双算子（一次提交、4 文件、1 个算子名就把 8 条全判过）· CANN `8.5.0`（平台）vs `9.0.0`（我方）（8.5.0 下编译通过且全绿，`PipeBarrier` / `DataCopyPad` / `TQue` 行为一致）。

### 4.4 ⭐ 教训：`QuePosition` 必须逐条对照"谁写这块 UB / 谁读这块 UB"来推

纯搬运 kernel 里 `TQue<VECIN>` 的序挂的是 **VEC 消费**，对 MTE1 完全不生效 —— 静态通读、甚至"看起来对"的事件配对都不够。
⇒ **这类流水线序不能靠"背事件名"补**，只能"改一次跑一次直到计数为 0"；而且**任何只跑一轮的全绿都不算证据**（V0 的失配计数每轮不同 ⇒ 必须同参重复跑，V3 就是这么确认的，§11.10.1）。

---

## 5. 测试面

### 5.1 官方覆盖 ↔ 我方用例（题面 §6 逐条都有真机 PASS）

| 题面场景 | 我方用例 |
|---|---|
| 小规模 `S=64, D=256, m=2` | `fwd/bwd-{fp16,bf16}-small` ✅ |
| 中规模 `S=1024, D=4096, m=4` | `fwd/bwd-{fp16,bf16}-medium` ✅ |
| 大规模 `S=8192, D=7168, m=8` | `fwd/bwd-{fp16,bf16}-large` ✅（含 `0/469762048` 位级） |
| 扩展倍数 `m=2/4/8` | `bwd-fp16-m2/m4` + `fwd-fp16-ROW-m8` ✅（另覆盖 m=1/3/5/16） |
| 边界 `S=1`、`D=1` | `fwd-fp16-S1D1`（0/2）、`bwd-*-D1` ✅ |
| 非对齐维度 | `D=7167` / `D=100` / `D=70001` / `D=33` ✅ |
| float16 / bfloat16 累加精度 | 反向全部 fp32 域累加，两组 dtype × 正反向全覆盖 ✅ 位级 |

`tools/reference.py`（**权威判据来源**）内置 **12 组 shape × 2 dtype = 24 组**用例（超集，含官方三档规模）：
`(4,8,2) (4096,7168,4) (7,100,2) (1000,7167,3) (13,128,4) (1,1,2) (1,4096,4) (128,256,1) (64,512,16) (32,1,4) (64,16,8) (8192,7168,4)`
- 判据：`torch.allclose(rtol=1e-3, atol=1e-3)`（两侧先转 fp32），前向另需 `torch.equal`（**逐 bit**）。
- CLI：`python3 tools/reference.py --list | --selftest | --export DIR | --check-forward NPY | --check-backward NPY --case N --dtype {fp16,bf16}`。
- ⚠️ 官方的中档 `(1024,4096,4)` 与高档 `(8192,7168,8)` **未被 reference.py 精确覆盖**（最接近 `(8192,7168,4)`）；真机/仿真 harness 两组都跑过 ⇒ 缺口只在这份 Python 参考里。

### 5.2 分组与条数（**条数会随 harness 版本漂移，别把旧数字当基线**）

| 组 | 内容 | 条数（2026-09-20 真机） |
|---|---|---|
| `quick` | 官方三档 + 边界 + 强制 STREAM/ELEMENT 切分 + fp16 饱和 | 25（仿真侧同组 29/30/35，因 harness 版本与分组不同） |
| `medium` / `large` | 中/大档 × 正反向 × 双 dtype | 4 / 4 |
| `mtile` | 多 tile（`D=70000/70001/33000/33001`，含**奇数尾块**） | 12 |
| `bnd` | UB 预算边界对 | 15 |
| `ub48` | `ub_size==0` 兜底 48KB 预算组 | 4 |

⇒ §11.10.3 的 **7 轮 89 条** = quick×2 + medium + mtile + bnd + ub48 + large；§14.4 的 **45 条** = 同一批去掉重复 quick。
⚠️ **只有 `D` 取奇数才能逼出奇数尾块**：`op_host:90` 的 else 分支写死 `t = std::min<uint64_t>(2048, D)`（不是按预算算出的 24576/32768）⇒ `2048×k` 恒为偶。`D=70001/33001` 两个用例正是补这个缺口。

### 5.3 三条真有翻车机理的风险（其余已被实测关掉）

| 风险 | 机理 | 现状 |
|---|---|---|
| 反向 `sum` / `mean` | 差恰好 `m` 倍 ⇒ 选错**必定零分** | ✅ 题面 + 平台双重裁定为 sum（§2.2 / §13.1） |
| 非 32B 对齐 | `D` 非 16 倍数时行首非 32B 对齐，裸 `DataCopy` 会崩（官方也点名"非对齐维度"） | ✅ 全走 `DataCopyPad`；`D=100/7167/1/33` 真机位级 PASS |
| 逐 bit 比对 | BF16 只有 8 位尾数，直加与 fp32 累加差最后 1 ulp | ✅ 反向 fp32 域累加 + 与 `xgrad_ref_fp32` 对拍，24 条 `maxdiff=0` |

（`ccec` 隐式 `uint32_t→float`、`m=1` 退化、小 shape 用不满核、UB 容量 `m=16` 四条已在 §11 全部实测通过，不再单列。）

### 5.4 tile / 预算边界（**纯算术钉死，真机逐条对上**）

整行 vs 多 tile 的分界就是 `D * elem_size ≤ ub_budget = ub_size/4`。910B 真机 UB = 256KB ⇒ 预算 **64KB** ⇒ fp16/bf16 整行上限 **`D ≤ 32768`**；官方三档 D（256/4096/7168）全部走整行，多 tile 分支只在兜底或超大 D 时生效。

| 用例 | ub_budget | 预期 tile / num / tail | 实测 |
|---|---|---|---|
| `BND-D32768-m2`（`D*2 == budget` 取等） | 64KB | 32768 / 1 / 32768（**整行最后一格**） | 真机位级 PASS；**仿真 FAIL = 假失败**（§8.3） |
| `BND-D32769-m2`（超预算 2 字节） | 64KB | 2048 / 17 / **1**（尾块只剩 1 元素） | 真机 PASS |
| ub48 组 `D=24576` / `D=24577` | 48KB（兜底） | 整行取等 / `2048, num=13, tail=1` | 真机 PASS（4 条全绿） |
| `D=70001` / `D=33001` / `D=26001` | 64KB | `2048/35/369`、`17/233`、`13/1425` | 与容器/真机输出的 `tile=` 一致 ⇒ **harness 镜像与 op_host 同源无漂移** |

---

## 6. 工程材料索引与留证

### 6.1 目录树（整理后 `code1/` 只含第一题）

```
code1/
├─ op_host/mhc_expand.cpp                       ← 提交文件
├─ op_kernel/mhc_expand.cpp                     ← 提交文件
├─ op_kernel/mhc_expand_tiling.h                ← 提交文件
├─ op_kernel/tiling_key_mhc_expand.h            ← 提交文件
├─ op_kernel/mhc_expand.cpp.bak_pre_fwd_event   ← V0（修复前原始版，**长期回滚点**，§11.10.7）
├─ npu_debug/mhc_expand.cpp.events_v1           ← V1（`SetFlag/WaitFlag` 实测版，已弃，§11.10.1）
├─ cpu_debug/                                   ← 仿真机对拍（非提交）
│   ├─ test_mhc_expand_cpu.cpp                  ← 同源 harness：#include 内核 + 镜像 TilingFunc
│   ├─ build_cpu.sh / run_cpu.sh                ← CANN 路径探测 + `$(uname -m)-linux` 自适应（两侧仿真机通用）
│   ├─ run_all_groups.sh / probe_args.sh        ← 分组驱动（组名白名单 + 真实用例计数）+ 参数诊断
│   ├─ recovered_r4/                            ← 云端 tpm0u 第二轮 5 组日志捞回归档（§8.2，md5 双侧复验）
│   └─ logs/ quick_matrix_cloud_20260920.log    ← 仿真侧逐用例留档
├─ npu_debug/                                   ← 真机侧（非提交）
│   ├─ test_mhc_expand_npu.cpp                  ← ACL 启动器（dlopen 取 aclnn 入口，⛔ 不得做成链接期依赖 → §11.9.2）
│   ├─ build_npu.sh / run_npu.sh                ← 编译（含源陈旧判定）+ 运行期组 custom OPP 包
│   ├─ preflight_npu.sh                         ← 到手第一条只读命令（§11.8）
│   ├─ prof_npu.sh / prof_sum.js / prof_matrix.sh  ← msprof 采集器 + `op_summary.csv` 摘要器 + blockDim 扫描驱动
│   ├─ archive/                                 ← 唯一的散落 `.bak_*` 归集处（4 份 git 里没有的历史工装 + 2 份远端 scratch 的 tar.xz，§18.7）
│   ├─ prof/<tag>_<ts>/PROF_*/                  ← msprof 原始产物（8 份 csv/轮，md5 双侧核对）
│   └─ logs/                                    ← 真机 + 平台日志（本地 74 份），结论的唯一原始出处
├─ tools/reference.py                           ← 参考实现 + 用例导出 + 对拍自检（**权威判据来源**）
└─ CMakeLists.txt
```

⚠️ `pkg/custom/…`（运行期组出的 OPP 包，`ASCEND_CUSTOM_OPP_PATH` 指这里）**只存在于设备构建树 `~/ops_comp/code1/`**，本地没有副本（每次由 `run_npu.sh` 现组）。

### 6.2 工装 md5（**非提交**，2026-09-20 23:40 本地实测）

```
cpu_debug/test_mhc_expand_cpu.cpp   4668173231e634e3d448962cc50dc965  ← 含 3 个预算边界用例 + 模式分组 + 逐用例计时
cpu_debug/build_cpu.sh              1fe721462b86e1ff00ce95fb3a4aa13f
cpu_debug/run_cpu.sh                dca1abe8130b508764c5e595a142e60a
cpu_debug/run_all_groups.sh         deb5bd526794498c321f64c4a4fe0fce  ← 组名白名单 + 真实用例计数（§8.4 假成功修复版）
cpu_debug/probe_args.sh             b770ff2af9e758fdbfc6f207cf7936f3  ← 打印脚本实际收到的位置参数
npu_debug/test_mhc_expand_npu.cpp   d23a164734cae2389c0eae98e5768960  ← 演变：c7a3e698… → 21:10 改 IEEE inf → 21:25 加 prof 组 → 23:2x 同步少开核定则注释
npu_debug/build_npu.sh              221777360927eefe1084a2a9ceb5a34d  ← 含源陈旧判定（§11.10.4 坑①）
npu_debug/run_npu.sh                417ceccbbd9c4bfc20f13dec5d0f68a2  ← 运行期组 custom OPP 包（解 561002）
npu_debug/preflight_npu.sh          b488fed4f41cb02972f4bf2682e4dd0f
npu_debug/prof_npu.sh               f1ed94b7d2efcf847565136972d353b3
npu_debug/prof_matrix.sh            b2acd6dc8b3c1fadb316908873ff1ea3
npu_debug/prof_sum.js               f60085cc19d4ca9cf5bcf31242921e13  ← ⚠️ 旧记录 a8ffb96c… 是"加 blk 参数之前"的版本
tools/reference.py                  e4230a47ccb6d3a8616287a4e2449bcc
```

### 6.3 留证日志（**结论的唯一原始出处**；⚠️ `.gitignore` 忽略 `*.log` ⇒ 只在工作区、不在 git）

```
§8.2 仿真三轮   cpu_debug/quick_matrix_cloud_20260920.log（33 行，逐例 tile/mode/blk）
                cpu_debug/recovered_r4/  quick 4e605205 / medium 3af4a0c7 / mtile 060c7388 / ub48 40b0a972
                                        / large 8ccfd1d3 / driver 75995258      ← 远端=本地 md5 逐一复验
                cpu_debug/logs/sim_quick_full_barrier_20260920_2109.log  dadd14dc…  ← V3 仿真 quick 29 条 ALL PASS（CR=0）
§11.9/11.10     npu_debug/logs/npu_allgroups_fixed_20260920_2056.log     877c11df…  ← ⭐ 7 轮 89 条 ALL PASS（当前状态出处）
                npu_fix_quick_20260920_2047.log   ac3a83d6…  ← V1 事件配对 fail=9（不足）
                npu_nbar_quick1_20260920_205522.log 93d56431… ← V2 窄 barrier err=9
                npu_nbar_err_20260920_2056.log    ee9153b7…  ← V2 `sync failed` 原文（带缩进，须 grep 不能 `^\[`）
                npu_barrier_quick_20260920_2052.log 30f80131… ← V3 `PIPE_ALL` 首次 ALL PASS
§11.11 msprof   logs/msprof_ab_v0_v3_fp16large.txt 00583bfa…  ← §11.11.2 那张表的原始 stdout
                prof/{v0,v3}_20260920_*/          ← 各 794K / 8 份 csv，md5 双侧一致
                                                       v0 f91707b4 e5d25911 8633d888 04200357
                                                       v3 4ce454fb d44f54af a5933b70 60f7dd20
§13 首交        logs/submit1_result_8of8_20260920.log 32c2594f…  ← ⭐ 8/8 Pass + 逐 case precision_ratio=1
                logs/submit1_analysis_20260920.log    0b7c722b…  ← §13.2/§13.3 判因原始数据（123 条 last_submission 时间轴）
                logs/submit1_nowait_20260920.log      364b6289…  ← 回执；submit1_query1.log 35660d86… ← status=Running（判分 ~4min）
§14 扫描+第二次 logs/blockdim_sweep_20260920.log      b17e9f15…  ← ⭐ §14.2 扫描表 + §14.4 复测 + §14.5 否决证据
                prof/blockdim_20260920/{b2..b40,rule}_c*/  ← 36 份 op_summary csv（3.6MB，远端逐字节回捞，**未入库**）
                logs/npu_all_20260920_{233408,233428}.log  **同一个 md5 a0a91cca…** ← 同参复跑逐字节一致 ⇒ 无新增非确定性
                logs/npu_quick_233030 067b8f55 · medium_233058 f53128ad · large_233101 f9603e92
                       · mtile_233122 e735b0a5 · bnd_233125 d2d1727e · ub48_233128 a04c6aea  ← 规则版全组 ALL PASS
                logs/submit2_result_20260920.log       916c5e7b…  ← ⭐ 第二次提交 8/8 Pass（case1=3.82 / case5=3.96）
                logs/submit2_nowait_20260920.log       b3217373…  ← 回执
```

---

## 7. 目录边界与历史材料

原先混在本目录下的**别题材料已分流**：`_sfa/` → `refs/sfa/`（第三题）；`_harness/` → `refs/harness_sinkhorn/`（第二题测试脚手架）；`_incoming/`、`_submission326701/`、`_payload.tar.gz`、`mhc_sinkhorn_CPU全过_修复版.run` ❌ **已删除**（内容与提交源重复，判据一律用 `md5sum`）。

⇒ 后果要说清：比赛平台下载 zip 的早期快照**本地已无副本**，"与通过版逐字节一致"**无法本地复核**；**当前四个提交文件以 §3.2 的 md5 为唯一事实源**。若用户重新下载 zip 要再核一次，先问、不自动对齐。

> 各题自己的 `codeN.md` 才是该题的当前口径。本文件只讲**第一题**。

---

## 8. 仿真机验证（本地仿真机 + 云端仿真机，2026-09-20）

### 8.1 方法与两个环境

**同源对拍、提交内核零改动**：`cpu_debug/test_mhc_expand_cpu.cpp` 直接 `#include "op_kernel/mhc_expand.cpp"`，tiling 决策**镜像** op_host `TilingFunc`；`ICPU_RUN_KF` 以 blockDim ≤ 20 跑核（CPU 仿真核数上限 < 50）。

| 环境 | 形态 | CANN 位置 |
|---|---|---|
| 本地仿真机 | x86_64 / 6vCPU / 24G（现场读 IP，勿硬编码） | `~/Ascend/cann/cann-9.0.0`（⚠️ 部分文档写 `/usr/local/Ascend/...`，这台**不存在**） |
| 云端仿真机 devenv 容器 | aarch64 / 16 核 / 32G，`npu-smi` 不存在 = 纯 CPU 仿真 | `~/Ascend/cann-9.0.0`；`libpem_davinci.so` 在 `aarch64-linux/simulator/dav_2201/lib/` |

`build_cpu.sh` / `run_cpu.sh` 已固化全部编译坑（`-DASCENDC_CPU_DEBUG -D__NPU_ARCH__=2201 -D_GLIBCXX_USE_CXX11_ABI=0` + `-lpem_davinci -lcpudebug*` 系列）并做 CANN 路径自动探测 + `$(uname -m)-linux` 架构自适应 ⇒ 两侧通用；推送用 tar 管道 + md5 校验。跨环境配方与踩坑已归入 `算子开发工作流.md` §3.6 / §4.3 / §6。

### 8.2 三轮结果

| 轮 | 环境 | 结果 |
|---|---|---|
| 1（优化 A/B 后重跑） | 云端容器 xq82l | quick 35 / medium 4 / mtile 15 / ub48 17 逐条 `maxdiff=0.00000 mismatch=0`；large **前向 2 条位级 PASS（`0/469762048`，22 秒）**、反向跑到中途 SSH 掉线未取回。⚠️ 该容器同日回收，5 份日志丢失 ⇒ 结果以上表为准 |
| 2（18:12，5 组一次跑完） | 云端容器 tpm0u | quick / medium / ub48 `=== ALL PASS ===`；mtile **仅 1 条 FAIL**（`bwd-fp16-BND-D32768-m2` ⇒ 仿真假失败，§8.3）；large 前向位级 PASS、反向未跑完（后由真机收口）。**5 组日志 20:30 全部 tar 管道捞回 + 6 份 md5 双侧复验**（§6.3） |
| 3（V3 barrier 版重编） | 云端 tpm0u | `build rc=0`、`error:` 0 行 → quick 全量 **29 条 ALL PASS、FAIL=0**（11 前向 + 18 反向）⇒ **`PipeBarrier` 在 `ICPU` 下可编可过、不破坏仿真** |

本地仿真机（6vCPU）全量矩阵跑到 bwd-medium 后按用户决策终止（exit 137）：**fwd-medium / fwd-large（8192×7168×8，469M 元素）/ bwd-medium 均 PASS**；bwd-large 改由小规模多块用例覆盖 + 真机验证。
quick 矩阵覆盖维度：官方小规模、非对齐 `D=100/7167`、边界 `S=1/D=1/m=1/m=16`、强制 STREAM/ELEMENT 切分（aiv=4）、**fp16 饱和**、**多块切分 6 用例**（`bwd MT tile=1536×3` 尾块 1024、`MT-ODD` 尾块 1023 奇数、`fwd MT-STREAM/ELEMENT`、bf16 MT×2）⇒ bwd-large 独有的 `dTileNum>1` 路径已小规模锁定。

### 8.3 ⭐ 只有仿真能给的三条口径

1. ⚠️ **仿真对流水线竞争原理上无保护力**：捞回日志末尾数十行 `[TmSim]: Run in serial mode.` 是**直接书面证据** —— 前向在仿真 **`0/469762048` 位级 PASS**、真机 **98.9% FAIL**（§11.9.4），两侧**完全反向**的分歧 ⇒ **"仿真全绿"对本 kernel 的前向不是证据，本题正确性判据只认真机。**
2. ⚠️ **`bwd-fp16-BND-D32768-m2` 是仿真"假失败"**：仿真 `mismatch=63276/65536 maxdiff=1.875`，真机同参（`S=2 D=32768 m=2 blk=2 mode=2 tile=32768`，整行取等点）**`mismatch=0/65536` 位级精确 PASS** ⇒ 该用例**以真机为准**，**不要再查、不要为它改核**；它是"整行 tile 恰好等于 64KB 预算"这个取等点上仿真器/镜像的产物。**反向其余 23 条两侧一致（全绿）。**
3. ✅ **fp16 饱和行为**只有仿真能量到（`x` 全 32768：m=1 和在量程内精确 / m=2 和 65536 → 饱和 `0x7bff=65504`）；真机同用例给 **IEEE `inf`** ⇒ 两侧各按各自硬件语义对拍（§11.10.5）。

### 8.4 本轮挖到 / 推翻的判断（通用条目已收录工作流 §6，这里只留本题结论）

- **仿真耗时此前被高估一个量级**：旧文档按"中/大档小时级"推断，实测 medium 全组 **46s**、large 前向（10.6 亿 DMA 元素）**22s**。根因：日志里的 `kern=` / `cpu_s` 是 `clock()`，**多线程仿真只计主线程** ⇒ 拿它推墙钟必然离谱。⇒ 中大档**不必留给真机**；`large` 真正慢的只有**反向 2 条**（单条十分钟级），而它已被真机收口 ⇒ **不再补跑**。
- **两次误判已作废**：① "两台容器都在 large 组期间被压满 16 核、导致 sshd 无法应答" ⇒ 真相是 tpm0u 的 5 组早在 18:12–18:14 **两分钟内跑完**，那 41 分钟是**我这边隧道不可见**（"我连不上 ≠ 机器被压满"）；② 据此写下的"下一步降并发单独跑 large"随之撤销（只对已被真机收口的反向 large 有意义）。
- **假成功串成链**：驱动用 `GROUPS=(…)` 存组名 —— `GROUPS` 是 bash **内置特殊变量**（当前进程 gid 列表），赋值被立即重置回 `(1000)` ⇒ 循环只迭代出一个不存在的组名 `1000`；更致命的是 harness **不拒绝未知 mode**（五个布尔全 false ⇒ 零用例，末尾照样 `=== ALL PASS ===` + `exit 0`）。✅ 已修（**工具层，非提交代码**）：改名 `MODES` + 组名白名单（不合法记 `SKIP bad-mode`）+ 每组 `grep -c 'maxdiff='` 统计真实执行数（为 0 记 `SUSPECT zero-cases`），诊断脚本 `probe_args.sh` 留仓。**⚠️ 待用户拍板（不自动改）**：harness 侧把"未知 mode"改成非零退出。
- 另两条：**`ssh -n` 把 stdin 接成 `/dev/null`** ⇒ `本地文件 | ssh 'cat > 远端'` 写出**空文件**（远端 md5 = 空串的 `d41d8cd9…`）；**stdout 接 `head`** 会把仿真进程杀掉 ⇒ 曾产出"只有 6 条"的假 quick 全过日志（`sim_quick_barrier_20260920_2058.log`），**不得当证据**，以 21:09 那份为准。

### 8.5 ⭐ 历史坑：反向 fp16 一度 FAIL 是**测试数据坏了，内核无 bug**

- 现象：`bwd-fp16` m=2~5 FAIL（`got=65504(0x7bff) ref=131008(0x7fff)`），且失配数与核数无关（blk=1..20 恒为 1698/16384）。
- 根因：harness 填充表达式 `(uint32_t)(i*37+11)%29 - 14` 是**无符号**运算 —— residue<14 回绕成 ~4.29e9，×0.125 = 5.37e8，**超 fp16 量程**。宿主 `(half)` 转换在 [65536,131072) 产出 0x7fff（非 IEEE），内核 `CAST_RINT` 饱和到 0x7bff=65504；**两种合法约定只在超量程处分歧**。
- 证据链 5 条：① `bad_fill=15819/32768` **恰为 14/29**（坏值在跑核前就存在）；② x 跑核前后位级快照完全一致（内核不写输入）；③ m-sweep（m=1/8/16/32 PASS，m=2/3/4/5 FAIL）失配数逐个被剩余定理算准；④ 常数填充 PASS；⑤ bf16 全 PASS（8 位指数装得下 5e8）。
- 修复：填充改**有符号** `(int64_t)((i*37+11)%29) - 14`（最大 |和| = 56，量程内）+ 加显式饱和用例（`fill_mode==2`）锁定内核饱和行为。
- ⇒ 对真机的启示：官方用例只要输入在 fp16 量程内（评测数据通常如此），反向求和不会触饱和分歧；即使触发，内核按 IEEE 饱和也是标准行为。

---

## 9. 两项优化（2026-09-20，均已进提交源）

### 9.1 优化 A：tiling 整行条件修正（host）

**问题**：原条件 `m * D * elem_size <= ub_budget` 过保守 —— 核内同时只持有 **1 份 D**（前向读 1 次写 m 次、反向逐副本读），不需要 m 份同时在 UB。
**修正**：`D * elem_size <= ub_budget`；`max_t = ub_budget / elem_size`（不再除以 m）。

| 大档 `S=8192, D=7168, m=8, fp16` | 修正前 | 修正后 |
|---|---|---|
| dTileLen / dTileNum | 2048 / 4 | 7168 / **1** |
| 前向 DMA 每行 | 4 read + 32 write = 36 | 1 read + 8 write = **9** |
| 反向 DMA 每行 | 16 read + 4 write = 20 | 8 read + 1 write = **9** |
| 内层 D 循环 | 4 次 | **0**（消除） |

中档 `(1024, 4096, 4)` 同样受益：`dTileNum 4→1`。附带收益：`D=4096, m=16` 由退化多 tile 改走整行 `tile=4096` 且 PASS（`mtile` 组）。

### 9.2 优化 B：反向双缓冲流水线（kernel）

**问题**：原反向逐副本串行 —— DMA 读 k → VEC Cast+Add → DMA 读 k+1 → VEC Cast+Add → …，DMA 与 VEC 不重叠。
**修正**：预取 k+1 的 DMA 与 k 的 VEC 计算并行，双缓冲交替使用 `in_que_` 的两个 slot：循环前预取 k=0；每轮 `DeQue` 当前 → `AllocTensor`+`DataCopyPad` 预取下一个 → `Cast`+`Add` → `FreeTensor`；最后一轮由 `k < m-1` 守卫不启动 DMA。
**效果**：每轮耗时 ≈ `max(DMA, Cast+Add)` 而非 `DMA + Cast + Add`。

### 9.3 ⚠️ 收益**尚未定档**

两项优化各自的加速比**没有回退重测**（§11.11 只做了"加 barrier vs 不加"）⇒ 目前只有 §9.1 的算术与真机 PASS 支撑，**没有定量**。
**唯一相关的已定量结论**：§11.11.3 显示当前前向已贴住"这个核能达到的上限"（V3 946GB/s vs V0 998GB/s，差额正好是 barrier 的钱；反向 971GB/s，与前向仅差 2.7%）⇒ 带宽侧已无空间。要定档见 §4.3 第一条。

---

## 10. ⭐ 开源参考池：`ops-transformer-master`（2026-09-20）

> 📌 **定位（用户 2026-09-20 定调）**：这批开源材料**有很大的参考价值**，但**最好不要照抄** —— 可以抄的是**部分细节**（切分策略、搬运手法、阈值取法这类"怎么做"），实现仍要自己出。下面每条都标了"可借鉴什么"，**不作为提交源**。
> 📦 本节所引路径/符号依赖外部库，**它不入库**（`.gitignore` 已忽略 `ops-transformer-master/`）。换机器或新 clone 后自行拉取 `https://gitcode.com/cann/ops-transformer`，本地这份版本 = **9.2.0**（`version.cmake:11`）。⚠️ **行号会随版本漂移** ⇒ 按符号名 grep（`UseReadOnce`、`USE_PERMANENT_X`），别把"找不到"当成"不存在"。

### 10.0 合规前提（主办方要求"不能引用闭源软件"）—— ✅ 满足

`ops-transformer` 是华为 CANN **官方开源** transformer 算子库，许可证 **CANN Open Software License Agreement Version 2.0**：

| 条款要点 | 对本比赛的影响 |
|---|---|
| 授权 worldwide、royalty-free，可 download / use / **modify** / integrate / distribute；范围限"developing software **solely for use in systems with Huawei AI Processors**" | ✅ 昇腾真机场景**正好落在授权范围内** |
| 不得用于开发运行在**非华为处理器**上的软件 | ✅ 不受影响 |
| **不得移除/篡改版权声明**；分发须附协议副本、保留 notices | ⚠️ 若借用了文件骨架，**必须保留原 Huawei 版权头**，建议加一行 `Adapted from CANN ops-transformer (CANN OSL v2.0)` |
| 对华为提专利诉讼即终止授权；无单独专利条款 | 无可操作影响 |

三方依赖清单 `Third_Party_Open_Source_Software_List.yaml` 只有 abseil / googletest / eigen / makeself / json / protobuf / libboundscheck，且全是 build/test 支撑、**不在算子源码路径内** → **无 GPL 污染、无禁止竞品条款**；`OAT.xml` 声明全仓 license=CANN-2.0。
> ⚠️ 注意：这是**开源**（源码可见）而**非 OSI 认证**许可证。"不引用闭源软件"这条**满足**；至于"引用开源代码"是否需要在提交里额外声明，题面无明文 ⇒ **保守做法是保留版权头**。

### 10.1 本题**没有**同名实现

全仓库 grep `mhc_expand` / `MhcExpand` / `hc_expand` 命中 **0**；`Tile` / `Repeat` / `Broadcast` / `kv_expand` 亦无同名算子 ⇒ 第一题只能找**语义邻近**的，不存在"直接对标件"。
（对照：第二题 `mhc/mhc_sinkhorn`、第三题 `attention/sparse_flash_attention` 都有同名实现，见 `code2.md §10` / `code3.md §10`。）

### 10.2 邻近候选与"能借的那一个细节"

本题语义已确认是**纯带宽题**：前向 `x[S,D] → o[S,m,D]` 复制广播（无乘加），反向沿中间轴 m 元求和。据此筛出的候选（按有用程度排序，**只留符号名不留行号**）：

| 路径 | 它算什么 / 重合度 | 可借鉴 |
|---|---|---|
| `experimental/mhc/mhc_post/kernel/mhc_post_kernel.cpp` | `out[b*N+n]=x[b]*h[n]`，1→N 广播缩放；`h≡1` 即本题前向。**高** | 双策略切核（per-stream vs read-once）、`UseReadOnce` 的 **4MB 阈值**、bf16 经 fp32 Cast 的路径、host 自适应 blockDim |
| `experimental/mhc/mhc_pre/kernel/mhc_pre_kernel.cpp` | `out[b]=Σ_s h[s]·x[b*N+s]`，`h≡1` 即本题反向。**高** | 用 `s==0` 的 `Muls` 代替 `Duplicate` 清零、尾块 `DataCopyPad`。⚠️ 它**无双缓冲流水**，本题反向已优于它（§9.2） |
| `mhc/mhc_post/op_kernel/arch22/mhc_post_arch22.h`（+ 对应 `op_tiling`） | 官方正式算子，含 1→n 广播。**中高** | `USE_PERMANENT_X` 让 x 常驻 UB、内层只写不读；tiling 折半 dOuter + fp32 常驻判定；`SetL2CacheHint(CACHE_MODE_DISABLE)` |
| `mhc/block_attention_residuals(_grad)/op_kernel/arch22/…_hslice.h` | 残差注意力正/反向。**低-中** | **Kahan 补偿求和**，直接对应题面 §6 的 bf16/fp16 累加精度场景 |
| `mhc/mhc_post_backward`、`moe/moe_token_unpermute_grad`、`mc2/moe_distribute_combine_v2` | 沿 n fp32 累加 / scatter-add。**低** | `[n,tileC]` 缓冲 + `Duplicate` 清零 + 批量 Cast（对应本题反向累加段）；仅切核参考 |
| `attention/*/op_kernel/arch22/…` 的 `blockCount>1 + 一个 stride=0` DMA | arch22 纯搬运手法。**中** | ⛔ 曾设想"一条 DMA 写 m 份副本" ⇒ **已被 §14.5 真机否决**：arch22 上 `DataCopyParams` 的 gap/stride 字段语义**不可依赖文档推断** |

⇒ **真正需要自出的只有两点**：① 一条 DMA 出 m 份副本的广播写法（**已试并已否决**，§14.5）；② 反向的精度补偿求和（Kahan，或直接沿用现有 fp32 累加，§9.2）。
⛔ 以上都属于"抄细节不抄实现"，且**均需用户认可后才动 `code1/` 的代码**（§0 约束 4/5）。

### 10.3 架构可编性核查：能编

`mhc/*/op_kernel` 69 个文件全部含 `__aicore__`；仓库根 `CMakeLists.txt` 明确 **`ascend910b → arch22`**，对应宏 `__CCE_AICORE__ == 220`（全仓 133 处）。`arch35` 是 950 的新式 regbase 写法（276 处），**不可直搬 910B**。
本题 `code1/op_kernel/mhc_expand.cpp` 与官方 `mhc/mhc_post/op_kernel/mhc_post.cpp` **同风格**（`KERNEL_TASK_TYPE_DEFAULT` + `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT`）⇒ 910B3 可编，风格无需调整（✅ 已由 §11.8 真机编译门证实）。
> 📎 顺带修正一条易混事实：官方 `mhc/mhc_sinkhorn` 的 `docs/aclnnMhcSinkhorn.md` 标注 **A2/A3 不支持**、仅 float32、`n∈{4,6,8}` —— 这与本题**第二题**的题面口径**不同源**，不要拿官方 sinkhorn 的约束去改自己的契约（详见 `code2.md §10`）。

---

## 11. ⭐ 真机（NPU）实测（`02aeb` / Ascend 910B3 / NPU ID=7，2026-09-20 19:38–21:30）

> 📌 本节原有 §11.1~§11.7 是"资源到位前的准备清单"（隧道前置、芯片闸、preflight 设计、构建两条路、上机矩阵规划、msprof 规划、时间盒），**已由下面的实测结果取代**；其中的通用做法归并在 `算子开发工作流.md` §4（连接/防假成功）、§6（通用坑）、§7（纪律）。
> ⚠️ **编号保持不连续是有意为之**：§11.8~§11.11 以及 §11.9.x / §11.10.x 是本文与源码注释的交叉引用锚点，**不要顺手重排**。

### 11.8 preflight + 编译门实测（19:38–19:53）

| 判据 | 实测值 |
|---|---|
| 芯片 | **Ascend 910B3**，**NPU ID=7**（⚠️ 旧记录 ID=2 作废），`npu-smi 25.5.0`，HBM 64GB（已用 3203MB），`/dev/davinci7` |
| 机器 | aarch64 / 16 vCPU / **122GB** 内存 / `/home` 余 179G |
| CANN | `~/Ascend/cann-9.0.0`（另有 `~/Ascend/ascend-toolkit` 并存），`ccec` / `atc` / `msopst` / `msprof` 齐备，clang 15.0.5，cmake 3.20.5 |
| 芯片口径 | ⚠️ 控制台里曾有一台标 **910C（Atlas A3）**（`DevEnvC_3GFCN`）⇒ **到手第一条命令先跑 preflight 拿芯片名，再谈编译**（工程两处写死 `ascend910b`：`CMakeLists.txt` + `op_host` 的 `.AddConfig`）。实测 `02aeb` = 910B3 ⇒ **口径正确，提交源四文件零改动**（推送后 md5 逐字节 = §3.2 基线） |
| 编译门 | `source set_env.sh` → `cmake -S . -B build_out -DCMAKE_PREFIX_PATH=$ASCEND_HOME_PATH && cmake --build build_out -j8` ⇒ **`rc=0` / 20 秒 / 产物时间戳全新**（非假成功）：`build_out/tmp/vendors/custom/…/ascend910b/mhc_expand/` 下 **4 个 `.o`**（`backward×dtype` 四分支）+ `binary_info_config.json` + `libcust_opapi.so`(1.6MB) |
| aclnn 接口 | `build_out/autogen/aclnn_mhc_expand.h`，标准两段式：`aclnnMhcExpandGetWorkspaceSize(const aclTensor *x, int64_t mhcMult, bool backward, const aclTensor *out, uint64_t *wsSize, aclOpExecutor **exe)` / `aclnnMhcExpand(void *ws, uint64_t wsSize, aclOpExecutor *exe, aclrtStream)` |

**两个现场坑**（已收录工作流 §6）：① `npu-smi` 裸跑报 `libc_sec.so: cannot open shared object file` —— 那两个库在**驱动目录**（`/usr/local/Ascend/driver/lib64/{common,driver}`），不在 CANN 的 `set_env.sh` 里；② **`find_package(ASC REQUIRED)` 不需要 `ASCConfig*.cmake`**（全 CANN grep 不到这个文件名），`npu_op_package` 宏实际在 `tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake`，**configure 只要带上 `-DCMAKE_PREFIX_PATH=$ASCEND_HOME_PATH` 就能解析**。

### 11.9 首跑 + 全量精度矩阵（19:55–20:25）：**反向 24 条全绿、前向 14 条全红**

**结论先说**：本题**第一次在真机上跑起算子**（此前从未上过机）。跑完 5 组 38 条后，真机给出一个仿真侧**完全看不见**的事实 —— **反向 24 条全 PASS（逐位精确），前向 14 条全 FAIL，无一例外。**
> ✅ **后续**：本节的"前向全红"已于同晚 21:10 修复复跑全绿（§11.10）。本节保留为**问题定位的原始证据**（尤其"仿真原理上看不见这类竞争"这条口径），**不要引用它的 FAIL 数字作为当前状态**。

#### 11.9.1 启动器（`code1/npu_debug/` 三件，全部**非提交**；md5 见 §6.2）

`test_mhc_expand_npu.cpp`（ACL/aclnn 启动器：用例矩阵 + 参考实现与 `cpu_debug` **逐条同源**、输出行格式一致便于跨侧对拍；额外字段 `ws=`、`nan_unwritten=` —— 用 `0x7fff`（fp16 NaN）预置输出缓冲，核没写到的区域一眼暴露）+ `build_npu.sh`（探测 `set_env.sh` → 按 `uname -m` 拼 include/lib → 校验 op 包**新时间戳 + `nm -D` 见 `aclnnMhcExpand`** 防假成功 → 链接 `-lnnopbase -lascendcl -ldl`）+ `run_npu.sh`（运行期组 custom OPP 包 + 设 `ASCEND_CUSTOM_OPP_PATH`）。
用法：`bash npu_debug/run_npu.sh {quick|medium|large|mtile|bnd|ub48|prof}`（日志自动 tee）。⚠️ **构建树是 `~/ops_comp/code1`，不是 `~/code1`**（§11.10.4 坑②）。

#### 11.9.2 561002「Do not find tiling func」—— 首跑全红，根因是**加载方式**不是算子

现象：25/25 用例 `st=561002`。逐项排除过 `ASCEND_CUSTOM_OPP_PATH` 指 `tmp/vendors/custom`、`build_out`、`build_out/op_host`、直接指 `.so` —— 全无效。
**根因（证据链）**：CANN 9.0 的 `REGISTER_OP_LIB(custom).RegOpLibInit(...)` + `ops::OpAICoreDef::SetTiling` 在 DSO 静态初始化期写 **`LocalRegistry`**，**只有框架自己通过 `ASCEND_CUSTOM_OPP_PATH` dlopen 该 so 时才把这张表提交进全局注册表**；启动器若把 `libcust_opapi.so` 做成**链接期依赖**，它虽被加载，注册器却只落在本 DSO 的 LocalRegistry ⇒ 框架查不到 tiling func。
**解法（已固化进脚本）**：⛔ **绝不做成链接期依赖** —— 启动器用 `dlopen`+`dlsym` 取入口（路径走 `MHC_OPAPI_SO`），`run_npu.sh` 在**运行前**按 CANN 自带 `tikcpp/ascendc_kernel_cmake/fwk_modules/scripts/install.sh` 的布局组包，让框架自己 dlopen：

```
npu_debug/pkg/custom/op_api/lib/libcust_opapi.so      ← ASCEND_CUSTOM_OPP_PATH=$PKG/custom
npu_debug/pkg/custom/op_impl/ai_core/tbe/{kernel,config}
```

改完 `err=0`，接口层彻底打通。**kernel 侧无需 OPP 路径**：`ACLNN_WITH_BINARY` 已把 4 个 `.o` 与 `binary_info_config.json` 以 `_binary_*_start/end` 符号内嵌进 `libcust_opapi.so`。

#### 11.9.3 精度矩阵（`blk=50`）

**反向 24 条全绿，`maxdiff` 逐条 = 0.00000（位级精确，不是容差内通过）**：官方三档（含 `large` 各 **0/58720256**）✅ 6/6 · m 扫 1,2,3,4,5,16 ✅ · 非对齐 D=7167/100/1/33 ✅ · 多 tile（D=70000/70001/33000/33001 × m=2/8，双 dtype）✅ 8/8 · **UB 预算边界对**（32768 整行 vs 32769 退化、24576 整行 vs 24577、26001）✅ 5/5。
⇒ §8.2 仿真侧三轮都没收口的 `bwd-*-large`（8192×7168×8）**真机毫秒级通过**；§5.4 靠纯算术推出的预算边界真机全对上。

**前向 14 条全红，且非确定**：`fwd-fp16-S1D1`（S=1 D=1 m=2，**最小可能形状**）1/2 —— 连"1 个元素复制 2 份"都错；`fwd-fp16-small` 4944/32768（15%，**历轮 8178 / 2046 / 2809 / 4944 每轮不同**）；`medium` ~88%；`large` **464593143 / 469762048 = 98.9%**（S 越大越接近全错）；其余（m=1/m=16/ROW-m8/D7167/D100/BND/UB48）14%~50% 全 FAIL。

#### 11.9.4 ⭐ 前向失败机理：**MTE1 读 UB 与 MTE2 写 UB 之间没有任何序保证**

`op_kernel/mhc_expand.cpp:104-122`（前向**纯搬运**，中间没有 VEC 运算）：

```cpp
auto in_buf = in_que_.AllocTensor<DT_X>();     // in_que_ = TQue<VECIN,2>  (:168)
DataCopyPad(in_buf, x_gm_[src_off], cp, pp);   // :110  MTE2  GM -> UB
in_que_.EnQue(in_buf); auto x_local = in_que_.DeQue<DT_X>();
for (k...) DataCopyPad(o_gm_[dst_off], x_local, cp);   // :119  MTE1  UB -> GM
in_que_.FreeTensor(x_local);                   // :121  只保证 VEC 已消费，不保证 MTE1 已读完
```

`VECIN` 队列的序**挂到 VEC 消费**上；前向没有 VEC 指令 ⇒ **MTE1 可能在 MTE2 落地前就开读**，读到 UB 里的陈旧内容。三条独立证据互锁：
1. **错值是"UB 脏数据"而非"搬错行"**：`large` 首帧 `got=0.0078 / 0.0000 / 0.0039`，即位图案 `0x0001 / 0x0000 / 0x0002`（fp16 次正规）。输入按 §11.9.3 的填充**全是 0.125 的整数倍**，输出里绝无可能出现这些值 ⇒ **不是偏移算错，是源头就没数据**。
2. **写没落地的区域可区分**：输出预置 `0x7fff`，全部用例 `nan_unwritten=0` ⇒ 空间**确实被写过**，只是写的是脏数据。
3. **规模相关性**：ROW 模式每任务把同一块 UB 连发 m 次 MTE1，`large`（S=8192, m=8, tile=7168）窗口最宽 ⇒ 98.9%，小形状只 15%；**同形重复跑计数每次不同** ⇒ 竞争而非算术错误。

**为什么反向不受影响**（`:126-165`）：反向在 VECIN 缓冲上插了 `Cast`/`Add`（`:154-155`），**VEC 消费即构成 MTE2→VEC 的序**，写出又走 `TQue<VECOUT>`（`:159-164`，保护 VEC→MTE1）⇒ 全链有序，实测 24/24 位级精确。**这也解释了仿真为什么全绿**：`ICPU_RUN_KF` 不实现流水线事件、串行执行 ⇒ 该竞争在 CPU 仿真上**原理上不可能被发现**（§8.3①）。
**排除项**：`out` 总字节 939,524,096 < 2^31 ⇒ 32 位字节偏移不溢出；`dst_off`/`src_off` 均为 `int64_t`；`tile=7168` → `cur_h*2=14336` 为 32B 对齐，非 pad 分支。**与寻址无关。**

#### 11.9.5 顺带坐实的两个口径

- **真机 UB = 256KB 成立**：host 预算 `ub_size/4` = 64KB ⇒ D=32768（fp16 整行 64KB）走整行、D=32769 退化为 `tile=2048`，与 §9.1 推导**逐位吻合** ⇒ §9 全部 tiling 推导无需重算。
- **fp16 溢出语义分歧（确定性，非竞争）**：`bwd-fp16-sat-m2`（输入 32768，m=2 ⇒ 65536 溢出）真机 `Cast(CAST_RINT)` 给 **+inf**（512/512 全错），CPU 仿真与本地参考给 **65504（饱和）**；同组 `sat-m1`（不溢出）PASS ⇒ 只有溢出分支分歧。✅ 已定档：用户拍板"改仿真参考为 IEEE inf"，改完该用例 FAIL → **0/512 PASS**（§11.10.5）。

#### 11.9.6 拍板记录（三条，2026-09-20 21:10 前全部关闭）

| # | 决策点 | 结果 |
|---|---|---|
| 1 | 前向竞争怎么修：(a) `SetFlag/WaitFlag` 事件配对 / (b) 前向改走 `TQue<VECOUT>` / (c) `PipeBarrier<PIPE_ALL>` | 用户选 **(a)** ⇒ **实测不足**；(b) 未试（V1 已覆盖同族思路）；窄 barrier ⇒ **运行期 trap**；最终落到 **(c)** 全绿。⚠️ **与批准方案有偏离**，声明见 §11.10.7 |
| 2 | fp16 溢出：改**核**（加饱和）还是改**参考**（接受 inf） | 用户选 **改参考为 IEEE inf** ⇒ 已改并复跑达标（§11.10.5） |
| 3 | `cpu_debug` harness 未知 mode 静默返回 0，是否改非零退出 | **未做**（低优先，真机侧已能兜底）⇒ 留在待办，不阻塞提交 |

### 11.10 ⭐ 前向竞争的修复实测（20:40–21:10，同机 `02aeb` + 云端仿真机 `tpm0u`）

**结论先说**：`ForwardOneBlock` 在 `DeQue` 之后加 **1 行 `PipeBarrier<PIPE_ALL>()`**（另 2 行注释，**完整 diff = +3 行**，反向与 host 一行未动）⇒ 真机 **7 轮 89 条每轮 `ALL PASS fail=0 err=0`**、云端仿真机 quick **29 条 ALL PASS**。

#### 11.10.1 四个版本的对照实验（同一台真机、同一份 quick 矩阵 25 条）

| 版本 | 前向改动 | quick 汇总 | `fwd-fp16-small` mismatch | 判定 |
|---|---|---|---|---|
| **V0** 原始 | — | `HAS FAIL fail=9 err=0` | 4944/32768（历轮 8178 / 2046 / 2809 / 4944 **每轮不同**） | 竞争 |
| **V1** 事件配对（**用户批准**） | `SetFlag/WaitFlag` 双向配对（`MTE2_MTE1` 管"落地才能读"，`MTE1_MTE2` 管"读完才能覆写"）+ `fwd_warmed_` 首块标志 | `HAS FAIL fail=9 err=0` | **2540**（`fwd-bf16-small` 12883） | ❌ **不足**：计数降了一半，但**不为 0、仍非确定** |
| **V2** 窄 barrier | `PipeBarrier<PIPE_MTE2>()` + `PipeBarrier<PIPE_MTE1>()` | `HAS FAIL fail=0 err=9` | 32768/**32768** 且 `kern=0.000s`、`got=nan`、`nan_unwritten=32768`（核根本没写 ⇒ 整块留在预置 NaN） | ❌ **运行期 trap**：`    [fwd-fp16-small] sync failed` |
| **V3** `PIPE_ALL`（**采纳**） | 1 行 `PipeBarrier<PIPE_ALL>()` | **`ALL PASS fail=0 err=0`** | **0 / maxdiff=0.00000** | ✅ 全绿 |

**V1 为什么不足 —— 两条已坐实、一条仍是假设**：
- 坐实①（**API 口径，下次直接照抄**）：CANN 9.0 真实签名是 `SetFlag<HardEvent event>(int32_t eventID)` —— **事件号是实参、不是模板参数**（`asc/include/basic_api/kernel_operator_block_sync_intf.h`；`enum class HardEvent : uint8_t` 在 `asc/impl/basic_api/kernel_event.h`），且 `eventID` 受 `ASCENDC_ASSERT(0 <= eventID < QUE_MAX_EVENT)` 约束（`QUE_MAX_EVENT` 按 arch 是 8 或 4）。**按旧记忆写成模板实参会直接编不过。**
- 坐实②：计数 4944→2540（`fwd-bf16-small` 同向 12883）说明**事件确实生效了一部分但没有归零** ⇒ "不足"是**纯实测**结论（9 条前向仍全 FAIL）。**具体哪一道序没关净，未逐条证实。**
- 假设（**未继续验证，别当结论**）：要彻底关净需把 `AllocTensor` 也纳入事件保护窗口 = 绕开 `BufferT`/`TQue` 手写 UB 管理，**风险与收益不成比例** ⇒ 停在 V3。
- ⇒ **教训**见 §4.4。**V2 的报错为什么差点漏掉**：`sync failed` 那行**带前导缩进**，`grep -E "^\["` 会把它们全滤掉 ⇒ 我一度误报"只是数值错、没报错"。查运行期错误要 `grep -n "sync failed"` 或 `ASCEND_GLOBAL_LOG_LEVEL=1` 再看 `head -25`。

#### 11.10.2 为什么 V3 的"串行化代价"在本 kernel 里比名义上小

§11.9.6 当初对 (c) 的担心是"把 m 次写出串行化、大 S 下带宽有代价"。实际放置点是 **`DeQue` 之后、`for k` 之前**：① 它只把"开始读 UB"推迟到"上一次 MTE1 全部完成"，**`for k` 内 m 次 `DataCopyPad` 之间仍然流水**；② 前向**没有 VEC 指令** ⇒ `PIPE_ALL` 相比单条 `PIPE_MTE1/2` **不额外等待任何在途流水线**。⚠️ 但**跨任务之间确实多了一道全同步**（每个 `ForwardOneBlock` 一次）⇒ 真实开销由 §11.11 定档 = **+5.5%**。

#### 11.10.3 修复后真机全量矩阵（`npu_debug/logs/npu_allgroups_fixed_20260920_2056.log`）

| 轮 | 组 | 用例数 | 结果 |
|---|---|---|---|
| 1–2 | `quick` ×2（**同参重复跑 = 确定性对照**） | 25 / 25 | `ALL PASS fail=0 err=0`，两轮逐条计数一致 |
| 3–7 | `medium` / `mtile` / `bnd` / `ub48` / `large` | 4 / 12 / 15 / 4 / 4 | 各轮 `ALL PASS fail=0 err=0` |
| | **合计** | **89** | **7/7 轮全绿**；`mismatch=0` 本身即蕴含"无漏写区域"（输出预置 `0x7fff`，漏写必计入 mismatch） |

前向位级精确抽样（全部 `maxdiff=0.00000`）：`fwd-fp16-S1D1` **0/2** · `fwd-fp16-MTL-D70000-m2` **0/140000** · `fwd-fp16-large` / `fwd-bf16-large` 各 **0/469762048**。反向延续 §11.9.3 的 24/24，本轮含 `bwd-fp16-sat-m2` **0/512 PASS**（§11.10.5）。

#### 11.10.4 两个"假成功"运维坑（下次直接照抄）

1. **`build_npu.sh` 只判"包是否存在" ⇒ 改动没进产物却报 done**：第一次重编后 `build_out/libcust_opapi.so` 时间戳仍是 19:40:44 未变。已加**源陈旧判定**：`find op_kernel op_host CMakeLists.txt -type f \( -name '*.cpp' -o -name '*.h' -o -name 'CMakeLists.txt' \) -newer build_out/libcust_opapi.so` 非空 → 强制 `rm -rf build_out` 重建；日志里要看到 `### stale: … -> 强制重建` + `ts: 1789908908 -> 1789909037` 两行才算真编过。
2. **远端有两份目录**：`~/code1`（早期 `cd ~ && tar xf -` 落错的位置）与 **`~/ops_comp/code1`（真正构建/运行的树）**。我一度对**错的那份**做 md5 复验并据此宣布"推送一致" ⇒ **md5 必须在构建树内核**。现已两侧统一 `~/ops_comp/code1`（真机 `02aeb` 与仿真机 `tpm0u` 均已核对 = `53ee60e5…`）。

#### 11.10.5 fp16 溢出：按批准改**参考**，用例转 PASS

`npu_debug/test_mhc_expand_npu.cpp` 的 `ref_backward` 在 `fill_mode==2`（输入常数 32768）下改为 `m*32768 > 65504 → INFINITY`；CPU 仿真侧 `cpu_debug` 的饱和期望分支**保留不动**（仿真的 `Cast` 确实饱和到 65504，**两侧各按各自的硬件语义对拍**）。改完真机 `bwd-fp16-sat-m2` 由 512/512 全错 → **0/512 PASS**，坐实"真机 = IEEE inf"。
⚠️ 这只解决**我方对拍**：**比赛平台判 `inf` 还是 `65504` 仍未核实**（§4.3），两次提交未被触发（§13.1）。

#### 11.10.6 仿真侧复验

云端仿真机 `tpm0u` 用**修复后的 kernel** 重编（`build rc=0`、`error:` 0 行）→ quick 全量 **29 条 ALL PASS、FAIL=0**。日志 `cpu_debug/logs/sim_quick_full_barrier_20260920_2109.log`，md5 `dadd14dc…` **远端=本地、CR=0**。⚠️ 更早那份只有 6 条的日志**不得当证据**（§8.4 的 `head` 坑）。

#### 11.10.7 偏离声明与回滚

用户批准 (a) 事件配对；实测 (a) 不足、最终落地当时列为 (c) 的 `PipeBarrier<PIPE_ALL>` ⇒ **偏离已明写在此，供追溯**。
**回滚只需** `cp op_kernel/mhc_expand.cpp.bak_pre_fwd_event op_kernel/mhc_expand.cpp`（回到 V0，前向会重新全红）。中间版本留档：`op_kernel/mhc_expand.cpp.bak_pre_fwd_event` = V0、`npu_debug/mhc_expand.cpp.events_v1` = V1。

### 11.11 `msprof` 性能基线与 barrier A/B（21:26–21:30）

**目的**：把 §11.10.2 遗留的唯一未定量项定档，同时落性能基线。用户批准口径：**"按 fp16-large 跑 V0 对 V3"**。
**形状与流量**：fp16 `large` = `S=8192, D=7168, m=8`，host tiling 决策 `blk=40`。单任务读写合计 `x`=117,440,512B + `o`=939,524,096B = **1,056,964,608B ≈ 1.057GB**（反向量级相同）。每版重复下发 **21 次**，统计**剔除首任务**（冷启动）后的 20 样本。

#### 11.11.1 采集通路（三件全部**非提交**，md5 见 §6.2）

- **启动器 `prof` 组**：`run_case()` 新增 `reps` 形参与 **rc=3=PROF** 语义 —— 只做"重复下发 + `aclrtSynchronizeStream`"、**不对拍**。放进程内的原因：`msprof` 要采**单进程多次下发同一算子**，外部循环脚本拿不到逐任务记录；`MHC_REPS` 默认 21（下限夹到 2）。
- **`prof_npu.sh`**：新式用法 `msprof [args] <app> [app args]`，`--task-time=on --ai-core=on`；`command -v msprof` 为空即 `exit 2`；OPP 包按 `[ "$SRC" -nt "$DST" ]` **自动刷新**（§11.10.4 坑① 的同族防线）；`--output` 落**工程内** `npu_debug/prof/<tag>_<ts>/`；采完 csv 计数为 0 → 打 `PROF_MISSING` 并 `exit 5`。
- **`prof_sum.js`**：`op_summary.csv` 摘要器 —— 列名带 `(us)` 后缀需先剥（`aiv_mte2_time(us)`，**按前缀匹配**）；方向按 `Input Shapes` **维度数**判（2=前向 / 3=反向）；`drop = a => a.slice(1)` 剔首任务；输出 mean/min/p50/max 与各 DMA 通道均值。

命令形态：`msprof --task-time=on --ai-core=on --output=npu_debug/prof/<tag>_<ts> ./npu_debug/test_npu 50 64 prof`。

#### 11.11.2 实测数据（单位 µs，n=21 / 剔首 20）

| 版本 | 方向 | 剔首 mean | min | p50 | max | `aiv_time` | `aiv_mte2_time` | `aiv_mte3_time` | `aiv_vec_time` |
|---|---|---|---|---|---|---|---|---|---|
| **V3**（当前提交版，带 barrier） | fwd | **1117.3** | 1111.0 | 1116.6 | 1124.3 | 1095.1 | 213.2 | 1088.4 | 0.0 |
| **V0**（无 barrier，前向结果错误） | fwd | **1059.0** | 1054.4 | 1058.9 | 1064.7 | 1041.0 | 229.6 | 1038.3 | 0.0 |
| **V3** | bwd | **1088.3** | 1079.6 | 1086.9 | 1098.6 | 1056.3 | 1013.7 | — | 385.0 |
| **V0** | bwd | **1085.6** | 1079.5 | 1085.7 | 1094.6 | 1056.2 | 1013.6 | — | 385.0 |

原始摘要：`npu_debug/logs/msprof_ab_v0_v3_fp16large.txt`（`00583bfa…`）。

#### 11.11.3 三条定档

1. **barrier 代价 = 前向 +58.3µs / +5.5%**（1117.3 vs 1059.0）。两版分布**完全不重叠**（V3 `min`=1111.0 > V0 `max`=1064.7）⇒ 不是采样噪声，是真代价。
2. **环境未漂移，所以归因成立**：反向是本轮的天然对照组（barrier 一行不在反向路径上），V3 vs V0 只差 **+0.25%**（1088.3 vs 1085.6）⇒ 那 +5.5% 不能推给"两次采集之间机器变慢"。
3. **时间落点在写出 DMA**：barrier 的增量几乎全进 `aiv_mte3_time`（1038.3 → **1088.4**，+50µs），`mte2` 反而从 229.6 降到 213.2 —— 全同步把 MTE2 与上一次 MTE1 拉开，读入通道不再和写出抢口。有效带宽（剔首 mean）：**前向 V3 946GB/s / V0 998GB/s，反向 971GB/s**；前向与反向仅差 2.7%。
   ⇒ **V0 已经贴住这个核能达到的上限，barrier 把它拉回到与反向同级**。剩下 5% 是"要正确性就得付"的钱，**不要试图用更窄的 barrier 换回来**（§11.10.1 已实测窄 barrier 运行期 trap）。
> ⚠️ **本表不是 §9 两项优化的收益证据**（§11.11 只做了"加 barrier vs 不加"）⇒ 见 §4.3 第一条。

#### 11.11.4 下一条性能线索 → **已闭合**

本节原设想"把逐副本 `for k` 的 m 次 `DataCopyPad` 合并为一次 `DataCopyExtParams{blockCount=m, srcStride=0}`，收益上界 ~50µs"。⇒ **§14.5 已用真机把这条路线否决**（arch22 的 gap/stride 语义不可依赖文档推断，且 §14.1 证明 DMA 调用数不是瓶颈）。**不要再回到这里翻案。**

#### 11.11.5 运维：切换、还原、留证

- **换 V0 前先保命**：`cp op_kernel/mhc_expand.cpp op_kernel/mhc_expand.cpp.v3.keep`；采集完**还原 + 构建树内 md5 复验 = `53ee60e5…`（必须在 `~/ops_comp/code1` 内核）+ `run_npu.sh quick` 25/25 `ALL PASS`** 三条齐了才判"环境复原"，随后删 keep（⚠️ keep 已删，长期回滚点仍是 `.bak_pre_fwd_event`）。
- 判读口径：**纯向量核的 `aicore_time(us)` 恒为 0，要看 `aiv_time(us)`**；`aic_*` 全 0 是正常（本 kernel 无 Cube），别据此判"profiler 没采到"。产物 **当场 `tar` 管道回本地 + 逐文件 md5 双侧复验**（§6.3）。

---

## 12. 比赛平台提交通道与平台实态

### 12.1 换题三步（`cannjudge-submit` CLI / API，✅ 已实测可用）

1. **按题目名查 `problem_id`**：`GET /api/problems/name/{problemName}`；`problemName` = **算子名去下划线全小写**（题 1 = `mhcexpand` ✅ 一次命中；`mhc_expand` / `mhc-expand` 均 **HTTP 404**；题 3 = `sparseflashattention`）。
   ```python
   problem = client.get_problem("mhcexpand")   # 换成目标题名
   problem_id = problem["_id"]
   client.submit(problem_id=..., kernel_cpp=..., tiling_h=..., tiling_key_h=..., host_cpp=...)
   ```
   ⚠️ **只有 `problem_id` 要换**；四个文件参数对应本地工程（题 1 的字段映射见 §3.1）。
2. **确认账号在目标题的参赛名单里**：查不到该题才可能是权限问题，届时先核对赛区，**不要**把"查不到"直接当成"题不存在"。
3. **CLI 实体位置**：真机 `02aeb` 的 `/mnt/workspace/cann-learning-hub/skills/cannjudge-submit/` —— `python3 cannjudge_cli.py {info,login,logout,download,submit,query,rank}`；`query` **必须带 `--submission-id`**。RSA 密文登录后的会话在 `~/.cannjudge/session.json`（09-19 22:45 签的，实测未过期）。密钥与私钥路径见 `连接信息.md`，**内容绝不进日志/文档**。

> ⛔ 此前"只能交第三题"的判断是**错的** —— 用户 2026-09-20 澄清：是提交命令里 `problem_id` 被写死成第三题的值，**与权限无关**。
> 隧道侧的通用口径（只用 `devspace_tunnel.ps1 -Role npu|cpu` 自举、"账号环境列表里查不到 devEnvId" = 桌面 VS Code 换了登录账号而非回收、**不重试轰炸**）见 `算子开发工作流.md` §4 与 `reference-devspace-environments`。21:58–22:00 那次隧道阻塞已于 22:10 由用户重连解除。

### 12.2 题 1 在比赛平台上的实态（22:15 `info` 实测）

| 字段 | 值 | 含义 / 影响 |
|---|---|---|
| `problemName` / `problem_id` | **`mhcexpand`** / **`6a7c1a74a52e0f540a89d39b`**（`ID=301`，`contest_id=6a7bf087…`） | 提交只需把它传给 `--problem-id` |
| `code_template` | `"custom_template"`（字符串） | `--project-type auto` 据此走 **`registry`（传统工程）**分支，四字段自动 glob 匹配成功 |
| `cann_version` | **`8.5.0`** | ⚠️ 我方全程在 **9.0.0** 上编译验证 ⇒ 平台比我方低一个大版本。`PipeBarrier<PIPE_ALL>` / `DataCopyPad` / `TQue` 在 8.x 即存在 ⇒ ✅ **已由两次提交坐实零风险**（8.5.0 下编译通过且全绿） |
| `score_mode` / `use_baseline` | `0` / `false` | 语义未文档化，先记录不解读 |
| `ongoing` | **`false`** | ⇒ 计分冻结，见 §13.3 |
| 精度判据 | **题面全文（4168 字）不含任何 `rtol`/`atol`/容差数值**，只写"精度保障""float16/bfloat16 累加精度"；`inf`/`65504`/饱和**零处提及** | ⇒ 判据形式由实交反推（`precision_ratio`，§13.1）；溢出分叉大概率不覆盖（§4.3） |
| `last_submission` | 榜单分：`85.2`、`68.92`、`0`… 共 123 条 | 已知有人过 85 ⇒ 本题可拿到非零分，不是"全员 0"的死亡题 |

> ⚠️ **官方题面本身有污染**：`mhcexpand` 的 `desc` 末尾挂着一段 `softmax(src, index=None, ptr=None, dim=0)` 的"公开题面接口"说明和"题面固定 ε=1e-16 → 评测侧 attr_eps"的对照 —— **那是第二题 mhc_sinkhorn 的契约**，被人工拼贴进了第一题题面。⇒ ①别把它当第一题的接口要求；②印证"比赛平台按什么方式调用算子"确实只能实测，**题面文字不可全信**。

---

## 13. ⭐ 首次提交实测（22:15–22:25，`submission_id=6aafea71b0477ec41ea07f63`）

**结果：`状态: Pass`，8/8 测试用例全过，每条 `precision_ratio: 1`。** 原始输出 `npu_debug/logs/submit1_result_8of8_20260920.log`（远端=本地 md5 `32c2594f…`），判因分析 `submit1_analysis_20260920.log`（`0b7c722b…`）。提交源 = §3.2 首交口径（host `a16c371d…` + kernel `53ee60e5…`）。

### 13.1 一次性裁掉四条平台口径（全部并入 §4.3 的"已裁定"清单）

`sum` 口径正确（8/8 未判错）· 精度判据字段就是 `precision_ratio`＝**逐元素通过比例**，我方拿到 `1` ⇒ 任何更严口径下同样满分 · CANN `8.5.0` 编译无碍 · 单算子 + `backward` 属性一次判过 8 条（调用形态不再需要观测）。

### 13.2 性能实态与两条计时口径（⭐ 逐 case 对照表以 §14.7 为准）

- **`best_time` 对**所有**提交都是同一组常数 ⇒ 它是比赛平台的固定基准，不是"榜上最快"**；`time` 单位是 **µs**。
- 首交 8 条：**中大档 6 条已贴基准 1–4%（case 3、7 还反超榜首）**；**唯一的结构性缺口是两条小档（case 1、5）慢 3.0–3.7 倍**（绝对差只有 3.4µs / 2.8µs ⇒ 是**每任务固定开销**，不是带宽）。
- ⚠️ 这个 3 倍**不能**用 §11.11 的 barrier +5.5% 解释（量级差一个数量级）⇒ 定位与处置见 **§14**。

### 13.3 `score=0` 的判因（**不是**我方性能被判差）

探针对照：榜上 **90.15 / 88.17 / 85.20** 三条的 `query` 结果里**每条 case 的 `score` 也全是 0** ⇒ `score` 不在 per-case 层。再把 `info` 的 123 条 `last_submission` 按 ObjectId 时间戳摊开：**非零分 22 条全部 ≤ 2026-09-03 05:48**（区间 44.08–90.15）；**零分 101 条从 08-12 一直连到 09-20 14:15（就是我方这条）**；与 `contest.ongoing = false` 一致（`end_time` 名义上还是 09-28）。
⇒ **计分自 09-03 起对所有人停止写入**，榜首今天重交同样是 0 分，**不能读成"性能不达标"**。有效信号只剩 `precision_ratio` 与 `time`/`best_time`。

---

## 14. ⭐ 小档固定开销定位与"少开核"定则（23:16–23:40，真机 `02aeb`，第二次提交）

> 缘起：用户 23:00 下达"已拿到 NPU，直接连真机开始优化"。§13.2 那条"零成本第一步"（用现成 msprof 通路测小档逐任务时间）本轮做完，并顺它推出了**一条只改 host 的落地规则**。

### 14.1 小档差距 100% 在固定开销，且**不是** DMA 调用数

`fwd/bwd-fp16-small`（S=64, D=256, m=2）读写合计 96KB ⇒ 纯带宽时间 ~0.1µs，而实测 Task Duration 4.1–4.9µs。拆开看（blk=40 时）：aiv ≈ 2.3µs（mte2 ≈ 1.1、mte3 ≈ 0.2、vec ≈ 0）+ 派发/序言 ≈ 1.7µs。
⚠️ 关键否定结论：**首交版在 blk=40 下每核只摊到 1~2 个任务、3~6 次 DMA 调用**，所以"减少调用次数"这类批处理手法对小档**没有靶子** —— 靶子是**开的核太多**（每多一个 block ≈ **45ns** 派发 + 每核固定序言）。
🔑 同一轮由 msprof `Block Num` 裁定：**本机 AIV 实为 40 核**（不是 config 里的 50；harness 的 `blk=50` 只是注入值）。

### 14.2 blockDim 扫描（msprof `Task Duration` 剔首 mean，单位 µs，同一会话内横比）

| blockDim | fwd-fp16 | bwd-fp16 | fwd-bf16 | bwd-bf16 | 合计 |
|---|---|---|---|---|---|
| 2 | 7.0 | 11.8 | 7.0 | 11.8 | 37.6 |
| 4 | 4.4 | 6.7 | 4.3 | 6.7 | 22.1 |
| 8 | 3.2 | 4.4 | 3.3 | 4.3 | 15.2 |
| 12 | **3.0** | 4.0 | 3.1 | 4.1 | 14.2 |
| **16** | 3.2 | **3.4** | **2.9** | **3.6** | **13.1** ← 谷底 |
| 24 | 3.4 | 4.0 | 3.7 | 3.7 | 14.8 |
| 40（首交实际） | 4.1 | 4.6 | 4.4 | 4.9 | 18.0 |

⇒ 曲线 **12~16 见底、24 起回抬**；首交的 40 核处在**明显过并行**区。原始 csv 36 份已回捞：`npu_debug/prof/blockdim_20260920/{b2..b40,rule}_c*/`，stdout 摘要 `npu_debug/logs/blockdim_sweep_20260920.log`。

### 14.3 落地规则：**每核 IO 不足 6KB 就不再开核**（只改 `op_host/mhc_expand.cpp`）

```cpp
io_bytes    = (1 + m) * S * D * elem_size      // 前向 x+o 与反向 x+o 同一条式子
block_dim   = min(block_dim, max(1, io_bytes / 6144))
```

6144 = 6KB/核，由 §14.2 反推：小档 96KB / 6KB = **16 核**，正落在实测谷底；medium(42MB) / large(8.4GB) 离拐点两个数量级 ⇒ 规则对中大档是**恒等变换**，仍满 40 核。生效阈值：只在 `io_bytes < 40×6KB = 245KB` 时才可能改变结果。
⚠️ **核函数、tiling 结构体、tiling key 一字未改**（`53ee60e5…` / `f759a052…` / `267e0125…` 与首交逐字节一致）⇒ 首交已证过 CANN 8.5.0 能编过，本轮残留风险只剩 host 侧一段整数算术。

### 14.4 规则版复测（`prof_matrix.sh rule 0..7`，无环境变量）

| 用例 | blk | 本轮 | 首交(blk=40) | 变化 |
|---|---|---|---|---|
| fwd-fp16-small | 16 | 3.2 | 4.1 | **−22%** |
| bwd-fp16-small | 16 | 3.5 | 4.6 | **−24%** |
| fwd-bf16-small | 16 | 2.9 | 4.4 | **−34%** |
| bwd-bf16-small | 16 | 3.6 | 4.9 | **−27%** |
| fwd-fp16-medium | 40 | 13.2 | — | 规则未触及 |
| bwd-fp16-medium | 40 | 23.4 | — | 规则未触及 |
| fwd-fp16-large | 40 | **1117.8** | §11.11 基线 1117.3 | +0.04%（环境未漂移的对照） |
| bwd-fp16-large | 40 | **1084.8** | §11.11 基线 1088.3 | −0.3% |

正确性闸门（同一 build）：`quick` 25 条 **ALL PASS fail=0 err=0**、`medium/large/mtile/bnd/ub48` = 4+4+12+15+4 全绿、`all` 模式**连跑两遍**各 45 条全绿（同参复验 ⇒ 无新增非确定性，两份日志**同一个 md5 `a0a91cca…`**），前向用例 `maxdiff=0.00000 / bitmis=0` 位级精确保持不变。

### 14.5 ⛔ 被真机否决的路线：分块 + 跨步 DMA（`splitMode=3`，已整段回滚）

设想是"一任务读 T 行、写 m 次跨行"（每任务 DMA 调用数固定为 `1+m`，与 T 无关），用 `DataCopy` + `DataCopyExtParams/Params{blockCount, blockLen, srcGap, dstGap}` 表达跨行。真机裁定：

| 注入的 T | 现象 |
|---|---|
| 2 | `mismatch 26810/32768`（82%），`maxdiff=3.5` |
| 8 / 16 | 该用例起整卡被毒化，后续用例 `launch failed st=361001` |
| ≥32 | 稳定设备异常：`errcode 0x200000000` **"The write address of the MTE instruction is out of range"**（aivec，`mte error info 0x1000000ec`，多核同一 PC） |

两种字段语义解释（`Gap` = 块间隔 vs `Stride` = 块首距）**各自都只能解释一半现象**，尤其是 **T=1 时 `blockCount=1`、间隙项根本不参与，结果仍非确定性 mismatch** ⇒ 不是"换个字段值"能救的。加上 §14.1 已证明**调用数不是瓶颈** ⇒ 该路线的**前提就不成立**，3 个构建周期后止损，kernel 与 tiling.h 回滚到基线 md5（`.bak_pre_chunk` 两份，回滚后 md5 复核一致）。
⚠️ 沉淀一条口径：**arch22 上 `DataCopyParams` 的 gap/stride 字段语义不可依赖文档推断** —— CANN 开源包里 `srcGap/srcStride` 是同字段 union、`c_api` 头无 doxygen、`pto` 侧只在 ND2NZ 编码里出现 ⇒ 只能靠**受控探针实验**定档，而探针本身的成本高于本轮收益。
> **2026-09-21 修订（§14.9）**：这条"疑案"已用一次单核受控探针结案。上表里那句"T=1 时间隙项不参与却仍非确定性 mismatch"**不再需要当作反证** —— §14.9 用 4 行改动在同一台机器上复现了"地址布局确定、内容不确定"的分离，根因是**多块 DMA 的源侧会前进、读到不属于本块的 UB 邻居**，而不是"语义整体不可依赖"。本节结论（该路线回滚、不进提交）**不变**。

### 14.6 归因注入已全部清除（提交前闸门）

`[ABL-SK]` 的 `MHC_SK_T` / `MHC_SK_BLK` 两个 `getenv` 开关与 `#include <cstdlib>` **已整段删除**；提交源四文件合规 grep（见 §3.3 的命令）**零命中**。留在 `npu_debug/` 里的 `MHC_PCASE`/`MHC_REPS`/`MHC_OPAPI_SO` 属调试工装，不在提交面内。

### 14.7 第二次提交实测（`submission_id=6aaffd92b0477ec41eb14eb0`，23:36 交、23:41 出分）—— ⭐ **逐 case 对照表的权威版本**

**结果：`状态: Pass`、8/8 全过、每条 `precision_ratio: 1`。** 原始输出 `code1/npu_debug/logs/submit2_result_20260920.log`（远端=本地 md5 `916c5e7b…`）。dry-run 恰好 4 字段，`tiling_h`/`tiling_key_h`/`kernel_cpp` 的 sha256 与首交逐项一致，**只有 `host_cpp` 变化**：9143B/`32cce2e9…` → **10343B/`278b7b84…`**。

| Case | 首交 `time` | **本轮 `time`** | 变化 | 平台基准 `best_time` | 对基准比值 | 榜首 `time` |
|---|---|---|---|---|---|---|
| 1 | 5.02 | **3.82** | **−23.9%** | 1.36 | 3.69× → **2.81×** | 1.42 |
| 2 | 17.80 | 17.60 | −1.1% | 14.02 | 1.27× → 1.26× | 16.06 |
| 3 | 1377.98 | 1377.24 | −0.05% | 1363.56 | 1.01× | **1388.44（我们更快）** |
| 4 | 48.20 | 49.48 | **+2.7%** ⚠️ | 45.36 | 1.06× → 1.09× | 48.08 |
| 5 | 4.44 | **3.96** | **−10.8%** | 1.48 | 3.00× → **2.68×** | 1.60 |
| 6 | 891.26 | 887.32 | −0.4% | 863.88 | 1.03× → 1.027× | 883.78 |
| 7 | 733.90 | 735.02 | +0.15% | 726.94 | 1.01× | **738.76（我们更快）** |
| 8 | 1038.24 | 1037.14 | −0.1% | 995.84 | 1.04× | 1016.66 |

**判读**：
1. ✅ **两条目标小档按预期落地**：case 1 −24%（真机同口径 −22%）、case 5 −11%（真机 −24%）。case 5 平台侧少赚 13 个点 ⇒ 平台计时里有一截**与核数无关**的固定量（≈0.5µs 级），与"真机 4.9µs ↔ 平台 5.02µs"的对表结论一致。
2. ✅ **中大档 6 条按规则应当一动不动**，实测 5 条在 ±1.1% 内 ⇒ 规则是恒等变换的判断成立。
3. ⚠️ **case 4 +2.7% 判为平台噪声，不是回归**：规则只在 `io_bytes < 245KB` 时生效，而 case 4 用时 49µs、IO 远大于该阈值 ⇒ 改前改后走**同一条代码路径**；同轮 case 7 反向 +0.15%、case 3/6/8 全在 ±0.4%，量级一致。⛔ 但这条**未被复测排除**：要坐实只能重交取分布，而 §13.3 已证计分冻结 ⇒ 不值得为它花提交位。
4. ⛔ **`score` 仍为 0**，与 §13.3 判因一致，**不能读成性能变差**。

### 14.8 剩余缺口与下一步

- **题 1 已无可动的低成本项**：小档剩下的 ~3.8µs 里，派发 + 每核序言占 ~1.7µs、DMA 落地 ~1.1µs，而 96KB 的纯带宽只有 0.1µs。要再往下只能动"每任务固定成本"本身（核函数序言、tiling 读取、UB 分配），而 §14.5 已证明这条路在 arch22 上连受控实验都不好做 ⇒ **投入产出比极低**。
- 中大档 6 条已在基准 1–9% 内、两条反超榜首 ⇒ **带宽侧无空间**。
- ⭐ **下一条线索（未实测、未排期、比题 1 残余 2µs 更值钱）**：前向/反向仍是逐副本 `for k` 各一次 `DataCopyPad`（`blockCount` 固定 1，§4.1）。合并为一次 `DataCopyExtParams{blockCount=m, srcStride=0}` 的设想在**跨行**形态下被 §14.5 否决，但**同块连续 m 份**（前向写出本就是 `m*D` 连续，§2.1）这一形态**没有**被单独测过 —— 若将来要试，必须先按 §14.5 的口径做受控探针。
  → **2026-09-21 已测掉：见 §14.9。该形态在真机上不成立，原因是 arch22 这条指令没有"源冻结"编码，不是单位填错。**
- ⇒ **题 1 建议转入"是否还有别的题更值"的判断**（题 2 向量版性能等真机、题 3 的 P0.5 修 `CalcUbNeed`）。

### 14.9 ✅ P1 受控探针结案：arch22 的多块 DMA **没有"源冻结"编码** ⇒ 合并副本这条路从"没测过"变成"原理上不成立"（2026-09-21 01:20 真机时钟，`02aeb`）

> 缘起：用户选定"先跑 P1 探针，低成本裁掉疑案"。靶子是 §14.5 那句未解读数（"T=1 时间隙项根本不参与却仍非确定性 mismatch"）与 §14.8 那条 ⭐ 未测线索。**成本：1 次构建 + 8 次进程启动 ≈ 4 分钟；没有花提交位（`ongoing=false` 下分数不动，见 §13.3）。**

**方法（单核受控，全在隔离树 `~/ops_comp/probe1`）**：形状固定 `S=64, D=256, fp16, 前向` ⇒ `dTileNum=1`、`splitMode=ROW`（每行一个任务）；host 侧把 `block_dim` 强制成 1；把 `ForwardOneBlock` 的逐副本 `for k` 换成**一次** `DataCopyPad(o_gm_[dst0], x_local, DataCopyExtParams{m, rb, srcStride=0, dstStride=X, 0})`（rb = `cur_h*elem_size` = 512 B）。X 的取值随 `m` 变，使**三种互斥口径各自只有一条能成立**：

| m | dstStride 填法 | 若成立则正确的那种口径 | 结果 | mismatch | `nan_unwritten` |
|---|---|---|---|---|---|
| 8 | 走原逐副本循环（正对照） | — | **PASS** `maxdiff=0.00000` | 0/131072 | 0 |
| 2 | `0` | 块间隔(gap)/字节 | FAIL | 16128/32768 = 63×256 | **0** |
| 3 | `rb` = 512 B | 绝对步距/字节 | FAIL | 31488（复测 32000）/49152 | **256** |
| 4 | `rb/32` = 16 | 绝对步距/32B 块 | FAIL | 47888/65536 | 1536 |

**裁定**：
1. ⛔ **三种口径无一通过** ⇒ "把 m 次副本 DMA 并成 1 次"在前向**不成立**。§14.5 当年的回滚判对，§14.8 的 ⭐ 线索正式关闭。
2. 🔑 **`srcStride=0` 不是"源冻结"，而是"源也按 blockLen 前进"** ⇒ 这条指令**没有广播形态**。证据链（m=2）：`nan_unwritten=0` 且每行 `[0,256)` 全对 ⇒ dst 侧确实按 rb 连续铺开；错的只有后副本的**内容**——而 `got[256..263] = [-1.75,-0.75,0.25,1.25,-1.375,-0.375,0.625,1.625]` 与 harness 生成器 `f(i)=((37i+11)%29-14)*0.125` 在 `i=256..263` 的取值**逐个相同**（= 输入 `x` 的第 1 行）⇒ 第二块读的是**源块之后相邻的 rb 字节**，不是 `x_local`。
   ⚠️ "邻居内存里为何正好是 `x[i+1]`"有两种都能自洽的机制（队列另一节点被下一任务的载入覆盖 ⇒ 无 barrier 的竞争；或源地址直接在 GM 侧前进），**本轮未区分**——因为对决策等价：**源停不住**。
3. ✅ **`DataCopyExtParams` 的 dst 字段 = 块间隔(gap)，单位=字节**（这是本轮最硬的一条单位结论）：m=3 填 `512 B` 时 `nan_unwritten=256` = **恰好一行一槽**，只有 gap 语义能给出这个数（行 i 的三块落 `+0/+2rb/+4rb`，每行的中间槽由上一行的第三块补上，只有最后一行的第三槽无人写）；"绝对步距/字节"在该填法下**必须 PASS**（它没通过），"绝对步距/32B 块"会留下几千个 NaN（实测 256）。
4. 🔑 **地址布局确定、内容不确定**（这正是 §14.5 那堆混乱读数的最小复现）：m=3 复测 `nan_unwritten` 恒为 256，而 `mismatch` 从 31488 变到 32000 ⇒ 落点由参数决定、读到的内容由竞争决定。m=2 三次启动 `16128/32768` 逐次相同（该形态下竞争结果稳定）。§14.5 的"越界 + 非确定性"两条现在分别是**单位混用**（`DataCopy` 下 blockLen×32、`DataCopyPad` 下按字节）与**源侧读邻居**所致，不必再解释成"arch22 语义整体不可依赖"。
5. ⚠️ 诚实标注：m=4 的 `nan_unwritten=1536` 不落在任何单一线性模型上（gap/字节模型预测 3072）⇒ 块数更多时源、目两侧前进会叠加，**本轮不假装已完全建模**；上面 1~4 条的裁定不依赖 m=4。
6. ⚠️ 另一条未直接复核的口径：日志里的 `blk=16/21/26/48` 是 **harness 的预测值**（`test_mhc_expand_npu.cpp:58` 明写），设备实际 `block_dim` 由 host 补丁强制为 1；本轮没用 msprof `Block Num` 去坐实"只开了 1 个核"。这不影响裁定（每行的落点/内容只由任务号决定，任务间不重叠），且 8 次启动 `err=0`、无 `launch failed`、无设备异常 ⇒ 卡片未被毒化。
7. ⭐ **gap 语义下唯一还能表达的合并形态（本轮未测，前向测不到）**：反向"一行的 m 个跨行副本 → UB 连续" = `{blockCount=m, blockLen=rb, srcStride=(m-1)*rb, dstStride=0}` —— 源每块前进 `m*rb` 正好落在下一个副本上、目的连续 ⇒ 与上面的机制**同构而反向**，是把 m 次读并成 1 次的唯一写法。目标只有 bwd-medium（本地 23.4 µs，mte2 17.3 / scalar 16.8）值得打；bwd-large 的 mte2 已 96–99% 饱和 ⇒ 并调用数不会赚。**要不要花一个构建周期测它 = 下一个决策点，而 `ongoing=false` 意味着它只影响文档完整性、不影响分数。**

**工装与闸门（全部留在调试面，不进提交）**：
- 隔离树 = 服务端 `cp -a ~/ops_comp/code1 ~/ops_comp/probe1` 后删 `build_out/` 与 `npu_debug/pkg/`；打补丁前先确认三份源 md5 与基线逐字节一致。
- `code1/npu_debug/probe_dma/`：`mk_probe.py`（6 条锚点**先全部验唯一**再落笔，输出 `all 6 anchors unique`，每文件留 `.bak_pre_probe`）、`fwd_probe.cpp`（探针本体）、`run_probe.sh`（一次构建 + 每变体独立进程 + 控制在前，`VAR="2 3 2 3"` 可只复测）。
- 探针版三文件 md5：kernel `53ee60e5→770b564a`、host `c76d30be→16e1da6f`、harness `d23a1647→65e33fcc`（4 个变体共用同一份构建）。
- **提交面完好（本地 + 构建树 `~/ops_comp/code1` 双向核对）**：kernel `53ee60e5…` / host `c76d30be…` / tiling.h `f759a052…` / tiling_key `267e0125…` 与 §14.7 定档值一致，合规 grep（`printf|fflush|fprintf|std::cout|TODO|FIXME|#if 0|getenv|PROBE`）**零命中**。
- 证据：`code1/npu_debug/logs/probe_dma_20260921_012037.log`（md5 `534ee28c395fde5cba778b897feef27c`，远端=本地，4 变体各 1 次）与复测 `probe_dma_20260921_013132.log`（md5 `b54856065ef374407cafcdcce2cfe935`，`m=2/3` 各 2 次；日志里 `op package up-to-date, ts=01:20:58` + `libcust_opapi.so ts` 未变 ⇒ 与首轮**同一份构建**，差异只可能是运行期）。隔离树 `~/ops_comp/probe1` 当时保留未删（探针可复现的前提），§18.7 已归档为 `npu_debug/archive/probe1_sub3_20260921.tar.xz`（含探针三文件与 118 份运行日志）后回收。

> **交接状态（2026-09-20 23:45 → 2026-09-21 文档整理）**：本轮 = "查小档固定开销"那条零成本第一步的完整执行（定位 → 扫描 → 定则 → 回归 → 提交），提交源只有 `op_host/mhc_expand.cpp` 一个文件变化（`c76d30be…`）。
> 2026-09-21：本文按"删无用 + 合并同类"重构（A 类），通用坑移入 `算子开发工作流.md §6`（B 类）；**结论、数字、md5 一字未改**，§10.0 与 §14 等被外部引用的锚点全部保留。

> **交接状态（2026-09-21 01:20–01:35）**：本节 = §14.5 疑案与 §14.8 ⭐ 线索的结案。**题 1 的性能结论收口**：小档靠"少开核"已吃掉 −11~−24%，剩余 2.4 µs 差距在"每核固定成本"里（§14.1/§14.2），而 §14.9 证明"合并 DMA 调用"这条唯一的低成本改形路线原理上不成立 ⇒ **题 1 无剩余可动项**，下一步只在"是否顺手测 §14.9 第 7 条的反向单次 gather"与"转题 2/题 3"之间选。

---

## 15. 本题专属的运维/判读坑（通用条目见工作流 §6，此处只留"题 1 才会遇到"的）

| 坑 | 现象 | 正确写法 |
|---|---|---|
| 拿 harness 的 `blk=50` 当硬件事实 | 以为 910B3 有 50 个 AIV | **`num_aiv` 实为 40**，由 msprof `Block Num` 裁定（§14.1）；`SetBlockDim` 超量不会报错但只是排队 |
| 对"远端另一份目录"做 md5 复验 | `~/code1` vs 构建树 `~/ops_comp/code1`，对错了等于没验 | **md5 必须在构建树内核**（§11.10.4 坑②） |
| `build_npu.sh` 只判"包是否存在" | 改动没进 `.so` 也报 done | 源比产物新 → 强制重建 + 产物新时间戳 + `nm -D` 三件齐（§11.10.4 坑①） |
| 把 `libcust_opapi.so` 做成链接期依赖 | 25/25 `st=561002` | 一律 `dlopen`/`dlsym` + 让框架自己按 `ASCEND_CUSTOM_OPP_PATH` 加载（§11.9.2） |
| 外部循环脚本调 `msprof` | 拿不到逐任务记录 | **`msprof [args] <app>` 单进程内连发**（启动器 `prof` 组，§11.11.1） |
| 看 `aicore_time` 判"profiler 没采到" | 纯向量核恒为 0、`aic_*` 全 0 | **看 `aiv_time(us)`**；`aic_*`=0 是正常（本 kernel 无 Cube）（§11.11.5） |
| 用 `grep -E "^\["` 查运行期错误 | 带前导缩进的 `sync failed` 全被滤掉 ⇒ 误报"没报错" | `grep -n "sync failed"` 或 `ASCEND_GLOBAL_LOG_LEVEL=1`（§11.10.1） |
| 只跑一轮就宣布全绿 | V0 失配计数每轮不同 | **同参重复跑**（V3 靠 quick×2 确认、§14.4 靠 `all`×2 同一 md5 确认） |
| 换 V0 采数据后忘了还原 | 提交源变成"结果错误版" | keep + 还原 + **构建树内 md5 复验 + quick 25 条 ALL PASS** 三条齐才判"环境复原"（§11.11.5） |
| 把平台 `best_time` 当榜上最快 | 以为已经输了 3 倍 | 它是**对所有提交都相同的固定基准**；真榜看 `last_submission`（§13.2） |
| 以为 `DataCopyExtParams{srcStride=0}` = 源冻结（广播） | 合并成 1 次 DMA 后**每行第一副本全对、后副本内容错、`nan_unwritten=0`** | arch22 多块 DMA **两侧都按 `blockLen+gap` 前进**，广播形态不存在（§14.9）；gap 单位=字节，且 `blockLen` 在 `DataCopy` 下×32、在 `DataCopyPad` 下按字节 |
| 以为"跑过 = 比过数值" | **`run_case(reps>1)` 在计时分支直接 `return 0`** ⇒ `prof`/`prof2` 打出的 `ALL PASS \| fail=0` 只代表"启动没报错"。删光前向全部序的 `none` 在 c0/c4 上照样 "ALL PASS"（§17.1） | 计时分支现在打完绩行**继续走数值校验**，判决行带 `timed=1`；引用任何 `ALL PASS` 前先确认它来自带 `timed=` 的新 harness |
| 在隔离树里"重编了就以为跑了新核" | `MhcExpand_<hash>.o` 文件名里的哈希**不覆盖内核源码**，而 `build_npu.sh` 从不刷新 `npu_debug/pkg`（运行期真正加载处）⇒ optA 的 pkg 停在 02:16 那版，把"未打补丁的已提交核"跑出 `rc=124`，连带 R5~R8 四组读数全废（§17.2） | `build_npu.sh` §1.5 每次 `rm -rf pkg` 重建并断言 pkg `.o` == build_out `.o`；`one_case.sh`/`prof_matrix.sh` 跑前硬校验并打印 `kernel_o=`，不一致 `exit 95/9` |
| 用 ssh 嵌套引号写"多轮 × 多用例"的 for 循环 | 8 轮 prof2 全跑成 case 0，`ROUND$r_EXIT` 变成空串 ⇒ 看着像"全通过" | **这类循环一律落成仓库里的脚本再推上去执行**（本轮起 `npu_debug/opt_a/run_gate.sh`），远端只留一条 `bash xxx.sh` |
| 拿 `prof_sum.js` 复取历史 sweep 读数时把目录下 4 个 CSV 一起传 | node 在缺 `Input Shapes` 列的那份上抛 `TypeError` 退出 ⇒ 整目录零输出，看起来像"CSV 已被清掉" | 复用时照 `prof_matrix.sh` 的过滤器：先 `head -1 \| grep 'Task Duration'` 逐份筛，再按目录传（§18.2 的 sweep1 表就是这么救回来的） |
| A/B 驱动 `run_r10.sh` 的 grep 过滤器 | 只剩时刻和 `流水线均值`，`剔首 mean` 整行被滤掉 ⇒ 一轮上机白跑 | 过滤器必须含 `fwd: n=\|bwd: n=`；已修，但**新写 sweep 脚本时先确认判决行在过滤集里** |
| 拿 `ub_size/4` 这类"看起来随意的预算常数"当可调项 | 前向攒批要 4 块环，把 `/4` 放宽到 `/2` 就是静默 UB 溢出 | 该常数现在是 §18.1 环深的上限，**两侧注释已写明"要一起改"**；动预算前先数核内同时持有几块 tile |

---

## 16. ⭐ 第三次提交：A1+A2 采纳（2026-09-21 02:04–02:10，真机 `02aeb`）

**采纳内容**（只动 `op_kernel/mhc_expand.cpp` 的前向，`op_host`/`tiling`/`tiling_key` 三份 sha256 与第二次提交逐字节相同）：

| 项 | 改动 | 真机 A/B 判定 |
|---|---|---|
| **A1** | `Init()` 里前向不再 `InitBuffer` 反向用的 4 块 UB（`in/out_que_`、`acc/tmp_`） | **0**：fwd-small 2.9 vs 2.9µs，两组交替配对里都没有差别 ⇒ 不进结论，只作为"前向少占 UB"的整洁改动留下 |
| **A2** | 前向去 `TQue`（队列簿记）改两块 `TBuf` 轮转 + 每任务一次 `PipeBarrier<PIPE_ALL>` | **−0.2µs（−6%）**：fwd-small 2.9/2.9 vs 3.1/3.1，两组交替同向；bwd-small 作对照组不变 ⇒ 采纳 |

**提交 3**：`submission_id=6ab0208fb0477ec41ec2b551`，`状态: Pass`、8/8 用例、每条 `precision_ratio: 1`。dry-run 清单：`kernel_cpp` 9,829 B / sha256 `1f2409785e8b5fef7350d1e58f45b909eda4ac457de0e4b87dcf258143fe273d`，其余三份未变。

**平台逐 case 对照（µs，sub2 → sub3）**：

| case | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| sub2 | 3.82 | 17.60 | 1377.24 | 49.48 | 3.96 | 887.32 | 735.02 | 1037.14 |
| sub3 | 3.52 | 17.78 | 1387.20 | 49.40 | 4.16 | 893.28 | 735.44 | 1039.64 |
| Δ | **−7.9%** | +1.0% | +0.7% | −0.2% | **+5.1%** | +0.7% | +0.1% | +0.2% |

⇒ case1（前向小档）**−7.9%** 是 A2 的独立旁证（与真机 −6% 同量级同方向）；其余七条在 ±1% 噪声内，只有 case5（反向小档）+5.1% 但没有对应改动，判为平台侧噪声。**平台 `score` 仍为 0**（§13.3 停写分数），本轮全部收益都是"证据收益"。

---

## 17. ⛔ R5~R9 路线结案（负结果）：arch22 纯搬运路径上，能替 `PIPE_ALL` 的只有 `PIPE_ALL`

### 17.1 先把判据修对：计时档从来没比过数值

`test_mhc_expand_npu.cpp::run_case()` 的 `reps>1` 分支打完结行就 `return 0`，**一次数值比对都不做**。所以 §11.11/§14.2/§14.4 里所有 `prof`/`prof2` 的 `ALL PASS`，含义只是"400 次启动没有一个返回错误"。发现方式：把前向 barrier 整个删掉、也不加任何序（`none`）在 c0/c4 上仍然 "ALL PASS"。
修法：绩行照打，之后**继续走 memcpy + 全量比对**，判决行加 `timed=1` 标记。改完立刻得到分层清晰的判据（下表）。

### 17.2 第二个判据级缺陷：pkg 里的 kernel `.o` 会悄悄是旧的

`MhcExpand_<hash>.o` 的 `<hash>` 只覆盖 tiling/key，**不覆盖内核源码**；而运行期加载的是 `npu_debug/pkg`，`build_npu.sh` 过去完全不管这个目录（只有 `prof_matrix.sh` 按 `.so` 时间戳条件复制）。结果 optA 隔离树的 pkg 从 02:16 起就没再变过：**R5/R6/R7/R8 的全部读数执行的是同一份 02:16 二进制**，包括那次"未打补丁的对照也 `rc=124`"。
修法（三处，已推到 `code1`/`optA` 两棵树）：`build_npu.sh` §1.5 每次 `rm -rf pkg` 重建 + 断言 pkg `.o` 与 build_out 逐字节一致；`one_case.sh` 跑前硬校验并打印 `kernel_o=`，不一致 `exit 95`；`prof_matrix.sh` 同样 `exit 9`。
修完的对照：同一份已提交内核在 optA 从 `rc=124` 变 `rc=0 / per=0.0197ms`，与 code1 树的 0.0198ms 交叉一致 ⇒ 环境恢复可信，本轮结论才成立。

### 17.3 判别矩阵（c4 `fwd-fp16-medium` S=1024 D=4096 m=4，共 16,777,216 输出元素）

| 模态 | 序提供方 | mismatch | 判读 |
|---|---|---|---|
| `base` | 每任务 `PipeBarrier<PIPE_ALL>` | **0** | 采纳版，提交 3 的就是它 |
| `none` | 无 | 15,145,589（90%） | 阴性对照成立：c4 撞得开冒险窗口（c0 撞不开，§11.9 的"14 条全红"靠的是大形状） |
| `ev1` | 只有一对相邻 `SetFlag/WaitFlag<MTE2_MTE3>(2)` | 15,450,624（92%） | **≈ `none` ⇒ `MTE2_MTE3` 事件对在这条路径上不产生序** |
| `db2` | 上面那对 + 跨任务 `MTE3_MTE2` WAR token | 393,216（2.3%） | WAR 方向**有效**（错量掉 95.5%）；残错就是没被提供的 RAW 环 |
| `db3` | 同 `db2`，但 RAW 的 id 随 `rot` 轮转 | 393,216 | 排除"同一 id 两笔在飞被并掉"这一猜测 |
| `db4` | 同 `db2`，但两方向用**不相交** id（RAW 4/5、WAR 0/1） | 393,216 | 排除"两方向共用一张 flag 表互相吞" |
| `db2e`/`db3e` | 各自 + 整段收尾一道 `PIPE_ALL` | 391,680 / 393,216 | 排除"内核退休时 MTE3 未排空"是主因（收尾 barrier 不改变失败量） |
| `mb2` | 窄 barrier `PipeBarrier<PIPE_MTE2>()` + WAR token | 1,114,341（6.6%；c0 上 24,831/32,768=76%） | 窄 barrier **能编译**（修正 §11.10 V2 的"窄 barrier 必 trap"只适用于双参数式），但对纯搬运路径不提供所需序，比 `db*` 更差 |

错法形态也定了性：`db*` 的 `mismatch=393216 = 24 × 16384` 恰好是**整行**，首个错元素 `i=15335424 = 936×16384` 落在行首，`got` 是另一行的**合法值**、`nan_unwritten=0` ⇒ 读侧读到上一代 UB 内容（RAW），不是漏写、不是尾序、不是覆写。

### 17.4 裁定与推论

1. **`MTE3_MTE2`（写完再给 MTE2 用）事件在 arch22 上有效**；**`MTE2_MTE3`（搬完再给 MTE3 读）事件在这条纯搬运路径上无效** ⇒ 前向所需的正是后一环 ⇒ **手写事件对不能替掉 `PIPE_ALL`**。这同时解释了 §11.10 悬着的两件事：V1"事件配对实测不足"、V2"窄 barrier 运行期 trap"（V2 用的是双参数式；单参数 `PipeBarrier<PIPE_MTE2>` 不 trap 但也不管序）。
2. 因此 §14.9 量的"c4 前向 mte2 4.7 + mte3 8.4 ≈ task 12.8µs、理论地板 8.4µs（−34%）"在**当前 API 语义下拿不到**——不是没写对代码，是这条通路缺一个"MTE2 落地→MTE3 起步"的可编程序。**R5~R9 路线关闭**，`opt_a/apply_r{5..9}.py` 全部只作为负结果工装留存。
3. 还剩下的、**不依赖任何新序原语**的杠杆：**把 barrier 从"每块一道"摊成"每 B 块一道"**（批内先灌 B 块 UB，一道 `PIPE_ALL`，再吐出 B×m 个副本）。`PIPE_ALL` 本来就同时买断了 RAW 和 WAR ⇒ 批处理不改变任何序语义，安全性可证，代价按 B 相减。§11.11 量的 barrier 代价是 **+5.5%**，这是本轮唯一还挂着的可量化收益。
4. 方法论上被这一轮救回来的东西：**任何"PASS"先问它比没比数值；任何"新核"先问 pkg 里那份 `.o` 是不是刚编的。**两条都已经写进 §15 的坑表和脚本里的硬闸门。

---

## 18. ⭐ 第四次提交：R10 前向 barrier 批处理采纳（2026-09-21 04:24–05:25，真机 `02aeb`）

**动机**就是 §17.4 剩下的第 3 条：不换序原语，只把 `PIPE_ALL` 从"每块一道"摊成"每 B 块一道"。

### 18.1 采纳形态（改动全在前向：驱动 + UB 环深 + 两个编译期常量；`op_host` 只有两处注释变化；`tiling_h`/`tiling_key_h` 与提交 2/3 逐字节相同）

```cpp
template <uint32_t BS>                       // 批大小走模板参数，不走运行时变量
__aicore__ inline void ProcessForwardN() {
    constexpr uint32_t MASK = BS * 2 - 1;    // 环深 = 2B
    ...  FwdIn((r+u)&MASK, ...) → PipeBarrier<PIPE_ALL>() → FwdOut((r+u)&MASK, ...)
}
__aicore__ inline void ProcessForward() {    // ≥12KB/tile 才攒批，否则退回 BS=1
    if (FwdBigTile()) ProcessForwardN<FWD_BATCH>(); else ProcessForwardN<1>();
}
```

环必须是 **2B** 而不是 B：驱动的形状是"灌本批 → barrier → 吐本批"，下一批的 `FwdIn` 发在**它自己那道 barrier 之前** ⇒ 若环只有 B 块，本批的 MTE3 还在读、下一批的 MTE2 就覆写同一格（WAR）。开 2B 后这一对永远不撞，且"上一批的吐"与"本批的灌"第一次真正并行，而序的强度与逐块 barrier 逐条等价。

### 18.2 四轮交替 A/B（`prof_matrix.sh` `Task Duration` **剔首 mean**，µs；同一 sweep 内交替配对，跨 sweep 只比同向不比绝对值）

| sweep（时刻 / 形态） | 模态 | c0 fwd-small | c4 fwd-medium | c6 fwd-large |
|---|---|---|---|---|
| **1**（04:24，运行时批大小） | `base` | — | 13.3 / 13.3 | 1119.0 / 1118.4 |
| | `b1` B=1 R=2 | — | 13.0 / 13.0 | 1117.9 / 1118.1 |
| | `b2` B=2 R=4 | — | 13.4 / 13.4 | 1097.0 / 1092.1 |
| **2**（04:36，改编译期模板分派） | `base` | 2.9 / 3.0 | 13.6 / 13.4 | 1116.5 / 1121.8 |
| | `b1` | 2.9 / 2.9 | 13.6 / 13.7 | 1119.2 / 1120.4 |
| | `b1r4` B=1 R=4 | 3.0 / 3.0 | 13.7 / 13.6 | 1119.7 / 1121.2 |
| | `b2` | 2.8 / 2.8 | 13.8 / 13.3 | **1092.2 / 1090.1** |
| | `ad`（=b2+12KB 门） | 3.0 / 3.1 | 13.4 / 13.1 | 1091.9 / 1090.7 |
| **3**（04:48，四模态复现） | `base` | 3.0 / 2.9 | 13.1 / 13.6 | 1119.6 / 1118.2 |
| | `b2` | 2.8 / 2.8 | 13.3 / 13.1 | 1095.6 / 1089.2 |
| | `ad` | 3.0 / 2.9 | 13.4 / 13.9 | 1095.2 / 1091.6 |
| | `fin`（清理注释/死支后的候选） | 3.0 / 3.0 | 13.1 / 13.2 | **1090.5 / 1091.6** |
| **4**（05:11，提交面确认） | `base` | 3.0 / 3.0 | 13.5 / 13.6 | 1112.2 / 1118.2 |
| | `fin` | 2.8 / 2.8 | 12.8 / 12.7 | 1089.4 / 1086.0 |
| | `sub`（**交上去的那一份文件**） | 2.8 / 3.0 | 12.9 / 13.5 | **1089.5 / 1090.9** |

**c6 四轮均值对（剔首 mean）**：base 1118.7 / 1119.2 / 1118.9 / 1115.2；采纳形态 1094.6(`b2`@1) / 1091.3(`ad`@2) / 1091.1(`fin`@3) / 1089.0(`fin`+`sub`@4 四样本) ⇒ **−2.2% ~ −2.5%，四轮、两种代码形状、九个模态全部同向**，而同 sweep 内 `base` 自己两轮的散布只有 ±3µs（±0.3%）。小/中档没有任何一轮出现同向差异。

### 18.3 三条非显然结论（其中一条否证我自己上一轮的口头结论）

1. **收益 100% 归"少开 barrier"，与环深无关。**`b1r4`（只把环开到 4 块、barrier 仍每块一道）= 1120.5µs，与 `b1` 1119.8、`base` 1119.2 同在噪声里；`b2` 才掉到 1091。**多占的 2 块 UB 免费，省下的一半 barrier 值 2.5%** —— 这两个问题以前是搅在一起讨论的，本轮第一次被拆开。
2. ⛔ **"攒批在中档有 ~0.5µs 代价"这句不成立，是跨 sweep 漂移**。它的来源是 sweep1 `b1` c4=13.0 对 sweep2 `b1` c4=13.6，但同一对比里 `base` 自己从 13.3 漂到 13.5 ⇒ 同形状同代码的跨 sweep 漂移就有 ±0.3µs，与所谓"代价"同量级。按 sweep 内配对，`b2−base` 在 c4 = +0.1 / −0.2 / −0.1µs ⇒ **中档既无代价也无收益**。于是 `FWD_THRESH`（≥12KB 才攒批）买到的**不是 µs**，而是两件事：小 tile 不白占 2 块 UB，以及对 §14.3 那条"小档少开核"定则的**影响面严格为零**（小/中档走的是与提交 3 完全同一条 BS=1 代码路径）。
3. ⭐ **溯源可以闭合到机器码，不必停在"审过没改逻辑"**：`fin`(源 md5 `8810171e`) 与提交面 `sub`(`024335fb`，人工清掉过期注释和 discarded 死支) 的 kernel `.o` **集合 md5 同为 `363663f846bc`** ⇒ 同一份机器码，第四轮里两者 c6 差 2.8µs 恰好就是这份"同码"给出的噪声尺度。其后两处注释修正（把"核内同时只持有 1 份 D"改成"前向环同时握 4 块 = host `ub_size/4` 预算的上限"）复编后 `.o` 仍是 `363663f846bc`。**host 那个 `/4` 从此是带约束的数字**：放宽它 = 放弃前向攒批，两者必须一起改，这一点已写进两侧注释。

### 18.4 门禁（全部在**提交树** `~/ops_comp/code1` 上跑，不是在隔离树）

**溯源链**：`daf2b8ed…`（提交 3 的 kernel）+ 注入器 `npu_debug/opt_a/apply_r10.py@md5 c9652eb1` 的 `fin` 模态 → `8810171e…`（取读数的那一份）→ 人工清理过期注释与 discarded 死支 → `024335fb…` → §18.3 的约束注释修正 → **`038d65a0…`**（提交的 kernel）。`op_host` 只有注释变化：`c76d30be…` → **`3c2dadc8…`**。从 `8810171e` 起（`fin`→`sub`→`038d65a0`）kernel `.o` 集合 md5 **恒为 `363663f846bc + 9423e52876c3`**，与被门禁/被计时的每一份构建逐字节相同；`base`(`daf2b8ed`) 是另一份 `306619d0a29b + 4b433cc5…`。

- `run_gate.sh 1 1` @05:18:24，`kernel=038d65a08b71 host=3c2dadc8f2cd`：`all` 45 条 `=== ALL PASS | cases=45 fail=0 err=0 ===` + 8 条 prof2 全部 `timed=1 … mismatch=0 … PASS`（含 `fwd-fp16-large 0/469762048`、`bwd-fp16-large 0/58720256` 位级）⇒ `### gate FAIL=0`，日志 `npu_debug/logs/gate_r10_20260921_051824`。
- 合规 grep（`printf|fflush|fprintf|std::cout|TODO|FIXME|#if 0|getenv|[ABL-|MHC_SK`）对**四个提交文件**零命中（`cpu_debug/test_mhc_expand_cpu.cpp` 有 26 处 `printf`，它不在提交清单里）。
- dry-run 清单恰好 4 字段，`kernel_cpp` 12,187B/sha256 `ab8654dc…`、`host_cpp` 10,409B/`3b3fa433…`、`tiling_h` `be5f660b…`、`tiling_key_h` `b79c2cbd…`，**与本地四文件 sha256 逐项一致**；四文件本地/远端 md5 平。

### 18.5 第四次提交实测（`submission_id=6ab04e5fb0477ec41ed0b659`，05:25 出分）

**`状态: Pass`、8/8、每条 `precision_ratio: 1`。** 原始回执 `code1/npu_debug/logs/submit4_query_20260921.log`（远端=本地 md5 `661181b8…`）。

| case | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| sub3 | 3.52 | 17.78 | 1387.20 | 49.40 | 4.16 | 893.28 | 735.44 | 1039.64 |
| sub4 | 3.74 | 16.46 | 1382.86 | 47.68 | 4.02 | 891.02 | 739.62 | 1040.78 |
| Δ | +6.3% | −7.4% | −0.31% | −3.5% | −3.4% | −0.25% | +0.57% | +0.11% |

1. ⚠️ **真机的 −2.5% 没有在任何一条平台用例上出现**：四条 1ms 量级用例是平台最稳的一档（§14.7 跨提交散布 ±0.4%），本轮读数是 −0.31%~+0.57%，**没有 −2% 量级的位移**。
2. 同两轮里 case1（+6.3%）与 case2（−7.4%）都发生在**改动走不到的路径**上（tile <12KB ⇒ BS=1 支路，与 sub3 是同一份代码）⇒ 平台在几十 µs 档的分辨率本来就是 ±7% 级，本轮平台侧**不携带新信息**。
3. ⛔ **不再花提交位去追这个不一致**。判不动的原因很具体：平台 8 条用例的形状从未拿到（§12.2 只有对所有提交相同的 `best_time` 常数），而本地↔平台按量级配出来的比值互相矛盾（fwd-small 1.25×、fwd-medium 1.29×、fwd-large 1.27×、case4 要 2.06× 才能配上、case6/7 本地根本没有对应量级）⇒ 平台用的不是我方 prof2 那三档，**四条 ms 级用例里有没有"tile≥12KB 的前向"这件事在现有信息下判不了**。要判只能改门限重交试探，而 §13.3 计分冻结 ⇒ **试探的回报是 0**。真机才是本项目的性能判据，平台只负责判"过不过"（四次 8/8 已经买断了这一点）。

### 18.6 前向路线结案：这是带宽墙，不是 barrier 墙

- c6 每轮搬运 `1,056,964,608 B`（读 117.4MB + 写 939.5MB）。四轮 base 均值 **1118.0µs = 945 GB/s**（与 §11.11 独立测到的 946 GB/s 对得上）；采纳版四轮 8 样本均值 1089.0µs = **970 GB/s = 910B 官方 1024 GB/s 的 94.7%**。
- 逐管道同向：sweep4 同轮样本 `mte3` 1087.5 → 1061.9µs（−2.4%），与 task time 同幅 ⇒ 拿到的确实是 MTE3 侧等待的消失，不是新的并发度。
- **反向也已经贴墙**：本轮同一份二进制上 `bwd-fp16-large` 剔首 1087.1µs（未改动的路径，与 §11.11 的 1088.3 一致）= **972 GB/s = 94.9% 规格**。
- B=4 不做：收益按 1/B 收敛（B=2 拿到 −2.5%，B=4 上界再 −1.2%），而环 8 块需要把 host 的 `ub_size/4` 放宽到 `/8` ⇒ `dTileLen` 会被腰斩、A/B 不再干净。**在 94.7% 带宽贴上用一次"会破坏形状可比性"的改动换 ≤1.2%**，判不值。

⇒ **题 1 前向路线结案**（连同 §17 的事件对负结果、§14.9 的源冻结否定）：中大档两条路径都到 95% 规格，唯一残余缺口在 §14.8 判过 ROI 极低的小档固定开销。本轮之后**题 1 没有已知的、可量化的、不需要新硬件语义的待动项**。

### 18.7 现场清理与归档（2026-09-21 08:20–08:28，本地 + 真机 `02aeb`）

路线已收口，把历轮实验留下的散落文件清了一遍。**判据：删之前先证明"内容在别处存在"**（git 里、archive 目录里、或归档 tar 里做过逐文件 md5 往返比对）。

**本地（`code1/`）**

| 处置 | 文件（md5 前 8 位） | 依据 |
|---|---|---|
| 删除 | `cpu_debug/test_mhc_expand_cpu.cpp.bak_pre_ieee` 46681732、`npu_debug/mhc_expand.cpp.bak_pre_chunk` 53ee60e5、`npu_debug/mhc_expand_tiling.h.bak_pre_chunk` f759a052 | git 里逐字节存在（`git show e330077:code1/op_kernel/mhc_expand.cpp` = 53ee60e5；f759a052 更简单——它和**当前提交的 `op_kernel/mhc_expand_tiling.h` 是同一份**） |
| 删除（未跟踪的散落副本） | `op_kernel/mhc_expand.cpp.bak_pre_a` 53ee60e5、`op_kernel/mhc_expand.cpp.bak_pre_r10` daf2b8ed、`npu_debug/code1.md.bak_pre_reorg` c63cc337、5 个 `nul` | 三份都有正本：前两者 = git 里的 `e330077`/`bc2057b` 两版内核，`code1.md.bak_pre_reorg` = `40a383b^:code1.md`（重构前那份 1122 行版，实测同 md5）；`nul` 是 Windows 下 `>nul` 写错的产物 |
| 移入 `npu_debug/archive/`（加 `cpu_debug__`/`npu_debug__` 前缀保原路径） | `npu_debug__mhc_expand.cpp.bak_pre_abl` a16c371d、`cpu_debug__test_mhc_expand_cpu.cpp.bak_pre_mtile` 93ecb1e0、`npu_debug__test_mhc_expand_npu.cpp.bak_pre_ieee` c7a3e698、`npu_debug__test_mhc_expand_npu.cpp.bak_pre_prof2` 9491224d | git 里**没有**这四份（HEAD 的 code1 只跟踪 8 个 `.bak_*`，这四份的内容分别是被回滚的事件对内核、mtile 分支 harness、IEEE 溢出对拍、prof2 启动器）⇒ 唯一副本，只能搬不能删 |
| 移位（改名保留原语义） | `npu_debug/mhc_expand.cpp.bak_pre_fwd_event` 0fb9e6ec → **`op_kernel/mhc_expand.cpp.bak_pre_fwd_event`** | §562 的回滚命令写的就是 `op_kernel/` 路径，原来放在 `npu_debug/` 是历史错位；现在命令与磁盘一致 |

**真机 `02aeb`**（`~/ops_comp/`，清理前 196M+284M+127M+44K+56K，清理后 **code1 196M / optA 284M / logs 56K / archive 820K**）

- 新建 `~/ops_comp/archive/`，两份 tar.xz **并已拉回本地 `code1/npu_debug/archive/`（md5 双侧平：`62f059b7…` 139,548B、`0765f073…` 690,444B）**：
  - `probe1_sub3_20260921.tar.xz` = §14.9 的隔离探针树 `probe1`（源码 + `probe_dma` 注入器 + 118 份运行日志）与提交 3 的暂存树 `sub3_020552`；184 条目。**删前先做往返校验**：解出 `/tmp/arkchk` 后 `probe1/op_kernel/mhc_expand.cpp` `770b564a…`、`probe_dma/mk_probe.py` `a5c31d3d…`、`sub3_020552/op_kernel/mhc_expand.cpp` `daf2b8ed…` 三份与在树文件逐一相同。
  - `tmp_scratch_20260921.tar.xz` = `/tmp` 上 57 个本轮实验散落件（`*.log`、`fin/sub/sub2_kernel.cpp`、`host_base.cpp`），**删除清单直接由 tar 条目生成**（`tar tJf … | sed 's#^tmp/##' | xargs -r rm -f`），避免 glob 误伤别的会话同期写的 `/tmp/*.log`。
- 删除：`probe1`、`sub3_020552`；`code1/op_kernel/mhc_expand.cpp.bak_pass{53ee60e5,daf2b8ed}`；`optA/op_kernel/mhc_expand.cpp.bak_pre_{a1,r5,r7,r8,r9,r10}`（6 份，md5 只取到 53ee60e5 与 daf2b8ed 两个值，都在本地 git）。
- **保留（勿删）**：`optA/op_kernel/mhc_expand.cpp.bak_pre_r6`（`run_r10.sh`/`run_r7.sh`/`run_sweep.sh` 的 `BASE`，4 处引用）与自动生成的 `.bak_pre_r*` 名册（各 `apply_r*.py` 每次运行自写，删了不影响，留着不影响）；`/tmp/r10cache/`（A/B 构建缓存，按 `apply_r10.py` 的 md5 分目录，重跑四轮交替计时全靠它）；`/tmp/{cannbot,fpsync,node-compile-cache,opencode,vscode-typescript1000}`——**不是本会话的**（`fpsync` root 所有、9 月就在；`cannbot/locks` 08:18 还在被写），不动。
- 顺手做的唯一一件"抢救"：把 `probe1/npu_debug/prof/` 里**本地没有**的 7 个 `swF*_c0_22*` 前向 blockDim 扫描目录（546 文件）拉回 `npu_debug/prof/blockdim_20260920/`；核对后确认远端 114MB prof 的其余部分（`b2..b40`、`rule_*`、`sm_*`、`v0/v3`）与本地同名同尺寸 ⇒ 归档 tar **不含 prof**，省掉 114MB 重复品。
- **清理后的提交树健康复验**：`~/ops_comp/code1` 四文件 md5 `038d65a0 / 3c2dadc8 / f759a052 / 267e0125` 未变，kernel `.o` 集合仍是 §18.4 那一对 `363663f846bc + 9423e52876c3`，且 `build_out/` 与运行期 `npu_debug/pkg/` 两处取值一致。

**恢复口径**：本地被删项一律 `git show <commit>:<path> > <path>`（`e330077`→53ee60e5、`bc2057b`→daf2b8ed、`24356fc`→038d65a0）；远端 `probe1`/`sub3`/scratch 用 `tar xJf code1/npu_debug/archive/<name>.tar.xz -C <dir>`；本地 archive 里那 4 份 `__` 前缀文件按前缀还原回 `cpu_debug/` 或 `npu_debug/` 即可。

**本地两处"删了就没了"（按既有约定留在盘上、不入库）**：`npu_debug/prof/`（§14.2 blockDim 扫描的 msprof 原始产物，实测 26MB，现含从 `probe1` 抢救回的 7 个 `swF*` 共 546 文件）与 `op_kernel/mhc_expand.cpp.bak_pre_fwd_event`（V0 长期回滚点）。要重做性能 A/B 靠的是远端 `/tmp/r10cache`（构建缓存）+ `prof/`（重新解析读数的唯一原始出处），两者都别清。


---

## 附：节号使用说明

- **锚点不要重排**：`§10.0`（合规，AGENT.MD 引用）、`§14`（`op_host` 与 npu harness 的注释里写着"见 code1.md §14"）、`§11.8~§11.11` 及其 `.x` 子节（全篇互引）是**外部/内部引用锚点**。
- §11 缺 `11.1~11.7`、§13 缺 `13.4` 是**有意留空**（准备清单与已关闭的交接注记被删，编号保留以免引用漂移）。
- 通用坑一律指向 `算子开发工作流.md §6`；连接与环境指向 `连接信息.md` 与 `reference-devspace-environments`。本文件只留**题 1 专属**的事实与证据。
