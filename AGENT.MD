# AGENT 入口（**先读这个**）

本仓库是**昇腾 Ascend C 算子开发比赛**，共 3 道题，通过 SSH 隧道连一台 910B3 NPU 机器。

---

## 1. 阅读顺序

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | **`AGENT.MD`**（本文件） | 入口与硬约束 |
| 2 | [算子开发工作流.md](算子开发工作流.md) | **通用**三步流程 + 通用坑 + 通用教训（跨三道题） |
| 3 | [连接信息.md](连接信息.md) | 连接参数与凭据**去哪读**（不硬编码任何值） |
| 4 | [code1.md](code1.md) | 第一题 `mhc_expand`：状态 / 已排除 / 下一步证据 |
| 5 | [code2.md](code2.md) | 第二题 `mhc_sinkhorn`：同上 |
| 6 | [code3.md](code3.md) | 第三题 `sparse_flash_attention`：同上 |

**参考资料**（按需查，不是必读）：

| 文件 | 何时看 |
|---|---|
| [official_problem_statement.md](official_problem_statement.md) | 需要**官方题面原文**时（权威口径；第一题 / 第二题 / 第三题三章） |
| [refs/README.md](refs/README.md) | 要**跑真机回归**、或想知道 `refs/` 里有什么时 |
| [refs/sfa/sfa_ref.py](refs/sfa/sfa_ref.py) | 需要**第三题参考实现 / 重新生成用例**时 |
| [refs/harness_sinkhorn/](refs/harness_sinkhorn) | 需要**第二题测试脚手架**时 |
| `ops-transformer-master/` | ⭐ 需要**官方开源对标实现 / 性能手法**时（CANN 官方算子库，三题的开源参考池）。**只读参考，不是提交源**；逐题可用性见各 `codeN.md §10`，合规条款见 `code1.md §10.0` |

> 📌 **分工**：`算子开发工作流.md` 讲"**怎么做**"，`codeN.md` 讲"**这题现在到哪了、什么已被排除、下一步要什么证据**"。
> `codeN.md` 随题目推进**持续更新**；根目录通用文档很少改动。
> 需要哪一题的细节就只读那一个 `codeN.md`，**不要全读**（省 token）。

---

## 2. ⛔ 硬约束

0. **统一术语（用户 2026-09-20 规定，全仓库文档一律照此称呼）**：
   - **仿真机** = CPU 仿真环境（云端 devenv 容器）——⛔ **已停用（2026-09-22 用户决定：一直有真机，不再跑仿真）**；本地 VMware 那台当年也跑过仿真，现叫**通道机**（见下）
   - **通道机** = 本地 VMware Ubuntu VM，专职两件事：**第二账号的 devspace 登录位**（homedir 隔离，见 `连接信息.md` §2.3）+ **第二工位 Qoder**
   - **真机** = NPU 环境（910B3，SSH 隧道进入）
   - **比赛平台** = 代码提交/评测网站（CANNJudge）；旧文档中的"网站 / 比赛网站 / 提交平台"均指此
   - 🕘 **历史行文**：`code1.md` / `code2.md` / `code3.md` / `README.md` 里出现的"本地仿真机 / 云端仿真机"是当年仿真时代的记录，**不回改**；**2026-09-22 起不再有任何仿真环节**（用户决定一直有真机），验证一律直连真机
1. **三道题严格分家**，绝不跨界：
   - 第一题 `mhc_expand` → 提交代码在 `code1/`
   - 第二题 `mhc_sinkhorn` → `code2/`
   - 第三题 `sparse_flash_attention` → `code 3/code/`（文档在 `code 3/`）
2. **提交里绝不能有任何调试输出**：`printf` / `fflush` / `fprintf` / `std::cout` 一律禁止，`TODO` / `FIXME` / `#if 0` / `调试` 字样也要清掉。否则比赛平台判**"提交代码中存在不合规内容"**。
3. **隧道一律先用 `devspace_tunnel.ps1` 自举，不要一上来就找用户**（2026-09-20 更新）：`.ssh/config` 的 Host 别名自带 `Match !exec` 自举，脚本封装了"拉起 + 等端口 + ssh 验证 + `bootstrap.log` 判读"。用法：**"连真机" → `-Role npu`**（CPU 仿真环境已停用，不再有 `-Role cpu` 的候选）、只看状态 `-List`。环境表（短名 → 角色 / 可否派活 / 备注）在脚本顶部 `$EnvTable`，**加新环境只在那里补一行**；`连接信息.md` §2 是同一张表的文档口径。
   - ⭐ **派活闸门**：`-Role` 只会选中 `use='free'` 的环境；`use='reserved'`（正被别的会话当通道）和**未登记的新环境**会被跳过并打印 `[门]`。⛔ **Agent 不得为了"有环境用"而用 `-Env` 点名越过闸门** —— 被挡下就等于没环境，回去告诉用户缺什么，让用户决定（改表 / 开新环境 / 授权占用）。
   - 用户直接点名环境（"用 e6z6k"）时按 `-Env <短名>` 执行，那是用户的决定，不是越闸。
   - ⛔ 脚本报 `development environment no longer exists` ⇒ **环境本体已被回收，只有用户能在 IDE / 网页重新 Start**，此时停下来找用户；**不得重试轰炸**。真机连上之后**跑什么**仍按 §2.4/§2.5 的纪律先问用户。
4. **改代码前先说清"要改什么、为什么、怎么验"**（2026-09-21 修订，原为"不要自动改代码"）：三题均已进入"改代码 → 上机验证"阶段，Agent **可以**改提交源，但**不得静默改**——动手前给出改动点与验收判据，改完立刻按 `codeN.md` 的口径留证据（md5 基线 + 真机结果；仿真已停用，不再有仿真证据），破坏性或不可逆操作仍**必须先后请示**。各题自己的约束表（如 `code1.md §0`）优先于本条。
5. **不要连续提交**：每次提交后等用户确认；每次提交前先 `--dry-run`。
6. **密钥与私钥路径不写死在文档里**，一律引用 `连接信息.md` 的相对路径。
7. **git 主干在 Windows**（用户 2026-09-20 定）：提交/推送只在这台机器上做；**通道机（VM）里的那份是只读工位副本**，只 pull 不 push（VM 侧 `origin` 的 push URL 已被停用，恢复方法见 `连接信息.md` §2.3）。多路会话并行改文档时仍按"只 add 自己那摊"。

---

## 3. 目录导航

```
.
├─ AGENT.MD                  ← 本文件（入口）
├─ 算子开发工作流.md          ← 通用流程与教训
├─ 连接信息.md                ← 连接参数 / 凭据 的相对路径说明
├─ code1.md / code2.md / code3.md   ← 各题状态（随推进更新）
│
├─ code1/                    ← 第一题 mhc_expand（提交代码）
├─ code2/                    ← 第二题 mhc_sinkhorn（提交代码）
├─ code 3/                   ← 第三题 SFA：code/ 是提交实现，其余是工程文档
│
├─ 密钥.txt / public.pem / encrypt_password.py   ← 比赛平台凭据与加密脚本
├─ devspace_tunnel.ps1       ← ⭐ 隧道自举 + 派活闸门：-Role cpu|npu / -Env <短名> / -List / -Diag / -Watch；环境表 = 顶部 $EnvTable
│                                 角色表与环境状态见 连接信息.md §2
├─ official_problem_statement.md   ← ⭐ 三题官方题面全文（权威口径）
├─ ops-transformer-master/   ← ⭐ CANN 官方**开源**算子库（三题的开源参考池，**只读、非提交源**）
│                               逐题可用性见各 `codeN.md §10`；合规条款见 `code1.md §10.0`
│                               ⚠️ **不入库**（已 gitignore）：clone 本仓库后需自行拉 `gitcode.com/cann/ops-transformer`；本地这份 = v9.2.0，文档里的行号会随版本漂移，查不到就 grep 符号名
└─ refs/                     ← 可跑的引用材料（不是提交代码）
   ├─ README.md              ← ⭐ refs 说明 + 用法 + 重建告诫
   ├─ sfa/                   ← 第三题：参考实现 + 真机回归用例 + 诊断脚本
   │                            ⭐ sfa_ref.py（Python 参考）、run.sh（构建+对拍）
   │                            ⚠️ test_sfa_real.cpp 是**重建版**（原文件误删，见 refs/README.md §3）
   │                            ⭐ gen_case.py + cases/（用当前格式生成用例）
   └─ harness_sinkhorn/      ← 第二题 mhc_sinkhorn 的本地测试脚手架
                                ⭐ mhc_sinkhorn_scalar.cpp、sinkhorn_ref.py、run.sh
```

> ⚠️ `refs/`、`official_problem_statement.md` 与 `ops-transformer-master/` 是**参考资料**，不参与构建、**不是提交源**。
> 提交源只在 `code1/`、`code2/`、`code 3/code/`。
> `ops-transformer-master/` 受 **CANN OSL v2.0** 约束（开源、可参考可改，**不得移除版权头**）；**参考为主、不整段照抄**，只借部分细节（用户 2026-09-20 定调）。

### 3.1 ⚠️ 目录归属提醒

整理后，**各题目录已只含自己的东西**：

| 目录 | 现在含有什么 |
|---|---|
| `code1/` | 第一题提交源（`op_host/`、`op_kernel/`、`CMakeLists.txt`）+ `tools/reference.py` |
| `code2/` | 第二题提交源（`op_host/`、`op_kernel/`、`CMakeLists.txt`） |
| `code 3/` | 第三题：`code/`（提交实现）+ `submit/`（提交包）+ 构建脚本 |

> 原先混在 `code1/` 下的第二、三题材料**已归入 `refs/`**，其余历史快照与冗余副本**已删除**。

---

## 4. 构建产物噪音

`build/`、`build_out/`、`__pycache__/` 等目录下有大量构建产物，会让 `glob` / 搜索很吵 —— 搜索时**主动排除**这些目录。

`ops-transformer-master/` 同理：它是**上万文件的外部开源库**，搜"本题实现在哪"时**必须排除**，否则会淹掉 `code1/`、`code2/`、`code 3/` 的命中。只有**明确要查开源对标实现**时才定向进它（入口见各 `codeN.md §10`，别从头 grep）。

`code1/npu_debug/prof/` 同理（**已在磁盘上、但已 gitignore 不入库**）：msprof 每次运行产上百个 `.db`/`.done`/`.complete`/`.csv` 中间件，本地这份 900+ 文件 26MB。数值结论已全部摘录进 `code1.md` 的表格（§14/§18/§19），**要复现就在真机重跑 `prof_npu.sh`**，别去仓库里找原始档。

---

## 5. 当前进展（一句话版）

> 📌 **每题的唯一权威状态 = 各自的 `codeN.md §1`**（状态 / 已排除 / 下一步证据都在那）。本表只放"一眼看去在哪"的一行指针，**不复制题级细节**；两边冲突时以 `codeN.md` 为准。最后同步：2026-09-22（题 1 已到 `§26`：p20 第 1.5/1.5b 阶真机读数落档）。

| 题 | 一句话状态 | 权威状态 |
|---|---|---|
| 第一题 `mhc_expand` | 正确性完全收口（平台十二次提交均 8/8，在榜 **33 / 75.50**）；性能收到"大档 + 中间档"；§24 的"c5 固定开销分解"p20 第 1 阶**已跑完并判负**（`§25`），第 1.5 阶把那个常数换成**两条**：反向 `每行 = 145ns + 107ns×m`、**字节项无符号**（`§26.2`，29 格 RMS 0.081µs）⇒ 当前候选 = **H5 反向合批**，价签已由 `h5_price.js` 钉成 `T = A + B·blk + f·(145+107m)·S/blk`：**过 1.6 分门槛 ⇔ 行项残余 `f ≤ 20%`（本地 ΔT ≥ 0.96µs）**，"只摊一半"那两档 0.79~1.14 分不值得动提交树（`§26.5/§26.8`，**代码未写**） | [code1.md](code1.md) §1 |
| 第二题 `mhc_sinkhorn` | 正确性 + 性能双双收口（平台向量版连续三次 5/5，step-7 在榜）；四条性能路线全部有结论 | [code2.md](code2.md) §1 |
| 第三题 `sparse_flash_attention` | 比赛平台 6/6 全 Pass；剩真机复验 + 性能优化（方案以 `code3.md §14.5` 为准） | [code3.md](code3.md) §1 |

> ⚠️ 题级细节一律进对应 `codeN.md §1` 看，不要在本文件里找；历史存档（非权威、仅证据）在 `code1/archive/` 与 `code2/archive/`。
