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

> 📌 **分工**：`算子开发工作流.md` 讲"**怎么做**"，`codeN.md` 讲"**这题现在到哪了、什么已被排除、下一步要什么证据**"。
> `codeN.md` 随题目推进**持续更新**；根目录通用文档很少改动。
> 需要哪一题的细节就只读那一个 `codeN.md`，**不要全读**（省 token）。

---

## 2. ⛔ 硬约束

1. **三道题严格分家**，绝不跨界：
   - 第一题 `mhc_expand` → 提交代码在 `code1/`
   - 第二题 `mhc_sinkhorn` → `code2/`
   - 第三题 `sparse_flash_attention` → `code 3/code/`（文档在 `code 3/`）
2. **提交里绝不能有任何调试输出**：`printf` / `fflush` / `fprintf` / `std::cout` 一律禁止，`TODO` / `FIXME` / `#if 0` / `调试` 字样也要清掉。否则平台判**"提交代码中存在不合规内容"**。
3. **NPU 隧道需人工建立，端口会变**：连不上（`Connection refused`）**直接问用户，不要重试**；`Test-NetConnection` 探不到，别用它判断。详见 `连接信息.md` §2。
4. **不要自动改代码**：Agent 只负责提交与报告，改代码由用户决定。
5. **不要连续提交**：每次提交后等用户确认；每次提交前先 `--dry-run`。
6. **密钥与私钥路径不写死在文档里**，一律引用 `连接信息.md` 的相对路径。

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
├─ 密钥.txt / public.pem / encrypt_password.py   ← 提交平台凭据与加密脚本
├─ official_problem_statement.md   ← ⭐ 三题官方题面全文（权威口径）
└─ refs/                     ← 可跑的引用材料（不是提交代码）
   ├─ README.md              ← ⭐ refs 说明 + 用法 + 重建告诫
   ├─ sfa/                   ← 第三题：参考实现 + 真机回归用例 + 诊断脚本
   │                            ⭐ sfa_ref.py（Python 参考）、run.sh（构建+对拍）
   │                            ⚠️ test_sfa_real.cpp 是**重建版**（原文件误删，见 refs/README.md §3）
   │                            ⭐ gen_case.py + cases/（用当前格式生成用例）
   └─ harness_sinkhorn/      ← 第二题 mhc_sinkhorn 的本地测试脚手架
                                ⭐ mhc_sinkhorn_scalar.cpp、sinkhorn_ref.py、run.sh
```

> ⚠️ `refs/` 与 `official_problem_statement.md` 是**参考资料**，不参与构建、**不是提交源**。
> 提交源只在 `code1/`、`code2/`、`code 3/code/`。

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

---

## 5. 当前进展（一句话版）

| 题 | 状态 |
|---|---|
| 第一题 `mhc_expand` | 代码看似完成，**真机从未验证** —— 唯一未知项，性价比最高 |
| 第二题 `mhc_sinkhorn` | 真机 **5/5 通过**，性能 303µs → 目标 ~1.5µs（病根 = 指令数） |
| 第三题 `sparse_flash_attention` | 真机自测全过，**平台仍 Wrong Answer**；bf16 假设**已被实验 0 排除**，当前最优先假设 = 输出差常数倍缩放 |

> 细节与证据见各自的 `codeN.md`。
