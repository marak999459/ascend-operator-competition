# `refs/` —— 可跑的参考资料（**不是提交代码**）

> 提交源只在 `code1/`、`code2/`、`code 3/code/`。本目录的东西**不参与构建、不会提交**。
> 它的用途只有两个：**参考实现**与**真机回归材料**。


**相关文档**：[AGENT.MD](../AGENT.MD)（入口） · [算子开发工作流.md](../算子开发工作流.md) · [official_problem_statement.md](../official_problem_statement.md)（官方题面） · 第三题 [code3.md](../code3.md) · 第二题 [code2.md](../code2.md)

---

## 1. 目录内容

### 1.1 `sfa/` —— 第三题 `sparse_flash_attention`

| 文件 | 用途 |
|---|---|
| `sfa_ref.py` | ⭐ Python 独立参考实现：生成用例（`gen_case` / `write_case`）+ 闭式解自检 |
| `test_sfa_real.cpp` | ⚠️ **重建版**真机对拍 harness（见 §3） |
| `run.sh` | 真机构建 + 跑 8 用例对拍（编译 `test_sfa_real.cpp`，输出 `PASS=/FAIL(`） |
| `build.sh` | 真机构建 + 组装 vendor（`CMAKE FAIL` / `MAKE FAIL` 显式报错） |
| `bench.cpp`、`gen_big.py` | 性能 bench 与大规模用例生成 |
| `c1_min.bin` … `c8_sbs2.bin`、`case_small.bin`、`mini.bin` | 回归用例数据（⚠️ 旧格式，见 §3） |
| `cases/` | 由 `gen_case.py` 生成的新格式用例（`run.sh` 优先吃这里） |
| `gen_case.py` | 用当前 `sfa_ref.py` 重新生成一套用例到 `cases/` |
| `verify_online_softmax.py`、`verify_by_pid.py`、`cmp_pid_dump.py`、`diag_c6.py` | 诊断脚本（在线 softmax 状态、按 PID 落盘、用例诊断） |
| `run_matrix.sh`、`build_test.sh` | 批量跑 / 双 tag 对比构建 |

### 1.2 `harness_sinkhorn/` —— 第二题 `mhc_sinkhorn`

| 文件 | 用途 |
|---|---|
| `mhc_sinkhorn_scalar.cpp` | ⭐ 当前 5/5 正确版 kernel（纯标量、无 `SyncAll`） |
| `sinkhorn_ref.py` | Python 参考实现（`gen` / `check` 子命令） |
| `test_mhc_sinkhorn.cpp` | C++ harness（aclnn 调用、fp16 解码、行和自检） |
| `run.sh` / `build_test.sh` | 一键构建 + 组装 vendor + 跑用例矩阵 |
| `mhc_sinkhorn_host_orig.cpp`、`mhc_sinkhorn_host.cpp` | host tiling 桩（6 字段版 / 带 `maxChunksPerCore` 版） |
| `mhc_sinkhorn_tiling_fixed.h` | 带 `maxChunksPerCore` 的 tiling 结构体 |
| `mhc_sinkhorn_fixed.cpp` | ⚠️ **中间版本**（统一轮数 + 保留 `SyncAll`、**未修写回截断**）—— 勿当终版 |
| `patch_trace.py`、`patch_trace2.py` | 给每个 `SyncAll` 前插打点，定位死锁位置 |

---

## 2. 怎么用（第三题）

```bash
# 1) 生成用例（用当前 sfa_ref.py，格式与重建 harness 一致）
python3 refs/sfa/gen_case.py

# 2) 推到真机后构建 + 对拍
bash ~/sfa_real/build.sh 2>&1 | grep -aE "CMAKE FAIL|MAKE FAIL|构建 OK"
bash ~/sfa_real/run.sh    2>&1 | grep -aE "PASS=|FAIL\("
```

> ⚠️ 跑之前必须 `source ~/Ascend/cann-9.0.0/set_env.sh`，且 `LD_LIBRARY_PATH` 要**追加**而不是覆盖（见 `code3.md` §4.4）。

---

## 3. ⚠️ 重要：`test_sfa_real.cpp` 是**重建版**，且旧 `.bin` 格式不匹配

### 3.1 为什么是重建版

原始 `test_sfa_real.cpp`（约 14.7KB）在 **2026-09-19 的目录整理中被误删**，无备份、无 Git 历史。
现文件是依据以下**仍然可靠的信息**重建的：

1. 用例文件格式 —— `sfa_ref.py` 的 `write_case()`
2. 用法 —— `run.sh`：`./test_sfa <case.bin>`，退出码 `0` = PASS
3. 输出格式 —— `run.sh` 用 `grep -aE "最大相对误差|超差元素|墙钟"` 提取
4. 输入契约与属性顺序 —— `official_problem_statement.md` 与 `code 3/code/op_host/sparse_flash_attention.cpp` 的 OpDef

⛔ **它没有在真机上编译/运行验证过。** 首次使用请先用一个已知用例确认能出 `PASS`，再信任其结论。

### 3.2 为什么旧 `.bin` 可能跑不通

实测：`refs/sfa/*.bin` 的文件头是 **`SFA_CASE 1`**，payload 比当前 `sfa_ref.py` 生成的（`SFA_CASE 2`）**少 16 字节**，差值来源未确定（各文件差值不随形状线性变化）。
→ 用旧 `.bin` 直接喂重建 harness，会因布局不符而读到错位数据。

**因此重建 harness 内建了精确字节数自校验**：

- 吻合 → 正常跑
- 不吻合 → 报 `[FAIL] payload 布局不吻合` 并提示重新生成用例，**不会静默给出错误结论**

**正确做法**：先 `python3 refs/sfa/gen_case.py` 生成新格式用例，再用新用例跑对拍。

### 3.3 如果你还留着原始的 `test_sfa_real.cpp`

真机上 `~/sfa_real/` 里可能仍有原文件（或已编译的 `test_sfa` 二进制）。
**若有，直接用它**，并把本重建版替换掉 —— 原版才是真机验证过的。

---

## 4. 怎么用（第二题）

```bash
bash refs/harness_sinkhorn/run.sh <kernel文件> [host文件] [tiling文件]
python3 refs/harness_sinkhorn/sinkhorn_ref.py gen   <batch> <n> <iters> <eps> <out.bin> <ref.bin> [seed]
python3 refs/harness_sinkhorn/sinkhorn_ref.py check <batch> <n> <iters> <eps> <in.bin> <ref.bin>
```

> ⚠️ 该 harness 固定走 **fp16**（`ACL_FLOAT16`），而官方题面要求**仅 FLOAT32**。做 fp32 复测需改 harness，见 `code2.md` §4.3。
