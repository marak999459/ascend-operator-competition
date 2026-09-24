#!/usr/bin/env python3
# P122 静态关闭 ⇒ LOG 15.101 追加 + 索引行重指（锚点校验）
import sys, io
P = "code 3/probes/LOG.md"
src = io.open(P, encoding="utf-8").read()

ENTRY = """
### 15.101 🚫 P122 不发：候选 (c)「跨核 barrier」在平台执行路径上**根本不存在** ⇒ 静态关闭，并立 P123 = 依赖链计量器

- **改了什么**：**没动代码** —— 这一条的全部动作是读自己的核函数（`code 3/code/op_kernel/sparse_flash_attention.cpp`，`f815bf1eaba0f8`，1151 行，与榜上逐字节相同）。
- **当场数出来的三条事实**：① `grep -oE "HardEvent::[A-Za-z0-9_]+" | uniq -c` = `MTE2_V 6 / V_MTE2 6 / MTE3_V 12 / V_MTE3 12` = 36，**全是同一颗向量核内部 V↔MTE** ⇒ 事件层面本算子**没有任何跨核通道**（AIC 在 L174-177 `unitEnd_ = 0; return;`，连管线都不进）。② 全文件唯一的跨核结构是 `SyncAll()`(L268)，它在 `for (r = 1u; r < ks_; ++r)` 体内，而 L260 是 `if (ks_ < 2u) { return; }`；`ks_ ≥ 2` 只在 L171 那五个合取全真时成立 ⇒ **两道独立充分条件各杀一次**：平台不给 LSE 输出 ⇒ `lseOn_ = false`；`total0 = B·S1·nHeadBlk ≥ rows ≥ 41 > 40 = coreNum`（P108，6~9σ）⇒ 等式 `coreNum == ksh·total0` 恒假。③ 另两处 `PipeBarrier`（L768 `PIPE_ALL`、L789 `PIPE_V`）在 `MergeToken`(L729) 内 ⇒ 同一条死路径；**热路径上只剩 L1030 一道 `PipeBarrier<PIPE_V>`**（`SoftmaxPv` 第 3 段，每 chunk 1 次）。
- **判**：**(c) 关闭、P122 一发不花**（省一发预算，也省一次"发了才知道它不执行"的脸红）。⛔ 别把它读成"barrier 不重要"—— 它读出来的是**不存在**，与 P118 那种"存在但 ≈0"是两种强度不同的否定。
- 🔴 **方法学（比这条结论值钱，也解释了最后三把尺子为什么全是 ≈0）**：**复制族计量器在构造上读不到"同步/等待"**。多复制一道旗标或一道 barrier，若此刻流水里没有它在等的事，它就是**免费**的（P120 的 ≤0.019 %/对正是这个形状）；barrier 的价只在**删掉它**时显形，而删除永不逐位惰性（输出必变）⇒ 开销类里只有"带发射成本的那部分"（旗标对、标量 GM 读）可被这一族定价。⇒ §1.1 那句"≈71 % 无价"从今天起**不许再往"还有哪段活没定价"的方向读**，它的意思是"没有任何具名机制挂着了"。
- ⇒ **下一发 P123 = 依赖链计量器**（第一个不落在"复制工作量"上的判别器，与 P119 **只差一个变量**）：`SoftmaxPv` 第 5 段之后追加 N 遍 `Muls(o[i*rowC], o[i*rowC], 1.0f, rowC)`（i<nbCur）。惰性论证用本仓库自己 L339 那句"IEEE 下 `x*1.0f` 对**所有**浮点值逐位恒等"（L347/L360 已拿它做透传），且写的就是**真累加器** ⇒ 不需要 sink、也不可 CSE；关键是 `dst == src` ⇒ **N 份互相串行**，而 P119 的复制体写 `kfBuf_` 末行 ⇒ 可自由重叠。按吞吐账每 chunk 是 `nbCur×rowC` 个 fp32，而 PV 是 `m×nbCur×rowC` ⇒ **无依赖价 ≈ PV ÷ m ≈ 0.06 %** ⇒ 读数只要过 6 % 的门，成因**只能是链**。判据：Δ ≈ 0 ⇒ 链不是钳子，71 % 连"结构"这个候选一起没了 ⇒ 停止定价、转纯结构阅读；Δ ≫ 0 ⇒ 链是钳子，解法现成 = **双累加器交错**（每 chunk 自依赖深度 1 拆 2、指令数不变，代价是求和顺序变 ⇒ 回本地过 27 档超差门）。⛔ 照 `#15.87` 的规矩先本地四臂标定 Δ∝N 再发货。
- **文档**：§1.1 候选句、§4 尾巴、§5 #27、§6 两行（P122 划掉 + P123 开 P0）、AGENT.MD 第三题行全部就地改；⚠️ 顺带清掉两处过期断言（§6 那条"已读到 26 % ⇒ 剩 ≈74 %"的旧running total，与 §1.2"每行 token 数与 `S2` 未实测**是**走不走 HBM 的分界"—— 后者已被 P118 否证）。本轮**无发次** ⇒ 树与榜上逐字节相同，无回发义务。
"""

IDX_OLD = "**P122 = 跨核 barrier 计量器**（候选集最后一条；⛔ 按空闲 id 成对复制真握手，幅度必须先把 Σ 推到 ≥6 % 才够格发）（L4917）。 编号从 `15.101` 起。"
IDX_NEW = "**P122 不发**（(c) 跨核 barrier 静态关闭：旗标全同核、`SyncAll` 在 `ks_ ≥ 2` 里而平台 `ks = 1` ⇒ §5 #27）⇒ 下一发 **P123 = 依赖链计量器**（`Muls(o_i, o_i, 1.0f)` ×N，与 P119 只差依赖这一个变量）（L4928）。 编号从 `15.102` 起。"

n = 0
if "### 15.101" in src:
    print("entry already present, skip"); sys.exit(1)
assert src.endswith("\n")
if src.count(IDX_OLD) != 1:
    print("BAD index anchor count=%d" % src.count(IDX_OLD)); sys.exit(1)
src = src.replace(IDX_OLD, IDX_NEW)
src = src.rstrip("\n") + "\n" + ENTRY.rstrip("\n") + "\n"
print("ok")
if "--apply" in sys.argv:
    io.open(P, "w", encoding="utf-8").write(src)
    print("written")
