#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P6 第 1 号待钉项探针（任务 #23）：MIX 形态下 AIV 的 5~10× 到底是"索引墙"还是"并行度墙"。

用法: mk_probe_mixgeo.py <pol0|pol1|pol2>   —— 把打完探针补丁的 kernel 打到 stdout
⚠️ 产物**只推到远端副本** ~/sfa_real/code/...，本地提交源一个字节都不写（探针纪律 §15.14(f)）。

钉死的头文件事实（CANN 9.0.0，本安装）：
    basic_api/dav_c220/kernel_operator_sys_var_impl.h:55-68   GetBlockIdxImpl()
        AIV: return get_block_idx() * GetTaskRationImpl() + get_subblockid();   // 0 .. 2*BD-1
        AIC: return get_block_idx();                                            // 0 .. BD-1
    basic_api/kernel_operator_sys_var_intf_impl.h:53-61       GetBlockNum() = get_block_num()
        ⇒ **返回的是"组数" BD，AIC/AIV 同一个值**（`SyncAllImpl` 里写得很明白：
           `totalBlocks = isAIVOnly ? GetBlockNum() : GetTaskRationImpl() * GetBlockNum()`）
⇒ 于是提交版 kernel 的那三行
        coreNum = GetBlockNum();  coreIdx = GetBlockIdx();  if (coreIdx >= coreNum) return;
   在 MIX 下把**一半 AIV 块直接判死**（bi ≥ BD 全部空转），而且活下来的 bi=0..BD-1 只覆盖
   前 BD/2 个物理核（相邻 blockIdx 是**同一个 AI 核的两个向量单元）⇒ p1 的 16 单元落在
   8 个核上而不是 16 个。§15.22 那个"AIV 有效并行度暴跌 5~10×"里有多少是这个索引 bug，
   就是本探针要裁的。

三档（都 MIX 1AIC:2AIV + shim clearWorkspace，AIC 空转、AIV 跑真实算子）：
  pol0  原样（coreNum = GetBlockNum()）           ⇒ 复现 §15.22 的 5~10×，作为基准
  pol1  coreNum = ratio * GetBlockNum()           ⇒ 80 块全部做工（每单元恰好一次）
  pol2  同 pol1，但把 slot 折成 `sub*BD + group`  ⇒ 相邻单元落到**相邻物理核**，
        小点（p1 的 16 单元）不再挤在前 8 个核上
组数 BD 由 op_host 读环境变量 `SFA_BD` 决定（**同一个构建扫 BD**，不必每档重编）。
"""
import io
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'pol1'
ALL = ('pol0', 'pol1', 'pol2')
assert MODE in ALL, MODE

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code')
SRC = os.path.join(BASE, 'op_kernel', 'sparse_flash_attention.cpp')
s = io.open(SRC, encoding='utf-8').read()

KER_INC = '#include "kernel_operator.h"\n'
assert s.count(KER_INC) == 1, 'include anchor miss'
s = s.replace(KER_INC, KER_INC + (
    '\n// [MIXGEO] 入口桩在 MIX 形态下调 matmul::clearWorkspace(workspace)；本探针不用 KFC/Matmul，\n'
    '// 给空实现即可（§15.24：真清空间的框架路径在 arch22 上会把下一次 launch 毒化）。\n'
    'namespace matmul { __aicore__ inline void clearWorkspace(GM_ADDR) {} }\n'), 1)

A_ENTRY = '    REGISTER_TILING_DEFAULT(SparseFlashAttentionTilingData);'
assert s.count(A_ENTRY) == 1, 'entry anchor miss'
s = s.replace(A_ENTRY,
              '    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);   // [MIXGEO]\n' + A_ENTRY, 1)

# ---- 核间切分三档 ----
A_SPLIT = ('        const uint32_t total = B_ * S1_ * nHeadBlk_;\n'
           '        const uint32_t coreNum = GetBlockNum();\n'
           '        const uint32_t coreIdx = GetBlockIdx();\n')
assert s.count(A_SPLIT) == 1, 'split anchor miss'
NEW = {
    'pol0': A_SPLIT,   # 原样 = 基准
    'pol1': (
        '        const uint32_t total = B_ * S1_ * nHeadBlk_;\n'
        '        const uint32_t grp = static_cast<uint32_t>(GetBlockNum());\n'
        '        const uint32_t rat = static_cast<uint32_t>(GetTaskRatio());   // AIV in MIX = 2\n'
        '        const uint32_t coreNum = grp * rat;\n'
        '        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());\n'),
    'pol2': (
        '        const uint32_t total = B_ * S1_ * nHeadBlk_;\n'
        '        const uint32_t grp = static_cast<uint32_t>(GetBlockNum());\n'
        '        const uint32_t rat = static_cast<uint32_t>(GetTaskRatio());   // AIV in MIX = 2\n'
        '        const uint32_t coreNum = grp * rat;\n'
        '        const uint32_t raw = static_cast<uint32_t>(GetBlockIdx());    // = group*rat + sub\n'
        '        // 折成 sub*BD + group：相邻单元 ⇒ 相邻**物理核**（raw 的相邻 blockIdx 同核）\n'
        '        const uint32_t coreIdx = (raw % rat) * grp + (raw / rat);\n'),
}[MODE]
s = s.replace(A_SPLIT, NEW, 1)

sys.stdout.write(s)
