#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P19-M0：把 P18 的提交版 kernel 打进 MIX 形态，验"AIV 算法在 MIX 构建上零退化"。

用法: mk_probe_mix.py <mxa|mxb>   —— 把打完探针补丁的 kernel 打到 stdout
⚠️ 产物**只推到远端副本** ~/sfa_real/code/...，本地提交源一个字节都不写（探针纪律 §15.14(f)）。

两档共同点（全部来自 §15.26/§15.27 已钉死的事实，不重新发明）：
  1) 入口 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` + `matmul::clearWorkspace` 空桩
     （§15.23/§15.24：官方 adv_api 路径在 arch22 第 2 次 launch 挂，workspace 保持 0）；
  2) 核间切分索引修正：MIX 下 `GetBlockNum()` = **组数 BD**，AIV 的 `GetBlockIdx()` =
     `group*2 + sub ∈ [0, 2·BD)` ⇒ `coreNum` 必须是 `2·BD`（§15.26(a) 推导 + §15.27 真机复验，
     代价实测 1.96×）。ratio 这里**写死 2**而不是调 `GetTaskRatio()`：AIC 侧那个函数给的是
     自己那份（=1），会让两档核数算子在同一份 tiling 下不一致 ⇒ mxb 的屏障会差一块。

两档的差别只有一条 —— **P11v2 的归并屏障用哪一种**：
  mxa  保留 `SyncAll()`（默认模板参 `isAIVOnly=true`，`dav_c220/kernel_operator_sync_impl.h:328-331`
       = `ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0, SYNC_AIV_ONLY_ALL))` + `wait_flag_dev`）
       ⇒ **纯硬件 AIV 屏障，一个字节内存都不碰**（§15.25 判死的是 `SoftSyncAllImpl` 那条要
          gmWorkspace 的软实现，不是这条），但 AIC 块**绝不能**调它 ⇒ AIC 分支置 `ks_=1` 让它
          从 `Process()` 的 `if (ks_ < 2u) return;` 处退出。
  mxb  换成 `SyncAll<false>()`（`…:334-345`：AIC 等 AIV→自己广播→AIV 等 AIC_AIV）
       ⇒ 框架的"AIC+AIV 全栅"协议，AIC **必须**一起调 ⇒ AIC 分支不动 ks_。

要裁的问题（M0 的验收口径）：
  ① MIX 构建下现有 AIV 算法（P18：奇偶交错分片 + 输出张量归并）fp16+fp32 双遍 PASS=8 +
     golden 逐位一致，且**反复 launch 不被毒化**（harness 每次跑 1+3 次再比对，第二次挂就是 §15.24 的形态）；
  ② 计时不退化：`SFA_MIXBD=1` 把 host 的 `SetBlockDim` 折成组数（`BD = ceil(AIV块/2)`），
     与 base（AIV-only）同场次对比 p1/p2/p4/p6/big1。
"""
import io
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'mxa'
ALL = ('mxa', 'mxb')
assert MODE in ALL, MODE

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code')
SRC = os.path.join(BASE, 'op_kernel', 'sparse_flash_attention.cpp')
s = io.open(SRC, encoding='utf-8').read()

# ---- 1) MIX 入口 + clearWorkspace 空桩 ----
KER_INC = '#include "kernel_operator.h"\n'
assert s.count(KER_INC) == 1, 'include anchor miss'
s = s.replace(KER_INC, KER_INC + (
    '\n// [MIXM0] MIX 入口桩会调 matmul::clearWorkspace(workspace)；本探针不用 KFC/Matmul，\n'
    '// 给空实现即可（§15.24：真清空间的框架路径在 arch22 上会把下一次 launch 毒化）。\n'
    'namespace matmul { __aicore__ inline void clearWorkspace(GM_ADDR) {} }\n'), 1)

A_ENTRY = '    REGISTER_TILING_DEFAULT(SparseFlashAttentionTilingData);'
assert s.count(A_ENTRY) == 1, 'entry anchor miss'
s = s.replace(A_ENTRY,
              '    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);   // [MIXM0]\n' + A_ENTRY, 1)

# ---- 2) 索引修正：coreNum = 组数 × ratio ----
A_IDX = ('        const uint32_t total0  = B_ * S1_ * nHeadBlk_;\n'
         '        const uint32_t coreNum = GetBlockNum();\n'
         '        const uint32_t coreIdx = GetBlockIdx();\n')
assert s.count(A_IDX) == 1, 'idx anchor miss'
s = s.replace(A_IDX, (
    '        const uint32_t total0  = B_ * S1_ * nHeadBlk_;\n'
    '        // [MIXM0] MIX：GetBlockNum() = 组数 BD，AIV 的 GetBlockIdx() = group*2+sub ⇒ 块数 = 2·BD\n'
    '        const uint32_t grp = static_cast<uint32_t>(GetBlockNum());\n'
    '        const uint32_t rat = 2u;   // KERNEL_TYPE_MIX_AIC_1_2 ⇒ 每组 2 个 AIV（不查 GetTaskRatio：\n'
    '                                   //  AIC 侧它返回 1，两档核数会在同一份 tiling 下算歪）\n'
    '        const uint32_t coreNum = grp * rat;\n'
    '        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());\n'), 1)

if MODE == 'mxa':
    # AIC 不参与 AIV-only 屏障：把它的 ks_ 压回 1，Process 在 SyncAll 之前就返回
    A_AIC = ('        if ASCEND_IS_AIC {\n'
             '            unitBegin_ = 0; unitEnd_ = 0; unitStep_ = 1;\n'
             '            return;\n')
    assert s.count(A_AIC) == 1, 'aic anchor miss'
    s = s.replace(A_AIC,
                  A_AIC.replace('unitStep_ = 1;\n',
                                'unitStep_ = 1;\n            ks_ = 1u;   // [MIXM0] 别让 AIC 走到 SyncAll<true>\n'), 1)
else:
    A_SYNC = '        SyncAll();\n'
    assert s.count(A_SYNC) == 1, 'sync anchor miss'
    s = s.replace(A_SYNC, '        SyncAll<false>();   // [MIXM0] 全栅协议：AIC 也必须调一次\n', 1)

sys.stdout.write(s)
