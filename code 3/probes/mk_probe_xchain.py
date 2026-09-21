#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 **只给远端副本用** 的 P11v2 前置探针（任务 #26）：AIV→AIV 的跨核**数据**通道。

为什么还要再探一次（§15.30(j) 不是已经通了吗）：
  · `xcoremm3` 证的是 **MIX 下 AIC→AIV**，用 **FFTS 旗标**做握手；
  · P11v2 要的是 **AIV-only 下 AIV→AIV**，握手换成 **`SyncAll`**（§15.17(d) 已实测 1/16/40
    块都能过 barrier，代价 ≈2.5~3.3 µs/launch），暂存换成**算子自己的输出张量**
    （§15.21/§15.24 已钉死 `workspace` 形参在 AIV-only 下是常量 `0x1000000`，不可用）。
  ⇒ 三件事同时成立才有 P11v2：① 屏障在 AIV-only + 40 块下真的对齐；② **别人写的字节我读得回来**；
    ③ 两条发布路径（V 流水标量写 / MTE3 `DataCopy`）各自能不能被屏障兜住。

探针出口布局（big1：nlse = B*S1*N1 = 1024 个 float，两条出口各 1024）：
  sumGm_[c]            = 1000 + c              —— 标量发布（V 流水 SetValue）
  sumGm_[n + c]        = 1 + 读回(邻居 p) - 期望 —— **1 ⇒ 通；0 ⇒ 这块没写；其它 ⇒ 错**
  sumGm_[80 + c*16+j]  = 从 maxGm_[p*16+j] 中继回来的值 —— **等于 p + j*0.001 ⇒ 批量通道通**
  maxGm_[c*16 + j]     = c + j*0.001           —— 批量发布（MTE3 DataCopy，32B 对齐）

模式：
  both     标量 + 批量，带 SyncAll（默认）
  nosync   与 both 逐字相同但**去掉 SyncAll** ⇒ 阴性对照：若它也"全 0/全对"，说明读数根本
           不依赖屏障，本次探针无意义；正常应当大面积错。
  scalar   只跑标量那半（隔离批量路径）
"""
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'both'
assert MODE in ('both', 'nosync', 'scalar'), 'unknown mode: %s' % MODE

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code',
                   'op_kernel', 'sparse_flash_attention.cpp')
src = open(SRC, encoding='utf-8').read()

PROBE = r'''
    // ==================== [XCHAIN] 远端探针专用（绝不进提交源） ====================
    // AIV 块 c 发布 -> 全局屏障 -> 读邻居 p=(c+1)%n，两条路径分别裁定。探针跑完直接 return，
    // 不碰真实计算（输出必然是错的，本档只看 [CUBE] CHAIN 那一屏）。
    __aicore__ inline void XChainProbe()
    {
        const uint32_t n = GetBlockNum();
        const uint32_t c = GetBlockIdx();
        if (n == 0u || n > 40u || c >= n) { return; }
        const uint32_t p = (c + 1u) % n;
        // ⚠️ 真机实测（本探针第一版就挂在这条上）：ccec 在 aicore 函数里禁止
        //    **无符号变量**与 float 之间的转换（"cast between floating and unsigned integer
        //    variable is not allowed"）。两条合法写法：① 转 intrinsic 的返回值 ——
        //    `GetBlockIdx()` 声明是 int64_t（`kernel_operator_sys_var_intf.h:37`），有符号→float
        //    不触发这条禁令（干净源 :122 就是这么写的）；② 根本不做转换，用 float 加法推。
        //    ⇒ 邻居的期望值走 ②：自己的 float +1.0f，唯独回绕到 0 的那一块特判。
        const float cf = static_cast<float>(GetBlockIdx());
        float pf = cf + 1.0f;
        if (p == 0u) { pf = 0.0f; }
        // ---- 1) 发布 ----
        // (a) V 流水标量写：sumGm_[c] = 1000 + c
        sumGm_.SetValue(c, 1000.0f + cf);
        // (b) MTE3 批量写：maxGm_[c*16 + j] = c + j*0.001（16 个 float = 64 B，槽起点 64B 对齐）
        //     同理不用 cast：j 的 float 值由累加器给。
        LocalTensor<float> ub = kfBuf_.Get<float>();
        float fj = 0.0f;
        for (uint32_t j = 0; j < 16u; ++j) {
            ub.SetValue(j, cf + fj);
            fj += 0.001f;
        }
        SetFlag<HardEvent::V_MTE3>(4);              // UB 标量写 -> MTE3 搬出之前必须自己收口
        WaitFlag<HardEvent::V_MTE3>(4);
        DataCopy(maxGm_[static_cast<size_t>(c) * 16u], ub, DataCopyParams{1, 2, 0, 0});
        PipeBarrier<PIPE_ALL>();                     // 先把本块的标量写/搬出都推出去，再进屏障
        @BARR@
        // ---- 2) 读邻居的货 ----
        // ⚠️ 读回值 +1.0f 偏置：出口缓冲每次 launch 前被 memset 成 0，而"通道通"的偏差恰好也是
        //    0 ⇒ 不加偏置就分不开"写了 0"和"这块根本没执行到这一句"（本轮第一跑就撞上了：
        //    pubN=2 却有 37 个读回槽是 0，两个解释都能自圆其说）。加了偏置之后
        //    1.0=通、0=没写、其余=错，三态可分。
        const float got = sumGm_.GetValue(p);
        sumGm_.SetValue(n + c, 1.0f + got - (1000.0f + pf));   // 1 = 标量通道通
        @BULK@
    }

'''

BULK = r'''        // 中继：邻居的批量发布 -> UB -> 自己的 sumGm_ 槽（不与任何人的发布区重叠）
        DataCopy(ub, maxGm_[static_cast<size_t>(p) * 16u], DataCopyParams{1, 2, 0, 0});
        SetFlag<HardEvent::MTE2_MTE3>(3);
        WaitFlag<HardEvent::MTE2_MTE3>(3);
        DataCopy(sumGm_[80u + static_cast<size_t>(c) * 16u], ub, DataCopyParams{1, 2, 0, 0});
'''

BARR = '\n        SyncAll();   // XCHAIN barrier\n'

body = PROBE.replace('@BARR@', '' if MODE == 'nosync' else BARR.strip())
body = body.replace('@BULK@', '' if MODE == 'scalar' else BULK.rstrip())

# 自检 0：ccec 的 aicore 禁令 —— float 不得与**无符号变量**互转（本探针第一版就死在这）。
for bad in ('static_cast<float>(c)', 'static_cast<float>(p)', 'static_cast<float>(j)',
            'static_cast<float>(n)', 'static_cast<float>(uint32'):
    if bad in body:
        raise SystemExit('[FAIL] 探针里有非法转换 %s（float<->无符号 变量）' % bad)

A_METHOD = '    __aicore__ inline void Process()\n    {\n'
assert src.count(A_METHOD) == 1, 'anchor miss (Process): %d' % src.count(A_METHOD)
src = src.replace(A_METHOD, body + '\n' + A_METHOD, 1)

A_ENTRY = '    op.Process();\n'
assert src.count(A_ENTRY) == 1, 'anchor miss (entry): %d' % src.count(A_ENTRY)
src = src.replace(A_ENTRY, '    op.XChainProbe();\n', 1)

# 自检：花括号必须平衡（不平衡 = ccec 静默挂在别处）
if src.count('{') != src.count('}'):
    raise SystemExit('[FAIL] 探针源花括号不平衡 %d vs %d' % (src.count('{'), src.count('}')))
sys.stdout.write(src)
