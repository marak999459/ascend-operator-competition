#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P6 前置探针（任务 #21）：arch22 手写 Cube 流水在本构建流程上是否可用。

用法: mk_probe_cube.py <launch|bare|shim|wspeek|wsnw|wsmix|wshigh|wsaiv|wsmixo|mmad|aivlive>   —— 把打完探针补丁的 kernel 打到 stdout
⚠️ 产物**只推到远端副本** ~/sfa_real/code/...，本地提交源一个字节都不写（探针纪律，§15.14(f)）。

各模式（全部加 MIX 入口宏）：
  launch   + **官方** matmul_intf.h；AIC 只往 GM 写到达标记；AIV 停工。
           ⇒ 裁定"MIX 形态能不能被启动、AIC 到底有没有块、blockIdx 怎么映射"。
  bare     同 launch，但 **AIC 一个字都不写** —— 用来把"框架在入口桩里插的
           matmul::clearWorkspace 越界"和"我们自己在 cube 核上写 GM 越界"分开（二分）。
  shim     不引官方头，改为给 `matmul::clearWorkspace` 一个空实现（本算子不碰 KFC）。
           ⇒ 已裁定：launch/bare 挂的那个 MTE DDR 越界在框架的 ClearWorkspaceImpl 里，
             空实现后 MIX 反复启动全部干净 ⇒ **cube 相关档位一律走 shim**。
  wspeek   shim + 入口报址档：AIC/AIV 各自报出 (workspace-attention_out) 与 (tiling-attention_out)，
           不碰 workspace ⇒ 裁定 MIX 下 KFC 基址寄存器到底有没有被装上（§15.23(i)）。
  wsnw     **kernel 代码与 wspeek 完全相同**，只是 host 侧声明改成 `16 MB + 128 KB` ⇒
           这是 wsmix 的**对照组**：若 wsnw 反复 launch 干净而 wsmix 挂在 rep0，就证明"挂"是
           **我们自己写 usrWorkspace** 造成的，不是"声明 16 MB"或 MIX 形态本身。
  wsmix    shim + 入口报址 + **真往 workspace 形参指向的 GM 写数据**，host 声明 `16 MB + 128 KB`
           （保留区之后才有可写空间）⇒ 裁定"MIX 下官方 workspace 通道能否当跨核暂存"。
  wsmixo   同 wsmix，但走**官方** matmul_intf.h（真 clearWorkspace）⇒ 回答"声明够大以后还要不要 shim"。
  mmad     shim + GM--Nd2Nz-->L1--LoadData-->L0A/L0B--Mmad-->L0C--Fixpipe-->GM 全链，
           结果落在 softmax_sum_out[16 + i*64 + j]，由 harness 侧和 CPU 参考值对拍。
  aivlive  shim + 只加入口宏，AIV 照常跑**真实算子**（正确性 + 计时）
           ⇒ 裁定 MIX 形态下 AIV 路径是否不受影响（1AIC:2AIV 会不会把 AIV 挤成 26 块）。
"""
import io
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'mmad'
ALL = ('launch', 'bare', 'shim', 'wspeek', 'wsnw', 'wsmix', 'wshigh', 'wsaiv', 'wsmixo',
       'xcore', 'xcoreb', 'xcorec', 'xcored', 'xcoree', 'xcoref', 'xcoreg', 'xcorei', 'xcorej',
       'xcorep', 'cubeloop3', 'cubeloop3nb', 'cubeloop2', 'cubemmad3', 'xcoremm3', 'cubethr',
       'cubethr2', 'cubethr3', 'cubethr4', 'cubethr5', 'cubethr6', 'mmad', 'aivlive')
assert MODE in ALL, MODE
MIX = ALL                     # 全部加 MIX 入口宏
# 官方 matmul_intf.h（真 clearWorkspace）只留给 launch/bare/wsmixo 做二分证据；
# 其余档一律用空实现 —— 真机实证官方那条路在"声明 0/4/20 MB"下第 2 次 launch 必挂（§15.20(b)）。
OFFICIAL_INC = ('launch', 'bare', 'wsmixo')
SHIM_INC = tuple(m for m in ALL if m not in OFFICIAL_INC)
ENTRY_PROBE = ('wspeek', 'wsnw', 'wsmix', 'wshigh', 'wsaiv', 'wsmixo')     # 装入口报址块
WRITE_WS = ('wsmix', 'wshigh', 'wsaiv', 'wsmixo')                          # 其中真写 workspace 的
# 写点基址：wshigh 把两组写点整体抬到 usrWorkspace + 64 KB，用来二分"第 2 次 launch 必挂"
# 是不是因为我们踩了框架在 usrWorkspace **头部**维护的 MIX/KFC 控制结构。
WS_BASE = {'wsmix': 0, 'wshigh': 16384, 'wsaiv': 0, 'wsmixo': 0}
# 谁写：wsaiv 只让 AIV 写（AIC 一个字不碰）⇒ 另一路二分。
WS_WHO = {'wsaiv': 'aiv'}
# xcore = 只测正向（AIC 写普通输出张量 + CrossCoreSetFlag ⇒ AIV 等待并读回）；
# xcoreb = 再加反向回执（AIV → AIC 的 CrossCoreSetFlag，AIC 等它）⇒ 真流水要的是双向。
#          ⛔ 真机卡死（rc=124）：它只让偶数号 AIV 回执 ⇒ 违反"AIV→AIC 是**扇入**"（§15.26(b)）。
# xcorec = 修好扇入（两个 AIV 都回执）+ **同一对旗标在一次 launch 内往返 3 轮** ⇒ P6 要的原语。
#   真机结论（§15.29）：**握手全绿**（3 轮 + 反复 launch 不挂），但 AIV 从"AIC 的标量 SetValue"
#   读回 0 ⇒ 标量写跨核不可见/不排序。P6 的写方是 Fixpipe，不是标量写 ⇒ 再加 xcored。
# xcored = 同一套 3 轮握手，但 AIC 的**数据**改走 `Mmad → Fixpipe → GM`（P6 的真实形态），
#   并且 AIV 把读回的值 echo 回 sumGm ⇒ 与 host 自己读到的同一格对照 ⇒ 一次跑分清
#   "旗标通不通"（xcorec 已答）与"**Fixpipe 写的数跨核到底看不看得见**"（P6 押在这上面）。
XCORE = ('xcore', 'xcoreb', 'xcorec', 'xcored', 'xcoree', 'xcoref', 'xcoreg', 'xcorei', 'xcorej',
         'xcorep', 'cubeloop3', 'cubeloop3nb', 'cubeloop2', 'cubemmad3', 'xcoremm3')
AIV_IDLE = ('launch', 'bare', 'shim', 'wspeek', 'wsnw', 'wsmix', 'wshigh', 'wsaiv', 'wsmixo',
            'xcore', 'xcoreb', 'xcorec', 'xcored', 'xcoree', 'xcoref', 'xcoreg', 'xcorei', 'xcorej',
            'xcorep', 'cubeloop3', 'cubeloop3nb', 'cubeloop2', 'cubemmad3', 'xcoremm3', 'cubethr',
            'cubethr2', 'cubethr3', 'cubethr4', 'cubethr5', 'cubethr6', 'mmad')

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code')
SRC = os.path.join(BASE, 'op_kernel', 'sparse_flash_attention.cpp')
s = io.open(SRC, encoding='utf-8').read()

KER_INC = '#include "kernel_operator.h"\n'
assert s.count(KER_INC) == 1, 'include anchor miss'
# 根因（9.0.0 代码实证）：只要 kernel 被判定为 MIX，构建侧生成的入口桩就会插
#   `#ifdef MIX_CORE_MACRO / if constexpr (g_coreType == AIC) matmul::clearWorkspace(workspace);`
#   （asc_op_compile_base/asc_op_compiler/compile_op.py:450-456），
# 而配套的 `#include ".../matmul_intf.h"` 只在 **c310 且开 dump** 时才写（同文件 576-579），
# ⇒ arch22 上纯 AIV 算子改成 MIX 必然报 `use of undeclared identifier 'matmul'`。
# 官方 arch22 MIX 算子（内置 SFA）靠自带 `lib/matmul_intf.h` 绕过；本安装没有 lib/ 目录，
# 公开路径是 adv_api/matmul_intf.h，它在 __NPU_ARCH__==2201 时拉进 AscendC::clearWorkspace。
if MODE in OFFICIAL_INC:
    s = s.replace(KER_INC, KER_INC + '#include "adv_api/matmul_intf.h"   // [CUBEPROBE] MIX 入口桩要 matmul::clearWorkspace\n', 1)
# shim 档：不引官方头，改为在 namespace matmul 里给桩要调的符号一个空实现。
# 动机（真机实证）：launch/bare 两档都在**第 2 次 launch**稳定挂在 clearWorkspace 内部的
# "MTE DDR address out of range"（bare 档 AIC 一个字都不写、同样挂 ⇒ 越界点在框架插的
# AscendC::ClearWorkspaceImpl 里，不是我们的 cube 代码），而本算子只用裸 Mmad/Fixpipe，
# 完全不碰 KFC 消息队列 ⇒ 清空实现语义上是安全的。
if MODE in SHIM_INC:
    s = s.replace(KER_INC, KER_INC + (
        '\n// [CUBEPROBE] 入口桩会调 matmul::clearWorkspace(workspace)；本算子不用 KFC/Matmul API，\n'
        '// ⇒ 给一个空实现，MIX 形态就不依赖 op 声明的 workspace 布局。\n'
        'namespace matmul { __aicore__ inline void clearWorkspace(GM_ADDR) {} }\n'), 1)

# ---- 1) 入口：声明 MIX 1AIC:2AIV（官方 arch22 同款，B:sparse_flash_attention.cpp:41）----
A_ENTRY = '    REGISTER_TILING_DEFAULT(SparseFlashAttentionTilingData);'
assert s.count(A_ENTRY) == 1, 'entry anchor miss'
if MODE in MIX:
    s = s.replace(A_ENTRY,
                  '    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);   // [CUBEPROBE]\n' + A_ENTRY, 1)

# ---- 2) 入口探针：每块一条 **8 个 int32** 的记录（AIC 基址 100 / AIV 基址 500 ⇒ blockDim ≤ 48 不重叠）
# 动机（§15.21/§15.23）：AIV-only 形态下 workspace 形参恒等于 0x1000000（KFC 寄存器为 0 + 16 MB 保留区），
# 而 MIX 形态下运行**可能**给每个线程装好 KFC 基址 ⇒ P6 的 GM 暂存能不能走官方通道，全看这一档。
# ⚠️ 第一版按"每块 4 格"跨步却写了 6 格 ⇒ 相邻块互相覆盖，读数像"只有 b0/b1 活着"（本轮踩过）。
# ⚠️ 第二版让两条通道挤在同一段上（AIV 分支写 float[64+bi]，入口块写 int32[400+8bi]）⇒ 分不清
#    "块没启动"和"标记被另一条通道覆盖"。现在各占一段，且**到达 / 存活**两位分开：
#      [+0/+1] (workspace - attention_out) lo/hi（字节）  [+4] blockIdx   [+5] 7 = 到达探针
#      [+2/+3] (tiling - attention_out)    lo/hi          [+6] 9 = 活着走出对 usrWorkspace 的真写
def ENTRY(write_ws, base=0, who='both'):
    w = ''
    if write_ws:
        cond = 'if ASCEND_IS_AIV {' if who == 'aiv' else '{'
        w = '''        %s   // 真写 usrWorkspace：AIC 从 float[%d] 起每块 64 格，AIV 再错开 8 KB ⇒ 一次跑分清两个通道
            uint32_t slot = %du + 64u * bi;
            float tag = 5.0f;
            if ASCEND_IS_AIV { slot += 2048u; tag = 6.0f; }
            GlobalTensor<float> wg;
            wg.SetGlobalBuffer(pws + slot);
            wg.SetValue(0, tag);
            GlobalTensor<int32_t> wb;
            wb.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(pws + slot) + 1);
            wb.SetValue(0, static_cast<int32_t>(bi));
            rg.SetValue(s + 6, 9);   // [CUBEPROBE] 活着走出 workspace 写（挂了这格就没有）
        }
''' % (cond, base, base)
    return '''
    {   // [CUBEPROBE][WSENTRY]
        const uint32_t bi = GetBlockIdx();
        __gm__ float *pws = reinterpret_cast<__gm__ float *>(workspace);
        __gm__ float *pout = reinterpret_cast<__gm__ float *>(attentionOut);
        const int64_t d = pws - pout;
        const int64_t dt = (reinterpret_cast<__gm__ float *>(tiling)) - pout;
        GlobalTensor<int32_t> rg;
        rg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(softmaxSumOut));
        uint32_t s = 100u + 8u * bi;
        if ASCEND_IS_AIV { s = 500u + 8u * bi; }
        rg.SetValue(s + 0, static_cast<int32_t>(d & 0xFFFFFFFF));
        rg.SetValue(s + 1, static_cast<int32_t>(d >> 32));
        rg.SetValue(s + 2, static_cast<int32_t>(dt & 0xFFFFFFFF));
        rg.SetValue(s + 3, static_cast<int32_t>(dt >> 32));
        rg.SetValue(s + 4, static_cast<int32_t>(bi));
        rg.SetValue(s + 5, 7);
''' + w + '''    }
'''


if MODE in ENTRY_PROBE:
    s = s.replace(A_ENTRY, A_ENTRY + ENTRY(MODE in WRITE_WS, WS_BASE.get(MODE, 0),
                                           WS_WHO.get(MODE, 'both')), 1)

# ---- 3) AIC 分支：跑探针；AIV 分支：停工，独占 max/sum 缓冲当探针出口 ----
A_AIC = ('        if ASCEND_IS_AIC {\n'
         '            unitBegin_ = 0; unitEnd_ = 0; unitStep_ = 1;\n'
         '            return;\n'
         '        }\n')
assert s.count(A_AIC) == 1, 'AIC anchor miss'
if MODE in AIV_IDLE:
    NEW_AIC = ('        if ASCEND_IS_AIC {\n'
               '            unitBegin_ = 0; unitEnd_ = 0; unitStep_ = 1;\n'
               + ('            XcoreAic();\n' if MODE in XCORE else
                  ('            CubeProbe();\n'
                   if MODE not in ('bare', 'wsmix', 'wshigh', 'wsaiv', 'wsmixo') else '')) +
               '            return;\n'
               '        }\n'
               '        if ASCEND_IS_AIV {   // [CUBEPROBE] AIV 停工，把 sum 缓冲让给探针当出口\n'
               '            unitBegin_ = 0; unitEnd_ = 0; unitStep_ = 1;\n'
               + ('            XcoreAiv();\n' if MODE in XCORE else
                  ('            sumGm_.SetValue(64u + GetBlockIdx(), 6.0f);   // [CUBEPROBE] AIV 的 blockIdx 空间\n'
                   if MODE in ('launch', 'bare', 'shim') else '')) +
               '            return;\n'
               '        }\n')
    s = s.replace(A_AIC, NEW_AIC, 1)

# ---- 4) 探针函数本体 ----
PROBE_MIN = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    // launch 档：只证明 "MIX 形态下 AIC 真有块在跑、且 AIC 能写 GM"。
    // 出口 = softmax_sum_out：[0]=7 到达、[1+blockIdx%8]=5 谁被启动。
    __aicore__ inline void CubeProbe()
    {
        sumGm_.SetValue(0, 7.0f);
        const uint32_t bi = GetBlockIdx();
        sumGm_.SetValue(1u + (bi % 8u), 5.0f);
    }

'''
PROBE = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    // C[i][j] = Σ_k A[i][k]·B[j][k]，A = key 行 0..15 列 0..15，B = key 行 16..31 列 0..15。
    // 出口 = softmax_sum_out：[0]=7 到达、[1+blockIdx%8]=5 谁被启动、[15]=9 全链走完、
    // [16 + i*64 + j] = C[i][j]（fp32，ND）。对拍在 harness 侧做（cube_harness_patch.py）。
    __aicore__ inline void CubeProbe()
    {
        sumGm_.SetValue(0, 7.0f);
        const uint32_t bi = GetBlockIdx();
        sumGm_.SetValue(1u + (bi % 8u), 5.0f);
        if constexpr (sizeof(DT_QUERY) == 2u) {   // Mmad 的 A/B 只收 fp16/bf16
            TBuf<TPosition::A1> bufl1;
            TBuf<TPosition::A2> bufL0A;
            TBuf<TPosition::B2> bufL0B;
            TBuf<TPosition::CO1> bufL0C;
            pipe_.InitBuffer(bufl1, 4096);
            pipe_.InitBuffer(bufL0A, 2048);
            pipe_.InitBuffer(bufL0B, 2048);
            pipe_.InitBuffer(bufL0C, 4096);
            LocalTensor<DT_QUERY> l1 = bufl1.Get<DT_QUERY>();
            const uint32_t rowD = static_cast<uint32_t>(D_);   // 512
            // 一个 16x16 的 fp16 分形 = 16 个 32B 行块 = 256 个元素
            Nd2NzParams nz;
            nz.ndNum = 1;
            nz.nValue = 16;
            nz.dValue = 16;
            nz.srcDValue = rowD;
            nz.dstNzC0Stride = 16;
            nz.dstNzNStride = 1;
            nz.srcNdMatrixStride = 0;
            nz.dstNzMatrixStride = 0;
            DataCopy(l1, kGm_, nz);
            DataCopy(l1[256], kGm_[16u * rowD], nz);
            SetFlag<HardEvent::MTE2_MTE1>(0);
            WaitFlag<HardEvent::MTE2_MTE1>(0);
            LocalTensor<DT_QUERY> l0a = bufL0A.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> l0b = bufL0B.Get<DT_QUERY>();
            LoadData2DParams ld;
            ld.startIndex = 0;
            ld.repeatTimes = 1;
            ld.srcStride = 1;
            ld.dstGap = 0;
            ld.ifTranspose = false;
            ld.sid = 0;
            ld.addrMode = 0;
            LoadData(l0a, l1, ld);
            LoadData(l0b, l1[256], ld);
            SetFlag<HardEvent::MTE1_M>(1);
            WaitFlag<HardEvent::MTE1_M>(1);
            LocalTensor<float> l0c = bufL0C.Get<float>();
            MmadParams mp;
            mp.m = 16;
            mp.n = 16;
            mp.k = 16;
            mp.cmatrixInitVal = true;
            mp.cmatrixSource = false;
            mp.unitFlag = 0b11;
            Mmad(l0c, l0a, l0b, mp);
            SetFlag<HardEvent::M_FIX>(2);
            WaitFlag<HardEvent::M_FIX>(2);
            FixpipeParamsV220 fx;
            fx.nSize = 16;
            fx.mSize = 16;
            fx.srcStride = 16;
            fx.dstStride = 64;      // 行距 64 个 float = 256B（>16，好暴露转置/步长错）
            fx.ndNum = 1;
            fx.srcNdStride = 0;
            fx.dstNdStride = 0;
            fx.unitFlag = 0b11;
            Fixpipe(sumGm_[16], l0c, fx);
            PipeBarrier<PIPE_ALL>();
            sumGm_.SetValue(15, 9.0f);
        }
    }

'''
A_MEMBER = '    TPipe pipe_;'
assert s.count(A_MEMBER) == 1, 'member anchor miss'
# ⚠️ 上一版只给 launch/shim 注入 CubeProbe()，而 wspeek/wsnw 的 AIC 分支同样会调它
#    ⇒ 真机 `error: use of undeclared identifier 'CubeProbe'`，那一档读数全是旧 .so 的假数。
# ---- 4b) xcore / xcoreb：Cube 与 Vector 之间"过一块普通输出张量 + FFTS 旗标"的中继 ----
# 为什么还有戏（§15.24 把 usrWorkspace 判死）：内置 arch22 SFA 用的
#   `CrossCoreSetFlag<2, pipe>(id)` / `CrossCoreWaitFlag<2, pipe>(id)` 在本安装的
#   `dav_c220/kernel_operator_sync_impl.h:429-440` 是**真指令**（`ffts_cross_core_sync` /
#   `wait_flag_dev`），不是别的 arch 上那种 `ASCENDC_ASSERT(false)` 占位；而"AIC 往普通 GM 张量写
#   + 反复 launch"已由 shim/mmad 两档证明干净（poison 只跟"写 usrWorkspace"绑定）。
# ⇒ P6 的中间量传输可以不走 workspace：**每个单元独占 LSE/输出张量的一小段**，AIC 把 score 写进
#   本单元那段、发旗标，AIV 等旗标读走算 softmax，（xcoreb）再回执给 AIC。
# 出口（都在 softmax_sum_out 的 float 视图）：
#   [0]      = 7.0  AIC 活着走出旗标回合      [8+bi]   = 9.0 AIC 到达位
#   [700+bi] = AIV 从 GM 读回的**值**（3..10 ⇒ 配对 g=bi/2 成立；0 ⇒ 读到垃圾）
#   [764+bi] = 6.0  AIV 到达位
PROBE_XC = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    // AIC → GM → AIV 的中继：不碰 workspace，用 softmax_max_out（fp32 LSE 输出）当暂存。
    __aicore__ inline void XcoreAic()
    {
        const uint32_t bi = GetBlockIdx();
        static constexpr float TAGV[8] = {3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
        const float t = TAGV[bi & 7u];        // 每块一个可区分的值 ⇒ 一次跑读出配对关系
        const uint32_t base = 64u * bi;
        for (uint32_t i = 0; i < 16u; ++i) { maxGm_.SetValue(base + i, t); }
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_FIX>(5);
        %s
        sumGm_.SetValue(0, 7.0f);
        sumGm_.SetValue(8u + (bi & 7u), 9.0f);
    }

    __aicore__ inline void XcoreAiv()
    {
        const uint32_t bi = GetBlockIdx();
        CrossCoreWaitFlag<2, PIPE_V>(5);
        const uint32_t g = bi >> 1;           // 1AIC:2AIV 的猜测配对：AIV bi ↔ AIC g
        sumGm_.SetValue(700u + bi, maxGm_.GetValue(64u * g + 3u));
        sumGm_.SetValue(764u + bi, 6.0f);
        %s
    }

'''
# ---- 4d) xcored：把"跨核数据"换成 P6 的真实形态 = AIC 的 `Mmad → Fixpipe → GM` ----
# xcorec 已经答完"旗标"这一半（3 轮复用同一对 + 双向扇入 + 反复 launch 不挂），但它的数据格
# 读回 0 ⇒ AIC 的**标量 SetValue** 对同组 AIV 不可见（而 host 在任务结束后读得到，见
# `AIC alive=7`）⇒ P6 不能靠标量写跨核，必须走 Fixpipe（官方 arch22 SFA 也正是这么交接的）。
# 本档：AIC 用与 `mmad` 档**完全相同**的一份数据算同一个 16x16 C（A = key 行 0..15，
#   B = key 行 16..31，K=16），每轮把 C 整块 Fixpipe 到 `sumGm_[16 + i*64 + j]`（= mmad 档
#   的出口布局 ⇒ harness 里现成的 `mismatchC` 直接校验 host 侧看到的值），
#   同时 AIV 每轮 echo 它读到的 `C[r][r]` 到 `sumGm_[600 + bi*4 + r]`。
# ⇒ 判据：`rd=vN[..,..,..]` 三格都等于 host 侧那同一格的期望值 ⇒ **Cube 写的数 AIV 读得到**；
#   读到 0 ⇒ 还缺一道 flush；读到上一轮的值 ⇒ 旗标没排住 Fixpipe。
# ⚠️ 两侧都用 `if constexpr (sizeof(DT_QUERY)==2)` 包起来：Mmad 的 A/B 只收 fp16/bf16，
#   若只包 AIC 侧，fp32 实例的 AIV 会死等一个没人发的旗标。
PROBE_XCD = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    __aicore__ inline void XcoreAic()
    {
        const uint32_t bi = GetBlockIdx();
        sumGm_.SetValue(8u + (bi & 15u), 9.0f);
        if constexpr (sizeof(DT_QUERY) == 2u) {
            TBuf<TPosition::A1> bufl1;
            TBuf<TPosition::A2> bufL0A;
            TBuf<TPosition::B2> bufL0B;
            TBuf<TPosition::CO1> bufL0C;
            pipe_.InitBuffer(bufl1, 4096);
            pipe_.InitBuffer(bufL0A, 2048);
            pipe_.InitBuffer(bufL0B, 2048);
            pipe_.InitBuffer(bufL0C, 4096);
            LocalTensor<DT_QUERY> l1 = bufl1.Get<DT_QUERY>();
            const uint32_t rowD = static_cast<uint32_t>(D_);
            Nd2NzParams nz;
            nz.ndNum = 1;
            nz.nValue = 16;
            nz.dValue = 16;
            nz.srcDValue = rowD;
            nz.dstNzC0Stride = 16;
            nz.dstNzNStride = 1;
            nz.srcNdMatrixStride = 0;
            nz.dstNzMatrixStride = 0;
            DataCopy(l1, kGm_, nz);
            DataCopy(l1[256], kGm_[16u * rowD], nz);
            SetFlag<HardEvent::MTE2_MTE1>(0);
            WaitFlag<HardEvent::MTE2_MTE1>(0);
            LocalTensor<DT_QUERY> l0a = bufL0A.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> l0b = bufL0B.Get<DT_QUERY>();
            LoadData2DParams ld;
            ld.startIndex = 0;
            ld.repeatTimes = 1;
            ld.srcStride = 1;
            ld.dstGap = 0;
            ld.ifTranspose = false;
            ld.sid = 0;
            ld.addrMode = 0;
            LoadData(l0a, l1, ld);
            LoadData(l0b, l1[256], ld);
            SetFlag<HardEvent::MTE1_M>(1);
            WaitFlag<HardEvent::MTE1_M>(1);
            LocalTensor<float> l0c = bufL0C.Get<float>();
            MmadParams mp;
            mp.m = 16;
            mp.n = 16;
            mp.k = 16;
            mp.cmatrixInitVal = true;
            mp.cmatrixSource = false;
            mp.unitFlag = 0b11;
            Mmad(l0c, l0a, l0b, mp);
            SetFlag<HardEvent::M_FIX>(2);
            WaitFlag<HardEvent::M_FIX>(2);
            FixpipeParamsV220 fx;
            fx.nSize = 16;
            fx.mSize = 16;
            fx.srcStride = 16;
            fx.dstStride = 64;
            fx.ndNum = 1;
            fx.srcNdStride = 0;
            fx.dstNdStride = 0;
            fx.unitFlag = 0b11;
            for (uint32_t r = 0; r < 3u; ++r) {
                Fixpipe(sumGm_[16], l0c, fx);       // C 落 GM（mmad 档同一布局）
                PipeBarrier<PIPE_ALL>();
                CrossCoreSetFlag<2, PIPE_FIX>(5);   // 广播给本组两个 AIV
                CrossCoreWaitFlag<2, PIPE_FIX>(6);   // 等本组两个 AIV 都回执（扇入）
            }
        }
        PipeBarrier<PIPE_ALL>();
        sumGm_.SetValue(0, 7.0f);
    }

    __aicore__ inline void XcoreAiv()
    {
        const uint32_t bi = GetBlockIdx();
        const uint32_t g = bi >> 1;
        if constexpr (sizeof(DT_QUERY) == 2u) {
            for (uint32_t r = 0; r < 3u; ++r) {
                CrossCoreWaitFlag<2, PIPE_V>(5);
                const float v = sumGm_.GetValue(16u + r * 64u + r);   // C[r][r]
                PipeBarrier<PIPE_ALL>();
                sumGm_.SetValue(600u + bi * 4u + r, v);
                CrossCoreSetFlag<2, PIPE_MTE3>(6);
            }
        }
        sumGm_.SetValue(760u + bi, 6.0f);
    }

'''
if MODE == 'xcored':
    s = s.replace(A_MEMBER, PROBE_XCD + A_MEMBER, 1)
elif MODE in ('xcoree', 'xcoref', 'xcoreg', 'xcorei', 'xcorej', 'xcorep',
                   'cubeloop3', 'cubeloop3nb', 'cubeloop2', 'cubemmad3'):
    # ---- 4e) xcoree：修掉 xcored 的"出口布局自毁"，再把两个变量拆开 ----
    # `xcored` 真机 rc=124（挂）+ `/tmp/cp*.txt` 0 字节（stdout 全缓冲，被 timeout 打死时没落盘）。
    # 复盘发现**它的布局本来就不成立**：C 落在 `sumGm_[16 + i*64 + j]`（i,j<16）⇒ 浮点
    #   16..991 整片被占，把 AIV 的 echo 槽 600..662 与到达位 760..775 **全覆盖**；
    #   而且那三格到达位是 AIV 用标量写的、写完才被下一轮 Fixpipe 抹掉 ⇒ 就算不挂也什么都读不出。
    # 本档一次改三件事，每件都对应一个之前说不清的岔口：
    #   ① **布局**：C 的 `dstStride` 从 64 改成 16、基址 32 ⇒ 只占 32..271，标记区（0/8..23/
    #      600../680../760..）全在 C 之外；harness 侧基址与行距改走环境变量
    #      `SFA_CBASE`/`SFA_CSTRIDE`（默认 16/64 ⇒ `mmad` 档的对拍口径一个字都不动）。
    #   ② **单向**：AIC 每轮广播旗标 5、**不等回执** ⇒ 把"cube 流水与 AIV 共存"和"扇入回执"
    #      两个变量拆开（`xcorec` 已单独证明后者在没有 cube 时是通的）。
    #   ③ **AIV 两条读路径各存一份**：标量 `GetValue` → `600+bi*4+r`；`DataCopy`(MTE2) 进 UB
    #      再读 → `680+bi*4+r` ⇒ 一次跑分清"Fixpipe 写的数 AIV 看不看得见"与
    #      "是不是只有走 MTE2 才看得见"（内置 arch22 SFA 的向量侧正是后者：DataCopy 进 UB 再算）。
    # ⚠️ 三轮写的是**同一块 C、同一个 l0c** ⇒ 值与轮次无关：本档只裁可见性，不裁顺序
    #    （顺序那半边由 `xcorec` 的查表标签答过了）。
    PROBE_XEE = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    __aicore__ inline void XcoreAic()
    {
        const uint32_t bi = GetBlockIdx();
        sumGm_.SetValue(8u + (bi & 15u), 9.0f);
        if constexpr (sizeof(DT_QUERY) == 2u) {
            TBuf<TPosition::A1> bufl1;
            TBuf<TPosition::A2> bufL0A;
            TBuf<TPosition::B2> bufL0B;
            TBuf<TPosition::CO1> bufL0C;
            pipe_.InitBuffer(bufl1, 4096);
            pipe_.InitBuffer(bufL0A, 2048);
            pipe_.InitBuffer(bufL0B, 2048);
            pipe_.InitBuffer(bufL0C, 4096);
            LocalTensor<DT_QUERY> l1 = bufl1.Get<DT_QUERY>();
            const uint32_t rowD = static_cast<uint32_t>(D_);
            Nd2NzParams nz;
            nz.ndNum = 1;
            nz.nValue = 16;
            nz.dValue = 16;
            nz.srcDValue = rowD;
            nz.dstNzC0Stride = 16;
            nz.dstNzNStride = 1;
            nz.srcNdMatrixStride = 0;
            nz.dstNzMatrixStride = 0;
            DataCopy(l1, kGm_, nz);
            DataCopy(l1[256], kGm_[16u * rowD], nz);
            SetFlag<HardEvent::MTE2_MTE1>(0);
            WaitFlag<HardEvent::MTE2_MTE1>(0);
            LocalTensor<DT_QUERY> l0a = bufL0A.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> l0b = bufL0B.Get<DT_QUERY>();
            LoadData2DParams ld;
            ld.startIndex = 0;
            ld.repeatTimes = 1;
            ld.srcStride = 1;
            ld.dstGap = 0;
            ld.ifTranspose = false;
            ld.sid = 0;
            ld.addrMode = 0;
            LoadData(l0a, l1, ld);
            LoadData(l0b, l1[256], ld);
            SetFlag<HardEvent::MTE1_M>(1);
            WaitFlag<HardEvent::MTE1_M>(1);
            LocalTensor<float> l0c = bufL0C.Get<float>();
            MmadParams mp;
            mp.m = 16;
            mp.n = 16;
            mp.k = 16;
            mp.cmatrixInitVal = true;
            mp.cmatrixSource = false;
            mp.unitFlag = 0b11;
            Mmad(l0c, l0a, l0b, mp);
            SetFlag<HardEvent::M_FIX>(2);
            WaitFlag<HardEvent::M_FIX>(2);
            FixpipeParamsV220 fx;
            fx.nSize = 16;
            fx.mSize = 16;
            fx.srcStride = 16;
            fx.dstStride = 16;       // 行距 16 个 float = 64B ⇒ C 只占 sumGm_[32..271]
            fx.ndNum = 1;
            fx.srcNdStride = 0;
            fx.dstNdStride = 0;
            fx.unitFlag = 0b11;
            for (uint32_t r = 0; r < 3u; ++r) {
                Fixpipe(sumGm_[32], l0c, fx);
                PipeBarrier<PIPE_ALL>();
                CrossCoreSetFlag<2, PIPE_FIX>(5);   // 只广播，不等回执（单向）
            }
        }
        PipeBarrier<PIPE_ALL>();
        sumGm_.SetValue(0, 7.0f);
    }

    __aicore__ inline void XcoreAiv()
    {
        const uint32_t bi = GetBlockIdx();
        if constexpr (sizeof(DT_QUERY) == 2u) {
            // 事件号 3/4：0..2 是 AIC 那半边的 cube 链在用，5/6 是跨核旗标 ⇒ 本组两个 AIV 各占一个，
            // ⛔ 不能都写死同一个号（若 arch22 的事件寄存器是**按物理核**而非按 subcore 分的，
            //    同核两个 AIV 会互相吃掉对方的 wait）。
            const uint8_t fid = static_cast<uint8_t>(3u + (bi & 1u));
            // %UBDECL%
            for (uint32_t r = 0; r < 3u; ++r) {
                CrossCoreWaitFlag<2, PIPE_V>(5);
                const float sv = sumGm_.GetValue(32u + 17u * r);      // C[r][r]，标量直读 GM
                // %UBREAD%
                sumGm_.SetValue(600u + bi * 4u + r, sv);
                sumGm_.SetValue(680u + bi * 4u + r, mv);
            }
        }
        PipeBarrier<PIPE_ALL>();
        sumGm_.SetValue(760u + bi, 6.0f);
    }

'''
    # AIV 的"MTE2 读"那半边做成可开关：xcoree 全开，xcoref 关掉（只留标量直读）。
    # 为什么要有 xcoref：`xcoree` 与 `xcored` 都在**设备侧**死锁（stdbuf 之后看得见 host 已经
    # 走到 launch、打完 wsSize 就再没输出），而单向档已经把"扇入回执"排除了 ⇒ 剩下的嫌疑里
    # 最贵的一条是"AIV 为读 GM 而 InitBuffer 了一块 UB，同时 AIC 正占着 L1/L0A/L0B/L0C"
    # （`mmad` 档能过，恰恰因为它的 AIV 一个字节的 UB 都不分配、也不等任何旗标）。
    # ⇒ xcoref = xcoree **只**去掉 AIV 的 UB/DataCopy：
    #     通过 ⇒ 死锁点定位到"AIV 用 UB/MTE2 与 AIC 用 L1/L0 共存"，顺带拿到标量可见性判据；
    #     仍挂 ⇒ 与 UB 无关，下一步换"旗标换成 FIX 之外的 pipe / AIC 不做 Mmad"继续二分。
    XEE_UBDECL = ('            pipe_.InitBuffer(probeBuf_, 128);\n'
                  '            LocalTensor<float> ub = probeBuf_.Get<float>();')
    XEE_UBREAD = ('                DataCopy(ub, sumGm_[32u + 16u * r], DataCopyParams{1, 2, 0, 0});'
                  '   // C[r][0..15] 走 MTE2\n'
                  '                SetFlag<HardEvent::MTE2_V>(fid);\n'
                  '                WaitFlag<HardEvent::MTE2_V>(fid);\n'
                  '                const float mv = ub.GetValue(r);   // 同一格，MTE2 路径')
    body = PROBE_XEE
    if MODE in ('xcoref', 'xcorei', 'xcorej', 'xcorep',
                     'cubeloop3', 'cubeloop3nb', 'cubeloop2', 'cubemmad3'):
        body = body.replace('            // %UBDECL%', '').replace(
            '                // %UBREAD%',
            '                const float mv = -7.0f;   // [xcoref] MTE2 那半边关掉，哨兵值 -7 一眼可辨')
    else:
        body = body.replace('            // %UBDECL%', XEE_UBDECL).replace(
            '                // %UBREAD%', XEE_UBREAD)
    if MODE == 'xcoreg':
        # 为什么要这一档：xcoree 里 AIV 的 MTE2->V 同步用的是我**自己挑的** 3/4 号事件
        # （"怕同核两个 AIV 撞号"），而提交版 kernel 一直是两个 AIV 各用 0 号、跑了无数遍都干净
        # ⇒ arch22 上 MTE2->V 这一对的事件号空间很可能就是**按 subcore 分**的，那我"避撞"反而
        #   越界了（超范围的 event id 在 2201 上是静默错，不是编译错误）。xcoreg 换回 0 号：
        #   通过 ⇒ xcoree 的挂与 UB/MTE2 无关，就是事件号；仍挂 ⇒ 事件号排除，剩下 UB 分配共存。
        body = body.replace('static_cast<uint8_t>(3u + (bi & 1u))', 'static_cast<uint8_t>(0u)')
    if MODE in ('xcorei', 'xcorej'):
        # ---- 4f) xcorei / xcorej = xcoref 的**少轮版**：AIC 只 set N 次、AIV 只 wait N 次 ----
        # N=1（`xcorei`）天生配平 ⇒ 把"旗标计数不锁存"这条与"cube 链"彻底分开；
        # N=2（`xcorej`）量**最小致命轮次**（1 轮已知通、3 轮已知挂）。
        ROUNDS = {'xcorei': '1u', 'xcorej': '2u'}[MODE]
        n = body.count('for (uint32_t r = 0; r < 3u; ++r) {')
        assert n == 2, '%s: 轮次循环锚点 %d != 2' % (MODE, n)
        body = body.replace('for (uint32_t r = 0; r < 3u; ++r) {',
                            'for (uint32_t r = 0; r < %s; ++r) {' % ROUNDS)
    if MODE == 'xcorep':
        # ---- 4g) xcorep = P6 真正要的那个形态：cube 链 + **配平的双向握手**（xcorec 已证的协议）----
        # xcorec(通) 与 xcoree/f(挂) 之间一次改了两件事：①AIC 的写方从标量 `SetValue` 换成
        # `Mmad→Fixpipe`；②AIC 从"每轮等回执"变成"连发 3 个 set 不等"。本档**只**改①：
        # 旗标侧与 xcorec 逐字同形（set(5) 广播 → 扇入 wait(6)，两个 AIV 都回执），
        # 数据侧换成 Fixpipe。
        #   通 ⇒ 交接协议成立，P6 解锁（代价：每轮一次往返，AIC/AIV 无法深度重叠 ⇒ 收益打折但要实测）；
        #   挂 ⇒ "Fixpipe 与 wait_flag 共存"这条结构解释坐实，§15.22~§15.30 整条 Cube 线按不可行结案。
        AIC_OLD = '                CrossCoreSetFlag<2, PIPE_FIX>(5);   // 只广播，不等回执（单向）'
        AIV_OLD = '                sumGm_.SetValue(680u + bi * 4u + r, mv);'
        assert body.count(AIC_OLD) == 1 and body.count(AIV_OLD) == 1, 'xcorep: 锚点 miss'
        body = body.replace(AIC_OLD, AIC_OLD +
                            '\n                CrossCoreWaitFlag<2, PIPE_FIX>(6);   // [xcorep] 扇入：等本组两个 AIV 都读过')
        body = body.replace(AIV_OLD, AIV_OLD +
                            '\n                PipeBarrier<PIPE_ALL>();            // [xcorep] 读落地再回执'
                            '\n                CrossCoreSetFlag<2, PIPE_MTE3>(6);  // [xcorep] 两个 AIV 都 set ⇒ 凑齐扇入')
    if MODE in ('cubeloop3', 'cubeloop3nb', 'cubeloop2'):
        # ---- 4h) cubeloop3 = 同一套 cube 链**原地跑 3 轮**，但**一个旗标都不发**、AIV 停工 ----
        # 为什么必须有这一档（§15.30(c) 差分矩阵缺的那一格）：
        #   xcorei (1 轮 + 旗标 + AIV 等) = **通**，而且 AIV 标量读到了 Fixpipe 的真值；
        #   xcoree/f (3 轮 + 旗标 + AIV 等) = 挂；xcorep (3 轮 + 配平双向) = 挂；
        #   xcorec (3 轮 + 旗标 + AIV 等，**无 cube**) = 通
        # ⇒ 翻掉"通→挂"的那个变量是**轮次数**，但所有 3 轮档都同时带着"旗标 + cube"两件事，
        #   所以"cube 链自己跑 3 轮会不会挂"从来没被单独裁过。本档把跨核整个摘掉：
        #   🔎 真机裁定（§15.30(g)）= **挂** ⇒ 跨核彻底无辜，根因在"链跑第 2 轮"本身。
        assert body.count('for (uint32_t r = 0; r < 3u; ++r) {') == 2, 'cubeloop3: 轮次锚点 miss'
        AIC_SET = '                CrossCoreSetFlag<2, PIPE_FIX>(5);   // 只广播，不等回执（单向）\n'
        assert body.count(AIC_SET) == 1, 'cubeloop3: AIC 旗标锚点 miss'
        body = body.replace(AIC_SET, '')                  # AIC 只跑链，不发旗标
        i0 = body.index('    __aicore__ inline void XcoreAiv()')
        i1 = body.rindex('\n    }\n') + len('\n    }')
        body = body[:i0] + (
            '    __aicore__ inline void XcoreAiv()\n'
            '    {\n'
            '        const uint32_t bi = GetBlockIdx();\n'
            '        sumGm_.SetValue(760u + bi, 6.0f);   // [cubeloop3] 只报到：不等旗标、不读 GM\n'
            '    }') + body[i1:]
    if MODE == 'cubeloop2':
        # ---- 4k) cubeloop2 = cubeloop3 的**两轮版**（同样一个旗标都不发）----
        # `xcorej`（2 轮 + 单向旗标）已挂，但那一档里旗标和第二轮是绑在一起的；
        # 本档把旗标摘干净再量 2 轮 ⇒ "第二次 `Fixpipe` 发射"本身就是充分条件还是必要条件。
        assert body.count('for (uint32_t r = 0; r < 3u; ++r) {') == 1, 'cubeloop2: AIC 轮次锚点 miss'
        body = body.replace('for (uint32_t r = 0; r < 3u; ++r) {', 'for (uint32_t r = 0; r < 2u; ++r) {')
    if MODE == 'cubeloop3nb':
        # ---- 4i) cubeloop3nb = cubeloop3 只把循环里那道 `PipeBarrier<PIPE_ALL>` 换成
        #   `SetFlag/WaitFlag<HardEvent::M_FIX>` 一对（**同为 3 轮、同为无旗标**）。
        # 动机：`cubeloop3` rc=124 ⇒ 挂点必在循环体那两个动作之一：`Fixpipe` 或 `PipeBarrier`。
        #   MIX 下 AIC 侧的 `PIPE_ALL` 到底覆盖哪些流水**没有文档**（§15.4 已证 2201 不插自动
        #   跨流水同步）⇒ 若它把"等 MTE2 侧空闲"也算进去，第 2 轮就会自己锁死。换成只等
        #   M→FIX 的事件对 = "确认上一条 Fixpipe 收工"的最小同步。
        assert body.count('                PipeBarrier<PIPE_ALL>();') == 1, 'cubeloop3nb: 循环内 barrier 锚点 miss'
        body = body.replace('                PipeBarrier<PIPE_ALL>();',
                            '                SetFlag<HardEvent::M_FIX>(2);\n'
                            '                WaitFlag<HardEvent::M_FIX>(2);   // [cubeloop3nb] 只等 M→FIX')
    if MODE == 'cubemmad3':
        # ---- 4j) cubemmad3 = **每轮把整条链重做一遍**（Nd2Nz → LoadData → Mmad → Fixpipe），
        #   仍然 3 轮、仍然**一个跨核旗标都不发**。
        # 动机：`cubeloop3` 的循环体只读**同一块 `l0c`** 三次（`Mmad` 只在循环外发了一次）。
        #   内置 arch22 SFA 的 cube 侧从来是"一块 L0C 配一次 Fixpipe"，没有示范过"Fixpipe 连读
        #   同一块 L0C" ⇒ 这是 `cubeloop3` 挂的最省事解释。
        # ⇒ 通：**P6 直接解锁**，规则就一条"每次 `Fixpipe` 前必须有自己的 `Mmad`"（恰好就是 P6 的
        #    真实形态：每个 chunk 的 S 都是一次独立 Mmad 的产物）；
        #    挂：L0C 复用被排除，剩下"`Fixpipe` 重复发射/重复落点"本身 ⇒ 下一档换每轮不同落点。
        A0 = '            Nd2NzParams nz;'
        A1 = '            for (uint32_t r = 0; r < 3u; ++r) {'
        IA0 = body.index('    __aicore__ inline void XcoreAic()')
        IA1 = body.index('    __aicore__ inline void XcoreAiv()')
        aic = body[IA0:IA1]                     # ⚠️ 只在 AIC 那半边做重组：AIV 也有一个同缩进的 r 循环
        assert aic.count(A0) == 1 and aic.count(A1) == 1, 'cubemmad3: AIC 锚点 miss'
        i0, i1 = aic.index(A0), aic.index(A1)
        head = aic[:i0] + A1 + '\n'                                     # 缓冲/句柄 + 循环头
        chain = '\n'.join(('    ' + ln) if ln.strip() else ln           # 整条链 + fx 参数 ⇒ 多一级缩进
                           for ln in aic[i0:i1].split('\n'))
        after = aic[i1 + len(A1):]                                      # 循环体剩余 + 各级收尾
        body = body[:IA0] + head + chain + after + body[IA1:]
        body = body.replace('                CrossCoreSetFlag<2, PIPE_FIX>(5);   // 只广播，不等回执（单向）\n', '')
        i0 = body.index('    __aicore__ inline void XcoreAiv()')
        i1 = body.rindex('\n    }\n') + len('\n    }')
        body = body[:i0] + (
            '    __aicore__ inline void XcoreAiv()\n'
            '    {\n'
            '        const uint32_t bi = GetBlockIdx();\n'
            '        sumGm_.SetValue(760u + bi, 6.0f);   // [cubemmad3] 只报到：不等旗标\n'
            '    }') + body[i1:]
    s = s.replace(A_MEMBER, body + A_MEMBER, 1)
    # AIV 侧要一块自己的 UB 当 MTE2 落点（只有全开的 xcoree 引用它）
    if MODE in ('xcoree', 'xcoreg'):
        s = s.replace('    TBuf<TPosition::VECCALC> qBuf_',
                      '    TBuf<TPosition::VECCALC> probeBuf_;   // [CUBEPROBE] xcoree：AIV 的 MTE2 落点\n'
                      '    TBuf<TPosition::VECCALC> qBuf_', 1)

elif MODE == 'xcoremm3':
    # ---- 4l) xcoremm3 = **P6 的那个原语本身**：每轮重发整条链（`cubemmad3` 已证它不挂）
    #   + 每轮**内容不同**（A/B 的行按 r 平移）+ `xcorec` 那套**配平双向握手**（set(5) → 扇入 wait(6)）。
    # 为什么现在必须有这一档：§15.30(g)/(h) 把死因钉在"同一块 `l0c` 在没有新 `Mmad` 翻转搬出标志
    #   时被第 2 次 `Fixpipe` 读走"，而 `cubemmad3` 通 / `cubeloop3` 挂 只证明了**AIC 自己**能连做
    #   3 次"链 + Fixpipe"。P6 要的是**这三次各自交棒给 AIV**，而旗标那半边从来没在"每轮有 Mmad"
    #   的前提下测过（`xcorep` 挂的时候带着 L0C 那个 bug ⇒ 它的"挂"不能算旗标的数）。
    # 判据（一次跑同时给三件事的答案，靠"每轮内容不同"才分得开）：
    #   `P6mismatch=0` ⇒ 轮次序对（echo[r] == 第 r 行的期望，不是复读了同一格）+ 数据面成立
    #     + 握手在"每轮一次完整链"下配平 ⇒ **P6 解冻**，生产侧照这一档的形态写；
    #   `rc=124` ⇒ 旗标与"每轮重发链"仍不能共存（这时才真轮到 §15.29 末尾那条最坏结论）；
    #   通但 `P6mismatch>0` ⇒ 交接活了但**排序/一致性**不成立，看 echo 是哪一轮错来定（全部相同
    #     = AIV 读到同一轮 = 旗标不锁存；部分 0 = 按核共享行的老毛病 §15.30(e)(2)）。
    # 出口布局（`softmax_sum_out` 的 float 视图，big1 nlse=1024，**每格都按 8 float = 32 B 独占**）：
    #   [0]=7 AIC 走完 / [8+bi*8]=9 AIC 到达 / C_r 基址 [128+r*256]（r<3 ⇒ 128..895）/
    #   AIV echo [896+bi*8+r] 写 **v+1000**（"读到 0"与"没写"可分）/ AIV 到达 [960+bi*8]=6
    #   ⇒ 本档 BD≤6（再大会撞出 1024）。AIV 每轮读 C_r 的 (r+1,r+1) 格 = 偏移 17*(r+1)。
    PROBE_P6 = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    __aicore__ inline void XcoreAic()
    {
        const uint32_t bi = GetBlockIdx();
        sumGm_.SetValue(8u + bi * 8u, 9.0f);   // 按核独占 32 B 行（§15.30(e)(2)）
        if constexpr (sizeof(DT_QUERY) == 2u) {
            TBuf<TPosition::A1> bufl1;
            TBuf<TPosition::A2> bufL0A;
            TBuf<TPosition::B2> bufL0B;
            TBuf<TPosition::CO1> bufL0C;
            pipe_.InitBuffer(bufl1, 4096);
            pipe_.InitBuffer(bufL0A, 2048);
            pipe_.InitBuffer(bufL0B, 2048);
            pipe_.InitBuffer(bufL0C, 4096);
            LocalTensor<DT_QUERY> l1 = bufl1.Get<DT_QUERY>();
            const uint32_t rowD = static_cast<uint32_t>(D_);
            for (uint32_t r = 0; r < 3u; ++r) {
                Nd2NzParams nz;
                nz.ndNum = 1;
                nz.nValue = 16;
                nz.dValue = 16;
                nz.srcDValue = rowD;
                nz.dstNzC0Stride = 16;
                nz.dstNzNStride = 1;
                nz.srcNdMatrixStride = 0;
                nz.dstNzMatrixStride = 0;
                DataCopy(l1, kGm_[r * 32u * rowD], nz);              // A = key 行 32r+0..15
                DataCopy(l1[256], kGm_[(r * 32u + 16u) * rowD], nz);   // B = key 行 32r+16..31
                SetFlag<HardEvent::MTE2_MTE1>(0);
                WaitFlag<HardEvent::MTE2_MTE1>(0);
                LocalTensor<DT_QUERY> l0a = bufL0A.Get<DT_QUERY>();
                LocalTensor<DT_QUERY> l0b = bufL0B.Get<DT_QUERY>();
                LoadData2DParams ld;
                ld.startIndex = 0;
                ld.repeatTimes = 1;
                ld.srcStride = 1;
                ld.dstGap = 0;
                ld.ifTranspose = false;
                ld.sid = 0;
                ld.addrMode = 0;
                LoadData(l0a, l1, ld);
                LoadData(l0b, l1[256], ld);
                SetFlag<HardEvent::MTE1_M>(1);
                WaitFlag<HardEvent::MTE1_M>(1);
                LocalTensor<float> l0c = bufL0C.Get<float>();
                MmadParams mp;
                mp.m = 16;
                mp.n = 16;
                mp.k = 16;
                mp.cmatrixInitVal = true;
                mp.cmatrixSource = false;
                mp.unitFlag = 0b11;               // 翻转"可搬出"⇒ 本次 Fixpipe 的前提（§15.30(h)）
                Mmad(l0c, l0a, l0b, mp);
                SetFlag<HardEvent::M_FIX>(2);
                WaitFlag<HardEvent::M_FIX>(2);
                FixpipeParamsV220 fx;
                fx.nSize = 16;
                fx.mSize = 16;
                fx.srcStride = 16;
                fx.dstStride = 16;
                fx.ndNum = 1;
                fx.srcNdStride = 0;
                fx.dstNdStride = 0;
                fx.unitFlag = 0b11;
                Fixpipe(sumGm_[128u + r * 256u], l0c, fx);   // 每轮**不同落点** ⇒ echo 能带上轮次
                PipeBarrier<PIPE_ALL>();
                CrossCoreSetFlag<2, PIPE_FIX>(5);            // 广播"本轮 C 已落地"
                CrossCoreWaitFlag<2, PIPE_FIX>(6);           // 扇入：等本组两个 AIV 都读完
            }
        }
        PipeBarrier<PIPE_ALL>();
        sumGm_.SetValue(0, 7.0f);
    }

    __aicore__ inline void XcoreAiv()
    {
        const uint32_t bi = GetBlockIdx();
        for (uint32_t r = 0; r < 3u; ++r) {
            CrossCoreWaitFlag<2, PIPE_V>(5);
            const float v = sumGm_.GetValue(128u + r * 256u + 17u * (r + 1u));   // C_r[r+1][r+1]
            sumGm_.SetValue(896u + bi * 8u + r, v + 1000.0f);                    // +1000 = 已写编码
            PipeBarrier<PIPE_ALL>();
            CrossCoreSetFlag<2, PIPE_MTE3>(6);                                   // 回执（两个 AIV 都 set）
        }
        PipeBarrier<PIPE_ALL>();
        sumGm_.SetValue(960u + bi * 8u, 6.0f);
    }

'''
    s = s.replace(A_MEMBER, PROBE_P6 + A_MEMBER, 1)

elif MODE == 'cubethr':
    # ---- 4m) cubethr = **P6 的收益探针**：把 big1 的 score 工作量按真实 tile 形态压给 AIC 跑，
    #   量"Cube 做完整份 QK^T 要多少 µs"，用来决定 P6 到底值不值得动 kernel（§15.30(k)）。
    # 为什么不能只算理论值：`xcoremm3` 证的是**能不能交接**，不是**快不快**。§15.10(b) 的差分
    #   计时说 score 占 AIV 时间 71 %（big1 口径 0.9851 ms）⇒ 只要 Cube 侧显著小于它，P6 就是
    #   本项目剩下最大的一根杠杆；但若手写 Cube 链的实际吞吐差到与 AIV 同量级（每 tile 一次
    #   `Fixpipe` + `PipeBarrier` 的代价没人量过），P6 就是白改。
    # 形态（**故意取得比 P6 真实形态更悲观**，这样量出来的时间是小看 Cube 的）：
    #   每块 128 轮，每轮 = 2 次 16×512 的 ND→NZ（32 KB，**tile 之间零复用** ⇒ MTE 侧比真实
    #   形态多算 4 倍）+ 32 次 k 累加 `Mmad`（`cmatrixSource` 累加、只有最后一次 `unitFlag=0b11`
    #   翻转搬出 ⇒ 与内置 `ComputeMm1` 的 k 循环同形）+ 1 次 `Fixpipe` + 1 次 `PipeBarrier<PIPE_ALL>`。
    #   每轮 = 16×16×512 = 131 072 MAC ⇒ 128 轮 = 16.8 M MAC = big1 score 的 **1/8**（BD=8 时
    #   8 块并行 ⇒ 一次 launch 的时间就是"Cube 做完整份 score"的时间）。
    # 读数口径：看 **`--- 反复 launch ---`（`test_sfa_dev cases/big1.bin 5 none`）那一步的
    #   `时间: 平均 X ms`**（launch#1 那步被 `SFA_CUBE_EXIT=1` 提前 return，没有计时）。
    #   ⚠️ 这一档 `rc=1` 是**预期**（探针档不算算子 ⇒ harness 的 diff 必失败），只看时间行。
    PROBE_THR = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    // Cube 吞吐档：AIC 跑 128 × (16x16x512 累加 + 1 次 Fixpipe)，AIV 停工。
    __aicore__ inline void CubeProbe()
    {
        sumGm_.SetValue(0, 7.0f);
        if constexpr (sizeof(DT_QUERY) == 2u) {
            TBuf<TPosition::A1> bufl1;
            TBuf<TPosition::A2> bufL0A;
            TBuf<TPosition::B2> bufL0B;
            TBuf<TPosition::CO1> bufL0C;
            pipe_.InitBuffer(bufl1, 32768);      // 两块 16×512 的 fp16 = 32 KB
            pipe_.InitBuffer(bufL0A, 2048);
            pipe_.InitBuffer(bufL0B, 2048);
            pipe_.InitBuffer(bufL0C, 4096);
            LocalTensor<DT_QUERY> l1 = bufl1.Get<DT_QUERY>();
            const uint32_t rowD = static_cast<uint32_t>(D_);
            Nd2NzParams nz;
            nz.ndNum = 1;
            nz.nValue = 16;
            nz.dValue = 512;                     // 整宽一次搬（内置同形：dValue=srcD、C0Stride=align16(n)）
            nz.srcDValue = rowD;
            nz.dstNzC0Stride = 16;
            nz.dstNzNStride = 1;
            nz.srcNdMatrixStride = 0;
            nz.dstNzMatrixStride = 0;
            LoadData2DParams ld;
            ld.startIndex = 0;
            ld.repeatTimes = 1;
            ld.srcStride = 1;
            ld.dstGap = 0;
            ld.ifTranspose = false;
            ld.sid = 0;
            ld.addrMode = 0;
            MmadParams mp;
            mp.m = 16;
            mp.n = 16;
            mp.k = 16;
            mp.cmatrixSource = false;
            FixpipeParamsV220 fx;
            fx.nSize = 16;
            fx.mSize = 16;
            fx.srcStride = 16;
            fx.dstStride = 16;                   // 只占 sumGm_[32..271]，不碰标记区
            fx.ndNum = 1;
            fx.srcNdStride = 0;
            fx.dstNdStride = 0;
            fx.unitFlag = 0b11;
            LocalTensor<DT_QUERY> l0a = bufL0A.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> l0b = bufL0B.Get<DT_QUERY>();
            LocalTensor<float> l0c = bufL0C.Get<float>();
            for (uint32_t it = 0; it < 128u; ++it) {
                const uint32_t off = (it & 63u) * 16u * rowD;   // 两个 tile 都从 key 里取，内容不影响计时
                DataCopy(l1, kGm_[off], nz);
                DataCopy(l1[8192], kGm_[2048u * rowD + off], nz);
                SetFlag<HardEvent::MTE2_MTE1>(0);
                WaitFlag<HardEvent::MTE2_MTE1>(0);
                for (uint32_t k = 0; k < 32u; ++k) {
                    LoadData(l0a, l1[8192u + k * 256u], ld);
                    LoadData(l0b, l1[k * 256u], ld);
                    SetFlag<HardEvent::MTE1_M>(1);
                    WaitFlag<HardEvent::MTE1_M>(1);
                    mp.cmatrixInitVal = (k == 0u);
                    mp.cmatrixSource = false;               // 内置 ComputeMm1 的 k 循环恒 false（累加靠
                                                            // cmatrixInitVal=false，不是靠 source=true）
                    mp.unitFlag = (k == 31u) ? 0b11 : 0b10;   // 只有"可搬出"那一次翻转
                    Mmad(l0c, l0a, l0b, mp);
                    // ⚠️ 必须等 M 把 L0A/L0B 读走才能发下一次 LoadData：本档 32 轮复用**同一对**
                    //   L0 缓冲，真机第一次跑就是因此直接抛异常（不是挂）：
                    //   `fftsplus aicore error … errorStr: L0B read/write conflict in the MTE
                    //    (same address)`、`cube error info: 0x40400f3`（harness v8 才看得见这行）。
                    //   内置 `ComputeMm1` 用的是**双槽 + `SetFlag<HardEvent::M_MTE1>` 反向事件**
                    //   （`abL0BufIter % 2`）来免掉这道串行；这里先取悲观口径（串行 = 吞吐下界），
                    //   若下界已经足够小就不必再做双槽版。⇒ P6 的硬约束之一。
                    PipeBarrier<PIPE_M>();
                }
                SetFlag<HardEvent::M_FIX>(2);
                WaitFlag<HardEvent::M_FIX>(2);
                Fixpipe(sumGm_[32], l0c, fx);
                PipeBarrier<PIPE_ALL>();
            }
        }
        PipeBarrier<PIPE_ALL>();
        sumGm_.SetValue(0, 7.0f);
    }

'''
    s = s.replace(A_MEMBER, PROBE_THR + A_MEMBER, 1)

elif MODE == 'cubethr2':
    # ---- 4n) cubethr2 = `cubethr` 的**按内置形态修好版**：L0A/L0B **双槽** + `M_MTE1` 反向事件 ----
    # `cubethr`（单槽）真机两次都抛 `L0B read/write conflict`（§15.30(l)），而且加了
    #   `PipeBarrier<PIPE_M>()` 之后错误码从 `0x8000004000 / …in the MTE (same address)`
    #   变成 `0x4000`（不再是"同地址"）⇒ 串行化 M 只是把冲突挪了个位置，不是修好。
    # 内置 `ComputeMm1` 的做法是**两块 L0A/L0B 交替**：`Mmad` 之后 `SetFlag<HardEvent::M_MTE1>`、
    #   下一轮装同一槽之前 `WaitFlag` 它（`…service_cube_mla.h:795-814`，槽号 `abL0BufIter % 2`）。
    #   ⇒ 本档照抄这个形态，顺手把"P6 的 AIC 侧到底长什么样"也一并验证了。
    # 事件号配平纪律（§15.26/§15.30）：进入外循环**前先 `SetFlag(3)/SetFlag(4)` 各一次**当"预热"，
    #   于是循环里可以**无条件** `WaitFlag` ⇒ 每个 set 恰好被一个 wait 消费，跨 128 轮不漂移。
    # 🔴 **事件号合法范围 = 0..7**（本轮真机 + 源码双向坐实）：
    #   `impl/basic_api/kernel_macros.h:110-114`  `#if (__NPU_ARCH__ == 2201) … constexpr int32_t
    #   QUE_MAX_EVENT = 8;`，`TPipe::FetchEventID` 断言 `lastId < QUE_MAX_EVENT`，而
    #   `SetFlag/WaitFlag` 最终就是 `set_flag_dev(id)/wait_flag_dev(id)`（`dav_c220/
    #   kernel_operator_sync_impl.h`）。**同文件 114-117 另外留着 11/12/13/14 给
    #   `SyncAll`/`Barrier`**（`SYNC_AIC_FLAG=11`、`SYNC_AIV_FLAG=12`、`SYNC_AIC_AIV_FLAG=13`、
    #   `SYNC_AIV_ONLY_ALL=14`）⇒ 自研旗标既不能用 8~10（越界：wait 永远不满足 ⇒ **静默挂死**，
    #   本档第一版用 7/8 就是这个死法，rc=124 且不报任何设备错误码），也不能碰 11~14。
    #   ⇒ 可用窗口只有 **0..7**，本档占 0(MTE2→MTE1)/1(MTE1→M)/2(M→FIX)/3、4(反向双槽)/5、6(跨核)。
    # 口径：每块 128 轮 × (32 个 16×16×16 累加 + 1 次 Fixpipe) = 16.8 M MAC = big1 score 的 1/8
    #   ⇒ BD=8 时**一次 launch 的 `时间:` 就是"Cube 做完整份 score"的时间**（MTE 侧仍按零复用算，
    #   比 P6 真实形态多 4 倍 ⇒ 量出来是**偏慢**的保守值）。
    PROBE_THR2 = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    // Cube 吞吐档（双槽版）：AIC 跑 128 × (32×Mmad 累加 + 1 Fixpipe)，AIV 停工。
    __aicore__ inline void CubeProbe()
    {
        sumGm_.SetValue(0, 7.0f);
        if constexpr (sizeof(DT_QUERY) == 2u) {
            TBuf<TPosition::A1> bufl1;
            TBuf<TPosition::A2> bufA0;
            TBuf<TPosition::A2> bufA1;
            TBuf<TPosition::B2> bufB0;
            TBuf<TPosition::B2> bufB1;
            TBuf<TPosition::CO1> bufL0C;
            pipe_.InitBuffer(bufl1, 32768);      // 两块 16×512 的 fp16 = 32 KB
            pipe_.InitBuffer(bufA0, 2048);
            pipe_.InitBuffer(bufA1, 2048);
            pipe_.InitBuffer(bufB0, 2048);
            pipe_.InitBuffer(bufB1, 2048);
            pipe_.InitBuffer(bufL0C, 4096);
            LocalTensor<DT_QUERY> l1 = bufl1.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> a0 = bufA0.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> a1 = bufA1.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> b0 = bufB0.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> b1 = bufB1.Get<DT_QUERY>();
            LocalTensor<float> l0c = bufL0C.Get<float>();
            const uint32_t rowD = static_cast<uint32_t>(D_);
            Nd2NzParams nz;
            nz.ndNum = 1;
            nz.nValue = 16;
            nz.dValue = 512;
            nz.srcDValue = rowD;
            nz.dstNzC0Stride = 16;
            nz.dstNzNStride = 1;
            nz.srcNdMatrixStride = 0;
            nz.dstNzMatrixStride = 0;
            LoadData2DParams ld;
            ld.startIndex = 0;
            ld.repeatTimes = 1;
            ld.srcStride = 1;
            ld.dstGap = 0;
            ld.ifTranspose = false;
            ld.sid = 0;
            ld.addrMode = 0;
            MmadParams mp;
            mp.m = 16;
            mp.n = 16;
            mp.k = 16;
            mp.cmatrixSource = false;            // 头文件口径：1 = C 取自 bias 表，0 = C 在 L0C
            FixpipeParamsV220 fx;
            fx.nSize = 16;
            fx.mSize = 16;
            fx.srcStride = 16;
            fx.dstStride = 16;
            fx.ndNum = 1;
            fx.srcNdStride = 0;
            fx.dstNdStride = 0;
            fx.unitFlag = 0b11;
            SetFlag<HardEvent::M_MTE1>(3);       // 预热：让下面的 WaitFlag 能无条件写
            SetFlag<HardEvent::M_MTE1>(4);
            for (uint32_t it = 0; it < 128u; ++it) {
                const uint32_t off = (it & 63u) * 16u * rowD;
                DataCopy(l1, kGm_[off], nz);
                DataCopy(l1[8192], kGm_[2048u * rowD + off], nz);
                SetFlag<HardEvent::MTE2_MTE1>(0);
                WaitFlag<HardEvent::MTE2_MTE1>(0);
                for (uint32_t kp = 0; kp < 16u; ++kp) {
                    const uint32_t k0 = kp * 2u;
                    const uint32_t k1 = k0 + 1u;
                    WaitFlag<HardEvent::M_MTE1>(3);           // 槽 0 上一次的 Mmad 已读走
                    LoadData(a0, l1[8192u + k0 * 256u], ld);
                    LoadData(b0, l1[k0 * 256u], ld);
                    SetFlag<HardEvent::MTE1_M>(1);
                    WaitFlag<HardEvent::MTE1_M>(1);
                    mp.cmatrixInitVal = (k0 == 0u);
                    mp.unitFlag = 0b10;
                    Mmad(l0c, a0, b0, mp);
                    // ⚠️ `PipeBarrier<PIPE_M>` 必须在**反向 set 之前**：`SetFlag<M_MTE1>` 的语义是
                    //   "M 流水**走到**这条指令"，不是"之前的 Mmad 已经真的读完 L0B"。小 tile
                    //   （m/16*n/16 < 10）时排队深度会让 set 早于完成 ⇒ 下一轮重装打正在读的槽。
                    //   内置在完全相同的位置就是这么做死的：`…service_cube_mla.h:810-813`
                    //   `if ((mmadParams.m/16)*(mmadParams.n/16) < 10) { PipeBarrier<PIPE_M>(); }`
                    //                    //   紧接着 `SetFlag<HardEvent::M_MTE1>`。本档 16×16 ⇒ 条件恒真 ⇒ 无条件加。
                    PipeBarrier<PIPE_M>();
                    SetFlag<HardEvent::M_MTE1>(3);
                    WaitFlag<HardEvent::M_MTE1>(4);           // 槽 1 同理
                    LoadData(a1, l1[8192u + k1 * 256u], ld);
                    LoadData(b1, l1[k1 * 256u], ld);
                    SetFlag<HardEvent::MTE1_M>(1);
                    WaitFlag<HardEvent::MTE1_M>(1);
                    mp.cmatrixInitVal = false;
                    mp.unitFlag = (k1 == 31u) ? 0b11 : 0b10;  // 只有"可搬出"那一次翻转
                    Mmad(l0c, a1, b1, mp);
                    PipeBarrier<PIPE_M>();                  // 同上：set 只代表"M 走到这里"
                    SetFlag<HardEvent::M_MTE1>(4);
                }
                SetFlag<HardEvent::M_FIX>(2);
                WaitFlag<HardEvent::M_FIX>(2);
                Fixpipe(sumGm_[32], l0c, fx);
                PipeBarrier<PIPE_ALL>();
            }
            // ⚠️ 收口：循环里 set 比 wait 多出的那两次（每次迭代末尾各 set 一次 3/4，最后一轮的
            //   消费点在循环外）在这里吃掉 ⇒ **事件计数跨 launch 零残留**。§15.30(j) 的
            //   "可反复 launch"性质就靠这类配平撑着，留余量会让下一次 launch 的第一次 wait 假通过。
            WaitFlag<HardEvent::M_MTE1>(3);
            WaitFlag<HardEvent::M_MTE1>(4);
        }
        PipeBarrier<PIPE_ALL>();
        sumGm_.SetValue(0, 7.0f);
    }

'''
    s = s.replace(A_MEMBER, PROBE_THR2 + A_MEMBER, 1)

elif MODE == 'cubethr3':
    # ---- 4o) cubethr3 = **P6 真实形态的吞吐档**：一轮 = 一次成对 `LoadData`（16×512）+
    #   4 个 `Mmad(k=128)` 累加进同一块 `l0c` + 1 次 `Fixpipe`。
    # 为什么这一档才是"要量的那个数"（`cubethr`/`cubethr2` 都在问别的问题）：
    #   ① 那两档每轮发 **32 次** `Mmad(16×16×16)`，而 16×16×16 是 Cube 阵列**最小**的一笔 ⇒
    #      指令数占满、单元利用率最低，量出来是 Cube 的最差面，不是 P6 会落到的那一面。
    #   ② 要"一轮 32 次 Mmad 复用同一对 L0A/L0B"就必须每轮重装 ⇒ 正撞 §15.30(l) 的
    #      `L0B read/write conflict`，而修它的反向事件在本档因旗标号越界（§15.30(n)）挂死。
    #   ③ 本档**一轮只装一次** L0（16×512 = 4 个 k 切片），4 个 `Mmad` 全是**读**同一对 L0 ⇒
    #      一轮之内没有任何 L0 覆写 ⇒ 一个反向事件都不需要；跨轮覆写由每轮末尾的
    #      `PipeBarrier<PIPE_ALL>` 兜住 —— 这正是 `cubemmad3`/`xcoremm3` 已在真机上跑通的骨架，
    #      本档只是把轮数 3 → 128、每轮 k 16 → 512。
    # k 切片的宽度 128 不是随手取的：内置 arch22 SFA 的 `kL0Size` 就是 128
    #   （`…service_cube_mla.h:868/904`）⇒ 2201 上 `Mmad(m=16,n=16,k=128)` 是官方在用、
    #   确定合法的形状，不用去赌 `k=512` 的合法性。
    # 口径：每块 128 轮 × 16×16×512 = **16.8 M MAC** = big1 score 的 1/8 ⇒ BD=8 时一次 launch
    #   的 `时间:` 就是"Cube 做完整份 score"的时间。仍比 P6 真实形态**悲观**两处：
    #   MTE 侧零复用（多 4 倍搬运）+ `m=n=16`（真实形态是 128×64 一类的大 tile，单位 MAC 的
    #   `Fixpipe`/`LoadData` 摊得更薄）。⇒ 本档时间 = Cube 用量的**上界**。
    PROBE_THR3 = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    // Cube 吞吐档（真实形态）：AIC 跑 128 × (16×512 装载 + 4×Mmad(k=128) 累加 + 1 Fixpipe)。
    __aicore__ inline void CubeProbe()
    {
        sumGm_.SetValue(0, 7.0f);
        if constexpr (sizeof(DT_QUERY) == 2u) {
            TBuf<TPosition::A1> bufl1;
            TBuf<TPosition::A2> bufL0A;
            TBuf<TPosition::B2> bufL0B;
            TBuf<TPosition::CO1> bufL0C;
            pipe_.InitBuffer(bufl1, 32768);      // 两块 16×512 的 fp16 = 32 KB
            pipe_.InitBuffer(bufL0A, 16384);     // 16×512 fp16
            pipe_.InitBuffer(bufL0B, 16384);
            pipe_.InitBuffer(bufL0C, 4096);      // 16×16 float
            LocalTensor<DT_QUERY> l1 = bufl1.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> l0a = bufL0A.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> l0b = bufL0B.Get<DT_QUERY>();
            LocalTensor<float> l0c = bufL0C.Get<float>();
            const uint32_t rowD = static_cast<uint32_t>(D_);
            Nd2NzParams nz;
            nz.ndNum = 1;
            nz.nValue = 16;
            nz.dValue = 512;
            nz.srcDValue = rowD;
            nz.dstNzC0Stride = 16;
            nz.dstNzNStride = 1;
            nz.srcNdMatrixStride = 0;
            nz.dstNzMatrixStride = 0;
            LoadData2DParams ld;
            ld.startIndex = 0;
            ld.repeatTimes = 32;                 // 内置口径：kSize / (32/sizeof(fp16)) = 512/16
            ld.srcStride = 1;
            ld.dstGap = 0;
            ld.ifTranspose = false;
            ld.sid = 0;
            ld.addrMode = 0;
            MmadParams mp;
            mp.m = 16;
            mp.n = 16;
            mp.k = 128;                          // = 内置 kL0Size
            mp.cmatrixSource = false;
            FixpipeParamsV220 fx;
            fx.nSize = 16;
            fx.mSize = 16;
            fx.srcStride = 16;
            fx.dstStride = 16;                   // 只占 sumGm_[32..287]，不碰标记区
            fx.ndNum = 1;
            fx.srcNdStride = 0;
            fx.dstNdStride = 0;
            fx.unitFlag = 0b11;
            for (uint32_t it = 0; it < 128u; ++it) {
                const uint32_t off = (it & 63u) * 16u * rowD;
                DataCopy(l1, kGm_[off], nz);
                DataCopy(l1[8192], kGm_[2048u * rowD + off], nz);
                SetFlag<HardEvent::MTE2_MTE1>(0);
                WaitFlag<HardEvent::MTE2_MTE1>(0);
                LoadData(l0a, l1[8192u], ld);
                LoadData(l0b, l1[0u], ld);
                SetFlag<HardEvent::MTE1_M>(1);
                WaitFlag<HardEvent::MTE1_M>(1);
                for (uint32_t j = 0; j < 4u; ++j) {
                    mp.cmatrixInitVal = (j == 0u);          // 第一刀清 C，后面三刀在 L0C 上累加
                    mp.unitFlag = (j == 3u) ? 0b11 : 0b10;  // 只有"可搬出"那一刀翻转搬出标志
                    // 第 j 刀的 k 切片 = L0 里第 j*8 个 16×16 块起（元素偏移 j*8*256 = j*2048）
                    Mmad(l0c, l0a[j * 2048u], l0b[j * 2048u], mp);
                }
                SetFlag<HardEvent::M_FIX>(2);
                WaitFlag<HardEvent::M_FIX>(2);
                Fixpipe(sumGm_[32], l0c, fx);
                PipeBarrier<PIPE_ALL>();       // 兜住"下一轮覆写 L0A/L0B"（本轮内全是读）
            }
        }
        PipeBarrier<PIPE_ALL>();
        sumGm_.SetValue(0, 7.0f);
    }

'''
    s = s.replace(A_MEMBER, PROBE_THR3 + A_MEMBER, 1)

elif MODE in ('cubethr4', 'cubethr5'):
    # ---- 4p) cubethr4/5 = **把 M 流水从 MTE 里剥出来单量**（§15.30(o) 的关键一格）----
    # 为什么 `cubethr3` 的 0.1321 ms **不能**直接当"Cube 做整份 score 的时间"：两重错配 ——
    #   ① 量纲：harness 的 MAC 估算式（`test_sfa_dev.cpp:373-375`，`B*S1*N1*SBS*min(COUNT,S2)*(2D+64)`）
    #      对 big1 给 2.282e9 ⇒ **score 那半边 = 1.074 G MAC**（tokens = SBS·COUNT = 2048，不是 256）。
    #      本探针每块 128 轮 × 131 072 = 16.8 M、8 块 = **134 M = 真 score 的 1/8**。
    #   ② 配比：本档每轮"搬 32 KB GM→L1 只做 131 072 MAC"= **0.25 B/MAC**，而真实 score 的
    #      GM 流量 ≈ K(8头×2048×512×2B) + Q ≈ 18 MB / 1.074 G = **0.017 B/MAC ⇒ 薄 15 倍**。
    #      ⇒ 0.132 ms 里 MTE 是主角，Cube 阵列根本没被压满（粗算 8 块 × 16.8 M / 0.132 ms
    #        = 每 AIC 核 254 GFLOP/s，对着单核十几 TFLOP/s 的峰值是 **~2 %**）。
    # ⇒ 这一档把复用做到极致：**一次装载 L0（16×512）后连打 ROUNDS 轮 Mmad**，L0A/L0B 全程
    #    **只读不写** ⇒ 一个反向事件都不需要（§15.30(n) 那条旗标坑绕开），MTE 侧只占开头一次
    #    32 KB 搬运 ⇒ 量到的就是"M 流水 + Fixpipe + 每轮同步"的净时间。
    # 轮数取得让**总量正好等于 big1 的真 score**：每块 1024 轮 × 4 × (16×16×128) = 134.2 M，
    #   BD=8 ⇒ **1.074 G MAC = 整份 score** ⇒ 屏幕上那个 `时间:` 就是"Cube 换掉 AIV 做 score"
    #   要花的钱，不用再做任何外推（判据：≪ 0.56 ms（§15.10(b) 的 score 占比）⇒ P6 净赚）。
    # 两档只差一处，用来分"Mmad 自己有多快"和"每轮一次 Fixpipe 有多贵"：
    #   cubethr4 = **P6 的真实形状**：每轮 4 刀累加 + 1 次 `Fixpipe`（1024 次搬出，与真实
    #              score 的 Fixpipe 次数逐一对上：2.097 M 输出元素 / 256 = 8192 次 / 8 核 = 1024）。
    #   cubethr5 = Mmad 刀数**完全相同**，但把 `Fixpipe` 整条摘到循环外只做一次 ⇒ 4 与 5 之差
    #              = "每轮搬出"的代价，也是 P6 该不该把 tile 开大（一次 Fixpipe 覆盖 128×64）的依据。
    # ⚠️ 合法性：§15.30(h) 禁的是"同一块 `l0c` 没有自己的 `Mmad` 还被第 2 次 `Fixpipe` 读走"；
    #    本档每轮**都**先重发 4 刀（最后一刀 `unitFlag=0b11`）再搬出 ⇒ 与 `cubemmad3` 同形。
    _THRV = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    // Cube 净吞吐档：装载一次 L0，连打 1024 × (4 刀 Mmad(k=128) 累加 [+ 1 次 Fixpipe])。
    __aicore__ inline void CubeProbe()
    {
        sumGm_.SetValue(0, 7.0f);
        if constexpr (sizeof(DT_QUERY) == 2u) {
            TBuf<TPosition::A1> bufl1;
            TBuf<TPosition::A2> bufL0A;
            TBuf<TPosition::B2> bufL0B;
            TBuf<TPosition::CO1> bufL0C;
            pipe_.InitBuffer(bufl1, 32768);
            pipe_.InitBuffer(bufL0A, 16384);
            pipe_.InitBuffer(bufL0B, 16384);
            pipe_.InitBuffer(bufL0C, 4096);
            LocalTensor<DT_QUERY> l1 = bufl1.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> l0a = bufL0A.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> l0b = bufL0B.Get<DT_QUERY>();
            LocalTensor<float> l0c = bufL0C.Get<float>();
            const uint32_t rowD = static_cast<uint32_t>(D_);
            Nd2NzParams nz;
            nz.ndNum = 1;
            nz.nValue = 16;
            nz.dValue = 512;
            nz.srcDValue = rowD;
            nz.dstNzC0Stride = 16;
            nz.dstNzNStride = 1;
            nz.srcNdMatrixStride = 0;
            nz.dstNzMatrixStride = 0;
            LoadData2DParams ld;
            ld.startIndex = 0;
            ld.repeatTimes = 32;
            ld.srcStride = 1;
            ld.dstGap = 0;
            ld.ifTranspose = false;
            ld.sid = 0;
            ld.addrMode = 0;
            MmadParams mp;
            mp.m = 16;
            mp.n = 16;
            mp.k = 128;                          // = 内置 kL0Size（2201 上官方在用的形状）
            mp.cmatrixSource = false;
            FixpipeParamsV220 fx;
            fx.nSize = 16;
            fx.mSize = 16;
            fx.srcStride = 16;
            fx.dstStride = 16;
            fx.ndNum = 1;
            fx.srcNdStride = 0;
            fx.dstNdStride = 0;
            fx.unitFlag = 0b11;
            DataCopy(l1, kGm_[0], nz);
            DataCopy(l1[8192], kGm_[2048u * rowD], nz);
            SetFlag<HardEvent::MTE2_MTE1>(0);
            WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData(l0a, l1[8192u], ld);        // ★ 全程只有这一次 L0 覆写
            LoadData(l0b, l1[0u], ld);
            constexpr bool kOut = (@FIXMODE@ == 4);      // thr4 每轮搬出；thr5 只在最后一轮翻一次
            SetFlag<HardEvent::MTE1_M>(1);
            WaitFlag<HardEvent::MTE1_M>(1);
            for (uint32_t it = 0; it < 1024u; ++it) {
                for (uint32_t j = 0; j < 4u; ++j) {
                    mp.cmatrixInitVal = (j == 0u);
                    mp.unitFlag = ((j == 3u) && (kOut || it == 1023u)) ? 0b11 : 0b10;
                    Mmad(l0c, l0a[j * 2048u], l0b[j * 2048u], mp);
                }
                @FIX@
            }
            @TAIL@
        }
        PipeBarrier<PIPE_ALL>();
        sumGm_.SetValue(0, 7.0f);
    }

'''
    FIX_IN = ('                SetFlag<HardEvent::M_FIX>(2);\n'
              '                WaitFlag<HardEvent::M_FIX>(2);\n'
              '                Fixpipe(sumGm_[32], l0c, fx);   // [thr4] 每轮搬出一次\n')
    TAIL_OUT = ('            SetFlag<HardEvent::M_FIX>(2);\n'
                '            WaitFlag<HardEvent::M_FIX>(2);\n'
                '            Fixpipe(sumGm_[32], l0c, fx);   // [thr5] 全程只搬出这一次\n')
    if MODE == 'cubethr4':
        body = _THRV.replace('@FIXMODE@', '4').replace('@FIX@', FIX_IN).replace('@TAIL@', '')
    else:
        # thr5 的循环体**一道同步都不发**：L0A/L0B 全程只读、L0C 的读写全在 M 流水上按序 ⇒ 这一档
        # 就是"M 流水连打 4096 刀要多少时间"，与 thr4 之差 = 每轮搬出的代价。
        body = _THRV.replace('@FIXMODE@', '5').replace('@FIX@', '').replace('@TAIL@', TAIL_OUT)
    s = s.replace(A_MEMBER, body + A_MEMBER, 1)

elif MODE == 'cubethr6':
    # ---- 4q) cubethr6 = **P6 的真实 tile 形状**：`thr4` 把 `thr5` 贵 10 倍的那口锅（16×16 的
    #   小搬出）指出来了，这一档按"一次 `Fixpipe` 搬一整块 128×64"重量同一个 1.074 G MAC。
    # 真机前读数（`cube_probe_thr45_155128.log`，BD=8）：
    #   thr5（4096 刀 Mmad、全程 1 次 Fixpipe）= **0.0714 ms**（批量 0.0483）
    #   thr4（**同样多**的刀、每轮 1 次 16×16 Fixpipe）= **0.7866 ms**（批量 0.7617）
    #   ⇒ M 流水本身跑完整份 score 只要 0.05~0.07 ms（AIV 那边 §15.10(b) 是 ~0.56 ms），
    #      而**搬出**一个人吃掉 0.715 ms = 每核 1024 次 / 每次 ~0.7 µs。⇒ 裁定 P6 的唯一的
    #      未知数就是"把 1024 次 16×16 换成 32 次 128×64 之后，这笔还剩下多少"。
    # 形状（每一轮 = P6 的一个 score tile，与内置 `ComputeMm1` 同构）：
    #   A 片 = Q 的 128 行 × 512 维，**循环外一次装、32 轮全程复用**（真实形态就是 Q 沿 n 复用）
    #   B 片 = K 的 512 × 64，每轮一次 GM→L1（64 KB）⇒ 每核 32 轮 = 2 MB、8 核 16 MB ≈
    #          big1 真 K 流量 8头×2048×512×2B = 16.8 MB ⇒ **MTE 侧不再是零复用假账**
    #   每轮 4 刀 `Mmad(m=128,n=64,k=128)`（= 内置 `M_SPLIT_SIZE=128` / `kL0Size=128` 的合法形状）
    #   + 1 次 `Fixpipe(mSize=128,nSize=64)` 落到 `vGm_`（16 KB/轮，sumGm_ 只有 1024 格放不下）
    # 同步：L0A/L0B **每轮覆写** ⇒ 沿用真机验过的 `PipeBarrier<PIPE_ALL>` 收口（thr3 那套）；
    #   ⚠️ 这里取**每刀一次**而不是每轮一次，因为同一轮内 j+1 刀的 `LoadData` 会盖 j 刀正在读的
    #   L0 —— §15.30(l) 的 `L0B read/write conflict`。代价 = 这一档仍是**偏慢**的下界口径。
    # 判据（同 thr4/5 的口径）：这一档 ≪ 0.56 ms ⇒ P6 动手；同量级或更大 ⇒ Cube 线按
    #   "M 流水够快、但 arch22 的 L0C→GM 搬出扛不动"结案，转任务 #24（P15 沿 D 轴切块）。
    PROBE_THR6 = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    // Cube 真实 tile 档：32 × (4 刀 Mmad(128×64×128) + 1 次 Fixpipe(128×64))，8 核合计 1.074 G MAC。
    __aicore__ inline void CubeProbe()
    {
        sumGm_.SetValue(0, 7.0f);
        if constexpr (sizeof(DT_QUERY) == 2u) {
            TBuf<TPosition::A1> bufl1;
            TBuf<TPosition::A2> bufL0A;
            TBuf<TPosition::B2> bufL0B;
            TBuf<TPosition::CO1> bufL0C;
            pipe_.InitBuffer(bufl1, 196608);     // A 片 128 KB + B 片 64 KB
            pipe_.InitBuffer(bufL0A, 32768);     // 128×128 fp16
            pipe_.InitBuffer(bufL0B, 16384);     // 128×64 fp16
            pipe_.InitBuffer(bufL0C, 32768);     // 128×64 fp32
            LocalTensor<DT_QUERY> l1a = bufl1.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> l1b = l1a[65536];
            LocalTensor<DT_QUERY> l0a = bufL0A.Get<DT_QUERY>();
            LocalTensor<DT_QUERY> l0b = bufL0B.Get<DT_QUERY>();
            LocalTensor<float> l0c = bufL0C.Get<float>();
            const uint32_t rowD = static_cast<uint32_t>(D_);
            Nd2NzParams nzA;                     // A：128 行 × 512 列，整片一次
            nzA.ndNum = 1;
            nzA.nValue = 128;
            nzA.dValue = 512;
            nzA.srcDValue = rowD;
            nzA.dstNzC0Stride = 128;
            nzA.dstNzNStride = 1;
            nzA.srcNdMatrixStride = 0;
            nzA.dstNzMatrixStride = 0;
            Nd2NzParams nzB;                     // B：512 行（k）× 64 列（n）
            nzB.ndNum = 1;
            nzB.nValue = 512;
            nzB.dValue = 64;
            nzB.srcDValue = rowD;
            nzB.dstNzC0Stride = 512;
            nzB.dstNzNStride = 1;
            nzB.srcNdMatrixStride = 0;
            nzB.dstNzMatrixStride = 0;
            LoadData2DParams ldA;
            ldA.startIndex = 0;
            ldA.repeatTimes = 64;                // 128×128 fp16 = 64 个 16×16 块
            ldA.srcStride = 1;
            ldA.dstGap = 0;
            ldA.ifTranspose = false;
            ldA.sid = 0;
            ldA.addrMode = 0;
            LoadData2DParams ldB;
            ldB.startIndex = 0;
            ldB.repeatTimes = 32;                // 128×64 fp16 = 32 块
            ldB.srcStride = 1;
            ldB.dstGap = 0;
            ldB.ifTranspose = false;
            ldB.sid = 0;
            ldB.addrMode = 0;
            MmadParams mp;
            mp.m = 128;
            mp.n = 64;
            mp.k = 128;
            mp.cmatrixSource = false;
            FixpipeParamsV220 fx;
            fx.nSize = 64;
            fx.mSize = 128;
            fx.srcStride = 128;                  // L0C 里 m 方向的行距（= m 对齐值）
            fx.dstStride = 128;                  // GM 里两列之间隔 128 个元素
            fx.ndNum = 1;
            fx.srcNdStride = 0;
            fx.dstNdStride = 0;
            fx.unitFlag = 0b11;
            DataCopy(l1a, kGm_[0], nzA);
            SetFlag<HardEvent::MTE2_MTE1>(0);
            WaitFlag<HardEvent::MTE2_MTE1>(0);
            for (uint32_t r = 0; r < 32u; ++r) {
                DataCopy(l1b, kGm_[2048u * rowD + (r & 3u) * 512u * rowD], nzB);
                SetFlag<HardEvent::MTE2_MTE1>(0);
                WaitFlag<HardEvent::MTE2_MTE1>(0);
                for (uint32_t j = 0; j < 4u; ++j) {
                    LoadData(l0a, l1a[j * 16384u], ldA);
                    LoadData(l0b, l1b[j * 8192u], ldB);
                    SetFlag<HardEvent::MTE1_M>(1);
                    WaitFlag<HardEvent::MTE1_M>(1);
                    mp.cmatrixInitVal = (j == 0u);
                    mp.unitFlag = (j == 3u) ? 0b11 : 0b10;
                    Mmad(l0c, l0a, l0b, mp);
                    PipeBarrier<PIPE_ALL>();     // 下一刀的 LoadData 会覆写这一刀正在读的 L0
                }
                SetFlag<HardEvent::M_FIX>(2);
                WaitFlag<HardEvent::M_FIX>(2);
                Fixpipe(vGm_[r * 8192u], l0c, fx);
                PipeBarrier<PIPE_ALL>();
            }
        }
        PipeBarrier<PIPE_ALL>();
        sumGm_.SetValue(0, 7.0f);
    }

'''
    s = s.replace(A_MEMBER, PROBE_THR6 + A_MEMBER, 1)

elif MODE == 'xcorec':
    # ---- 4c) xcorec：P6 真正要的**原语** = 同一对旗标在一次 launch 内**往返 3 轮** +
    #   AIV→AIC 走**扇入**（本组两个 AIV 都 set，AIC 一次 wait 才醒）。
    #   依据（§15.26(b)）：`xcoreb` 卡死的因 = 只让偶数号 AIV 发回执 ⇒ 扇入永远凑不齐。
    #   这一档一次钉掉 P6 依赖的三件事：
    #     ① 扇入修法成立（两个 AIV 都 set ⇒ AIC 醒，不再挂）；
    #     ② 旗标计数**不因轮次漂移**（3 轮复用同一对 5/6 ⇒ 若 set/wait 配不平，第 2 轮就死）；
    #     ③ **跨 launch 不漂移**（harness 的 1 + 5 reps + 20×3 批量 ⇒ 上一次任务不许留余量）。
    #   出口（softmax_sum_out 的 float 视图，big1 有 1024 格 ⇒ BD≤16）：
    #     [0]=7.0 AIC 走完全部轮   [8+bi]=9.0 AIC 到达   [760+bi]=6.0 AIV 到达
    #     [600+bi*4+r] = AIV 第 r 轮从本组 GM 分片读回的值（期望 3.0+r ⇒ 数据与**顺序**都对；
    #                    读到 5.0 = AIC 提前覆盖了这一轮 ⇒ 回执里缺一道 PipeBarrier）
    PROBE_XC3 = r'''
    // ==================== [CUBEPROBE] 远端探针专用（绝不进提交源） ====================
    // 3 轮 ping-pong，同一对旗标 (5: AIC→AIV 广播, 6: AIV→AIC 扇入)，暂存走本组独占的
    // softmax_max_out 分片（每块 64 格）——**完全不碰 workspace**（§15.24 已判死那条路）。
    // ⚠️ 轮次标签一律查表：aicore 里 `static_cast<float>(uint32_t)` 直接被编译器拒绝
    //    （`cast between floating and unsigned integer variable is not allowed in aicore function`，
    //     真机实证）⇒ 想要"每轮一个可区分的值"只能用 float 常量数组。
    __aicore__ inline void XcoreAic()
    {
        static constexpr float RV[3] = {3.0f, 4.0f, 5.0f};
        const uint32_t bi = GetBlockIdx();
        const uint32_t base = 64u * bi;
        sumGm_.SetValue(8u + (bi & 15u), 9.0f);
        for (uint32_t r = 0; r < 3u; ++r) {
            maxGm_.SetValue(base + r, RV[r]);
            PipeBarrier<PIPE_ALL>();
            CrossCoreSetFlag<2, PIPE_FIX>(5);       // 广播：本组两个 AIV 都醒
            CrossCoreWaitFlag<2, PIPE_FIX>(6);      // 扇入：两个 AIV 都 set 才醒
        }
        PipeBarrier<PIPE_ALL>();
        sumGm_.SetValue(0, 7.0f);
    }

    __aicore__ inline void XcoreAiv()
    {
        const uint32_t bi = GetBlockIdx();
        const uint32_t g = bi >> 1;                 // §15.26(a)：AIV bi = group*2 + sub
        for (uint32_t r = 0; r < 3u; ++r) {
            CrossCoreWaitFlag<2, PIPE_V>(5);
            const float v = maxGm_.GetValue(64u * g + r);
            PipeBarrier<PIPE_ALL>();                // 读落地之后再回执，否则下一轮会覆盖
            sumGm_.SetValue(600u + bi * 4u + r, v);
            CrossCoreSetFlag<2, PIPE_MTE3>(6);      // 两个 AIV 都 set ⇒ 凑齐一次扇入
        }
        sumGm_.SetValue(760u + bi, 6.0f);
    }

'''
    s = s.replace(A_MEMBER, PROBE_XC3 + A_MEMBER, 1)
elif MODE in XCORE:
    s = s.replace(A_MEMBER, (PROBE_XC % (
        'CrossCoreWaitFlag<2, PIPE_FIX>(6);' if MODE == 'xcoreb' else '(void)0;',
        'if ((bi & 1u) == 0u) { PipeBarrier<PIPE_ALL>(); CrossCoreSetFlag<2, PIPE_MTE3>(6); }'
        if MODE == 'xcoreb' else '(void)0;')) + A_MEMBER, 1)
elif MODE in ('launch', 'shim', 'wspeek', 'wsnw'):
    s = s.replace(A_MEMBER, PROBE_MIN + A_MEMBER, 1)
elif MODE == 'mmad':
    s = s.replace(A_MEMBER, PROBE + A_MEMBER, 1)

sys.stdout.write(s)
