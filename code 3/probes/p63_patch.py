#!/usr/bin/env python3
# P63 遥测补丁（只打在远端 ~/sfa_real 的副本上，本地提交源绝不碰）。
#
# 到 p61 为止的实测把挂点收窄成一件事：**两侧各自到底跑了几个片**。
#   free6   通  ：门关 + AIV 不等 READY + 两颗都 set 常量 6 + 排空只 wait 一次
#   nostale 挂 ：同 free6 但一个 set 都不发 ⇒ free6 的"通"是真到达，不是残留
#   cred6   挂 ：完整协议，credit 线挪到 6/8
#   dbl     挂 ：完整协议，两颗共用常量 6、每片等两次
#   free5   挂 ：门关 + AIV 不等 READY，但 set 用运行时号 5/7、排空只 wait(5)
# 而 header 已经把块号口径钉死（AIV 的 GetBlockIdx = group*2+sub），官方
# dav_c220/kernel_operator_sync_impl.h:158-165 也证明 AIV→AIC 用 mode 2 + 常量号
# 是通的（框架自己的 EARLY_START_AIV_TO_AIC 就是 set(PIPE_MTE3, msg(2,12)) /
# 对侧 wait_flag_dev(12)）。⇒ 剩下的候选只有"片数对不上"。
#
# 这一档不再猜：把两侧的累计片数、单元数、块号写进 GM 读回来。
#   · AIC 侧的 credit 等待整个关掉（门关 + 排空删掉）⇒ 一定能收尾；
#   · AIV 侧保留【自然协议】（等 READY、按 sub_ 分号交 credit），片数就是真片数；
#   · 遥测槽 = 本单元那一行 attention_out 的尾部 256 个元素：环每片只写一行前
#     16*nTile ≤ 768 个元素（nTile ≤ n_blk ≤ 48），尾 256 个两侧都不碰，而
#     attention_out 是输出张量 ⇒ harness 的 write 模式会把它落盘（query 是输入，
#     落不了盘，所以槽选 outGm_）。
#     槽内布局：AIV sub0 记录 [0,16)、sub1 [16,32)、AIC [64,80)，marker 77/77/78。
import io
import os
import sys

SRC = os.environ.get('P63_SRC', '/tmp/kernel_p63_orig.cpp')
DST = os.environ.get('P63_DST', 'code/op_kernel/sparse_flash_attention.cpp')

src = io.open(SRC, encoding='utf-8').read()


def rep(t, a, b, n):
    assert t.count(a) == 1, 'anchor %s hits %d times' % (n, t.count(a))
    return t.replace(a, b, 1)


# ---- 1) 成员 ----
MEMB = '    uint32_t aChunk_ = 0u;'
t = rep(src, MEMB, MEMB + '\n'
        '    uint32_t telRow_ = 0u;    // [p63]\n'
        '    uint32_t telUnits_ = 0u;  // [p63]\n'
        '    uint32_t telAicU_ = 0u;   // [p63]\n'
        '    uint32_t telSets_ = 0u;   // [p63]', 'memb')

# ---- 2) 遥测函数（挂在 Process 前面，类内前向无碍）----
PROC = '    __aicore__ inline void Process()\n    {'
TEL = r'''
    // ---- [p63] 遥测：只在远端探针副本里存在，判据是记录内容而不是数值 ----
    __aicore__ inline void TelWrite(uint32_t off, const uint32_t *rec)
    {
        // ⚠️ aicore 禁 "无符号 ↔ 浮点" 直接转（实测报 cast between floating and unsigned
        //    integer is not allowed），但**有符号 -> 浮点**是通的（本文件 Init 里
        //    `static_cast<float>(GetBlockIdx())` 就是这个形状）⇒ 中间垫一次 int32。
        for (uint32_t i = 0u; i < 16u; ++i) {
            outGm_.SetValue(static_cast<int64_t>(off + i),
                            static_cast<DT_QUERY>(static_cast<float>(static_cast<int32_t>(rec[i]))));
        }
    }

    __aicore__ inline uint32_t TelTailBase(uint32_t row)
    {
        const uint32_t rowW = static_cast<uint32_t>(N1_) * static_cast<uint32_t>(D_);
        return row + rowW - 256u;
    }

    __aicore__ inline void TelDumpAiv()
    {
        if (!cubeOn_ || telUnits_ == 0u) { return; }
        uint32_t rec[16];
        rec[0] = 77u;
        rec[1] = static_cast<uint32_t>(GetBlockIdx());
        rec[2] = static_cast<uint32_t>(GetBlockNum());
        rec[3] = static_cast<uint32_t>(GetSubBlockIdx());
        rec[4] = sub_;
        rec[5] = aChunk_;
        rec[6] = telUnits_;
        rec[7] = unitBegin_;
        rec[8] = unitEnd_;
        rec[9] = unitStep_;
        rec[10] = headBase_;
        rec[11] = static_cast<uint32_t>(nb_);
        rec[12] = static_cast<uint32_t>(N1_);
        rec[13] = static_cast<uint32_t>(nBlk_);
        rec[14] = static_cast<uint32_t>(sparseCount_);
        rec[15] = telSets_;          // 真正走到 credit set 那一行的次数
        TelWrite(TelTailBase(telRow_) + sub_ * 16u, rec);
    }

    __aicore__ inline void TelDumpAic()
    {
        if (telAicU_ == 0u) { return; }
        uint32_t rec[16];
        rec[0] = 78u;
        rec[1] = static_cast<uint32_t>(GetBlockIdx());
        rec[2] = static_cast<uint32_t>(GetBlockNum());
        rec[3] = static_cast<uint32_t>(GetSubBlockIdx());
        rec[4] = cstep_;
        rec[5] = cp_;
        rec[6] = telAicU_;
        rec[7] = cu_;
        rec[8] = nHeadBlk_;
        rec[9] = static_cast<uint32_t>(N1_);
        rec[10] = static_cast<uint32_t>(nBlk_);
        rec[11] = static_cast<uint32_t>(sparseCount_);
        rec[12] = 0u;
        rec[13] = 0u;
        rec[14] = 0u;
        rec[15] = static_cast<uint32_t>(D_);
        TelWrite(TelTailBase(cs1_) + 64u, rec);
    }

'''
t = rep(t, PROC, TEL + PROC, 'proc')

# ---- 3) AIV 侧：记下行基址 + 单元数；收尾打一份 ----
PT0 = '        const uint32_t s1Base   = static_cast<uint32_t>(((static_cast<uint64_t>(b) * S1_ + s) * N1_) * D_);'
t = rep(t, PT0, PT0 + '\n        telRow_ = s1Base; ++telUnits_;   // [p63]', 'pt')
KRET = '        if (ks_ < 2u) { return; }'
t = rep(t, KRET, '        TelDumpAiv();   // [p63]\n' + KRET, 'aiv')

# ---- 4) AIC 侧：数单元 + 收尾打一份；同时把 credit 等待整个关掉 ----
LOOP = ('            for (; cu_ < total; cu_ += cstep_) {\n'
        '                CubeUnitBegin(ctx);\n'
        '                while (CubeOneChunk(ctx)) { }\n'
        '            }')
t = rep(t, LOOP, ('            for (; cu_ < total; cu_ += cstep_) {\n'
                  '                CubeUnitBegin(ctx);\n'
                  '                while (CubeOneChunk(ctx)) { }\n'
                  '                ++telAicU_;   // [p63]\n'
                  '            }'), 'loop')

DRAIN = ('            const uint32_t d = (cp_ < sfa::SFA_RING) ? cp_ : sfa::SFA_RING;\n'
         '            for (uint32_t i = 0u; i < d; ++i) {\n'
         '                CrossCoreWaitFlag<2, PIPE_S>(sfa::CF_CRED);\n'
         '                CrossCoreWaitFlag<2, PIPE_S>(sfa::CF_CRED + 2u);\n'
         '            }')
t = rep(t, DRAIN, '            TelDumpAic();   // [p63] 排空整个关掉：这一档只看片数', 'drain')

CSET = '            CrossCoreSetFlag<2, PIPE_MTE3>(sfa::CF_CRED + 2u * sub_);'
t = rep(t, CSET, '            ++telSets_;   // [p63]\n' + CSET, 'cset')

GATE = '        if (cp_ >= sfa::SFA_RING) {'
t = rep(t, GATE, '        if (false) {   // [p63] 门关', 'gate')

# AIV 不再等 READY：片数由【自己的扫描】决定，与 READY 的到达次数无关，所以这一改动
# 不动被测的量，只保证"哪怕两侧片数不等这一档也能收尾"（收不了尾就一条记录都读不到）。
R_WAIT = '        CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_READY);'
t = rep(t, R_WAIT, '        // [p63] AIV 不等 READY', 'rwait')

io.open(DST, 'w', encoding='utf-8').write(t)
print('PATCHED_TEL lines %d -> %d' % (len(src.splitlines()), len(t.splitlines())))
