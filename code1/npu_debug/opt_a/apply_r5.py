#!/usr/bin/env python3
# [R5] 前向的跨任务序：把 PipeBarrier<PIPE_ALL> 换成两条真依赖事件对
#   RAW  MTE2 -> MTE3 ：本任务写的 UB 落地后才能被本任务的写出读走（不可省）
#   WAR  MTE3 -> MTE2 ：两任务前那一次读排空后才能覆写同一块轮转缓冲（留 1 个 token 余量）
# 依据：msprof 前向计数 mte2/mte3 都在动而 vec=0 ⇒ 生产者 MTE2、消费者 MTE3；
#       §11.10 的 V1 当年配的是 MTE2_MTE1/MTE1_MTE2（错引擎），所以"事件配对不足"未成立。
#   python3 apply_r5.py diag   ← 保留 PIPE_ALL，只把两条事件对"挂"在旁边（判用法是否可行）
#   python3 apply_r5.py full   ← 去掉 PIPE_ALL，完全靠事件对定序（判时序假设是否成立）
# 依据：msprof 前向计数 mte2/mte3 都在动而 vec=0 ⇒ 生产者 MTE2、消费者 MTE3；
#       §11.10 的 V1 当年配的是 MTE2_MTE1/MTE1_MTE2 —— 而 arch22 的 EventToIndexAiv() 里
#       压根没有这两个分支（落到 return HardEventAiv::MAX = 非法槽位），所以"事件配对不足"
#       是用错事件类型的产物，不是硬件结论。
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KJ = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
BASE_MD5 = "daf2b8ed3995d4698c435c2d42d67b23"

OLD_SEQ = """        // 前向纯搬运：两块 UB 轮转，不经 TQue（省每任务的队列簿记）
        auto x_local = (slot_ & 1) ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // 这道 barrier 同时管两件事：等 MTE2 落地再让 MTE3 读，以及保证两任务前对同一缓冲
        // 的上一次读已排空（本路径没有 VEC 动作，TQue 的事件序本来就挂不上，换 TBuf 不减少序）
        PipeBarrier<PIPE_ALL>();
"""

# 事件号用字面量：0/1 给按缓冲奇偶的 WAR 对，2 给紧邻的 RAW 对（QUE_MAX_EVENT=4 或 8，均合法；
# 不拿 constexpr 类成员当设备侧实参 —— v1 就是这么挂的，先排除掉这个变量）
NEW_SEQ_DIAG = """        // 前向纯搬运：两块 UB 轮转，不经 TQue（省每任务的队列簿记）
        const uint32_t ev = slot_ & 1;
        if (slot_ >= 2) WaitFlag<HardEvent::MTE3_MTE2>(ev == 0 ? 0 : 1);
        auto x_local = (slot_ & 1) ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // [R5-diag] PIPE_ALL 原样保留，事件对只作为"能否触发"的旁证
        PipeBarrier<PIPE_ALL>();
        SetFlag<HardEvent::MTE2_MTE3>(2);
        WaitFlag<HardEvent::MTE2_MTE3>(2);
"""

NEW_SEQ_FULL = """        // 前向纯搬运：两块 UB 轮转，不经 TQue（省每任务的队列簿记）
        // WAR：本块缓冲上一次被 MTE3 读走（两任务前）必须排空才能覆写；头两轮无在途读
        const uint32_t ev = slot_ & 1;
        if (slot_ >= 2) WaitFlag<HardEvent::MTE3_MTE2>(ev == 0 ? 0 : 1);
        auto x_local = (slot_ & 1) ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // RAW：本任务的输入由 MTE2 落地后才允许 MTE3 读，这条等待省不掉
        SetFlag<HardEvent::MTE2_MTE3>(2);
        WaitFlag<HardEvent::MTE2_MTE3>(2);
"""

OLD_TAIL = """            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
    }
"""

NEW_TAIL = """            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
        SetFlag<HardEvent::MTE3_MTE2>(ev == 0 ? 0 : 1);
    }
"""

PATCHES = [(OLD_SEQ, NEW_SEQ_DIAG), (OLD_TAIL, NEW_TAIL)]
if (sys.argv[1] if len(sys.argv) > 1 else "diag") == "full":
    PATCHES = [(OLD_SEQ, NEW_SEQ_FULL), (OLD_TAIL, NEW_TAIL)]


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


bak = KJ + ".bak_pre_r5"
if os.path.exists(bak):
    shutil.copy2(bak, KJ)          # 可重入：先回到 base 再打
with open(KJ, encoding="utf-8") as f:
    txt = f.read()
before = md5(txt)
if before != BASE_MD5:
    print("BASE MISMATCH: kernel md5=%s, expected %s (R5 只能打在 A1+A2 采纳版上)" % (before, BASE_MD5))
    sys.exit(2)
for old, new in PATCHES:
    n = txt.count(old)
    if n != 1:
        print("ANCHOR FAIL (%d hits) :: %r" % (n, old[:60]))
        sys.exit(3)
for old, new in PATCHES:
    txt = txt.replace(old, new, 1)

if not os.path.exists(bak):
    shutil.copy2(KJ, bak)
with open(KJ, "w", encoding="utf-8") as f:
    f.write(txt)

print("### applied R5  kernel md5 %s -> %s" % (before, md5(txt)))
print("### sanity: PIPE_ALL left in forward = %d, SetFlag = %d, WaitFlag = %d" % (
    txt.count("PipeBarrier<PIPE_ALL>()"), txt.count("SetFlag<"), txt.count("WaitFlag<")))
