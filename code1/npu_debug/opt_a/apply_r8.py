#!/usr/bin/env python3
# [R8] 判"手写 SetFlag/WaitFlag 在 arch22 上挂死"到底是用法问题还是**事件池没被初始化**问题。
# 观察链（同一台 02aeb，采纳版 daf2b8ed）：
#   §11.10 V1：前向**还留着 TQue**，事件对配的是 MTE2_MTE1/MTE1_MTE2（arch22 AIV 无此分支
#     ⇒ EventToIndexAiv 落到 MAX 非法槽）⇒ 结果是"计数减半但不挂"。
#   R5-full / R6(补收支平衡) / R7-raw / R7-relay / R7-war：**全部 rc=124 挂死**，
#     而同一份构建里反向（补丁没碰的代码）`rc=0 ALL PASS` ⇒ 挂死只可能来自前向那对事件。
# ⇒ 两条路都指向同一个未验假设：**A2 之后前向一个 TQue 都不建，而事件池要由队列初始化才能用。**
#   python3 apply_r8.py q    ← 只给前向补一个不使用的 TQue 初始化（对照组：加了队列会不会挂）
#   python3 apply_r8.py qe   ← 同上 + 紧邻 RAW 事件对（判"队列初始化是否是事件的先决条件"）
# 锚点不唯一 / base md5 不符 => 直接退出，不产生半拉子源码。只在隔离树里跑。
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KJ = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
BASE_MD5 = "daf2b8ed3995d4698c435c2d42d67b23"

OLD_INIT = """        } else {
            pipe_.InitBuffer(fwd_b0_, tiling_.dTileLen * elem_size_);
            pipe_.InitBuffer(fwd_b1_, tiling_.dTileLen * elem_size_);
        }
"""
NEW_INIT = """        } else {
            pipe_.InitBuffer(fwd_b0_, tiling_.dTileLen * elem_size_);
            pipe_.InitBuffer(fwd_b1_, tiling_.dTileLen * elem_size_);
            // [R8] 前向建一个**不使用**的队列，只为触发框架的事件池初始化
            pipe_.InitBuffer(probe_que_, 2, tiling_.dTileLen * elem_size_);
        }
"""

OLD_SEQ = """        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // 这道 barrier 同时管两件事：等 MTE2 落地再让 MTE3 读，以及保证两任务前对同一缓冲
        // 的上一次读已排空（本路径没有 VEC 动作，TQue 的事件序本来就挂不上，换 TBuf 不减少序）
        PipeBarrier<PIPE_ALL>();
"""
NEW_SEQ_Q = """        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        PipeBarrier<PIPE_ALL>();
"""
NEW_SEQ_QE = """        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // [R8] 紧邻 RAW 一对（有队列初始化之后还能不能挂？）
        SetFlag<HardEvent::MTE2_MTE3>(2);
        WaitFlag<HardEvent::MTE2_MTE3>(2);
        PipeBarrier<PIPE_ALL>();
"""

OLD_MEM = """    TBuf<TPosition::VECCALC> fwd_b0_;
"""
NEW_MEM = """    TQue<QuePosition::VECIN, 2> probe_que_;
    TBuf<TPosition::VECCALC> fwd_b0_;
"""

MODE = sys.argv[1] if len(sys.argv) > 1 else "q"
if MODE == "q":
    PATCHES = [(OLD_INIT, NEW_INIT), (OLD_MEM, NEW_MEM), (OLD_SEQ, NEW_SEQ_Q)]
elif MODE == "qe":
    PATCHES = [(OLD_INIT, NEW_INIT), (OLD_MEM, NEW_MEM), (OLD_SEQ, NEW_SEQ_QE)]
else:
    print("mode must be q|qe")
    sys.exit(1)


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


bak = KJ + ".bak_pre_r8"
if os.path.exists(bak):
    shutil.copy2(bak, KJ)
with open(KJ, encoding="utf-8") as f:
    base_txt = f.read()
before = md5(base_txt)
if before != BASE_MD5:
    print("BASE MISMATCH: kernel md5=%s, expected %s" % (before, BASE_MD5))
    sys.exit(2)
txt = base_txt
for old, new in PATCHES:
    n = txt.count(old)
    if n != 1:
        print("ANCHOR FAIL (%d hits) :: %r" % (n, old[:60]))
        sys.exit(3)
for old, new in PATCHES:
    txt = txt.replace(old, new, 1)
with open(KJ, "w", encoding="utf-8") as f:
    f.write(txt)
if not os.path.exists(bak):
    with open(bak, "w", encoding="utf-8") as f:
        f.write(base_txt)

print("### applied R8-%s  kernel md5 %s -> %s" % (MODE, before, md5(txt)))
print("### sanity: PIPE_ALL=%d SetFlag=%d WaitFlag=%d probe_que_=%d" % (
    txt.count("PipeBarrier<PIPE_ALL>()"), txt.count("SetFlag<"), txt.count("WaitFlag<"),
    txt.count("probe_que_")))
