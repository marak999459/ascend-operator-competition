#!/usr/bin/env python3
# [R6] 前向跨任务序：RING 块轮转 + 收支平衡的两条事件对，替代整条 PipeBarrier<PIPE_ALL>。
# 与 R5 的差别 = R5 只写了环、没写收尾，而 harness 一个进程内连发 400 次 ⇒ 每次 launch 末尾
# 留下未消费的 WAR token，下一次 launch 的 set/wait 全部错位（这正是 R5-full 三次 rc=124 的
# 头号嫌疑）。R6 在任务循环后按"未消费数 = min(RING, 本 launch 任务数)"精确吃掉，收支平衡。
#   RAW  MTE2 -> MTE3 ：本任务读的 UB 必须由本任务的载入落地（紧邻 set→wait，天然平衡）
#   WAR  MTE3 -> MTE2 ：RING 个任务前那次读排空后才能覆写同一块缓冲（跨任务，需收尾）
# 靶子（02aeb 真机实测，采纳版 daf2b8ed）：fwd-medium 任务 12.8µs，mte2 4.7 + mte3 8.4 = 13.1
#   ≈ 任务时长 ⇒ 两条 MTE 几乎零重叠；重叠到位的理论地板是 max(4.7,8.4)=8.4µs（-34%）。
# 用法：python3 apply_r6.py 2      # RING=2（默认，UB 占用与采纳版相同）
#      python3 apply_r6.py 3      # RING=3（更深流水，前向 UB 多用一块）
# 锚点不唯一 / base md5 不符 => 直接退出，不产生半拉子源码。只在隔离树里跑。
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KJ = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
BASE_MD5 = "daf2b8ed3995d4698c435c2d42d67b23"   # 采纳版（A1+A2），R6 只能打在它上面

RING = sys.argv[1] if len(sys.argv) > 1 else "2"
RING_N = int(RING)
EV_RAW = RING_N          # RAW 用的事件号与 WAR 的 0..RING-1 不重叠

OLD_INIT = """        } else {
            pipe_.InitBuffer(fwd_b0_, tiling_.dTileLen * elem_size_);
            pipe_.InitBuffer(fwd_b1_, tiling_.dTileLen * elem_size_);
        }
"""
NEW_INIT = """        } else {
            for (uint32_t j = 0; j < %s; ++j) {
                pipe_.InitBuffer(fwd_b_[j], tiling_.dTileLen * elem_size_);
            }
        }
""" % RING

OLD_SEQ = """        // 前向纯搬运：两块 UB 轮转，不经 TQue（省每任务的队列簿记）
        auto x_local = (slot_ & 1) ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // 这道 barrier 同时管两件事：等 MTE2 落地再让 MTE3 读，以及保证两任务前对同一缓冲
        // 的上一次读已排空（本路径没有 VEC 动作，TQue 的事件序本来就挂不上，换 TBuf 不减少序）
        PipeBarrier<PIPE_ALL>();
"""
NEW_SEQ = """        // [R6] RING 块轮转 + 真依赖事件对：WAR 只在覆写前等 RING 个任务前的那次读
        const uint32_t rot = slot_ %% %s;
        if (slot_ >= %s) WaitFlag<HardEvent::MTE3_MTE2>(rot);
        auto x_local = fwd_b_[rot].Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        SetFlag<HardEvent::MTE2_MTE3>(%d);
        WaitFlag<HardEvent::MTE2_MTE3>(%d);
""" % (RING, RING, EV_RAW, EV_RAW)

OLD_TAIL = """            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
    }
"""
NEW_TAIL = """            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
        SetFlag<HardEvent::MTE3_MTE2>(rot);
    }
"""

OLD_DRAIN = """        }
    }

private:
"""
NEW_DRAIN = """        }
        // [R6] 收尾：吃掉环上未消费的 WAR token，使每次 launch 事件收支平衡（可连发多次）
        if constexpr (!BACKWARD) {
            const uint32_t pend = (slot_ < %s) ? slot_ : %s;
            for (uint32_t j = 1; j <= pend; ++j) WaitFlag<HardEvent::MTE3_MTE2>((slot_ - j) %% %s);
        }
    }

private:
""" % (RING, RING, RING)

OLD_MEM = """    TBuf<TPosition::VECCALC> fwd_b0_;
    TBuf<TPosition::VECCALC> fwd_b1_;
"""
NEW_MEM = """    TBuf<TPosition::VECCALC> fwd_b_[%s];
""" % RING

PATCHES = [(OLD_INIT, NEW_INIT), (OLD_SEQ, NEW_SEQ), (OLD_TAIL, NEW_TAIL),
           (OLD_DRAIN, NEW_DRAIN), (OLD_MEM, NEW_MEM)]


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


bak = KJ + ".bak_pre_r6"
if os.path.exists(bak):
    shutil.copy2(bak, KJ)          # 可重入：先回到 base 再打
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

print("### applied R6(RING=%s)  kernel md5 %s -> %s" % (RING, before, md5(txt)))
print("### sanity: PIPE_ALL=%d SetFlag=%d WaitFlag=%d fwd_b_=%d fwd_b0_=%d" % (
    txt.count("PipeBarrier<PIPE_ALL>()"), txt.count("SetFlag<"), txt.count("WaitFlag<"),
    txt.count("fwd_b_["), txt.count("fwd_b0_")))
