#!/usr/bin/env python3
# [R7] 事件可用性判定：三种"前向跨任务序"写法，谁的哪一条在 arch22 上真的会触发。
# 共同点：**都保留 PipeBarrier<PIPE_ALL>** ⇒ 正确性由 barrier 兜住，所以
#   "挂死(rc=124) / 通过" 这个二元结果只能归因到**注入的事件对本身**，不会与流水竞争混淆。
# 为什么需要：R5-full 三次 rc=124、R6（补了收支平衡收尾）仍 rc=124 ⇒ "事件配对不足"
#   这条 §11.10 的旧结论要么成立、要么 MTE2↔MTE3 之间压根没有可用的硬事件通道。
#   不先判死这一条，任何"去掉 barrier 换流水"的尝试都是在赌。
#   python3 apply_r7.py raw     ← 只加紧邻的 MTE2->MTE3 一对（R5 里 WAR 之外的另一半）
#   python3 apply_r7.py relay   ← 经 VEC 中继：MTE2->V 再 V->MTE3（TQue 实际在用的那两条）
#   python3 apply_r7.py war     ← 只加跨任务 MTE3->MTE2 一对 + 收尾（判 WAR 是否单独致命）
# 锚点不唯一 / base md5 不符 => 直接退出，不产生半拉子源码。只在隔离树里跑。
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KJ = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
BASE_MD5 = "daf2b8ed3995d4698c435c2d42d67b23"   # 采纳版（A1+A2）

OLD_SEQ = """        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // 这道 barrier 同时管两件事：等 MTE2 落地再让 MTE3 读，以及保证两任务前对同一缓冲
        // 的上一次读已排空（本路径没有 VEC 动作，TQue 的事件序本来就挂不上，换 TBuf 不减少序）
        PipeBarrier<PIPE_ALL>();
"""

NEW_RAW = """        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // [R7-raw] 只测这一对能不能触发；barrier 保留 ⇒ 时序仍由 barrier 保证
        SetFlag<HardEvent::MTE2_MTE3>(2);
        WaitFlag<HardEvent::MTE2_MTE3>(2);
        PipeBarrier<PIPE_ALL>();
"""

NEW_RELAY = """        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // [R7-relay] MTE2 落地 -> VEC -> MTE3：两条 TQue 已经在用的事件，中间经 VEC 中继
        SetFlag<HardEvent::MTE2_V>(2);
        WaitFlag<HardEvent::MTE2_V>(2);
        SetFlag<HardEvent::V_MTE3>(3);
        WaitFlag<HardEvent::V_MTE3>(3);
        PipeBarrier<PIPE_ALL>();
"""

NEW_WAR = """        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // [R7-war] 只加跨任务 WAR（本任务读走之后才允许 RING 个任务前那块被覆写），barrier 保留
        if (slot_ >= 2) WaitFlag<HardEvent::MTE3_MTE2>(slot_ & 1);
        PipeBarrier<PIPE_ALL>();
"""

OLD_TAIL = """            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
    }
"""
NEW_TAIL_WAR = """            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
        SetFlag<HardEvent::MTE3_MTE2>((slot_ - 1) & 1);
    }
"""

OLD_DRAIN = """        }
    }

private:
"""
NEW_DRAIN_WAR = """        }
        // [R7-war] 收尾吃掉环上未消费的 WAR token（launch 内收支平衡）
        if constexpr (!BACKWARD) {
            const uint32_t pend = (slot_ < 2) ? slot_ : 2;
            for (uint32_t j = 1; j <= pend; ++j) WaitFlag<HardEvent::MTE3_MTE2>((slot_ - j) & 1);
        }
    }

private:
"""

MODE = sys.argv[1] if len(sys.argv) > 1 else "raw"
if MODE == "raw":
    PATCHES = [(OLD_SEQ, NEW_RAW)]
elif MODE == "relay":
    PATCHES = [(OLD_SEQ, NEW_RELAY)]
elif MODE == "war":
    PATCHES = [(OLD_SEQ, NEW_WAR), (OLD_TAIL, NEW_TAIL_WAR), (OLD_DRAIN, NEW_DRAIN_WAR)]
else:
    print("mode must be raw|relay|war")
    sys.exit(1)


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


bak = KJ + ".bak_pre_r7"
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

print("### applied R7-%s  kernel md5 %s -> %s" % (MODE, before, md5(txt)))
print("### sanity: PIPE_ALL=%d SetFlag=%d WaitFlag=%d" % (
    txt.count("PipeBarrier<PIPE_ALL>()"), txt.count("SetFlag<"), txt.count("WaitFlag<")))
