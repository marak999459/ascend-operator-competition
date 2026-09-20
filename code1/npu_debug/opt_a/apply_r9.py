# [题1/R9 工装，非提交面] 前向流水：把 PipeBarrier<PIPE_ALL> 拆成显式事件对。
#
# 为什么现在才敢拆：R7 已裁定相邻 MTE2->MTE3 事件对在 arch22 上可用（raw 过、不 trap），
# 而 R7-war（barrier + 跨任务 WAR）挂死 ⇒ 嫌疑落在"barrier 与手写事件抢同一组 event id"。
# 本注入器的三个模态都不留 barrier，因此能同时回答：
#   1) 手写事件对单独能不能承担正确性序（ev1 / db2）；
#   2) 去掉 barrier 后 MTE2 与 MTE3 能不能真重叠（db2，目标见 §14.9：fwd-medium 12.8us -> 8.4us）；
#   3) none 作为**阴性对照**：若连它都 ALL PASS，说明 c0 根本没在校验数值，
#      那么此前所有"PASS"读数都得重新一一判定（这条必须先跑）。
#
#   python3 apply_r9.py none|ev1|db2
# 锚点不唯一 / base md5 不符 => 直接退出，不产生半拉子源码。只在隔离树里跑。
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KJ = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
BASE_MD5 = "daf2b8ed3995d4698c435c2d42d67b23"   # 采纳版（A1+A2）

OLD_BLOCK = """        // 前向纯搬运：两块 UB 轮转，不经 TQue（省每任务的队列簿记）
        auto x_local = (slot_ & 1) ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // 这道 barrier 同时管两件事：等 MTE2 落地再让 MTE3 读，以及保证两任务前对同一缓冲
        // 的上一次读已排空（本路径没有 VEC 动作，TQue 的事件序本来就挂不上，换 TBuf 不减少序）
        PipeBarrier<PIPE_ALL>();
"""

NEW_NONE = """        auto x_local = (slot_ & 1) ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // [R9-none] 阴性对照：只删 barrier，不加任何序 -> 期望数值 FAIL（若 PASS 则 c0 没在校验）
"""

NEW_EV1 = """        auto x_local = (slot_ & 1) ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // [R9-ev1] 只用一对相邻事件替掉 barrier（RAW：等 MTE2 落地再让 MTE3 读）
        // 注意：这块没有 WAR 保护，两任务前覆写同一缓冲的冒险仍在 —— 交给用例自己暴露
        SetFlag<HardEvent::MTE2_MTE3>(2);
        WaitFlag<HardEvent::MTE2_MTE3>(2);
"""

NEW_DB2 = """        const uint32_t rot = slot_ & 1;
        // [R9-db2] 双缓冲流水：覆写前等这一格上一次 MTE3 读空（WAR，必须在 DataCopy 之前）
        if (slot_ >= 2) WaitFlag<HardEvent::MTE3_MTE2>(rot);
        auto x_local = rot ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        SetFlag<HardEvent::MTE2_MTE3>(2);
        WaitFlag<HardEvent::MTE2_MTE3>(2);
"""

NEW_DB3 = """        const uint32_t rot = slot_ & 1;
        // [R9-db3] 与 db2 同构，唯一区别：RAW 事件 id 随 rot 轮转（db2 两笔都用 id 2）。
        // 动机：db2 的错法是"整行错、got 是别的行的合法值"= RAW（MTE3 读到未落地的 UB），
        //      而 393216 = 24 × 16384 恰好是整数行 ⇒ 不是尾序、不是覆写。
        //      若 arch22 的 flag 是电平量（同一 id 上两笔在飞会并成一次到达），
        //      那么"每任务一对 id=2"就会让 MTE3 等到错的代次 —— 换轮转 id 即可判定。
        if (slot_ >= 2) WaitFlag<HardEvent::MTE3_MTE2>(rot);
        auto x_local = rot ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        SetFlag<HardEvent::MTE2_MTE3>(rot);
        WaitFlag<HardEvent::MTE2_MTE3>(rot);
"""

NEW_DB4 = """        const uint32_t rot = slot_ & 1;
        // [R9-db4] 与 db3 唯一区别：两个方向用**不相交**的 id（RAW 用 4/5，WAR 用 0/1）。
        // 判别点：db2/db3/db3e 三种写法 mismatch 计数完全相同（393216=整 24 行）——
        // 确定性的错法不是竞争而是结构性的，最像"两个方向其实共用一张硬件 flag 表，
        // 同 id 互相吞"。若 db4 仍错同样的量，就基本可以判"手写事件对在这条纯搬运
        // 路径上根本没提供序"，R 路线到此为止。
        if (slot_ >= 2) WaitFlag<HardEvent::MTE3_MTE2>(rot);
        auto x_local = rot ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        SetFlag<HardEvent::MTE2_MTE3>(2 + rot);
        WaitFlag<HardEvent::MTE2_MTE3>(2 + rot);
"""

NEW_MB2 = """        const uint32_t rot = slot_ & 1;
        // [R9-mb2] 分工：RAW 用**只等 MTE2 的窄 barrier**，WAR 用 MTE3_MTE2 事件。
        // 依据（本轮实测）：MTE2_MTE3 事件对不提供序（ev1 15.45M 错 ≈ none 15.15M），
        // 而 MTE3_MTE2 事件确实提供（db2/db3/db4 一致压到 0.393M 且与 id 编号无关）。
        // 关键差别 vs 采纳版：窄 barrier **不等 MTE3** ⇒ MTE3 可以跨任务跑在前面 = 真重叠。
        if (slot_ >= 2) WaitFlag<HardEvent::MTE3_MTE2>(rot);
        auto x_local = rot ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        PipeBarrier<PIPE_MTE2>();
"""

OLD_TAIL = """            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
    }
"""
NEW_TAIL_DB2 = """            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
        SetFlag<HardEvent::MTE3_MTE2>(rot);   // 交出本格的 WAR token，供 2 个任务后使用
    }
"""

OLD_DRAIN = """        }
    }

private:
"""
NEW_DRAIN = """        }
        // [R9-db2] 收尾吃掉环上未消费的 WAR token，保证 launch 内收支平衡
        if constexpr (!BACKWARD) {
            const uint32_t pend = (slot_ < 2) ? slot_ : 2;
            for (uint32_t j = 1; j <= pend; ++j) WaitFlag<HardEvent::MTE3_MTE2>((slot_ - j) & 1);
        }
    }

private:
"""

# db2e = db2 + 整段收尾一次 barrier。
# 动机来自 db2 的失败形状：mismatch=393216 = 96 个 dTile（4096 tile 的 2.3%），
# 而 96 ≈ 每核最后那一两块 —— 说明错的不是任务间的序（事件对已经把 90% 的坑填了），
# 而是**内核退休时 MTE3 的拷贝还没排空**。所以把 barrier 从"每任务一道"降成"整段一道"：
# 既补上这段尾序，又保住 §11.11 量到的 barrier 代价下降（原来每任务一次 = 值 +5.5%）。
NEW_DRAIN_DB2E = """        }
        // [R9-db2e] 先吃掉环上未消费的 WAR token，再一道 barrier 兜住"内核退休时 MTE3 未排空"
        if constexpr (!BACKWARD) {
            const uint32_t pend = (slot_ < 2) ? slot_ : 2;
            for (uint32_t j = 1; j <= pend; ++j) WaitFlag<HardEvent::MTE3_MTE2>((slot_ - j) & 1);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
"""

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
PATCH = {"none": [(OLD_BLOCK, NEW_NONE)],
         "ev1": [(OLD_BLOCK, NEW_EV1)],
         "db2": [(OLD_BLOCK, NEW_DB2), (OLD_TAIL, NEW_TAIL_DB2), (OLD_DRAIN, NEW_DRAIN)],
         "db2e": [(OLD_BLOCK, NEW_DB2), (OLD_TAIL, NEW_TAIL_DB2), (OLD_DRAIN, NEW_DRAIN_DB2E)],
         "db3": [(OLD_BLOCK, NEW_DB3), (OLD_TAIL, NEW_TAIL_DB2), (OLD_DRAIN, NEW_DRAIN)],
         "db3e": [(OLD_BLOCK, NEW_DB3), (OLD_TAIL, NEW_TAIL_DB2), (OLD_DRAIN, NEW_DRAIN_DB2E)],
         "db4": [(OLD_BLOCK, NEW_DB4), (OLD_TAIL, NEW_TAIL_DB2), (OLD_DRAIN, NEW_DRAIN)],
         "mb2": [(OLD_BLOCK, NEW_MB2), (OLD_TAIL, NEW_TAIL_DB2), (OLD_DRAIN, NEW_DRAIN)]}
if MODE not in PATCH:
    print("usage: apply_r9.py none|ev1|db2|db2e|db3|db3e|db4|mb2"); sys.exit(1)


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


with open(KJ, encoding="utf-8") as f:
    txt = f.read()
before = md5(txt)
if before != BASE_MD5:
    print("BASE MISMATCH: kernel md5=%s, expected %s" % (before, BASE_MD5))
    sys.exit(2)
for old, new in PATCH[MODE]:
    n = txt.count(old)
    if n != 1:
        print("ANCHOR FAIL (%d hits) :: %r" % (n, old[:56]))
        sys.exit(3)

bak = KJ + ".bak_pre_r9"
if not os.path.exists(bak):
    shutil.copy2(KJ, bak)
for old, new in PATCH[MODE]:
    txt = txt.replace(old, new, 1)
with open(KJ, "w", encoding="utf-8") as f:
    f.write(txt)

print("### applied R9-%s  kernel md5 %s -> %s" % (MODE, before, md5(txt)))
print("### sanity: PIPE_ALL=%d MTE2_MTE3_set=%d MTE3_MTE2_set=%d wait_MTE3_MTE2=%d" % (
    txt.count("PipeBarrier<PIPE_ALL>()"),
    txt.count("SetFlag<HardEvent::MTE2_MTE3>"),
    txt.count("SetFlag<HardEvent::MTE3_MTE2>"),
    txt.count("WaitFlag<HardEvent::MTE3_MTE2>")))
