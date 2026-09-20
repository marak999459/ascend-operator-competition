#!/usr/bin/env python3
# [OPT A] 小档"每核标量序言 + 每任务标量簿记"两条候选改动的注入器（只在隔离树里用）。
#   A1 = 前向不再 InitBuffer 三个它根本不碰的缓冲（out_que_/acc_buf_/tmp_buf_）
#   A2 = 前向单块搬运不走 TQue（去掉 Alloc/EnQue/DeQue/Free 每任务标量簿记）
# 用法：python3 apply_a.py A1        # 只打 A1
#      python3 apply_a.py A2        # 在已打 A1 的树上再打 A2
# 锚点不唯一/找不到 => 直接退出，不产生半拉子源码。提交面树永不运行本脚本。
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KJ = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")

OLD_INIT = """        // UB 缓冲（双缓冲）
        pipe_.InitBuffer(in_que_, 2, tiling_.dTileLen * elem_size_);
        pipe_.InitBuffer(out_que_, 2, tiling_.dTileLen * elem_size_);
        pipe_.InitBuffer(acc_buf_, tiling_.dTileLen * sizeof(float));
        pipe_.InitBuffer(tmp_buf_, tiling_.dTileLen * sizeof(float));
"""

NEW_INIT_A1 = """        // UB 缓冲（双缓冲）：前向是纯搬运，out_que_/acc_buf_/tmp_buf_ 全程不碰 => 不建
        pipe_.InitBuffer(in_que_, 2, tiling_.dTileLen * elem_size_);
        if constexpr (BACKWARD) {
            pipe_.InitBuffer(out_que_, 2, tiling_.dTileLen * elem_size_);
            pipe_.InitBuffer(acc_buf_, tiling_.dTileLen * sizeof(float));
            pipe_.InitBuffer(tmp_buf_, tiling_.dTileLen * sizeof(float));
        }
"""

NEW_INIT_A2 = """        // UB 缓冲：反向走 TQue 流水线（in/out 双缓冲 + 两块 float 暂存）；
        // 前向是纯搬运，只要两块轮转缓冲，其余一个都不建
        if constexpr (BACKWARD) {
            pipe_.InitBuffer(in_que_, 2, tiling_.dTileLen * elem_size_);
            pipe_.InitBuffer(out_que_, 2, tiling_.dTileLen * elem_size_);
            pipe_.InitBuffer(acc_buf_, tiling_.dTileLen * sizeof(float));
            pipe_.InitBuffer(tmp_buf_, tiling_.dTileLen * sizeof(float));
        } else {
            pipe_.InitBuffer(fwd_b0_, tiling_.dTileLen * elem_size_);
            pipe_.InitBuffer(fwd_b1_, tiling_.dTileLen * elem_size_);
        }
"""

OLD_FWD = """        auto in_buf = in_que_.AllocTensor<DT_X>();
        DataCopyPad(in_buf, x_gm_[src_off], cp, pp);
        in_que_.EnQue(in_buf);
        auto x_local = in_que_.DeQue<DT_X>();
        // 前向是纯搬运、中间没有 VEC 指令，TQue<VECIN> 的序只挂在 VEC 消费上，所以 MTE2
        // 落地与上一次 MTE1 读完这两道序得自己补；本路径无 VEC 动作，ALL 与单条 barrier 等价
        PipeBarrier<PIPE_ALL>();
"""

NEW_FWD = """        // 前向纯搬运：两块 UB 轮转，不经 TQue（省每任务的队列簿记）。
        // 轮转保证"本任务写 A、下一任务写 B"，且 MTE 同管按程序序执行 => 无需额外 barrier
        auto x_local = (slot_ & 1) ? fwd_b1_.Get<DT_X>() : fwd_b0_.Get<DT_X>();
        ++slot_;
        DataCopyPad(x_local, x_gm_[src_off], cp, pp);
        // TQue 的序本来就只挂在 VEC 消费上，这里换成 TBuf 后 MTE2 落地这道序仍要自己补
        PipeBarrier<PIPE_ALL>();
"""

OLD_TAIL = """            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
        in_que_.FreeTensor(x_local);
    }
"""

NEW_TAIL = """            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
    }
"""

OLD_MEMBERS = """    TBuf<TPosition::VECCALC> acc_buf_;
    TBuf<TPosition::VECCALC> tmp_buf_;
"""

NEW_MEMBERS = """    TBuf<TPosition::VECCALC> acc_buf_;
    TBuf<TPosition::VECCALC> tmp_buf_;
    TBuf<TPosition::VECCALC> fwd_b0_;
    TBuf<TPosition::VECCALC> fwd_b1_;
"""

OLD_SLOT = """        tiling_ = tiling;
        elem_size_ = sizeof(DT_X);
"""

NEW_SLOT = """        tiling_ = tiling;
        elem_size_ = sizeof(DT_X);
        slot_ = 0;
"""

OLD_SLOTDECL = """    uint32_t task_begin_;
"""

NEW_SLOTDECL = """    uint32_t task_begin_;
    uint32_t slot_;
"""

PATCH_A1 = [(OLD_INIT, NEW_INIT_A1)]
# A2 必须在 A1 之后打（锚点就是 A1 落下的那段），这样三步 base/A1/A1+A2 都能各自单独归因
PATCH_A2 = [(NEW_INIT_A1, NEW_INIT_A2), (OLD_FWD, NEW_FWD), (OLD_TAIL, NEW_TAIL),
            (OLD_MEMBERS, NEW_MEMBERS), (OLD_SLOT, NEW_SLOT), (OLD_SLOTDECL, NEW_SLOTDECL)]


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()

var = sys.argv[1] if len(sys.argv) > 1 else "A1"
patches = PATCH_A1 if var == "A1" else PATCH_A2

with open(KJ, encoding="utf-8") as f:
    txt = f.read()
before = md5(txt)

for old, new in patches:
    n = txt.count(old)
    if n != 1:
        print("ANCHOR FAIL (%d hits) :: %r" % (n, old[:60]))
        sys.exit(3)
for old, new in patches:
    txt = txt.replace(old, new, 1)

if var == "A1":
    bak = KJ + ".bak_pre_a1"
    if not os.path.exists(bak):
        shutil.copy2(KJ, bak)
with open(KJ, "w", encoding="utf-8") as f:
    f.write(txt)

print("### applied %s  kernel md5 %s -> %s" % (var, before, md5(txt)))
print("### sanity: InitBuffer in file = %d, TQue ops in forward = %d, fwd_b0_ = %d" % (
    txt.count("pipe_.InitBuffer"), txt.count("in_que_.AllocTensor") + txt.count("in_que_.FreeTensor"),
    txt.count("fwd_b0_")))
