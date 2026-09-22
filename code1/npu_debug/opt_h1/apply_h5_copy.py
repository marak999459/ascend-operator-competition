# -*- coding: utf-8 -*-
# [题1/R21 H5a 工装，非提交面] 反向 m==1 的"整块搬运"：每核连续 L 行从 **L 次(读→VEC→写)** 改成
# **1 条连续读 + 1 条连续写、零 VEC 发起**。权威落档 code1.md §26.8。
#
# 为什么这一刀是可证的：m==1 时 BackwardOneBlock 只做
#   Duplicate(acc,0) -> Cast(fp16->fp32, CAST_NONE) -> Add(acc,0,tmp) -> Cast(fp32->fp16, CAST_RINT)
# 输入输出同 dtype，CAST_NONE 是精确扩位、CAST_RINT 回同一 dtype 是恒等往返 ⇒ **结果逐 bit 等于输入**。
# 于是整条 VEC 是白跑的，而"该搬多少"与"从哪搬到哪"在 m==1 时两侧同址（base = i*m*D = i*D）
# ⇒ L 行在 GM 上本来就连续 ⇒ 这次改动**不依赖任何未定档的 DMA 语义**（跟 H3 同一理由）。
#
# 单变量在哪：只换"每核这段连续行的发起形状"。流水结构逐字照抄已过门禁的前向合批（§22）：
# 两格轮换 + 每批两道 PIPE_ALL。资格只由**形状**决定（不看每核行数），所以同一形状下所有核
# 走同一种缓冲布局，UB 预算不随核漂移。
#
# 预算：单格 <= FWD_MERGE_BYTES(16384)，两格 = 32768 <= host 给单块的 ub_size/4 = 49152 < 196608。
# 走这一支时 in_que_/out_que_/acc/tmp **一概不建** ⇒ 比原路径(6*tile)更省 UB。
#
# 用法（隔离树根目录；必须在 apply_h1_knobs.py on 之后跑）：
#   python3 apply_h5_copy.py check   # 只验锚点，不落盘
#   python3 apply_h5_copy.py on
#   python3 apply_h5_copy.py off     # 从 .bak_h5 还原并复核 md5
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
F_HOST = os.path.join(ROOT, "op_host", "mhc_expand.cpp")
F_KERN = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
F_HDR  = os.path.join(ROOT, "op_kernel", "mhc_expand_tiling.h")
MARK = "[PROBE ONLY] H5a"
KNOB = "MHC_BWD_COPY"

HDR_OLD = ("    uint32_t probeNoMerge; // [PROBE ONLY] 1 => MergeRows 直接返回 0（同构建 A/B 的 base 臂）\n};\n")
HDR_NEW = ("    uint32_t probeNoMerge; // [PROBE ONLY] 1 => MergeRows 直接返回 0（同构建 A/B 的 base 臂）\n"
           "    uint32_t probeBwdC;    // [PROBE ONLY] H5a：1 => 反向 m==1 整行走一次连续读+一次连续写\n};\n")
HOST_OLD = '        tiling->probeNoMerge = std::getenv("MHC_NO_MERGE") ? 1u : 0u;\n'
HOST_NEW = (HOST_OLD +
            '        tiling->probeBwdC = std::getenv("%s") ? 1u : 0u;   // [PROBE ONLY] H5a\n' % KNOB)

# ---- kernel：三处（缓冲二选一 / 分派 / 新函数 + 成员）----
BUF_OLD = ("        if constexpr (BACKWARD) {\n"
           "            pipe_.InitBuffer(in_que_, 2, tiling_.dTileLen * elem_size_);\n"
           "            pipe_.InitBuffer(out_que_, 2, tiling_.dTileLen * elem_size_);\n"
           "            pipe_.InitBuffer(acc_buf_, tiling_.dTileLen * sizeof(float));\n"
           "            pipe_.InitBuffer(tmp_buf_, tiling_.dTileLen * sizeof(float));\n")
BUF_NEW = ("        if constexpr (BACKWARD) {\n"
           "            bwd_c_ = BwdCopyRows(tiling_.dTileLen * elem_size_);   // [PROBE ONLY] H5a\n"
           "            if (bwd_c_ != 0) {\n"
           "                // 整块搬运只用两格轮换，一格 = 本批 L 行；原路径那四格一概不建\n"
           "                const uint32_t cb = bwd_c_ * tiling_.dTileLen * elem_size_;\n"
           "                pipe_.InitBuffer(fwd_b0_, cb);\n"
           "                pipe_.InitBuffer(fwd_b1_, cb);\n"
           "            } else {\n"
           "            pipe_.InitBuffer(in_que_, 2, tiling_.dTileLen * elem_size_);\n"
           "            pipe_.InitBuffer(out_que_, 2, tiling_.dTileLen * elem_size_);\n"
           "            pipe_.InitBuffer(acc_buf_, tiling_.dTileLen * sizeof(float));\n"
           "            pipe_.InitBuffer(tmp_buf_, tiling_.dTileLen * sizeof(float));\n"
           "            }\n")
DISPATCH_OLD = "        if constexpr (BACKWARD) ProcessBackward();\n"
DISPATCH_NEW = ("        if constexpr (BACKWARD) { if (bwd_c_ != 0) ProcessBackwardCopy(); else ProcessBackward(); }   // [PROBE ONLY] H5a\n")
FN_OLD = "    // 反向单块：逐 m 读 o_grad[i, k, jt]"
FN_NEW = (
  "    // [PROBE ONLY] H5a 资格：探针开 + 反向 + ROW 切分 + 整行(dTileNum==1) + **m==1** +\n"
  "    // tile<=FWD_SMALL_THRESH(6144B) + 32B 对齐。三条「为什么这么窄」：\n"
  "    //   ① m==1 是**可证恒等**的前提（见文件头），m>=2 要的是 H5 本体、不在这一刀里；\n"
  "    //   ② 不看「本核拿到几行」（前向 MergeRows 要看 n>=2）：L=1 时这一支退化成 1 读 1 写、语义不变，\n"
  "    //      而把资格与每核行数解耦，才能保证同一形状下所有核的 UB 布局一致；\n"
  "    //   ③ 6144B 字节门沿用前向，于是 D>3072(fp16) 的 ms 档**解析式恒等**，自带恒等对照组。\n"
  "    __aicore__ inline uint32_t BwdCopyRows(uint32_t tile_bytes) const {\n"
  "        if (!tiling_.probeBwdC) return 0;\n"
  "        if (tiling_.splitMode != 0 || tiling_.dTileNum != 1) return 0;\n"
  "        if (tiling_.m != 1) return 0;\n"
  "        if (tile_bytes > FWD_SMALL_THRESH || (tile_bytes & 31u) != 0) return 0;\n"
  "        if (task_end_ <= task_begin_) return 0;\n"
  "        const uint32_t cap = FWD_MERGE_BYTES / tile_bytes;\n"
  "        const uint32_t n = task_end_ - task_begin_;\n"
  "        return (n < cap) ? n : cap;\n"
  "    }\n\n"
  "    // [PROBE ONLY] H5a：本核连续 L 行一次搬完。读侧一条 blockCount=1 的连续 DMA（这 L 行在\n"
  "    // o_grad 里本来就相邻），写侧同理落到 x_grad 的同一段 ⇒ 每核每批 = 1 读 + 1 写、零 VEC。\n"
  "    // 两格轮换 + 每批两道 PIPE_ALL 逐字照抄前向合批（arch22 纯搬运只认 PIPE_ALL，§17）。\n"
  "    __aicore__ inline void ProcessBackwardCopy() {\n"
  "        const uint32_t rb = tiling_.dTileLen * elem_size_;\n"
  "        const uint32_t d = tiling_.D;\n"
  "        DataCopyPadExtParams<DT_X> pp{false, 0, 0, static_cast<DT_X>(0)};\n"
  "        uint32_t t = task_begin_, slot = 0;\n"
  "        bool first = true;\n"
  "        while (t < task_end_) {\n"
  "            uint32_t L = task_end_ - t;\n"
  "            if (L > bwd_c_) L = bwd_c_;\n"
  "            auto buf = (slot == 0) ? fwd_b0_.Get<DT_X>() : fwd_b1_.Get<DT_X>();\n"
  "            if (!first) PipeBarrier<PIPE_ALL>();   // 两格轮换：读回本格之前先等上一批 MTE3 腾空\n"
  "            first = false;\n"
  "            DataCopyExtParams cp{1, L * rb, 0, 0, 0};\n"
  "            const int64_t off = static_cast<int64_t>(t) * d;\n"
  "            DataCopyPad(buf, x_gm_[off], cp, pp);\n"
  "            PipeBarrier<PIPE_ALL>();               // 本批落地之后才允许 MTE3 读它\n"
  "            DataCopyPad(o_gm_[off], buf, cp);\n"
  "            t += L;\n"
  "            slot ^= 1u;\n"
  "        }\n"
  "    }\n\n"
  "    // 反向单块：逐 m 读 o_grad[i, k, jt]")
MEM_OLD = "    TBuf<TPosition::VECCALC> tmp_buf_;\n"
MEM_NEW = ("    TBuf<TPosition::VECCALC> tmp_buf_;\n"
           "    uint32_t bwd_c_ = 0;   // [PROBE ONLY] H5a：非 0 = 本核走整块搬运，值 = 每批行数 L\n")

EDITS = [
 (F_HDR,  HDR_OLD, HDR_NEW),
 (F_HOST, HOST_OLD, HOST_NEW),
 (F_KERN, BUF_OLD, BUF_NEW),
 (F_KERN, DISPATCH_OLD, DISPATCH_NEW),
 (F_KERN, FN_OLD, FN_NEW),
 (F_KERN, MEM_OLD, MEM_NEW),
]

def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read().replace(b"\r\n", b"\n")).hexdigest()

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE not in ("check", "on", "off"):
    print("usage: apply_h5_copy.py check|on|off"); sys.exit(1)

if MODE == "off":
    bad = 0
    for path in (F_HOST, F_KERN, F_HDR):
        bak = path + ".bak_h5"
        if not os.path.exists(bak):
            print("### %s NO .bak_h5" % os.path.basename(path)); bad = 1; continue
        want = md5(bak)
        shutil.copy2(bak, path)
        os.utime(path, None)          # copy2 会带上备份的旧 mtime，骗过 build_npu.sh 的重建判据（§23.27-③）
        got = md5(path)
        print("### %s restored -> %s %s" % (os.path.basename(path), got[:12],
                                             "OK" if got == want else "MISMATCH"))
        if got != want: bad = 1
    for path in (F_HOST, F_KERN, F_HDR):
        n = open(path, encoding="utf-8").read().count(MARK)
        if n: print("### 残留 MARK %s in %s" % (n, os.path.basename(path))); bad = 1
    sys.exit(1 if bad else 0)

texts, fails = {}, 0
for path, old, new in EDITS:
    txt = texts.get(path)
    if txt is None:
        if not os.path.exists(path):
            print("NO FILE %s" % path); fails += 1; continue
        txt = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    if new in txt:
        texts[path] = txt
        print("### skip（该编辑已在）%s" % os.path.relpath(path, ROOT)); continue
    n = txt.count(old)
    if n != 1:
        hint = "  ← 先跑 apply_h1_knobs.py on（本补丁的锚点里有两条来自旋钮面）" if n == 0 else ""
        print("ANCHOR FAIL in %s (%d hits)%s :: %r" % (path, n, hint, old[:56])); fails += 1; continue
    texts[path] = txt.replace(old, new, 1)
if fails or MODE == "check":
    print("### check edits=%d fails=%d" % (len(EDITS), fails))
    sys.exit(1 if fails else 0)

for path in (F_HOST, F_KERN, F_HDR):
    if not os.path.exists(path + ".bak_h5"):
        shutil.copy2(path, path + ".bak_h5")
for path, txt in texts.items():
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    os.utime(path, None)              # 同上：写完必须让源比产物新
    print("### wrote %-34s md5=%s" % (os.path.relpath(path, ROOT), md5(path)[:12]))
print("### H5a edits=%d getenv=%d copy_fn=%d m1_gate=%d" % (
    len(EDITS),
    sum(1 for p, t in texts.items() for L in t.splitlines() if KNOB in L),
    sum(1 for p, t in texts.items() for L in t.splitlines() if "ProcessBackwardCopy" in L),
    sum(1 for p, t in texts.items() for L in t.splitlines() if "tiling_.m != 1" in L)))
