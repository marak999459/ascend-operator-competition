# -*- coding: utf-8 -*-
# [题1/R17 H3 工装，非提交面] 反向"整行一次 gather"探针：每行 m 条读 DMA -> 1 条。
#
# 为什么只打这一支：核数轴已由 code1.md §23.25 封存（同 tpc 只差启动块数就有 15.5%/27.1% 的
# 锯齿），H3 换的是**每行几次 DMA 发起**、核数一个字节不动 ⇒ 锯齿在同一 sweep 内是常量偏移。
#
# 为什么"gather"退化成一条连续读：反向输入 x = o_grad 是 [S, m, D] 布局，同一 s 的 m 个副本
# 在 GM 里本来就相邻（整行模式 dTileLen==dTailLen==D）⇒ 不需要 §14.9 的跨步编码，只是把
# blockLen 从 D*es 换成 m*D*es。所以本次改动**不依赖任何未定档的 DMA 语义**。
#
# 单变量在哪：流水结构逐字照抄旧路径（预取下一行、环深 2、DeQue 后才放开），
# VEC 指令数一条不少（还是 m 次 Cast + m 次 Add，只是数据源从"m 个队列格"变成"一格里的 m 段"）
# ⇒ 两臂唯一差别是每行读发起数 m -> 1。
#
# 用法（隔离树根目录；必须在 apply_h1_knobs.py on 之后跑）：
#   python3 apply_h3_gather.py check   # 只验锚点，不落盘
#   python3 apply_h3_gather.py on
#   python3 apply_h3_gather.py off     # 从 .bak_h3 还原并复核 md5
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
F_HOST = os.path.join(ROOT, "op_host", "mhc_expand.cpp")
F_KERN = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
F_HDR  = os.path.join(ROOT, "op_kernel", "mhc_expand_tiling.h")
MARK = "[PROBE ONLY] H3"

# 锚点全部取"未被 R17 旋钮改过"的原文，故在 提交面 与 旋钮面 上都唯一。
EDITS = [
 (F_HDR,
  "    uint32_t splitMode;    // 0 = ROW_SPLIT, 1 = ROW_STREAM_SPLIT(前向), 2 = ELEMENT_SPLIT\n",
  "    uint32_t splitMode;    // 0 = ROW_SPLIT, 1 = ROW_STREAM_SPLIT(前向), 2 = ELEMENT_SPLIT\n"
  "    uint32_t probeBwdG;    // [PROBE ONLY] H3：1 => 反向整行走单次 gather（同构建 A/B）\n"),

 (F_HOST,
  "        tiling->probeNoMerge = std::getenv(\"MHC_NO_MERGE\") ? 1u : 0u;\n",
  "        tiling->probeNoMerge = std::getenv(\"MHC_NO_MERGE\") ? 1u : 0u;\n"
  "        tiling->probeBwdG = std::getenv(\"MHC_BWD_GATHER\") ? 1u : 0u;   // [PROBE ONLY] H3\n"),

 # ---- kernel：环缓冲按资格二选一（gather 开时 in_que_ 不建，省下的字节正好还给大格）----
 (F_KERN,
  "        if constexpr (BACKWARD) {\n"
  "            pipe_.InitBuffer(in_que_, 2, tiling_.dTileLen * elem_size_);\n",
  "        if constexpr (BACKWARD) {\n"
  "            bwd_g_ = BwdGatherBytes(tiling_.dTileLen * elem_size_);   // [PROBE ONLY] H3\n"
  "            if (bwd_g_ != 0) pipe_.InitBuffer(g_que_, 2, bwd_g_);\n"
  "            else pipe_.InitBuffer(in_que_, 2, tiling_.dTileLen * elem_size_);\n"),

 (F_KERN,
  "    __aicore__ inline void ProcessBackward() {\n",
  "    __aicore__ inline void ProcessBackward() {\n"
  "        if (bwd_g_ != 0) { ProcessBackwardGather(); return; }   // [PROBE ONLY] H3\n"),

 # ---- 新函数 + 资格式 + 成员 ----
 (F_KERN,
  "    // 反向单块：逐 m 读 o_grad[i, k, jt]",
  "    // [PROBE ONLY] H3 资格：反向 + ROW 切分 + 整行(dTileLen==dTailLen==D) + m>=2 + UB 装得下。\n"
  "    // 环 2 格大缓冲 = 2*m*tile，外加 out 环 2*tile 与 acc/tmp 各 2*tile(float)，合计 tile*(2m+6)。\n"
  "    // 上限取 128KB（物理 UB 196608B 的 2/3），于是 ms 档的 D=4096/m=8（tile=8192B -> 需 172KB）\n"
  "    // 自动落在门外 —— 大档那六条**解析式恒等**，是这次改动自带的对照组。\n"
  "    __aicore__ inline uint32_t BwdGatherBytes(uint32_t tile_bytes) const {\n"
  "        if (!tiling_.probeBwdG) return 0;\n"
  "        if (tiling_.splitMode != 0 || tiling_.dTileNum != 1) return 0;\n"
  "        if (tiling_.dTileLen != tiling_.D || tiling_.dTailLen != tiling_.D) return 0;\n"
  "        const uint32_t m = tiling_.m;\n"
  "        if (m < 2) return 0;\n"
  "        const uint64_t need = static_cast<uint64_t>(m) * tile_bytes;\n"
  "        if (need * 2 + static_cast<uint64_t>(tile_bytes) * 6 > 131072u) return 0;\n"
  "        return static_cast<uint32_t>(need);\n"
  "    }\n\n"
  "    // [PROBE ONLY] H3：反向整行单次 gather。每行 1 条读 DMA（旧路径 m 条）+ 1 条写；\n"
  "    // 预取下一行、环深 2 —— 与旧路径同一种流水，只是发起数 m -> 1，VEC 指令数不变。\n"
  "    __aicore__ inline void ProcessBackwardGather() {\n"
  "        const uint32_t m = tiling_.m, d = tiling_.D, h = tiling_.dTileLen;\n"
  "        DataCopyPadExtParams<DT_X> pp{false, 0, 0, static_cast<DT_X>(0)};\n"
  "        DataCopyExtParams rd{1, m * h * elem_size_, 0, 0, 0};\n"
  "        DataCopyExtParams one{1, h * elem_size_, 0, 0, 0};\n"
  "        auto acc = acc_buf_.Get<float>();\n"
  "        auto tmp = tmp_buf_.Get<float>();\n"
  "        auto nxt = g_que_.AllocTensor<DT_X>();\n"
  "        DataCopyPad(nxt, x_gm_[static_cast<int64_t>(task_begin_) * m * d], rd, pp);\n"
  "        g_que_.EnQue(nxt);\n"
  "        for (uint32_t t = task_begin_; t < task_end_; ++t) {\n"
  "            auto cur = g_que_.DeQue<DT_X>();\n"
  "            if (t + 1 < task_end_) {\n"
  "                nxt = g_que_.AllocTensor<DT_X>();\n"
  "                DataCopyPad(nxt, x_gm_[static_cast<int64_t>(t + 1) * m * d], rd, pp);\n"
  "                g_que_.EnQue(nxt);\n"
  "            }\n"
  "            Duplicate(acc, 0.0f, h);\n"
  "            for (uint32_t k = 0; k < m; ++k) {\n"
  "                Cast(tmp, cur[k * h], RoundMode::CAST_NONE, h);\n"
  "                Add(acc, acc, tmp, h);\n"
  "            }\n"
  "            g_que_.FreeTensor(cur);\n"
  "            auto out_buf = out_que_.AllocTensor<DT_X>();\n"
  "            Cast(out_buf, acc, RoundMode::CAST_RINT, h);\n"
  "            out_que_.EnQue(out_buf);\n"
  "            auto o_local = out_que_.DeQue<DT_X>();\n"
  "            DataCopyPad(o_gm_[static_cast<int64_t>(t) * d], o_local, one);\n"
  "            out_que_.FreeTensor(o_local);\n"
  "        }\n"
  "    }\n\n"
  "    // 反向单块：逐 m 读 o_grad[i, k, jt]"),

 (F_KERN,
  "    TQue<QuePosition::VECIN, 2> in_que_;\n",
  "    TQue<QuePosition::VECIN, 2> in_que_;\n"
  "    TQue<QuePosition::VECIN, 2> g_que_;      // [PROBE ONLY] H3：单次 gather 的环\n"
  "    uint32_t bwd_g_ = 0;                     // [PROBE ONLY] H3：非 0 = 本核走 gather 路径，值 = 每格字节\n"),
]

def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read().replace(b"\r\n", b"\n")).hexdigest()

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE not in ("check", "on", "off"):
    print("usage: apply_h3_gather.py check|on|off"); sys.exit(1)

if MODE == "off":
    bad = 0
    for path in (F_HOST, F_KERN, F_HDR):
        bak = path + ".bak_h3"
        if not os.path.exists(bak):
            print("### %s NO .bak_h3" % os.path.basename(path)); bad = 1; continue
        want = md5(bak)
        shutil.copy2(bak, path)
        got = md5(path)
        print("### %s restored -> %s %s" % (os.path.basename(path), got[:12],
                                             "OK" if got == want else "MISMATCH"))
        if got != want: bad = 1
    for path in (F_HOST, F_KERN, F_HDR):
        n = open(path, encoding="utf-8").read().count(MARK)
        if n: print("### 残留 MARK %s in %s" % (n, os.path.basename(path))); bad = 1
    sys.exit(1 if bad else 0)

texts = {}
fails = 0
for path, old, new in EDITS:
    txt = texts.get(path)
    if txt is None:
        txt = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    if new in txt:
        texts[path] = txt
        print("### skip（该编辑已在）%s" % os.path.relpath(path, ROOT)); continue
    n = txt.count(old)
    if n != 1:
        print("ANCHOR FAIL in %s (%d hits) :: %r" % (path, n, old[:56])); fails += 1; continue
    texts[path] = txt.replace(old, new, 1)
if fails or MODE == "check":
    print("### check edits=%d fails=%d" % (len(EDITS), fails))
    sys.exit(1 if fails else 0)

for path in (F_HOST, F_KERN, F_HDR):
    if not os.path.exists(path + ".bak_h3"):
        shutil.copy2(path, path + ".bak_h3")
for path, txt in texts.items():
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    print("### wrote %-34s md5=%s" % (os.path.relpath(path, ROOT), md5(path)[:12]))
print("### H3 edits=%d getenv=%d gather_fn=%d" % (
    len(EDITS),
    sum(1 for p, t in texts.items() for L in t.splitlines() if "MHC_BWD_GATHER" in L),
    sum(1 for p, t in texts.items() for L in t.splitlines() if "ProcessBackwardGather" in L)))
