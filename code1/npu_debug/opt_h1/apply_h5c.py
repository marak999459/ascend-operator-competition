# -*- coding: utf-8 -*-
# [题1/R22 H5c 工装，非提交面] 反向 m>=2 的"每批 L 行"：把每核每行的
#   m 条串行读 + m 道窄 VEC  换成  m 条步进 gather（一次收 L 行）+ **一次宽 VEC(L·D)** + 一次连续写
# 权威落档 code1.md §27.6（立项）/ §28（读数）。
#
# 为什么钱在 VEC 不在 DMA（H5b 判负的那一步）：§26.2 律 `每行 = 145 + 107·m ns` 里 `107·m` 那半
# 挂在"每副本一道 Cast+Add"上，只合 DMA 动不到它（§27.6 预测 0.46µs ≈ 0.71 分）。这一刀把
# **VEC 发起条数 ÷L**，是 `m>=2` 侧唯一还在野的机制候选。
#
# 这一刀要的 DMA 形态 = §14.9 第 7 条那句"唯一还能表达的合并形态（本轮未测）"：
#   {blockCount=L, blockLen=rb, srcStride=(m-1)*rb, dstStride=0}
# 源每块前进 rb+gap = m*rb 正好落到下一行的同一副本层，目的连续 ⇒ 语义由 §14.9-②③ 的机制推出来，
# **但没有真机读数**（当年只测了 dst 侧跨步）。⇒ **数值门禁先于计时，且 mismatch!=0 时先换
# 回退方案（1 条连续读 L·m·D + m 次 UB→UB 步进 gather，UB 侧 gap 单位 = 32B 块，题2 §11.22）
# 再谈机制。**
#
# 位级恒等的理由（可证）：逐元素对 k 的求和顺序一字未动 —— gather 只是把第 k 层的 L 行摆进 UB
# 相邻段，宽 VEC 对 L·D 个元素做的还是同一个 Cast/Add；`dTileNum==1` ⇒ `dTailLen==dTileLen==D`
# （host :88-102 整行模式），于是行内长度与行间步距同为 D，`base = 行号*m*D`。
#
# 单变量在哪：只有"每批几行"。TQue 双缓冲流水线结构逐字照抄已过 8/8 门禁的 BackwardOneBlock
# （预取第 k+1 副本与第 k 层 VEC 并行），host 侧一行未改 ⇒ blk 仍由现律决定，靠 MHC_FORCE_BLK 臂控。
#
# 资格：探针位 ∧ splitMode==0 ∧ dTileNum==1 ∧ **m>=2**（与 H5a 的 m==1 互斥 ⇒ 同一形状只有一支非 0）
#       ∧ tile<=6144B ∧ 32B 对齐 ∧ cap>=2 ∧ UB 装得下每批 8 份 tile 字节。
# 预算：in×2 + out×2 + acc(float, 2 份) + tmp(float, 2 份) = 8·L·tile <= 98304 = ub_size/2，
#       比 §23.26 反向预算用的 131072 还保守 ⇒ 只可能漏配、不可能超配挂死。
#
# 用法（隔离树根目录；必须在 apply_h1_knobs.py on 与 apply_h5_copy.py on 之后跑）：
#   python3 apply_h5c.py check   # 只验锚点，不落盘
#   python3 apply_h5c.py on
#   python3 apply_h5c.py off     # 从 .bak_h5c 还原并复核 md5
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
F_HOST = os.path.join(ROOT, "op_host", "mhc_expand.cpp")
F_KERN = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
F_HDR  = os.path.join(ROOT, "op_kernel", "mhc_expand_tiling.h")
MARK = "[PROBE ONLY] H5c"
KNOB = "MHC_BWD_WIDE"

HDR_OLD = "    uint32_t probeBwdC;    // [PROBE ONLY] H5a：1 => 反向 m==1 整行走一次连续读+一次连续写\n};\n"
HDR_NEW = ("    uint32_t probeBwdC;    // [PROBE ONLY] H5a：1 => 反向 m==1 整行走一次连续读+一次连续写\n"
           "    uint32_t probeBwdW;    // [PROBE ONLY] H5c：1 => 反向 m>=2 每批 L 行合一次宽 VEC\n};\n")
HOST_OLD = '        tiling->probeBwdC = std::getenv("%s") ? 1u : 0u;   // [PROBE ONLY] H5a\n' % "MHC_BWD_COPY"
HOST_NEW = (HOST_OLD +
            '        tiling->probeBwdW = std::getenv("%s") ? 1u : 0u;  // [PROBE ONLY] H5c\n' % KNOB)

# ---- kernel：四处（资格函数 + 新函数 / 缓冲三选一 / 分派 / 成员）----
CAP_OLD = ("            bwd_c_ = BwdCopyRows(tiling_.dTileLen * elem_size_);   // [PROBE ONLY] H5a\n")
CAP_NEW = ("            bwd_c_ = BwdCopyRows(tiling_.dTileLen * elem_size_);   // [PROBE ONLY] H5a\n"
           "            bwd_w_ = BwdWideRows(tiling_.dTileLen * elem_size_);   // [PROBE ONLY] H5c\n")

BUF_OLD = ("            } else {\n"
           "            pipe_.InitBuffer(in_que_, 2, tiling_.dTileLen * elem_size_);\n"
           "            pipe_.InitBuffer(out_que_, 2, tiling_.dTileLen * elem_size_);\n"
           "            pipe_.InitBuffer(acc_buf_, tiling_.dTileLen * sizeof(float));\n"
           "            pipe_.InitBuffer(tmp_buf_, tiling_.dTileLen * sizeof(float));\n")
BUF_NEW = ("            } else if (bwd_w_ != 0) {\n"
           "                // 同一条 TQue 流水线，只是每格容量从 1 行扩到 L 行（cap 已按 8*L*tile<=98304 夹过）\n"
           "                const uint32_t wb = bwd_w_ * tiling_.dTileLen * elem_size_;\n"
           "                pipe_.InitBuffer(in_que_, 2, wb);\n"
           "                pipe_.InitBuffer(out_que_, 2, wb);\n"
           "                pipe_.InitBuffer(acc_buf_, bwd_w_ * tiling_.dTileLen * sizeof(float));\n"
           "                pipe_.InitBuffer(tmp_buf_, bwd_w_ * tiling_.dTileLen * sizeof(float));\n"
           "            } else {\n"
           "            pipe_.InitBuffer(in_que_, 2, tiling_.dTileLen * elem_size_);\n"
           "            pipe_.InitBuffer(out_que_, 2, tiling_.dTileLen * elem_size_);\n"
           "            pipe_.InitBuffer(acc_buf_, tiling_.dTileLen * sizeof(float));\n"
           "            pipe_.InitBuffer(tmp_buf_, tiling_.dTileLen * sizeof(float));\n")

DISPATCH_OLD = "        if constexpr (BACKWARD) { if (bwd_c_ != 0) ProcessBackwardCopy(); else ProcessBackward(); }   // [PROBE ONLY] H5a\n"
DISPATCH_NEW = ("        if constexpr (BACKWARD) { if (bwd_c_ != 0) ProcessBackwardCopy();\n"
                "                                 else if (bwd_w_ != 0) ProcessBackwardWide();\n"
                "                                 else ProcessBackward(); }   // [PROBE ONLY] H5a/H5c\n")

FN_OLD = "    // 反向单块：逐 m 读 o_grad[i, k, jt]"
FN_NEW = (
  "    // [PROBE ONLY] H5c 资格：三条\"为什么这么窄\"：\n"
  "    //   ① **m>=2** 与 H5a 的 m==1 互斥 ⇒ 两臂同构建下永远只有一支非 0，交叉污染在构造上不可能；\n"
  "    //   ② cap<2 直接 return 0：L=1 时 srcStride 根本不参与（等于没测这一刀要的 DMA 形态），\n"
  "    //      而每批 0.3µs 的固定成本（§27.3-②）白摊在 1 行上 ⇒ 那一格没有讨论价值；\n"
  "    //   ③ 字节闸沿用前向的 6144B ⇒ D>3072(fp16) 的 ms 档**解析式恒等**，自带恒等对照组。\n"
  "    // 预算：in×2 + out×2 + acc(float=2 份 tile) + tmp(2 份) = 8*L*tile <= 98304 = ub_size/2。\n"
  "    __aicore__ inline uint32_t BwdWideRows(uint32_t tile_bytes) const {\n"
  "        if (!tiling_.probeBwdW) return 0;\n"
  "        if (tiling_.splitMode != 0 || tiling_.dTileNum != 1) return 0;\n"
  "        if (tiling_.m < 2) return 0;\n"
  "        if (tile_bytes > FWD_SMALL_THRESH || (tile_bytes & 31u) != 0) return 0;\n"
  "        if (task_end_ <= task_begin_) return 0;\n"
  "        const uint32_t cap = 12288u / tile_bytes;   // 8 * L * tile_bytes <= 98304\n"
  "        if (cap < 2) return 0;\n"
  "        const uint32_t n = task_end_ - task_begin_;\n"
  "        return (n < cap) ? n : cap;\n"
  "    }\n\n"
  "    // [PROBE ONLY] H5c：本核连续 L 行、每个副本**一条**步进 gather 进 UB：\n"
  "    //   {blockCount=L, blockLen=rb, srcStride=(m-1)*rb, dstStride=0} ⇒ 源每块前进 m*rb 正好是\n"
  "    //   下一行的同一副本层、目的连续。随后对整块 L*D 做一次宽 VEC，最后一次连续写\n"
  "    //   （x_grad 里这 L 行本来就相邻）。流水线结构与 BackwardOneBlock 逐字同构，只有\"每批几行\"变了。\n"
  "    __aicore__ inline void ProcessBackwardWide() {\n"
  "        const uint32_t m = tiling_.m;\n"
  "        const uint32_t d = tiling_.D;\n"
  "        const uint32_t rb = tiling_.dTileLen * elem_size_;\n"
  "        DataCopyPadExtParams<DT_X> pp{false, 0, 0, static_cast<DT_X>(0)};\n"
  "        uint32_t t = task_begin_;\n"
  "        while (t < task_end_) {\n"
  "            uint32_t L = task_end_ - t;\n"
  "            if (L > bwd_w_) L = bwd_w_;\n"
  "            const uint32_t n_el = L * tiling_.dTileLen;\n"
  "            DataCopyExtParams g{static_cast<uint16_t>(L), rb, (m - 1) * rb, 0, 0};\n"
  "            const int64_t base = static_cast<int64_t>(t) * m * d;   // 本批首行的第 0 副本\n"
  "            auto acc = acc_buf_.Get<float>();\n"
  "            Duplicate(acc, 0.0f, n_el);\n"
  "            auto next_dma_buf = in_que_.AllocTensor<DT_X>();\n"
  "            DataCopyPad(next_dma_buf, x_gm_[base], g, pp);\n"
  "            in_que_.EnQue(next_dma_buf);\n"
  "            for (uint32_t k = 0; k < m; ++k) {\n"
  "                auto cur_dma_buf = next_dma_buf;\n"
  "                auto compute_buf = in_que_.DeQue<DT_X>();\n"
  "                if (k < m - 1) {\n"
  "                    next_dma_buf = in_que_.AllocTensor<DT_X>();\n"
  "                    DataCopyPad(next_dma_buf, x_gm_[base + static_cast<int64_t>(k + 1) * d], g, pp);\n"
  "                    in_que_.EnQue(next_dma_buf);\n"
  "                }\n"
  "                auto tmp = tmp_buf_.Get<float>();\n"
  "                Cast(tmp, compute_buf, RoundMode::CAST_NONE, n_el);\n"
  "                Add(acc, acc, tmp, n_el);\n"
  "                in_que_.FreeTensor(compute_buf);\n"
  "            }\n"
  "            auto out_buf = out_que_.AllocTensor<DT_X>();\n"
  "            Cast(out_buf, acc, RoundMode::CAST_RINT, n_el);\n"
  "            out_que_.EnQue(out_buf);\n"
  "            auto o_local = out_que_.DeQue<DT_X>();\n"
  "            DataCopyExtParams w{1, L * rb, 0, 0, 0};\n"
  "            // UB->GM 方向**没有带 padParams 的重载**（c220 impl :542 只有 ext-params 那一支）；\n"
  "            // GM->UB 那两支则逐字等价（:419-423 内部就是 padParams{false,0,0,0}）。\n"
  "            DataCopyPad(o_gm_[static_cast<int64_t>(t) * d], o_local, w);\n"
  "            out_que_.FreeTensor(o_local);\n"
  "            t += L;\n"
  "        }\n"
  "    }\n\n"
  "    // 反向单块：逐 m 读 o_grad[i, k, jt]")

MEM_OLD = "    TBuf<TPosition::VECCALC> tmp_buf_;\n"
MEM_NEW = ("    TBuf<TPosition::VECCALC> tmp_buf_;\n"
           "    uint32_t bwd_w_ = 0;   // [PROBE ONLY] H5c：非 0 = 本核走 gather+宽 VEC，值 = 每批行数 L\n")

EDITS = [
 (F_HDR,  HDR_OLD, HDR_NEW),
 (F_HOST, HOST_OLD, HOST_NEW),
 (F_KERN, CAP_OLD, CAP_NEW),
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
    print("usage: apply_h5c.py check|on|off"); sys.exit(1)

if MODE == "off":
    bad = 0
    for path in (F_HOST, F_KERN, F_HDR):
        bak = path + ".bak_h5c"
        if not os.path.exists(bak):
            print("### %s NO .bak_h5c" % os.path.basename(path)); bad = 1; continue
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
        hint = "  ← 本补丁的锚点来自 H5a+旋钮面，先跑 apply_h1_knobs.py on 与 apply_h5_copy.py on" if n == 0 else ""
        print("ANCHOR FAIL in %s (%d hits)%s :: %r" % (path, n, hint, old[:56])); fails += 1; continue
    texts[path] = txt.replace(old, new, 1)
if fails or MODE == "check":
    print("### check edits=%d fails=%d" % (len(EDITS), fails))
    sys.exit(1 if fails else 0)

for path in (F_HOST, F_KERN, F_HDR):
    if not os.path.exists(path + ".bak_h5c"):
        shutil.copy2(path, path + ".bak_h5c")
for path, txt in texts.items():
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    os.utime(path, None)              # 同上：写完必须让源比产物新
    print("### wrote %-34s md5=%s" % (os.path.relpath(path, ROOT), md5(path)[:12]))
print("### H5c edits=%d getenv=%d wide_fn=%d m2_gate=%d stride=%d" % (
    len(EDITS),
    sum(1 for p, t in texts.items() for L in t.splitlines() if KNOB in L),
    sum(1 for p, t in texts.items() for L in t.splitlines() if "ProcessBackwardWide" in L),
    sum(1 for p, t in texts.items() for L in t.splitlines() if "tiling_.m < 2" in L),
    sum(1 for p, t in texts.items() for L in t.splitlines() if "(m - 1) * rb, 0, 0" in L)))
