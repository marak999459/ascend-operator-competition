#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [题1/R17 工装，非提交面] 给隔离树 probe2 的 prof2 表**追加**四条"合批态小 io"形状（idx 8..11）。
#
# 为什么是新脚本而不是 add_band_cases.py on2：R16 收尾把 probe2 同步回提交面之后，
# `add_band_cases.py` 的整条插入链都死了 —— 它的 `on` 拿 `bwd-merge-128x256-m2` 当插入锚点，
# 那个名字属于更早的一版 harness（现已 0 命中），而 `on2/on3/on4/on5` 又各自以"上一条已插入"为锚。
# 更硬的一条约束：**prof2 的 idx 是位置编号**，旧 §23.8b 的 c19..c21 只有在那五条 band 形状都在表里
# 时才指到 tiny-io 那组。与其伪造历史编号，不如明写新编号 —— 本脚本一律**追加到表尾**，
# 编号从当前 8 条之后连续排（⇒ 新形状是 c8..c11，与 §23.8b 的 c19..c21 **同号不同形状**，
# 任何跨轮引用都必须看名字不看号）。
#
# 用法（在树根执行，与 apply_h1_knobs.py 正交、顺序无关）：
#   python3 npu_debug/opt_h1/add_r17_cases.py on
#   python3 npu_debug/opt_h1/add_r17_cases.py off
#   python3 npu_debug/opt_h1/add_r17_cases.py list     # 打印 idx -> name/io/tile 对照
import sys

HARNESS = "npu_debug/test_mhc_expand_npu.cpp"
ANCHOR = "R17 H1'' 合批态 blk 下界阶梯"
LAST = '{"bwd-fp16-large",  true,  DT_FP16, 8192, 7168, 8, 21},'
BLOCK = """            // ==== {anchor}（idx 8..11）：S 固定 64（>=num_aiv 的 1.6 倍，路由天然 ROW 且不是混淆因子），
            //      只切 io，配合 MHC_FORCE_BLK 量"合批态 blk 下界该不该是 8"（code1.md §23.15-3）。
            //      四条都是 merge-qualified：dTileNum==1、tile%32==0、tile<=6144、merge_cap*2<=S。
            {{"fwd-h17-64x32-m2",   false, DT_FP16,  64,   32, 2, 201}},  //  8 io=12KiB  tile=64B   现律 blk=8 tpc=8
            {{"fwd-h17-64x64-m2",   false, DT_FP16,  64,   64, 2, 201}},  //  9 io=24KiB  tile=128B  现律 blk=8 tpc=8
            {{"fwd-h17-64x128-m2",  false, DT_FP16,  64,  128, 2, 201}},  // 10 io=48KiB  tile=256B  现律 blk=8 tpc=8
            {{"fwd-h17-64x512-m2",  false, DT_FP16,  64,  512, 2, 201}},  // 11 io=192KiB tile=1024B 现律 blk=8 tpc=8
""".format(anchor=ANCHOR)


def main():
    act = sys.argv[1] if len(sys.argv) > 1 else "on"
    src = open(HARNESS, encoding="utf-8", newline="").read()
    has = ANCHOR in src
    if act == "list":
        print("### anchor %s  patched=%d  表尾锚点 hits=%d" % (ANCHOR, 1 if has else 0, src.count(LAST)))
        return 0
    if act == "on":
        if has:
            print("### already patched: " + ANCHOR); return 0
        p = src.find(LAST)
        if p < 0 or src.count(LAST) != 1:
            print("### ABORT: prof2 表尾锚点不唯一 (%d)" % src.count(LAST)); return 5
        end = src.find("\n", p)
        open(HARNESS, "w", encoding="utf-8", newline="").write(
            src[:end + 1] + BLOCK + src[end + 1:])
        print("### patched: +4 cases idx 8..11 (%s)" % ANCHOR)
        return 0
    if act == "off":
        if not has:
            print("### already clean: " + ANCHOR); return 0
        i = src.find("            // ==== " + ANCHOR)
        j = src.find("fwd-h17-64x512-m2")
        j = src.find("\n", src.find("},", j))
        open(HARNESS, "w", encoding="utf-8", newline="").write(src[:i] + src[j + 1:])
        print("### reverted: -4 cases (%s)" % ANCHOR)
        return 0
    print("usage: add_r17_cases.py on|off|list"); return 1


if __name__ == "__main__":
    sys.exit(main())
