#!/usr/bin/env python3
# [题1/R15 工装，非提交面] 给隔离树 probe2 的 prof2 表追加 3 条"外推带"形状（idx 16/17/18）。
#
# 为什么需要：§23.5 的拟合域是 io=96~216KiB，而提交面的新规则 `blk=isqrt(io/4096)` 在
# io ∈ [0.5MB, 6.55MB) 这一段**会真的少开核**（isqrt 恒小于线性律，直到 io=268MB；而 6.55MB 之上
# isqrt 已被 num_aiv=40 夹住 ⇒ 只有这一段是新行为）。平台 case 的 io 我们不知道，这段没测过就是
# 盲交。三条形状把 1.1 / 2.6 / 4.7MB 钉在 `merge_ok` 区间内（tile<=6144 且整行）。
#
# 用法：python3 add_band_cases.py on|off|on2|off2|on3|off3
#   on/off  = idx 16..18（外推带 1.1~5.2MB）      on2/off2 = idx 19..21（小 io 12/24/48KB）
#   on3/off3 = idx 22..24（反向 m 轴判律，code1.md §23.10）
#   on4/off4 = idx 25..28（反向小 S，§23.11）      on5/off5 = idx 29..31（反向 blk>=S，§23.11 p8）
import sys

HARNESS = "npu_debug/test_mhc_expand_npu.cpp"
ANCHOR = "R15 band 探针形状"
ANCHOR2 = "R15 tiny-io 阶梯"
ANCHOR3 = "R16 反向 m 轴判律"
ANCHOR4 = "R16 反向小 S"
ANCHOR5 = "R16 反向 blk>=S"
BLOCK = """            // ==== {anchor}（idx 16..18）：isqrt 唯一少开核的外推带 io=1.1/2.6/4.7MB ====
            {{"fwd-band-64x512-m16",   false, DT_FP16,  64,  512, 16, 101}},  // 16 io=1.09MB isqrt=16 vs 线性=40
            {{"fwd-band-256x1024-m4",   false, DT_FP16, 256, 1024,  4, 101}},  // 17 io=2.62MB isqrt=25 vs 线性=40
            {{"fwd-band-256x2048-m4",   false, DT_FP16, 256, 2048,  4, 101}},  // 18 io=5.24MB isqrt=35 vs 线性=40
""".format(anchor=ANCHOR)
# 提交 8 的 case1 反弹 +38.8% 唯一能解释它的是"合批段拿掉了 R13 那条 blk>=8 下界"，
# 而 p1 的阶梯最低只到 io=96KB ⇒ io=12/24/48KB 的合批谷底根本没测过。
# S 一律 64（>=num_aiv，天然 ROW）⇒ 这一组里 io 是唯一自变量，路由不是混淆因子。
BLOCK2 = """            // ==== {anchor}（idx 19..21）：合批态小 io 阶梯，S 固定 64 只切 io ====
            {{"fwd-tiny-64x32-m2",     false, DT_FP16,  64,   32, 2, 401}},  // 19 io=12KB tile=64B
            {{"fwd-tiny-64x64-m2",     false, DT_FP16,  64,   64, 2, 401}},  // 20 io=24KB tile=128B
            {{"fwd-tiny-64x128-m2",    false, DT_FP16,  64,  128, 2, 401}},  // 21 io=48KB tile=256B
""".format(anchor=ANCHOR2)
# §23.10：p1b 三条全在 m=2，于是"谷底 tpc=4"(L1) 与"谷底每核发起数=12"(L2) 分不开。
# 只切 m（S 固定 64≥num_aiv ⇒ 路由不是混淆因子，D 固定 256 与 c1 同）就能分开：
#   L1 一律 blk=S/4=16；L2 blk=S(1+m)/12 = 16(m2) / 27(m4) / 48→40(m8)。
# 第三条切 D=128（两律同给 16，而现行律给 8）量的是"现行律少开核"那一侧。
BLOCK3 = """            // ==== {anchor}（idx 22..24）：反向只切 m / D，判谷底跟 tpc 还是跟每核发起数 ====
            {{"bwd-maxis-64x256-m4",   true, DT_FP16,  64,  256,  4, 201}},  // 22 io=160KB L1=16 L2=27 今日=26
            {{"bwd-maxis-64x256-m8",   true, DT_FP16,  64,  256,  8, 201}},  // 23 io=288KB L1=16 L2=48 今日=40(夹住)
            {{"bwd-maxis-64x128-m2",   true, DT_FP16,  64,  128,  2, 201}},  // 24 io=48KB  L1=L2=16 今日=8
""".format(anchor=ANCHOR3)
# §23.10 判读 4 之后发现的下一个洞：p6 四条反向形状 S 全 ≥64，于是"现行律给 tpc=2"
# 那一侧（S<num_aiv ⇒ block_dim 先被 total_tasks=S 夹住）从来没进过扫描。
# 而 case1 现在收窄到 `S∈[16,50) ∧ io∈[64,256KB)` 这一类，它的反向孪生正好落在这段里。
BLOCK4 = """            // ==== {anchor}（idx 25..28）：反向小 S（现行律 tpc=1024/D，看这段对不对得上谷底）====
            {{"bwd-S32-D256-m2",  true, DT_FP16,  32,  256,  2, 201}},  // 25 io=48KiB 今日 blk=8  tpc=4
            {{"bwd-S48-D256-m2",  true, DT_FP16,  48,  256,  2, 201}},  // 26 io=72KiB 今日 blk=12 tpc=4
            {{"bwd-S32-D512-m2",  true, DT_FP16,  32,  512,  2, 201}},  // 27 io=96KiB 今日 blk=16 tpc=2
            {{"bwd-S48-D128-m2",  true, DT_FP16,  48,  128,  2, 201}},  // 28 io=36KiB 今日 blk=8  tpc=6
""".format(anchor=ANCHOR4)
# §23.11 的裁定留下的最后一个洞：p6+p7 九条反向形状的谷底全在 blk≈16，于是"反向律 = 定核数 16"
# 比"tpc 律"和"每核字节律"都更统一。但九条里 S 最小是 32 ⇒ **blk=16 从来没越过 total_tasks=S**。
# 一旦 blk ≥ S，多出来的核是纯空转（每核任务数 0，仍要过 barrier），而 §22.3 的空核直测说
# 每多一核 +75~90ns ⇒ 这条律在微小 S 上必须有上限形式。三条形状专门去钉这个上限：
#   c29 S=16：blk=16 恰好 tpc=1（边界）；c30 S=8：blk=16 ⇒ 8 个空核（越界一倍）；
#   c31 S=24：blk=16 ⇒ tpc=2 但有 8 个空核（中间态，且 io=72KiB 让现行律给 12）。
# 现行面（floor 8 / io÷6KiB）在三条上分别是 8 / 8 / 12 ⇒ 都在"安全侧"，本轮只测形状不改面。
BLOCK5 = """            // ==== {anchor}（idx 29..31）：反向 blk 逼近/越过 total_tasks，钉"16 核律"的上限 ====
            {{"bwd-S16-D256-m2",  true, DT_FP16,  16,  256,  2, 201}},  // 29 io=24KiB blk=16=>tpc=1(边界)
            {{"bwd-S8-D256-m2",   true, DT_FP16,   8,  256,  2, 201}},  // 30 io=12KiB blk=16=>8 个空核
            {{"bwd-S24-D512-m2",  true, DT_FP16,  24,  512,  2, 201}},  // 31 io=72KiB blk=16=>tpc=2+8 空核
""".format(anchor=ANCHOR5)


def main():
    act = sys.argv[1] if len(sys.argv) > 1 else "on"
    src = open(HARNESS, encoding="utf-8", newline="").read()
    if act in ("on5", "off5"):
        return patch(act == "on5", ANCHOR5, BLOCK5, "bwd-S48-D128-m2", "bwd-S24-D512-m2")
    if act in ("on4", "off4"):
        return patch(act == "on4", ANCHOR4, BLOCK4, "bwd-maxis-64x128-m2", "bwd-S48-D128-m2")
    if act in ("on3", "off3"):
        return patch(act == "on3", ANCHOR3, BLOCK3, "fwd-tiny-64x128-m2", "bwd-maxis-64x128-m2")
    if act in ("on2", "off2"):
        return patch(act == "on2", ANCHOR2, BLOCK2, "fwd-band-256x2048-m4", "fwd-tiny-64x128-m2")
    if act == "off":
        return patch(False, ANCHOR, BLOCK, "fwd-band-256x2048-m4", "fwd-band-256x2048-m4")
    return patch(True, ANCHOR, BLOCK, "bwd-merge-128x256-m2", "fwd-band-256x2048-m4")


def patch(add, anchor, block, insert_after, last_line):
    """add=在 insert_after 那行之后整块插入；off=从本 anchor 的注释行删到 last_line 那行末尾。"""
    src = open(HARNESS, encoding="utf-8", newline="").read()
    has = anchor in src
    if add:
        if has:
            print("### already patched: " + anchor)
            return 0
        tail = src.find(insert_after)
        end = src.find("\n", src.find("},", tail))
        if tail < 0 or end < 0:
            print("### ABORT: 找不到锚点 " + insert_after)
            return 5
        open(HARNESS, "w", encoding="utf-8", newline="").write(
            src[:end + 1] + block + src[end + 1:])
        print("### patched: +cases (%s)" % anchor)
    else:
        if not has:
            print("### already clean: " + anchor)
            return 0
        i = src.find("            // ==== " + anchor)
        j = src.find(last_line)
        j = src.find("\n", src.find("},", j))
        open(HARNESS, "w", encoding="utf-8", newline="").write(src[:i] + src[j + 1:])
        print("### reverted: -cases (%s)" % anchor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
