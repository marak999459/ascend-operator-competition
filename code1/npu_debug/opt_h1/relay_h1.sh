#!/bin/bash
# [题1/R15 工装，非提交面] 串行接力：等 p1 跑完 -> p1b(反向阶梯) -> p2(H4: S<num_aiv 走 ROW)
# 为什么必须串行：两条 msprof 同时跑会共用同一张 NPU，读数互相污染，A/B 直接失效。
set +u
cd "$(dirname "$0")/../.." || exit 1
P1LOG=/tmp/r15_p1.log

echo "### relay start $(date +%H:%M:%S) 等 p1 done 标记"
for i in $(seq 1 240); do
    grep -q "run_h1_sweep done tag=p1_io64kb" "$P1LOG" 2>/dev/null && break
    sleep 15
done
grep -q "run_h1_sweep done tag=p1_io64kb" "$P1LOG" 2>/dev/null || { echo "### relay ABORT: p1 没在 60 分钟内收尾"; exit 9; }
echo "### p1 done $(date +%H:%M:%S)"

# p1b：反向阶梯（合批不参与反向 -> 只跑 mg=on 一臂，省一半时间；off 臂天然是同构建噪声控制）
CASES="1 14 15" BLKS="4 6 8 12 16 24 40" MERGES="on" ROWS=0 ROUNDS=2 AUTO=1 \
    bash npu_debug/opt_h1/run_h1_sweep.sh p1b_bwd 2>&1 | tee /tmp/r15_p1b.log
echo "### p1b done $(date +%H:%M:%S)"

# p2：H4 原型 —— S<num_aiv 的前向今天走 STREAM/ELEMENT，强制它走 ROW（合批资格随之打开）
#      row=0 那一臂是"今日形态"的同构建对照，所以 merge 两臂都要跑。
CASES="10 13" BLKS="4 6 8 12 24 40" MERGES="on off" ROWS="0 1" ROUNDS=2 AUTO=1 \
    bash npu_debug/opt_h1/run_h1_sweep.sh p2_h4 2>&1 | tee /tmp/r15_p2.log
echo "### relay done $(date +%H:%M:%S)"
