#!/bin/bash
# [题1/R11-C 工装，非提交面] 采纳形态的定档扫描：把"核数 × 批大小"的谷点和反向的核数曲线一次跑完。
# 为什么这一轮要 3 个来回配对：R11-A/B 已经看到**跨构建漂移 ±0.2us**（同一份 b4s 二进制，
# 08:52 读 2.6、09:10 读 2.8），单轮读数分不开 8/12/16 三格；只有同轮交替才作数。
#   nohup setsid bash npu_debug/opt_a/run_r11c.sh > /tmp/r11c_rounds.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
echo "########## C1 前向小档 (核数 × 批大小) 3 轮交替  $(date +%H:%M:%S)"
bash npu_debug/opt_a/run_r11b.sh 3 "base b2s b4s" "8 12 16" "0 2"
echo "########## C2 反向小档 核数曲线 2 轮交替        $(date +%H:%M:%S)"
bash npu_debug/opt_a/run_r11b.sh 2 "base" "nat 8 12 16 24 40" "1 3"
echo "########## all done $(date +%H:%M:%S)"
