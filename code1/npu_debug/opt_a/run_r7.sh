#!/bin/bash
# [R7 扫描] 三种事件写法 × 一次构建 × 单用例判死（一条用例 ~9s，"### rc=124" 即设备侧死等）。
# 只在隔离树跑：cd ~/ops_comp/optA && bash npu_debug/opt_a/run_r7.sh
# 为什么先单用例：整组门禁被挂死拖成 3×7min；one_case.sh 9s 就能定性，判死成本降一个数量级。
# ⚠️ 两条踩过坑（03:14 那轮三个"挂死"全是废数据）：
#   ① 注入器 BASE MISMATCH 会 exit 2 但**不落盘** ⇒ 必须显式回 base + 每次核对 md5，否则
#      构建/跑的是上一版残留核，读数全归错人；
#   ② one_case.sh 把 rc 打在最后一行 ⇒ **不能**拿 PIPESTATUS[0]（那是 tail 的 rc，恒 0）。
set +u
cd "$(dirname "$0")/../.." || exit 1
BASE=op_kernel/mhc_expand.cpp.bak_pre_r6      # == 采纳版 daf2b8ed
LOG=npu_debug/logs/r7_sweep_$(date +%H%M%S).log
: > "$LOG"
for MODE in raw relay war; do
    cp "$BASE" op_kernel/mhc_expand.cpp || { echo "no base backup" | tee -a "$LOG"; exit 1; }
    echo "########## R7-$MODE  $(date +%H:%M:%S)  base_md5=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12)" | tee -a "$LOG"
    python3 npu_debug/opt_a/apply_r7.py "$MODE" 2>&1 | tee -a "$LOG"
    if [ "${PIPESTATUS[0]}" != "0" ]; then echo "### INJECTOR FAILED, skip" | tee -a "$LOG"; continue; fi
    KMD5=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12)
    bash npu_debug/build_npu.sh > /tmp/r7_build_$MODE.log 2>&1
    echo "### build rc=$?  kernel_md5=$KMD5  error_lines=$(grep -c 'error:' /tmp/r7_build_$MODE.log)" | tee -a "$LOG"
    grep -E "^### rc=" npu_debug/logs/one_case_*.log 2>/dev/null | tail -1 >/dev/null
    OUT=$(timeout 90 bash npu_debug/one_case.sh 0 60 2>&1 | grep -E "^### rc=|ALL PASS|HAS FAIL|PROF ")
    echo "### one_case(fwd-small): $OUT" | tee -a "$LOG"
    case "$OUT" in
        *"rc=0"*) echo "### VERDICT $MODE: 不挂死" | tee -a "$LOG" ;;
        *)        echo "### VERDICT $MODE: 挂死/异常" | tee -a "$LOG"
                  pkill -9 -f 'ops_comp/[a-zA-Z0-9_]*/npu_debug/[t]est_npu' 2>/dev/null
                  pkill -9 -f '[m]sprof --task-time' 2>/dev/null; sleep 2 ;;
    esac
done
echo "### done $(date +%H:%M:%S)  log=$LOG" | tee -a "$LOG"
cp op_kernel/mhc_expand.cpp.bak_pre_r6 op_kernel/mhc_expand.cpp
