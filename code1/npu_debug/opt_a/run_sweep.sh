#!/bin/bash
# [通用扫描] 对注入器逐个取模态：回 base → 打补丁 → 构建 → 单用例判死。
# 只在隔离树跑：cd ~/ops_comp/optA && bash npu_debug/opt_a/run_sweep.sh apply_r8.py q qe
# 为什么先单用例：整组门禁被挂死拖成 3×7min；one_case.sh 的 fwd-small 9s 就能定性。
# ⚠️ 两条踩过坑（03:14 那轮三个"挂死"全是废数据）：
#   ① 注入器 BASE MISMATCH 会 exit 2 但**不落盘** ⇒ 必须显式回 base + 每轮核对 md5；
#   ② one_case.sh 把 rc 打在最后一行 ⇒ 不能拿 PIPESTATUS[0]（那是 tail 的 rc，恒 0）。
set +u
cd "$(dirname "$0")/../.." || exit 1
INJ=$1; shift
BASE=op_kernel/mhc_expand.cpp.bak_pre_r6      # == 采纳版 daf2b8ed
[ -f "$BASE" ] || { echo "no base backup $BASE"; exit 1; }
LOG=npu_debug/logs/sweep_$(basename "$INJ" .py)_$(date +%H%M%S).log
: > "$LOG"
for MODE in "$@"; do
    cp "$BASE" op_kernel/mhc_expand.cpp
    echo "########## $INJ $MODE  $(date +%H:%M:%S)  base=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12)" | tee -a "$LOG"
    python3 "npu_debug/opt_a/$INJ" "$MODE" 2>&1 | tee -a "$LOG"
    if [ "${PIPESTATUS[0]}" != "0" ]; then echo "### INJECTOR FAILED, skip" | tee -a "$LOG"; continue; fi
    KM=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12)
    bash npu_debug/build_npu.sh > /tmp/sw_build.log 2>&1
    echo "### build rc=$?  kernel=$KM  error=$(grep -c 'error:' /tmp/sw_build.log)" | tee -a "$LOG"
    OUT=$(timeout 90 bash npu_debug/one_case.sh 0 60 2>&1 | grep -E "^\[PROF|ALL PASS|HAS FAIL|^### rc")
    echo "$OUT" | tee -a "$LOG"
    pkill -9 -f 'ops_comp/[a-zA-Z0-9_]*/npu_debug/[t]est_npu' 2>/dev/null
    pkill -9 -f '[m]sprof --task-time' 2>/dev/null
    sleep 2
done
cp "$BASE" op_kernel/mhc_expand.cpp
echo "### done $(date +%H:%M:%S)  log=$LOG  (树已还原 base)" | tee -a "$LOG"
