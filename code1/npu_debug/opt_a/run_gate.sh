#!/bin/bash
# [题1/R10 门禁工装，非提交面] 提交面真机门禁一条龙：全组矩阵 ×N + prof2 逐条数值+计时 ×M。
#   cd ~/ops_comp/code1 && bash npu_debug/opt_a/run_gate.sh [all轮数] [prof2轮数]
# 为什么要有这条：2026-09-21 那次用 ssh 内联 for 循环，`$r`/`$c` 被外层 shell 吞掉，
# 结果 8 条 prof2 全跑成 case 0 —— 多层引号里的循环变量必须落成脚本再推，不能现拼。
# 判据（两条都要，缺一不算过）：
#   ① run_npu.sh all 每轮 === ALL PASS | fail=0 err=0 ===
#   ② one_case.sh 每条判决行带 timed=1 且 mismatch=0（§17.1：只有 timed= 的判决行才比过数值）
set +u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)
ROUNDS=${1:-2}
PROFS=${2:-1}
STAMP=$(date +%Y%m%d_%H%M%S)
LOGD=npu_debug/logs/gate_r10_$STAMP
mkdir -p "$LOGD"
FAIL=0

echo "### gate root=$ROOT kernel=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12) host=$(md5sum op_host/mhc_expand.cpp | cut -c1-12) $(date +%H:%M:%S)"
bash npu_debug/build_npu.sh > "$LOGD/build.log" 2>&1
if [ $? != 0 ]; then echo "### BUILD FAILED (见 $LOGD/build.log)"; exit 6; fi
grep -E "pkg synced|kernel .o md5" "$LOGD/build.log" | tail -2

for r in $(seq 1 "$ROUNDS"); do
    bash npu_debug/run_npu.sh all > "$LOGD/all_r$r.log" 2>&1
    V=$(grep -E "^=== ALL PASS|^=== HAS FAIL" "$LOGD/all_r$r.log" | tail -1)
    echo "### all round=$r : ${V:-NO-VERDICT}"
    echo "$V" | grep -q "ALL PASS | cases=[0-9]* fail=0 err=0" || FAIL=1
done

for p in $(seq 1 "$PROFS"); do
    for c in 0 1 2 3 4 5 6 7; do
        bash npu_debug/one_case.sh "$c" 150 > "$LOGD/prof2_p${p}_c$c.log" 2>&1
        V=$(grep "timed=1" "$LOGD/prof2_p${p}_c$c.log" | head -1)
        echo "### prof2 p=$p c=$c : ${V:-NO-VERDICT-LINE}"
        echo "$V" | grep -q "timed=1.*mismatch=0.* PASS" && echo "$V" | grep -q "timed=1" || FAIL=1
    done
done

echo "### gate FAIL=$FAIL  logs=$LOGD  $(date +%H:%M:%S)"
exit $FAIL
