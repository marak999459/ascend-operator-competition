#!/bin/bash
# [题1/R10 A/B 工装，非提交面] base / b1 / b2 在 c4(fwd-medium) + c6(fwd-large) 上交替取 msprof。
#   cd ~/ops_comp/optA && bash npu_debug/opt_a/run_r10.sh [rounds]
# 为什么缓存 build 产物：一轮 build ~50s，两轮交替要 6 次；把每个模态的 build_out/tmp + pkg
# 存到 /tmp/r10_<m> 后，第 2 轮只还原目录。还原后 prof_matrix.sh 自带的
# "pkg kernel .o == build_out kernel .o" 硬校验仍然生效，缓存错了会 rc=9 中止而不是静默跑错核。
set +u
cd "$(dirname "$0")/../.." || exit 1
ROUNDS=${1:-2}
BASE=op_kernel/mhc_expand.cpp.bak_pre_r6      # 采纳版 daf2b8ed
CASES="4 6"
LOG=npu_debug/logs/r10_$(date +%H%M%S).log
mkdir -p npu_debug/logs
: > "$LOG"

prep() {
    local m=$1
    if [ ! -d "/tmp/r10_$m/tmp" ]; then
        cp "$BASE" op_kernel/mhc_expand.cpp
        if [ "$m" != "base" ]; then
            python3 npu_debug/opt_a/apply_r10.py "$m" >>"$LOG" 2>&1 || { echo "### INJECT FAIL $m" | tee -a "$LOG"; return 1; }
        fi
        bash npu_debug/build_npu.sh > "/tmp/r10_build_$m.log" 2>&1
        if [ $? != 0 ]; then echo "### BUILD FAIL $m" | tee -a "$LOG"; tail -5 "/tmp/r10_build_$m.log" | tee -a "$LOG"; return 1; fi
        mkdir -p "/tmp/r10_$m"
        cp -r build_out/tmp "/tmp/r10_$m/tmp" && cp -r npu_debug/pkg "/tmp/r10_$m/pkg"
        echo "### built+cached $m kernel=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12)" | tee -a "$LOG"
    else
        rm -rf build_out/tmp npu_debug/pkg
        cp -r "/tmp/r10_$m/tmp" build_out/tmp && cp -r "/tmp/r10_$m/pkg" npu_debug/pkg
    fi
    # 还原后必须能证明"设备上跑的就是这一份源码"：先构建再校验 pkg，两边不一致直接跳过该模态。
    K=$(md5sum build_out/tmp/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
    P=$(md5sum npu_debug/pkg/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
    if [ -z "$K" ] || [ "$K" != "$P" ]; then echo "### SKIP $m pkg/build_out mismatch $K/$P" | tee -a "$LOG"; return 1; fi
    echo "### ready $m kernel_src=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12) kernel_o=$(echo $K | cut -c1-12)" | tee -a "$LOG"
}

for r in $(seq 1 "$ROUNDS"); do
    for m in base b1 b2; do
        prep "$m" || continue
        echo "@@@@ round=$r mode=$m $(date +%H:%M:%S)" | tee -a "$LOG"
        bash npu_debug/prof_matrix.sh "r10${m}R${r}" $CASES 2>&1 |
            grep -E "^### |^\s+(fwd|bwd):|流水线均值|PROF_MISSING|ALL PASS|HAS FAIL|^\[PROF" |
            tee -a "$LOG"
        pkill -9 -f 'ops_comp/[a-zA-Z0-9_]*/npu_debug/[t]est_npu' 2>/dev/null
        pkill -9 -f '[m]sprof --task-time' 2>/dev/null
        sleep 2
    done
done
cp "$BASE" op_kernel/mhc_expand.cpp
echo "### done $(date +%H:%M:%S)  log=$LOG  (树已还原 base)" | tee -a "$LOG"
