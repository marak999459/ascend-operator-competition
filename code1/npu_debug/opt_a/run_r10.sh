#!/bin/bash
# [题1/R10 A/B 工装，非提交面] 若干模态在 c0(fwd-small)+c4(fwd-medium)+c6(fwd-large) 上交替取 msprof。
#   cd ~/ops_comp/optA && bash npu_debug/opt_a/run_r10.sh [rounds] "mode mode..."
#   例：bash npu_debug/opt_a/run_r10.sh 2 "base ad fin"
# 为什么缓存 build 产物：一轮 build ~50s，多轮交替要几十次；把每个模态的 build_out/tmp + pkg +
# 源码存到 /tmp/r10cache/<注入器md5>/<mode> 后，后续轮次只还原目录。
# 缓存根带注入器哈希：改过 apply_r10.py 就不可能"拿旧二进制冒充新代码"（§17.2 那轮的教训）。
# 还原后 prof_matrix.sh / one_case.sh 自带的 "pkg kernel .o == build_out kernel .o" 硬校验仍生效。
set +u
cd "$(dirname "$0")/../.." || exit 1
ROUNDS=${1:-2}
MODES="${2:-base b1 b2}"
BASE=op_kernel/mhc_expand.cpp.bak_pre_r6      # 采纳版 daf2b8ed
INH=$(md5sum npu_debug/opt_a/apply_r10.py | awk '{print substr($1,1,8)}')
CD=/tmp/r10cache/$INH
SUBSRC=${SUBSRC:-/tmp/sub_kernel.cpp}     # mode=sub 时编译的"提交面源码"
CASES="0 4 6"
LOG=npu_debug/logs/r10_$(date +%H%M%S).log
mkdir -p npu_debug/logs "$CD"
: > "$LOG"
echo "### injector=$INH modes='$MODES' rounds=$ROUNDS cache=$CD" | tee -a "$LOG"

prep() {
    local m=$1
    if [ ! -d "$CD/$m/tmp" ]; then
        cp "$BASE" op_kernel/mhc_expand.cpp
        if [ "$m" = sub ]; then
            # sub = 直接编译"提交面那一份文件"（$SUBSRC），不经注入器：
            # 用于证明"交上去的源码"和"取过读数的源码"实测同速，而不是靠注释一句"只改了注释"。
            [ -f "$SUBSRC" ] || { echo "### NO SUBSRC $SUBSRC" | tee -a "$LOG"; return 1; }
            cp "$SUBSRC" op_kernel/mhc_expand.cpp
        elif [ "$m" != "base" ]; then
            python3 npu_debug/opt_a/apply_r10.py "$m" >>"$LOG" 2>&1 || { echo "### INJECT FAIL $m" | tee -a "$LOG"; return 1; }
        fi
        bash npu_debug/build_npu.sh > "/tmp/r10_build_$m.log" 2>&1
        if [ $? != 0 ]; then echo "### BUILD FAIL $m" | tee -a "$LOG"; tail -5 "/tmp/r10_build_$m.log" | tee -a "$LOG"; return 1; fi
        # 新构建必须先过数值门禁再进缓存/计时：只比"启动没报错"的 PASS 在本项目已废（§17.1），
        # 这里强制要求判决行带 timed=1 且 PASS（harness 已改成计时后仍回读比对）。
        for gc in 0 4 6; do
            L=$(timeout 150 bash npu_debug/one_case.sh "$gc" 120 2>&1 | grep -E "timed=1|^### rc")
            V=$(echo "$L" | grep "timed=1" | head -1)
            echo "### gate $m c$gc: ${V:-NO-VERDICT-LINE}" | tee -a "$LOG"
            echo "$V" | grep -q "timed=1.*PASS" || { echo "### GATE FAIL $m c$gc —— 不缓存、不计时" | tee -a "$LOG"; return 1; }
        done
        mkdir -p "$CD/$m"
        cp -r build_out/tmp "$CD/$m/tmp" && cp -r npu_debug/pkg "$CD/$m/pkg" &&
        cp op_kernel/mhc_expand.cpp "$CD/$m/src.cpp"
        echo "### built+cached $m kernel=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12)" | tee -a "$LOG"
    else
        rm -rf build_out/tmp npu_debug/pkg
        cp -r "$CD/$m/tmp" build_out/tmp && cp -r "$CD/$m/pkg" npu_debug/pkg
        # 源码一并还原：缓存命中时若只还原二进制，kernel_src 那一列会是上一个模态的 md5 ⇒ 归因错人
        cp "$CD/$m/src.cpp" op_kernel/mhc_expand.cpp
    fi
    # 还原后必须能证明"设备上跑的就是这一份源码"：先构建再校验 pkg，两边不一致直接跳过该模态。
    K=$(md5sum build_out/tmp/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
    P=$(md5sum npu_debug/pkg/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
    if [ -z "$K" ] || [ "$K" != "$P" ]; then echo "### SKIP $m pkg/build_out mismatch $K/$P" | tee -a "$LOG"; return 1; fi
    echo "### ready $m kernel_src=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12) kernel_o=$(echo $K | cut -c1-12)" | tee -a "$LOG"
}

for r in $(seq 1 "$ROUNDS"); do
    for m in $MODES; do
        prep "$m" || continue
        echo "@@@@ round=$r mode=$m $(date +%H:%M:%S)" | tee -a "$LOG"
        bash npu_debug/prof_matrix.sh "r10${m}R${r}" $CASES 2>&1 |
            grep -E "^### |fwd: n=|bwd: n=|流水线均值|PROF_MISSING|ALL PASS|HAS FAIL|^\[PROF" |
            tee -a "$LOG"
        pkill -9 -f 'ops_comp/[a-zA-Z0-9_]*/npu_debug/[t]est_npu' 2>/dev/null
        pkill -9 -f '[m]sprof --task-time' 2>/dev/null
        sleep 2
    done
done
cp "$BASE" op_kernel/mhc_expand.cpp
echo "### done $(date +%H:%M:%S)  log=$LOG  (树已还原 base)" | tee -a "$LOG"
