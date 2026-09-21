#!/bin/bash
# [题1/R11 A/B 工装，非提交面] "成本地板"探针 + 小 tile 攒批候选，多模态交替取 msprof。
#   cd ~/ops_comp/optR11 && bash npu_debug/opt_a/run_r11.sh [rounds] "mode mode..." [cases]
#   例：bash npu_debug/opt_a/run_r11.sh 2 "base b4s nob"          # 默认用例 0 1 2 4
#       bash npu_debug/opt_a/run_r11.sh 1 "b4s" "0 1 2 4 6"       # 指定用例
# 模态见 apply_r11.py。**候选模态(b2s/b4s)必须先过数值门禁才进缓存**（§17.1：只比"启动没报错"
# 的 PASS 在本项目已废）；探针模态(nob/prol/empty)是**故意破坏数值**的地板读数，永不采纳，
# 所以跳过门禁并在判决行打 PROBE 标记，避免被误读成"跑对了"。
# 缓存按注入器 md5 分目录（§17.2 的教训：改过注入器就不可能拿旧二进制冒充新代码）。
set +u
cd "$(dirname "$0")/../.." || exit 1
ROUNDS=${1:-2}
MODES="${2:-base b2s b4s nob prol empty}"
CASES="${3:-0 1 2 4}"
BASE=op_kernel/mhc_expand.cpp.bak_r11base     # 提交面 038d65a0
INH=$(md5sum npu_debug/opt_a/apply_r11.py | awk '{print substr($1,1,8)}')
CD=/tmp/r11cache/$INH
LOG=npu_debug/logs/r11_$(date +%H%M%S).log
GATED="b2s b4s"
mkdir -p npu_debug/logs "$CD"
: > "$LOG"
echo "### injector=$INH modes='$MODES' cases='$CASES' rounds=$ROUNDS cache=$CD" | tee -a "$LOG"

is_gate() { case " $GATED " in *" $1 "*) return 0;; *) return 1;; esac; }

prep() {
    local m=$1
    if [ ! -d "$CD/$m/tmp" ]; then
        cp "$BASE" op_kernel/mhc_expand.cpp
        [ "$m" = base ] || { python3 npu_debug/opt_a/apply_r11.py "$m" >>"$LOG" 2>&1 ||
            { echo "### INJECT FAIL $m" | tee -a "$LOG"; return 1; }; }
        bash npu_debug/build_npu.sh > "/tmp/r11_build_$m.log" 2>&1
        if [ $? != 0 ]; then echo "### BUILD FAIL $m" | tee -a "$LOG";
            tail -8 "/tmp/r11_build_$m.log" | tee -a "$LOG"; return 1; fi
        if is_gate "$m"; then
            for gc in 0 2 4; do
                V=$(timeout 150 bash npu_debug/one_case.sh "$gc" 120 2>&1 | grep "timed=1" | head -1)
                echo "### gate $m c$gc: ${V:-NO-VERDICT-LINE}" | tee -a "$LOG"
                echo "$V" | grep -q "timed=1.*PASS" ||
                    { echo "### GATE FAIL $m c$gc —— 不缓存、不计时" | tee -a "$LOG"; return 1; }
            done
        else
            echo "### PROBE $m —— 数值门禁按设计不适用，本模态只取时间，禁止当作'跑对了'" | tee -a "$LOG"
        fi
        mkdir -p "$CD/$m"
        cp -r build_out/tmp "$CD/$m/tmp" && cp -r npu_debug/pkg "$CD/$m/pkg" &&
        cp op_kernel/mhc_expand.cpp "$CD/$m/src.cpp"
        echo "### built+cached $m kernel=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12)" | tee -a "$LOG"
    else
        rm -rf build_out/tmp npu_debug/pkg
        cp -r "$CD/$m/tmp" build_out/tmp && cp -r "$CD/$m/pkg" npu_debug/pkg
        cp "$CD/$m/src.cpp" op_kernel/mhc_expand.cpp
    fi
    K=$(md5sum build_out/tmp/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
    P=$(md5sum npu_debug/pkg/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
    if [ -z "$K" ] || [ "$K" != "$P" ]; then echo "### SKIP $m pkg/build_out mismatch $K/$P" | tee -a "$LOG"; return 1; fi
    echo "### ready $m kernel_src=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12) kernel_o=$(echo $K | cut -c1-12)" | tee -a "$LOG"
}

for r in $(seq 1 "$ROUNDS"); do
    for m in $MODES; do
        prep "$m" || continue
        echo "@@@@ round=$r mode=$m $(date +%H:%M:%S)" | tee -a "$LOG"
        bash npu_debug/prof_matrix.sh "r11${m}R${r}" $CASES 2>&1 |
            grep -E "^### |fwd: n=|bwd: n=|流水线均值|PROF_MISSING|ALL PASS|HAS FAIL|^\[PROF" |
            tee -a "$LOG"
        pkill -9 -f 'ops_comp/[a-zA-Z0-9_]*/npu_debug/[t]est_npu' 2>/dev/null
        pkill -9 -f '[m]sprof --task-time' 2>/dev/null
        sleep 2
    done
done
cp "$BASE" op_kernel/mhc_expand.cpp
echo "### done $(date +%H:%M:%S)  log=$LOG  (树已还原 base)" | tee -a "$LOG"
