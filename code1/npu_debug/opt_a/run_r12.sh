#!/bin/bash
# [题1/R12 A/B 工装，非提交面] 两个**完整源码形态**（提交面 vs 候选）同轮交替取 msprof。
#   cd ~/ops_comp/optR11 && bash npu_debug/opt_a/run_r12.sh [rounds] "base cand" "0 1 2 3 4 6"
# 与 run_r11b.sh 的区别：那边是"一份基线 + 注入器出的模态"（探针），这边候选是**真改过的
# op_kernel + op_host 两个文件**，要评的就是它能不能变成提交面 —— 所以两边都必须过数值门禁，
# 且 A/B 只能同轮交替（R11 实测跨构建漂移 ±0.2~0.3us，比我们要拿的 0.4us 只小一点点）。
# 变体文件对（放在各自原位，靠后缀区分）：
#   base : op_kernel/mhc_expand.cpp.bak_r11base   + op_host/mhc_expand.cpp.bak_r12base
#   cand : op_kernel/mhc_expand.cpp.cand          + op_host/mhc_expand.cpp.cand
# 缓存按两份源码的 md5 联合分目录（§17.2：换过源码就不许拿旧二进制冒充）。
set +u
cd "$(dirname "$0")/../.." || exit 1
ROUNDS=${1:-3}
VARIANTS="${2:-base cand}"
CASES="${3:-0 1 2 3 4 6}"
K_BASE=op_kernel/mhc_expand.cpp.bak_r11base
H_BASE=op_host/mhc_expand.cpp.bak_r12base
K_CAND=op_kernel/mhc_expand.cpp.cand
H_CAND=op_host/mhc_expand.cpp.cand
LOG=npu_debug/logs/r12_$(date +%H%M%S).log
mkdir -p npu_debug/logs
: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }

pair() { case "$1" in
    base) echo "$K_BASE $H_BASE";;
    cand) echo "$K_CAND $H_CAND";;
    *) return 1;; esac; }

apply_variant() {
    local kv hv
    read -r kv hv < <(pair "$1") || return 1
    [ -f "$kv" ] && [ -f "$hv" ] || { say "### MISSING $1 pair ($kv/$hv)"; return 1; }
    cp "$kv" op_kernel/mhc_expand.cpp && cp "$hv" op_host/mhc_expand.cpp
}

prep() {
    local v=$1 kv hv KM HM CD
    read -r kv hv < <(pair "$v") || return 1
    KM=$(md5sum "$kv" | awk '{print substr($1,1,12)}')
    HM=$(md5sum "$hv" | awk '{print substr($1,1,12)}')
    CD=/tmp/r12cache/$v-$KM-$HM
    if [ ! -d "$CD/tmp" ]; then
        apply_variant "$v" || return 1
        bash npu_debug/build_npu.sh > "/tmp/r12_build_$v.log" 2>&1
        if [ $? != 0 ]; then say "### BUILD FAIL $v";
            tail -8 "/tmp/r12_build_$v.log" | tee -a "$LOG"; return 1; fi
        # 数值门禁：前向 + 反向各过一条 timed=1 PASS 才算这个变体能被计时
        for gc in 0 1 4 6; do
            V=$(timeout 200 bash npu_debug/one_case.sh "$gc" 180 2>&1 | grep "timed=1" | head -1)
            say "### gate $v c$gc: ${V:-NO-VERDICT-LINE}"
            echo "$V" | grep -q "timed=1.*mismatch=0.* PASS" ||
                { say "### GATE FAIL $v c$gc —— 不缓存、不计时"; return 1; }
        done
        mkdir -p "$CD"
        rm -rf "$CD/tmp" "$CD/pkg"
        cp -r build_out/tmp "$CD/tmp" &&
        cp build_out/libcust_opapi.so "$CD/libcust_opapi.so" &&
        cp -r npu_debug/pkg "$CD/pkg" &&
        cp op_kernel/mhc_expand.cpp "$CD/kernel.cpp" && cp op_host/mhc_expand.cpp "$CD/host.cpp" ||
            { say "### CACHE FAIL $v"; return 1; }
        say "### built+gated+cached $v kernel=$KM host=$HM"
    else
        apply_variant "$v" || return 1
        rm -rf build_out/tmp npu_debug/pkg
        cp -r "$CD/tmp" build_out/tmp && cp -r "$CD/pkg" npu_debug/pkg &&
        cp "$CD/libcust_opapi.so" build_out/libcust_opapi.so
    fi
    local K=$(md5sum build_out/tmp/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
    local P=$(md5sum npu_debug/pkg/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
    [ -n "$K" ] && [ "$K" = "$P" ] || { say "### SKIP $v pkg/build_out .o mismatch"; return 1; }
    [ "$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12)" = "$KM" ] || { say "### SKIP $v kernel src drift"; return 1; }
    [ "$(md5sum op_host/mhc_expand.cpp | cut -c1-12)" = "$HM" ] || { say "### SKIP $v host src drift"; return 1; }
    say "### ready $v kernel=$KM host=$HM kernel_o=$(echo $K | cut -c1-12)"
}

say "### variants='$VARIANTS' cases='$CASES' rounds=$ROUNDS $(date +%H:%M:%S)"
for r in $(seq 1 "$ROUNDS"); do
    for v in $VARIANTS; do
        prep "$v" || continue
        say "@@@@ round=$r variant=$v $(date +%H:%M:%S)"
        bash npu_debug/prof_matrix.sh "r12${v}R${r}" $CASES 2>&1 |
            grep -E "^### |fwd: n=|bwd: n=|流水线均值|PROF_MISSING|ALL PASS|HAS FAIL|^\[PROF" |
            tee -a "$LOG"
        pkill -9 -f 'ops_comp/[a-zA-Z0-9_]*/npu_debug/[t]est_npu' 2>/dev/null
        pkill -9 -f '[m]sprof --task-time' 2>/dev/null
        sleep 2
    done
done
say "### done $(date +%H:%M:%S)  log=$LOG  (树停在最后一个变体，未自动还原)"
