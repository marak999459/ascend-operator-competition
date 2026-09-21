#!/bin/bash
# [题1/R11-B 工装，非提交面] (核数 × 批大小) **联合**扫描。
#   cd ~/ops_comp/optR11 && bash npu_debug/opt_a/run_r11b.sh [rounds] "modes" "blks" [cases]
#   例：bash npu_debug/opt_a/run_r11b.sh 1 "base b4s empty" "nat 2 4 8 16 40" "0"
#
# 为什么还要这一轮（不是重复 §14.2）：§14.2 那条"blk 谷底在 12~16"是在 **BS=1、每核一道
# barrier** 的驱动下测的；R11-A 证明攒批能拿 0.40us，而 empty 地板又以 ~96ns/核 上涨。
# 两个效应方向相反，且核数决定"每核几个块"、批大小决定"几道 barrier"——
# 单扫任一维都会看到假最优，必须联合扫。`nat` 列 = 不强制（= §14.3 的 6KB/核 规则）。
#
# host 侧走 apply_r11_host.py 的 MHC_FORCE_BLK 探针（getenv 属归因注入，提交前整段删除，§14.6）。
# 所以每个模态构建完必须硬校验**运行期那份 pkg .so 真带探针**：strings 命中 0 => 跑的是没
# 补丁的 host，这一格读数全部作废（本项目已因"跑的是另一份二进制"废过一整轮，§17.2）。
set +u
cd "$(dirname "$0")/../.." || exit 1
ROUNDS=${1:-1}
MODES="${2:-base b4s empty}"
BLKS="${3:-nat 2 4 8 12 16 24 40}"
CASES="${4:-0}"
BASE=op_kernel/mhc_expand.cpp.bak_r11base          # 提交面 038d65a0
INH=$(md5sum npu_debug/opt_a/apply_r11.py | awk '{print substr($1,1,8)}')
GATED="base b2s b4s"        # base 也要过：证的是"host 带 getenv 补丁、env 未设时行为不变"
LOG=npu_debug/logs/r11b_$(date +%H%M%S).log
mkdir -p npu_debug/logs
: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }

probe_hits() {   # pkg 里那份 .so 是否带 MHC_FORCE_BLK 探针
    strings -a npu_debug/pkg/custom/op_api/lib/libcust_opapi.so 2>/dev/null |
        grep -c MHC_FORCE_BLK
}

# ---- 0) host 探针：整轮只打一次，所有模态共用 ----
HP=$(python3 npu_debug/opt_a/apply_r11_host.py on 2>&1 | tee -a "$LOG")
say "$HP"
HOSTM=$(echo "$HP" | grep -o 'HOST_PROBE_MD5=[0-9a-f]*' | cut -d= -f2 | cut -c1-8)
[ -n "$HOSTM" ] || { say "### HOST PROBE ABORT —— 拿不到 HOST_PROBE_MD5，别扫"; exit 2; }
CD=/tmp/r11bcache/$INH-$HOSTM
mkdir -p "$CD"
say "### injector=$INH hostprobe=$HOSTM blks='$BLKS' cases='$CASES' rounds=$ROUNDS cache=$CD"

is_gate() { case " $GATED " in *" $1 "*) return 0;; *) return 1;; esac; }

prep() {
    local m=$1
    if [ ! -d "$CD/$m/tmp" ]; then
        cp "$BASE" op_kernel/mhc_expand.cpp
        [ "$m" = base ] || { python3 npu_debug/opt_a/apply_r11.py "$m" >>"$LOG" 2>&1 ||
            { say "### INJECT FAIL $m"; return 1; }; }
        bash npu_debug/build_npu.sh > "/tmp/r11b_build_$m.log" 2>&1
        if [ $? != 0 ]; then say "### BUILD FAIL $m";
            tail -8 "/tmp/r11b_build_$m.log" | tee -a "$LOG"; return 1; fi
        local h=$(probe_hits)
        [ "$h" = "1" ] || { say "### HOST-PROBE MISSING $m hits=$h —— pkg .so 没带探针，读数会全等，停"; return 1; }
        if is_gate "$m"; then
            for gc in 0 2 4; do
                V=$(env -u MHC_FORCE_BLK timeout 150 bash npu_debug/one_case.sh "$gc" 120 2>&1 | grep "timed=1" | head -1)
                say "### gate $m c$gc(nat): ${V:-NO-VERDICT-LINE}"
                echo "$V" | grep -q "timed=1.*PASS" ||
                    { say "### GATE FAIL $m c$gc —— 不缓存、不计时"; return 1; }
            done
            # 再在强制小核数下过两条： blk 变了任务区间也变，这两格验的是"探针没把边界跑飞"。
            # c1 是反向：R11 各模态都不碰反向代码，所以反向在这里的作用是**证明强制核数本身
            # 不破坏数值** —— 否则整条 blk 曲线（含 §14.3 规则的重新推导）都建立在飞掉的输出上。
            for gc2 in 0 1; do
                V=$(MHC_FORCE_BLK=4 timeout 150 bash npu_debug/one_case.sh "$gc2" 120 2>&1 | grep "timed=1" | head -1)
                say "### gate $m c$gc2(blk=4): ${V:-NO-VERDICT-LINE}"
                echo "$V" | grep -q "timed=1.*PASS" ||
                    { say "### GATE FAIL $m c$gc2@4 —— 强制核数下数值就飞了，不缓存"; return 1; }
            done
            V=$(MHC_FORCE_BLK=40 timeout 150 bash npu_debug/one_case.sh 1 120 2>&1 | grep "timed=1" | head -1)
            say "### gate $m c1(blk=40): ${V:-NO-VERDICT-LINE}"
            echo "$V" | grep -q "timed=1.*PASS" ||
                { say "### GATE FAIL $m c1@40"; return 1; }
        else
            say "### PROBE $m —— 数值门禁按设计不适用，本模态只取时间，禁止当作'跑对了'"
        fi
        mkdir -p "$CD/$m"
        rm -rf "$CD/$m/tmp" "$CD/$m/pkg"
        cp -r build_out/tmp "$CD/$m/tmp" &&
        cp build_out/libcust_opapi.so "$CD/$m/libcust_opapi.so" &&
        cp -r npu_debug/pkg "$CD/$m/pkg" &&
        cp op_kernel/mhc_expand.cpp "$CD/$m/src.cpp" || { say "### CACHE FAIL $m"; return 1; }
        say "### built+cached $m kernel=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12) hostprobe_hits=$h"
    else
        rm -rf build_out/tmp npu_debug/pkg
        cp -r "$CD/$m/tmp" build_out/tmp && cp -r "$CD/$m/pkg" npu_debug/pkg &&
        cp "$CD/$m/libcust_opapi.so" build_out/libcust_opapi.so &&
        cp "$CD/$m/src.cpp" op_kernel/mhc_expand.cpp
    fi
    local K=$(md5sum build_out/tmp/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
    local P=$(md5sum npu_debug/pkg/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
    [ -n "$K" ] && [ "$K" = "$P" ] || { say "### SKIP $m pkg/build_out mismatch $K/$P"; return 1; }
    local S=$(md5sum build_out/libcust_opapi.so | awk '{print substr($1,1,12)}')
    local PS=$(md5sum npu_debug/pkg/custom/op_api/lib/libcust_opapi.so | awk '{print substr($1,1,12)}')
    [ "$S" = "$PS" ] || { say "### SKIP $m build_out/pkg .so differ $S/$PS"; return 1; }
    [ "$(probe_hits)" = "1" ] || { say "### SKIP $m probe vanished on restore"; return 1; }
    say "### ready $m src=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12) kernel_o=$(echo $K | cut -c1-12) so=$S"
}

run_blk() {   # $1=mode $2=blk $3=round
    if [ "$2" = nat ]; then unset MHC_FORCE_BLK; TAGB=nat
    else export MHC_FORCE_BLK="$2"; TAGB=b"$2"; fi
    say "@@@@ round=$3 mode=$1 blk=$TAGB $(date +%H:%M:%S)"
    bash npu_debug/prof_matrix.sh "r11b_${1}_${TAGB}_R${3}" $CASES 2>&1 |
        grep -E "^### |fwd: n=|bwd: n=|流水线均值|PROF_MISSING|ALL PASS|HAS FAIL|^\[PROF" |
        tee -a "$LOG"
    pkill -9 -f 'ops_comp/[a-zA-Z0-9_]*/npu_debug/[t]est_npu' 2>/dev/null
    pkill -9 -f '[m]sprof --task-time' 2>/dev/null
    sleep 2
}

for r in $(seq 1 "$ROUNDS"); do
    for m in $MODES; do
        prep "$m" || continue
        for b in $BLKS; do run_blk "$m" "$b" "$r"; done
    done
done

# ---- 收尾：内核回 base、host 探针整段删掉，并**重建**产物 ----
# 为什么收尾要重建：否则 pkg 里留下一份带 getenv 的 .so，下一轮只要忘了 build 就会拿它当提交面读数。
unset MHC_FORCE_BLK
cp "$BASE" op_kernel/mhc_expand.cpp
python3 npu_debug/opt_a/apply_r11_host.py off >>"$LOG" 2>&1
bash npu_debug/build_npu.sh > /tmp/r11b_build_restore.log 2>&1
say "### restore build rc=$? host=$(md5sum op_host/mhc_expand.cpp | cut -c1-12) kernel=$(md5sum op_kernel/mhc_expand.cpp | cut -c1-12) probe_hits=$(probe_hits)"
say "### done $(date +%H:%M:%S)  log=$LOG"
