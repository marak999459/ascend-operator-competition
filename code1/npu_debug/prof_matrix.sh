#!/bin/bash
# 逐用例独立 msprof 会话（非提交脚本）。
#   bash npu_debug/prof_matrix.sh <tag> [用例序号...]
# 序号表见 test_mhc_expand_npu.cpp 的 prof2 分支；默认 0..7。
# MHC_NOWALL=1  -> 只跑不带 msprof 的进程（拿主机侧 per= 墙钟，秒级出结果）
# 为什么一条用例一个会话：prof_sum.js 按 Input Shapes 维度数分 fwd/bwd，
# 混形状会把小档和大档的逐任务时间平均成一列。
set +u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

for f in "$HOME/Ascend/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/cann/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/ascend-toolkit/set_env.sh" \
         /usr/local/Ascend/cann-9.0.0/set_env.sh; do
    [ -f "$f" ] && { source "$f"; break; }
done

PKG=$ROOT/npu_debug/pkg
SRC=$ROOT/build_out/libcust_opapi.so
DST=$PKG/custom/op_api/lib/libcust_opapi.so
[ -f "$SRC" ] || { echo "build_out/libcust_opapi.so missing, run build_npu.sh first"; exit 3; }
if [ ! -f "$DST" ] || [ "$SRC" -nt "$DST" ]; then
    echo "### refreshing OPP pkg ($SRC -> $DST)"
    rm -rf "$PKG"; mkdir -p "$(dirname "$DST")" "$PKG/custom/op_impl/ai_core/tbe"
    cp "$SRC" "$DST" || { echo "copy failed"; exit 4; }
    for d in kernel config; do
        [ -d "$ROOT/build_out/tmp/vendors/custom/op_impl/ai_core/tbe/$d" ] && \
            cp -r "$ROOT/build_out/tmp/vendors/custom/op_impl/ai_core/tbe/$d" \
                  "$PKG/custom/op_impl/ai_core/tbe/" 2>/dev/null
    done
fi
export ASCEND_CUSTOM_OPP_PATH=$PKG/custom
export MHC_OPAPI_SO=$DST
export LD_LIBRARY_PATH=$ROOT/build_out:$PKG/custom/op_api/lib:$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-4}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}

TAG=${1:-untagged}; shift
CASES="${*:-0 1 2 3 4 5 6 7}"
BIN=$ROOT/npu_debug/test_npu
KB=$ROOT/build_out/tmp/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand
PKB=$PKG/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand
A=$(md5sum $KB/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
B=$(md5sum $PKB/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
[ -z "$A" ] || [ "$A" = "$B" ] || { echo "### PROF ABORT: pkg kernel .o != build_out ($A vs $B) 先跑 build_npu.sh"; exit 9; }
echo "### kernel md5: $(md5sum $ROOT/op_kernel/mhc_expand.cpp | cut -d' ' -f1)  host md5: $(md5sum $ROOT/op_host/mhc_expand.cpp | cut -d' ' -f1)  kernel_o: $(echo $A | cut -c1-12)"
echo "### so ts: $(date -r "$DST" '+%H:%M:%S')  bin ts: $(date -r "$BIN" '+%H:%M:%S')  tag=$TAG cases=$CASES"

for i in $CASES; do
    if [ "$MHC_NOWALL" = "1" ]; then
        MHC_PCASE=$i "$BIN" 50 64 prof2 2>&1 | grep -E "^\[PROF|=== " | sed "s/^/c$i /"
        continue
    fi
    OUT=$ROOT/npu_debug/prof/${TAG}_c${i}_$(date +%H%M%S)
    mkdir -p "$OUT"
    MHC_PCASE=$i msprof --task-time=on --ai-core=on --output="$OUT" "$BIN" 50 64 prof2 2>&1 |
        grep -E "^\[PROF|=== |Profiling|Failed|error" | sed "s/^/c$i /"
    CSVLIST=$(find "$OUT" -name "*.csv" 2>/dev/null | while read -r c; do
                  head -1 "$c" | grep -q "Task Duration" && echo "$c"
              done)
    if [ -z "$CSVLIST" ]; then echo "c$i PROF_MISSING $OUT"; continue; fi
    node "$ROOT/npu_debug/prof_sum.js" $CSVLIST 2>&1 | sed "s/^/c$i /"
done
