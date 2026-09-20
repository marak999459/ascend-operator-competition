#!/bin/bash
# msprof 采 fp16-large 的 aicore_time（V0 无 barrier vs V3 有 barrier 的 A/B）。非提交脚本。
# 用法：bash npu_debug/prof_npu.sh <tag>   tag 例：v3 / v0
# 判据（防假成功）：产物目录里必须出现含 aicore/Machine/CORE 列的 csv，否则打印 PROF_MISSING 并非零退出。
# 同 build_npu.sh：CANN 的 set_env.sh 不兼容 set -u
set +u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

for f in "$HOME/Ascend/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/cann/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/ascend-toolkit/set_env.sh" \
         /usr/local/Ascend/cann-9.0.0/set_env.sh; do
    [ -f "$f" ] && { source "$f"; break; }
done
command -v msprof >/dev/null || { echo "NO msprof on PATH"; exit 2; }

PKG=$ROOT/npu_debug/pkg
SRC=$ROOT/build_out/libcust_opapi.so
DST=$PKG/custom/op_api/lib/libcust_opapi.so
[ -f "$SRC" ] || { echo "build_out/libcust_opapi.so missing, run build_npu.sh first"; exit 3; }
# 包必须跟当前产物同版本：换过 kernel 重编后要重新拷，否则采到的是旧二进制
if [ ! -f "$DST" ] || [ "$SRC" -nt "$DST" ]; then
    echo "### refreshing OPP pkg ($SRC -> $DST)"
    mkdir -p "$(dirname "$DST")" "$PKG/custom/op_impl/ai_core/tbe"
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
export MHC_REPS=${MHC_REPS:-21}

TAG=${1:-untagged}
TS=$(date +%Y%m%d_%H%M%S)
OUT=$ROOT/npu_debug/prof/${TAG}_${TS}
mkdir -p "$OUT"
echo "### prof tag=$TAG reps=$MHC_REPS out=$OUT"
echo "### kernel md5 under test: $(md5sum $ROOT/op_kernel/mhc_expand.cpp | cut -d' ' -f1)  pkg so ts: $(date -r "$DST" +%H:%M:%S)"

msprof --task-time=on --ai-core=on --output="$OUT" "$ROOT/npu_debug/test_npu" 50 64 prof 2>&1 | tail -14
rc=${PIPESTATUS[0]}
echo "### msprof rc=$rc"

CSV=$(find "$OUT" -name "*.csv" 2>/dev/null | wc -l)
echo "### csv files under $OUT = $CSV"
find "$OUT" -name "*.csv" 2>/dev/null | sed "s|$OUT/||" | head -20
if [ "$CSV" = "0" ]; then echo "PROF_MISSING"; exit 5; fi
grep -al "aicore\|MACH_AICORE\|Machine" "$OUT"/*.csv "$OUT"/*/*.csv 2>/dev/null | head -5 \
    || { echo "PROF_NO_AICORE_COLUMN"; exit 6; }
echo "### PROF_OK $OUT"
