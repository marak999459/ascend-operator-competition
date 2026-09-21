#!/bin/bash
# 真机跑用例矩阵：$1=组(quick/medium/large/mtile/bnd/ub48/all) $2=预测用的 aiv 数 $3=UB 预算 KB
# 日志当场 tee 进工程目录（机器/容器易失，证据不能只留在机器上）
# 同 build_npu.sh：set_env.sh 不兼容 set -u
set +u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

for f in "$HOME/Ascend/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/cann/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/ascend-toolkit/set_env.sh" \
         /usr/local/Ascend/cann-9.0.0/set_env.sh; do
    [ -f "$f" ] && { source "$f"; break; }
done
ARCH=$(uname -m)
export LD_LIBRARY_PATH=$ROOT/build_out:$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
# 排错时改 1（debug）重跑；平时 4 只留 error，避免污染对拍日志
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-4}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}

MODE=${1:-quick}; AIV=${2:-50}; UBB=${3:-64}
LOG=npu_debug/logs/npu_${MODE}_$(date +%Y%m%d_%H%M%S).log
[ -x npu_debug/test_npu ] || { echo "test_npu missing, run build_npu.sh first"; exit 5; }

# ---- 组装 custom OPP 包：框架只有在按 ASCEND_CUSTOM_OPP_PATH 自己 dlopen 算子 so 时
# 才会把静态注册器写的 LocalRegistry 提交进全局注册表（否则 561002 Do not find tiling func）
PKG=$ROOT/npu_debug/pkg
SRC=$ROOT/build_out/libcust_opapi.so
DST=$PKG/custom/op_api/lib/libcust_opapi.so
if [ ! -f "$DST" ] || [ "$SRC" -nt "$DST" ]; then
    rm -rf "$PKG"; mkdir -p "$PKG/custom/op_api/lib" "$PKG/custom/op_impl/ai_core/tbe"
    cp "$SRC" "$DST" || { echo "copy opapi so failed"; exit 6; }
    for d in kernel config; do
        [ -d "$ROOT/build_out/tmp/vendors/custom/op_impl/ai_core/tbe/$d" ] && \
        cp -r "$ROOT/build_out/tmp/vendors/custom/op_impl/ai_core/tbe/$d" \
              "$PKG/custom/op_impl/ai_core/tbe/" 2>/dev/null
    done
    echo "### pkg assembled: $(ls -la --time-style=+%H:%M:%S $DST | awk '{print $6,$5}')"
fi
export ASCEND_CUSTOM_OPP_PATH=$PKG/custom
export MHC_OPAPI_SO=$DST
export LD_LIBRARY_PATH=$PKG/custom/op_api/lib:$LD_LIBRARY_PATH
echo "### run mode=$MODE aiv=$AIV ub=${UBB}KB log=$LOG"
./npu_debug/test_npu "$AIV" "$UBB" "$MODE" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "### run_npu rc=$rc"
exit $rc
