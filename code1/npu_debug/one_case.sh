#!/bin/bash
# 单条用例探针（非提交工装）：只跑 prof2 分支的第 N 条，行缓冲 + 超时 + 把 slog 打到 stdout。
#   bash npu_debug/one_case.sh [用例序号] [超时秒]
# 存在的理由：整组门禁被 timeout 砍掉时，harness 的 stdout 是全缓冲的 ⇒ 日志 0 行，
# 分不清"设备死等"还是"运行期报错"。这条按单用例、行缓冲重跑一次即可定性。
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
export ASCEND_CUSTOM_OPP_PATH=$PKG/custom
export MHC_OPAPI_SO=$PKG/custom/op_api/lib/libcust_opapi.so
export LD_LIBRARY_PATH=$ROOT/build_out:$PKG/custom/op_api/lib:$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=1

IDX=${1:-0}
TMO=${2:-90}
# 运行前硬校验：设备真正加载的是 pkg 里的 kernel .o，它必须等于 build_out 里刚编出来的那份。
# 不校验就会"跑旧核并把它记成新补丁的读数"（2026-09-21 optA 树栽过，见 build_npu.sh §1.5）。
KB=build_out/tmp/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand
PKB=$PKG/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand
A=$(md5sum $ROOT/$KB/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
B=$(md5sum $PKB/*.o 2>/dev/null | awk '{print $1}' | sort | tr -d '\n')
if [ -z "$A" ] || [ "$A" != "$B" ]; then
    echo "### rc=95 PKG STALE (build_out=$A pkg=$B) —— 先跑 build_npu.sh，别信任何读数"
    exit 95
fi
echo "### one_case root=$ROOT idx=$IDX timeout=${TMO}s so=$(md5sum $MHC_OPAPI_SO 2>/dev/null | cut -c1-8) kernel_o=$(echo $A | cut -c1-12) bin=$(date -r $ROOT/npu_debug/test_npu '+%H:%M:%S')"
MHC_PCASE=$IDX stdbuf -oL -eL timeout "$TMO" "$ROOT/npu_debug/test_npu" 50 64 prof2
echo "### rc=$?  (124=被 timeout 砍=设备侧死等)"
