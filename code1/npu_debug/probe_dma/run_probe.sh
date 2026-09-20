#!/bin/bash
# [PROBE P1] 在隔离树里 build 一次、4 个变体各起独立进程跑（一条变体把核毒化了也不会传染下一条）。
# 用法：bash npu_debug/probe_dma/run_probe.sh            # 默认 m=8 2 3 4
#      VAR="2 3" bash npu_debug/probe_dma/run_probe.sh   # 只跑指定变体
# 日志只落 npu_debug/logs/，别给 stdout 接 head 之类会提前退出的管道。
set +u
cd "$(dirname "$0")/../.." || exit 1
[ -f npu_debug/probe_dma/fwd_probe.cpp ] || { echo "run this from the probe tree"; exit 7; }
grep -q "\[PROBE\]" op_kernel/mhc_expand.cpp || { echo "kernel not patched, run mk_probe.py first"; exit 7; }

mkdir -p npu_debug/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG=npu_debug/logs/probe_dma_$TS.log
VARS=${VAR:-"8 2 3 4"}

{
    echo "================================================================================"
    echo "P1 探针 · arch22 DataCopyExtParams 字段口径裁定（单核受控）$(date '+%F %T')"
    echo "树=$(pwd) 主机=$(hostname) 变体(m)=$VARS"
    echo "形状固定 S=64 D=256 fp16 前向：dTileNum=1、splitMode=ROW、block_dim 强制 1"
    echo "kernel/harness/host md5:"
    md5sum op_kernel/mhc_expand.cpp op_host/mhc_expand.cpp npu_debug/test_mhc_expand_npu.cpp
    echo "--------------------------------------------------------------------------------"
} | tee "$LOG"

bash npu_debug/build_npu.sh >> "$LOG" 2>&1
rc=$?
echo "### build rc=$rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { tail -30 "$LOG"; exit 8; }

# 编译产物里确认探针真的进了 so（框架用的是 pkg 里那份）
ls -la --time-style=+%F_%T build_out/libcust_opapi.so | tee -a "$LOG"

for v in $VARS; do
    echo "" | tee -a "$LOG"
    echo "################ variant m=$v  dstStride 字段取值: $([ $v = 2 ] && echo 0 || ([ $v = 3 ] && echo rb=512B || ([ $v = 4 ] && echo rb/32=16 || echo "原逐副本循环(对照)"))) ################" | tee -a "$LOG"
    MHC_PROBE=$v bash npu_debug/run_npu.sh probe 50 64 >> "$LOG" 2>&1
    echo "### variant m=$v rc=$?" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "--------------------------------------------------------------------------------" | tee -a "$LOG"
echo "### 摘要（PASS/mismatch/越界/异常）" | tee -a "$LOG"
grep -nE "variant m=|probe-fp16|maxdiff|mismatch|nan_unwritten|not written|ALL PASS|fail=|errcode|out of range|sync failed|launch failed" "$LOG" | tee -a "$LOG"
echo "### log=$LOG md5=$(md5sum $LOG | awk '{print $1}')"
