#!/bin/bash
# SFA 对拍测试矩阵（probe_v3：文件系统同步，规避 CPU 仿真多进程竞态）
# 用法: run_matrix.sh [NB] [N_BLK] [重复次数]
set -u
WORK=/home/fszqsn/sfa_work
NB="${1:-8}"
NBLK="${2:-4}"
REP="${3:-1}"
H=/home/fszqsn/Ascend/cann/cann-9.0.0
export LD_LIBRARY_PATH="$H/toolkit/tools/tikicpulib/lib/Ascend910B1:$H/toolkit/tools/tikicpulib/lib:$H/lib64:${LD_LIBRARY_PATH:-}"
cd "$WORK" || exit 1

CASES="${CASES:-c1_min c2_chunk c3_mode3 c4_shortkv c5_blocks c6_multiB c7_norope c8_sbs2 case_small mini}"

echo "############ SFA 对拍  NB=$NB N_BLK=$NBLK rep=$REP ############"
pass=0; fail=0
for f in $CASES; do
  [ -f "$f.bin" ] || { printf "%-12s SKIP\n" "$f"; continue; }
  sc=0
  line=""
  for r in $(seq 1 "$REP"); do
    timeout 900 ./op_kernel/probe_v3 "$f.bin" "$NB" "$NBLK" > /tmp/m_$f.txt 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then sc=$((sc+1)); fi
    line=$(grep -a -E "abs=" /tmp/m_$f.txt | tail -1)
  done
  if [ "$sc" -eq "$REP" ]; then st="PASS"; pass=$((pass+1)); else st="FAIL($sc/$REP)"; fail=$((fail+1)); fi
  printf "%-12s %-9s %s\n" "$f" "$st" "$line"
done
echo "-----------------------------------------------------------"
echo "PASS=$pass  FAIL=$fail"
[ "$fail" -eq 0 ] && exit 0 || exit 2
