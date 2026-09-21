#!/bin/bash
# 成对测量：候选 kernel vs 基线 kernel，同一 session 顺序跑（跨 session 读数会漂）。
# 两边都用 host_cur.cpp 重建 + 6/6 正确性闸门 + msprof 设备时长。
# 用法: pair_run.sh <cand_kernel> <cand_label> <base_kernel> <base_label> [reps]
set -u
T=/home/developer/mhc_test
CAND="${1:?usage: pair_run.sh <cand> <clabel> <base> <blabel> [reps]}"
CLAB="${2:?missing cand label}"
BASE="${3:?missing base kernel}"
BLAB="${4:?missing base label}"
REPS="${5:-20}"
H=$T/host_cur.cpp
mkdir -p $T/log
echo "=== $(date -Iseconds) pair start cand=$CLAB base=$BLAB reps=$REPS"
bash $T/prof_bench.sh "$CAND" "$CLAB" "$REPS" 1 "$H" > "$T/log/pair_$CLAB.log" 2>&1;  echo "cand rc=$?"
bash $T/prof_bench.sh "$BASE" "$BLAB" "$REPS" 1 "$H" > "$T/log/pair_$BLAB.log" 2>&1;  echo "base rc=$?"
grep -h "@@@@\|^prof \|abort" "$T/log/pair_$CLAB.log" "$T/log/pair_$BLAB.log"
echo "=== $(date -Iseconds) pair done"
