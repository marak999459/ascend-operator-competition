#!/bin/bash
# P56-S1 闸门链：把 MIX 形态的真提交源推到远端、构建、跑 27 例双 dtype 权威闸门。
# 用途：确认 K1/K2/K3/H1 四条落到提交源之后 AIV 路径零退化（本地口径），
#       然后才允许往同一份源里加 Cube（M1d）代码。
# 日志里 "SNAP_MD5" 之后本地源可以改（远端用的是已经推过去的那一份）。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
LOG="$REPO/code 3/npu_debug/logs/p56_s1_gate.log"
{
  echo "=== start $(date +%H:%M:%S) ==="
  md5sum "$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp" \
         "$REPO/code 3/code/op_host/sparse_flash_attention.cpp"
  bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -12
  echo "SNAP_MD5 已推送，本地源可以改"
  bash "$REPO/code 3/npu_debug/npu.sh" build 2>&1 | tail -25
  echo "=== build done $(date +%H:%M:%S) ==="
  bash "$REPO/code 3/probes/p32_gate.sh" 2>&1
  echo "=== gate done $(date +%H:%M:%S) ==="
} > "$LOG" 2>&1
