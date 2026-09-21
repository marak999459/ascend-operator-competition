#!/bin/bash
# fp32 设备闸门（比赛契约 = float32 进/出；历史 6/6 全是 fp16 口径，fp32 分支从未上过机）。
# A 段：DT=fp32 常量输入 6 配置矩阵（判据 rc=0 且 harness 自评 PASS）
# B 段：DT=fp32 随机输入 + sinkhorn_ref.py --f32 元素级对拍（判据 异常矩阵 0/batch）
# 用法: f32_gate.sh <kernel文件> [host文件] [build:1|0]
set -u
T=/home/developer/mhc_test
B=/home/developer/mhc_build
CANN=/home/developer/Ascend/cann-9.0.0
KERNEL="${1:?usage: f32_gate.sh <kernel> [host] [build]}"
HOST="${2:-$T/host_cur.cpp}"
BUILD="${3:-1}"
source $CANN/set_env.sh 2>/dev/null
V=$B/myopp/vendors/custom
LBL=$(basename "$KERNEL" .cpp)

if [ "$BUILD" = "1" ]; then
  bash $T/run.sh "$KERNEL" "$HOST" $T/tiling_cur.h > $T/log/f32build_$LBL.log 2>&1
  brc=$?
  ok=$(grep -ac "成功" $T/log/f32build_$LBL.log)
  echo "@@@@ f32_gate build_rc=$brc fp16_cases_ok=$ok kernel_md5=$(md5sum "$KERNEL" | cut -c1-8) host_md5=$(md5sum "$HOST" | cut -c1-8)"
  [ "$ok" = "6" ] || { echo "### abort: fp16 矩阵不是 6/6（见 log/f32build_$LBL.log）"; exit 1; }
fi

export ASCEND_CUSTOM_OPP_PATH=$V
export LD_LIBRARY_PATH=$V/op_api/lib:$CANN/aarch64-linux/lib64:$LD_LIBRARY_PATH
export DT=fp32
cd $T/harness || exit 1
[ -f test_sink ] || { echo "GATE FAIL=no test_sink"; exit 1; }

PASS=0; TOT=0
for cfg in "8 8 20" "1024 8 20" "64 4 20" "100 6 20" "1 8 20" "40 6 3"; do
  set -- $cfg; TOT=$((TOT + 1))
  timeout 120 ./test_sink $1 $2 $3 1e-6 > $T/log/f32_c_$1_$2_$3.txt 2>&1
  rc=$?
  dev=$(grep -a '与 1/n' $T/log/f32_c_$1_$2_$3.txt | sed -n 's/.*最大偏差 = \([0-9.eE+-]*\).*/\1/p')
  ds=$(grep -a '双随机性' $T/log/f32_c_$1_$2_$3.txt | sed -n 's/.*最大偏差 = \([0-9.eE+-]*\).*/\1/p')
  if [ "$rc" = "0" ] && grep -aq '<<< PASS' $T/log/f32_c_$1_$2_$3.txt; then v=PASS; PASS=$((PASS + 1)); else v=FAIL; fi
  echo "const batch=$1 n=$2 iters=$3 | verdict=$v rc=$rc dev_vs_1n=${dev:-NA} double_stich=${ds:-NA}"
done
echo "@@@@ f32 常量输入 cases_ok=$PASS/$TOT"

for cfg in "64 8 20" "40 6 7" "33 4 20"; do
  set -- $cfg
  python3 $T/sinkhorn_ref.py gen $1 $2 $3 1e-6 /tmp/ri_$1_$2_$3.bin /tmp/rr_$1_$2_$3.bin 777 --f32 > /dev/null \
    || { echo "rand batch=$1 n=$2 iters=$3 | verdict=GEN_FAIL"; continue; }
  timeout 120 ./test_sink $1 $2 $3 1e-6 /tmp/ri_$1_$2_$3.bin > $T/log/f32_r_$1_$2_$3.txt 2>&1
  rc=$?
  if [ "$rc" != "0" ]; then
    echo "rand batch=$1 n=$2 iters=$3 | verdict=FAIL rc=$rc"
    continue
  fi
  python3 $T/sinkhorn_ref.py check $1 $2 $3 1e-6 /tmp/ri_$1_$2_$3.bin /tmp/rr_$1_$2_$3.bin --f32 \
    | sed "s/^/rand batch=$1 n=$2 iters=$3 | /"
done
echo "@@@@ f32_gate done kernel=$LBL"
