#!/bin/bash
# fp32 large-g gate for the strided n<N_MAX path: f32_gate.sh only reaches g<=3 because
# its n<8 configs have batch <= 100 (batchPerCore = ceil(batch/40)). These two shapes put
# 26 matrices in one burst, which is what the timing grid measures.
set -u
T=/home/developer/mhc_test
B=/home/developer/mhc_build
CANN=/home/developer/Ascend/cann-9.0.0
source $CANN/set_env.sh 2>/dev/null
export ASCEND_CUSTOM_OPP_PATH=$B/myopp/vendors/custom
export LD_LIBRARY_PATH=$B/myopp/vendors/custom/op_api/lib:$CANN/aarch64-linux/lib64:$LD_LIBRARY_PATH
export DT=fp32
cd $T/harness || exit 1
[ -f test_sink ] || { echo "GATE FAIL=no test_sink"; exit 1; }

PASS=0; TOT=0
for cfg in "1024 6 20" "1024 4 20"; do
  set -- $cfg; TOT=$((TOT + 1))
  python3 $T/sinkhorn_ref.py gen $1 $2 $3 1e-6 /tmp/ri_$1_$2.bin /tmp/rr_$1_$2.bin 777 --f32 > /dev/null \
    || { echo "rand batch=$1 n=$2 | verdict=GEN_FAIL"; continue; }
  timeout 300 ./test_sink $1 $2 $3 1e-6 /tmp/ri_$1_$2.bin > $T/log/f32_r_$1_$2.txt 2>&1
  rc=$?
  if [ "$rc" != "0" ]; then echo "rand batch=$1 n=$2 | verdict=FAIL rc=$rc"; continue; fi
  out=$(python3 $T/sinkhorn_ref.py check $1 $2 $3 1e-6 /tmp/ri_$1_$2.bin /tmp/rr_$1_$2.bin --f32)
  echo "rand batch=$1 n=$2 | $out"
  echo "$out" | grep -q 'PASS' && PASS=$((PASS + 1))
done
echo "@@@@ f32_big done $PASS/$TOT"
