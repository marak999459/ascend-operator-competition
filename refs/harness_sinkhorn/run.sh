#!/bin/bash
# 构建指定 kernel 版本并跑一组用例
# 用法: run.sh <kernel文件> [host文件] [tiling文件]
set -u
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
CANN=/home/developer/Ascend/cann-9.0.0
B=/home/developer/mhc_build
T=/home/developer/mhc_test
V=$B/myopp/vendors/custom

KERNEL="${1:-$T/fixed_kernel.cpp}"
HOST="${2:-$T/host_orig.cpp}"
TILING="${3:-$T/orig_tiling.h}"

cp "$KERNEL" $B/op_kernel/mhc_sinkhorn.cpp
cp "$HOST"   $B/op_host/mhc_sinkhorn.cpp
cp "$TILING" $B/op_kernel/mhc_sinkhorn_tiling.h

cd $B && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release >/tmp/cmake.log 2>&1 || { echo "CMAKE FAIL"; grep -a -A5 error /tmp/cmake.log|head -20; exit 1; }
make -j8 >/tmp/make.log 2>&1 || { echo "MAKE FAIL"; grep -a -i error /tmp/make.log|head -20; exit 1; }
[ -f $B/build/libcust_opapi.so ] || { echo "NO SO"; exit 1; }
echo "构建 OK"

rm -rf $B/myopp && mkdir -p $V
cp -r $B/build/tmp/vendors/custom/* $V/ 2>/dev/null
mkdir -p $V/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64 $V/op_api/lib $V/op_api/include
cp $B/build/op_host/libcustom_ascendc_cust_optiling.so \
   $V/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
cp $B/build/libcust_opapi.so $V/op_api/lib/
cp $B/build/autogen/aclnn_mhc_sinkhorn.h $V/op_api/include/

cd $T/harness
rm -f test_sink   # 先删旧二进制：否则编译失败会被残留 test_sink 掩盖（HARNESS FAIL 门失效）
g++ -std=c++17 -O2 test_mhc_sinkhorn.cpp -o test_sink \
  -I$CANN/aarch64-linux/include -I$V/op_api/include \
  -L$CANN/aarch64-linux/lib64 -L$V/op_api/lib \
  -lascendcl -lnnopbase -lcust_opapi 2>&1 | head -6
[ -f test_sink ] || { echo "HARNESS FAIL"; exit 1; }

export ASCEND_CUSTOM_OPP_PATH=$V
export LD_LIBRARY_PATH=$V/op_api/lib:$CANN/aarch64-linux/lib64:$LD_LIBRARY_PATH

mkdir -p $T/log
for cfg in "8 8 20" "1024 8 20" "64 4 20" "100 6 20" "1 8 20" "8192 8 20"; do
  set -- $cfg
  printf "  batch=%-6s n=%-2s iters=%-3s -> " "$1" "$2" "$3"
  timeout 70 ./test_sink $1 $2 $3 1e-6 > $T/log/r_$1_$2.txt 2>&1
  rc=$?
  # CASE= 是给上层门用的机器可读判据：rc=0 只代表"没崩"，数值 PASS 必须看 harness 的
  # <<< PASS 标记（历史上门只看"成功"二字，把一条数值全错的 kernel 放过去过）。
  case $rc in
    0)   if grep -aq '<<< PASS' $T/log/r_$1_$2.txt; then v=CASE=PASS; else v=CASE=FAIL; fi
         echo "$v 成功   | $(grep -a '最大偏差' $T/log/r_$1_$2.txt | tail -1 | cut -c1-110)";;
    2)   echo "CASE=FAIL 设备崩溃";;
    124) echo "CASE=FAIL 死锁超时";;
    *)   echo "CASE=FAIL 退出码 $rc";;
  esac
done

# ---- 计时阶段（BENCH=1 时执行）----
# 口径：single_* = 每次 launch 后同步；burst_avg = 连发 reps 次再同步（含启动开销下界）
# 要看纯 kernel 时长用 PROF=1 再跑一遍（msprof 产物在 $T/prof）
if [ "${BENCH:-0}" = "1" ]; then
  g++ -std=c++17 -O2 $T/bench_sinkhorn.cpp -o bench_sink \
    -I$CANN/aarch64-linux/include -I$V/op_api/include \
    -L$CANN/aarch64-linux/lib64 -L$V/op_api/lib \
    -lascendcl -lnnopbase -lcust_opapi 2>&1 | grep -a "error:" | head -6
  [ -f bench_sink ] || { echo "BENCH HARNESS FAIL"; exit 1; }
  echo "=== 计时 (reps=${REPS:-50} dtype=${DT:-fp16}) ==="
  for cfg in "20 6 20" "1 4 20" "1 8 20" "64 8 20" "100 6 20" "1024 8 20" "8192 8 20" "20 6 100"; do
    set -- $cfg
    if [ "${PROF:-0}" = "1" ]; then
      timeout 300 msprof --output=$T/prof --application="./bench_sink $1 $2 $3 ${REPS:-20} ${DT:-fp16}" \
        > $T/log/prof_$1_$2_$3.log 2>&1
      printf "  batch=%-6s n=%-2s iters=%-3s | " "$1" "$2" "$3"
      grep -ah mhc_sinkhorn $T/prof/*/msprof*.csv 2>/dev/null | grep -a "aicore\|AiCore\|op_time" | head -2
      [ -s "$T/log/prof_$1_$2_$3.log" ] && grep -a "batch=" $T/log/prof_$1_$2_$3.log | head -1
    else
      timeout 300 ./bench_sink $1 $2 $3 ${REPS:-50} ${DT:-fp16} 2>&1 | grep -a "batch=\|FAIL"
    fi
  done
fi
