#!/bin/bash
# 在远程机器上构建并验证 mhc_sinkhorn 的原始版 / 修复版
set -u
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
CANN=/home/developer/Ascend/cann-9.0.0
B=/home/developer/mhc_build
V=$B/myopp/vendors/custom

build_and_test () {
  local tag="$1"      # orig / fixed
  echo "############################################################"
  echo "### 构建并测试: $tag"
  echo "############################################################"
  cd $B
  rm -rf build && mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release >/tmp/cmake_$tag.log 2>&1 || { echo "CMAKE FAIL"; tail -20 /tmp/cmake_$tag.log; return 1; }
  make -j8 >/tmp/make_$tag.log 2>&1 || { echo "MAKE FAIL"; tail -30 /tmp/make_$tag.log; return 1; }
  echo "构建 OK"

  # 组装 vendor 目录
  rm -rf $B/myopp && mkdir -p $V
  cp -r $B/build/tmp/vendors/custom/* $V/ 2>/dev/null
  mkdir -p $V/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64
  cp $B/build/op_host/libcustom_ascendc_cust_optiling.so \
     $V/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
  mkdir -p $V/op_api/lib $V/op_api/include
  cp $B/build/libcust_opapi.so $V/op_api/lib/
  cp $B/build/autogen/aclnn_mhc_sinkhorn.h $V/op_api/include/

  # kernel 名
  echo -n "kernel: "; ls $V/op_impl/ai_core/tbe/kernel/ascend910b/mhc_sinkhorn/*.o 2>/dev/null | head -2 | xargs -n1 basename | tr '\n' ' '; echo

  # 编译 harness
  cd /home/developer/mhc_test/harness
  g++ -std=c++17 -O2 test_mhc_sinkhorn.cpp -o test_sink \
    -I$CANN/aarch64-linux/include -I$V/op_api/include \
    -L$CANN/aarch64-linux/lib64 -L$V/op_api/lib \
    -lascendcl -lnnopbase -lcust_opapi 2>&1 | head -10
  [ -f test_sink ] || { echo "HARNESS BUILD FAIL"; return 1; }

  export ASCEND_CUSTOM_OPP_PATH=$B/myopp/vendors/custom
  export LD_LIBRARY_PATH=$V/op_api/lib:$CANN/aarch64-linux/lib64:$LD_LIBRARY_PATH

  for cfg in "8 8 20" "1024 8 20" "64 4 20" "100 6 20"; do
    set -- $cfg
    printf "  batch=%-5s n=%-2s iters=%-3s -> " "$1" "$2" "$3"
    timeout 70 ./test_sink $1 $2 $3 1e-6 > /tmp/t_${tag}_$1_$2.txt 2>&1
    rc=$?
    case $rc in
      0)   echo "成功";;
      2)   echo "设备崩溃";;
      124) echo "死锁超时";;
      *)   echo "退出码 $rc";;
    esac
  done
}

case "${1:-both}" in
  orig)  build_and_test orig ;;
  fixed) build_and_test fixed ;;
  *)     build_and_test orig; build_and_test fixed ;;
esac
