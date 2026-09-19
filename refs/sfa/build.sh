#!/bin/bash
# SFA 真机构建 + 组装 vendor（照搬上道题 mhc_test/run.sh 的流程）
# 用法: build.sh
set -u
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
CANN=/home/developer/Ascend/cann-9.0.0
B=/home/developer/sfa_real/code
V=/home/developer/sfa_real/vendor/custom

cd "$B" || { echo "NO CODE DIR"; exit 2; }
rm -rf build && mkdir build && cd build

echo "=== CMAKE ==="
cmake .. -DCMAKE_BUILD_TYPE=Release > /tmp/sfa_cmake.log 2>&1
if [ $? -ne 0 ]; then
  echo "CMAKE FAIL"; grep -a -i -m5 "error" /tmp/sfa_cmake.log | head -20; exit 1
fi

echo "=== MAKE ==="
make -j8 > /tmp/sfa_make.log 2>&1
if [ $? -ne 0 ]; then
  echo "MAKE FAIL -- error 摘要:"
  grep -a -i "error" /tmp/sfa_make.log | head -30
  echo "... 完整日志: /tmp/sfa_make.log ($(wc -l < /tmp/sfa_make.log) 行)"
  exit 1
fi

[ -f "$B/build/libcust_opapi.so" ] || { echo "NO libcust_opapi.so"; exit 1; }
echo "构建 OK"

# ---- 组装 vendor 目录 ----
rm -rf "$V" && mkdir -p "$V"
cp -r "$B/build/tmp/vendors/custom/"* "$V/" 2>/dev/null
mkdir -p "$V/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64" "$V/op_api/lib" "$V/op_api/include"
cp "$B/build/op_host/libcustom_ascendc_cust_optiling.so" \
   "$V/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
cp "$B/build/libcust_opapi.so" "$V/op_api/lib/"
cp "$B/build/autogen/aclnn_sparse_flash_attention.h" "$V/op_api/include/" 2>/dev/null \
  || { echo "缺 autogen aclnn 头"; ls "$B/build/autogen/" | head; exit 1; }

echo "vendor 组装完成: $V"
echo "--- aclnn 头 ---"
ls -la "$V/op_api/include/"
echo "--- 生成的 kernel 二进制 ---"
find "$B/build" -name "*.o" -o -name "*.json" 2>/dev/null | grep -i sparse | head -10
