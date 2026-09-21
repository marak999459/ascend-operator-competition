#!/bin/bash
# mhc_expand CPU 仿真构建脚本（VM 上执行）
set -o pipefail
# CANN 根路径按环境自动探测：VM 布局 ~/Ascend/cann/cann-9.0.0，云主机布局 ~/Ascend/cann-9.0.0
CANN_ENV=""
for p in "$HOME/Ascend/cann/cann-9.0.0/set_env.sh" "$HOME/Ascend/cann-9.0.0/set_env.sh" "$HOME/Ascend/cann/set_env.sh"; do
    [ -f "$p" ] && CANN_ENV="$p" && break
done
[ -n "$CANN_ENV" ] || { echo "ERROR: set_env.sh not found"; exit 1; }
source "$CANN_ENV"
cd "$(dirname "$0")/.."

# 架构自适应：x86_64 主机 -> x86_64-linux，aarch64 主机 -> aarch64-linux
X86="$ASCEND_HOME_PATH/$(uname -m)-linux"
TIKI_INC="$ASCEND_HOME_PATH/toolkit/tools/tikicpulib/lib/include"
INCLUDES="-I$X86/include -I$X86/include/graph \
-I$X86/asc -I$X86/asc/include -I$X86/asc/include/basic_api \
-I$X86/asc/include/interface -I$X86/asc/include/c_api \
-I$X86/asc/include/simt_api -I$X86/asc/include/utils \
-I$X86/asc/impl/basic_api -I$X86/asc/impl/basic_api/dav_c220 \
-I$X86/asc/impl/utils -I$X86/asc/impl/c_api \
-I$X86/asc/impl/c_api/instr_impl/npu_arch_2201 \
-I$X86/ascendc/include -I$X86/ascendc/include/basic_api \
-I$X86/ascendc/include/basic_api/impl -I$X86/ascendc/include/c_api \
-I$TIKI_INC -I./op_kernel"

BLOG="$(dirname "$0")/build_cpu.log"   # 日志写工程目录：/tmp 会被容器重启清空
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -DASCENDC_CPU_DEBUG -D__NPU_ARCH__=2201 \
    -include "$X86/include/graph/c_types.h" \
    $INCLUDES \
    -L"$ASCEND_HOME_PATH/toolkit/tools/tikicpulib/lib/Ascend910B1" \
    -L"$ASCEND_HOME_PATH/toolkit/tools/tikicpulib/lib" \
    -L"$ASCEND_HOME_PATH/toolkit/tools/simulator/Ascend910B1/lib" \
    -L"$X86/simulator/dav_2201/lib" \
    -L"$ASCEND_HOME_PATH/lib64" \
    -Wl,--allow-shlib-undefined \
    cpu_debug/test_mhc_expand_cpu.cpp \
    -lpem_davinci -lcpudebug -lcpudebug_stubreg -lcpudebug_cceprint -lcpudebug_npuchk \
    -lc_sec -lpthread -ldl \
    -o test_expand_cpu 2> "$BLOG"

rc=$?
echo "build exit: $rc"
grep -E "error:|undefined reference" "$BLOG" | head -20
exit $rc
