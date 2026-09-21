#!/bin/bash
# mhc_sinkhorn 仿真机构建（本地仿真机 / 云端仿真机通用，x86_64 与 aarch64 自适应）
# 编译的是提交源码本身：cpu_debug harness 直接 #include op_kernel/mhc_sinkhorn.cpp
set -o pipefail

CANN_ENV=""
for p in "$HOME/Ascend/cann/cann-9.0.0/set_env.sh" "$HOME/Ascend/cann-9.0.0/set_env.sh" "$HOME/Ascend/cann/set_env.sh"; do
    [ -f "$p" ] && CANN_ENV="$p" && break
done
[ -n "$CANN_ENV" ] || { echo "ERROR: set_env.sh not found"; exit 1; }
source "$CANN_ENV"
cd "$(dirname "$0")/.."

# 提交源码目录。cwd 已 cd 到包根，候选按两种真实布局排：
#   云端 ~/ops_comp/code2/{cpu_debug,op_kernel}  → "op_kernel"（与 code1 逐字节同构）
#   本地 .../refs/harness_sinkhorn/cpu_debug     → "../../code2/op_kernel"
KD=""
for d in "$KERNEL_DIR" "op_kernel" "../../code2/op_kernel"; do
    [ -n "$d" ] && [ -f "$d/mhc_sinkhorn.cpp" ] && KD="$(cd "$d" && pwd)" && break
done
[ -n "$KD" ] || { echo "ERROR: op_kernel/mhc_sinkhorn.cpp not found (可设 KERNEL_DIR 指定)"; exit 1; }
echo "kernel dir: $KD"

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
-I$TIKI_INC -I$KD"

BLOG="$(dirname "$0")/build_cpu.log"   # 日志写工程目录：/tmp 会被容器重启清空
g++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -DASCENDC_CPU_DEBUG -D__NPU_ARCH__=2201 \
    -include "$X86/include/graph/c_types.h" \
    $INCLUDES \
    -L"$ASCEND_HOME_PATH/toolkit/tools/tikicpulib/lib/Ascend910B1" \
    -L"$ASCEND_HOME_PATH/toolkit/tools/tikicpulib/lib" \
    -L"$ASCEND_HOME_PATH/toolkit/tools/simulator/Ascend910B1/lib" \
    -L"$X86/simulator/dav_2201/lib" \
    -L"$ASCEND_HOME_PATH/lib64" \
    -Wl,--allow-shlib-undefined \
    cpu_debug/test_mhc_sinkhorn_cpu.cpp \
    -lpem_davinci -lcpudebug -lcpudebug_stubreg -lcpudebug_cceprint -lcpudebug_npuchk \
    -lc_sec -lpthread -ldl \
    -o test_sinkhorn_cpu 2> "$BLOG"

rc=$?
echo "build exit: $rc"
grep -E "error:|undefined reference" "$BLOG" | head -30
exit $rc
