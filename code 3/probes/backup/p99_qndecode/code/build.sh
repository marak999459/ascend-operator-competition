#!/bin/bash
source /home/fszqsn/Ascend/cann/cann-9.0.0/set_env.sh
cd ~/sparse_flash_attention/op_kernel

TIKI_INC="$ASCEND_HOME_PATH/toolkit/tools/tikicpulib/lib/include"
KERNEL_DIR="$PWD"
X86="$ASCEND_HOME_PATH/x86_64-linux"

INCLUDES="-I$X86/include \
-I$X86/include/graph \
-I$X86/asc \
-I$X86/asc/include \
-I$X86/asc/include/basic_api \
-I$X86/asc/include/interface \
-I$X86/asc/include/c_api \
-I$X86/asc/include/simt_api \
-I$X86/asc/include/utils \
-I$X86/asc/impl/basic_api \
-I$X86/asc/impl/basic_api/dav_c220 \
-I$X86/asc/impl/utils \
-I$X86/asc/impl/c_api \
-I$X86/asc/impl/c_api/instr_impl/npu_arch_2201 \
-I$X86/ascendc/include \
-I$X86/ascendc/include/basic_api \
-I$X86/ascendc/include/basic_api/impl \
-I$X86/ascendc/include/c_api \
-I$TIKI_INC \
-I$KERNEL_DIR"

echo "=== COMPILE ==="
g++ -std=c++17 -DASCENDC_CPU_DEBUG -D__NPU_ARCH__=2201 \
    -include $X86/include/graph/c_types.h \
    $INCLUDES \
    -L$ASCEND_HOME_PATH/toolkit/tools/tikicpulib/lib/Ascend910B1 \
    -L$ASCEND_HOME_PATH/toolkit/tools/tikicpulib/lib \
    -L$ASCEND_HOME_PATH/lib64 \
    -Wl,--allow-shlib-undefined \
    test_sparse_flash_attention.cpp \
    -lcpudebug -lcpudebug_stubreg \
    -lc_sec \
    -lpthread -ldl \
    -o test_sparse_flash_attention 2>&1 | head -30

echo "exit: $?"
ls -la test_sparse_flash_attention 2>/dev/null
