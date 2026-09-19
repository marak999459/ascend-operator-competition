#!/bin/bash
# CPU 仿真构建 + 运行（可复用，路径参数化）
# 用法: build_test.sh <workdir> [run_args...]
#   workdir  : 含 op_kernel/ 的目录
#   run_args : 透传给被测程序
set -u
WORK="${1:-/home/fszqsn/sfa_work}"
shift 2>/dev/null || true

source /home/fszqsn/Ascend/cann/cann-9.0.0/set_env.sh 2>/dev/null
H="$ASCEND_HOME_PATH"
KD="$WORK/op_kernel"
TIKI_INC="$H/toolkit/tools/tikicpulib/lib/include"
X86="$H/x86_64-linux"

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
-I$KD"

TESTFILE="${TESTFILE:-test_sparse_flash_attention.cpp}"
OUTBIN="${OUTBIN:-test_sfa}"

cd "$KD" || { echo "NO WORKDIR $KD"; exit 2; }

# ⚠️ 防陈旧构建：先删掉旧二进制。否则编译失败时 run 步骤会跑到【上一次】的产物，
#    造出"结果自相矛盾"的假结论（HANDOFF §7 教训 2 就这样坑过一次）。
rm -f "$KD/$OUTBIN"

echo "=== BUILD ($(md5sum $TESTFILE | cut -c1-8)) $(date +%H:%M:%S) ==="
LOG="$KD/build.log"
g++ -std=c++17 -DASCENDC_CPU_DEBUG -D__NPU_ARCH__=2201 \
    -include $X86/include/graph/c_types.h \
    $INCLUDES \
    -L$H/toolkit/tools/tikicpulib/lib/Ascend910B1 \
    -L$H/toolkit/tools/tikicpulib/lib \
    -L$H/lib64 \
    -Wl,--allow-shlib-undefined \
    "$TESTFILE" \
    -lcpudebug -lcpudebug_stubreg \
    -lc_sec \
    -lpthread -ldl \
    -o "$OUTBIN" >"$LOG" 2>&1
RC=$?

if [ $RC -ne 0 ] || [ ! -f "$OUTBIN" ]; then
  echo "BUILD FAIL (rc=$RC) -- error 摘要:"
  # 只抓真错误（error:），不带上下文噪音
  grep -n "error:" "$LOG" | head -25
  echo "..."
  echo "完整日志: $LOG ($(wc -l <"$LOG") 行)"
  exit 1
fi
echo "BUILD OK -> $KD/$OUTBIN  (warnings: $(grep -c 'warning:' "$LOG"), 二进制时间 $(date -r "$OUTBIN" +%H:%M:%S))"
echo "=== RUN ==="
export LD_LIBRARY_PATH="$H/toolkit/tools/tikicpulib/lib/Ascend910B1:$H/toolkit/tools/tikicpulib/lib:$H/lib64:${LD_LIBRARY_PATH:-}"
timeout 600 "./$OUTBIN" "$@"
echo "=== rc=$? ==="
