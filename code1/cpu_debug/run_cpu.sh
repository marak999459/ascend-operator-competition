#!/bin/bash
# mhc_expand CPU 仿真运行脚本（VM 上执行）
# 与 build_cpu.sh 相同的 CANN 探测顺序
CANN_ENV=""
for p in "$HOME/Ascend/cann/cann-9.0.0/set_env.sh" "$HOME/Ascend/cann-9.0.0/set_env.sh" "$HOME/Ascend/cann/set_env.sh"; do
    [ -f "$p" ] && CANN_ENV="$p" && break
done
[ -n "$CANN_ENV" ] || { echo "ERROR: set_env.sh not found"; exit 1; }
source "$CANN_ENV"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/$(uname -m)-linux/simulator/dav_2201/lib:$ASCEND_HOME_PATH/toolkit/tools/tikicpulib/lib/Ascend910B1:$ASCEND_HOME_PATH/toolkit/tools/tikicpulib/lib:$ASCEND_HOME_PATH/toolkit/tools/simulator/Ascend910B1/lib:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH"
cd "$(dirname "$0")/.."
./test_expand_cpu "$@"
echo "=== exit: $? ==="
