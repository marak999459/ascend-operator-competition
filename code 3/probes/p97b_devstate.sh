#!/bin/bash
# P97b：设备状态分诊 —— "只有 MIX/cube 起不来"还是"整卡废了"？
# 全程只动远端副本（npu.sh build 每次都重打 SoC 补丁，sync 会还原），提交源字节不碰。
set -u
REPO=/home/fszqsn/ops_comp/ascend-operator-competition
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ns() { timeout "${T:-300}" ssh -F $S -o ConnectTimeout=25 $H "$@" 2>&1 | grep -av Warning; }

RUN='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; \
 export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; \
 export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH; \
 export ASCEND_SLOG_PRINT_TO_STDOUT=1; export ASCEND_GLOBAL_LOG_LEVEL=4; \
 for c in p1 p6 w3 k_n16s64; do \
   timeout 90 ./test_sfa_dev cases/$c.bin 1 diff > /tmp/run_$c.log 2>&1; echo "$c rc=$? | $(grep -aE "平均|超差|逐位一致|==. (PASS|FAIL)|\[FAIL\]" /tmp/run_$c.log | tr "\n" " ")"; \
 done; \
 echo "---- p6 全文里的报错行 ----"; \
 grep -aiE "ERROR|errcode|exception|aicore|hang|timeout|reset" /tmp/run_p6.log | head -12'

echo "########## 1) 当前构建（cube on）$(date +%H:%M:%S) ##########"
ns "$RUN"
echo "########## 2) 远端把 SFA_CUBE_ON 改 0 后重建 $(date +%H:%M:%S) ##########"
ns "cd ~/sfa_real && sed -i 's/^constexpr uint32_t SFA_CUBE_ON = 1U;/constexpr uint32_t SFA_CUBE_ON = 0U;/' code/op_host/sparse_flash_attention.cpp && grep -n 'SFA_CUBE_ON = ' code/op_host/sparse_flash_attention.cpp | head -2"
( cd "$REPO" && timeout 1800 bash "code 3/npu_debug/npu.sh" build ) 2>&1 | grep -aE '构建 OK|error:' | head -3
ns "$RUN"
echo "########## 3) 还原提交源构建 $(date +%H:%M:%S) ##########"
( cd "$REPO" && timeout 300 bash "code 3/npu_debug/npu.sh" sync ) >/dev/null 2>&1
( cd "$REPO" && timeout 1800 bash "code 3/npu_debug/npu.sh" build ) 2>&1 | grep -aE '构建 OK|error:' | head -2
