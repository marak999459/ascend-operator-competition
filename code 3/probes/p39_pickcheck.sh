#!/bin/bash
# P39 探针：在真机上回读 P38 补丁后的 AUTO 选档，与离线复刻 p38_model_fit.py 的预测对账。
# 起因：P38 给代价模型补上了缺失的"chunk 数"这一维（§15.55），离线说只有 4/31 个本地案改档
#       （big1 / p_n64 / p_n512 / p_n1024：8/40/1 → 4/48/1）。**离线复刻不可全信**
#       （§15.53 已把 p25_model_check.py 的 pick() 判过死刑），所以这一发只读真机打印。
# 计时：REPS=1，纯读档，不产出任何性能/正确性结论。
# ⚠️ 只作用于远端副本 ~/sfa_real/code；本地提交源绝不落这个补丁。跑完必须 npu.sh sync + build 复原。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
OUT=${OUT:-/tmp/p39_pickcheck.txt}
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'

{
echo "=== 0) 打 PICK 补丁 + 重建 ==="
timeout 2400 ssh -F $S $H "cd ~/sfa_real && python3 ~/code\ 3/probes/p23_sweep_patch.py code/op_host/sparse_flash_attention.cpp && grep -c SFA_FORCE code/op_host/sparse_flash_attention.cpp && bash build.sh > /tmp/p39_build.log 2>&1; tail -2 /tmp/p39_build.log"
echo "=== 1) 逐案回读 AUTO 选档 ==="
timeout 3000 ssh -F $S $H "$ENVR; \
  for cs in \$(ls cases | sed 's/\.bin\$//' | sort); do \
    line=\$(timeout 300 ./test_sfa_dev cases/\$cs.bin 1 none 2>&1 | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+ sbs=[0-9]+ rows=[0-9]+ count=[0-9]+ need=[0-9]+ safe=[0-9]+ e=[0-9]+' | head -1); \
    echo \"PICK \$cs :: \$line\"; \
  done"
} 2>&1 | grep -av Warning | tee -a $OUT
echo "P39_DONE"
