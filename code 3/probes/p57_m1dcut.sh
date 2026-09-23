#!/bin/bash
# P57：M1d 首发在真机上【第一个用例就卡死】。这一发把跨核旗标按侧剪掉，
# 用三档联合读数把"旗标路由错"与"AIC 自己算不完"分开（判据表见 m1d_cut.py 文件头）。
# ⚠️ 纯诊断：剪掉旗标之后的数值一律不作正确性结论（没有流控 ⇒ 环槽必然竞争）。
# ⚠️ 只改远端副本 ~/sfa_real/code，跑完复原；本地提交源一行都不动。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
CS=${CS:-"r1_min r2_chunk r3_mode3"}
OUT=/tmp/p57_m1dcut.txt
: > $OUT

# 0) 推诊断脚本 + 存一份"未剪"的原始 kernel（复原用）
tar cf - -C "$REPO/code 3" probes/m1d_cut.py probes/p57_run.sh \
  | timeout 180 ssh -F $S -o ConnectTimeout=25 $H "cd ~ && tar xf -" 2>&1 | grep -av Warning
timeout 120 ssh -F $S $H "cd ~/sfa_real && cp -f $K m1d_orig.cpp && sed -i 's/\r\$//' ~/probes/p57_run.sh && grep -c 'CrossCore' m1d_orig.cpp" 2>&1 \
  | grep -av Warning | sed 's/^/ORIG_CrossCore_lines=/'

for mode in all aiv aic; do
  echo "########## CUT $mode ##########" | tee -a $OUT
  timeout 200 ssh -F $S $H "cd ~/sfa_real && cp -f m1d_orig.cpp $K && \
    python3 ~/probes/m1d_cut.py $K $mode && grep -c '^// \[CUT\]' $K" 2>&1 \
    | grep -av Warning | tail -2 | sed 's/^/  /' | tee -a $OUT
  timeout 900 ssh -F $S $H "cd ~/sfa_real && bash build.sh" 2>&1 \
    | grep -aE "构建 OK|error:|FAILED" | head -3 | sed 's/^/  /' | tee -a $OUT
  timeout 700 ssh -F $S $H "$ENVR; CASES='$CS' TO=90 bash ~/probes/p57_run.sh" 2>&1 \
    | grep -av Warning | sed 's/^/  /' | tee -a $OUT
done
timeout 120 ssh -F $S $H "cd ~/sfa_real && cp -f m1d_orig.cpp $K && grep -c 'CrossCore' $K" 2>&1 \
  | grep -av Warning | sed 's/^/RESTORED_CrossCore_lines=/'
echo "P57_CUT_DONE"
