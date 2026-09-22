#!/bin/bash
# P40 同-session A/B：P38 改档的四个案（big1 / p_n64 / p_n512 / p_n1024）+ w4，
# 在【旧档 8/40/1】与【新档 4/48/1】之间交替各测 N 遍，取逐案最小值比。
# 为什么强制档而不是 AUTO vs AUTO：远端当前是带 PICK 补丁的构建，AUTO 已 = 新档；
#   要拿到"旧档"这一臂只能 SFA_FORCE_*，两臂同用强制口径才是同-session 可比。
# 交错顺序（旧,新,旧,新…）是为了抵消热漂移；⚠️ 只读性能，不做正确性结论。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
OUT=${OUT:-/tmp/p40_ab.txt}
PASSES=${PASSES:-4}
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
: > $OUT
for p in $(seq 1 $PASSES); do
  echo "=== PASS $p ===" | tee -a -i $OUT
  timeout 2400 ssh -F $S $H "$ENVR; \
    for cs in big1 p_n64 p_n512 p_n1024 w4; do \
      for arm in '8 40' '4 48'; do \
        set -- \$arm; \
        line=\$(env SFA_FORCE_NB=\$1 SFA_FORCE_NBLK=\$2 SFA_FORCE_KS=1 timeout 300 ./test_sfa_dev cases/\$cs.bin 3 none 2>&1 | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+|平均 [0-9.]+ ms|ERROR' | head -2 | tr '\n' ' '); \
        echo \"AB \$cs nb=\$1 kb=\$2 :: \$line\"; \
      done; \
    done" 2>&1 | grep -av Warning | tee -a -i $OUT
done
echo "P40_DONE"
