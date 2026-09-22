#!/bin/bash
# P37 探针：在 **5~16 ms 量级**（= 平台六点的量级）把 (nb, n_blk, ks) **三轴全展开**重扫一遍。
# 起因（§15.54(7) 的 P34）：P34 为了省时间把 ks 钉死在 2、nb 只扫到 2，结果发现
#   w4(行×头=1024) 上 AUTO=nb8/k40/ks1 = 10.495 ms，而 nb2/k48/ks2 = 9.688 ms（−7.7 %）——
#   **一次改了三个变量**，分不清是 nb 档选错还是 ks 被 `Ks2Allowed` 门卡住。
# 这一发把三轴全部叉乘，每格回读 SFA_PICK（pick != 标签 ⇒ 该格作废）。
# ⚠️ 计时专用：用例 expect 是哑值，**这一发不产出任何正确性结论**；赢家另案在 golden 案上验。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
OUT=/tmp/p37_fullbig.txt
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
: > $OUT
for pass in 1 2; do
echo "=== PASS $pass ===" | tee -a $OUT
timeout 2400 ssh -F $S $H "$ENVR; \
for cs in w2 w3 w4; do \
  for nb in 1 2 4 8; do \
    for kb in 32 40 48; do \
      for ks in 1 2; do \
        line=\$(env SFA_FORCE_NB=\$nb SFA_FORCE_NBLK=\$kb SFA_FORCE_KS=\$ks timeout 300 ./test_sfa_dev cases/\$cs.bin 3 none 2>&1 | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+|平均 [0-9.]+ ms|ERROR|err=' | head -3 | tr '\n' ' '); \
        echo \"G \$cs nb=\$nb kb=\$kb ks=\$ks :: \$line\"; \
      done; \
    done; \
  done; \
done" 2>&1 | grep -av Warning | tee -a $OUT
done
echo "P37_DONE"
