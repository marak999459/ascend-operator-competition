#!/bin/bash
# P34 探针：把 (nb, n_blk) 网格搬到**大形状**上重量，检验 §15.54(4) 假设①
#   "每个 nb 取最大可行 k"这条贪心是**发射受限小区间**的性质（§15.51 的 24 格全在 0.12~0.74 ms），
#   平台六点 7.8~14.7 ms 若已进入带宽/延迟受限区，最优点可能往左走。
# 口径：
#   * 用例是 gen_bigshape.py 造的**纯计时**案（expect 是哑值 ⇒ 只看 ms，不看超差）；
#   * 每格都回读完整 SFA_PICK 行，**pick != 标签的格一律作废**（P33 里 p2 的 nb>=2 格就是这样废的：
#     超出该形状头数上限的 nb 会被 host 静默降级成"兜底档 nb=1 nblk=16"）。
#   * ks 固定 2（ks=1 在每一案上都慢 1.8 倍，与本题无关）。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
OUT=/tmp/p34_big.txt
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
: > $OUT
echo "=== G) 造大形状计时案（纯输入 + 哑 expect）===" | tee -a $OUT
timeout 1200 ssh -F $S $H "cd ~/sfa_real && python3 gen_bigshape.py w1 w2 w3 w4 w5" 2>&1 \
  | grep -av Warning | tee -a $OUT
for pass in 1 2; do
echo "=== PASS $pass ===" | tee -a $OUT
timeout 2400 ssh -F $S $H "$ENVR; \
for cs in w1 w2 w3 w4 w5; do \
  a=\$(timeout 300 ./test_sfa_dev cases/\$cs.bin 3 none 2>&1 | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+|平均 [0-9.]+ ms' | tr '\n' ' '); \
  echo \"AUTO \$cs \$a\"; \
  for nb in 1 2; do \
    for kb in 16 32 40 48; do \
      line=\$(env SFA_FORCE_NB=\$nb SFA_FORCE_NBLK=\$kb SFA_FORCE_KS=2 timeout 300 ./test_sfa_dev cases/\$cs.bin 3 none 2>&1 | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+|平均 [0-9.]+ ms|ERROR|err=' | head -3 | tr '\n' ' '); \
      echo \"GRID \$cs nb=\$nb kb=\$kb :: \$line\"; \
    done; \
  done; \
done" 2>&1 | grep -av Warning | tee -a $OUT
done
echo "P34_BIG_DONE"
