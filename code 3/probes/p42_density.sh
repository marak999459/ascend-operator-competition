#!/bin/bash
# P42 探针：验 §15.55(7) 登记的最后一条风险 —— 新模型的 `toks = sparse_count × sbs` 是【上界】，
# host 看不见稀疏列表里有多少条是有效项（`big1` 就是活例子：COUNT=2048，但 `gen_big.py nblk=256`
# 只有 256 条有效）。若"有效项 ≪ COUNT"，chunk 数会被高估 ⇒ 模型会**过度推大 `k`、并顺带
# 把 `nb=4/k=48` 相对 `nb=8/k=40` 的优势放大**（P38 改档正是这一对）。
# 做法：形状全同（B=1 S1=128 S2=8192 N1=8 D=512 SBS=1 COUNT=2048），只把有效 token 数
#   扫 {64, 256, 1024, 2048} 四档，每档读 AUTO 选档 + 强制扫 4 个候选档，看 AUTO 是否仍贴最快格。
# ⚠️ 计时专用（`none` 口径），正确性一律回 golden 案；只作用于远端副本，跑完必须 dev.sh build 复原。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
OUT=${OUT:-/tmp/p42_density.txt}
PASSES=${PASSES:-2}
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'

{
echo "=== 0) 造密度族（gen_big.py <name> <nblk>，形状全同）==="
timeout 1200 ssh -F $S $H "cd ~/sfa_real && source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; for n in 'd64 64' 'd256 256' 'd1024 1024' 'd2048 2048'; do set -- \$n; python3 gen_big.py \$1 \$2 | tail -1; done"
echo "=== 1) 打 FORCE/PICK 补丁 + 重建 ==="
timeout 2400 ssh -F $S $H "cd ~/sfa_real && python3 ~/code\ 3/probes/p23_sweep_patch.py code/op_host/sparse_flash_attention.cpp && bash build.sh > /tmp/p42_build.log 2>&1; tail -1 /tmp/p42_build.log"
} 2>&1 | grep -av Warning | tee -a $OUT

for p in $(seq 1 $PASSES); do
  echo "=== PASS $p ===" | tee -a $OUT
  timeout 2400 ssh -F $S $H "$ENVR; \
    for cs in d64 d256 d1024 d2048; do \
      a=\$(timeout 300 ./test_sfa_dev cases/\$cs.bin 3 none 2>&1 | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+|平均 [0-9.]+ ms' | head -2 | tr '\n' ' '); \
      echo \"AUTO \$cs :: \$a\"; \
      for arm in '4 48 1' '8 40 1' '8 32 1' '2 48 1'; do \
        set -- \$arm; \
        line=\$(env SFA_FORCE_NB=\$1 SFA_FORCE_NBLK=\$2 SFA_FORCE_KS=\$3 timeout 300 ./test_sfa_dev cases/\$cs.bin 3 none 2>&1 | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+|平均 [0-9.]+ ms|ERROR' | head -2 | tr '\n' ' '); \
        echo \"G \$cs nb=\$1 kb=\$2 ks=\$3 :: \$line\"; \
      done; \
    done" 2>&1 | grep -av Warning | tee -a $OUT
done
echo "P42_DONE"
