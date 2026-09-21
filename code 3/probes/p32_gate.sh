#!/bin/bash
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
GOLD="r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads p1 p2 p4 p6 big1"
EXPC="q1h q2h q3h p_n64 p_n512 p_n1024 e1empty e2one e3two e4odd e5s2one e6many e7padq e8padkv"
for f in 0 1; do
  tag=$([ $f = 1 ] && echo fp32 || echo fp16)
  echo "########## GATE1 $tag ##########"
  timeout 2200 ssh -F $S $H "$ENVR; for cs in $GOLD; do line=\$(env $( [ $f = 1 ] && echo SFA_F32=1 ) ./test_sfa_dev cases/\$cs.bin 1 diff 2>&1 | grep -aoE 'out : dtype=[a-z0-9]+ 超差 [0-9]+/[0-9]+|超差 [0-9]+/[0-9]+|逐位一致 [^ ]+|不\*\*逐位一致\*\*|FAIL|PASS' | tr '\n' ' '); printf '%-6s %-10s %s\n' $tag \$cs \"\$line\"; done" 2>&1 | grep -av Warning
done
echo "########## GATE2 expect(双dtype) ##########"
timeout 2200 ssh -F $S $H "$ENVR; for cs in $EXPC; do a=\$(./test_sfa_dev cases/\$cs.bin 1 diff 2>&1 | grep -aoE '超差 [0-9]+/[0-9]+' | head -1); b=\$(env SFA_F32=1 ./test_sfa_dev cases/\$cs.bin 1 diff 2>&1 | grep -aoE '超差 [0-9]+/[0-9]+' | head -1); printf '%-10s fp16=%-10s fp32=%-10s\n' \$cs \"\$a\" \"\$b\"; done" 2>&1 | grep -av Warning
echo "P32_GATE_DONE"
