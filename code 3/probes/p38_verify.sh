#!/bin/bash
# P38 闸门：把 P37 选出来的**赢家档**放到全部 golden 案上验正确性（逐位一致 + 超差）。
# 为什么必须单独验：P34/P37 用的是**哑 expect 计时案**，一个正确性结论都不产出；
#   而 `kvShard=1` 那句 host 注释明写"降级路径不赌并行度，退回与参考实现逐位一致的那条路"
#   ⇒ 强开某档有可能**破逐位一致**（§15.18/§15.35(c)）。
# 用法：bash p38_verify.sh <nb> <n_blk> <ks>     （例：bash p38_verify.sh 2 48 1）
# 判读：每行必须 `超差 0/N` 且 `PASS`；出现"与 golden 不逐位一致"要逐条列出来对比现档。
set -u
NB=${1:-2}; KB=${2:-48}; KS=${3:-1}
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
OUT=/tmp/p38_verify_${NB}_${KB}_${KS}.txt
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
GOLD='r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads p1 p2 p4 p6 big1 q1h q2h q1 q2 e1 e2 e3 e4 e5 e6 e7 e8'
: > $OUT
echo "=== 强制档 nb=$NB n_blk=$KB ks=$KS 的 golden 正确性（fp16 / fp32 各一遍）===" | tee -a $OUT
for f in 0 1; do
  tag=$([ "$f" = 1 ] && echo fp32 || echo fp16)
  timeout 2400 ssh -F $S $H "$ENVR; for cs in $GOLD; do \
    if [ $f = 1 ]; then out=\$(env SFA_F32=1 SFA_FORCE_NB=$NB SFA_FORCE_NBLK=$KB SFA_FORCE_KS=$KS timeout 200 ./test_sfa_dev cases/\$cs.bin 1 diff 2>&1); \
    else out=\$(env SFA_FORCE_NB=$NB SFA_FORCE_NBLK=$KB SFA_FORCE_KS=$KS timeout 200 ./test_sfa_dev cases/\$cs.bin 1 diff 2>&1); fi; \
    pick=\$(printf '%s' \"\$out\" | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+' | head -1); \
    tol=\$(printf '%s' \"\$out\" | grep -aoE '超差 [0-9]+/[0-9]+' | head -2 | tr '\n' ' '); \
    bit=\$(printf '%s' \"\$out\" | grep -ac '不逐位一致'); \
    res=\$(printf '%s' \"\$out\" | grep -aoE 'PASS|FAIL' | head -1); \
    printf '%-5s %-11s %-9s %-30s 非逐位行=%s %s\n' \"$tag\" \"\$cs\" \"\$res\" \"\$tol\" \"\$bit\" \"\$pick\"; \
  done" 2>&1 | grep -av Warning | tee -a $OUT
done
echo "=== 阴性对照：必须至少有一条 grep 能命中，否则上面的 0 是假的 ===" | tee -a $OUT
timeout 300 ssh -F $S $H "$ENVR; env SFA_FORCE_NB=1 SFA_FORCE_NBLK=99 SFA_FORCE_KS=1 timeout 200 ./test_sfa_dev cases/p1.bin 1 diff 2>&1 | grep -aoE 'SFA_PICK[^|]{0,60}|PASS|FAIL' | head -2" 2>&1 | grep -av Warning | tee -a $OUT
echo "NB=$NB KB=$KB KS=$KS" >> $OUT
echo "P38_VERIFY_DONE"
