#!/bin/bash
# P105（通用 kv_shard 深度）的发次闸门 = `p32_gate.sh` 的骨架 + 两件事：
#   ① GOLDEN 段补进 rows 阶梯档 `pm41/pm81/pm161/pm4full` —— 这些是**平台 replica 形状**且
#      只有它们真会走 `ks>=2` 的链 ⇒ 光测 p1/p2 不够（p1/p2 在旧档里走 ks=2，新档走 ks=5）。
#   ② 最后加一段 GATE3 = 同一构建两遍的 AUTO 选档计时 —— 证明"提交的那份字节自己选出的档"
#      就是网格里那个深档（网格读的是 SFA_FORCE_* 强制出来的档，旋钮不在提交源里）。
# ⚠️ 口径：`pm*` 一族的 expect 是 `p104_run.py` 故意填的**哑零**（那份 runner 的第 23 行自己
#    就写着"expect 全零 ⇒ 只读时间不读判据"）⇒ 这一族只读**时间**和"是否逐位一致"，
#    报"超差 22 万"不是回归（§7.3）。`big1` 有真 expect。判据看 `超差`，不看 `PASS` 字样。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
GOLD="r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads p1 p2 p4 p6 big1 pm41 pm81 pm161 pm4full"
EXPC="q1h q2h q3h p_n64 p_n512 p_n1024 e1empty e2one e3two e4odd e5s2one e6many e7padq e8padkv"
# 0) 重建 dev harness（§15.76(a)）：闸门读的是 ~/sfa_real/test_sfa_dev 这个**二进制**，
#    上一轮的 cube 探针会留下带 [CUBE] 回读判据的旧版本 ⇒ 数值全对也照样报 FAIL。
LINK="-I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include -I\$HOME/sfa_real/vendor/custom/op_api/include -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 -L\$HOME/sfa_real/vendor/custom/op_api/lib -lascendcl -lnnopbase -lcust_opapi"
echo "########## GATE0 harness 重建 ##########"
timeout 600 ssh -F $S $H "$ENVR; g++ -std=c++17 -O2 test_sfa_dev.cpp -o test_sfa_dev $LINK 2>&1 | head -20; ls -la test_sfa_dev" 2>&1 | grep -av Warning
[ -n "$(timeout 300 ssh -F $S $H "$ENVR; ./test_sfa_dev cases/big1.bin 1 diff 2>&1 | grep -a '\[CUBE\]' | head -1" 2>&1 | grep -av Warning)" ] && \
  { echo ">>> 探针残留：harness 输出里仍有 [CUBE] 段 ⇒ 闸门不可信，中止"; exit 1; }
for f in 0 1; do
  tag=$([ $f = 1 ] && echo fp32 || echo fp16)
  echo "########## GATE1 $tag ##########"
  timeout 2200 ssh -F $S $H "$ENVR; for cs in $GOLD; do line=\$(env $( [ $f = 1 ] && echo SFA_F32=1 ) ./test_sfa_dev cases/\$cs.bin 1 diff 2>&1 | grep -aoE 'out : dtype=[a-z0-9]+ 超差 [0-9]+/[0-9]+|超差 [0-9]+/[0-9]+|逐位一致 [^ ]+|不\*\*逐位一致\*\*|FAIL|PASS' | tr '\n' ' '); printf '%-6s %-10s %s\n' $tag \$cs \"\$line\"; done" 2>&1 | grep -av Warning
done
echo "########## GATE2 expect(双dtype) ##########"
timeout 2200 ssh -F $S $H "$ENVR; for cs in $EXPC; do a=\$(./test_sfa_dev cases/\$cs.bin 1 diff 2>&1 | grep -aoE '超差 [0-9]+/[0-9]+' | head -1); b=\$(env SFA_F32=1 ./test_sfa_dev cases/\$cs.bin 1 diff 2>&1 | grep -aoE '超差 [0-9]+/[0-9]+' | head -1); printf '%-10s fp16=%-10s fp32=%-10s\n' \$cs \"\$a\" \"\$b\"; done" 2>&1 | grep -av Warning
echo "########## GATE3 AUTO 选档计时(同一构建两遍) ##########"
timeout 1200 ssh -F $S $H "$ENVR; for r in 1 2; do echo \"-- 第 \$r 遍 --\"; for cs in p1 p2 pm41 pm81 pm161 p4 p6 big1; do t=\$(./test_sfa_dev cases/\$cs.bin 5 none 2>&1 | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//'); printf '%-8s %10s\n' \$cs \"\$t\"; done; done" 2>&1 | grep -av Warning
echo "P105_GATE_DONE"
