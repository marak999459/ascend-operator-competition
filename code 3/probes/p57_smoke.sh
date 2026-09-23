#!/bin/bash
# M1d 交接形态的**冒烟**档：先重建 harness（§15.76(a) 那条假 FAIL 的根源），再逐案带
# per-case timeout 跑 4 个点。死锁的特征是"某一行永远不出现"⇒ 这里每案独立 90 s 超时，
# 超时就是挂（不像 p32_gate.sh 那样一案挂掉烧掉整臂）。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
LINK="-I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include -I\$HOME/sfa_real/vendor/custom/op_api/include -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 -L\$HOME/sfa_real/vendor/custom/op_api/lib -lascendcl -lnnopbase -lcust_opapi"
CASES="${CASES:-r1_min r2_chunk r3_mode3 r4_shortkv big1}"
TMO="${TMO:-90}"
timeout 900 ssh -F $S $H "$ENVR; g++ -std=c++17 -O2 test_sfa_dev.cpp -o test_sfa_dev $LINK 2>&1 | head -20; ls -la test_sfa_dev; \
  for cs in $CASES; do \
    timeout $TMO ./test_sfa_dev cases/\$cs.bin 1 diff > /tmp/\$cs.log 2>&1; rc=\$?; \
    out=\$(grep -aoE '超差 [0-9]+/[0-9]+|逐位一致 [^ ]+|FAIL|PASS' /tmp/\$cs.log | tr '\n' ' '); \
    printf '%-12s rc=%-4s %s\n' \$cs \$rc \"\$out\"; \
  done" 2>&1 | grep -av Warning
echo "SMOKE_DONE"
