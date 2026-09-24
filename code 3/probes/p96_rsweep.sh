#!/bin/bash
# P96：回程环行距锁死成 n_blk ⇒ ScoreFromRing 从"每头一条 DataCopy"折成一条。
# 判据：时间对 P93 的 full 臂（w3 2.8507 / w4 4.4321 / k_n16s64 4.5353 / p6 0.7358 ms），
#      数值只看 `超差` 必须为 0（换的是搬运布局，不该动数学）。
# 没有 arm 循环：这一发直接改的是提交源，所以只 sync+build 一次，跑完再核 sha256。
set -u
REPO=/home/fszqsn/ops_comp/ascend-operator-competition
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
CS="${CS:-w3 w4 k_n16s64 k_n8s128 p6}"
REPS="${REPS:-2}"
KERREL=code/op_kernel/sparse_flash_attention.cpp
ns() { timeout "${T:-1800}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "########## sync $(date +%H:%M:%S) ##########"
( cd "$REPO" && timeout 300 bash "code 3/npu_debug/npu.sh" sync ) 2>&1 | grep -a 'op_kernel/sparse' | head -2
echo "########## build $(date +%H:%M:%S) ##########"
( cd "$REPO" && timeout 1800 bash "code 3/npu_debug/npu.sh" build ) 2>&1 | grep -aE '构建 OK|构建失败|error:' | head -5
echo "########## sweep $(date +%H:%M:%S)  CS=$CS ##########"
ns "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; \
    export ASCEND_CUSTOM_OPP_PATH=\$HOME/sfa_real/vendor/custom; \
    export LD_LIBRARY_PATH=\$HOME/sfa_real/vendor/custom/op_api/lib:\$LD_LIBRARY_PATH; \
    for r in \$(seq 1 $REPS); do for c in $CS; do \
      o=\$(timeout 300 ./test_sfa_dev cases/\$c.bin 1 diff 2>&1); \
      t=\$(printf '%s' \"\$o\" | grep -aoE '平均 [0-9.]+ ms' | head -1); \
      d=\$(printf '%s' \"\$o\" | grep -aoE '超差 [0-9]+/[0-9]+' | head -1); \
      [ -z \"\$t\" ] && t=\"NO-READING | \$(printf '%s' \"\$o\" | tail -1 | tr '\n\r' '  ')\"; \
      printf 'r%s %-10s %-14s %-16s\n' \$r \"\$c\" \"\$t\" \"\$d\"; \
    done; done"
ns "cd ~/sfa_real && grep -ac 'P96' $KERREL && sha256sum $KERREL | cut -c1-12"
