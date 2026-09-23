#!/bin/bash
# P93：#69 的可用部分 —— cube 臂自己的段级消融（full / cubekg / cubek1），同场次三臂。
# cubel0/cubefx/cubenull 三档在 P86 那轮已废（挂死/当场报错），不再重试。
# 判据只有时间：这些档输出必然是错的 ⇒ 只读"平均 X ms"，绝不看数值、绝不 write。
# 用途：钉住"AIC 侧到底还有多少余量"，从而裁 M2（PV 上 Cube 要再花 512~1088 MAC/ht）值不值。
set -u
REPO=/home/fszqsn/ops_comp/ascend-operator-competition
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
CS="${CS:-w3 w4 k_n16s64 p6}"
KERREL=code/op_kernel/sparse_flash_attention.cpp
ns() { timeout "${T:-1800}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

restore() {
  echo "########## 还原提交源构建 $(date +%H:%M:%S) ##########"
  ( cd "$REPO" && timeout 300 bash "code 3/npu_debug/npu.sh" sync ) 2>&1 | grep -aE 'op_kernel/sparse' | head -1
  ( cd "$REPO" && timeout 1800 bash "code 3/npu_debug/npu.sh" build ) 2>&1 | grep -aE '构建 OK|error:' | head -3
  ns "cd ~/sfa_real && grep -ac '\[ABL\]' $KERREL; sha256sum $KERREL | cut -c1-12"
}
trap restore EXIT

for m in "$@"; do
  echo "########## arm = $m  $(date +%H:%M:%S) ##########"
  ( cd "$REPO" && timeout 300 bash "code 3/npu_debug/npu.sh" sync ) 2>&1 | grep -a 'op_kernel/sparse' | head -1
  if [ "$m" != full ]; then
    B64=$(base64 -w0 "$REPO/code 3/probes/p86_cubeabl.py")
    ns "mkdir -p ~/sfa_real/probes && echo $B64 | base64 -d > ~/sfa_real/probes/p86_cubeabl.py && \
        cd ~/sfa_real && python3 probes/p86_cubeabl.py $m > /tmp/abl.cpp && \
        test -s /tmp/abl.cpp && mv /tmp/abl.cpp ~/sfa_real/$KERREL && \
        grep -ac '\[ABL\]' ~/sfa_real/$KERREL" || { echo "档 $m 打桩失败，作废"; continue; }
  fi
  ( cd "$REPO" && timeout 1800 bash "code 3/npu_debug/npu.sh" build ) 2>&1 | grep -aE '构建 OK|error:' | head -5
  ns "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; \
      export ASCEND_CUSTOM_OPP_PATH=\$HOME/sfa_real/vendor/custom; \
      export LD_LIBRARY_PATH=\$HOME/sfa_real/vendor/custom/op_api/lib:\$LD_LIBRARY_PATH; \
      for r in 1 2; do for c in $CS; do \
        o=\$(timeout 300 ./test_sfa_dev cases/\$c.bin 1 diff 2>&1); \
        t=\$(printf '%s' \"\$o\" | grep -aoE '平均 [0-9.]+ ms' | head -1); \
        [ -z \"\$t\" ] && t=\"NO-READING | \$(printf '%s' \"\$o\" | tail -1 | tr '\n\r' '  ')\"; \
        printf 'r%s %-9s %-10s %s\n' \$r $m \$c \"\$t\"; \
      done; done"
done
