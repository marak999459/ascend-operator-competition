#!/bin/bash
# P96 分诊跑法：每个 (臂, 用例) 一条独立 ssh —— aicore 异常会把设备弄脏一小会儿，
# 同一会话里后续的 ./test_sfa_dev 全部跟着报 [FAIL]（本轮实测就是这个形状），
# 所以"谁真的挂"必须一个用例一次进程来读。
set -u
REPO=/home/fszqsn/ops_comp/ascend-operator-competition
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
KERREL=code/op_kernel/sparse_flash_attention.cpp
ns() { timeout "${T:-300}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

B64=$(base64 -w0 "$REPO/code 3/probes/p96_abl.py")
for arm in "$@"; do
  echo "########## arm=$arm  $(date +%H:%M:%S) ##########"
  ( cd "$REPO" && timeout 300 bash "code 3/npu_debug/npu.sh" sync ) 2>&1 | grep -ac 'op_kernel/sparse'
  ns "mkdir -p ~/sfa_real/probes && echo $B64 | base64 -d > ~/sfa_real/probes/p96_abl.py && \
      cd ~/sfa_real && python3 probes/p96_abl.py $arm ~/sfa_real/$KERREL /tmp/abl96.cpp && \
      test -s /tmp/abl96.cpp && mv /tmp/abl96.cpp ~/sfa_real/$KERREL && \
      sha256sum ~/sfa_real/$KERREL | cut -c1-12"
  ( cd "$REPO" && timeout 1800 bash "code 3/npu_debug/npu.sh" build ) 2>&1 | grep -aE '构建 OK|构建失败|error:' | head -3
  for c in p6 k_n16s64 w3; do
    r=$(ns "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; \
        export ASCEND_CUSTOM_OPP_PATH=\$HOME/sfa_real/vendor/custom; \
        export LD_LIBRARY_PATH=\$HOME/sfa_real/vendor/custom/op_api/lib:\$LD_LIBRARY_PATH; \
        timeout 90 ./test_sfa_dev cases/$c.bin 1 diff 2>&1 | \
        grep -aoE '平均 [0-9.]+ ms|超差 [0-9]+/[0-9]+|\[FAIL\] sync.*' | head -3 | tr '\n' '|'")
    printf '%-9s %-10s %s\n' "$arm" "$c" "$r"
    ns "echo ping > /tmp/ping96.txt; cat /tmp/ping96.txt" >/dev/null
    sleep 12
  done
done
