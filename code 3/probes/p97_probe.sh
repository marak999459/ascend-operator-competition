#!/bin/bash
# 远端探针：把"跑 N 遍某个用例、只回一行读数"收到一个文件里，避免 ssh 引号/printf 二次解析
# （上一版就是被本地 printf 把远端的 %s 一起吃掉，四遍读数全空）。
# 用法（远端）：bash ~/p97/probe.sh <case> <遍数> <单遍墙钟上限秒>
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
cd ~/sfa_real || exit 9
export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom
export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
CASE=${1:-p6}; N=${2:-1}; TT=${3:-60}
for i in $(seq 1 "$N"); do
  timeout "$TT" ./test_sfa_dev cases/$CASE.bin 1 diff > /tmp/pr_$CASE.log 2>&1; rc=$?
  echo "$CASE#$i rc=$rc |$(grep -aoE '平均 [0-9.]+ ms|==> (PASS|FAIL)|不逐位一致|\[FAIL\] [a-z]+' /tmp/pr_$CASE.log | tr '\n' ' ')|"
done
