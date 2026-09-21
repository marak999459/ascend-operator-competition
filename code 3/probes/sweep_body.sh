#!/bin/bash
# P17a 扫描主体：**只在远端跑**。先打印模型自选的 (nb,n_blk,ks)，再扫 (nb, ks) 网格。
# 由 run_nbsweep.sh 推到 ~/sfa_real/sweep_body.sh 后调用。
set -u
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
cd ~/sfa_real
export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom
export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
# AUTO_ONLY=1 时只打印模型自选档 + 计时（拿 pick 表用，不扫网格）。
for c in "$@"; do
  echo "########## $c ##########"
  ./test_sfa_dev "cases/$c.bin" 3 none 2>&1 | grep -aE 'SFA_PICK|批量口径|时间: 平均|超差' | tr -s ' ' | sed 's/^/  AUTO  /'
  [ "${AUTO_ONLY:-0}" = "1" ] && continue
  for nb in 1 2 4 8; do
    for ks in 1 2; do
      out=$(SFA_FORCE_NB=$nb SFA_FORCE_KS=$ks timeout 900 ./test_sfa_dev "cases/$c.bin" 3 none 2>&1)
      pk=$(echo "$out" | grep -a SFA_PICK | head -1 | tr -s ' ')
      bm=$(echo "$out" | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/  最小.*//')
      sp=$(echo "$out" | grep -a '时间: 平均' | head -1 | sed 's/.*平均 //; s/  最小.*//')
      printf '  nb=%-2s ks=%s  %-42s 批量 %8s  单发 %8s\n' "$nb" "$ks" "$pk" "$bm" "$sp"
    done
  done
done
