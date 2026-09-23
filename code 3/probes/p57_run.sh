#!/bin/bash
# P57 的远端执行体：跑一个用例，回 "rc + 耗时 + 判据行"。
# rc=124 ⇒ timeout 判死（§15.16 的卡死形态）；rc=0 且判据行有货 ⇒ 跑完了。
# 单独成文是因为这些命令要经两层 ssh 引号，内联会被撕碎。
set -u
cd ~/sfa_real || exit 3
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom
export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
TO=${TO:-90}
# 用例从 CASES 环境变量取（走 ssh 时 "$@" 会被引号撕碎，见文件头注释）
for cs in ${CASES:-r1_min r2_chunk}; do
  t0=$(date +%s)
  log=$(timeout $TO ./test_sfa_dev cases/$cs.bin 1 diff 2>&1)
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  verdict=$(printf '%s' "$log" | grep -aoE '超差 [0-9]+/[0-9]+|逐位一致 [^ ]+|\*\*逐位一致\*\*|PASS|FAIL|ERROR[^ ]*' | tr '\n' ' ')
  printf '%-10s rc=%-4s %4ss  %s\n' "$cs" "$rc" "$dt" "$verdict"
done
