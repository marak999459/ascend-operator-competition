#!/bin/bash
# P104 计时：同一批 pm* 用例（只差索引表的连续率）在**当前构建**上跑两遍。
# 用法： LABEL=pre bash ~/p104_time.sh            （改前基线）
#       LABEL=post bash ~/p104_time.sh           （run-merge 落地后 = 上下界）
# ⓪ 先跑一发 p6 确认设备活着（§7 第 8 条）。REPS 可调，两臂必须同值。
set -u
cd ~/sfa_real || exit 1
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
export ASCEND_CUSTOM_OPP_PATH=~/sfa_real/vendor/custom
export LD_LIBRARY_PATH=~/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
LABEL="${LABEL:-run}"; REPS="${REPS:-10}"
CASES="${*:-pm1 pm2 pm4 pm8 pm64 pm256 pm1024 pmfull}"

[ -f ~/p104_run.py ] && python3 ~/p104_run.py

echo "== $LABEL 构建 = $(sha256sum code/op_kernel/sparse_flash_attention.cpp | cut -c1-12) kernel"
timeout 240 ./test_bench cases/p6.bin 3 2>&1 | tail -2 | sed 's/^/   探活 p6: /'

for pass in 1 2; do
  echo "===== $LABEL 第 $pass 遍 (reps=$REPS) ====="
  for c in $CASES; do
    [ -f "cases/$c.bin" ] || { printf "%-10s (无用例)\n" "$c"; continue; }
    out=$(timeout 240 ./test_bench "cases/$c.bin" "$REPS" 2>&1); rc=$?
    printf "%-10s rc=%d  %s\n" "$c" "$rc" \
      "$(printf '%s' "$out" | tr '\n' ' ' | sed 's/.*SBS/SBS/;s/估算.*//;s/  */ /g')"
  done
done
