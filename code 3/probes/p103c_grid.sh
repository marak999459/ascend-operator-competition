#!/bin/bash
# P103c：cube ↔ AIV 的**形状网格** —— 同一份 A′ 字节，只动 host 的并行度门常量：
#   GATE=9999 ⇒ 门永远关 = 纯 AIV 基线；GATE=5 ⇒ 门永远开 = 纯 cube（形态门/CubeBlock 仍兜着）。
#   两臂各跑两遍、同一场次（§7 第 1 条）。⚠️ 每次 sed 后 grep 命中数必须 =1（§7 第 2 条）。
#   ⓪ 每批先跑一发 p6（cube 臂小用例）确认没被上一发挂死污染（§7 第 8 条）。
# 用法： GATE=9999 bash ~/p103c_grid.sh [reps] [用例...]
set -u
cd ~/sfa_real || exit 1
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
export ASCEND_CUSTOM_OPP_PATH=~/sfa_real/vendor/custom
export LD_LIBRARY_PATH=~/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
HOST=code/op_host/sparse_flash_attention.cpp
WANT="${GATE:-9999}"; REPS="${1:-5}"; shift || true
CASES="${*:-p6 p4s128c16 p4s128c128 p4s128c512 p4s1c2048 p4s1c16384 p16s1c4096 p16s1c16384 p32s1c16384 p16s128c128 p32s128c128 p32s128c512 p4n1s128c512}"
CUR=$(grep -o 'CUBE_MIN_WAVES_PCT = [0-9]*' "$HOST" | grep -o '[0-9]*$')
echo "== GATE 当前=$CUR 目标=$WANT"
if [ "$CUR" != "$WANT" ]; then
  sed -i "s/CUBE_MIN_WAVES_PCT = ${CUR}ULL/CUBE_MIN_WAVES_PCT = ${WANT}ULL/" "$HOST"
  NOW=$(grep -c "CUBE_MIN_WAVES_PCT = ${WANT}ULL" "$HOST")
  echo "   sed 后命中数=$NOW（必须 =1）"
  [ "$NOW" = "1" ] || { echo "   门没改成，中止"; exit 2; }
  bash build.sh > /tmp/p103c_build.log 2>&1 || { echo "   build 失败"; tail -25 /tmp/p103c_build.log; exit 3; }
  echo "   build ok"
fi
grep -o 'CUBE_MIN_WAVES_PCT = [0-9]*' "$HOST"

for pass in 1 2; do
  echo "===== GATE=$WANT 第 $pass 遍 ====="
  for c in $CASES; do
    [ -f "cases/$c.bin" ] || { printf "%-14s (无用例)\n" "$c"; continue; }
    out=$(timeout 240 ./test_bench "cases/$c.bin" "$REPS" 2>&1); rc=$?
    printf "%-14s rc=%d  %s\n" "$c" "$rc" \
      "$(printf '%s' "$out" | tr '\n' ' ' | sed 's/.*SBS/SBS/;s/估算.*//;s/  */ /g')"
  done
done
