#!/bin/bash
# P103b：cube 线在**平台形状**上的胜负 —— 同一份 A′ 字节，只动 host 的并行度门常量
#   80 % ↔ 5 %（5 % = 实际放行；形态门与 CubeBlock 仍然兜着），两臂各跑两遍。
#   ⓪ 每批先跑一发 p6 确认 cube 臂活着（§7 第 8 条），会挂的档只放最后。
#   ⚠️ 每次 sed 后必须 grep 命中数 = 1 才算改了（§7 第 2 条）。
# 用法： bash p103b_gate_ab.sh <A|B> [reps]     A = 门 80%（cube 关）  B = 门 5%（cube 开）
set -u
cd ~/sfa_real || exit 1
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
export ASCEND_CUSTOM_OPP_PATH=~/sfa_real/vendor/custom
export LD_LIBRARY_PATH=~/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
HOST=code/op_host/sparse_flash_attention.cpp
ARM="${1:-B}"; REPS="${2:-5}"
WANT=80; [ "$ARM" = "B" ] && WANT=5
CUR=$(grep -o 'CUBE_MIN_WAVES_PCT = [0-9]*' "$HOST" | grep -o '[0-9]*$')
echo "== 臂 $ARM：门 当前=$CUR 目标=$WANT"
if [ "$CUR" != "$WANT" ]; then
  sed -i "s/CUBE_MIN_WAVES_PCT = ${CUR}ULL/CUBE_MIN_WAVES_PCT = ${WANT}ULL/" "$HOST"
  NOW=$(grep -c "CUBE_MIN_WAVES_PCT = ${WANT}ULL" "$HOST")
  echo "   sed 后命中数=$NOW（必须 =1）"
  [ "$NOW" = "1" ] || { echo "   门没改成，中止"; exit 2; }
  bash build.sh > /tmp/p103b_build.log 2>&1 || { echo "   build 失败"; tail -20 /tmp/p103b_build.log; exit 3; }
  echo "   build ok"
fi
grep -o 'CUBE_MIN_WAVES_PCT = [0-9]*' "$HOST"

run() {
  for c in "$@"; do
    [ -f "cases/$c.bin" ] || { printf "%-14s (无用例)\n" "$c"; continue; }
    out=$(timeout 180 ./test_bench "cases/$c.bin" "$REPS" 2>&1); rc=$?
    printf "%-14s rc=%d  %s\n" "$c" "$rc" \
      "$(printf '%s' "$out" | tr '\n' ' ' | sed 's/.*SBS/SBS/;s/估算.*//;s/  */ /g')"
  done
}

echo "-- pass 1"
run p6 p4s128c16 p4s1c2048 p4s128c128 p4s1c16384 p4s128c512 p4n1s128c512 p16s128c128 p32s128c128
echo "-- pass 2"
run p6 p4s128c16 p4s1c2048 p4s128c128 p4s1c16384 p4s128c512 p4n1s128c512 p16s128c128 p32s128c128
