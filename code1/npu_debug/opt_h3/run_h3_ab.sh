#!/bin/bash
# [题1/R17 H3 工装，非提交面] 反向单次 gather 的同构建 A/B。
#
# 为什么这样切：核数轴已由 code1.md §23.25 封存（同 tpc 只差启动块数就给 15.5%/27.1% 锯齿），
# 所以两臂必须在**同一份机器码、同一个强制核数**上比，判据取"随 m 单调"而不是"某档快"：
#   m=2 每行只省 1 条发起（预期 ≈0，它同时是本轮的恒等控制），m=4 省 3 条，m=8 省 7 条。
#   若 on 臂的增益按 m=2 < m=4 < m=8 排开，那是发起数的签名；若三格乱跳，锯齿赢。
# 形状全部 S=64,D=1024 只切 m ⇒ io 随 m 变（320/640/1280KiB），所以读数一律归一到各自的
# off 臂，不比绝对值。
set +u
cd "$(dirname "$0")/../.." || exit 1
SO=npu_debug/pkg/custom/op_api/lib/libcust_opapi.so
strings "$SO" | grep -q MHC_BWD_GATHER || { echo "### ABORT: .so 里没有 MHC_BWD_GATHER —— H3 补丁没生效"; exit 8; }

BLK=${BLK:-40}
ROUNDS=${ROUNDS:-2}
REPS=${REPS:-41}
SHAPES=${SHAPES:-"bwd,fp16,64,1024,2 bwd,fp16,64,1024,4 bwd,fp16,64,1024,8"}
echo "### banner: run_h3_ab blk=$BLK reps=$REPS rounds=$ROUNDS shapes='$SHAPES'"
k=0
for shp in $SHAPES; do
  for r in $(seq 1 "$ROUNDS"); do
    for g in off on; do
      ev=""
      [ "$g" = "on" ] && ev=MHC_BWD_GATHER=1
      line=$(env MHC_SHAPE="$shp" MHC_PCASE=0 MHC_REPS=$REPS MHC_FORCE_BLK=$BLK \
                 MHC_CORE_FLOOR=8 MHC_MERGE_FLOOR=8 $ev \
                 bash npu_debug/prof_matrix.sh "h3_s${k}g${g}_r${r}" 0 2>&1 |
             grep -E "剔首 mean=|ALL PASS|FAIL=|mismatch=[1-9]" | head -2 | tr '\n' '|')
      mean=$(echo "$line" | sed -n 's/.*剔首 mean=\([0-9.]*\).*/\1/p')
      p50=$(echo "$line" | sed -n 's/.*p50=\([0-9.]*\).*/\1/p')
      got=$(echo "$line" | sed -n 's/.*blk=\([0-9]*\).*/\1/p')
      ver=$(echo "$line" | grep -oE "ALL PASS|FAIL=[0-9]+" | head -1)
      echo "H3 s$k g=$g blk=$BLK got=$got ${ver:-NOVERDICT} mean=$mean p50=$p50 round=$r shape=$shp"
    done
  done
  k=$((k + 1))
done
echo "### run_h3_ab done"
