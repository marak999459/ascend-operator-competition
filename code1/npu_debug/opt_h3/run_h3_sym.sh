#!/bin/bash
# [题1/R18 临时探针] 奇偶相位自平衡的 A/B：每个 round 跑 4 次 = off,on,on,off
# ⇒ 每个臂恰好占一个"慢位"和一个"快位"，块内均值对相位无偏。
set +u
cd ~/ops_comp/probe2 || exit 1
SO=npu_debug/pkg/custom/op_api/lib/libcust_opapi.so
strings "$SO" | grep -q MHC_BWD_GATHER || { echo "ABORT no probe"; exit 8; }
one() {  # $1=shape $2=arm $3=tag $4=round $5=slot
  local ev=""
  [ "$2" = "on" ] && ev=MHC_BWD_GATHER=1
  line=$(env MHC_SHAPE="$1" MHC_PCASE=0 MHC_REPS=41 MHC_FORCE_BLK=40 MHC_CORE_FLOOR=8 MHC_MERGE_FLOOR=8 $ev \
        bash npu_debug/prof_matrix.sh "h3s_$3_$2_r${4}_$5" 0 2>&1 |
        grep -E "剔首 mean=|ALL PASS|FAIL=" | head -2 | tr '\n' '|')
  mean=$(echo "$line" | sed -n 's/.*剔首 mean=\([0-9.]*\).*/\1/p')
  got=$(echo "$line" | sed -n 's/.*blk=\([0-9]*\).*/\1/p')
  ver=$(echo "$line" | grep -oE "ALL PASS|FAIL=[0-9]+" | head -1)
  echo "S $3 r$4 slot$5 arm=$2 got=$got ${ver:-NOVERDICT} mean=$mean shape=$1"
}
for r in 1 2 3; do
  for s in m1:64,1024,1 a2:64,1024,2 a4:64,1024,4 a8:64,1024,8 t1a8:40,1024,8; do
    tag=${s%%:*}; shp="bwd,fp16,${s#*:}"
    one "$shp" off "$tag" "$r" 1; one "$shp" on "$tag" "$r" 2
    one "$shp" on "$tag" "$r" 3; one "$shp" off "$tag" "$r" 4
  done
done
echo "### h3_sym done"
