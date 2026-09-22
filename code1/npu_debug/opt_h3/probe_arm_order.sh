#!/bin/bash
# [题1/R18 临时探针] 判两件事：① off/on 的差里有没有"臂位置"这一项（用 m=1 这条 gather 原理上不可达的形状）
# ② D=4096 上我的资格算式到底在 m 几开始失效（m=5 应合闸、m=6 应不合闸）
set +u
cd ~/ops_comp/probe2 || exit 1
SO=npu_debug/pkg/custom/op_api/lib/libcust_opapi.so
strings "$SO" | grep -q MHC_BWD_GATHER || { echo "ABORT no probe"; exit 8; }
one() {  # $1=shape $2=arm $3=seq $4=round
  local ev=""
  [ "$2" = "on" ] && ev=MHC_BWD_GATHER=1
  line=$(env MHC_SHAPE="$1" MHC_PCASE=0 MHC_REPS=41 MHC_FORCE_BLK=40 MHC_CORE_FLOOR=8 MHC_MERGE_FLOOR=8 $ev \
        bash npu_debug/prof_matrix.sh "h3p_s${5}_$2_r${4}" 0 2>&1 |
        grep -E "剔首 mean=|ALL PASS|FAIL=" | head -2 | tr '\n' '|')
  mean=$(echo "$line" | sed -n 's/.*剔首 mean=\([0-9.]*\).*/\1/p')
  ver=$(echo "$line" | grep -oE "ALL PASS|FAIL=[0-9]+" | head -1)
  echo "P $5 seq=$3 arm=$2 ${ver:-NOVERDICT} mean=$mean shape=$1"
}
for r in 1 2 3; do
  for pat in "off on" "on off"; do
    for a in $pat; do one "bwd,fp16,64,1024,1" "$a" "$(echo $pat | tr ' ' '')" "$r" m1; done
  done
done
echo "--- boundary D=4096 ---"
for r in 1 2; do
  for pat in "off on" "on off"; do
    for a in $pat; do one "bwd,fp16,64,4096,5" "$a" "$(echo $pat | tr ' ' '')" "$r" d5; done
    for a in $pat; do one "bwd,fp16,64,4096,6" "$a" "$(echo $pat | tr ' ' '')" "$r" d6; done
  done
done
echo "--- s5 rerun (D=4096,m=8) swapped order ---"
for r in 1 2; do
  one "bwd,fp16,64,4096,8" on  "onoff" "$r" d8
  one "bwd,fp16,64,4096,8" off "onoff" "$r" d8
done
echo "PROBE_RC=$?"
echo "### h3_probe done"
