#!/bin/bash
# [题1/p20 第 1.5 阶临时探针，非提交面] 把 §25.3 拟合出的 k = 270ns/(核·行) 推到两条新轴上：
#   m 轴（每行几次副本 ⇒ k 该怎么长）与 S 轴（"每核行数"是不是真自变量）。
# 全部走 MHC_SHAPE 环境变量 ⇒ 0 次构建。块结构沿用 run_p20_ladder.sh 的 8-run 二级平衡块
#   （R,X,X,R | X,R,R,X），REF 钉在 stage-1 那一格 bwd,fp16,64,384,1 @ blk=16 ⇒ 跨轮可比。
set +u
cd ~/ops_comp/probe2 || exit 1
SO=npu_debug/pkg/custom/op_api/lib/libcust_opapi.so
strings "$SO" | grep -q MHC_FORCE_BLK || { echo "ABORT no knob"; exit 8; }
strings "$SO" | grep -q MHC_BWD_GATHER && { echo "ABORT H3 probe still in .so"; exit 9; }
echo "### RUNNER p20b_maxis v1 host=$(hostname) start=$(date +%H:%M:%S) cells=18 (IDENT+3x3 m轴+2x3 S轴+2 io分离)"
RSH=bwd,fp16,64,384,1
RBLK=16
one() {  # $1=shape $2=blk $3=tag $4=round $5=slot
  local line mean got ver
  line=$(env MHC_SHAPE="$1" MHC_PCASE=0 MHC_REPS=41 MHC_FORCE_BLK="$2" MHC_CORE_FLOOR=8 MHC_MERGE_FLOOR=8 \
        bash npu_debug/prof_matrix.sh "p20b_$3_b${2}_r${4}_$5" 0 2>&1 |
        grep -E "剔首 mean=|ALL PASS|FAIL=" | head -2 | tr '\n' '|')
  mean=$(echo "$line" | sed -n 's/.*剔首 mean=\([0-9.]*\).*/\1/p')
  got=$(echo "$line" | sed -n 's/.*blk=\([0-9]*\).*/\1/p')
  ver=$(echo "$line" | grep -oE "ALL PASS|FAIL=[0-9]+" | head -1)
  echo "S $3 b$2 r$4 slot$5 arm=$3 got=$got ${ver:-NOVERDICT} mean=$mean shape=$1"
}
pair() {  # $1=shape $2=blk $3=tag
  one "$RSH" "$RBLK" REF 1 1; one "$1" "$2" "$3" 1 2
  one "$1" "$2" "$3" 1 3; one "$RSH" "$RBLK" REF 1 4
  one "$1" "$2" "$3" 2 1; one "$RSH" "$RBLK" REF 2 2
  one "$RSH" "$RBLK" REF 2 3; one "$1" "$2" "$3" 2 4
}

# 恒等对照（必须 0）：与 REF 逐字节同构
pair "$RSH" 16 IDENT
# m 轴：S=64,D=384 ⇒ io = (m+1)·48 KiB；m=1 那一档 stage-1 已给（3.95/3.38/4.03）
for m in 2 4 8; do for blk in 8 16 32; do pair "bwd,fp16,64,384,${m}" "$blk" "m${m}"; done; done
# S 轴：m=1,D=384 ⇒ io = S·0.75 KiB；S=64 那一档同上
for S in 32 128; do for blk in 8 16 32; do pair "bwd,fp16,${S},384,1" "$blk" "S${S}"; done; done
# io/副本分离：m=8 但把 D 减半 ⇒ 同 m、同 rows、io 差 2 倍 ⇒ 看 k 是跟"每行几次副本"还是跟字节
for blk in 16 32; do pair "bwd,fp16,64,192,8" "$blk" "m8D192"; done
echo "### p20b done"
