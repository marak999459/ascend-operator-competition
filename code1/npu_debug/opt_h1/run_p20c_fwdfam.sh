#!/bin/bash
# [题1/p20 第 1.5b 阶临时探针，非提交面] 同一份 .so 里把"合批开/关"当 A/B 臂，量**每核行数摊薄**
#   到底值多少钱 —— 这是 §25.7 候选 H5（反向行合批）目前唯一**不需要新构建**就能拿到的经验上界。
# 为什么用前向：MergeRows() 只在前向生效（op_host:125 merge_ok 含 !backward），
#   所以 MHC_NO_MERGE 对反向天然是 no-op ⇒ 顺手当**负对照**（Δ 必须为 0）。
# 预测（若前向也带 stage-1 那个 k=270ns/(核·行)，且合批把每核 tpc 行的开销摊薄成 tpc/L）：
#   S=64 ⇒ 每核行数 tpc=64/blk：blk=8→8 行、L=8 ⇒ ΔT≈+1.89µs；blk=16→4 行、L=4 ⇒ +0.81µs；blk=32→2 行、L=2 ⇒ +0.27µs
#   （cap = FWD_MERGE_BYTES/tile = 16384/768 = 21 ⇒ 这三档都由 tpc 封顶，L 正好等于 tpc）
# ⚠️ 解释力有限：NO_MERGE 臂仍走 FwdBatch() 的攒批（R11 那条 BS），所以量到的是"合批相对攒批再赚多少"，
#   不是"逐行相对合批赚多少"。读数一律按 §25.8 的方式只当**下界**用。
set +u
cd ~/ops_comp/probe2 || exit 1
SO=npu_debug/pkg/custom/op_api/lib/libcust_opapi.so
strings "$SO" | grep -q MHC_FORCE_BLK || { echo "ABORT no FORCE_BLK knob"; exit 8; }
strings "$SO" | grep -q MHC_NO_MERGE  || { echo "ABORT no NO_MERGE knob"; exit 9; }
strings "$SO" | grep -q MHC_BWD_GATHER && { echo "ABORT H3 probe still in .so"; exit 10; }
echo "### RUNNER p20c_fwdfam v1 host=$(hostname) start=$(date +%H:%M:%S) cells=12 (合批 on/off 同构建 A/B：1 IDENT + 3 m1 + 3 m4 + 2 S轴 + 1 反向负对照 + 2 m8)"

one() {  # $1=shape $2=blk $3=tag $4=额外env（'-' 表示无） $5=round $6=slot
  local line mean got ver xenv
  xenv="$4"; [ "$xenv" = "-" ] && xenv=""
  line=$(env MHC_SHAPE="$1" MHC_PCASE=0 MHC_REPS=41 MHC_FORCE_BLK="$2" MHC_CORE_FLOOR=8 MHC_MERGE_FLOOR=8 $xenv \
        bash npu_debug/prof_matrix.sh "p20c_$3_b${2}_r${4}_$5" 0 2>&1 |
        grep -E "剔首 mean=|ALL PASS|FAIL=" | head -2 | tr '\n' '|')
  mean=$(echo "$line" | sed -n 's/.*剔首 mean=\([0-9.]*\).*/\1/p')
  got=$(echo "$line" | sed -n 's/.*blk=\([0-9]*\).*/\1/p')
  ver=$(echo "$line" | grep -oE "ALL PASS|FAIL=[0-9]+" | head -1)
  echo "S $3 b$2 r$4 slot$5 arm=$3 got=$got ${ver:-NOVERDICT} mean=$mean shape=$1 xenv=${xenv:-none}"
}
# 8-run 二级平衡块：on 臂占 slot{1,4}/{2,3} 各一次，off 臂对称 ⇒ 两臂每轮各占奇偶槽
pair() {  # $1=shape $2=blk $3=tag $4=off 臂的额外 env
  one "$1" "$2" "${3}on" -    1 1;  one "$1" "$2" "${3}off" "$4" 1 2
  one "$1" "$2" "${3}off" "$4" 1 3; one "$1" "$2" "${3}on" -    1 4
  one "$1" "$2" "${3}off" "$4" 2 1; one "$1" "$2" "${3}on" -    2 2
  one "$1" "$2" "${3}on" -    2 3;  one "$1" "$2" "${3}off" "$4" 2 4
}
NM=MHC_NO_MERGE=1

pair "fwd,fp16,64,384,1"  16 IDENT '-'     # 恒等对照：两臂同构 ⇒ Δ 必须 0
for blk in 8 16 32; do pair "fwd,fp16,64,384,1" $blk m1 $NM; done
for blk in 8 16 32; do pair "fwd,fp16,64,384,4" $blk m4 $NM; done
pair "fwd,fp16,32,384,1"  16 S32  $NM      # 每核 2 行
pair "fwd,fp16,128,384,1" 16 S128 $NM      # 每核 8 行
pair "bwd,fp16,64,384,1"  16 bwd  $NM      # 负对照：反向不吃 NO_MERGE ⇒ Δ 必须 0
for blk in 8 16;   do pair "fwd,fp16,64,384,8" $blk m8 $NM; done
echo "### p20c done"
