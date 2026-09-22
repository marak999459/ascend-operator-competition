#!/bin/bash
# [题1/R21 p21 临时探针，非提交面] H5a（反向 m==1 整块搬运）的**同构建 A/B**：
#   off 臂 = 原 BackwardOneBlock 逐行链；on 臂 = 每核连续 L 行一次连续读 + 一次连续写。
#   两臂在同一份 .so 里由 tiling.probeBwdC 选路 ⇒ 没有跨构建漂移（§23.27 那条 0.45µs 相位
#   用 8-run 二级平衡块抵消：off,on,on,off | on,off,off,on，每臂各占两慢两快槽）。
# 判据（code1.md §26.8）：本地 ΔT ≥ 0.96µs（= 行项残余 f ≤ 20%）才写提交树；0.6~0.96 不提交、
#   回来改成 L 封顶 + 与 blk 联合重定；≈0 则 H5 判负。
# 全部走 MHC_SHAPE ⇒ 本脚本自身 0 次构建（构建在 apply_h5_copy.py on 之后做一次）。
set +u
cd ~/ops_comp/probe2 || exit 1
SO=npu_debug/pkg/custom/op_api/lib/libcust_opapi.so
strings "$SO" | grep -q MHC_BWD_COPY || { echo "ABORT: .so 里没有 H5a 探针（先 apply_h5_copy.py on 再 build_npu.sh）"; exit 8; }
strings "$SO" | grep -q MHC_FORCE_BLK || { echo "ABORT: 没有 MHC_FORCE_BLK 旋钮"; exit 8; }
strings "$SO" | grep -q MHC_BWD_GATHER && { echo "ABORT: H3 gather 探针还开着，会污染反向路径"; exit 9; }
echo "### RUNNER p21_h5a v1 host=$(hostname) start=$(date +%H:%M:%S) cells=8 (6x m=1 + 2x 构造恒等对照)"

one() {  # $1=shape $2=blk $3=arm(off|on) $4=round $5=slot $6=tag
  local line mean got ver mm shp
  shp="$1"
  if [ "$3" = "on" ]; then export MHC_BWD_COPY=1; else unset MHC_BWD_COPY; fi
  line=$(env MHC_SHAPE="$shp" MHC_PCASE=0 MHC_REPS=41 MHC_FORCE_BLK="$2" MHC_CORE_FLOOR=8 MHC_MERGE_FLOOR=8 \
        bash npu_debug/prof_matrix.sh "p21_$6_b${2}_$3_r${4}s${5}" 0 2>&1 |
        grep -E "剔首 mean=|ALL PASS|FAIL=|mismatch=" | head -3 | tr '\n' '|')
  unset MHC_BWD_COPY
  mean=$(echo "$line" | sed -n 's/.*剔首 mean=\([0-9.]*\).*/\1/p')
  got=$(echo "$line"  | sed -n 's/.*blk=\([0-9]*\).*/\1/p')
  ver=$(echo "$line"  | grep -oE "ALL PASS|FAIL=[0-9]+" | head -1)
  mm=$(echo "$line"   | sed -n 's/.*mismatch=\([0-9]*\)\/[0-9]*.*/mismatch=\1/p' | head -1)
  echo "S $6 b$2 r$4 slot$5 arm=$3 got=$got ${ver:-NOVERDICT} ${mm:-nommline} mean=$mean shape=$shp"
}

# 一个格 = 8 发：块1 off,on,on,off | 块2 on,off,off,on（每臂各占 1,4 与 2,3 槽位一次）
pair() {  # $1=shape $2=blk $3=tag
  one "$1" "$2" off 1 1 "$3"; one "$1" "$2" on  1 2 "$3"
  one "$1" "$2" on  1 3 "$3"; one "$1" "$2" off 1 4 "$3"
  one "$1" "$2" on  2 1 "$3"; one "$1" "$2" off 2 2 "$3"
  one "$1" "$2" off 2 3 "$3"; one "$1" "$2" on  2 4 "$3"
}

# ---- 主格族（m=1，反向整行；tpc 与 L 由 blk 决定，cap=FWD_MERGE_BYTES/tile=21）----
pair "bwd,fp16,64,384,1" 16 c1     # tpc=4  L=4   ← 与 p20/p20b 的 REF 同格，off 臂应读 3.37~3.44
pair "bwd,fp16,64,384,1" 8  c2     # tpc=8  L=8
pair "bwd,fp16,64,384,1" 32 c3     # tpc=2  L=2
pair "bwd,fp16,128,384,1" 16 c4    # tpc=8  L=8（S 轴：行项基数翻倍，on 臂省得更多才对）
pair "bwd,fp16,32,384,1"  8  c5    # tpc=4  L=4（小 S：判分档 S 未知，这一档看小 tpc 还剩多少）
pair "bwd,fp16,64,192,1"  16 c6    # D=192 ⇒ tile=384B、cap=42、tpc=4 ⇒ L=4（io 减半，字节项应仍不出现）

# ---- 两条"由构造不可能走新分支"的对照臂（§23.27-② 的硬规矩）----
pair "bwd,fp16,64,384,2" 16 x1     # m=2 ⇒ BwdCopyRows 里 m!=1 直接 return 0 ⇒ 两臂同一条机器码
pair "fwd,fp16,64,384,1" 16 x2     # 前向 ⇒ 探针位在 if constexpr(BACKWARD) 里，根本不参与
echo "### p21 done"
