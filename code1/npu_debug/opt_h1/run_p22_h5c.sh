#!/bin/bash
# [题1/R22 p22 临时探针，非提交面] H5c 的**同构建 A/B** 计时：
#   off 臂 = 原 BackwardOneBlock 逐行链（每核每行 m 条读 + m 道窄 VEC）；
#   on  臂 = 每批 L 行、每副本一条步进 gather + 一次宽 VEC(L·D) + 一次连续写。
#   两臂在同一份 .so 里由 tiling.probeBwdW 选路 ⇒ 无跨构建漂移；
#   §23.27 那条 ≈0.45µs 运行次序相位用 8-run 二级平衡块抵消（off,on,on,off | on,off,off,on）。
# 判据（§26.8-① + §27.9-1 的加严）：本地 ΔT >= 0.96µs，且**换算两端点都过 1.6 分**才算过线。
# 全部走 MHC_SHAPE ⇒ 本脚本自身 0 次构建。
set +u
cd ~/ops_comp/probe2 || exit 1
SO=npu_debug/pkg/custom/op_api/lib/libcust_opapi.so
strings "$SO" | grep -q MHC_BWD_WIDE || { echo "ABORT: .so 里没有 H5c 探针"; exit 8; }
strings "$SO" | grep -q MHC_FORCE_BLK || { echo "ABORT: 没有 MHC_FORCE_BLK 旋钮"; exit 8; }
strings "$SO" | grep -q MHC_BWD_GATHER && { echo "ABORT: H3 gather 探针还开着"; exit 9; }
echo "### RUNNER p22_h5c v1 host=$(hostname) start=$(date +%H:%M:%S) cells=9 (7x m>=2 + 2x 构造恒等对照)"

one() {  # $1=shape $2=blk $3=arm(off|on) $4=round $5=slot $6=tag
  local line mean got ver mm
  unset MHC_BWD_WIDE MHC_BWD_COPY
  if [ "$3" = "on" ]; then export MHC_BWD_WIDE=1; fi
  line=$(env MHC_SHAPE="$1" MHC_PCASE=0 MHC_REPS=41 MHC_FORCE_BLK="$2" MHC_CORE_FLOOR=8 MHC_MERGE_FLOOR=8 \
        bash npu_debug/prof_matrix.sh "p22_$6_b${2}_$3_r${4}s${5}" 0 2>&1 |
        grep -E "剔首 mean=|ALL PASS|FAIL=|mismatch=" | head -3 | tr '\n' '|')
  unset MHC_BWD_WIDE
  mean=$(echo "$line" | sed -n 's/.*剔首 mean=\([0-9.]*\).*/\1/p')
  got=$(echo "$line"  | sed -n 's/.*blk=\([0-9]*\).*/\1/p')
  ver=$(echo "$line"  | grep -oE "ALL PASS|FAIL=[0-9]+" | head -1)
  mm=$(echo "$line"   | sed -n 's/.*mismatch=\([0-9]*\)\/[0-9]*.*/mismatch=\1/p' | head -1)
  echo "S $6 b$2 r$4 slot$5 arm=$3 got=$got ${ver:-NOVERDICT} ${mm:-nommline} mean=$mean shape=$1"
}

# 一个格 = 8 发：块1 off,on,on,off | 块2 on,off,off,on（每臂各占 1,4 与 2,3 槽位一次）
pair() {  # $1=shape $2=blk $3=tag
  one "$1" "$2" off 1 1 "$3"; one "$1" "$2" on  1 2 "$3"
  one "$1" "$2" on  1 3 "$3"; one "$1" "$2" off 1 4 "$3"
  one "$1" "$2" on  2 1 "$3"; one "$1" "$2" off 2 2 "$3"
  one "$1" "$2" off 2 3 "$3"; one "$1" "$2" on  2 4 "$3"
}

# ---- 主格族（m>=2，反向整行；tpc 由 blk 定，L = min(tpc, 12288/tile)）----
pair "bwd,fp16,64,256,2"  16  w1    # tpc=4  L=4   §27.6 预测 ΔT 0.78（blk=16 那族本就不到线）
pair "bwd,fp16,64,256,2"  8   w2    # tpc=8  L=8   §27.6 预测 2.21 ← 主判格
pair "bwd,fp16,64,256,4"  8   w3    # tpc=8  L=8   §27.6 预测 2.63（副本轴：off 行项翻倍）
pair "bwd,fp16,64,256,8"  8   w4    # tpc=8  L=8   m=8 ⇒ off 行项 8*107+145 = 1.0µs/行
pair "bwd,fp16,128,256,2" 16  w5    # tpc=8  L=8   S 轴：行项基数翻倍
pair "bwd,fp16,64,384,2"  8   w6    # tile=768B ⇒ cap=16、L=8（字节轴：off 应≈ x1 那格 3.93）
pair "bwd,fp16,256,256,2" 8   w7    # tpc=32 L=24 ⇒ **每核 2 批**（判"每批 0.3µs"是否按批数累加）

# ---- 两条"由构造不可能走新分支"的对照臂（§23.27-② 的硬规矩）----
pair "bwd,fp16,64,256,1"  8   x1    # m=1 ⇒ BwdWideRows 里 m<2 return 0（且 H5a 位未开）⇒ 同一份机器码
pair "fwd,fp16,64,256,2"  16  x2    # 前向 ⇒ 探针位在 if constexpr(BACKWARD) 内，根本不参与
echo "### p22 done $(date +%H:%M:%S)"
