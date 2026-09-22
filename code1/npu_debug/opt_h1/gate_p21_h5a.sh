#!/bin/bash
# [题1/R21 p21 临时探针，非提交面] H5a 的**数值门禁**（code1.md §26.8-④：数值先于计时）。
# 走 prof_matrix.sh 的 MHC_NOWALL=1 档（秒级，不出 msprof 会话），只看三件事：
#   1) 该走的分支真的走了吗 —— 靠 [PROF] 行的 mode=/tile= 与下面六格的**资格边界对偶**判：
#      g5(D=3072⇒tile 6144B，刚好过 BwdCopyRows 的 <=6144 闸) 与 g6(D=3200⇒6400B，过不了)
#      两格形状只差 128B，若分支入口写错（例如把边界比较反了），g6 会立刻现形。
#   2) 恒等：反向 m=1 的 on 臂输出必须逐 bit 等于 CPU 参考。这里 mismatch=0 就是位级证据，
#      不用另加 bitmis 字段 —— 默认 fill 的全部取值都是 0.125 的整数倍，而 TOL=1e-2，
#      所以"取错行/少写"的最小可能差 0.125 远大于容差；漏写区预置 0xffff=NaN 由 nan_cnt 单判。
#      两臂各 mismatch=0 ⇒ 两臂都等于同一参考 ⇒ 传递出 on≡off。
#   3) 不该动的没动：g3(m=2)、g4(前向)、g6(字节数超闸) 三格 on 臂必须与 off 臂同判据。
set +u
cd ~/ops_comp/probe2 || exit 1
strings npu_debug/pkg/custom/op_api/lib/libcust_opapi.so | grep -q MHC_BWD_COPY || { echo "ABORT: 没有 H5a 探针"; exit 8; }

g() {   # $1=shape $2=blk $3=arm $4=tag $5=本格应走哪条分支(expect)
  local out
  if [ "$3" = "on" ]; then export MHC_BWD_COPY=1; else unset MHC_BWD_COPY; fi
  out=$(env MHC_SHAPE="$1" MHC_PCASE=0 MHC_REPS=6 MHC_FORCE_BLK="$2" MHC_CORE_FLOOR=8 MHC_MERGE_FLOOR=8 \
        MHC_NOWALL=1 bash npu_debug/prof_matrix.sh "g_$4" 0 2>&1 |
        grep -E "^\[PROF|mismatch=|ALL PASS|HAS FAIL" | tr '\n' '#' | sed 's/#$//')
  unset MHC_BWD_COPY
  echo "G $4 arm=$3 exp=$5 shape=$1 blk=$2 :: $out"
}

echo "### GATE p21_h5a host=$(hostname) $(date +%H:%M:%S)"
# ---- 主格：与 p21 计时族同一形状同一 blk（off/on 必须都 PASS 且 maxdiff 相同）----
g "bwd,fp16,64,384,1" 16 off a_copy_off copy
g "bwd,fp16,64,384,1" 16 on  a_copy_on  copy
# ---- 资格边界对偶：6144B 过闸 / 6400B 不过闸（tpc=8 ⇒ 过闸那格还要走 4 个批次，正好多块路径）----
g "bwd,fp16,64,3072,1" 8 off b_edge_off nocopy
g "bwd,fp16,64,3072,1" 8 on  b_edge_on  copy
g "bwd,fp16,64,3200,1" 8 on  c_over_on  nocopy
# ---- 构造上不可能走新分支的三格 ----
g "bwd,fp16,64,384,2" 16 on  d_m2_on    nocopy   # m!=1 直接 return 0
g "fwd,fp16,64,384,1" 16 on  e_fwd_on   nocopy   # 探针位在 if constexpr(BACKWARD) 内
g "bwd,fp16,128,384,1" 16 on  f_s128_on copy     # S 翻倍 ⇒ tpc=8 ⇒ 单批 L=8
echo "### GATE done $(date +%H:%M:%S)"
