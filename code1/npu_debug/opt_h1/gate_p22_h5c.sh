#!/bin/bash
# [题1/R22 p22 临时探针，非提交面] H5c（反向 m>=2 步进 gather + 宽 L·D VEC）的**数值门禁**。
# 顺序照 §26.8-④：数值先于计时 —— mismatch!=0 时 ΔT 无意义（且那正是 §14.9-7 那条未测 DMA 形态）。
# 走 prof_matrix.sh 的 MHC_NOWALL=1 快路（秒级、不出 msprof 会话），只看三件事：
#   1) 该进的分支进了没：靠边界对偶 d(tile=6144B 过闸、cap=2 ⇒ 还要走 4 个批次) /
#      e(tile=6400B 不过闸)，两格只差 128B ⇒ 入口比较写反立刻现形；
#   2) 恒等：反向 m=2/4/8 的 on 臂必须逐 bit 等于 CPU 参考。本输入族 fill 全是 0.125 的整数倍、
#      TOL=1e-2 ⇒ mismatch=0 就是位级证据（§27.1 自记口径）；
#   3) 不该动的没动：f(m=1 ⇒ m<2 return 0)、g(前向)、h(mode=2 ⇒ return 0) 三格 on 臂同 off 臂。
# ⚠️ d 格同时是 UB 预算的最紧一格：8*L*tile = 8*2*6144 = 98304 = 本补丁的 cap 上界。
#    它若能 PASS 就说明"每批 8 份 tile 字节"这笔账没算错；算错的话这里是 InitBuffer 失败/trap。
set +u
cd ~/ops_comp/probe2 || exit 1
SO=npu_debug/pkg/custom/op_api/lib/libcust_opapi.so
strings "$SO" | grep -q MHC_BWD_WIDE || { echo "ABORT: .so 里没有 H5c 探针（先 apply_h5c.py on 再 build_npu.sh）"; exit 8; }
strings "$SO" | grep -q MHC_BWD_GATHER && { echo "ABORT: H3 gather 探针还开着，会污染反向路径"; exit 9; }

g() {   # $1=shape $2=blk $3=arm $4=tag $5=expect(进/不进分支)
  local out
  unset MHC_BWD_WIDE MHC_BWD_COPY
  if [ "$3" = "on" ]; then export MHC_BWD_WIDE=1; fi
  out=$(env MHC_SHAPE="$1" MHC_PCASE=0 MHC_REPS=6 MHC_FORCE_BLK="$2" MHC_CORE_FLOOR=8 MHC_MERGE_FLOOR=8 \
        MHC_NOWALL=1 bash npu_debug/prof_matrix.sh "g22_$4" 0 2>&1 |
        grep -E "^\[PROF|mismatch=|ALL PASS|HAS FAIL" | tr '\n' '#' | sed 's/#$//')
  unset MHC_BWD_WIDE
  echo "G $4 arm=$3 exp=$5 shape=$1 blk=$2 :: $out"
}

echo "### GATE p22_h5c host=$(hostname) $(date +%H:%M:%S)"
# ---- 主格：m=2/4/8 逐 bit 恒等（计时族同一形状同一 blk）----
g "bwd,fp16,64,256,2" 16 off a_m2_off  wide
g "bwd,fp16,64,256,2" 16 on  a_m2_on   wide
g "bwd,fp16,64,256,4" 8  on  b_m4_on   wide
g "bwd,fp16,64,256,8" 8  on  c_m8_on   wide
# ---- 边界对偶：6144B 过闸(cap=2 ⇒ 多批) / 6400B 不过闸；同时是 UB 最紧格 ----
g "bwd,fp16,64,3072,2" 8 on  d_edge_on  wide
g "bwd,fp16,64,3200,2" 8 on  e_over_on  nowide
# ---- 构造上不可能走新分支的三格 ----
g "bwd,fp16,64,256,1"  8  on  f_m1_on    nowide   # m<2 直接 return 0
g "fwd,fp16,64,256,2"  16 on  g_fwd_on   nowide   # 探针位在 if constexpr(BACKWARD) 内
g "bwd,fp16,32,256,2"  8  on  h_mode2_on nowide   # S=32 ⇒ splitMode=2 ⇒ return 0
# ---- 第二副本轴 + S 轴（tpc=8 ⇒ 单批 L=8，m=4 走 4 条 gather）----
g "bwd,fp16,128,256,4" 16 on  i_s128_on  wide
# ---- w7 那格的 mode= 前置判：p21b 的 mode 地图只扫到 S=128，S=256 未证 ⇒ 先证分支可达再计时 ----
g "bwd,fp16,256,256,2" 8  on  j_s256_on  wide
echo "### GATE done $(date +%H:%M:%S)"
