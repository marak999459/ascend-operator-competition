#!/bin/bash
# [题1/R21 p21b 离线+快跑工装，非提交面] 反向 host 路由的 **mode 地图**：
#   为什么值得单开一发 —— p21 里 c5(`bwd,fp16,32,384,1`) 两臂 ΔT 恰好 0.00，一开始像"行项在
#   小 S 上消失"，直接读 `[probe-bwd] … mode=` 才判明：S=32 走的是 `splitMode=2(ELEMENT_SPLIT)`，
#   `BwdCopyRows` 第一道闸就把 `splitMode != 0` 挡掉 ⇒ **那一发从未进新分支**，它是第三个对照组
#   而不是反例。这条 host 侧事实同时框住 H5a/H5c 的适用域（只有 mode=0 的档能被整块搬运吃到），
#   所以把它铺成一张表，而不是留在一次性读数里。
# 口径：MHC_NOWALL=1 走 host 墙钟快路（**不开 msprof**），一发自证只看 `[probe-*] … mode= tile= …
#   mismatch=` 那一行 ⇒ 每发秒级、不产生计时读数，**这张表里的 time 一律不作 µs 结论用**（§23.27-6
#   的相位/口径规矩只约束计时，这里根本没有计时）。
set +u
cd ~/ops_comp/probe2 || exit 1
strings npu_debug/pkg/custom/op_api/lib/libcust_opapi.so | grep -q MHC_BWD_COPY || { echo "ABORT: 没有 H5a 探针"; exit 8; }
strings npu_debug/pkg/custom/op_api/lib/libcust_opapi.so | grep -q MHC_BWD_GATHER && { echo "ABORT: H3 gather 探针还开着"; exit 9; }
echo "### RUNNER p21b_modemap host=$(hostname) start=$(date +%H:%M:%S) 只读路由，不作计时"

# $1=shape $2=blk $3=arm(on|off) $4=tag
g() {
  local line
  if [ "$3" = "on" ]; then export MHC_BWD_COPY=1; else unset MHC_BWD_COPY; fi
  line=$(env MHC_SHAPE="$1" MHC_PCASE=0 MHC_REPS=6 MHC_FORCE_BLK="$2" MHC_CORE_FLOOR=8 MHC_MERGE_FLOOR=8 \
        MHC_NOWALL=1 bash npu_debug/prof_matrix.sh "p21b_$4" 0 2>&1 |
        grep -E "\[probe-|mismatch=|ALL PASS|HAS FAIL" | tr '\n' '#' | sed 's/#$//')
  unset MHC_BWD_COPY
  echo "M $4 b$2 $3 shape=$1 :: ${line:-无读数}"
}

# ---- S 轴（D=384, m=1，H5a 主战场）：找 mode=0 的下边界 ----
for s in 1 8 16 24 32 40 48 56 64 96 128 256; do
  g "bwd,fp16,${s},384,1" 8 on "S${s}"
done
# ---- m 轴（判分档声明覆盖 m=2/4/8）：D=256 是题面小规模那一档 ----
for m in 1 2 4 8; do
  g "bwd,fp16,64,256,${m}" 16 on "m${m}"
  g "bwd,fp16,128,256,${m}" 16 on "M${m}"
done
# ---- D 轴两点：题面小规模 D=256 与中规模 D=4096（后者 tile 早已过 6144B 闸）----
g "bwd,fp16,1024,4096,4" 16 on "big4"
g "bwd,fp16,64,4096,1" 16 on "D4096"
echo "### p21b done $(date +%H:%M:%S)"
