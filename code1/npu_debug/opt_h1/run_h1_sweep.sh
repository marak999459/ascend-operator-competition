#!/bin/bash
# [题1/R15 工装，非提交面] H1' 的二维阶梯：核数 blk × 合批开关，同一份机器码换环境变量。
#
# 为什么要它：R14 上线合批之后，§14/R13 那条"每核 >=8KB IO 才值得开这个核"的拐点在合批态
# 不再成立（code1.md §22.7）。拐点往哪移只能把 blk 当自变量重扫。R11/R13 那种"跨构建 A/B"
# 在这里不够用：本轮要同时切 blk 和合批，跨构建漂移（±0.2us/7%）会盖掉小位移。
#
# 用法（树根目录）：bash npu_debug/opt_h1/run_h1_sweep.sh <tag>
#   环境变量：CASES="0 8 9 12"  BLKS="4 6 8 12 16 24 40"  MERGES="on off"
#             ROWS="0"  AUTO=1  ROUNDS=2
#   ROWS=1 -> MHC_FORCE_ROW（H4 原型：S<num_aiv 的前向也走 ROW）
#   AUTO=1 -> 先跑一组"不加任何探针"的今日面，用来复现 R14 的读数当同构建锚点
#   COREFLOOR=<n> / MFLOOR=<n> -> 把 R13 那条无条件 blk 下界 / sqrt 合批律的初值从 8 换掉
#       （两者默认 8 = 与提交面恒等；H1″ 的"走原式的下界检验"臂用它，硬阶梯仍用 BLKS）
#   SHAPES="bwd,fp16,64,256,2 bwd,fp16,64,512,2 ..."  <- R17：按形状扫，不碰 pcs[] 表也不重编译
#       非空时**忽略 CASES**，序号打成 `s<k>`（聚合时和 `c<i>` 是两套标签，别混）；
#       REPS=<n>（默认 201）是这一模式的连发次数，因为形状没有表里的默认值可继承。
#       为什么要它：R17 之前加一档形状 = 改表 + 重建（6~8 分钟），而"反向 blk<8"这一整段
#       从来没进过扫描（p7/p8 只到 8..40），要扫 4~5 条形状 × 8 档核数，重编译比扫描还久。
# 每行 12 列定长，本地聚合取中位数。核数一律读 msprof 设备上报的 blk=（判决行的 blk= 是
# harness 预测值，§19.7.3）。
set +u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)
for f in "$HOME/Ascend/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/cann/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/ascend-toolkit/set_env.sh"; do
    [ -f "$f" ] && { source "$f"; break; }
done
export MHC_OPAPI_SO=$ROOT/npu_debug/pkg/custom/op_api/lib/libcust_opapi.so

if ! strings "$MHC_OPAPI_SO" | grep -q MHC_FORCE_BLK; then
    echo "### ABORT: 运行期 .so 里没有 MHC_FORCE_BLK —— 探针没生效，读数会全是默认核数"
    exit 8
fi
if ! strings "$MHC_OPAPI_SO" | grep -q MHC_CORE_FLOOR; then
    # R17：这道闸原来查的是 MHC_MIN_IO —— 那个旋钮随 R15 的 merge_cap 改写一起消失了（§23.14-8），
    # 继续查它等于每次必 exit 8。现在查 MHC_CORE_FLOOR（与 MHC_FORCE_BLK 同一次编辑注入）。
    echo "### ABORT: 没有 MHC_CORE_FLOOR —— host 补丁只吃进了部分编辑"
    exit 8
fi
TAG=${1:-untagged}
CASES=${CASES:-"0 8 9 12"}
BLKS=${BLKS:-"4 6 8 12 16 24 40"}
MERGES=${MERGES:-"on off"}
ROWS=${ROWS:-"0"}
ROUNDS=${ROUNDS:-2}
SHAPES=${SHAPES:-""}
REPS=${REPS:-201}
echo "### banner: run_h1_sweep tag=$TAG cases='$CASES' shapes='$SHAPES' blks='$BLKS' merges='$MERGES' rows='$ROWS' rounds=$ROUNDS reps=$REPS auto=${AUTO:-0}"

if [ -n "$SHAPES" ] && ! strings npu_debug/test_npu | grep -q MHC_SHAPE; then
    echo "### ABORT: harness 二进制里没有 MHC_SHAPE —— test_npu 是旧构建，形状会被静默忽略"
    exit 8
fi

run_one() {   # $1=case(或 SHAPES 序号) $2=blk(auto=NA) $3=merge $4=row $5=round $6=sub-tag [$7=shape]
    local envs=""
    [ "$2" != "NA" ] && envs="MHC_FORCE_BLK=$2"
    [ "$3" = "off" ] && envs="$envs MHC_NO_MERGE=1"
    [ "$4" = "1" ] && envs="$envs MHC_FORCE_ROW=1"
    local pc="$1" lbl="c$1"
    if [ -n "${7:-}" ]; then
        pc=0; lbl="s$1"
        envs="$envs MHC_SHAPE=$7 MHC_REPS=$REPS"
    fi
    line=$(env $envs MHC_CORE_FLOOR=${COREFLOOR:-8} MHC_MERGE_FLOOR=${MFLOOR:-8} bash npu_debug/prof_matrix.sh "r15_${6}" "$pc" 2>&1 |
           grep -E "剔首 mean=" | head -1)
    mean=$(echo "$line" | sed -n 's/.*剔首 mean=\([0-9.]*\).*/\1/p')
    p50=$(echo "$line" | sed -n 's/.*p50=\([0-9.]*\).*/\1/p')
    dblk=$(echo "$line" | sed -n 's/.*blk=\([0-9]*\).*/\1/p')
    dir=$(echo "$line" | sed -n 's/^ *c[0-9]* *\([a-z]*\):.*/\1/p')
    echo "R15 $lbl blk=$2 mg=$3 row=$4 got=$dblk dir=$dir mean=$mean p50=$p50 round=$5"
}

if [ -n "$SHAPES" ]; then
    declare -a SARR
    k=0
    for shp in $SHAPES; do SARR[$k]="$shp"; echo "### shape s$k = $shp"; k=$((k + 1)); done
    NSHP=$k
    if [ "${AUTO:-0}" = "1" ]; then
        for r in $(seq 1 "$ROUNDS"); do
            for ((q = 0; q < NSHP; q++)); do run_one "$q" NA on "$ROWS" "$r" "auto_r${r}_s${q}" "${SARR[$q]}"; done
        done
    fi
    for r in $(seq 1 "$ROUNDS"); do
        for b in $BLKS; do
            for g in $MERGES; do
                for ((q = 0; q < NSHP; q++)); do
                    run_one "$q" "$b" "$g" "$ROWS" "$r" "s${q}b${b}g${g}_r${r}" "${SARR[$q]}"
                done
            done
        done
    done
    echo "### run_h1_sweep done tag=$TAG (SHAPES mode, NSHP=$NSHP)"
    exit 0
fi

if [ "${AUTO:-0}" = "1" ]; then
    for r in $(seq 1 "$ROUNDS"); do
        for i in $CASES; do run_one "$i" NA on "$ROWS" "$r" "auto_r${r}_c${i}"; done
    done
fi
for r in $(seq 1 "$ROUNDS"); do
    for b in $BLKS; do
        for g in $MERGES; do
            for w in $ROWS; do
                for i in $CASES; do run_one "$i" "$b" "$g" "$w" "$r" "b${b}g${g}w${w}_r${r}_c${i}"; done
            done
        done
    done
done
echo "### run_h1_sweep done tag=$TAG"
