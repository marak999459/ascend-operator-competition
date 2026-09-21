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
if ! strings "$MHC_OPAPI_SO" | grep -q MHC_MIN_IO; then
    echo "### ABORT: 没有 MHC_MIN_IO —— host 补丁只吃进了部分编辑"
    exit 8
fi
TAG=${1:-untagged}
CASES=${CASES:-"0 8 9 12"}
BLKS=${BLKS:-"4 6 8 12 16 24 40"}
MERGES=${MERGES:-"on off"}
ROWS=${ROWS:-"0"}
ROUNDS=${ROUNDS:-2}
echo "### banner: run_h1_sweep tag=$TAG cases='$CASES' blks='$BLKS' merges='$MERGES' rows='$ROWS' rounds=$ROUNDS auto=${AUTO:-0}"

run_one() {   # $1=case $2=blk(auto=NA) $3=merge $4=row $5=round $6=sub-tag
    local envs=""
    [ "$2" != "NA" ] && envs="MHC_FORCE_BLK=$2"
    [ "$3" = "off" ] && envs="$envs MHC_NO_MERGE=1"
    [ "$4" = "1" ] && envs="$envs MHC_FORCE_ROW=1"
    line=$(env $envs MHC_MIN_IO=${MINIO:-0} bash npu_debug/prof_matrix.sh "r15_${6}" "$1" 2>&1 |
           grep -E "剔首 mean=" | head -1)
    mean=$(echo "$line" | sed -n 's/.*剔首 mean=\([0-9.]*\).*/\1/p')
    p50=$(echo "$line" | sed -n 's/.*p50=\([0-9.]*\).*/\1/p')
    dblk=$(echo "$line" | sed -n 's/.*blk=\([0-9]*\).*/\1/p')
    dir=$(echo "$line" | sed -n 's/^ *c[0-9]* *\([a-z]*\):.*/\1/p')
    echo "R15 c$1 blk=$2 mg=$3 row=$4 got=$dblk dir=$dir mean=$mean p50=$p50 round=$5"
}

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
