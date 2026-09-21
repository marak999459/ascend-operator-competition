#!/bin/bash
# Device-only kernel timing for mhc_sinkhorn (题2). 每个形状独立 msprof 输出目录
# （run.sh 自带的 PROF 分支把所有形状写进同一个 $T/prof，grep 会串台 —— 这里按形状分目录）。
# 读数口径沿用题1：op_summary 的 Task Duration(us)、剔除首个任务（冷启动）后取 mean，
# 核数取设备上报的 Block Num，判决行一律带 timed=1。
# 用法: prof_bench.sh <kernel文件> <label> [reps] [build:1|0] [host文件]
set -u
T=/home/developer/mhc_test
B=/home/developer/mhc_build
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
KERNEL="${1:?usage: prof_bench.sh <kernel> <label> [reps] [build] [host]}"
LABEL="${2:?missing label}"
REPS="${3:-20}"
BUILD="${4:-1}"
HOST="${5:-$T/host_cur.cpp}"
# 计时口径 = 比赛契约 dtype。§11.3~§11.12 的历史读数全是 fp16（prof_bench 旧版写死），
# 而题面 float32 进/出（official_problem_statement.md:190,198）⇒ 默认 fp32，判决行带 dtype= 以免混口径。
DTY="${DTY:-fp32}"
V=$B/myopp/vendors/custom
export ASCEND_CUSTOM_OPP_PATH=$V
export LD_LIBRARY_PATH=$V/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:$LD_LIBRARY_PATH

if [ "$BUILD" = "1" ]; then
  # 让 run.sh 的 6 配置正确性矩阵跑在计时口径同一个 dtype 上（以前恒为 fp16，
  # 与 prof 的 dtype 无关 ⇒ "门过了"和"计时的这条路对了"是两件不相干的事）
  DT=$DTY bash $T/run.sh "$KERNEL" "$HOST" $T/tiling_cur.h > $T/log/profbuild_$LABEL.log 2>&1
  rc=$?
  ok=$(grep -ac 'CASE=PASS' $T/log/profbuild_$LABEL.log)
  echo "@@@@ label=$LABEL dtype=$DTY build_rc=$rc cases_ok=$ok kernel_md5=$(md5sum "$KERNEL" | cut -c1-8) host_md5=$(md5sum "$HOST" | cut -c1-8)"
  [ "$ok" = "6" ] || { echo "### abort: correctness matrix not 6/6 (见 $T/log/profbuild_$LABEL.log)"; exit 1; }
  # 计时前必须过 fp32 契约闸门（常量 6 配置 + 随机 3 配置对拍），build=0 复用已编好的包
  bash $T/f32_gate.sh "$KERNEL" "$HOST" 0 > $T/log/f32prof_$LABEL.log 2>&1
  f32ok=$(grep -ac 'cases_ok=6/6' $T/log/f32prof_$LABEL.log)
  f32rand=$(grep -ac 'PASS: 全部矩阵与参考一致' $T/log/f32prof_$LABEL.log)
  echo "@@@@ label=$LABEL f32_cases=$f32ok f32_rand=$f32rand/3"
  { [ "$f32ok" = "1" ] && [ "$f32rand" = "3" ]; } || { echo "### abort: fp32 闸门未过 (见 $T/log/f32prof_$LABEL.log)"; exit 1; }
fi

cd $T/harness || exit 1
[ -f bench_sink ] || { echo "PROF FAIL=no bench_sink"; exit 1; }
OUT=$T/prof/$LABEL
rm -rf "$OUT"; mkdir -p "$OUT"
FAIL=0
# 默认网格 = 历次 A/B 的口径，别动它（历史读数要能对上）。
# 只想加形状时用 SHAPES 覆盖（'|' 分隔三元组），这样判决行仍带同一个 label/dtype。
SHAPES="${SHAPES:-1 4 20|1 8 20|20 6 20|64 8 20|100 6 20|1024 8 20}"
IFS='|' read -r -a SHAPE_ARR <<< "$SHAPES"
for cfg in "${SHAPE_ARR[@]}"; do
  set -- $cfg
  d="$OUT/b$1_n$2_i$3"; mkdir -p "$d"
  timeout 300 msprof --task-time=on --ai-core=on --output="$d" \
    --application="./bench_sink $1 $2 $3 $REPS $DTY" > "$d/msprof.stdout" 2>&1
  mrc=$?
  CSV=$(find "$d" -name "op_summary_*.csv" 2>/dev/null | head -1)
  if [ "$mrc" != "0" ] || [ -z "$CSV" ]; then
    echo "prof batch=$1 n=$2 iters=$3 | timed=0 dtype=$DTY MSProf_MISSING rc=$mrc"; FAIL=1; continue
  fi
  python3 - "$CSV" "$1" "$2" "$3" "$LABEL" "$DTY" <<'PY'
import csv, sys
path, b, n, it, lab, dty = sys.argv[1:7]
def num(r, key):
    try:
        return float(r[key])
    except Exception:
        return None
pairs = []
with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    for rec in csv.DictReader(fh):
        clean = {(k or '').strip().split('(')[0].strip(): (v or '').strip() for k, v in rec.items()}
        if 'MhcSinkhorn' not in clean.get('Op Name', ''):
            continue
        try:
            pairs.append((float(clean['Task Duration']), clean))
        except Exception:
            pass
if len(pairs) < 2:
    print(f"prof batch={b} n={n} iters={it} | timed=0 dtype={dty} tasks={len(pairs)} label={lab}")
    sys.exit(0)
body = pairs[1:]
durs = [d for d, _ in body]
r = body[0][1]
mean = sum(durs) / len(durs)
cols = {k: num(r, k) for k in ('aiv_time', 'aiv_vec_time', 'aiv_scalar_time', 'aiv_mte2_time', 'aiv_mte3_time')}
blk = r.get('Block Num', '?')
extra = ' '.join(f"{k}={v:.3f}" for k, v in cols.items() if v is not None)
print(f"prof batch={b} n={n} iters={it} | timed=1 dtype={dty} label={lab} tasks={len(body)} mean={mean:.3f}us min={min(durs):.3f} max={max(durs):.3f} blk={blk} {extra}")
PY
  [ $? -eq 0 ] || FAIL=1
done
echo "### prof_done label=$LABEL FAIL=$FAIL out=$OUT"
