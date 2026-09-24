#!/bin/bash
# P105 网格主体：**只在远端跑**（由 p105_grid.sh 推过去后调用）。
# 每发打印：模型/旋钮选出的档 (nb,n_blk,ks) + 批量口径时间 + out/LSE 超差数。
# ⚠️ "强制的 ks 是否真的生效"不靠相信旋钮，靠两处硬校验：
#    ① pick[] 里必须出现本发的 nb/ks（nb 超过头数时 host 走降级路径，pick 会露出 nb=1）；
#    ② ks>1 时结果不可能与参考【逐位】相同 ⇒ 这里用"超差 0 但 ks>1"的形态自证切分在跑。
#    外加：合法组合由 units0*ks<=40 预先过滤（kernel 的门 coreNum==ks*total0 不满足时
#    会静默退回 ks=1，那种臂量出来的就是别的臂，是 P55 记过的自欺形态）。
set -u
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
cd ~/sfa_real || exit 1
export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom
export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
REPS="${REPS:-5}"
CORES=40

# 用例 -> rows:qN（p1/p2/p4/p6 = 平台六点的本地 replica；pm* = rows 阶梯）
spec_of() {
  case "$1" in
    p1) echo 4:4 ;;   p2) echo 8:2 ;;   p4) echo 16:4 ;;  p6) echo 32:4 ;;
    pm41) echo 4:4 ;;      pm4full) echo 4:4 ;;   pm81) echo 8:4 ;;  pm161) echo 16:4 ;;
    *) echo "?:?" ;;
  esac
}

run_auto() {
  local c="$1" out
  out=$(timeout 300 ./test_sfa_dev "cases/$c.bin" "$REPS" none 2>&1)
  printf '%-9s %-11s %s\n' "$c" "AUTO" "$(printf '%s' "$out" | grep -a SFA_PICK | head -1 | sed 's/.*SFA_PICK //')"
  printf '          %-11s %s\n' "" "$(printf '%s' "$out" | grep -aE '批量口径|out : |LSE :' | tr '\n' '|' | sed 's/  */ /g')"
}

run_arm() {
  local c="$1" nb="$2" ks="$3" out rc
  out=$(SFA_FORCE_NB="$nb" SFA_FORCE_KS="$ks" timeout 300 ./test_sfa_dev "cases/$c.bin" "$REPS" none 2>&1)
  rc=$?
  local pk bm bd le nblk
  pk=$(printf '%s' "$out" | grep -a SFA_PICK | head -1 | sed 's/.*SFA_PICK //')
  nblk=$(printf '%s' "$pk" | sed 's/.*nblk=\([0-9]*\).*/\1/')
  bm=$(printf '%s' "$out" | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//')
  bd=$(printf '%s' "$out" | grep -a 'out : ' | head -1 | sed 's/.*超差 //; s/ .*//')
  le=$(printf '%s' "$out" | grep -a 'LSE :' | head -1 | sed 's/.*LSE : max //; s/sum 超差/sum=/')
  # 生效校验：pick 里的 nb/ks 必须是本发强制的值
  if ! printf '%s' "$pk" | grep -aq "nb=$nb .*ks=$ks "; then
    printf '%-9s nb=%-2s ks=%-3s ⚠️未生效 pick[%s] rc=%s\n' "$c" "$nb" "$ks" "$pk" "$rc"
    return
  fi
  printf '%-9s nb=%-2s ks=%-3s nblk=%-3s 批量 %9s  超差 out=%-6s | %s\n' \
         "$c" "$nb" "$ks" "$nblk" "$bm" "$bd" "$le"
}

# 一个用例的全网格：nb 不能超过头数；ks 要满足 units0*ks <= 核数
sweep_case() {
  local c="$1" rows qn nb units0 ks
  rows=$(spec_of "$c" | cut -d: -f1); qn=$(spec_of "$c" | cut -d: -f2)
  if [ "$rows" = "?" ]; then echo "无用例 $c"; return; fi
  run_auto "$c"
  for nb in 1 2 4 8; do
    [ "$nb" -le "$qn" ] || continue
    units0=$(( rows * ( (qn + nb - 1) / nb ) ))
    for ks in 1 2 4 5 8 10; do
      # units0*ks > 核数时 kernel 的门（coreNum == ks*total0）不满足 ⇒ 静默退回 ks=1，
      # 这种臂量的就是别的臂（P55 记过的自欺形态），直接不发。
      if [ "$ks" != 1 ] && [ $(( units0 * ks )) -gt $CORES ]; then continue; fi
      run_arm "$c" "$nb" "$ks"
    done
  done
}

if [ "${SMOKE:-0}" = "1" ]; then
  echo "===== P105 烟囱：ks=2 回归 + 深链会不会挂 ====="
  run_arm p1 1 2
  run_arm p1 4 4
  run_arm p1 4 10
  run_arm pm41 4 8
  exit 0
fi

for c in p1 p2 pm41 pm4full pm81 p4 p6 pm161; do
  echo "########## $c ($(spec_of $c | tr ':' '/')) ##########"
  sweep_case "$c"
done
