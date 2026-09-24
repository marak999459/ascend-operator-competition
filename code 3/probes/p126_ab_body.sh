#!/bin/bash
# P126 两臂对表的**远端**主体（由 p126_ab.sh 推过去后调用；⛔ 只活在远端，不进提交源）。
# 每一臂读两件事：
#   · `none` 档 -> 批量口径时间（本地 Δ）
#   · `diff` 档 -> 对**耐久** golden 的"逐位一致 /3"计数 + 超差（归因：哪一例掉了逐位）
# 有 golden 的用例才跑 diff（没 golden 的用例 diff 会打空，读到 0/3 会被误读成"掉了逐位"）。
set -u
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
cd ~/sfa_real || exit 1
export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom
export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
REPS="${REPS:-5}"
CASES="${CASES:-p1 p2 p4 p6 big1 w3 r64c65}"
NBS="${NBS:-1}"
GOLDSET="${GOLDSET:-p1 p2 p4 p6 big1}"

arm() {
  local c="$1" fb="$2" out pk bm d bit dif
  if [ -z "$fb" ]; then
    out=$(timeout 900 ./test_sfa_dev "cases/$c.bin" "$REPS" none 2>&1)
  else
    out=$(SFA_FORCE_NB="$fb" timeout 900 ./test_sfa_dev "cases/$c.bin" "$REPS" none 2>&1)
  fi
  pk=$(printf '%s' "$out" | grep -a SFA_PICK | head -1 | sed 's/.*SFA_PICK //')
  bm=$(printf '%s' "$out" | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//')
  bit="-"; dif="-"
  case " $GOLDSET " in
    *" $c "*)
      if [ -z "$fb" ]; then d=$(timeout 900 ./test_sfa_dev "cases/$c.bin" 1 diff 2>&1)
      else d=$(SFA_FORCE_NB="$fb" timeout 900 ./test_sfa_dev "cases/$c.bin" 1 diff 2>&1); fi
      # ⚠️ 必须排掉"**不逐位一致**"—— 那行里也含"逐位一致"四个字，早先按行计数把
      #    p1/p2/p4 的"掉了逐位"读成了 3/3（本目录 p126_ab.txt 第一版就是这个口径）。
      bit=$(printf '%s' "$d" | grep -a 逐位一致 | grep -av 不逐位一致 | wc -l)
      dif=$(printf '%s' "$d" | grep -aoE '超差 [0-9]+/[0-9]+' | tr '\n' ' ')
      ;;
  esac
  printf '%-9s nb=%-5s 批量 %10s  逐位 %s/3  超差[%s] pick[%s]\n' "$c" "${fb:-auto}" "$bm" "$bit" "$dif" "$pk"
}

for c in $CASES; do
  arm "$c" ""
  for b in $(echo "$NBS" | tr ',' ' '); do arm "$c" "$b"; done
  arm "$c" ""          # 同场次尾基线（看这一族场次漂移）
done
