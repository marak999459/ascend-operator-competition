#!/bin/bash
# P109：**把选档标定搬到"平台真正所在的那一族"上**（rows >= 41）。
# 为什么现在做：P106 的门 + P107/P108 两发零响应 ⇒ 六点全在 rows>=41（本地靶族 p*/s* 全是
# rows<=32，从 9/23 起一直在扫错的那一族）。而 host 的 `nb` 贪心（"每个 nb 取最大可行 k"）
# 当年就是在 0.12~0.74 ms 的小形状上验的（`gen_bigshape.py` 自己的 docstring 就把这个问题
# 挂在 w 族上，但从没跑过）。这一发只回答一句：**rows 大了之后，模型选的 nb 还是不是最优？**
#
# 只夹 `SFA_FORCE_NB`，**不动 n_blk** ⇒ 每一档的 UB 需求都还是模型自己算出来的那个（夹小的
# 方向安全，夹大可能真越界）。SFA_PICK 打在选档之后 ⇒ 它就是生效凭据（与 p107 那条 n_blk
# 覆盖不同，那里必须另打 SFA_EFF）。
# ⚠️ 只跑在远端（由 p109_nbgrid.sh 推过来后调用）。
set -u
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
cd ~/sfa_real || exit 1
export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom
export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
REPS="${REPS:-5}"

# case -> "rows:qN:nb候选"
spec_of() {
  case "$1" in
    w2)             echo "32:4:1,2,4" ;;
    w3)             echo "128:4:1,2,4" ;;
    w4)             echo "128:8:1,2,4,8" ;;
    big1)           echo "128:8:1,2,4,8" ;;
    p32s1c16384)    echo "32:4:1,2,4" ;;      # rows=32 对照（P108 已证平台不在这一族）
    *)              echo "?:?" ;;
  esac
}

arm() {
  local c="$1" fb="$2" out pk bm
  if [ -z "$fb" ]; then
    out=$(timeout 900 ./test_sfa_dev "cases/$c.bin" "$REPS" none 2>&1)
  else
    out=$(SFA_FORCE_NB="$fb" timeout 900 ./test_sfa_dev "cases/$c.bin" "$REPS" none 2>&1)
  fi
  pk=$(printf '%s' "$out" | grep -a SFA_PICK | head -1 | sed 's/.*SFA_PICK //')
  bm=$(printf '%s' "$out" | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//')
  printf '%-14s force_nb=%-4s 批量 %10s  pick[%s]\n' "$c" "${fb:-auto}" "$bm" "$pk"
}

for c in ${ONLY:-w3 w4 big1 w2 p32s1c16384}; do
  echo "----- $c (rows:qN:nb = $(spec_of "$c")) -----"
  arm "$c" ""
  IFS=: read -r _ _ nbs < <(spec_of "$c")
  for b in $(echo "$nbs" | tr ',' ' '); do arm "$c" "$b"; done
  arm "$c" ""            # 同场次尾基线：看这一族的场次漂移有多大
done
