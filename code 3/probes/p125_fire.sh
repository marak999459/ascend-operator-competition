#!/bin/bash
# P125 发次（额度重置后一次跑完两臂）—— 为什么要这么发：探针臂**只能**从快照目录发
# （`~/sfa_real/code` 是提交源、必须保持 pristine），而 CLI 的 `submit` 把 429 的响应体吞了
# ⇒ 一律走 `p125_submit_diag.py`（打印 status/headers/BODY，成功后自己轮询到终态）。
# 链：本地 pristine 校验 → q 臂 → 读 → w 臂 → 读 → `p125_reissue.sh`（回发最好的字节）。
# ⚠️ 两臂都是"多加一遍活"的尺子 ⇒ 榜上挂的每一秒都比现挂慢；**跑完必须回发**，
#    且回发要核**外部真值**（§7 #11：看 `p98_rank.py` 的 `sub` 变没变），不许拿"后台任务完成"当发货成功。
# 用法：bash p125_fire.sh            （跳过 w 臂：SKIP_W=1；只发不发读：DRY=1）
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
K="$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp"
SHA='f815bf1eaba0f8bc'
nssh() { timeout "${TMO:-2400}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
DIAG="$REPO/code 3/probes/p125_submit_diag.py"
fire() {  # fire <dir> <tag>
  echo "########## 发 $2（$1）##########"
  timeout 2400 ssh -F "$S" -o ConnectTimeout=25 "$H" "python3 - $1 $2" < "$DIAG" 2>&1 \
    | grep -av Warning | tee "$REPO/code 3/probes/p125_fire_$2.txt"
}

[ "$(sha256sum "$K" | cut -c1-16)" = "$SHA" ] \
  || { echo ">>> 本地提交源不是 pristine（$SHA）⇒ 中止：探针字节只许活在快照目录里"; exit 1; }
echo "本地 pristine OK；快照 sha 当场核："
nssh 'cd ~/sfa_real && for d in p125q_code p125w_code; do printf "%-12s %s\n" $d $(sha256sum $d/op_kernel/sparse_flash_attention.cpp | cut -c1-16); done' \
  2>&1 | grep -av Warning
echo ">>> 期望 p125q_code=9f7c9c28… p125w_code=57880494…"
[ "${DRY:-}" = 1 ] && exit 0

fire ~/sfa_real/p125q_code q32
[ "${SKIP_W:-}" = 1 ] || fire ~/sfa_real/p125w_code w16

echo "########## 六点读数（从平台侧再读一遍，⛔ 不信上面那行的转录）##########"
nssh 'source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 ~/sfa_real/p98_rank.py' 2>&1 | grep -av Warning | tail -3

echo "########## 回发 pristine ##########"
bash "$REPO/code 3/probes/p125_reissue.sh"
echo "########## 回发后的外部真值 ##########"
nssh 'source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 ~/sfa_real/p98_rank.py' 2>&1 | grep -av Warning | tail -3
echo P125_FIRE_DONE
