# P125 submit（前置：p125_precheck.sh 已四枚吻合 + SOCEDEX=0 + MARK/REP/NBOUND 对 + FORBID=0）
# ⚠️ 429 退避实测 ≥12 min（§7 #11 那条的同一轮）⇒ 连发之前先看榜单现态，别硬撞。
# ⚠️ 输出**不做 cut**：P128 那轮 `cut -c1-90` 把 Submission ID 整个削掉了，只能回头查 ranking API
#    才拿到全串（`p128_fullsub.py`）。远端另存一份原始件到 ~/sfa_real/。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
PID=6a7c22d6a52e0f540a8a098d
CLI=/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit/cannjudge_cli.py
timeout "${TMO:-1500}" ssh -F "$S" -o ConnectTimeout=25 "$H" \
  "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --max-wait 1200 | tee ~/sfa_real/p125_submit_$1_$2.txt" 2>&1 \
  | grep -av Warning
