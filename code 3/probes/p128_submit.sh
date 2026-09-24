#!/bin/bash
# P128 submit（前置：p128_precheck.sh 已四枚吻合 + SOCEDEX=0 + FORBID=0 + P128=2 BESTUNITS=3 OLDGATE=0 OLDTAB=1 KRES=0）
# ⛔ 429 窗口按 ≥8~10 min 用（P127 实测距上一发 12 min 仍撞）⇒ 距上一发不足 10 min 就别跑。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
PID=6a7c22d6a52e0f540a8a098d
CLI=/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit/cannjudge_cli.py
timeout "${TMO:-1500}" ssh -F "$S" -o ConnectTimeout=25 "$H" \
  "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --max-wait 1200" 2>&1 | grep -av Warning | cut -c1-120
