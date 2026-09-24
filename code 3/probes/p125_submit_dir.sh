#!/bin/bash
# P125 发次（可从**任意目录**发）—— 存在的理由：平台是**每天 50 发**的硬额度（429 正文，§7 #13），
# 等重置的这段时间要在设备上铺下一臂，而 `~/sfa_real/code` 会被 sync 覆盖 ⇒ 发次必须从一个**当场核过 sha 的快照目录**里发。
# 用法：bash p125_submit_dir.sh <远端 project-dir（~/ 开头）> <日志名后缀>
# ⚠️ 输出不做 cut（P128 那轮 `cut -c1-90` 把 Submission ID 削掉了，见 §7 #11）。
set -u
DIR="${1:-}"; TAG="${2:-arm}"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
PID=6a7c22d6a52e0f540a8a098d
CLI=/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit/cannjudge_cli.py
case "$DIR" in '~'*) ;; *) echo ">>> 用法 p125_submit_dir.sh '~/sfa_real/xxx' TAG"; exit 1;; esac
timeout "${TMO:-1500}" ssh -F "$S" -o ConnectTimeout=25 "$H" \
  "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; \
   cd $DIR && sha256sum op_kernel/sparse_flash_attention.cpp | cut -c1-16; \
   python3 $CLI submit --problem-id $PID --project-dir $DIR --max-wait 1200 | tee ~/sfa_real/p125_submit_$TAG.txt" 2>&1 \
  | grep -av Warning
