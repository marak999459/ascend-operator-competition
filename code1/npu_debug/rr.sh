#!/bin/bash
# 远端执行包装（非提交工装）：把超时放在**服务器侧**，本地 timeout 砍掉 ssh 不会砍掉远端进程。
# 为什么需要：本地 timeout 只杀 ssh 客户端 ⇒ 远端 test_npu/msprof 变孤儿继续占卡，
# 之后每条 msprof 都拿不到 step_trace（"Failed to connect database"/PROF_MISSING），
# 而且会伪装成"核函数挂死"（2026-09-21 02:54 那次就是这么把 R5 的读数看歪的）。
# 用法：bash npu_debug/rr.sh <超时秒> '<在 ~/ops_comp 下要跑的命令>' [tree]
#   tree 默认 code1（提交面构建树），传 optA/probe1 进隔离树。
set +u
SEC=${1:-300}; shift
CMD="$1"
TREE=${2:-code1}
CFG="$HOME/.atomgitdevenv/.ssh/config"
HOST=devenvc_02aeb.e653dc673c3d43f5ab8f0ebd29b1250d.atomgit.0
ssh -F "$CFG" -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
  "pkill -9 -u \$USER -f 'ops_comp/[a-zA-Z0-9_]*/npu_debug/[t]est_npu' 2>/dev/null; \
   pkill -9 -u \$USER -f '[m]sprof --task-time' 2>/dev/null; sleep 1; \
   cd ~/ops_comp/$TREE && timeout --signal=KILL $SEC bash -c '$CMD'" 2>&1 |
  grep -vE "post-quantum|WARNING: connection|vulnerable|may need to be upgraded|Permanently added"
