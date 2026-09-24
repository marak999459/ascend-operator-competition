#!/bin/bash
# P97：可信基线 —— 同一台机器、同一协议下对照 head(P91 已验证态) 与 p96(当前工作树)。
#
# 为什么要这一发：P96 的读数自相矛盾（w3 挂死 4 min / k_n16s64 3.4 s 秒挂 / p6 一会儿
# "超差 0"一会儿跑不完），而我上一轮 p96_abl.py 的 orig 对照臂是**错臂**（它把 AIV 改成
# "宽 nBlk_ + 行距 nBlk_"，那一档的 AIC 行距却是 nTile ⇒ 读到环外甚至张量外），
# 所以那批 "[FAIL] sync" 既不能给 P96 定罪也不能开脱。
#
# 三条从上一轮的翻车里学来的协议（每条都对应一次作废的读数）：
#   · 内核字节走 **stdin** 推送，不再 base64 进 argv —— 单参数上限 MAX_ARG_STRLEN=128 KB，
#     1667 行的内核 base64 后 137 KB ⇒ `Argument list too long` ⇒ 那一轮"head 臂"其实
#     跑的还是 p96 的二进制。现在每臂都先验远端 sha256 再构建，对不上就整臂作废。
#   · slog 只开 **ERROR** 级（ASCEND_GLOBAL_LOG_LEVEL=4）：开 INFO 会刷屏，判据行被截掉。
#   · 一个用例一个进程 + 用例间 sleep：一次 aicore 异常会毒化同 session 的后续 launch。
#   · 每例跑 REPS 遍：p6 的"过/不过"本来就是这一轮唯一的不确定观测，必须重复才谈得上读数。
set -u
REPO=/home/fszqsn/ops_comp/ascend-operator-competition
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
KERREL=code/op_kernel/sparse_flash_attention.cpp
CS="${CS:-p6 k_n16s64}"
REPS="${REPS:-2}"
GAP="${GAP:-20}"
TT="${TT:-200}"          # 单用例墙钟上限（秒）

ns()  { timeout "${T:-300}" ssh -F $S -o ConnectTimeout=25 $H "$@" 2>&1 | grep -av Warning; }
nsin(){ timeout "${T:-300}" ssh -F $S -o ConnectTimeout=25 $H "$@" < "$SRC"; }

( cd "$REPO" && git show "HEAD:code 3/code/op_kernel/sparse_flash_attention.cpp" ) > /tmp/p97_head.cpp \
  || { echo "取 HEAD 内核失败"; exit 1; }
cp "$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp" /tmp/p97_p96.cpp
declare -A LSHA
for m in "$@"; do
  case $m in head|p96) ;; *) continue;; esac
  [ "$m" = head ] && SRC=/tmp/p97_head.cpp || SRC=/tmp/p97_p96.cpp
  LSHA[$m]=$(sha256sum < "$SRC" | cut -c1-12)
  printf 'arm=%-5s local %s lines %s\n' "$m" "${LSHA[$m]}" "$(wc -l < "$SRC")"
done
ns "mkdir -p ~/p97 && echo READY" | tail -1

for m in "$@"; do
  case $m in head|p96) ;; *) continue;; esac
  [ "$m" = head ] && SRC=/tmp/p97_head.cpp || SRC=/tmp/p97_p96.cpp
  echo "########## arm=$m (${LSHA[$m]}) $(date +%H:%M:%S) ##########"
  ( cd "$REPO" && timeout 300 bash "code 3/npu_debug/npu.sh" sync ) >/dev/null 2>&1
  nsin "cat > ~/p97/$m.cpp"
  RS=$(ns "cd ~/sfa_real && cp ~/p97/$m.cpp $KERREL && sha256sum $KERREL | cut -c1-12")
  echo "remote sha=$RS"
  if [ "$RS" != "${LSHA[$m]}" ]; then echo "!! 远端字节不等于本臂 ⇒ 整臂作废"; continue; fi
  ( cd "$REPO" && timeout 1800 bash "code 3/npu_debug/npu.sh" build ) 2>&1 | grep -aE '构建 OK|build ok|error:' | head -3
  for c in $CS; do
    for r in $(seq 1 "$REPS"); do
      out=$(ns "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; \
          export ASCEND_CUSTOM_OPP_PATH=\$HOME/sfa_real/vendor/custom; \
          export LD_LIBRARY_PATH=\$HOME/sfa_real/vendor/custom/op_api/lib:\$LD_LIBRARY_PATH; \
          export ASCEND_SLOG_PRINT_TO_STDOUT=1; export ASCEND_GLOBAL_LOG_LEVEL=4; \
          o=\$(timeout $TT ./test_sfa_dev cases/$c.bin 1 diff 2>&1); rc=\$?; \
          printf '%s\n' \"\$o\" | grep -aiE '逐位|超差|平均|==\\> (PASS|FAIL)|\\[FAIL\\]|ERROR\\(' | tail -8; \
          echo \"--- rc=\$rc 行数=\$(printf '%s\n' \"\$o\" | wc -l)\"")
      printf '%-5s %-10s r%s %s\n' "$m" "$c" "$r" "$(printf '%s' "$out" | tr '\n' '~')"
      sleep "$GAP"
    done
    health=$(ns "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; \
        npu-smi info 2>/dev/null | sed -n '7p;8p' | tr -s ' ' | tr '\n' '|'")
    printf '%-5s %-10s HEALTH %s\n' "" "" "$health"
  done
done

echo "########## 还原提交源构建 $(date +%H:%M:%S) ##########"
( cd "$REPO" && timeout 300 bash "code 3/npu_debug/npu.sh" sync ) >/dev/null 2>&1
( cd "$REPO" && timeout 1800 bash "code 3/npu_debug/npu.sh" build ) 2>&1 | grep -aE '构建 OK|error:' | head -2
