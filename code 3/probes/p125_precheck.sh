#!/bin/bash
# P125 发次链 §7.5 的"发前三查"：数值门已跑完（p125_probe.sh 第 4 步）⇒ **最后一次 sync（此后不再 build、不再 sed）**
# → 当场核四枚 sha256（本地 vs 远端）+ SoC sed 残留 0 + 禁用词 0 + 探针命中数 + host 三枚没被动 → dry-run。
# 用法：bash p125_precheck.sh q|w N
# ⛔ 这一发只做到 dry-run；submit 由 p125_submit.sh 单独跑（读数要留在另一段里）。
set -u
ARM="${1:-}"; N="${2:-}"
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
PID=6a7c22d6a52e0f540a8a098d
CLI=/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit/cannjudge_cli.py
FILES="op_kernel/sparse_flash_attention.cpp op_kernel/sparse_flash_attention_tiling.h op_kernel/tiling_key_sparse_flash_attention.h op_host/sparse_flash_attention.cpp"
case "$ARM" in q) WANT=1;; w) WANT=2;; *) echo ">>> 用法 p125_precheck.sh q|w N"; exit 1;; esac
case "$N" in (''|*[!0-9]*) echo ">>> N 必须是整数"; exit 1;; esac

echo "########## ① 最后一次 sync（此后不再 build） ##########"
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -3

echo "########## ② 本地四枚当场 sha256 ##########"
( cd "$REPO/code 3/code" && sha256sum $FILES | cut -c1-16,66- && wc -c $FILES )

echo "########## ③ 远端四枚 + SoC 残留 + 探针命中 + 禁用词 ##########"
nssh() { timeout "${TMO:-600}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
nssh "cd ~/sfa_real/code && sha256sum $FILES | cut -c1-16,66- ; \
  echo SOCEDEX=\$(grep -rc ascend910_93 op_host/ op_kernel/ CMakeLists.txt | grep -v ':0' | wc -l) ; \
  echo MARK=\$(grep -c 'P125 计量器' op_kernel/sparse_flash_attention.cpp) \
       REP=\$(grep -c 'p125r = 0u' op_kernel/sparse_flash_attention.cpp) \
       NBOUND=\$(grep -c \"p125r < ${N}u\" op_kernel/sparse_flash_attention.cpp) \
       FLAG=\$(grep -cE '(Set|Wait)Flag<HardEvent::' op_kernel/sparse_flash_attention.cpp) \
       HOSTP=\$(grep -c 'P12[5-9]' op_host/sparse_flash_attention.cpp) \
       OLD=\$(grep -cE 'P12[0-4]|p12[0-4]|P11[0-9]|skP119|probeRep_|Muls\(oc, oc|Muls\(dk, dk' op_kernel/sparse_flash_attention.cpp) ; \
  echo FORBID=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0|调试' op_kernel/sparse_flash_attention.cpp op_host/sparse_flash_attention.cpp | tr '\n' ' ')" 2>&1 | grep -av Warning
echo ">>> 期望 SOCEDEX=0 MARK=$WANT REP=$WANT NBOUND=$WANT FLAG=36 HOSTP=0 OLD=0 FORBID=0；"
echo ">>>      除 kernel 那一枚外，其余三枚必须仍是 f2b28a86 / 1046b349 / 3a8f53050c8f15ae（纯 kernel 发）"
echo "⛔ kernel 的 sha 每档都不同（N 写死在循环上界里）⇒ 这里只核'本地==远端'，不核它等于某个历史值。"

echo "########## ④ dry-run（四枚必须逐个吻合） ##########"
nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --dry-run" 2>&1 | grep -av Warning
echo "P125_PRECHECK_DONE ARM=$ARM N=$N"
