#!/bin/bash
# P120 发次链（§3 / §7.5 固定顺序）：闸门跑完 → **最后一次 sync（此后不再 build、不再 sed）**
# → 核四枚当场 sha256（本地 vs 远端）+ SoC sed 残留 0 + 禁用词 0 + 命中数 → dry-run 四枚逐个吻合。
# ⛔ 这一发只做到 dry-run；submit 由 p120_submit.sh 单独跑（读数要留在另一段里）。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
PID=6a7c22d6a52e0f540a8a098d
CLI=/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit/cannjudge_cli.py
nssh() { timeout "${TMO:-600}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
FILES="op_kernel/sparse_flash_attention.cpp op_kernel/sparse_flash_attention_tiling.h op_kernel/tiling_key_sparse_flash_attention.h op_host/sparse_flash_attention.cpp"

echo "########## ① 最后一次 sync（此后不再 build） ##########"
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -3

echo "########## ② 本地四枚当场 sha256 ##########"
( cd "$REPO/code 3/code" && sha256sum $FILES | cut -c1-16,66- && wc -c $FILES )

echo "########## ③ 远端四枚 + SoC 残留 + 命中数 + 禁用词 ##########"
nssh "cd ~/sfa_real/code && sha256sum $FILES | cut -c1-16,66- ; \
  echo SOCEDEX=\$(grep -rc ascend910_93 op_host/ op_kernel/ CMakeLists.txt | grep -v ':0' | wc -l) ; \
  echo MARK=\$(grep -c 'P120 flag meter' op_kernel/sparse_flash_attention.cpp) \
  SET4=\$(grep -c 'SetFlag<HardEvent::MTE2_V>(4)' op_kernel/sparse_flash_attention.cpp) \
  WAIT4=\$(grep -c 'WaitFlag<HardEvent::MTE2_V>(4)' op_kernel/sparse_flash_attention.cpp) \
  REAL=\$(grep -c 'SetFlag<HardEvent::V_MTE2>(0)' op_kernel/sparse_flash_attention.cpp) \
  ALLFLAG=\$(grep -cE '(Set|Wait)Flag<HardEvent::' op_kernel/sparse_flash_attention.cpp) \
  BUFF=\$(grep -c InitBuffer op_kernel/sparse_flash_attention.cpp) ; \
  echo RESIDUE=\$(grep -c 'P119 PV meter\|skP119\|rP119\|P118 gather meter\|P115 split\|P116 softmax-rep\|P114\|P112 探针\|P110 rep\|probeRep_' op_kernel/sparse_flash_attention.cpp) ; \
  echo FORBID=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0|调试' op_kernel/sparse_flash_attention.cpp op_host/sparse_flash_attention.cpp | tr '\n' ' ')" 2>&1 | grep -av Warning
N="${N:-127}"
echo ">>> 期望 MARK=1 SET4=$N WAIT4=$N REAL=2 ALLFLAG=$((36 + 2 * N)) BUFF=13 RESIDUE=0 FORBID=0 SOCEDEX=0"

echo "########## ④ dry-run（四枚必须逐个吻合） ##########"
nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --dry-run" 2>&1 | grep -av Warning
echo "P120_PRECHECK_DONE"
