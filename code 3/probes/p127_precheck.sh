#!/bin/bash
# P127 发次链（§7.5 固定顺序）：数值门已跑完（p127_gate.sh）⇒ **最后一次 sync（此后不再 build、不再 sed）**
# → 核四枚当场 sha256（本地 vs 远端）+ SoC sed 残留 0 + 禁用词 0 + 钉档命中数 → dry-run 四枚逐个吻合。
# ⛔ 这一发只做到 dry-run；submit 由 p127_submit.sh 单独跑（读数要留在另一段里）。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
PID=6a7c22d6a52e0f540a8a098d
CLI=/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit/cannjudge_cli.py
FILES="op_kernel/sparse_flash_attention.cpp op_kernel/sparse_flash_attention_tiling.h op_kernel/tiling_key_sparse_flash_attention.h op_host/sparse_flash_attention.cpp"

echo "########## ① 最后一次 sync（此后不再 build） ##########"
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -3

echo "########## ② 本地四枚当场 sha256 ##########"
( cd "$REPO/code 3/code" && sha256sum $FILES | cut -c1-16,66- && wc -c $FILES )

echo "########## ③ 远端四枚 + SoC 残留 + 钉档命中数 + 禁用词 ##########"
nssh() { timeout "${TMO:-600}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
nssh "cd ~/sfa_real/code && sha256sum $FILES | cut -c1-16,66- ; \
  echo SOCEDEX=\$(grep -rc ascend910_93 op_host/ op_kernel/ CMakeLists.txt | grep -v ':0' | wc -l) ; \
  echo P127=\$(grep -c 'P127 单元数下界档' op_host/sparse_flash_attention.cpp) \
       OLDANCHOR=\$(grep -c 'cost \* 20ULL' op_host/sparse_flash_attention.cpp) \
       BC0=\$(grep -c 'bestCost == 0' op_host/sparse_flash_attention.cpp) \
       OLDTAB=\$(grep -c '{32, 16, 8, 4, 2, 1}' op_host/sparse_flash_attention.cpp) \
       KRES=\$(grep -cE 'P1[12][0-9]|p12[0-9]|skP119|probeRep_|Muls\(oc, oc|Muls\(dk, dk' op_kernel/sparse_flash_attention.cpp) ; \
  echo FORBID=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0|调试' op_kernel/sparse_flash_attention.cpp op_host/sparse_flash_attention.cpp | tr '\n' ' ')" 2>&1 | grep -av Warning
echo ">>> 期望 P127=1 OLDANCHOR=0 BC0=2 OLDTAB=1 KRES=0 FORBID=0 SOCEDEX=0；"
echo ">>>      host=f0a25b0771b644d0，kernel 三枚 = f815bf1e / f2b28a86 / 1046b349（一发都没动）"
echo "⚠️ KRES 的门用 'P1[12][0-9]'：pristine kernel 里合法留着 'P105' 注释（发货字节的一部分，不是探针残留）。"
echo "⚠️ OLDTAB=1 是**对的**：P127 不动候选表（那是 P126 的钉法），只动 L304 的改档门。"

echo "########## ④ dry-run（四枚必须逐个吻合） ##########"
nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --dry-run" 2>&1 | grep -av Warning
echo "P127_PRECHECK_DONE"
