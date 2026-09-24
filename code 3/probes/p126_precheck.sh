#!/bin/bash
# P126 发次链（§7.5 固定顺序）：数值门已跑完（p126_gate.sh）⇒ **最后一次 sync（此后不再 build、不再 sed）**
# → 核四枚当场 sha256（本地 vs 远端）+ SoC sed 残留 0 + 禁用词 0 + 钉档命中数 → dry-run 四枚逐个吻合。
# ⛔ 这一发只做到 dry-run；submit 由 p126_submit.sh 单独跑（读数要留在另一段里）。
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

echo "########## ③ 远端四枚 + SoC 残留 + 钉档命中数 + 禁用词 ##########"
nssh "cd ~/sfa_real/code && sha256sum $FILES | cut -c1-16,66- ; \
  echo SOCEDEX=\$(grep -rc ascend910_93 op_host/ op_kernel/ CMakeLists.txt | grep -v ':0' | wc -l) ; \
  echo PIN=\$(grep -c 'NB_CAND\[\] = {1}' op_host/sparse_flash_attention.cpp) \
       NBLINES=\$(grep -c NB_CAND op_host/sparse_flash_attention.cpp) \
       OLDTAB=\$(grep -c '{32, 16, 8, 4, 2, 1}' op_host/sparse_flash_attention.cpp) \
       MARK=\$(grep -c 'P126' op_host/sparse_flash_attention.cpp) \
       KRES=\$(grep -cE 'P1[12][0-9]|p12[0-9]|skP119|probeRep_|Muls\(oc, oc|Muls\(dk, dk' op_kernel/sparse_flash_attention.cpp) ; \
  echo FORBID=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0|调试' op_kernel/sparse_flash_attention.cpp op_host/sparse_flash_attention.cpp | tr '\n' ' ')" 2>&1 | grep -av Warning
echo ">>> 期望 PIN=1 NBLINES=4 OLDTAB=0 MARK=1 KRES=0 FORBID=0 SOCEDEX=0；host=3969800637a4ce58，kernel 三枚 = f815bf1e / f2b28a86 / 1046b349"
echo "⚠️ KRES 的门用 'P1[12][0-9]' 而不是 'P1[0-2][0-9]'：pristine kernel 里合法地留着 4 处 'P105' 注释（那是发货字节的一部分，不是探针残留）。"

echo "########## ④ dry-run（四枚必须逐个吻合） ##########"
nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --dry-run" 2>&1 | grep -av Warning
echo "P126_PRECHECK_DONE"
