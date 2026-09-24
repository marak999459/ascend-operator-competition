#!/bin/bash
# P114 发次链（§3 / §7.5 的固定顺序）：所有探针跑完 → 最后一次 sync（此后不再 build）
# → 核四枚当场 sha256（本地 vs 远端 + SoC sed 残留 0 + 禁用词 0）→ dry-run（四枚必须逐个吻合）。
# ⛔ 这一发只做到 dry-run 为止；去掉 --dry-run 由人确认读数后单独跑（p114_push.sh）。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
PID=6a7c22d6a52e0f540a8a098d
CLI=/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit/cannjudge_cli.py
nssh() { timeout "${TMO:-600}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }

echo "########## ① 最后一次 sync（此后不再 build） ##########"
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -3

echo "########## ② 本地四枚当场 sha256 ##########"
( cd "$REPO/code 3/code" && sha256sum op_kernel/sparse_flash_attention.cpp op_kernel/sparse_flash_attention_tiling.h op_kernel/tiling_key_sparse_flash_attention.h op_host/sparse_flash_attention.cpp | cut -c1-16,66- && wc -c op_kernel/*.cpp op_kernel/*.h op_host/*.cpp )

echo "########## ③ 远端四枚 + SoC sed 残留 + 禁用词 ##########"
nssh "cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp op_kernel/sparse_flash_attention_tiling.h op_kernel/tiling_key_sparse_flash_attention.h op_host/sparse_flash_attention.cpp | cut -c1-16,66- ; \
  echo SOCEDEX=\$(grep -rc ascend910_93 op_host/ op_kernel/ CMakeLists.txt | grep -v ':0' | wc -l) ; \
  echo MARK_P114=\$(grep -c 'P114' op_kernel/sparse_flash_attention.cpp) RESIDUE=\$(grep -c 'P114 rep\|probeRep_\|P112 探针\|P110 rep' op_kernel/sparse_flash_attention.cpp) ; \
  echo FORBID=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0|调试' op_kernel/sparse_flash_attention.cpp op_host/sparse_flash_attention.cpp | tr '\n' ' ') ; \
  echo HITS_KB=\$(grep -c 'CopyGm2Ub(kb\[done \* D_\]' op_kernel/sparse_flash_attention.cpp) HITS_KR=\$(grep -c 'CopyGm2Ub(kr\[done \* Dr_\]' op_kernel/sparse_flash_attention.cpp) HITS_VB=\$(grep -c 'CopyGm2Ub(vb\[done \* D_\]' op_kernel/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning

echo "########## ④ dry-run ##########"
nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --dry-run" 2>&1 | grep -av Warning
