#!/bin/bash
# P128 读数读完 ⇒ **回发当前最好的字节**（latest 计分，探针版不许留在榜上）。
#   本脚本 = "P128 输了"那条分支：把 host 拉回 pristine 再发一遍（P105 lineage 的四枚）。
#   "P128 赢了"（Σ ≤ −4 %）⇒ **不发这个**：护栏字节就是新的最好字节，直接再跑一遍
#   `p128_submit.sh`（远端树此刻已是核过 sha 的护栏字节、SOCEDEX=0）补那第二遍确认。
# 链：p128_pin.py revert（带 sha 断言）→ 最后 sync（不再 build）→ 核四枚 sha + SoC 0 + 护栏 0
#     + 禁用词 0 → dry-run 逐个吻合 → submit
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
PID=6a7c22d6a52e0f540a8a098d
CLI=/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit/cannjudge_cli.py
nssh() { timeout "${TMO:-900}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
FILES="op_kernel/sparse_flash_attention.cpp op_kernel/sparse_flash_attention_tiling.h op_kernel/tiling_key_sparse_flash_attention.h op_host/sparse_flash_attention.cpp"

python3 "$REPO/code 3/probes/p128_pin.py" revert
( cd "$REPO/code 3/code" && echo "本地四枚：" && sha256sum $FILES | cut -c1-16,66- )

bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -1
nssh "cd ~/sfa_real/code && sha256sum $FILES | cut -c1-16,66- ; \
  echo SOCEDEX=\$(grep -rc ascend910_93 op_host/ op_kernel/ CMakeLists.txt | grep -v ':0' | wc -l) \
       P128=\$(grep -c 'P128' op_host/sparse_flash_attention.cpp) \
       BESTUNITS=\$(grep -c 'bestUnits' op_host/sparse_flash_attention.cpp) \
       OLDGATE=\$(grep -c 'if (bestCost == 0 || cost' op_host/sparse_flash_attention.cpp) \
       OLDTAB=\$(grep -c '{32, 16, 8, 4, 2, 1}' op_host/sparse_flash_attention.cpp) \
       KRES=\$(grep -cE 'P1[12][0-9]|p12[0-9]|skP119|probeRep_|Muls\(oc, oc|Muls\(dk, dk' op_kernel/sparse_flash_attention.cpp) ; \
  echo FORBID=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0|调试' op_kernel/sparse_flash_attention.cpp op_host/sparse_flash_attention.cpp | tr '\n' ' ')" 2>&1 | grep -av Warning
echo ">>> 期望 SOCEDEX=0 P128=0 BESTUNITS=0 OLDGATE=1 OLDTAB=1 KRES=0 FORBID=0；host 回到 3a8f53050c8f15ae"

nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --dry-run" 2>&1 | grep -aE 'sha256|bytes' | cut -c1-100
nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --max-wait 600" 2>&1 | grep -av Warning | cut -c1-90
